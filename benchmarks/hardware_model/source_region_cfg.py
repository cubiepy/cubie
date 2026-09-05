"""Compile source regions to one cyclic SSA program for lane replay."""

import ast
import hashlib
import inspect
import json
import math

import numpy as np

from cubie.cuda_simsafe import float32, int32

from benchmarks.hardware_model.expansion import (
    UNKNOWN,
    constant,
    source_receipt,
)
from benchmarks.hardware_model.workload import source_function


FULL = (1 << 32) - 1
WORDS = {"bool": 1, "int32": 1, "float32": 1, "uint64": 2, "memory": 0}


def captured_arrays(graph, context, names):
    """Recover exact alias extents from one actual captured helper call."""
    matches = [
        call
        for call in graph["calls"]
        if call["kind"] == "source_call" and call["context"] == context
    ]
    if len(matches) != 1:
        raise ValueError("Region needs one actual captured call context")
    call = matches[0]
    arrays = {}
    for name in names:
        binding = call["bindings"].get(name, {}).get("view")
        if binding is None:
            aliases = [
                row["view"]
                for row in graph["aliases"]
                if row["source"]["context"] == context
                and row["name"] == name
                and row["view"] is not None
            ]
            unique = {json.dumps(row, sort_keys=True) for row in aliases}
            if len(unique) == 1:
                binding = aliases[0]
        if (
            not isinstance(binding, dict)
            or binding["itemsize"] != 4
            or len(binding["shape"]) != 1
            or binding["bytes"] != 4 * binding["shape"][0]
        ):
            raise ValueError("Actual formal is not a flat four-byte alias")
        arrays[name] = dict(
            storage=binding["storage"],
            offset=binding["offset"],
            length=binding["shape"][0],
            dtype=binding["dtype"],
        )
    return arrays, dict(
        source_graph_sha256=hashlib.sha256(
            json.dumps(
                graph,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        context=context,
        call=call,
        arrays=arrays,
        workload_identity=graph["candidate_construction"]["workload_identity"],
    )


class RegionCFG:
    """Retain source branches and fixed loops before common allocation.

    This constructor admits an explicitly selected contiguous source region.
    Calls need a supported primitive form; no opaque effects are invented.
    Array bindings describe complete addressable aliases and initial cells.
    """

    def __init__(
        self,
        function,
        scalar_types,
        arrays,
        output_names,
        lines=None,
        false_form="retained",
        array_capture=None,
    ):
        if false_form != "retained":
            raise ValueError("False requires an explicit admitted form")
        self.function = function
        closure = inspect.getclosurevars(function)
        self.constants = dict(
            closure.builtins, **closure.globals, **closure.nonlocals
        )
        self.types = dict(scalar_types)
        if any(
            kind not in WORDS or kind == "memory"
            for kind in self.types.values()
        ):
            raise ValueError("Unknown scalar source type")
        self.inputs = dict(self.types)
        self.arrays = arrays
        self.outputs = list(output_names)
        self.blocks = []
        self.instructions = []
        self.loops = []
        self.scalar_constants = {}
        self.current = self.new_block()
        self.entry = self.current
        self.source = source_receipt(function)
        if array_capture is not None:
            if (
                array_capture["arrays"] != arrays
                or array_capture["call"]["source"] != self.source
            ):
                raise ValueError(
                    "Array capture does not bind this source function"
                )
        tree = source_function(function)
        body = tree.body
        if lines is not None:
            candidates = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.stmt)
                and lines[0] <= node.lineno <= lines[1]
            ]
            roots = [
                node
                for node in candidates
                if not any(
                    node is child
                    for parent in candidates
                    if parent is not node
                    for child in ast.walk(parent)
                )
            ]
            body = sorted(roots, key=lambda node: node.lineno)
            if not body or body[0].lineno != lines[0]:
                raise ValueError("Region start is not an actual statement")
            if any(node.end_lineno > lines[1] for node in body):
                raise ValueError("Region cuts through an actual statement")
        for binding in arrays.values():
            if set(binding) != {"storage", "offset", "length", "dtype"}:
                raise ValueError("Array binding requires exact byte geometry")
            if (
                binding["dtype"] not in ("float32", "int32")
                or any(
                    type(binding[key]) is not int or binding[key] < 0
                    for key in ("offset", "length")
                )
                or binding["offset"] % 4
            ):
                raise ValueError("Only aligned four-byte cells are admitted")
            memory = "@memory:" + binding["storage"]
            base = "@base:" + binding["storage"]
            self.inputs[memory] = self.types[memory] = "memory"
            self.inputs[base] = self.types[base] = "uint64"
        self.block(body)
        self.blocks[self.current]["terminator"] = dict(
            kind="return",
            values=self.outputs
            + sorted(key for key in self.inputs if key.startswith("@memory:")),
        )
        self.program = self.to_ssa()
        self.program.update(
            kind="source_region_cyclic_ssa",
            source=self.source,
            selected_lines=lines,
            arrays=arrays,
            loops=self.loops,
            array_capture=array_capture,
            compiler_form=dict(
                branches="divergent_structured",
                false_loops=false_form,
                arrays="addressable_alias_storage",
            ),
            scope="Selected complete statements, not an entire solver plan",
        )

    def new_block(self):
        identifier = len(self.blocks)
        self.blocks.append(
            dict(id=identifier, instructions=[], terminator=None)
        )
        return identifier

    def emit(self, opcode, inputs, dtype, node, output=None, **details):
        output = output or f"@temporary:{len(self.instructions)}"
        if output in self.types and self.types[output] != dtype:
            raise ValueError("Source assignment changes scalar type")
        self.types[output] = dtype
        instruction = dict(
            id=len(self.instructions),
            opcode=opcode,
            inputs=inputs,
            output=output,
            dtype=dtype,
            source=dict(line=node.lineno, syntax=ast.unparse(node)),
            **details,
        )
        self.instructions.append(instruction)
        self.blocks[self.current]["instructions"].append(instruction["id"])
        return output

    def literal(self, value, node):
        if isinstance(value, (bool, np.bool_)):
            dtype, value = "bool", bool(value)
        elif isinstance(value, (int, np.integer)):
            dtype, value = "int32", int(value)
            if not -(2**31) <= value < 2**31:
                raise ValueError("Literal exceeds int32 source range")
        elif isinstance(value, np.float32):
            dtype, value = "float32", float(value)
        else:
            raise ValueError("Raw Python floats require a source cast")
        return self.emit("literal", [], dtype, node, value=value)

    def raw(self, node):
        return constant(node, dict(self.constants, **self.scalar_constants))

    def expr(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.scalar_constants:
                return self.literal(self.scalar_constants[node.id], node)
            if node.id in self.types:
                return node.id
            raw = self.raw(node)
            if raw is not UNKNOWN:
                return self.literal(raw, node)
            raise ValueError("Unbound source name " + node.id)
        if isinstance(node, ast.Constant):
            return self.literal(node.value, node)
        if isinstance(node, ast.Subscript):
            binding, index = self.reference(node)
            return self.emit(
                "load",
                [
                    "@memory:" + binding["storage"],
                    "@base:" + binding["storage"],
                    index,
                ],
                binding["dtype"],
                node,
                memory=binding,
            )
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left), self.expr(node.right)
            dtype = self.types[left]
            if dtype != self.types[right] or dtype not in ("int32", "float32"):
                raise ValueError("Binary source promotion is unproved")
            opcode = type(node.op).__name__
            if opcode not in ("Add", "Sub", "Mult", "Div", "BitAnd", "BitOr"):
                raise ValueError("Unsupported source binary operation")
            if opcode == "Div" and dtype != "float32":
                raise ValueError("Integer division needs its own proof")
            return self.emit(opcode, [left, right], dtype, node)
        if isinstance(node, ast.UnaryOp):
            operand = self.expr(node.operand)
            opcode = type(node.op).__name__
            if opcode not in ("Not", "USub", "UAdd"):
                raise ValueError("Unsupported unary source operation")
            return self.emit(
                opcode,
                [operand],
                "bool" if opcode == "Not" else self.types[operand],
                node,
            )
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError("Chained comparison needs short-circuit CFG")
            inputs = [self.expr(node.left), self.expr(node.comparators[0])]
            if self.types[inputs[0]] != self.types[inputs[1]]:
                raise ValueError("Comparison promotion is unproved")
            return self.emit(type(node.ops[0]).__name__, inputs, "bool", node)
        if isinstance(node, ast.Call):
            target = self.raw(node.func)
            if (
                len(node.args) == 1
                and not node.keywords
                and target
                in (
                    np.float32,
                    np.int32,
                    float32,
                    int32,
                )
            ):
                dtype = (
                    "float32" if target in (np.float32, float32) else "int32"
                )
                raw = self.raw(node.args[0])
                if raw is not UNKNOWN:
                    return self.literal(target(raw), node)
                return self.emit(
                    "cast", [self.expr(node.args[0])], dtype, node
                )
            if (
                getattr(target, "__name__", None) == "selp"
                and len(node.args) == 3
                and not node.keywords
            ):
                inputs = [self.expr(argument) for argument in node.args]
                if (
                    self.types[inputs[0]] != "bool"
                    or self.types[inputs[1]] != self.types[inputs[2]]
                ):
                    raise ValueError("Select operand source types differ")
                return self.emit("Select", inputs, self.types[inputs[1]], node)
            raise ValueError(
                "Source call needs an explicit effect/type form: "
                + ast.unparse(node)
            )
        raise ValueError("Unsupported source expression " + ast.unparse(node))

    def reference(self, node):
        if (
            not isinstance(node.value, ast.Name)
            or node.value.id not in self.arrays
        ):
            raise ValueError("Array alias binding is unresolved")
        binding = dict(self.arrays[node.value.id])
        index = self.expr(node.slice)
        if self.types[index] != "int32":
            raise ValueError("Array index lacks int32 source type")
        return binding, index

    def assign(self, target, value, node):
        if isinstance(target, ast.Name):
            self.emit(
                "copy", [value], self.types[value], node, output=target.id
            )
        elif isinstance(target, ast.Subscript):
            binding, index = self.reference(target)
            if binding["dtype"] != self.types[value]:
                raise ValueError("Array store conversion is unproved")
            memory = "@memory:" + binding["storage"]
            self.emit(
                "store",
                [memory, "@base:" + binding["storage"], index, value],
                "memory",
                node,
                output=memory,
                memory=binding,
            )
        else:
            raise ValueError("Unsupported source assignment")

    def jump(self, target):
        self.blocks[self.current]["terminator"] = dict(
            kind="jump", target=target
        )

    def block(self, statements):
        for node in statements:
            if isinstance(node, ast.Assign):
                value = self.expr(node.value)
                for target in node.targets:
                    self.assign(target, value, node)
            elif isinstance(node, ast.AugAssign):
                synthetic = ast.copy_location(
                    ast.BinOp(left=node.target, op=node.op, right=node.value),
                    node,
                )
                self.assign(node.target, self.expr(synthetic), node)
            elif isinstance(node, ast.If):
                folded = self.raw(node.test)
                if isinstance(folded, (bool, np.bool_)):
                    self.block(node.body if folded else node.orelse)
                    continue
                predicate = self.expr(node.test)
                if self.types[predicate] != "bool":
                    raise ValueError("Source branch predicate is not bool")
                origin = self.current
                first_loop = len(self.loops)
                left, right, join = (self.new_block() for _ in range(3))
                self.blocks[origin]["terminator"] = dict(
                    kind="branch",
                    predicate=predicate,
                    true=left,
                    false=right,
                    join=join,
                    line=node.lineno,
                )
                self.current = left
                self.block(node.body)
                self.jump(join)
                self.current = right
                self.block(node.orelse)
                self.jump(join)
                self.blocks[origin]["terminator"]["contains_cyclic_loop"] = (
                    any(
                        row["compiler_form"] == "retained"
                        for row in self.loops[first_loop:]
                    )
                )
                self.current = join
            elif isinstance(node, ast.For):
                self.loop(node)
            elif isinstance(node, ast.Pass):
                continue
            elif isinstance(node, ast.Expr) and isinstance(
                node.value, ast.Constant
            ):
                continue
            else:
                raise ValueError(
                    "Unsupported source statement " + ast.unparse(node)
                )

    def loop(self, node):
        iterator = node.iter
        if (
            node.orelse
            or not isinstance(node.target, ast.Name)
            or not isinstance(iterator, ast.Call)
            or len(iterator.args) != 2
            or getattr(self.raw(iterator.func), "__name__", None)
            != "unroll_if"
        ):
            raise ValueError("Loop needs its actual captured unroll directive")
        bounds = iterator.args[0]
        if (
            not isinstance(bounds, ast.Call)
            or self.raw(bounds.func) is not range
            or bounds.keywords
        ):
            raise ValueError("Loop range is not source-resolved")
        numbers = [self.raw(arg) for arg in bounds.args]
        if any(not isinstance(value, (int, np.integer)) for value in numbers):
            raise ValueError("Loop cap is not a source constant")
        indices = range(*(int(value) for value in numbers))
        if indices.step <= 0:
            raise ValueError("Only positive source strides are admitted")
        flag = self.raw(iterator.args[1])
        if not isinstance(flag, tuple) or len(flag) != 2:
            raise ValueError("Captured loop flag is malformed")
        full = flag == (True, None)
        factor = 1 if flag[0] is False else flag[1]
        if not full and (type(factor) is not int or factor < 1):
            raise ValueError("Counted loop requires a positive source factor")
        self.loops.append(
            dict(
                line=node.lineno,
                source_flag=list(flag),
                range=[indices.start, indices.stop, indices.step],
                compiler_form="full" if full else "retained",
                copies=len(indices) if full else factor,
            )
        )
        if full:
            previous = self.scalar_constants.get(node.target.id, UNKNOWN)
            for index in indices:
                self.emit(
                    "copy",
                    [self.literal(index, node)],
                    "int32",
                    node,
                    output=node.target.id,
                )
                self.scalar_constants[node.target.id] = index
                self.block(node.body)
            if previous is UNKNOWN:
                self.scalar_constants.pop(node.target.id, None)
            else:
                self.scalar_constants[node.target.id] = previous
            return
        counter = f"@loop:{node.lineno}"
        self.emit(
            "copy",
            [self.literal(indices.start, node)],
            "int32",
            node,
            output=counter,
        )
        header, body, done = (self.new_block() for _ in range(3))
        self.jump(header)
        self.current = header
        test = self.emit(
            "Lt", [counter, self.literal(indices.stop, node)], "bool", node
        )
        self.blocks[header]["terminator"] = dict(
            kind="branch",
            predicate=test,
            true=body,
            false=done,
            join=done,
            line=node.lineno,
            loop_header=True,
        )
        self.current = body
        for lane in range(factor):
            offset = self.literal(lane * indices.step, node)
            induction = self.emit("Add", [counter, offset], "int32", node)
            # The source tail has a real predicate; no witness is folded.
            if lane:
                tail_origin = self.current
                tail_body, tail_join = self.new_block(), self.new_block()
                test = self.emit(
                    "Lt",
                    [induction, self.literal(indices.stop, node)],
                    "bool",
                    node,
                )
                self.blocks[tail_origin]["terminator"] = dict(
                    kind="branch",
                    predicate=test,
                    true=tail_body,
                    false=tail_join,
                    join=tail_join,
                    line=node.lineno,
                )
                self.current = tail_body
                self.emit(
                    "copy", [induction], "int32", node, output=node.target.id
                )
                self.block(node.body)
                self.jump(tail_join)
                self.current = tail_join
            else:
                self.emit(
                    "copy", [induction], "int32", node, output=node.target.id
                )
                self.block(node.body)
        increment = self.emit(
            "Add",
            [counter, self.literal(factor * indices.step, node)],
            "int32",
            node,
        )
        self.emit("copy", [increment], "int32", node, output=counter)
        self.jump(header)
        self.current = done

    def to_ssa(self):
        predecessors = [[] for _ in self.blocks]
        successors = []
        uses, definitions = [], []
        for block in self.blocks:
            term = block["terminator"]
            targets = (
                [term["target"]]
                if term["kind"] == "jump"
                else [term["true"], term["false"]]
                if term["kind"] == "branch"
                else []
            )
            successors.append(targets)
            for target in targets:
                predecessors[target].append(block["id"])
            seen, needed = set(), set()
            for identifier in block["instructions"]:
                op = self.instructions[identifier]
                needed.update(set(op["inputs"]) - seen)
                seen.add(op["output"])
            tail = (
                [term["predicate"]]
                if term["kind"] == "branch"
                else term.get("values", [])
            )
            needed.update(set(tail) - seen)
            uses.append(needed)
            definitions.append(seen)
        live = [set() for _ in self.blocks]
        rounds = 0
        while True:
            rounds += 1
            updated = [
                uses[i]
                | (
                    set().union(*(live[j] for j in successors[i]))
                    - definitions[i]
                )
                for i in range(len(live))
            ]
            if updated == live:
                break
            live = updated
        values, inputs = [], {}

        def value(name, producer):
            identifier = len(values)
            values.append(
                dict(
                    id=identifier,
                    variable=name,
                    dtype=self.types[name],
                    producer=producer,
                )
            )
            return identifier

        for name in sorted(self.inputs):
            inputs[name] = value(name, None)
        outputs = {
            op["id"]: value(op["output"], op["id"]) for op in self.instructions
        }
        last = [
            {
                self.instructions[i]["output"]: outputs[i]
                for i in block["instructions"]
            }
            for block in self.blocks
        ]
        phis = {}
        for block in self.blocks:
            identifier = block["id"]
            if len(predecessors[identifier]) > 1:
                for name in sorted(live[identifier]):
                    phis[identifier, name] = dict(
                        output=value(name, "phi"), variable=name, incoming={}
                    )
        entries = {}
        active = set()

        def entry(block, name):
            key = block, name
            if key in entries:
                return entries[key]
            if key in phis:
                result = phis[key]["output"]
            elif block == self.entry:
                if name not in inputs:
                    raise ValueError(
                        "Source read lacks an incoming definition: " + name
                    )
                result = inputs[name]
            else:
                if key in active or len(predecessors[block]) != 1:
                    raise ValueError("Unresolved cyclic source definition")
                active.add(key)
                result = exit_value(predecessors[block][0], name)
                active.remove(key)
            entries[key] = result
            return result

        def exit_value(block, name):
            return (
                last[block][name]
                if name in last[block]
                else entry(block, name)
            )

        for (block, name), phi in phis.items():
            phi["incoming"] = {
                str(pred): exit_value(pred, name)
                for pred in predecessors[block]
            }
        instructions, blocks = [], []
        for block in self.blocks:
            current = {}

            def read(name):
                return (
                    current[name]
                    if name in current
                    else entry(block["id"], name)
                )

            for identifier in block["instructions"]:
                op = self.instructions[identifier]
                result = dict(
                    op,
                    inputs=[read(name) for name in op["inputs"]],
                    output=outputs[identifier],
                )
                instructions.append(result)
                current[op["output"]] = outputs[identifier]
            term = dict(block["terminator"])
            if term["kind"] == "branch":
                term["predicate"] = read(term["predicate"])
            elif term["kind"] == "return":
                term["names"] = term["values"]
                term["values"] = [read(name) for name in term["values"]]
            blocks.append(
                dict(
                    block,
                    terminator=term,
                    phis=[
                        phi
                        for (owner, _), phi in phis.items()
                        if owner == block["id"]
                    ],
                    predecessors=predecessors[block["id"]],
                    successors=successors[block["id"]],
                )
            )
        instructions.sort(key=lambda row: row["id"])
        return dict(
            schema=1,
            entry=self.entry,
            blocks=blocks,
            instructions=instructions,
            values=values,
            inputs=inputs,
            source_name_liveness_rounds=rounds,
        )


def allocate_cfg(program, gpr_budget=255, predicate_budget=7):
    """Allocate one static cyclic SSA graph with fixed-point edge liveness."""
    if (
        type(gpr_budget) is not int
        or not 1 <= gpr_budget <= 255
        or type(predicate_budget) is not int
        or not 1 <= predicate_budget <= 7
    ):
        raise ValueError("Allocation budgets exceed admitted physical banks")
    blocks, ops, values = (
        program[key] for key in ("blocks", "instructions", "values")
    )
    use, defs, phi_defs = [], [], []
    for block in blocks:
        defined = {phi["output"] for phi in block["phis"]}
        phi_defs.append(set(defined))
        needed = set()
        for identifier in block["instructions"]:
            op = ops[identifier]
            needed.update(set(op["inputs"]) - defined)
            defined.add(op["output"])
        term = block["terminator"]
        needed.update(
            set(
                [term["predicate"]]
                if term["kind"] == "branch"
                else term.get("values", [])
            )
            - defined
        )
        use.append(needed)
        defs.append(defined)
    before, after = [set() for _ in blocks], [set() for _ in blocks]
    rounds = 0
    while True:
        rounds += 1
        new_after = []
        for block in blocks:
            edge = set()
            for target in block["successors"]:
                edge |= before[target] - phi_defs[target]
                edge.update(
                    phi["incoming"][str(block["id"])]
                    for phi in blocks[target]["phis"]
                )
            new_after.append(edge)
        new_before = [
            use[i] | (new_after[i] - defs[i]) for i in range(len(blocks))
        ]
        if new_before == before and new_after == after:
            break
        before, after = new_before, new_after
    neighbors = [set() for _ in values]

    def clique(members):
        members = {i for i in members if WORDS[values[i]["dtype"]]}
        for identifier in members:
            neighbors[identifier].update(members - {identifier})

    for block in blocks:
        live = set(after[block["id"]])
        term = block["terminator"]
        live.update(
            [term["predicate"]]
            if term["kind"] == "branch"
            else term.get("values", [])
        )
        clique(live)
        for identifier in reversed(block["instructions"]):
            op = ops[identifier]
            clique(live | {op["output"]} | set(op["inputs"]))
            live.discard(op["output"])
            live.update(op["inputs"])
        clique(live | phi_defs[block["id"]])
        for pred in block["predecessors"]:
            clique(
                phi_defs[block["id"]]
                | {phi["incoming"][str(pred)] for phi in block["phis"]}
            )
    colors = {}
    for identifier in sorted(
        range(len(values)),
        key=lambda i: (
            -len(neighbors[i]),
            -WORDS[values[i]["dtype"]],
            i,
        ),
    ):
        dtype = values[identifier]["dtype"]
        width = WORDS[dtype]
        if not width:
            continue
        bank = "P" if dtype == "bool" else "R"
        occupied = {
            word
            for other in neighbors[identifier]
            if other in colors and colors[other]["bank"] == bank
            for word in colors[other]["words"]
        }
        budget = predicate_budget if bank == "P" else gpr_budget
        first = next(
            (
                start
                for start in range(budget - width + 1)
                if all(
                    word not in occupied
                    for word in range(start, start + width)
                )
            ),
            None,
        )
        if first is None:
            raise ValueError(
                "Common CFG allocation needs explicit spill lowering"
            )
        colors[identifier] = dict(
            bank=bank, words=list(range(first, first + width))
        )
    return dict(
        kind="common_cyclic_ssa_allocation",
        locations=colors,
        gpr_words=max(
            (
                max(row["words"]) + 1
                for row in colors.values()
                if row["bank"] == "R"
            ),
            default=0,
        ),
        predicate_words=max(
            (
                max(row["words"]) + 1
                for row in colors.values()
                if row["bank"] == "P"
            ),
            default=0,
        ),
        fixed_point_rounds=rounds,
        live_in=[sorted(row) for row in before],
        live_out=[sorted(row) for row in after],
        interference=[sorted(row) for row in neighbors],
        spill_form="No spill admission; exhausted budget is explicit",
    )


def replay_lanes(
    program,
    allocation,
    inputs,
    entry_mask=FULL,
    branch_form="divergent_structured",
):
    """Execute declared lane values through one source CFG allocation.

    Predicated form retains zero-execution-mask instruction issues. Stores
    and copies write only executing lanes. Phi edge copies are parallel.
    Input payloads validate semantics; they never alter program allocation.
    """
    if type(entry_mask) is not int or not 0 < entry_mask <= FULL:
        raise ValueError("A replay requires entered warp lanes")
    if branch_form not in ("divergent_structured", "predicated_acyclic_arms"):
        raise ValueError("Unknown source branch compiler form")
    values, blocks, ops = (
        program[key] for key in ("values", "blocks", "instructions")
    )
    locations = {
        int(key): value for key, value in allocation["locations"].items()
    }
    registers, memory = {}, {}
    events = []

    def lanes(mask):
        return [lane for lane in range(32) if mask & (1 << lane)]

    def write(identifier, lane, value):
        if values[identifier]["dtype"] == "memory":
            memory[identifier, lane] = value
            return
        location = locations[identifier]
        for word in location["words"]:
            registers[location["bank"], word, lane] = identifier, value

    def read(identifier, lane):
        if values[identifier]["dtype"] == "memory":
            if (identifier, lane) not in memory:
                raise ValueError("Read of undefined lane memory version")
            return memory[identifier, lane]
        location = locations[identifier]
        records = [
            registers.get((location["bank"], word, lane))
            for word in location["words"]
        ]
        if any(
            record is None or record[0] != identifier for record in records
        ):
            raise ValueError(
                "Physical register does not hold the demanded lane value"
            )
        return records[0][1]

    if set(inputs) != set(program["inputs"]):
        raise ValueError("Replay input inventory differs from compiled source")
    for name, identifier in program["inputs"].items():
        payload = inputs[name]
        if not isinstance(payload, list) or len(payload) != 32:
            raise ValueError(
                "Every replay input needs exactly 32 lane payloads"
            )
        if identifier in allocation["live_in"][program["entry"]]:
            for lane in lanes(entry_mask):
                write(identifier, lane, payload[lane])

    def transfer(pred, target, mask):
        updates = [
            (phi["output"], lane, read(phi["incoming"][str(pred)], lane))
            for phi in blocks[target]["phis"]
            for lane in lanes(mask)
        ]
        for identifier, lane, result in updates:
            write(identifier, lane, result)

    def scalar(op, lane):
        kind = op["opcode"]
        if kind == "Select":
            predicate = bool(read(op["inputs"][0], lane))
            return read(op["inputs"][1 if predicate else 2], lane)
        args = [read(identifier, lane) for identifier in op["inputs"]]
        if kind == "literal":
            return op["value"]
        if kind in ("load", "store"):
            geometry = op["memory"]
            index = args[2]
            if type(index) is not int or not 0 <= index < geometry["length"]:
                raise ValueError("Runtime address exceeds source-bound alias")
            offset = geometry["offset"] + 4 * index
            if kind == "load":
                if offset not in args[0]:
                    raise ValueError(
                        "Actual cell lacks an initialized lane value"
                    )
                return args[0][offset]
            result = dict(args[0])
            result[offset] = args[3]
            return result
        functions = {
            "copy": lambda a: a,
            "cast": lambda a: a,
            "Add": lambda a, b: a + b,
            "Sub": lambda a, b: a - b,
            "Mult": lambda a, b: a * b,
            "Div": lambda a, b: a / b,
            "BitAnd": lambda a, b: a & b,
            "BitOr": lambda a, b: a | b,
            "Not": lambda a: not a,
            "USub": lambda a: -a,
            "UAdd": lambda a: +a,
            "Lt": lambda a, b: a < b,
            "LtE": lambda a, b: a <= b,
            "Gt": lambda a, b: a > b,
            "GtE": lambda a, b: a >= b,
            "Eq": lambda a, b: a == b,
            "NotEq": lambda a, b: a != b,
        }
        result = functions[kind](*args)
        dtype = op["dtype"]
        if dtype == "float32":
            result = float(np.float32(result))
            if not math.isfinite(result):
                raise ValueError(
                    "This region replay requires finite arithmetic"
                )
        elif dtype == "int32":
            result = int(result)
            if not -(2**31) <= result < 2**31:
                raise ValueError(
                    "Runtime integer exceeds declared source type"
                )
        return result

    returned = {}

    def walk(start, execution, issue, stop=None):
        current = start
        while current != stop:
            block = blocks[current]
            if not execution and branch_form == "divergent_structured":
                return
            for identifier in block["instructions"]:
                op = ops[identifier]
                results = [
                    (lane, scalar(op, lane)) for lane in lanes(execution)
                ]
                for lane, result in results:
                    write(op["output"], lane, result)
                events.append(
                    dict(
                        instruction=identifier,
                        block=current,
                        issue_mask=issue,
                        execution_mask=execution,
                        memory_mask=execution
                        if op["opcode"] in ("load", "store")
                        else None,
                    )
                )
            term = block["terminator"]
            if term["kind"] == "return":
                for name, identifier in zip(
                    term["names"], term["values"], strict=True
                ):
                    returned[name] = {
                        lane: read(identifier, lane)
                        for lane in lanes(execution)
                    }
                return
            if term["kind"] == "jump":
                target = term["target"]
                transfer(current, target, execution)
                current = target
                continue
            true_mask = sum(
                1 << lane
                for lane in lanes(execution)
                if read(term["predicate"], lane)
            )
            false_mask = execution & ~true_mask
            if term.get("loop_header"):
                if true_mask not in (0, execution):
                    raise ValueError(
                        "Source fixed-loop induction must be uniform"
                    )
                target = term["true"] if true_mask else term["false"]
                transfer(current, target, execution)
                current = target
                continue
            join = term["join"]
            converted = (
                branch_form == "predicated_acyclic_arms"
                and not term.get("contains_cyclic_loop", False)
            )
            for target, active in (
                (term["true"], true_mask),
                (term["false"], false_mask),
            ):
                transfer(current, target, active)
                child_issue = issue if converted else active
                if active or converted:
                    walk(target, active, child_issue, join)
            current = join

    walk(program["entry"], entry_mask, entry_mask)
    return dict(
        outputs=returned,
        events=events,
        entry_mask=entry_mask,
        compiler_branch_form=branch_form,
        allocation_reused=True,
        physical_lane_checks=True,
    )
