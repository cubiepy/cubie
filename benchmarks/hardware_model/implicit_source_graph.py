"""Extract typed implicit execution regions under explicit warp regimes.

This source observer never requests native specialization. Runtime branch
choices describe a declared execution path; they are not compiler facts.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cubie import Solver
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cuda_simsafe import activemask, all_sync, any_sync, selp, unroll_if

from benchmarks import placement_landscape as placement
from benchmarks.hardware_model.buffer_descriptors import registry_layout
from benchmarks.hardware_model.expansion import CapturedGraph, source_receipt
from benchmarks.hardware_model import implicit_workload as workload
from benchmarks.hardware_model import source_value_graph as source
from benchmarks.hardware_model.workload import UNKNOWN, source_function


SCRIPT = Path(__file__).resolve()
FULL = 2**32 - 1


def uniform_regime(descriptor, newton_bodies=1, krylov_bodies=1):
    """Declare full-participation, successful-path body counts per call."""
    if (type(newton_bodies) is not int or newton_bodies < 0
            or type(krylov_bodies) is not int or krylov_bodies < 0):
        raise ValueError("Iteration body counts must be nonnegative integers")

    def linear(role):
        count = 0 if role["solver_type"] == "lu" else krylov_bodies
        return dict(entry_mask=FULL, active_masks=[FULL] * count,
                    terminal_active_mask=0)

    scenarios = {}
    for call in descriptor["step_calls"]:
        role = descriptor["roles"][call["role"]]
        if role["solver_type"] == "newton":
            scenarios[call["id"]] = dict(
                entry_mask=FULL, active_masks=[FULL] * newton_bodies,
                terminal_active_mask=0,
                linear_calls=[linear(descriptor["roles"]["main_linear"])
                              for _ in range(newton_bodies)],
            )
        else:
            scenarios[call["id"]] = linear(role)
    return scenarios


class RegionValues(source.SourceValues):
    """Retain typed values while selecting one declared runtime path."""

    def __init__(self, graph, descriptor, scenarios, branch_choices=None):
        super().__init__(graph)
        self.descriptor = descriptor
        self.scenarios = scenarios
        self.regime = workload.evaluate_regime(
            descriptor, scenarios, step_entry_mask=FULL,
        )
        self.role_functions = {
            key: graph.callables[role["function"]]
            for key, role in descriptor["roles"].items()
        }
        self.frames = []
        self.templates = []
        self.entered_calls = []
        self.decisions = {}
        self.control_stack = []
        self.last_branch_event = None
        self.branch_choices = branch_choices or {}
        self.used_branch_choices = set()
        self.primitives = {
            id(source.python_function(function)): name
            for function, name in (
                (activemask, "ActiveMask"), (all_sync, "AllSync"),
                (any_sync, "AnySync"), (selp, "Select"),
            )
        }

    def location(self, node):
        result = super().location(node)
        if self.frames:
            frame = self.frames[-1]
            result["runtime_region"] = dict(
                role=frame["role"], call=frame["call"],
                body_index=frame.get("body_index"),
                entry_mask=frame["scenario"]["entry_mask"],
                phase=frame.get("phase", "entry"),
            )
        if self.templates:
            template = dict(
                function=self.templates[-1], line=result["line"],
                syntax=result["syntax"],
                fixed_indices=[item for item in self.loop_indices
                               if not item.get("recurrent", False)],
            )
            result["hot_template_identity"] = hashlib.sha256(
                json.dumps(template, sort_keys=True).encode()
            ).hexdigest()
            result["hot_template"] = template
            result["template_is_not_native_copy_identity"] = True
        return result

    def primitive(self, kind, arguments, node, dtype, **details):
        values = [self.scalar(value, node) for value in arguments]
        return self.operation(kind, values, node, dtype=dtype, **details)[0]

    def operation(self, kind, operands, node, dtype=None, raw=UNKNOWN,
                  ordering=(), **details):
        return super().operation(
            kind, operands, node, dtype=dtype, raw=raw,
            ordering=set(ordering) | set(self.control_stack), **details,
        )

    def binary(self, node, left, right):
        operands = [self.scalar(value, node) for value in (left, right)]
        types = [self.values[value["identity"]]["dtype"]
                 for value in operands]
        if set(types) == {"literal_int", "int32"} and isinstance(
            node.op, (ast.BitAnd, ast.BitOr),
        ):
            index = types.index("literal_int")
            literal = operands[index]["raw"]
            if type(literal) is not int or not -(2**31) <= literal < 2**31:
                self.unknown(node, "Bitwise literal exceeds int32")
            operands[index] = self.operation(
                "cast", [operands[index]], node, dtype="int32",
                raw=np.int32(literal),
                conversion="exact_int32_literal_bitwise_operand",
            )[0]
            return self.primitive(type(node.op).__name__, operands, node,
                                  "int32")
        if types == ["bool", "bool"] and isinstance(
            node.op, (ast.BitAnd, ast.BitOr),
        ):
            return self.primitive(type(node.op).__name__, operands, node,
                                  "bool")
        if types == ["int32", "int32"]:
            methods = {ast.Add: lambda a, b: a + b,
                       ast.Sub: lambda a, b: a - b,
                       ast.Mult: lambda a, b: a * b,
                       ast.BitAnd: lambda a, b: a & b,
                       ast.BitOr: lambda a, b: a | b}
            method = methods.get(type(node.op))
            if method is None:
                self.unknown(node, "Unsupported integer operation")
            raw = UNKNOWN
            exact = None
            if all(value["raw"] is not UNKNOWN for value in operands):
                exact = method(*(int(value["raw"]) for value in operands))
                if not -(2**31) <= exact < 2**31:
                    self.unknown(node, "Integer constant exceeds int32")
                raw = np.int32(exact)
            return self.operation(
                type(node.op).__name__, operands, node, dtype="int32",
                raw=raw,
                integer_semantics=(
                    "conditional low-32-bit storage-compatible alternative"
                ),
                native_integer_width="unresolved before specialization",
                exact_constant_result=exact,
            )[0]
        return super().binary(node, left, right)

    def pure_boolean(self, node, environment):
        """Admit eager evaluation only for exact known pure primitives."""
        for child in ast.walk(node):
            if isinstance(child, ast.NamedExpr):
                return False
            if isinstance(child, ast.Call):
                target = source.constant(
                    child.func, self.raw_environment(environment),
                )
                if target not in (abs, min, max, math.fabs, math.sqrt):
                    return False
                if child.keywords:
                    return False
        return True

    def expression(self, node, environment):
        if isinstance(node, ast.BinOp):
            return self.binary(node, self.expression(node.left, environment),
                               self.expression(node.right, environment))
        if isinstance(node, ast.Compare):
            folded = self.condition_constant(node, environment)
            if folded is not UNKNOWN:
                return source.item(folded)
            if len(node.ops) != 1:
                self.unknown(node, "Chained runtime comparison")
            arguments = [self.scalar(self.expression(value, environment),
                                     node)
                         for value in (node.left, node.comparators[0])]
            types = [self.values[value["identity"]]["dtype"]
                     for value in arguments]
            if set(types) == {"bool", "int32"}:
                position = types.index("bool")
                arguments[position] = self.operation(
                    "cast", [arguments[position]], node, dtype="int32",
                    conversion="exact_bool_to_int32_zero_or_one",
                )[0]
                types[position] = "int32"
            if types[0] != types[1] or types[0] not in (
                "float32", "int32", "uint32", "bool",
            ):
                self.unknown(node, f"Unproved comparison promotion {types}")
            kind = "Compare" + type(node.ops[0]).__name__
            if kind not in {"CompareEq", "CompareNotEq", "CompareLt",
                            "CompareLtE", "CompareGt", "CompareGtE"}:
                self.unknown(node, "Unsupported runtime comparison")
            return self.primitive(kind, arguments, node, "bool")
        if isinstance(node, ast.BoolOp):
            # Only pure expressions are admitted to eager boolean lowering.
            if not self.pure_boolean(node, environment):
                self.unknown(node, "Short-circuit expression has effects")
            result = self.expression(node.values[0], environment)
            for child in node.values[1:]:
                argument = self.expression(child, environment)
                values = [self.scalar(value, node)
                          for value in (result, argument)]
                if any(self.values[value["identity"]]["dtype"] != "bool"
                       for value in values):
                    self.unknown(node, "Boolean operation needs bool values")
                result = self.primitive(type(node.op).__name__, values,
                                        node, "bool")
            return result
        if isinstance(node, ast.IfExp):
            if not all(isinstance(arm, (ast.Name, ast.Constant))
                       for arm in (node.body, node.orelse)):
                self.unknown(node, "Conditional expression needs region merge")
            predicate = self.scalar(self.expression(node.test, environment),
                                    node)
            arms = [self.scalar(self.expression(arm, environment), node)
                    for arm in (node.body, node.orelse)]
            types = [self.values[value["identity"]]["dtype"]
                     for value in [predicate] + arms]
            if types[0] != "bool" or types[1] != types[2]:
                self.unknown(
                    node, "Conditional values have incompatible types",
                )
            return self.primitive(
                "Select", [predicate] + arms, node, types[1],
                conditional_origin="existing scalar value arms, no effects",
            )
        return super().expression(node, environment)

    def call(self, node, environment):
        target = self.expression(node.func, environment)["raw"]
        function = source.python_function(target)
        primitive = self.primitives.get(id(function))
        if primitive is None and target not in (
            abs, min, max, math.sqrt, math.fabs,
        ):
            return super().call(node, environment)
        if node.keywords:
            self.unknown(node, "Primitive keywords are unsupported")
        arguments = [self.scalar(self.expression(arg, environment), node)
                     for arg in node.args]
        types = [self.values[value["identity"]]["dtype"]
                 for value in arguments]
        if primitive == "ActiveMask":
            if arguments:
                self.unknown(node, "ActiveMask arguments")
            entry = (self.frames[-1]["scenario"]["entry_mask"]
                     if self.frames else self.regime["step_entry_mask"])
            result = self.primitive(primitive, [], node, "uint32",
                                    declared_active_entry_mask=entry)
            self.values[result["identity"]]["declared_lane_mask"] = entry
            return result
        if primitive in ("AllSync", "AnySync"):
            if types != ["uint32", "bool"] or not self.frames:
                self.unknown(node, f"Vote types {types}, "
                             f"role frames {len(self.frames)}")
            frame = self.frames[-1]
            entry = frame["scenario"]["entry_mask"]
            participating = self.values[arguments[0]["identity"]].get(
                "declared_lane_mask"
            )
            if participating != entry:
                self.unknown(node, "Vote mask differs from active entry")
            result = self.primitive(
                primitive, arguments, node, "bool",
                participating_mask=participating, active_entry_mask=entry,
                mask_equality="explicit uniform participating-path contract",
                primitive_source=source_receipt(function),
            )
            count = len(frame["scenario"]["active_masks"])
            index = frame.get("body_index")
            if primitive != "AllSync":
                self.unknown(node, "AnySync needs an explicit mask decision")
            decision = count == 0 if index is None else index >= count
            self.decisions[result["identity"]] = decision
            self.nodes[self.values[result["identity"]]["producer"]][
                "declared_result"
            ] = decision
            return result
        if primitive == "Select":
            if len(types) != 3 or types[0] != "bool" or types[1] != types[2]:
                self.unknown(node, "Select has incompatible typed arms")
            if types[1] not in ("float32", "int32", "uint32", "bool"):
                self.unknown(node, "Unsupported Select result type")
            return self.primitive(primitive, arguments, node, types[1])
        if any(dtype != "float32" for dtype in types):
            self.unknown(node, f"Primitive FP32 input types differ {types}")
        if target in (abs, math.fabs, math.sqrt) and len(arguments) == 1:
            kind = "Sqrt" if target is math.sqrt else "Abs"
        elif target in (min, max) and len(arguments) == 2:
            kind = "Minimum" if target is min else "Maximum"
        else:
            self.unknown(node, "Unsupported typed numeric primitive")
        return self.primitive(
            kind, arguments, node, "float32",
            callable_identity=target.__module__ + "." + target.__name__,
            source_spelling=ast.unparse(node.func),
        )

    def condition_constant(self, node, environment):
        return super().condition(node, environment)

    def condition(self, node, environment):
        self.last_branch_event = None
        folded = self.condition_constant(node, environment)
        if folded is not UNKNOWN:
            return folded
        predicate = self.scalar(self.expression(node, environment), node)
        if self.values[predicate["identity"]]["dtype"] != "bool":
            self.unknown(node, "Runtime branch does not have bool predicate")
        if predicate["identity"] in self.decisions:
            choice = self.decisions[predicate["identity"]]
            reason = "declared uniform convergence-mask regime"
        else:
            key = f"{Path(self.path).name}:{node.lineno}"
            if key not in self.branch_choices:
                self.unknown(node, "Runtime branch needs explicit choice "
                             + key)
            self.used_branch_choices.add(key)
            choice = self.branch_choices[key]
            if type(choice) is not bool:
                self.unknown(node, "Branch choice must be bool")
            reason = "explicit runtime-path assumption"
        self.controls.append(dict(
            kind="runtime_branch_choice", source=self.location(node),
            predicate=predicate["identity"], choice=choice, reason=reason,
            is_codegen_constant=False,
        ))
        _, self.last_branch_event = self.operation(
            "BranchDecision", [predicate], node, selected_path=choice,
            decision_reason=reason, is_codegen_constant=False,
        )
        return choice

    def invoke(self, function, bound, call_site=None):
        template = dict(
            function=self.function_ids[id(function)],
            source=source_receipt(function),
            parameters={},
        )
        for name, value in bound.items():
            if value["reference"] is not None:
                reference = value["reference"]
                template["parameters"][name] = {
                    key: reference[key]
                    for key in ("dtype", "shape", "bytes", "itemsize")
                }
            elif value["identity"] is not None:
                scalar = self.values[value["identity"]]
                template["parameters"][name] = dict(dtype=scalar["dtype"])
                if scalar["kind"] == "constant":
                    template["parameters"][name]["constant"] = scalar[
                        "constant"
                    ]
            else:
                snapshot = source.snapshot(value["raw"])
                template["parameters"][name] = (
                    snapshot if snapshot is not UNKNOWN else "opaque"
                )
        self.templates.append(template)
        role = next((name for name, target in self.role_functions.items()
                     if target is function), None)
        frame = None
        if role is not None:
            if role == "main_linear" and self.frames and (
                self.frames[-1]["role"] == "main_newton"
            ):
                parent = self.frames[-1]
                index = parent["body_index"]
                scenario = parent["scenario"]["linear_calls"][index]
                call = parent["call"] + f".linear{index}"
            else:
                calls = [entry for entry in self.descriptor["step_calls"]
                         if entry["role"] == role
                         and entry["id"] not in self.entered_calls]
                if not calls:
                    self.unknown(source_function(function),
                                 "Unexpected role call")
                call = calls[0]["id"]
                self.entered_calls.append(call)
                scenario = self.scenarios[call]
            if any(value != scenario["entry_mask"]
                   for value in scenario["active_masks"]):
                self.unknown(source_function(function),
                             "Nonuniform path needs masked region merge")
            frame = dict(role=role, call=call, scenario=scenario)
            self.frames.append(frame)
        try:
            return super().invoke(function, bound, call_site)
        finally:
            if frame is not None:
                self.frames.pop()
            self.templates.pop()

    def block(self, statements, environment):
        for node in statements:
            if isinstance(node, ast.If):
                condition = self.condition(node.test, environment)
                self.controls.append(dict(
                    kind="selected_source_path", source=self.location(node),
                    condition=bool(condition),
                ))
                branch = self.last_branch_event
                if branch is not None:
                    self.control_stack.append(branch)
                returned, value = self.block(
                    node.body if condition else node.orelse, environment,
                )
                if branch is not None:
                    self.control_stack.pop()
                if returned:
                    return returned, value
            elif isinstance(node, ast.For):
                iterator = node.iter
                if (not isinstance(iterator, ast.Call)
                        or self.expression(iterator.func, environment)["raw"]
                        is not unroll_if or len(iterator.args) != 2):
                    self.unknown(node, "Loop lacks actual unroll directive")
                bounds = iterator.args[0]
                if (not isinstance(bounds, ast.Call)
                        or self.expression(bounds.func, environment)["raw"]
                        is not range or bounds.keywords):
                    self.unknown(node, "Unsupported loop range")
                indices = list(range(*(self.index(arg, environment)
                                       for arg in bounds.args)))
                flag = source.constant(iterator.args[1],
                                       self.raw_environment(environment))
                if flag != (True, None):
                    self.unknown(node, "Fixed partial/rolled loop needs "
                                 "dynamic-index region lowering")
                frame = self.frames[-1] if self.frames else None
                recurrence = None
                if frame is not None:
                    role = self.descriptor["roles"][frame["role"]]
                    recurrence = role["iteration"].get("source_region")
                recurrent = (recurrence is not None
                             and node.lineno == recurrence["line"]
                             and self.stack[-1] == id(
                                 self.role_functions[frame["role"]]))
                if recurrent:
                    count = len(frame["scenario"]["active_masks"])
                    if len(indices) != role["cap"]:
                        self.unknown(node, "Recurrent source cap differs")
                    indices = indices[:min(role["cap"], count + 1)]
                self.controls.append(dict(
                    kind="recurrent_execution_trace" if recurrent else
                    "full_source_expansion", source=self.location(node),
                    indices=indices, directive=list(flag),
                    code_copies_are_execution_count=False,
                ))
                if node.orelse or len(indices) > 10000:
                    self.unknown(node, "Unsupported loop expansion")
                for index in indices:
                    if recurrent:
                        frame["body_index"] = index
                        frame["phase"] = (
                            "body" if index < count else "exit_vote"
                        )
                    self.assign(node.target, source.item(index), environment)
                    self.loop_indices.append(dict(line=node.lineno,
                                                  index=index,
                                                  recurrent=recurrent))
                    returned, value = self.block(node.body, environment)
                    self.loop_indices.pop()
                    if returned == "break":
                        break
                    if returned:
                        return returned, value
                if recurrent:
                    frame.pop("body_index", None)
                    frame["phase"] = "exit"
            elif isinstance(node, ast.Break):
                return "break", source.item(None)
            else:
                returned, value = super().block([node], environment)
                if returned:
                    return returned, value
        return False, source.item(None)


def describe_implicit_source(solver, scenarios, branch_choices=None):
    """Capture one actual implicit step under supplied runtime scenarios."""
    descriptor = workload.describe_implicit_workload(solver)
    step_owner = solver.kernel.single_integrator._algo_step
    graph = CapturedGraph()
    graph.add_function(step_owner.step_function, "algorithm_step")
    engine = RegionValues(graph, descriptor, scenarios, branch_choices)
    step, bound, caller = source.caller_bindings(engine, solver)
    returned = engine.invoke(step, bound)
    if set(engine.entered_calls) != set(scenarios):
        raise ValueError("Source path did not enter every supplied role call")
    unused = set(engine.branch_choices) - engine.used_branch_choices
    if unused:
        raise ValueError(f"Unused runtime branch choices: {sorted(unused)}")
    outputs = {value["identity"] for cell, value in engine.cells.items()
               if cell[0] in engine.boundary}
    if returned["raw"] is not None or returned["identity"] is not None:
        outputs.add(engine.scalar(returned, source_function(step))["identity"])
    if any(dispatcher.overloads for dispatcher in graph.dispatchers):
        raise ValueError("Native specialization appeared during extraction")
    consumers = {}
    value_edges = []
    for node in engine.nodes:
        for value in node["inputs"]:
            consumers.setdefault(value, set()).add(node["id"])
            producer = engine.values[value]["producer"]
            if producer is not None:
                value_edges.append(dict(
                    producer=producer, consumer=node["id"], value=value,
                ))
    certificates = []
    for call in engine.calls:
        if call["kind"] != "source_call":
            continue
        nodes = engine.nodes[call["first_node"]:call["end_node"]]
        membership = {node["id"] for node in nodes}
        used = {value for node in nodes
                for value in node["inputs"] + node["outputs"]}
        incoming = {value for value in used
                    if engine.values[value]["producer"] not in membership
                    and engine.values[value]["kind"] != "constant"}
        outgoing = set()
        for value in engine.values:
            identifier = value["id"]
            producer = value["producer"]
            if value["kind"] == "constant":
                continue
            existed = producer is None or producer < call["end_node"]
            later = identifier in outputs or any(
                node >= call["end_node"]
                for node in consumers.get(identifier, ())
            )
            if existed and later:
                outgoing.add(identifier)
                if producer is None or producer < call["first_node"]:
                    incoming.add(identifier)
        certificates.append(dict(
            context=call["context"], function=call["function"],
            node_ids=sorted(membership), live_ins=sorted(incoming),
            observable_outputs=sorted(outgoing),
            scope="lexical interval with conservative caller live-through",
        ))
    return dict(
        schema=1, kind="typed_implicit_execution_region",
        provenance=dict(extractor=source_receipt(describe_implicit_source),
                        dependencies=[source_receipt(source.SourceValues),
                                      source_receipt(
                                          workload.evaluate_regime
                                      )],
                        functions=graph.functions),
        workload=descriptor, caller=caller,
        registry=registry_layout(step_owner),
        allocations=list(engine.allocations.values()), calls=engine.calls,
        values=engine.values, nodes=engine.nodes, controls=engine.controls,
        value_edges=value_edges,
        aliases=engine.aliases, live_ins=engine.input_ids,
        certificates=certificates,
        required_control_nodes=[node["id"] for node in engine.nodes
                                if node["kind"] == "BranchDecision"],
        observable_values=sorted(outputs),
        final_cells=[dict(cell=list(cell), value=value["identity"],
                          boundary=cell[0] in engine.boundary)
                     for cell, value in sorted(engine.cells.items())],
        regime=engine.regime, branch_choices=engine.branch_choices,
        operation_inventory=[
            dict(kind=kind, inputs=list(inputs), outputs=list(outputs),
                 count=count)
            for (kind, inputs, outputs), count in Counter(
                (node["kind"],
                 tuple(engine.values[key]["dtype"]
                       for key in node["inputs"]),
                 tuple(engine.values[key]["dtype"]
                       for key in node["outputs"]))
                for node in engine.nodes
            ).items()
        ],
        compilation_check=dict(native_overloads=0,
                               batch_kernel_requested=False,
                               device_function_executed=False),
        semantic_contract=dict(
            path="one declared uniform successful path",
            floating_point=(
                "FP32 source operations; selected-path replay assumes finite "
                "inputs and intermediates"
            ),
            nan_sensitive_primitives=(
                "source callable identity is retained; Python min/max is not "
                "identified with fmin/fmax"
            ),
            integer_values=(
                "int32 storage-compatible abstraction; constant arithmetic "
                "is admitted only in signed range and dynamic arithmetic "
                "records a conditional low-32-bit alternative"
            ),
            native_integer_width="unresolved before specialization",
            vote_mask="participating mask equals declared active-entry mask",
        ),
        limitations=[
            "Runtime branch choices are workload hypotheses",
            "Executed recurrent bodies are not native code-copy counts",
            "No native predicate/GPR allocation is inferred here",
            "Nonuniform branches and partial fixed loops reject",
            "Caller cells are conservative exit observables, not a use proof",
        ],
    )


def main():
    """Construct and inspect one source-only implicit scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="lorenz")
    parser.add_argument("--algo", default="kvaerno3")
    parser.add_argument("--linear-solver", default="lu")
    parser.add_argument("--newton-bodies", type=int, default=1)
    parser.add_argument("--krylov-bodies", type=int, default=1)
    parser.add_argument("--branch-choices", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output / "codegen"
    cache.mkdir()
    previous = get_cache_root_override()
    solver = None
    try:
        set_cache_root(cache.resolve())
        system = placement.SYSTEMS[args.system]["build"]()
        kwargs = placement.solver_kwargs(args.system, args.algo)
        kwargs["linear_correction_type"] = workload.PUBLIC_LINEAR_TYPES[
            args.linear_solver
        ]
        solver = Solver(system, **kwargs)
        descriptor = workload.describe_implicit_workload(solver)
        scenarios = uniform_regime(descriptor, args.newton_bodies,
                                   args.krylov_bodies)
        choices = {} if args.branch_choices is None else json.loads(
            args.branch_choices.read_text()
        )
        result = describe_implicit_source(solver, scenarios, choices)
        workload.write_json(args.output / "graph.json", result)
        print(json.dumps(dict(status="TYPED_IMPLICIT_SOURCE_PASS",
                              nodes=len(result["nodes"]),
                              values=len(result["values"]))))
    finally:
        if solver is not None:
            solver.close()
        set_cache_root(previous)


if __name__ == "__main__":
    main()
