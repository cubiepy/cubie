"""Construct conditional native plans from complete typed ERK graphs.

The default CLI reads JSON and imports no CUDA package. ``construct``
starts an isolated source-construction worker, never a native compiler.
Modeled instructions, registers and addresses are explicit hypotheses.
"""

import argparse
import ast
from collections import Counter, defaultdict, deque, OrderedDict
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
WORDS = {"float32": 1, "int32": 1, "uint32": 1, "bool": 1}
LOCAL = "local"
SHARED = "shared"
RULE_SCOPE = "conditional_model_not_verified_native_lowering"


def digest(path):
    """Return the exact file-byte SHA256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Create a finite JSON receipt without overwriting evidence."""
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")


def exact_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def round_up(value, unit):
    return ((value + unit - 1) // unit) * unit


def constant_payload(value, dtype):
    """Normalize supported finite constants, preserving FP32 bits."""
    if isinstance(value, dict):
        value = value["value"]
    if dtype == "float32":
        raw = np.float32(value)
        if not np.isfinite(raw):
            raise ValueError("Nonfinite constant is unsupported")
        if raw != 0 and abs(raw) < np.finfo(np.float32).tiny:
            raise ValueError("Subnormal constant requires a flush rule")
        return dict(
            dtype=dtype, bits=struct.pack("<f", raw).hex(), value=float(raw)
        )
    if dtype in ("int32", "uint32", "bool"):
        number = int(value)
        low, high = (-(2**31), 2**31) if dtype == "int32" else (0, 2**32)
        if dtype == "bool":
            low, high = 0, 2
        if not low <= number < high:
            raise ValueError("Constant narrowing is not admitted")
        return dict(dtype=dtype, value=number)
    if dtype in ("literal_int", "literal_float"):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Invalid source literal")
        return dict(dtype=dtype, value=value)
    raise ValueError(f"Unsupported constant type {dtype}")


def validate_graph(graph):
    """Check complete graph, ordered identities and source byte receipts."""
    if (
        graph.get("kind") != "source_value_frontier_certificate"
        or graph.get("workload", {}).get("family") != "ERK"
        or graph.get("compilation_check", {}).get("native_overloads") != 0
    ):
        raise ValueError("Need a complete uncompiled ERK value graph")
    nodes, values = graph["nodes"], graph["values"]
    if [node["id"] for node in nodes] != list(range(len(nodes))):
        raise ValueError("Noncontiguous source nodes")
    if [value["id"] for value in values] != list(range(len(values))):
        raise ValueError("Noncontiguous source values")
    for node in nodes:
        for key in node["inputs"]:
            producer = values[key]["producer"]
            if producer is not None and producer >= node["id"]:
                raise ValueError("Source graph is not topological")
        for key in node["outputs"]:
            if values[key]["producer"] != node["id"]:
                raise ValueError("Source output/producer mismatch")
        if any(before >= node["id"] for before in node["order_predecessors"]):
            raise ValueError("Invalid source memory order")
    for control in graph["controls"]:
        if control["kind"] == "full_source_expansion" and control["flag"] != [
            True,
            None,
        ]:
            raise ValueError("NativePlan currently requires full expansion")
    receipts = [graph["provenance"]["extractor"], graph["caller"]["source"]]
    receipts += graph["provenance"]["dependencies"]
    receipts += [
        dict(
            path=fn["source"]["source_path"],
            sha256=fn["source"]["source_sha256"],
        )
        for fn in graph["provenance"]["functions"]
    ]
    checked = {}
    for record in receipts:
        if digest(record["path"]) != record["sha256"]:
            raise ValueError(f"Source bytes changed: {record['path']}")
        checked[record["path"]] = record["sha256"]
    return checked


def validate_construction(graph):
    """Bind source construction, placement and actual shared stride."""
    record = graph.get("native_plan_construction")
    if not record:
        raise ValueError("Missing source-construction receipt")
    directory = Path(record["cache_root"]).parent
    worker = directory / "construct.py"
    request_path = directory / "construction_request.json"
    request = json.loads(request_path.read_text())
    result = json.loads((directory / "worker.json").read_text())
    source = directory / "benchmark_source.py"
    if (
        digest(worker) != record["constructor_sha256"]
        or digest(request_path) != record["request_sha256"]
        or digest(source) != request["model_sha256"]
        or result["returncode"] != 0
        or result["worker_sha256"] != digest(worker)
    ):
        raise ValueError("Construction identity or worker completion differs")
    if result["command"] != [
        sys.executable,
        str(worker.resolve()),
        str(directory.resolve()),
    ]:
        raise ValueError("Executed constructor command differs")
    tree = ast.parse(source.read_text())
    literal = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "CONSTRUCTOR"
            for t in node.targets
        )
    )
    if worker.read_text() != literal:
        raise ValueError("Worker is not the retained constructor literal")
    for key in ("system", "algo", "placement"):
        if request[key] != record[key]:
            raise ValueError("Actual construction does not match its request")
    elements = exact_int(record["shared_elements_per_run"], "shared elements")
    padding = exact_int(record["shared_padding_elements"], "shared padding")
    if padding not in (0, 1) or record["shared_stride_bytes"] != 4 * (
        elements + padding
    ):
        raise ValueError("Shared stride is not the captured padded layout")
    if padding != int(elements != 0 and elements % 2 == 0):
        raise ValueError("Shared padding differs from the captured FP32 rule")
    extent = max(
        (
            allocation["view"]["offset"] + allocation["view"]["bytes"]
            for allocation in graph["allocations"]
            if allocation["view"]["storage"] == "caller:shared_scratch"
        ),
        default=0,
    )
    if extent != 4 * elements:
        raise ValueError("Shared capacity differs from actual captured views")
    entry = [
        item
        for item in graph["registry"]
        if item["owner"] == "step" and item["name"] == "stage_accumulator"
    ]
    if len(entry) != 1 or entry[0]["resolved_location"] != record["placement"]:
        raise ValueError("Requested placement differs from the resolved owner")
    source = record["shared_stride_source"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("The shared-stride source bytes changed")
    expected = (
        3
        if record["system"] == "lorenz"
        else int(
            record["system"][5:],
        )
    )
    if not (
        record["actual_system_size"]
        == graph["workload"]["n_states"]
        == expected
    ):
        raise ValueError("Constructed dimension differs")
    for allocation in graph["allocations"]:
        ref = allocation["view"]
        if (
            ref["storage"] == "caller:shared_scratch"
            and ref["offset"] + ref["bytes"] > 4 * elements
        ):
            raise ValueError("Shared view exceeds the actual per-run layout")
    return record


class Lowering:
    """Lower source values and memory effects into a typed model graph."""

    def __init__(self, graph, materialization="promote", contract=False):
        if materialization not in ("promote", "addressable"):
            raise ValueError("Unknown materialization scenario")
        self.graph = graph
        self.mode = materialization
        self.contract = contract
        self.nodes = []
        self.values = []
        self.source_values = {}
        self.source_nodes = {}
        self.read_values = {}
        self.rewrites = []
        self.constants = {}
        self.spaces = {}
        self.layouts = {}
        self.source = graph["values"]
        self.original_nodes = graph["nodes"]
        self.initial_memory = {}
        self.final_memory = {}
        self.promoted = set()
        self.bases = {}
        self.observables = set()
        # One modeled stack base is retained even in a promoted plan, so
        # allocating spill instructions never changes the input ABI.
        self.bases[LOCAL] = self.value("uint32", "base", "base:local")
        self.bind_layouts()

    def value(self, dtype, kind, semantic, **details):
        if dtype not in WORDS:
            raise ValueError(f"Unsupported lowered dtype {dtype}")
        key = len(self.values)
        self.values.append(
            dict(
                id=key,
                dtype=dtype,
                words=WORDS[dtype],
                kind=kind,
                semantic=semantic,
                **details,
            )
        )
        return key

    def literal(self, payload):
        token = json.dumps(payload, sort_keys=True)
        if token not in self.constants:
            self.constants[token] = self.value(
                payload["dtype"],
                "constant",
                "constant:" + token,
                constant=payload,
            )
        return self.constants[token]

    def bind_layouts(self):
        cursor = 0
        for allocation in self.graph["allocations"]:
            ref = allocation["view"]
            storage = ref["storage"]
            space = SHARED if storage == "caller:shared_scratch" else LOCAL
            if not (
                storage.startswith("local:")
                or storage
                in ("caller:persistent_local", "caller:shared_scratch")
            ):
                raise ValueError(f"Unproved storage space {storage}")
            extent = ref["offset"] + ref["bytes"]
            if storage not in self.layouts:
                self.layouts[storage] = dict(
                    storage=storage, space=space, bytes=extent
                )
            else:
                self.layouts[storage]["bytes"] = max(
                    extent,
                    self.layouts[storage]["bytes"],
                )
            self.spaces[storage] = space
            if self.mode == "promote" and space == LOCAL:
                self.promoted.add(storage)
        for storage, layout in self.layouts.items():
            if storage in self.promoted:
                layout["frame_offset"] = None
            elif layout["space"] == LOCAL:
                cursor = round_up(cursor, 4)
                layout["frame_offset"] = cursor
                cursor += layout["bytes"]
            else:
                layout["frame_offset"] = 0
        self.named_frame_bytes = round_up(cursor, 4)
        if any(
            v["space"] == SHARED and v["bytes"] for v in self.layouts.values()
        ):
            self.bases[SHARED] = self.value(
                "uint32",
                "base",
                "base:shared",
            )

    def mapped(self, source_value, node=None):
        if node is not None:
            for before in reversed(node["order_predecessors"]):
                original = self.original_nodes[before]
                if before in self.read_values and original["inputs"] == [
                    source_value
                ]:
                    return self.read_values[before]
        if source_value in self.source_values:
            return self.source_values[source_value]
        value = self.source[source_value]
        if value["kind"] == "constant":
            payload = constant_payload(value["constant"], value["dtype"])
            if payload["dtype"].startswith("literal_"):
                raise ValueError("An uncast literal reached a native operand")
            result = self.literal(payload)
        elif value["kind"] == "live_in":
            label = value["label"]
            if isinstance(label, list) and label[0] not in self.promoted:
                raise ValueError("Addressable cell used without its read")
            result = self.value(
                value["dtype"],
                "live_in",
                f"source:{source_value}",
                label=label,
            )
        else:
            raise ValueError(f"Unmapped source expression {source_value}")
        self.source_values[source_value] = result
        return result

    def semantic(self, source_value):
        value = self.source[source_value]
        if source_value in self.source_values:
            return self.values[self.source_values[source_value]]["semantic"]
        if value["kind"] == "live_in":
            return f"source:{source_value}"
        return self.values[self.mapped(source_value)]["semantic"]

    def emit(self, opcode, inputs, output, source_ids, memory=None):
        key = len(self.nodes)
        before = set()
        for source_id in source_ids:
            for parent in self.original_nodes[source_id]["order_predecessors"]:
                before.update(self.source_nodes.get(parent, []))
        for operand in inputs:
            producer = self.values[operand].get("producer")
            if producer is not None:
                before.add(producer)
        self.nodes.append(
            dict(
                id=key,
                opcode=opcode,
                inputs=list(inputs),
                outputs=[] if output is None else [output],
                predecessors=sorted(before),
                source_nodes=list(source_ids),
                memory=memory,
            )
        )
        if output is not None:
            self.values[output]["producer"] = key
        for source_id in source_ids:
            self.source_nodes[source_id] = [key]
        return key

    def memory(self, node):
        cell = node["cell"]
        storage, low, high, dtype = cell
        if dtype != "float32" or high - low != 4 or low % 4:
            raise ValueError("Only exact aligned FP32 cells are admitted")
        if storage not in self.layouts:
            raise ValueError("Memory cell lacks actual allocation identity")
        if high > self.layouts[storage]["bytes"]:
            raise ValueError("Memory access exceeds its captured allocation")
        if storage in self.promoted:
            self.source_nodes[node["id"]] = sorted(
                set(
                    parent
                    for before in node["order_predecessors"]
                    for parent in self.source_nodes.get(before, [])
                )
            )
            self.rewrites.append(
                dict(
                    rule="promote_exact_cell",
                    source_nodes=[node["id"]],
                    cell=cell,
                )
            )
            return
        space = self.spaces[storage]
        inputs = [self.bases[space]]
        key = json.dumps(cell)
        details = dict(
            kind="named",
            cell=cell,
            space=space,
            bytes=4,
            offset=self.layouts[storage]["frame_offset"] + low,
            expected_semantic=self.semantic(node["inputs"][0]),
        )
        if node["kind"] == "element_read_alias":
            source_value = node["inputs"][0]
            if self.source[source_value]["kind"] == "live_in":
                self.initial_memory[key] = details["expected_semantic"]
            output = self.value(dtype, "load", details["expected_semantic"])
            self.read_values[node["id"]] = output
            opcode = "LDL" if space == LOCAL else "LDS"
            details["access"] = "read"
        else:
            inputs.append(self.mapped(node["inputs"][0], node))
            output = None
            opcode = "STL" if space == LOCAL else "STS"
            details["access"] = "write"
        self.emit(opcode, inputs, output, [node["id"]], details)
        self.rewrites.append(
            dict(
                rule="addressable_exact_cell",
                source_nodes=[node["id"]],
                cell=cell,
                opcode=opcode,
            )
        )

    def build(self):
        consumers = defaultdict(list)
        for node in self.original_nodes:
            for value in node["inputs"]:
                consumers[value].append(node["id"])
        root = next(
            item
            for item in self.graph["certificates"]
            if item["function"] == "f0"
        )
        observable = set(root["observable_outputs"])
        fusion = {}
        if self.contract:
            for node in self.original_nodes:
                if node["kind"] != "Add":
                    continue
                for operand in node["inputs"]:
                    producer = self.source[operand]["producer"]
                    if producer is None:
                        continue
                    product = self.original_nodes[producer]
                    if (
                        product["kind"] == "Mult"
                        and self.source[operand]["dtype"] == "float32"
                        and consumers[operand] == [node["id"]]
                        and operand not in observable
                    ):
                        fusion[node["id"]] = producer
                        break
        skipped = set(fusion.values())
        for node in self.original_nodes:
            source_id = node["id"]
            if source_id in skipped:
                continue
            kind = node["kind"]
            if kind.startswith("element_"):
                self.memory(node)
                continue
            (output_id,) = node["outputs"]
            dtype = self.source[output_id]["dtype"]
            raw = self.source[node["inputs"][0]]
            if kind == "cast" and raw["kind"] == "constant":
                payload = constant_payload(raw["constant"], dtype)
                self.source_values[output_id] = self.literal(payload)
                self.source_nodes[source_id] = []
                self.rewrites.append(
                    dict(
                        rule="typed_literal_cast",
                        source_nodes=[source_id],
                        result=payload,
                    )
                )
                continue
            if kind == "cast" and raw["dtype"] == dtype:
                self.source_values[output_id] = self.mapped(
                    node["inputs"][0],
                    node,
                )
                self.source_nodes[source_id] = sorted(
                    set(
                        parent
                        for before in node["order_predecessors"]
                        for parent in self.source_nodes.get(before, [])
                    )
                )
                self.rewrites.append(
                    dict(rule="same_type_cast_alias", source_nodes=[source_id])
                )
                continue
            ids = [source_id]
            if source_id in fusion:
                product = self.original_nodes[fusion[source_id]]
                product_value = product["outputs"][0]
                inputs = [
                    self.mapped(value, product) for value in product["inputs"]
                ]
                inputs.append(
                    self.mapped(
                        next(
                            value
                            for value in node["inputs"]
                            if value != product_value
                        ),
                        node,
                    )
                )
                ids = [product["id"], source_id]
                opcode = "FFMA"
            else:
                inputs = [self.mapped(value, node) for value in node["inputs"]]
                opcodes = {
                    "Add": "FADD",
                    "Sub": "FSUB",
                    "Mult": "FMUL",
                    "USub": "FNEG",
                    "Div": "FDIV",
                }
                if dtype != "float32" or kind not in opcodes:
                    raise ValueError(f"No typed lowering for {kind}/{dtype}")
                opcode = opcodes[kind]
            output = self.value(dtype, "expression", f"source:{output_id}")
            self.source_values[output_id] = output
            if opcode == "FDIV":
                reciprocal = self.value(
                    dtype, "expression", f"reciprocal:{source_id}"
                )
                self.emit("MUFU.RCP", [inputs[1]], reciprocal, ids)
                self.emit("FMUL", [inputs[0], reciprocal], output, ids)
                rule = "approx_reciprocal_then_multiply"
            else:
                self.emit(opcode, inputs, output, ids)
                rule = "single_use_fma" if opcode == "FFMA" else "typed_op"
            self.rewrites.append(
                dict(
                    rule=rule,
                    source_nodes=ids,
                    dtype=dtype,
                    opcode=opcode,
                    qualification=RULE_SCOPE,
                )
            )
        boundary_values = set()
        for item in self.graph["final_cells"]:
            if not item["boundary"]:
                continue
            cell, value = item["cell"], item["value"]
            boundary_values.add(value)
            if cell[0] in self.promoted:
                self.observables.add(self.mapped(value))
            else:
                self.final_memory[json.dumps(cell)] = self.semantic(value)
        for value in observable - boundary_values:
            self.observables.add(self.mapped(value))
        # Keep only arithmetic needed by side effects or observable values.
        roots = {
            node["id"]
            for node in self.nodes
            if node["memory"] and node["memory"]["access"] == "write"
        }
        roots.update(
            self.values[value]["producer"]
            for value in self.observables
            if "producer" in self.values[value]
        )
        required, pending = set(), list(roots)
        while pending:
            current = pending.pop()
            if current not in required:
                required.add(current)
                pending.extend(self.nodes[current]["predecessors"])
        nodes = [node for node in self.nodes if node["id"] in required]
        used = set(self.observables) | set(self.bases.values())
        used.update(
            value
            for node in nodes
            for value in node["inputs"] + node["outputs"]
        )
        return dict(
            nodes=nodes,
            values=[v for v in self.values if v["id"] in used],
            observable_values=sorted(self.observables),
            initial_memory=self.initial_memory,
            final_memory=self.final_memory,
            layouts=list(self.layouts.values()),
            allocation_provenance=self.graph["allocations"],
            registry=self.graph["registry"],
            named_local_frame_bytes=self.named_frame_bytes,
            rewrites=self.rewrites,
            source_node_mapping=self.source_nodes,
            caller_cuts=[
                dict(
                    context=c["context"],
                    source_live_ins=c["live_ins"],
                    source_outputs=c["observable_outputs"],
                )
                for c in self.graph["certificates"]
            ],
            assumptions=[
                "Complete fully expanded ERK; captured aliases are exhaustive",
                "All known helpers inline; no external call ABI instructions",
                "FP32/int32 constants materialize by MOV when required",
                "FSUB/FNEG model FADD with signed/zero operand forms",
                "Division is reciprocal then multiply, no refinement",
                "Caller supplies bases; offsets fit native operands",
                "One common 32-bit local stack base remains live",
                "Shared base is one additional supplied 32-bit GPR",
                "No hidden kernel/caller values added to this step-only plan",
                "No CSE, reassociation or general algebraic folding",
            ],
        )


def native_schedule(plan):
    """Return a stable topological schedule, tied by source node order."""
    nodes = {node["id"]: node for node in plan["nodes"]}
    following = defaultdict(list)
    remaining = {}
    for key, node in nodes.items():
        remaining[key] = len(node["predecessors"])
        for before in node["predecessors"]:
            if before not in nodes:
                raise ValueError("Missing lowered dependency")
            following[before].append(key)
    ready = [key for key, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        key = heapq.heappop(ready)
        order.append(key)
        for after in following[key]:
            remaining[after] -= 1
            if remaining[after] == 0:
                heapq.heappush(ready, after)
    if len(order) != len(nodes):
        raise ValueError("Lowered graph has a cycle")
    return order


class Allocation:
    """Allocate words with farthest-next-use eviction and exact SSA tags."""

    def __init__(self, plan, order, budget):
        self.plan, self.order = plan, order
        self.budget = exact_int(budget, "register budget", 1)
        self.values = {value["id"]: value for value in plan["values"]}
        self.nodes = {node["id"]: node for node in plan["nodes"]}
        self.uses = defaultdict(deque)
        for position, node_id in enumerate(order):
            for value in set(self.nodes[node_id]["inputs"]):
                self.uses[value].append(position)
        for value in plan["observable_values"]:
            self.uses[value].append(len(order))
        self.pinned = {
            key
            for key, value in self.values.items()
            if value["kind"] == "base"
        }
        self.registers = {}
        self.free = list(range(budget))
        heapq.heapify(self.free)
        self.spilled = {}
        self.free_slots = []
        self.slots = 0
        self.trace = []
        self.peak = 0
        self.initial = {}

    def event(self, opcode, reads=(), writes=(), memory=None, node=None):
        self.trace.append(
            dict(
                id=len(self.trace),
                opcode=opcode,
                node=node,
                reads=[
                    dict(value=value, register=reg) for value, reg in reads
                ],
                writes=[
                    dict(value=value, register=reg) for value, reg in writes
                ],
                memory=memory,
            )
        )

    def release(self, value):
        if value in self.registers:
            heapq.heappush(self.free, self.registers.pop(value))
        if value in self.spilled:
            heapq.heappush(self.free_slots, self.spilled.pop(value))

    def reserve(self, protected):
        if not self.free:
            choices = [
                value
                for value in self.registers
                if value not in protected and value not in self.pinned
            ]
            if not choices:
                raise ValueError(
                    "Budget cannot hold this instruction's operands"
                )
            victim = max(
                choices,
                key=lambda value: (
                    self.uses[value][0] if self.uses[value] else math.inf,
                    value,
                ),
            )
            register = self.registers[victim]
            if self.uses[victim] and self.values[victim]["kind"] != "constant":
                if victim not in self.spilled:
                    if self.free_slots:
                        slot = heapq.heappop(self.free_slots)
                    else:
                        slot, self.slots = self.slots, self.slots + 1
                    self.spilled[victim] = slot
                    self.event(
                        "STL",
                        [(victim, register)],
                        memory=dict(
                            kind="spill",
                            access="write",
                            space=LOCAL,
                            bytes=4,
                            slot=slot,
                            offset=self.plan["named_local_frame_bytes"]
                            + 4 * slot,
                            expected_semantic=self.values[victim]["semantic"],
                        ),
                    )
            self.registers.pop(victim)
            heapq.heappush(self.free, register)
        return heapq.heappop(self.free)

    def ensure(self, value, protected):
        if value in self.registers:
            return self.registers[value]
        register = self.reserve(protected)
        self.registers[value] = register
        record = self.values[value]
        if record["kind"] == "constant":
            self.event("MOV", writes=[(value, register)])
        elif value in self.spilled:
            slot = self.spilled[value]
            self.event(
                "LDL",
                writes=[(value, register)],
                memory=dict(
                    kind="spill",
                    access="read",
                    space=LOCAL,
                    bytes=4,
                    slot=slot,
                    offset=self.plan["named_local_frame_bytes"] + 4 * slot,
                    expected_semantic=record["semantic"],
                ),
            )
        else:
            raise ValueError(f"Unmaterialized value {value}")
        self.peak = max(self.peak, len(self.registers))
        return register

    def run(self):
        entry = [
            key
            for key, value in self.values.items()
            if value["kind"] in ("live_in", "base")
            and (self.uses[key] or key in self.pinned)
        ]
        if len(entry) > self.budget:
            raise ValueError("Caller entry ABI exceeds the modeled budget")
        for value in sorted(entry):
            register = heapq.heappop(self.free)
            self.registers[value] = register
            self.initial[register] = value
        self.peak = len(entry)
        for position, key in enumerate(self.order):
            node = self.nodes[key]
            protected = set(node["inputs"])
            reads = [
                (value, self.ensure(value, protected))
                for value in node["inputs"]
            ]
            for value in set(node["inputs"]):
                if not self.uses[value] or self.uses[value][0] != position:
                    raise ValueError("Next-use ledger differs from schedule")
                self.uses[value].popleft()
            # Modeled instructions read operands before writing a result;
            # a dead source word may therefore hold its destination.
            for value in set(node["inputs"]):
                if not self.uses[value] and value not in self.pinned:
                    self.release(value)
            writes = []
            for value in node["outputs"]:
                register = self.reserve(set())
                self.registers[value] = register
                writes.append((value, register))
            self.peak = max(self.peak, len(self.registers))
            self.event(node["opcode"], reads, writes, node["memory"], key)
            for value in node["outputs"]:
                if not self.uses[value]:
                    self.release(value)
        final = {}
        for value in self.plan["observable_values"]:
            if value in self.registers:
                final[value] = dict(register=self.registers[value])
            elif value in self.spilled:
                final[value] = dict(spill_slot=self.spilled[value])
            elif self.values[value]["kind"] == "constant":
                final[value] = dict(constant=self.values[value]["constant"])
            else:
                raise ValueError("Observable value has no retained location")
        return dict(
            register_budget=self.budget,
            peak_words=self.peak,
            initial_registers=self.initial,
            final_registers=final,
            spill_slots=self.slots,
            spill_bytes=4 * self.slots,
            local_frame_bytes=self.plan["named_local_frame_bytes"]
            + 4 * self.slots,
            trace=self.trace,
        )


def verify_allocation(plan, allocation):
    """Replay register and memory tags independently of allocation choices."""
    values = {value["id"]: value for value in plan["values"]}
    budget = exact_int(allocation["register_budget"], "register budget", 1)
    slots = exact_int(allocation["spill_slots"], "spill slots")
    frame = exact_int(plan["named_local_frame_bytes"], "named local frame")
    if (
        exact_int(allocation["spill_bytes"], "spill bytes") != 4 * slots
        or exact_int(allocation["local_frame_bytes"], "local frame")
        != frame + 4 * slots
    ):
        raise ValueError("Allocated spill frame identity changed")

    def keyed_integers(mapping, name):
        result = {}
        for key, value in mapping.items():
            if type(key) is str and key.isdecimal() and str(int(key)) == key:
                key = int(key)
            exact_int(key, name)
            if key in result:
                raise ValueError(f"Duplicate {name}")
            result[key] = value
        return result

    def checked_register(register):
        exact_int(register, "register")
        if register >= budget:
            raise ValueError("Trace register is outside the allocated budget")
        used_registers.add(register)
        return register

    def checked_value(value):
        exact_int(value, "value")
        if value not in values:
            raise ValueError("Unknown allocated SSA value")
        return values[value]

    used_registers, used_slots = set(), set()
    initial = keyed_integers(allocation["initial_registers"], "entry register")
    final = keyed_integers(allocation["final_registers"], "exit value")
    required = set(plan["observable_values"])
    if set(final) != required:
        raise ValueError("Observable exit membership changed")
    used_values = required | {
        value for node in plan["nodes"] for value in node["inputs"]
    }
    entry = {
        key
        for key, value in values.items()
        if value["kind"] == "base"
        or value["kind"] == "live_in"
        and key in used_values
    }
    if (
        len(set(initial.values())) != len(initial)
        or set(initial.values()) != entry
    ):
        raise ValueError("Caller entry ABI membership changed")
    registers = {
        checked_register(reg): checked_value(value)["semantic"]
        for reg, value in initial.items()
    }
    named, spill = dict(plan["initial_memory"]), {}
    nodes = {node["id"]: node for node in plan["nodes"]}
    seen = set()
    for position, event in enumerate(allocation["trace"]):
        if exact_int(event["id"], "trace id") != position:
            raise ValueError("Allocated trace identity changed")
        for item in event["reads"] + event["writes"]:
            if set(item) != {"register", "value"}:
                raise ValueError("Invalid register operand schema")
            checked_register(item["register"])
            checked_value(item["value"])
        if len({item["register"] for item in event["writes"]}) != len(
            event["writes"]
        ):
            raise ValueError("Instruction results overlap registers")
        for operand in event["reads"]:
            if (
                registers.get(operand["register"])
                != values[operand["value"]]["semantic"]
            ):
                raise ValueError(
                    "Register replay consumed the wrong SSA value"
                )
        if event["node"] is not None:
            exact_int(event["node"], "lowered node")
            if event["node"] not in nodes:
                raise ValueError("Unknown lowered instruction")
            node = nodes[event["node"]]
            if (
                event["node"] in seen
                or event["opcode"] != node["opcode"]
                or event["memory"] != node["memory"]
            ):
                raise ValueError("Allocated instruction identity changed")
            if not set(node["predecessors"]) <= seen:
                raise ValueError("Allocated schedule broke a dependency")
            if [item["value"] for item in event["reads"]] != node["inputs"]:
                raise ValueError(
                    "Allocated instruction has different operands"
                )
            if [item["value"] for item in event["writes"]] != node["outputs"]:
                raise ValueError("Allocated instruction has different results")
            seen.add(event["node"])
        elif event["opcode"] == "MOV":
            if (
                event["reads"]
                or len(event["writes"]) != 1
                or event["memory"] is not None
                or values[event["writes"][0]["value"]]["kind"] != "constant"
            ):
                raise ValueError("Unproved constant rematerialization")
        else:
            memory = event["memory"]
            read = event["opcode"] == "LDL"
            if (
                event["opcode"] not in ("LDL", "STL")
                or not isinstance(memory, dict)
                or set(memory)
                != {
                    "kind",
                    "access",
                    "space",
                    "bytes",
                    "slot",
                    "offset",
                    "expected_semantic",
                }
                or memory["kind"] != "spill"
                or memory["space"] != LOCAL
                or memory["access"] != ("read" if read else "write")
                or len(event["reads"]) != (0 if read else 1)
                or len(event["writes"]) != (1 if read else 0)
            ):
                raise ValueError("Unproved allocation memory instruction")
        memory = event["memory"]
        if memory is not None:
            if memory["kind"] == "spill":
                slot = exact_int(memory["slot"], "spill slot")
                used_slots.add(slot)
                if (
                    exact_int(memory["offset"], "spill offset")
                    != frame + 4 * slot
                    or exact_int(memory["bytes"], "spill width") != 4
                    or slot >= slots
                ):
                    raise ValueError(
                        "Spill slot overlaps or exceeds its frame"
                    )
            state = spill if memory["kind"] == "spill" else named
            key = (
                memory["slot"]
                if memory["kind"] == "spill"
                else json.dumps(
                    memory["cell"],
                )
            )
            expected = memory["expected_semantic"]
            if memory["access"] == "read":
                if (
                    state.get(key) != expected
                    or len(event["writes"]) != 1
                    or values[event["writes"][0]["value"]]["semantic"]
                    != expected
                ):
                    raise ValueError(
                        "Load replay consumed the wrong cell version"
                    )
            elif memory["access"] == "write":
                if values[event["reads"][-1]["value"]]["semantic"] != expected:
                    raise ValueError("Store replay wrote the wrong value")
                state[key] = expected
            else:
                raise ValueError("Unknown memory access")
        for result in event["writes"]:
            registers[result["register"]] = values[result["value"]]["semantic"]
    if seen != set(nodes):
        raise ValueError("Allocation omitted a lowered instruction")
    for value, location in final.items():
        record = checked_value(value)
        if set(location) == {"register"}:
            observed = registers.get(checked_register(location["register"]))
        elif set(location) == {"spill_slot"}:
            slot = exact_int(location["spill_slot"], "exit spill slot")
            if slot >= slots:
                raise ValueError("Exit spill slot exceeds its frame")
            observed = spill.get(slot)
        elif set(location) == {"constant"} and record["kind"] == "constant":
            if location["constant"] != record["constant"]:
                raise ValueError("Observable constant identity changed")
            observed = record["semantic"]
        else:
            raise ValueError("Unproved observable exit location")
        if observed != record["semantic"]:
            raise ValueError("Observable register value was lost")
    if (
        used_slots != set(range(slots))
        or exact_int(allocation["peak_words"], "peak words", 1)
        != max(used_registers, default=-1) + 1
    ):
        raise ValueError("Allocated register or spill extent changed")
    for cell, expected in plan["final_memory"].items():
        if named.get(cell) != expected:
            raise ValueError("Observable memory value was lost")
    return dict(
        status="SSA_AND_MEMORY_CONSERVATION_PASS",
        instructions=len(seen),
        trace_events=len(allocation["trace"]),
        final_cells=len(plan["final_memory"]),
        final_values=len(allocation["final_registers"]),
    )


def hardware_model(manifest):
    """Extract target capacities from an existing query, without CUDA."""
    attrs = manifest.get(
        "device_attributes",
        manifest.get("compilation_identity", {}).get("device_attributes", {}),
    )
    capability = manifest.get(
        "compute_capability",
        manifest.get("compilation_identity", {}).get("compute_capability"),
    )
    if list(capability) != [8, 9]:
        raise ValueError("The hardware equations are restricted to SM89")
    names = (
        "MULTIPROCESSOR_COUNT",
        "WARP_SIZE",
        "MAX_THREADS_PER_MULTIPROCESSOR",
        "MAX_THREADS_PER_BLOCK",
        "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
        "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
        "RESERVED_SHARED_MEMORY_PER_BLOCK",
        "MAX_REGISTERS_PER_BLOCK",
    )
    result = {name: exact_int(attrs[name], name, 1) for name in names}
    if result["WARP_SIZE"] != 32:
        raise ValueError("Expected warp width 32")
    result.update(
        registers_per_sm=65536,
        subpartitions=4,
        register_unit_words=256,
        shared_unit_bytes=128,
        maximum_registers_per_thread=255,
        max_blocks_per_sm=24,
        unified_data_bytes=131072,
        supported_carveouts=[0, 8192, 16384, 32768, 65536, 102400],
    )
    return result


def residency(hardware, registers, block, shared_dynamic):
    """Apply conditional SM89 allocation quanta and launch capacity checks."""
    exact_int(registers, "registers", 1)
    exact_int(block, "block", 1)
    exact_int(shared_dynamic, "shared_dynamic")
    if block % 32 or block > hardware["MAX_THREADS_PER_BLOCK"]:
        raise ValueError("This model requires complete 32-thread warps")
    warps = block // 32
    rwarp = round_up(32 * registers, 256)
    register_limit = (4 * (16384 // rwarp)) // warps
    hardware_per_block = rwarp * round_up(warps, 4)
    if (
        registers > 255
        or hardware_per_block > hardware["MAX_REGISTERS_PER_BLOCK"]
    ):
        register_limit = 0
    nonshared = min(
        register_limit, hardware["MAX_THREADS_PER_MULTIPROCESSOR"] // block, 24
    )
    allocated = round_up(
        shared_dynamic + hardware["RESERVED_SHARED_MEMORY_PER_BLOCK"], 128
    )
    if shared_dynamic > hardware["MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"]:
        nonshared = 0
    candidates = [
        size
        for size in hardware["supported_carveouts"]
        if size <= hardware["MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"]
        and size >= allocated
    ]
    if not candidates or not nonshared:
        return dict(feasible=False, reason="conditional launch resource limit")
    preserving = [
        size for size in candidates if size // allocated >= nonshared
    ]
    carveout = min(preserving) if preserving else max(candidates)
    blocks = min(nonshared, carveout // allocated)
    return dict(
        feasible=True,
        registers=registers,
        allocated_registers_per_warp=rwarp,
        allocated_registers_per_block=rwarp * warps,
        shared_dynamic_bytes=shared_dynamic,
        shared_allocated_bytes=allocated,
        modeled_carveout_bytes=carveout,
        carveout_assumption="smallest preserving nonshared residency",
        legal_carveout_bytes=candidates,
        modeled_l1_bytes=hardware["unified_data_bytes"] - carveout,
        resident_blocks=blocks,
        resident_warps=blocks * warps,
        resident_threads=blocks * block,
    )


def sector_stream(
    trace,
    local_frame,
    shared_stride,
    geometry,
    block,
    waves=2,
    backing="resident_slots",
):
    """Emit sector ranges under two explicit local backing hypotheses."""
    if backing not in ("resident_slots", "trajectory_unique"):
        raise ValueError("Unknown local backing hypothesis")
    exact_int(waves, "waves", 2)
    events = []
    resident = geometry["resident_warps"]
    frame = round_up(local_frame, 4)
    for event in trace:
        memory = event["memory"]
        if memory is None:
            continue
        if memory["space"] == LOCAL:
            # CUDA's same-offset local words coalesce across a warp. The
            # model makes warps' frame segments disjoint; base reuse across
            # waves is a separate scenario, never inferred from gtid.
            sector = (memory["offset"] // 4) * 4
            descriptors = dict(
                first_sector_in_warp=sector,
                sectors=4,
                warp_segment_sectors=frame * 32 // 32,
            )
        else:
            addresses = [
                lane * shared_stride + memory["offset"] for lane in range(32)
            ]
            descriptors = dict(
                lane_offsets=addresses,
                banks=[(address // 4) % 32 for address in addresses],
                distinct_banks=len(
                    {(address // 4) % 32 for address in addresses}
                ),
                bank_geometry_assumption="32 banks, four-byte words",
            )
        events.append(
            dict(
                trace_event=event["id"],
                access=memory["access"],
                kind=memory["kind"],
                space=memory["space"],
                offset=memory["offset"],
                **descriptors,
            )
        )
    cache = OrderedDict()
    capacity = geometry["modeled_l1_bytes"] // 32
    counts = Counter()
    for wave in range(waves):
        # One step per warp. Cyclic instruction-ordinal interleaving is a
        # declared access-stream hypothesis, not a measured timing trace.
        for event in events:
            for warp in range(resident):
                counts[
                    f"{event['space']}_{event['access']}_warp_instructions"
                ] += 1
                if event["space"] != LOCAL:
                    counts[f"shared_{event['access']}_bank_wavefronts"] += max(
                        Counter(event["banks"]).values(),
                    )
                    continue
                identity = warp + (
                    wave * resident if backing == "trajectory_unique" else 0
                )
                base = identity * event["warp_segment_sectors"]
                for offset in range(event["sectors"]):
                    key = base + event["first_sector_in_warp"] + offset
                    counts[f"local_{event['access']}_sectors"] += 1
                    if key in cache:
                        dirty = cache.pop(key)
                        counts[f"local_{event['access']}_hits"] += 1
                    else:
                        dirty = False
                        counts[f"local_{event['access']}_misses"] += 1
                        if len(cache) == capacity:
                            _, evicted_dirty = cache.popitem(last=False)
                            counts["dirty_evictions"] += int(evicted_dirty)
                    cache[key] = dirty or event["access"] == "write"
    return dict(
        backing=backing,
        mapping=(
            "warp-local word-interleaved disjoint frames; resident slot "
            "reused on later waves"
            if backing == "resident_slots"
            else "warp-local frames disjoint for each trajectory wave"
        ),
        events=events,
        counts=dict(counts),
        waves=waves,
        scope="one modeled SM, one step per warp in each wave",
        cache_assumption="sector LRU, fully associative, write allocate, cold",
        cycle_interleaving="cyclic warp visits at each memory-event ordinal",
        retained_dirty_sectors=sum(cache.values()),
        block_size=block,
        limitations=[
            "No L2 prediction or measured hit-class assertion",
            "No integration-step history or physical address claim",
            "Load/store caches may implement different write policy",
        ],
    )


def service_estimate(trace, geometry, catalog=None):
    """Simulate a fixed stream when every opcode has sourced service data."""
    counts = Counter(event["opcode"] for event in trace)
    warps = geometry["resident_warps"]
    missing = sorted(
        set(counts) - set((catalog or {}).get("instructions", {}))
    )
    result = dict(
        opcode_counts_per_warp=dict(counts),
        instruction_bytes_model=16 * len(trace),
        issue_component_aggregate_sm_cycles=len(trace) * warps / 4,
        issue_component_status="conditional work capacity, not runtime",
        missing_service_symbols=[f"service[{op}]" for op in missing],
        excluded_terms=[
            "caller_address_setup",
            "outer_integration_control",
            "instruction_fetch",
            "cache_service_misses",
        ],
        cycles=None,
    )
    if missing:
        return result
    specifications = catalog["instructions"]
    for opcode in counts:
        record = specifications[opcode]
        if (
            not record.get("provenance")
            or not record.get("assumption")
            or record.get("scope") not in ("sm", "subpartition")
        ):
            raise ValueError(
                "Each service requires provenance/assumption/scope"
            )
        for field in ("latency_cycles", "initiation_cycles"):
            if (
                type(record[field]) not in (int, float)
                or not math.isfinite(record[field])
                or record[field] <= 0
            ):
                raise ValueError("Service values must be finite and positive")
    pcs = [0] * warps
    ready = [defaultdict(float) for _ in range(warps)]
    memory_ready = [defaultdict(float) for _ in range(warps)]
    pipelines = defaultdict(float)
    scheduler = [0.0] * 4
    finish = 0.0
    while any(pc < len(trace) for pc in pcs):
        choices = []
        for warp, pc in enumerate(pcs):
            if pc == len(trace):
                continue
            event = trace[pc]
            record = specifications[event["opcode"]]
            partition = warp % 4
            pipe = (
                partition if record["scope"] == "subpartition" else -1,
                record["pipeline"],
            )
            hazards = event["reads"] + event["writes"]
            memory_key = (
                None
                if event["memory"] is None
                else (event["memory"]["space"], event["memory"]["offset"])
            )
            start = max(
                scheduler[partition],
                pipelines[pipe],
                memory_ready[warp][memory_key],
                *(ready[warp][r["register"]] for r in hazards),
            )
            choices.append((start, warp, pipe))
        start, warp, pipe = min(choices)
        event = trace[pcs[warp]]
        record = specifications[event["opcode"]]
        stop = start + record["latency_cycles"]
        for output in event["writes"]:
            ready[warp][output["register"]] = stop
        if event["memory"]:
            key = (event["memory"]["space"], event["memory"]["offset"])
            memory_ready[warp][key] = stop
        scheduler[warp % 4] = start + 1
        pipelines[pipe] = start + record["initiation_cycles"]
        pcs[warp] += 1
        finish = max(finish, stop)
    result.update(
        cycles=finish,
        catalog=catalog,
        cycle_scope="one modeled SM resident wave, one step",
        scheduling="cyclic assignment, earliest ready issue, warp-ID tie",
        qualification="conditional pipeline service; exclusions remain",
    )
    return result


def predict(
    graph,
    hardware,
    mode="promote",
    contract=False,
    block=64,
    register_budget=None,
    catalog=None,
):
    """Return a checkable lowering, allocation and traffic plan."""
    sources = validate_graph(graph)
    construction = validate_construction(graph)
    fastmath = construction["jit_kwargs"].get("fastmath", [])
    if contract and "contract" not in fastmath:
        raise ValueError(
            "Contraction scenario conflicts with actual JIT flags"
        )
    if any(node["kind"] == "Div" for node in graph["nodes"]):
        if "arcp" not in fastmath:
            raise ValueError(
                "Reciprocal lowering requires the actual arcp flag"
            )
    lowering = Lowering(graph, mode, contract).build()
    order = native_schedule(lowering)
    # An unbounded pass measures this native plan's no-spill requirement;
    # it is not a source peak and not a bound on the installed compiler.
    limitless = Allocation(
        lowering, order, max(1, len(lowering["values"]))
    ).run()
    verify_allocation(lowering, limitless)
    budget = (
        min(255, max(1, limitless["peak_words"]))
        if register_budget is None
        else exact_int(register_budget, "register_budget", 1)
    )
    if budget > 255:
        raise ValueError("Modeled native budget exceeds the hardware maximum")
    allocation = Allocation(lowering, order, budget).run()
    conservation = verify_allocation(lowering, allocation)
    dynamic = max(4, construction["shared_stride_bytes"] * block)
    geometry = residency(
        hardware, max(1, allocation["peak_words"]), block, dynamic
    )
    streams, estimate = [], None
    if geometry["feasible"]:
        streams = [
            sector_stream(
                allocation["trace"],
                allocation["local_frame_bytes"],
                construction["shared_stride_bytes"],
                geometry,
                block,
                backing=backing,
            )
            for backing in ("resident_slots", "trajectory_unique")
        ]
        estimate = service_estimate(allocation["trace"], geometry, catalog)
    return dict(
        schema=1,
        kind="conditional_erk_native_plan",
        candidate=dict(
            materialization=mode,
            contraction=contract,
            actual_placement=construction["placement"],
            block_size=block,
        ),
        actionable_settings=dict(
            stage_accumulator_location=construction["placement"],
            block_size=block,
            unroll="captured full expansion",
        ),
        materialization_is_compiler_hypothesis=True,
        source_operation_counts=dict(
            Counter(node["kind"] for node in graph["nodes"])
        ),
        provenance=dict(
            model_source_sha256=digest(SCRIPT),
            sources=sources,
            construction=construction,
        ),
        hardware=hardware,
        lowering=lowering,
        native_schedule=order,
        modeled_no_spill_words=limitless["peak_words"],
        allocation=allocation,
        conservation=conservation,
        geometry=geometry,
        streams=streams,
        service=estimate,
        status="conditional_complete_step_model"
        if geometry["feasible"]
        else "conditional_geometry_infeasible",
        claim="Conditional step model, not a native bound or kernel time",
    )


def rank_plans(plans):
    """Rank available modeled service, retaining missing/excluded terms."""
    feasible = [
        (index, plan)
        for index, plan in enumerate(plans)
        if plan["geometry"]["feasible"]
    ]
    modeled = [
        (index, plan)
        for index, plan in feasible
        if plan["service"]["cycles"] is not None
    ]
    identities = set()
    catalogs = set()
    hardware_identities = set()
    for _, plan in feasible:
        record = plan["provenance"]["construction"]
        identities.add(
            json.dumps(
                {
                    key: record[key]
                    for key in (
                        "system",
                        "algo",
                        "actual_system_size",
                        "package_source_hash",
                        "jit_kwargs",
                        "toolchain_fingerprint",
                    )
                },
                sort_keys=True,
            )
        )
        hardware_identities.add(json.dumps(plan["hardware"], sort_keys=True))
    for _, plan in modeled:
        catalogs.add(json.dumps(plan["service"]["catalog"], sort_keys=True))
    if (
        len(identities) > 1
        or len(catalogs) > 1
        or len(hardware_identities) > 1
    ):
        raise ValueError(
            "Ranking requires one workload/compiler/service identity"
        )
    # Normalize each one-resident-wave estimate to the same total warp work.
    groups = defaultdict(list)
    for index, plan in modeled:
        scenario = json.dumps(
            {
                key: plan["candidate"][key]
                for key in ("materialization", "contraction")
            },
            sort_keys=True,
        )
        groups[scenario].append(
            dict(
                candidate=index,
                actionable_settings=plan["actionable_settings"],
                cycles_per_warp=plan["service"]["cycles"]
                / plan["geometry"]["resident_warps"],
            )
        )
    rankings = []
    for scenario, rows in groups.items():
        rows.sort(key=lambda row: (row["cycles_per_warp"], row["candidate"]))
        rankings.append(
            dict(
                scenario=json.loads(scenario),
                ranking=rows,
                conditional_recommendation=rows[0]["candidate"],
            )
        )
    return dict(
        service_rankings_by_lowering_scenario=rankings,
        recommendation_conditions="same catalog; exclusions cancel; full grid",
        missing_service_candidates=[
            index
            for index, plan in feasible
            if plan["service"]["cycles"] is None
        ],
        issue_component_ranking=sorted(
            (
                dict(
                    candidate=index,
                    instructions_per_warp=len(plan["allocation"]["trace"]),
                )
                for index, plan in feasible
            ),
            key=lambda item: (
                item["instructions_per_warp"],
                item["candidate"],
            ),
        ),
        issue_ranking_is_runtime_recommendation=False,
        compiler_alternatives_are_not_user_settings=True,
    )


CONSTRUCTOR = r"""
import hashlib
import json
from pathlib import Path
import sys
from cubie import Solver
from cubie._utils import package_source_hash
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cubie_cache import toolchain_fingerprint
from benchmarks import placement_landscape as placement
from benchmarks.hardware_model.source_value_graph import describe_source_values

def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(item) for key,item in value.items()}
    if isinstance(value, (set,frozenset)):
        return sorted(canonical(item) for item in value)
    if isinstance(value, (list,tuple)):
        return [canonical(item) for item in value]
    return value

output = Path(sys.argv[1]).resolve()
request = json.loads((output/'construction_request.json').read_text())
cache = output/'codegen'
cache.mkdir(exist_ok=False)
previous = get_cache_root_override()
solver = None
try:
    set_cache_root(cache)
    name = request['system']
    if name == 'lorenz':
        system = placement.SYSTEMS[name]['build']()
        template = name
    elif name.startswith('chain') and name[5:].isdigit():
        n = int(name[5:])
        if n < 3:
            raise ValueError('Chain requires at least three states')
        system = placement.build_chain(n, 3)
        template = 'chain32'
    else:
        raise ValueError('Need lorenz or chainN actual factory')
    kwargs = placement.solver_kwargs(template, request['algo'])
    kwargs['stage_accumulator_location'] = request['placement']
    solver = Solver(system, **kwargs)
    graph = describe_source_values(solver, max_states=1)
    stride = 4*(int(solver.kernel.shared_memory_elements) +
                int(solver.kernel.shared_memory_needs_padding))
    graph['native_plan_construction'] = dict(
        system=name, algo=request['algo'], placement=request['placement'],
        actual_system_size=int(graph['workload']['n_states']),
        shared_stride_bytes=stride,
        shared_elements_per_run=int(solver.kernel.shared_memory_elements),
        shared_padding_elements=int(solver.kernel.shared_memory_needs_padding),
        shared_stride_source=dict(
            path=str(Path(sys.modules['cubie.batchsolving.BatchSolverKernel'].__file__).resolve()),
            sha256=hashlib.sha256(Path(sys.modules['cubie.batchsolving.BatchSolverKernel'].__file__).read_bytes()).hexdigest(),
            lines=[975,982]),
        jit_kwargs=canonical(solver.kernel.jit_kwargs),
        toolchain_fingerprint=toolchain_fingerprint(),
        package_source_hash=package_source_hash(),
        cache_root=str(cache), constructor_sha256=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        request_sha256=hashlib.sha256(
            (output/'construction_request.json').read_bytes()).hexdigest(),
        native_overloads=0,
    )
    path=output/'graph.json'
    with path.open('x',encoding='utf-8') as handle:
        json.dump(graph,handle,separators=(',',':'),allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status='SOURCE_CONSTRUCTION_PASS',
                         nodes=len(graph['nodes']),
                         shared_stride_bytes=stride,
                         native_overloads=0)))
finally:
    if solver is not None:
        solver.close()
    set_cache_root(previous)
"""


def main():
    """Construct source or estimate retained graphs in a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    constructor = sub.add_parser("construct")
    constructor.add_argument("--system", required=True)
    constructor.add_argument("--algo", required=True)
    constructor.add_argument(
        "--placement", choices=("local", "shared"), default="local"
    )
    constructor.add_argument("--output", type=Path, required=True)
    estimator = sub.add_parser("estimate")
    estimator.add_argument("--graph", type=Path, required=True)
    estimator.add_argument("--hardware-manifest", type=Path, required=True)
    estimator.add_argument("--catalog", type=Path)
    estimator.add_argument(
        "--mode", choices=("promote", "addressable"), default="promote"
    )
    estimator.add_argument("--contract", action="store_true")
    estimator.add_argument("--block", type=int, default=64)
    estimator.add_argument("--register-budget", type=int)
    estimator.add_argument("--output", type=Path, required=True)
    ranking = sub.add_parser("rank")
    ranking.add_argument("--plans", type=Path, nargs="+", required=True)
    ranking.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.command == "rank":
        plans = [json.loads(path.read_text()) for path in args.plans]
        if (
            len({plan["provenance"]["model_source_sha256"] for plan in plans})
            != 1
        ):
            raise ValueError("Ranking mixes estimator implementations")
        result = rank_plans(plans)
        result["inputs"] = [
            dict(path=str(path.resolve()), sha256=digest(path))
            for path in args.plans
        ]
        write_json(args.output / "ranking.json", result)
        print(json.dumps(result))
        return
    if args.command == "construct":
        (args.output / "benchmark_source.py").write_bytes(SCRIPT.read_bytes())
        write_json(
            args.output / "construction_request.json",
            dict(
                system=args.system,
                algo=args.algo,
                placement=args.placement,
                model_sha256=digest(SCRIPT),
            ),
        )
        worker = args.output / "construct.py"
        worker.write_text(CONSTRUCTOR, encoding="utf-8")
        command = [
            sys.executable,
            str(worker.resolve()),
            str(args.output.resolve()),
        ]
        result = subprocess.run(
            command, cwd=REPO, capture_output=True, text=True, check=False
        )
        write_json(
            args.output / "worker.json",
            dict(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                worker_sha256=digest(worker),
            ),
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        print(result.stdout)
        return
    graph = json.loads(args.graph.read_text())
    hardware = hardware_model(json.loads(args.hardware_manifest.read_text()))
    catalog = (
        None if args.catalog is None else json.loads(args.catalog.read_text())
    )
    result = predict(
        graph,
        hardware,
        args.mode,
        args.contract,
        args.block,
        args.register_budget,
        catalog,
    )
    result["provenance"]["input_graph"] = dict(
        path=str(args.graph.resolve()),
        sha256=digest(args.graph),
    )
    result["provenance"]["hardware_manifest"] = dict(
        path=str(args.hardware_manifest.resolve()),
        sha256=digest(args.hardware_manifest),
    )
    write_json(args.output / "plan.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                no_spill_words=result["modeled_no_spill_words"],
                allocated_peak=result["allocation"]["peak_words"],
                spill_bytes=result["allocation"]["spill_bytes"],
                geometry=result["geometry"],
            )
        )
    )


if __name__ == "__main__":
    main()
