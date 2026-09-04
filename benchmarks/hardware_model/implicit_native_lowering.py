"""Lower typed implicit execution graphs with separate predicate storage.

This CPU-only component describes an explicit compiler alternative. It
accepts retained source graphs, never compiled labels or measured counts.
"""

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path

from benchmarks.hardware_model import native_plan as base


SCRIPT = Path(__file__).resolve()
BASE_SHA = "f547ee91e5f3a390d68c8113e8eb438bde03438935ca8d4b294e148fb9480471"
DTYPES = {"float32", "int32", "uint32", "bool"}


class Unresolved(ValueError):
    """A required compiler or simultaneous-operand fact is unproved."""


def bank(dtype):
    """Return the declared native register class, separately from type."""
    if dtype not in DTYPES:
        raise Unresolved(f"Unproved native type {dtype}")
    return "P" if dtype == "bool" else "R"


def validate_hot_template(source):
    """Recompute one source-template identity and its fixed indices."""
    template = source.get("hot_template")
    identity = source.get("hot_template_identity")
    fixed = [
        item
        for item in source.get("loop_indices", [])
        if item.get("recurrent") is False
    ]
    if (
        not isinstance(template, dict)
        or identity
        != hashlib.sha256(
            json.dumps(template, sort_keys=True).encode()
        ).hexdigest()
        or source.get("template_is_not_native_copy_identity") is not True
        or template.get("fixed_indices") != fixed
        or any(item.get("recurrent") is not False for item in fixed)
    ):
        raise ValueError("Source hot-template identity differs")


def runtime_key(source):
    """Return the declared role/call key for one runtime region."""
    region = source.get("runtime_region")
    if region is None:
        return None
    if (
        set(region)
        != {"role", "call", "body_index", "entry_mask", "phase"}
        or not isinstance(region["role"], str)
        or not isinstance(region["call"], str)
        or type(region["entry_mask"]) is not int
        or not 0 < region["entry_mask"] < 2**32
        or region["phase"] not in ("entry", "body", "exit_vote", "exit")
        or (
            region["body_index"] is not None
            and (
                type(region["body_index"]) is not int
                or region["body_index"] < 0
            )
        )
    ):
        raise ValueError("Runtime-region record is incomplete")
    return region["role"], region["call"]


def validate_regime(graph, nodes):
    """Reconstruct the selected iteration regime from graph execution."""
    regime = graph.get("regime", {})
    expected_regime_keys = {
        "status",
        "step_entry_mask",
        "calls",
        "warp_body_totals",
        "logged_lane_counters",
        "limitations",
    }
    step_mask = regime.get("step_entry_mask")
    if (
        set(regime) != expected_regime_keys
        or regime.get("status") != "EXPLICIT_SYMBOLIC_REGIME_EVALUATED"
        or type(step_mask) is not int
        or not 0 < step_mask < 2**32
        or regime.get("limitations")
        != [
            "Masks are supplied assumptions, not predicted convergence",
            "Different step-entry masks need separate warp instances",
            "Body totals are not dynamic native instruction counts",
        ]
    ):
        raise ValueError("Selected iteration regime header differs")

    workload = graph.get("workload", {})
    roles = workload.get("roles", {})
    if (
        workload.get("kind") != "actual_implicit_workload"
        or not isinstance(roles, dict)
        or not roles
        or any(item.get("role") != name for name, item in roles.items())
    ):
        raise ValueError("Runtime roles are not actual workload roles")

    region_nodes = defaultdict(list)
    entry_masks = defaultdict(set)
    body_indices = defaultdict(set)
    for node in nodes:
        key = runtime_key(node.get("source", {}))
        if key is None:
            continue
        region = node["source"]["runtime_region"]
        if region["role"] not in roles:
            raise ValueError("Runtime region has no workload role")
        if region["entry_mask"] & ~step_mask:
            raise ValueError("Runtime entry mask exceeds the step mask")
        region_nodes[key].append(node)
        entry_masks[key].add(region["entry_mask"])
        if region["phase"] == "body":
            body_indices[key].add(region["body_index"])
    if any(len(masks) != 1 for masks in entry_masks.values()):
        raise ValueError("Runtime call changes its entry mask")

    traces = {}
    for item in graph["controls"]:
        validate_hot_template(item["source"])
        if item["kind"] != "recurrent_execution_trace":
            continue
        key = runtime_key(item["source"])
        region = item["source"]["runtime_region"]
        role = roles.get(key[0], {})
        iteration = role.get("iteration")
        indices = item.get("indices")
        if (
            key in traces
            or key not in region_nodes
            or entry_masks[key] != {region["entry_mask"]}
            or region["body_index"] is not None
            or region["phase"] != "entry"
            or item.get("code_copies_are_execution_count") is not False
            or not isinstance(indices, list)
            or any(type(index) is not int for index in indices)
            or indices != list(range(len(indices)))
            or not isinstance(iteration, dict)
            or item.get("directive")
            != iteration["source_region"]["actual_closure_flag"]
        ):
            raise ValueError("Recurrent execution trace differs")
        traces[key] = item

    def call_shape(key):
        if key not in region_nodes or len(entry_masks[key]) != 1:
            raise ValueError("Regime call has no exact runtime region")
        role = roles[key[0]]
        solver = role.get("solver_type")
        if solver not in ("newton", "lu", "mr", "bicgstab"):
            raise ValueError("Regime call uses an unsupported solver role")
        bodies = body_indices[key]
        if None in bodies or bodies != set(range(len(bodies))):
            raise ValueError("Runtime body indices are not contiguous")
        body_count = len(bodies)
        mask = next(iter(entry_masks[key]))
        trace = traces.get(key)
        if solver == "lu":
            if trace is not None or body_count:
                raise ValueError("Direct solve has recurrent execution")
            return dict(
                body_iterations=0,
                direct_calls=1,
                returned_lane_counts=[
                    int(bool(mask & (1 << lane))) for lane in range(32)
                ],
            )
        if trace is None:
            raise ValueError("Iterative call lacks its recurrent trace")
        loop_nodes = []
        for node in region_nodes[key]:
            region = node["source"]["runtime_region"]
            if (
                node["kind"] == "BranchDecision"
                and node.get("decision_reason")
                == "declared uniform convergence-mask regime"
                and region["phase"] in ("body", "exit_vote")
            ):
                loop_nodes.append(node)
        loop_nodes.sort(
            key=lambda node: node["source"]["runtime_region"]["body_index"]
        )
        if len(loop_nodes) != len(trace["indices"]):
            raise ValueError("Loop-top vote count differs from trace")
        for index, node in enumerate(loop_nodes):
            region = node["source"]["runtime_region"]
            predicate = graph["values"][node["inputs"][0]]
            producer = predicate["producer"]
            vote = nodes[producer] if producer is not None else {}
            if (
                region["body_index"] != index
                or vote.get("kind") not in ("AllSync", "AnySync")
                or vote.get("outputs") != node["inputs"]
                or vote.get("declared_result") is not node["selected_path"]
                or (index < body_count and node["selected_path"] is not False)
                or (
                    index >= body_count
                    and node["selected_path"] is not True
                )
            ):
                raise ValueError("Loop-top vote/branch execution differs")
        if len(loop_nodes) not in (body_count, body_count + 1):
            raise ValueError("Loop-top votes do not delimit body execution")
        selected = role["iteration"]["control"][
            "selected_body_call_counts"
        ]
        common = dict(
            body_iterations=body_count,
            direct_calls=0,
            returned_lane_counts=[
                body_count if mask & (1 << lane) else 0
                for lane in range(32)
            ],
        )
        if solver == "newton":
            if selected != {
                "residual_function": 1,
                "linear_solver_fn": 1,
                "correction_norm_fn": 1,
            }:
                raise ValueError("Newton source-call multiplicity differs")
            return dict(
                body_iterations=body_count,
                loop_top_votes=len(loop_nodes),
                residual_calls=body_count,
                correction_norm_calls=body_count,
                body_control_entry_lanes=mask.bit_count(),
            )
        expected_selected = {
            "operator_apply": 1 if solver == "mr" else 2,
            "preconditioner": 1 if solver == "mr" else 2,
            "weighted_norm": 1 if solver == "mr" else 2,
        }
        if selected != expected_selected:
            raise ValueError("Linear source-call multiplicity differs")
        zero_guess = role.get("zero_initial_guess")
        if type(zero_guess) is not bool:
            raise ValueError("Linear initial-guess regime is not explicit")
        initial_operator = int(not zero_guess)
        initial_norm = 1 if zero_guess else 2
        return dict(
            **common,
            operator_calls=(
                body_count * expected_selected["operator_apply"]
                + initial_operator
            ),
            preconditioner_calls=(
                body_count * expected_selected["preconditioner"]
            ),
            norm_calls=(
                body_count * expected_selected["weighted_norm"]
                + initial_norm
            ),
            initial_votes=int(solver == "bicgstab" and zero_guess),
            loop_top_votes=len(loop_nodes),
            seed_region_executed=True,
            body_control_entry_lanes=mask.bit_count(),
            returned_counter_lanes=mask.bit_count(),
            counts_are_not_active_thread_instruction_counts=True,
        )

    newton_keys = sorted(key for key in region_nodes if key[0] == "main_newton")
    expected_calls = {}
    consumed = set()
    for key in newton_keys:
        record = call_shape(key)
        linear = sorted(
            candidate
            for candidate in region_nodes
            if candidate[0] == "main_linear"
            and candidate[1].startswith(f"{key[1]}.linear")
        )
        if [candidate[1] for candidate in linear] != [
            f"{key[1]}.linear{index}" for index in range(len(linear))
        ]:
            raise ValueError("Newton linear-call indices differ")
        record["linear_calls"] = [call_shape(candidate) for candidate in linear]
        expected_calls[key[1]] = record
        consumed.add(key)
        consumed.update(linear)
    for key in sorted(set(region_nodes) - consumed):
        if key[0] not in ("main_linear", "error_linear"):
            raise ValueError("Runtime call role is not modeled")
        if ".linear" in key[1]:
            raise ValueError("Nested linear call lacks its Newton parent")
        expected_calls[key[1]] = call_shape(key)
        consumed.add(key)
    if consumed != set(region_nodes) or regime["calls"] != expected_calls:
        raise ValueError("Selected call regime differs from execution graph")

    totals = dict(newton_bodies=0, krylov_bodies=0, direct_calls=0)
    logged = [[0] * 32, [0] * 32]
    for key in newton_keys:
        shape = call_shape(key)
        totals["newton_bodies"] += shape["body_iterations"]
        mask = next(iter(entry_masks[key]))
        for lane in range(32):
            if mask & (1 << lane):
                logged[0][lane] += shape["body_iterations"]
    for key in region_nodes:
        if key[0] == "main_newton":
            continue
        shape = call_shape(key)
        mask = next(iter(entry_masks[key]))
        totals["krylov_bodies"] += shape["body_iterations"]
        totals["direct_calls"] += shape["direct_calls"]
        count = shape["body_iterations"] + shape["direct_calls"]
        for lane in range(32):
            if mask & (1 << lane):
                logged[1][lane] += count
    if (
        regime["warp_body_totals"] != totals
        or regime["logged_lane_counters"] != logged
    ):
        raise ValueError("Iteration totals/counters differ from runtime regions")
    return regime


class TypedLowering(base.Lowering):
    """Retain typed native operands, exact cell aliases and order edges."""

    def __init__(self, graph, compiler, materialization="promote"):
        if base.digest(base.SCRIPT) != BASE_SHA:
            raise Unresolved("The reused scalar/cell implementation changed")
        self.compiler = compiler
        super().__init__(graph, materialization, compiler["fp32_contract"])

    def value(self, dtype, kind, semantic, **details):
        identifier = super().value(dtype, kind, semantic, **details)
        self.values[identifier].update(
            register_class=bank(dtype),
            gpr_words=int(dtype != "bool"),
            predicate_bits=int(dtype == "bool"),
            words=int(dtype != "bool"),
        )
        return identifier

    def emit(
        self, opcode, inputs, output, source_ids, memory=None, semantics=None
    ):
        ordering = {
            parent
            for key in source_ids
            for before in self.original_nodes[key]["order_predecessors"]
            for parent in self.source_nodes.get(before, [])
        }
        identifier = super().emit(
            opcode,
            inputs,
            output,
            source_ids,
            memory,
        )
        item = self.nodes[identifier]
        data = {
            self.values[value]["producer"]
            for value in inputs
            if "producer" in self.values[value]
        }
        item["value_predecessors"] = sorted(data)
        item["order_predecessors"] = sorted(ordering)
        item["semantics"] = semantics or {}
        if opcode in {
            "FADD",
            "FMUL",
            "FFMA",
            "FSETP",
            "SEL",
            "MUFU.RCP",
            "MUFU.SQRT",
            "LOP3",
        } and any(self.values[v]["dtype"] == "float32" for v in inputs):
            item["semantics"]["fp32_flush_subnormals"] = self.compiler[
                "fp32_flush_subnormals"
            ]
        item["source_contexts"] = [
            self.original_nodes[key].get("source", {}) for key in source_ids
        ]
        item["native_encoding_verified"] = False
        return identifier

    def memory(self, node):
        storage, low, high, dtype = node["cell"]
        if dtype not in ("float32", "int32", "uint32"):
            raise Unresolved("Addressable bool cells need a byte-storage rule")
        if high - low != 4 or low % 4 or low < 0:
            raise Unresolved("Memory needs an exact aligned 32-bit cell")
        if (
            storage not in self.layouts
            or high > self.layouts[storage]["bytes"]
        ):
            raise Unresolved("Memory exceeds the captured allocation")
        if storage in self.promoted:
            self.source_nodes[node["id"]] = sorted(
                {
                    parent
                    for before in node["order_predecessors"]
                    for parent in self.source_nodes.get(before, [])
                }
            )
            self.rewrites.append(
                dict(
                    rule="promote_exact_typed_cell",
                    source_nodes=[node["id"]],
                    cell=node["cell"],
                )
            )
            return
        space = self.spaces[storage]
        inputs = [self.bases[space]]
        semantic = self.semantic(node["inputs"][0])
        details = dict(
            kind="named",
            cell=node["cell"],
            space=space,
            bytes=4,
            dtype=dtype,
            offset=self.layouts[storage]["frame_offset"] + low,
            expected_semantic=semantic,
        )
        if node["kind"] == "element_read_alias":
            if self.source[node["inputs"][0]]["kind"] == "live_in":
                self.initial_memory[json.dumps(node["cell"])] = semantic
            output = self.value(dtype, "load", semantic)
            self.read_values[node["id"]] = output
            opcode = "LDL" if space == base.LOCAL else "LDS"
            details["access"] = "read"
        elif node["kind"] == "element_write_alias":
            inputs.append(self.mapped(node["inputs"][0], node))
            output = None
            opcode = "STL" if space == base.LOCAL else "STS"
            details["access"] = "write"
        else:
            raise Unresolved("Unknown source cell operation")
        self.emit(opcode, inputs, output, [node["id"]], details)

    def typed_operation(self, kind, inputs, dtype, node):
        types = [self.values[value]["dtype"] for value in inputs]
        numeric = dtype in ("float32", "int32", "uint32")
        if kind.startswith("Compare"):
            relation = kind[len("Compare") :]
            if relation not in ("Eq", "NotEq", "Lt", "LtE", "Gt", "GtE"):
                raise Unresolved("Unknown comparison relation")
            if len(types) != 2 or types[0] != types[1] or dtype != "bool":
                raise Unresolved("Comparison needs equal declared input types")
            if types[0] not in ("float32", "int32", "uint32"):
                raise Unresolved("Predicate comparison needs Boolean lowering")
            return ("FSETP" if types[0] == "float32" else "ISETP"), dict(
                relation=relation,
                operand_dtype=types[0],
                unordered_result=(
                    relation == "NotEq" if types[0] == "float32" else None
                ),
                predicate_result=True,
            )
        if kind in ("BitAnd", "BitOr", "And", "Or", "Not"):
            count = 1 if kind == "Not" else 2
            if types != [dtype] * count:
                raise Unresolved(
                    "Logic operands have different declared types"
                )
            if dtype == "bool":
                return "PLOP3", dict(boolean_operation=kind)
            if dtype in ("int32", "uint32") and kind in ("BitAnd", "BitOr"):
                return "LOP3", dict(bit_operation=kind, word_bits=32)
            raise Unresolved("Unknown numeric/Boolean logical operation")
        if kind == "Select":
            if types != ["bool", dtype, dtype]:
                raise Unresolved(
                    "Select needs a predicate and equal data types"
                )
            if dtype == "bool":
                return "PLOP3", dict(boolean_operation="Select")
            return "SEL", dict(
                payload_dtype=dtype,
                select_semantics="eager_typed_branches",
            )
        if kind in ("AllSync", "AnySync"):
            if types != ["uint32", "bool"] or dtype != "bool":
                raise Unresolved("Vote requires an explicit uint32 mask")
            mask = node.get("participating_mask")
            if (
                type(mask) is not int
                or not 0 < mask < 2**32
                or mask != node.get("active_entry_mask")
            ):
                raise Unresolved("Vote participating/entry masks differ")
            return "VOTE", dict(
                vote_operation="all" if kind == "AllSync" else "any",
                participating_mask=mask,
                declared_result=node["declared_result"],
                mask_precondition=(
                    "Every named lane enters the same vote; explicit mask "
                    "equals the participating execution mask"
                ),
                predicate_bank_assumption="P result; no UP routing inferred",
            )
        if kind == "ActiveMask":
            if types or dtype != "uint32":
                raise Unresolved("ActiveMask result must be an unsigned word")
            return "ACTIVEMASK", dict(
                result_dtype="uint32",
                declared_active_entry_mask=node["declared_active_entry_mask"],
                lowering_assumption="One abstract active-mask operation",
            )
        if kind == "Sqrt" and types == ["float32"] and dtype == "float32":
            return "MUFU.SQRT", dict(
                lowering_assumption=(
                    "Approximate native SQRT path, no refinement"
                ),
            )
        if kind == "Abs" and types == ["float32"] and dtype == "float32":
            return "LOP3", dict(
                bit_operation="clear_sign_bit",
                immediate_mask=0x7FFFFFFF,
                immediate_encoding_assumption=True,
            )
        if numeric and types == [dtype] * (
            1 if kind in ("USub", "UAdd") else 2
        ):
            if dtype == "float32":
                ops = dict(
                    Add="FADD",
                    Sub="FADD",
                    Mult="FMUL",
                    USub="FADD",
                    UAdd="MOV",
                )
                if kind in ops:
                    if kind == "USub":
                        inputs.append(
                            self.literal(base.constant_payload(0, dtype))
                        )
                    return ops[kind], dict(
                        source_operation=kind,
                        logical_operation=dict(
                            Add="a + b",
                            Sub="a + (-b)",
                            Mult="a * b",
                            USub="(-a) + 0",
                            UAdd="a",
                        )[kind],
                        operand_modifiers=dict(
                            Add=["identity", "identity"],
                            Sub=["identity", "negate"],
                            Mult=["identity", "identity"],
                            USub=["negate", "identity"],
                            UAdd=["identity"],
                        )[kind],
                        explicit_zero_input=(1 if kind == "USub" else None),
                    )
            elif kind in ("Add", "Sub", "Mult", "USub", "UAdd"):
                if kind == "UAdd":
                    return "MOV", dict(
                        source_operation=kind,
                        logical_operation="a",
                        operand_modifiers=["identity"],
                        word_bits=32,
                        signed=dtype == "int32",
                    )
                zero = self.literal(base.constant_payload(0, dtype))
                if kind == "USub":
                    inputs.extend([zero, zero])
                else:
                    inputs.append(zero)
                return "IMAD" if kind == "Mult" else "IADD3", dict(
                    source_operation=kind,
                    logical_operation=dict(
                        Add="a + b + 0",
                        Sub="a + (-b) + 0",
                        Mult="a * b + 0",
                        USub="(-a) + 0 + 0",
                    )[kind],
                    operand_modifiers=dict(
                        Add=["identity", "identity", "identity"],
                        Sub=["identity", "negate", "identity"],
                        Mult=["identity", "identity", "identity"],
                        USub=["negate", "identity", "identity"],
                    )[kind],
                    explicit_zero_inputs=(
                        [1, 2] if kind == "USub" else [2]
                    ),
                    wrap_bits=32,
                    signed=dtype == "int32",
                )
        raise Unresolved(
            f"No admitted typed operation {kind}/{types}->{dtype}"
        )

    def boundary_ref(self, source_value):
        """Describe a cut value without fabricating an addressable register."""
        value = self.source[source_value]
        cells = [
            item["cell"]
            for item in self.graph["final_cells"]
            if item["value"] == source_value
            and item["boundary"]
            and item["cell"][0] not in self.promoted
        ]
        if len(cells) == 1:
            return dict(source_value=source_value, named_cell=cells[0])
        if (
            value["kind"] == "live_in"
            and isinstance(value.get("label"), list)
            and value["label"][0] not in self.promoted
        ):
            return dict(source_value=source_value, named_cell=value["label"])
        if source_value in self.source_values:
            return dict(
                source_value=source_value,
                lowered_value=self.source_values[source_value],
            )
        raise Unresolved("Caller cut value has no proved location")

    def build(self):
        observable = set(self.graph["observable_values"])
        consumers = defaultdict(list)
        for node in self.original_nodes:
            for value in node["inputs"]:
                consumers[value].append(node["id"])
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
            key, kind = node["id"], node["kind"]
            if key in skipped:
                continue
            if kind.startswith("element_"):
                self.memory(node)
                continue
            if kind == "BranchDecision":
                inputs = [self.mapped(v, node) for v in node["inputs"]]
                if (
                    len(inputs) != 1
                    or self.values[inputs[0]]["dtype"] != "bool"
                ):
                    raise Unresolved("Branch decision needs one predicate")
                if node["outputs"]:
                    raise Unresolved("Branch decision cannot define a value")
                self.emit(
                    "BRA",
                    inputs,
                    None,
                    [key],
                    semantics=dict(
                        explicit_selected_path=True,
                        selected_path=node.get("selected_path"),
                        selection_reason=node["decision_reason"],
                        reconvergence_instructions="unresolved",
                    ),
                )
                continue
            if len(node["outputs"]) != 1:
                raise Unresolved("An expression needs one typed result")
            output_id = node["outputs"][0]
            dtype = self.source[output_id]["dtype"]
            raw = self.source[node["inputs"][0]] if node["inputs"] else None
            if kind == "cast" and raw["kind"] == "constant":
                payload = base.constant_payload(raw["constant"], dtype)
                self.source_values[output_id] = self.literal(payload)
                self.source_nodes[key] = []
                continue
            if kind == "cast" and raw["dtype"] == dtype:
                self.source_values[output_id] = self.mapped(
                    node["inputs"][0], node
                )
                self.source_nodes[key] = sorted(
                    {
                        parent
                        for before in node["order_predecessors"]
                        for parent in self.source_nodes.get(before, [])
                    }
                )
                continue
            source_ids = [key]
            if key in fusion:
                product = self.original_nodes[fusion[key]]
                product_value = product["outputs"][0]
                inputs = [self.mapped(v, product) for v in product["inputs"]]
                inputs.append(
                    self.mapped(
                        next(v for v in node["inputs"] if v != product_value),
                        node,
                    )
                )
                source_ids = [product["id"], key]
                opcode, semantics = "FFMA", dict(single_use_contraction=True)
            else:
                inputs = [self.mapped(v, node) for v in node["inputs"]]
                if kind in ("Minimum", "Maximum"):
                    if (
                        node.get("callable_identity")
                        not in ("builtins.min", "builtins.max")
                        or dtype != "float32"
                        or [self.values[v]["dtype"] for v in inputs]
                        != ["float32", "float32"]
                    ):
                        raise Unresolved(
                            "Min/max needs exact builtins binding"
                        )
                    condition = self.value(
                        "bool", "expression", f"minmax:{key}"
                    )
                    self.emit(
                        "FSETP",
                        inputs[::-1],
                        condition,
                        [key],
                        semantics=dict(
                            relation="Lt" if kind == "Minimum" else "Gt",
                            operand_dtype="float32",
                            ordered=True,
                        ),
                    )
                    inputs = [condition, inputs[1], inputs[0]]
                    opcode, semantics = (
                        "SEL",
                        dict(
                            payload_dtype="float32",
                            source_primitive=node["callable_identity"],
                            equal_or_unordered_retains_first=True,
                        ),
                    )
                elif kind == "Div" and dtype == "float32":
                    if [self.values[v]["dtype"] for v in inputs] != [
                        dtype
                    ] * 2:
                        raise Unresolved("Division operands need FP32 types")
                    reciprocal = self.value(dtype, "expression", f"rcp:{key}")
                    self.emit(
                        "MUFU.RCP",
                        [inputs[1]],
                        reciprocal,
                        [key],
                        semantics=dict(approximate=True, refinement=False),
                    )
                    inputs = [inputs[0], reciprocal]
                    opcode, semantics = "FMUL", dict(reciprocal_division=True)
                elif kind == "cast":
                    before = self.values[inputs[0]]["dtype"]
                    if before == "bool" and dtype in ("int32", "uint32"):
                        predicate = inputs[0]
                        true_word = self.literal(
                            base.constant_payload(1, dtype)
                        )
                        false_word = self.literal(
                            base.constant_payload(0, dtype)
                        )
                        inputs = [predicate, true_word, false_word]
                        opcode, semantics = (
                            "SEL",
                            dict(
                                predicate_to_word=True,
                                predicate_input=0,
                                true_payload_input=1,
                                false_payload_input=2,
                                canonical_true_word=1,
                                canonical_false_word=0,
                            ),
                        )
                    elif before in ("int32", "uint32") and dtype == "bool":
                        zero = self.literal(base.constant_payload(0, before))
                        inputs.append(zero)
                        opcode, semantics = "ISETP", dict(
                            relation="NotEq",
                            operand_dtype=before,
                            zero_operand_input=1,
                            zero_operand=0,
                        )
                    elif before in ("int32", "uint32") and dtype == "float32":
                        opcode, semantics = (
                            "I2F",
                            dict(rounding="nearest_even"),
                        )
                    else:
                        raise Unresolved(f"Unproved cast {before}->{dtype}")
                else:
                    opcode, semantics = self.typed_operation(
                        kind, inputs, dtype, node
                    )
            output = self.value(dtype, "expression", f"source:{output_id}")
            self.source_values[output_id] = output
            self.emit(opcode, inputs, output, source_ids, semantics=semantics)
        boundary = set()
        for item in self.graph["final_cells"]:
            if not item["boundary"]:
                continue
            cell, value = item["cell"], item["value"]
            boundary.add(value)
            if cell[0] in self.promoted:
                self.observables.add(self.mapped(value))
            else:
                self.final_memory[json.dumps(cell)] = self.semantic(value)
        for value in observable - boundary:
            self.observables.add(self.mapped(value))
        required = set(self.graph.get("required_control_nodes", []))
        if any(not self.source_nodes.get(key) for key in required):
            raise Unresolved("A required runtime control node was erased")
        return dict(
            nodes=self.nodes,
            values=self.values,
            materialization=self.mode,
            observable_values=sorted(self.observables),
            initial_memory=self.initial_memory,
            final_memory=self.final_memory,
            layouts=list(self.layouts.values()),
            named_local_frame_bytes=self.named_frame_bytes,
            source_node_mapping=[
                self.source_nodes.get(identifier, [])
                for identifier in range(len(self.original_nodes))
            ],
            source_value_mapping=[
                self.source_values.get(identifier)
                for identifier in range(len(self.source))
            ],
            source_read_mapping=[
                self.read_values.get(identifier)
                for identifier in range(len(self.original_nodes))
            ],
            rewrites=self.rewrites,
            caller_cuts=[
                dict(
                    **cut,
                    lowered_live_ins=[
                        self.boundary_ref(v) for v in cut["live_ins"]
                    ],
                    lowered_observable_outputs=[
                        self.boundary_ref(v) for v in cut["observable_outputs"]
                    ],
                )
                for cut in self.graph.get("certificates", [])
            ],
            declared_float_domain=self.graph["semantic_contract"][
                "floating_point"
            ],
            source_control=self.graph.get(
                "control", self.graph.get("controls")
            ),
            required_control_nodes=sorted(required),
            scope="typed execution trace under explicit compiler alternatives",
            native_instruction_bytes=None,
            limits=[
                "Dynamic iteration instances do not prove hot-code copies",
                "All source-selected control operations are retained",
                "Register classes and opcode families are "
                "lowering assumptions",
                "No latency, instruction-cache penalty or native label "
                "is input",
            ],
        )


class BankAllocation:
    """Allocate a fixed source-order alternative with two physical banks.

    Predicate eviction uses an explicit canonical 0/1 word and local
    memory. The policy is deterministic farthest-next-use, not optimal.
    """

    def __init__(self, lowered, gpr_budget, predicate_budget):
        self.graph = lowered
        self.values = lowered["values"]
        self.budget = dict(
            R=base.exact_int(gpr_budget, "GPR budget", 1),
            P=base.exact_int(predicate_budget, "predicate budget", 1),
        )
        self.slots = {"R": {}, "P": {}}
        self.home = {}
        self.memory = {}
        self.free_offsets = []
        self.next_offset = lowered["named_local_frame_bytes"]
        self.events = []
        self.peaks = dict(R=0, P=0)
        self.uses = defaultdict(deque)
        for node in lowered["nodes"]:
            for value in set(node["inputs"]):
                self.uses[value].append(node["id"])
        for value in lowered["observable_values"]:
            self.uses[value].append(len(lowered["nodes"]))
        self.pinned = {v["id"] for v in self.values if v["kind"] == "base"}
        self.position = -1
        self.protected = set()
        self.initial = []
        for value in self.values:
            key = value["id"]
            if value["kind"] not in ("live_in", "base"):
                continue
            if not self.uses[key] and key not in self.pinned:
                continue
            kind = bank(value["dtype"])
            free = self.free(kind)
            if not free:
                raise Unresolved("Entry values exceed explicit bank budget")
            reg = free[0]
            self.slots[kind][reg] = key
            self.initial.append(self.ref(key, kind, reg))
        self.update_peaks()

    def ref(self, value, kind, register):
        return dict(value=value, bank=kind, register=register)

    def free(self, kind):
        return [
            i for i in range(self.budget[kind]) if i not in self.slots[kind]
        ]

    def update_peaks(self):
        for kind in self.peaks:
            self.peaks[kind] = max(self.peaks[kind], len(self.slots[kind]))

    def location(self, value, kind=None):
        kinds = (kind,) if kind else ("R", "P")
        for item in kinds:
            for reg, stored in self.slots[item].items():
                if stored == value:
                    return self.ref(value, item, reg)
        return None

    def event(self, kind, opcode, reads, writes, kills=(), **detail):
        record = dict(
            id=len(self.events),
            kind=kind,
            opcode=opcode,
            source_position=self.position,
            reads=reads,
            writes=writes,
            kills=list(kills),
            **detail,
        )
        for ref in kills:
            del self.slots[ref["bank"]][ref["register"]]
        for ref in writes:
            self.slots[ref["bank"]][ref["register"]] = ref["value"]
        self.update_peaks()
        record["resident"] = {k: len(v) for k, v in self.slots.items()}
        self.events.append(record)

    def spill_offset(self, value):
        if value not in self.home:
            if self.free_offsets:
                offset = min(self.free_offsets)
                self.free_offsets.remove(offset)
            else:
                offset = self.next_offset
                self.next_offset += 4
            self.home[value] = offset
        return self.home[value]

    def reserve(self, kind):
        free = self.free(kind)
        if free:
            return free[0]
        candidates = [
            v
            for v in self.slots[kind].values()
            if v not in self.protected and v not in self.pinned
        ]
        if not candidates:
            raise Unresolved(f"Simultaneous {kind} operands exceed budget")
        victim = max(
            candidates,
            key=lambda v: (
                self.uses[v][0] if self.uses[v] else float("inf"),
                v,
            ),
        )
        self.evict(victim, kind)
        return self.free(kind)[0]

    def evict(self, value, kind):
        ref = self.location(value, kind)
        if not self.uses[value] or self.values[value]["kind"] == "constant":
            self.event("release", None, [], [], [ref])
            return
        offset = self.spill_offset(value)
        if self.memory.get(offset) == value:
            self.event("release", None, [], [], [ref])
            return
        if kind == "R":
            self.event(
                "spill",
                "STL",
                [ref, self.base_ref()],
                [],
                [ref],
                offset=offset,
                bytes=4,
                value=value,
            )
        else:
            self.protected.add(value)
            register = self.reserve("R")
            word = self.ref(value, "R", register)
            self.event(
                "predicate_to_word",
                "SEL",
                [ref],
                [word],
                canonical_words=[0, 1],
            )
            self.event(
                "spill",
                "STL",
                [word, self.base_ref()],
                [],
                [word, ref],
                offset=offset,
                bytes=4,
                value=value,
            )
            self.protected.remove(value)
        self.memory[offset] = value

    def base_ref(self):
        key = next(
            v["id"] for v in self.values if v["semantic"] == "base:local"
        )
        return self.location(key, "R")

    def ensure(self, value):
        record = self.values[value]
        kind = bank(record["dtype"])
        if kind == "P" and record["kind"] == "constant":
            return dict(
                value=value, bank="PT", constant=record["constant"]["value"]
            )
        ref = self.location(value, kind)
        if ref:
            return ref
        if record["kind"] == "constant":
            ref = self.ref(value, kind, self.reserve(kind))
            self.event(
                "constant", "MOV", [], [ref], payload=record["constant"]
            )
            return ref
        if value not in self.home:
            raise Unresolved("A value has neither register nor proved home")
        offset = self.home[value]
        if self.memory.get(offset) != value:
            raise Unresolved("Spill home no longer holds the requested value")
        if kind == "R":
            ref = self.ref(value, "R", self.reserve("R"))
            self.event(
                "reload",
                "LDL",
                [self.base_ref()],
                [ref],
                offset=offset,
                bytes=4,
                value=value,
            )
        else:
            # Reserve P first: evicting another predicate may itself need R.
            register = self.reserve("P")
            word = self.ref(value, "R", self.reserve("R"))
            self.event(
                "reload",
                "LDL",
                [self.base_ref()],
                [word],
                offset=offset,
                bytes=4,
                value=value,
            )
            ref = self.ref(value, "P", register)
            self.event(
                "word_to_predicate",
                "ISETP",
                [word],
                [ref],
                [word],
                relation="NotEqZero",
            )
        return ref

    def collect_dead(self):
        dead = [
            self.ref(v, kind, reg)
            for kind in ("R", "P")
            for reg, v in list(self.slots[kind].items())
            if not self.uses[v] and v not in self.pinned
        ]
        if dead:
            self.event("release", None, [], [], dead)
        for value, offset in list(self.home.items()):
            if not self.uses[value]:
                del self.home[value]
                self.memory.pop(offset, None)
                self.free_offsets.append(offset)
                self.events.append(
                    dict(
                        id=len(self.events),
                        kind="free_home",
                        opcode=None,
                        source_position=self.position,
                        reads=[],
                        writes=[],
                        kills=[],
                        offset=offset,
                        value=value,
                        resident={k: len(v) for k, v in self.slots.items()},
                    )
                )

    def build(self):
        for node in self.graph["nodes"]:
            self.position = node["id"]
            self.protected = set(node["inputs"])
            reads = [self.ensure(v) for v in node["inputs"]]
            writes = [
                self.ref(
                    v,
                    bank(self.values[v]["dtype"]),
                    self.reserve(bank(self.values[v]["dtype"])),
                )
                for v in node["outputs"]
            ]
            self.event(
                "source",
                node["opcode"],
                reads,
                writes,
                node=node["id"],
                memory=node["memory"],
                semantics=node["semantics"],
            )
            for value in set(node["inputs"]):
                if self.uses[value].popleft() != self.position:
                    raise Unresolved("The fixed schedule changed")
            self.protected = set()
            self.collect_dead()
        exits = []
        for value in self.graph["observable_values"]:
            ref = self.location(value, bank(self.values[value]["dtype"]))
            if ref:
                exits.append(dict(value=value, location=ref))
            elif value in self.home:
                exits.append(dict(value=value, offset=self.home[value]))
            elif self.values[value]["kind"] == "constant":
                exits.append(
                    dict(value=value, constant=self.values[value]["constant"])
                )
            else:
                raise Unresolved("Observable output was lost")
        return dict(
            budgets=self.budget,
            initial=self.initial,
            events=self.events,
            exits=exits,
            peak_resident=self.peaks,
            local_frame_bytes=self.next_offset,
            named_local_frame_bytes=self.graph["named_local_frame_bytes"],
            spill_slots_extent_bytes=(
                self.next_offset - self.graph["named_local_frame_bytes"]
            ),
            policy="fixed_source_order_farthest_next_use_separate_R_P",
            predicate_spill_representation="canonical_0_or_1_uint32",
            optimum_claim=False,
        )


def verify_typed_lowering(graph, lowered, compiler):
    """Check exact operand forms against typed source operations."""
    materialization = lowered.get("materialization")
    if materialization not in ("promote", "addressable"):
        raise ValueError("Typed lowering materialization differs")
    expected = TypedLowering(graph, compiler, materialization).build()
    if lowered != expected:
        raise ValueError("Typed lowering differs from exact source replay")
    source_values = graph["values"]
    values = lowered["values"]
    checked = 0

    def operation(source_node):
        mapped = lowered["source_node_mapping"][source_node["id"]]
        if len(mapped) != 1:
            raise ValueError("Typed source operation has no unique form")
        return lowered["nodes"][mapped[0]]

    def zero_input(node, position, dtype):
        value = values[node["inputs"][position]]
        if value.get("constant") != base.constant_payload(0, dtype):
            raise ValueError("Typed arithmetic zero operand differs")

    for source_node in graph["nodes"]:
        kind = source_node["kind"]
        if not source_node["outputs"]:
            continue
        dtype = source_values[source_node["outputs"][0]]["dtype"]
        mapped = lowered["source_node_mapping"][source_node["id"]]
        if kind in ("Add", "Mult") and len(mapped) == 1:
            candidate = lowered["nodes"][mapped[0]]
            if candidate["opcode"] == "FFMA":
                if (
                    compiler["fp32_contract"] is not True
                    or candidate["semantics"]
                    != {
                        "single_use_contraction": True,
                        "fp32_flush_subnormals": compiler[
                            "fp32_flush_subnormals"
                        ],
                    }
                    or len(candidate["inputs"]) != 3
                ):
                    raise ValueError("Contracted FP32 operand form differs")
                checked += 1
                continue
        if kind not in ("Add", "Sub", "Mult", "USub", "UAdd", "cast"):
            continue
        if kind == "cast":
            before = source_values[source_node["inputs"][0]]["dtype"]
            raw = source_values[source_node["inputs"][0]]
            if raw["kind"] == "constant" or before == dtype:
                if mapped:
                    raise ValueError("Erased typed cast emitted an operation")
                continue
            node = operation(source_node)
            if before == "bool" and dtype in ("int32", "uint32"):
                if (
                    node["opcode"] != "SEL"
                    or len(node["inputs"]) != 3
                    or node["semantics"]
                    != {
                        "predicate_to_word": True,
                        "predicate_input": 0,
                        "true_payload_input": 1,
                        "false_payload_input": 2,
                        "canonical_true_word": 1,
                        "canonical_false_word": 0,
                    }
                ):
                    raise ValueError("Boolean-to-word operand form differs")
                if values[node["inputs"][1]].get("constant") != {
                    "dtype": dtype,
                    "value": 1,
                }:
                    raise ValueError("Boolean true payload differs")
                zero_input(node, 2, dtype)
            elif before in ("int32", "uint32") and dtype == "bool":
                if (
                    node["opcode"] != "ISETP"
                    or len(node["inputs"]) != 2
                    or node["semantics"]
                    != {
                        "relation": "NotEq",
                        "operand_dtype": before,
                        "zero_operand_input": 1,
                        "zero_operand": 0,
                    }
                ):
                    raise ValueError("Word-to-Boolean operand form differs")
                zero_input(node, 1, before)
            elif before in ("int32", "uint32") and dtype == "float32":
                if node["opcode"] != "I2F" or len(node["inputs"]) != 1:
                    raise ValueError("Integer-to-FP32 operand form differs")
            else:
                raise ValueError("Unproved source cast reached validation")
            checked += 1
            continue
        if dtype not in ("float32", "int32", "uint32"):
            continue
        node = operation(source_node)
        if dtype == "float32":
            expected_opcode = dict(
                Add="FADD",
                Sub="FADD",
                Mult="FMUL",
                USub="FADD",
                UAdd="MOV",
            )[kind]
            expected_modifiers = dict(
                Add=["identity", "identity"],
                Sub=["identity", "negate"],
                Mult=["identity", "identity"],
                USub=["negate", "identity"],
                UAdd=["identity"],
            )[kind]
            expected_logical = dict(
                Add="a + b",
                Sub="a + (-b)",
                Mult="a * b",
                USub="(-a) + 0",
                UAdd="a",
            )[kind]
            expected_zero = 1 if kind == "USub" else None
            if (
                node["opcode"] != expected_opcode
                or len(node["inputs"]) != len(expected_modifiers)
                or node["semantics"]
                != {
                    "source_operation": kind,
                    "logical_operation": expected_logical,
                    "operand_modifiers": expected_modifiers,
                    "explicit_zero_input": expected_zero,
                    "fp32_flush_subnormals": compiler[
                        "fp32_flush_subnormals"
                    ],
                }
            ):
                raise ValueError("FP32 arithmetic operand form differs")
            if expected_zero is not None:
                zero_input(node, expected_zero, dtype)
        else:
            if kind == "UAdd":
                expected_opcode = "MOV"
                expected_logical = "a"
                expected_modifiers = ["identity"]
                zero_positions = None
            else:
                expected_opcode = "IMAD" if kind == "Mult" else "IADD3"
                expected_logical = dict(
                    Add="a + b + 0",
                    Sub="a + (-b) + 0",
                    Mult="a * b + 0",
                    USub="(-a) + 0 + 0",
                )[kind]
                expected_modifiers = dict(
                    Add=["identity", "identity", "identity"],
                    Sub=["identity", "negate", "identity"],
                    Mult=["identity", "identity", "identity"],
                    USub=["negate", "identity", "identity"],
                )[kind]
                zero_positions = [1, 2] if kind == "USub" else [2]
            if (
                node["opcode"] != expected_opcode
                or len(node["inputs"]) != len(expected_modifiers)
                or node["semantics"]
                != {
                    "source_operation": kind,
                    "logical_operation": expected_logical,
                    "operand_modifiers": expected_modifiers,
                    **(
                        {"explicit_zero_inputs": zero_positions}
                        if zero_positions is not None
                        else {}
                    ),
                    (
                        "word_bits" if kind == "UAdd" else "wrap_bits"
                    ): 32,
                    "signed": dtype == "int32",
                }
            ):
                raise ValueError("Integer arithmetic operand form differs")
            for position in zero_positions or []:
                zero_input(node, position, dtype)
        checked += 1
    return dict(status="PASS", typed_operand_forms=checked)


def verify_allocation(lowered, allocation):
    """Replay typed bank, spill and named-memory conservation witnesses."""
    values = lowered["values"]
    budgets = allocation["budgets"]
    slots = {"R": {}, "P": {}}
    memory = {}
    named = dict(lowered["initial_memory"])
    peaks = dict(R=0, P=0)
    extent = lowered["named_local_frame_bytes"]
    next_source = 0

    def admitted(ref):
        value = base.exact_int(ref["value"], "value")
        if value >= len(values):
            raise ValueError("Unknown allocation value")
        kind = ref["bank"]
        if kind == "PT":
            record = values[value]
            if (
                record["dtype"] != "bool"
                or record["kind"] != "constant"
                or ref
                != dict(
                    value=value,
                    bank="PT",
                    constant=record["constant"]["value"],
                )
            ):
                raise ValueError("Unproved predicate literal")
            return value, "predicate"
        if kind not in slots:
            raise ValueError("Unknown register bank")
        reg = base.exact_int(ref["register"], "register")
        if reg >= budgets[kind] or set(ref) != {"value", "bank", "register"}:
            raise ValueError("Register outside its declared bank")
        return value, "predicate" if kind == "P" else "word"

    def read(ref):
        tag = admitted(ref)
        if ref["bank"] == "PT":
            return tag
        actual = slots[ref["bank"]].get(ref["register"])
        if actual is None or actual[0] != ref["value"]:
            raise ValueError("Read lost or changed its typed value")
        return actual

    expected_entry = {
        v["id"]
        for v in values
        if v["kind"] in ("base", "live_in")
        and (
            v["kind"] == "base"
            or v["id"] in lowered["observable_values"]
            or any(v["id"] in n["inputs"] for n in lowered["nodes"])
        )
    }
    if {r["value"] for r in allocation["initial"]} != expected_entry:
        raise ValueError("Entry ABI membership differs")
    if len(allocation["initial"]) != len(expected_entry):
        raise ValueError("Duplicate entry value")
    for ref in allocation["initial"]:
        tag = admitted(ref)
        if ref["bank"] != bank(values[ref["value"]]["dtype"]):
            raise ValueError("Initial value has wrong bank")
        if ref["register"] in slots[ref["bank"]]:
            raise ValueError("Two entry values share a register")
        slots[ref["bank"]][ref["register"]] = tag
    for kind in slots:
        peaks[kind] = len(slots[kind])
    for index, event in enumerate(allocation["events"]):
        if event["id"] != index:
            raise ValueError("Noncontiguous allocation events")
        reads = [read(r) for r in event["reads"]]
        writes = [admitted(r) for r in event["writes"]]
        kind = event["kind"]
        if kind == "source":
            node = lowered["nodes"][next_source]
            if (
                event["node"] != next_source
                or event["source_position"] != next_source
                or event["opcode"] != node["opcode"]
                or event["memory"] != node["memory"]
                or event["semantics"] != node["semantics"]
                or [r[0] for r in reads] != node["inputs"]
                or [r[0] for r in writes] != node["outputs"]
            ):
                raise ValueError("Typed source operation changed")
            for ref in event["reads"] + event["writes"]:
                expected = bank(values[ref["value"]]["dtype"])
                if ref["bank"] not in (expected, "PT"):
                    raise ValueError("Source operand in wrong bank")
            if any(tag[1] == "canonical_bool" for tag in reads):
                raise ValueError("Word predicate used without conversion")
            detail = node["memory"]
            if detail:
                cell = json.dumps(detail["cell"])
                semantic = detail["expected_semantic"]
                if detail["access"] == "read":
                    if named.get(cell) != semantic:
                        raise ValueError("Named load lost its source semantic")
                    if values[node["outputs"][0]]["semantic"] != semantic:
                        raise ValueError("Named load result differs")
                else:
                    if values[node["inputs"][1]]["semantic"] != semantic:
                        raise ValueError("Named store value differs")
                    named[cell] = semantic
            next_source += 1
        elif kind == "constant":
            if (
                reads
                or len(writes) != 1
                or event["opcode"] != "MOV"
                or event["payload"] != values[writes[0][0]].get("constant")
                or event["writes"][0]["bank"] != "R"
            ):
                raise ValueError("Unproved literal materialization")
        elif kind == "predicate_to_word":
            if (
                len(reads) != 1
                or len(writes) != 1
                or reads[0] != (writes[0][0], "predicate")
                or event["reads"][0]["bank"] != "P"
                or event["writes"][0]["bank"] != "R"
                or event["opcode"] != "SEL"
                or event["canonical_words"] != [0, 1]
            ):
                raise ValueError("Unproved predicate conversion")
            writes = [(writes[0][0], "canonical_bool")]
        elif kind == "word_to_predicate":
            if (
                len(reads) != 1
                or len(writes) != 1
                or reads[0] != (writes[0][0], "canonical_bool")
                or event["writes"][0]["bank"] != "P"
                or event["opcode"] != "ISETP"
                or event["relation"] != "NotEqZero"
            ):
                raise ValueError("Unproved word-to-predicate conversion")
        elif kind in ("spill", "reload"):
            offset = base.exact_int(event["offset"], "spill offset")
            value = event["value"]
            if (
                offset < lowered["named_local_frame_bytes"]
                or offset % 4
                or event["bytes"] != 4
            ):
                raise ValueError("Spill overlaps named storage")
            extent = max(extent, offset + 4)
            if kind == "spill":
                if (
                    len(reads) != 2
                    or writes
                    or event["opcode"] != "STL"
                    or reads[0][0] != value
                    or event["reads"][0]["bank"] != "R"
                ):
                    raise ValueError("Unproved spill store")
                memory[offset] = reads[0]
            else:
                if (
                    len(reads) != 1
                    or len(writes) != 1
                    or event["opcode"] != "LDL"
                    or writes[0][0] != value
                    or event["writes"][0]["bank"] != "R"
                    or memory.get(offset, (None,))[0] != value
                ):
                    raise ValueError("Reload value differs from its spill")
                writes = [memory[offset]]
            if values[reads[-1][0]]["semantic"] != "base:local":
                raise ValueError("Spill address lacks the retained stack base")
        elif kind == "free_home":
            if reads or writes or event["opcode"] is not None:
                raise ValueError("Home release has an instruction")
            if memory.get(event["offset"], (None,))[0] != event["value"]:
                raise ValueError("Released a different spill lifetime")
            del memory[event["offset"]]
        elif kind == "release":
            if reads or writes or event["opcode"] is not None:
                raise ValueError("Register release has an instruction")
        else:
            raise ValueError("Unproved allocation operation")
        for ref in event["kills"]:
            read(ref)
            del slots[ref["bank"]][ref["register"]]
        for ref, tag in zip(event["writes"], writes):
            if ref["register"] in slots[ref["bank"]]:
                raise ValueError("Write overwrites a retained value")
            slots[ref["bank"]][ref["register"]] = tag
        residents = {k: len(v) for k, v in slots.items()}
        if residents != event["resident"]:
            raise ValueError("Resident count differs from the trace")
        for item in peaks:
            peaks[item] = max(peaks[item], residents[item])
    if next_source != len(lowered["nodes"]):
        raise ValueError("Source operations omitted")
    exits = allocation["exits"]
    if len(exits) != len(lowered["observable_values"]) or {
        e["value"] for e in exits
    } != set(lowered["observable_values"]):
        raise ValueError("Observable membership differs")
    for item in exits:
        value = item["value"]
        if set(item) == {"value", "location"}:
            if read(item["location"])[0] != value:
                raise ValueError("Observable register differs")
        elif set(item) == {"value", "offset"}:
            if memory.get(item["offset"], (None,))[0] != value:
                raise ValueError("Observable spill differs")
        elif set(item) == {"value", "constant"}:
            if item["constant"] != values[value].get("constant"):
                raise ValueError("Observable literal differs")
        else:
            raise ValueError("Unproved observable location")
    for cell, semantic in lowered["final_memory"].items():
        if named.get(cell) != semantic:
            raise ValueError("Final named memory differs")
    if (
        peaks != allocation["peak_resident"]
        or extent != allocation["local_frame_bytes"]
        or allocation["spill_slots_extent_bytes"]
        != extent - lowered["named_local_frame_bytes"]
    ):
        raise ValueError("Allocation resource extents differ")
    return dict(
        status="PASS",
        source_operations=next_source,
        allocation_events=len(allocation["events"]),
        peaks=peaks,
        local_frame_bytes=extent,
        limitation="typed alternative conservation, not actual native proof",
    )


def validate_source(graph):
    """Validate typed topology, aliases, cuts and actual source-file joins."""
    if (
        graph.get("kind") != "typed_implicit_execution_region"
        or graph["compilation_check"]["native_overloads"] != 0
        or graph["compilation_check"]["batch_kernel_requested"] is not False
        or graph["compilation_check"]["device_function_executed"] is not False
    ):
        raise ValueError("Need a complete source-only implicit region")
    nodes, values = graph["nodes"], graph["values"]
    if [n["id"] for n in nodes] != list(range(len(nodes))):
        raise ValueError("Source node IDs differ")
    if [v["id"] for v in values] != list(range(len(values))):
        raise ValueError("Source value IDs differ")
    if any(
        v["dtype"] not in DTYPES | {"literal_int", "literal_float"}
        for v in values
    ):
        raise ValueError("Source graph contains an untyped value")
    declared_live = {v["id"] for v in values if v["kind"] == "live_in"}
    if set(graph["live_ins"]) != declared_live:
        raise ValueError("Source live-in membership differs")
    ancestors = []
    accesses = defaultdict(list)
    consumers = defaultdict(set)
    value_edges = []
    for node in nodes:
        parents = set(node["order_predecessors"])
        for value in node["inputs"]:
            consumers[value].add(node["id"])
            producer = values[value]["producer"]
            if producer is not None:
                parents.add(producer)
                value_edges.append(
                    dict(producer=producer, consumer=node["id"], value=value)
                )
        if any(parent < 0 or parent >= node["id"] for parent in parents):
            raise ValueError("Source dependency is not topological")
        before = 0
        for parent in parents:
            before |= ancestors[parent] | (1 << parent)
        ancestors.append(before)
        for value in node["outputs"]:
            if values[value]["producer"] != node["id"]:
                raise ValueError("Source output producer differs")
        source = node.get("source", {})
        template = source.get("hot_template")
        identity = source.get("hot_template_identity")
        fixed = [
            item
            for item in source.get("loop_indices", [])
            if item.get("recurrent") is False
        ]
        if (
            not isinstance(template, dict)
            or identity
            != hashlib.sha256(
                json.dumps(template, sort_keys=True).encode()
            ).hexdigest()
            or source.get("template_is_not_native_copy_identity") is not True
            or template.get("fixed_indices") != fixed
            or any(item.get("recurrent") is not False for item in fixed)
        ):
            raise ValueError("Source hot-template identity differs")
        if node["kind"] == "BranchDecision":
            if (
                node["outputs"]
                or len(node["inputs"]) != 1
                or values[node["inputs"][0]]["dtype"] != "bool"
                or type(node.get("selected_path")) is not bool
                or not node.get("decision_reason")
                or node.get("is_codegen_constant") is not False
            ):
                raise ValueError("Runtime branch decision is not explicit")
        if node["kind"] == "ActiveMask":
            mask = node.get("declared_active_entry_mask")
            region = source.get("runtime_region")
            expected = (
                region["entry_mask"]
                if region is not None
                else graph["regime"]["step_entry_mask"]
            )
            if (
                node["inputs"]
                or len(node["outputs"]) != 1
                or values[node["outputs"][0]]["dtype"] != "uint32"
                or type(mask) is not int
                or not 0 < mask < 2**32
                or mask != expected
                or values[node["outputs"][0]].get("declared_lane_mask")
                != mask
            ):
                raise ValueError("ActiveMask source regime differs")
        if node["kind"] in ("AllSync", "AnySync"):
            mask = node.get("participating_mask")
            entry = node.get("active_entry_mask")
            region = source.get("runtime_region")
            if (
                len(node["inputs"]) != 2
                or len(node["outputs"]) != 1
                or values[node["inputs"][0]]["dtype"] != "uint32"
                or values[node["inputs"][1]]["dtype"] != "bool"
                or values[node["outputs"][0]]["dtype"] != "bool"
                or type(mask) is not int
                or not 0 < mask < 2**32
                or mask != entry
                or region is None
                or region.get("entry_mask") != entry
                or values[node["inputs"][0]].get("declared_lane_mask")
                != mask
                or type(node.get("declared_result")) is not bool
                or node.get("mask_equality")
                != "explicit uniform participating-path contract"
            ):
                raise ValueError("Vote source regime differs")
        if node["kind"].startswith("element_"):
            cell = node.get("cell")
            if (
                not isinstance(cell, list)
                or len(cell) != 4
                or cell[3] not in ("float32", "int32", "uint32")
                or type(cell[1]) is not int
                or type(cell[2]) is not int
                or cell[1] < 0
                or cell[2] - cell[1] != 4
                or cell[1] % 4
            ):
                raise ValueError("Source memory cell is not aligned typed32")
            accesses[json.dumps(cell)].append(node["id"])
    if graph["value_edges"] != value_edges:
        raise ValueError("Declared source value edges differ")
    # Every same-cell read follows the last write; each write follows all
    # reads and the prior write. This checks value edges separately from
    # memory-order frontiers without imposing lexical order on other cells.
    for ordered in accesses.values():
        writer, readers = None, []
        for identifier in ordered:
            node = nodes[identifier]
            required = [] if writer is None else [writer]
            if node["kind"] == "element_write_alias":
                required += readers
            if any(
                not (ancestors[identifier] >> item) & 1 for item in required
            ):
                raise ValueError("Same-cell alias order is not preserved")
            if node["kind"] == "element_write_alias":
                writer, readers = identifier, []
            else:
                readers.append(identifier)
    current_cells = {}
    for value in values:
        label = value.get("label")
        if value["kind"] != "live_in" or not isinstance(label, list):
            continue
        token = json.dumps(label)
        if token in current_cells:
            raise ValueError("A source cell has multiple live-in values")
        current_cells[token] = value["id"]
    for node in nodes:
        if not node["kind"].startswith("element_"):
            continue
        token = json.dumps(node["cell"])
        if node["kind"] == "element_read_alias":
            if (
                node["outputs"]
                or node["inputs"] != [current_cells.get(token)]
            ):
                raise ValueError("Source read does not consume current cell")
        elif node["kind"] == "element_write_alias":
            if node["outputs"] or len(node["inputs"]) != 1:
                raise ValueError("Source write does not define one cell value")
            current_cells[token] = node["inputs"][0]
        else:
            raise ValueError("Unknown source element operation")
    controls = set(graph["required_control_nodes"])
    actual_controls = {n["id"] for n in nodes if n["kind"] == "BranchDecision"}
    if controls != actual_controls:
        raise ValueError("Required runtime control membership differs")
    branch_nodes = [n for n in nodes if n["kind"] == "BranchDecision"]
    branch_controls = [
        item
        for item in graph["controls"]
        if item["kind"] == "runtime_branch_choice"
    ]
    node_choices = Counter(
        (
            node["source"]["path"],
            node["source"]["line"],
            node["inputs"][0],
            node["selected_path"],
            node["decision_reason"],
        )
        for node in branch_nodes
    )
    control_choices = Counter(
        (
            item["source"]["path"],
            item["source"]["line"],
            item["predicate"],
            item["choice"],
            item["reason"],
        )
        for item in branch_controls
        if item.get("is_codegen_constant") is False
        and type(item.get("choice")) is bool
    )
    if node_choices != control_choices:
        raise ValueError("Runtime branch control records differ")
    for key, choice in graph["branch_choices"].items():
        matching = [
            item
            for item in branch_controls
            if f"{Path(item['source']['path']).name}:{item['source']['line']}"
            == key
        ]
        if (
            type(choice) is not bool
            or not matching
            or any(item["choice"] is not choice for item in matching)
        ):
            raise ValueError("Explicit branch choice is not executed")
    observable = set(graph["observable_values"])
    roots = [cut for cut in graph["certificates"] if cut["function"] == "f0"]
    if len(roots) != 1 or set(roots[0]["observable_outputs"]) != observable:
        raise ValueError("Root caller-cut outputs differ")
    source_calls = [
        call for call in graph["calls"] if call["kind"] == "source_call"
    ]
    if len(source_calls) != len(graph["certificates"]):
        raise ValueError("Caller-cut/source-call membership differs")
    for cut, call in zip(graph["certificates"], source_calls):
        membership = set(range(call["first_node"], call["end_node"]))
        used = {
            value
            for identifier in membership
            for value in (
                nodes[identifier]["inputs"] + nodes[identifier]["outputs"]
            )
        }
        incoming = {
            value
            for value in used
            if values[value]["producer"] not in membership
            and values[value]["kind"] != "constant"
        }
        outgoing = set()
        for value in values:
            identifier = value["id"]
            producer = value["producer"]
            if value["kind"] == "constant":
                continue
            existed = producer is None or producer < call["end_node"]
            later = identifier in observable or any(
                node >= call["end_node"]
                for node in consumers.get(identifier, ())
            )
            if existed and later:
                outgoing.add(identifier)
                if producer is None or producer < call["first_node"]:
                    incoming.add(identifier)
        if (
            cut.get("context") != call["context"]
            or cut.get("function") != call["function"]
            or cut.get("node_ids") != sorted(membership)
            or cut.get("live_ins") != sorted(incoming)
            or cut.get("observable_outputs") != sorted(outgoing)
            or cut.get("scope")
            != "lexical interval with conservative caller live-through"
        ):
            raise ValueError("Caller cut is not a complete graph interval")
    extents = defaultdict(int)
    boundary_storage = set()
    for allocation in graph["allocations"]:
        view = allocation["view"]
        if allocation.get("boundary") is True:
            boundary_storage.add(view["storage"])
        extents[view["storage"]] = max(
            extents[view["storage"]], view["offset"] + view["bytes"]
        )
    final_cells = {}
    for item in graph["final_cells"]:
        storage, low, high, dtype = item["cell"]
        token = json.dumps(item["cell"])
        if (
            token in final_cells
            or item["value"] not in range(len(values))
            or values[item["value"]]["dtype"] != dtype
            or low < 0
            or high - low != 4
            or high > extents[storage]
            or type(item["boundary"]) is not bool
            or item["boundary"] != (storage in boundary_storage)
            or current_cells.get(token) != item["value"]
        ):
            raise ValueError("Final cell exceeds an actual typed allocation")
        final_cells[token] = item["value"]
    if final_cells != current_cells:
        raise ValueError("Final cells do not equal the source cell frontier")
    contract = graph.get("semantic_contract", {})
    if (
        contract.get("path") != "one declared uniform successful path"
        or contract.get("floating_point")
        != (
            "FP32 source operations; selected-path replay assumes finite "
            "inputs and intermediates"
        )
        or contract.get("nan_sensitive_primitives")
        != (
            "source callable identity is retained; Python min/max is not "
            "identified with fmin/fmax"
        )
        or "low-32-bit" not in contract.get("integer_values", "")
        or "unresolved" not in contract.get("native_integer_width", "")
        or contract.get("vote_mask")
        != "participating mask equals declared active-entry mask"
    ):
        raise ValueError("Source numeric/mask contract is not explicit")
    records = [graph["provenance"]["extractor"], graph["caller"]["source"]]
    records += graph["provenance"]["dependencies"]
    records += [
        dict(
            path=f["source"]["source_path"],
            sha256=f["source"]["source_sha256"],
        )
        for f in graph["provenance"]["functions"]
    ]
    records += [
        node["primitive_source"]
        for node in nodes
        if node["kind"] in ("AllSync", "AnySync")
    ]
    checked = {}
    for record in records:
        path = record["path"]
        if path not in checked:
            checked[path] = base.digest(path)
        if checked[path] != record["sha256"]:
            raise ValueError(f"Source bytes changed: {path}")
    known_paths = set(checked)
    if any(node["source"]["path"] not in known_paths for node in nodes):
        raise ValueError("Source operation is outside retained function files")
    if any(
        item["source"]["path"] not in known_paths
        for item in graph["controls"]
    ):
        raise ValueError("Source control is outside retained function files")
    regime = validate_regime(graph, nodes)
    return checked, regime


def make_plan(graph, architecture, compiler, materialization="promote"):
    """Build and replay an explicit two-bank typed compiler alternative."""
    checked, regime = validate_source(graph)
    required = {
        "name",
        "provenance",
        "gpr_budget",
        "predicate_budget",
        "gpr_scope",
        "predicate_scope",
    }
    if (
        set(architecture) != required
        or not architecture["provenance"]
        or architecture["gpr_scope"] != "per_thread_scenario"
        or architecture["predicate_scope"] != "per_thread_scenario"
    ):
        raise ValueError("Architecture needs explicit bank-budget provenance")
    required_compiler = {
        "name",
        "provenance",
        "fp32_flush_subnormals",
        "fp32_contract",
        "division",
        "sqrt",
        "numeric_literals",
        "predicate_literals",
        "integer_dynamic_width_bits",
        "predicate_spills",
        "schedule",
    }
    if (
        set(compiler) != required_compiler
        or not compiler["name"]
        or not isinstance(compiler["provenance"], list)
        or not compiler["provenance"]
        or type(compiler["fp32_flush_subnormals"]) is not bool
        or type(compiler["fp32_contract"]) is not bool
        or compiler["division"] != "approximate_reciprocal_multiply"
        or compiler["sqrt"] != "approximate_native_no_refinement"
        or compiler["numeric_literals"] != "materialized_gpr"
        or compiler["predicate_literals"] != "PT_or_inverted_PT"
        or compiler["integer_dynamic_width_bits"] != 32
        or compiler["predicate_spills"] != "canonical_uint32_local"
        or compiler["schedule"] != "source_order"
    ):
        raise ValueError("Compiler alternative is incomplete or unsupported")
    for record in compiler["provenance"]:
        if (
            "path" in record
            and base.digest(record["path"]) != record["sha256"]
        ):
            raise ValueError("Compiler-alternative source bytes changed")
    lowered = TypedLowering(graph, compiler, materialization).build()
    typed_forms = verify_typed_lowering(graph, lowered, compiler)
    templates = {}
    for node in lowered["nodes"]:
        for context in node["source_contexts"]:
            identity = context["hot_template_identity"]
            item = templates.setdefault(
                identity,
                dict(
                    hot_template_identity=identity,
                    source=context["hot_template"],
                    selected_trace_instances=0,
                    modeled_opcodes=Counter(),
                ),
            )
            item["selected_trace_instances"] += 1
            item["modeled_opcodes"][node["opcode"]] += 1
    for item in templates.values():
        item["modeled_opcodes"] = dict(item["modeled_opcodes"])
    allocation = BankAllocation(
        lowered,
        architecture["gpr_budget"],
        architecture["predicate_budget"],
    ).build()
    verified = verify_allocation(lowered, allocation)
    return dict(
        schema=1,
        kind="conditional_typed_implicit_native_plan",
        provenance=dict(
            lowerer=dict(path=str(SCRIPT), sha256=base.digest(SCRIPT)),
            base_sha256=BASE_SHA,
            source_files=checked,
        ),
        architecture=architecture,
        compiler_alternative=compiler,
        lowering=lowered,
        static_hot_templates=list(templates.values()),
        allocation=allocation,
        verification=dict(**verified, typed_forms=typed_forms),
        assumptions=dict(
            integer_arithmetic="32-bit wrapping; signed comparisons typed",
            floating_mode=(
                "FTZ" if compiler["fp32_flush_subnormals"] else "non_FTZ"
            ),
            floating_division=compiler["division"],
            floating_sqrt=compiler["sqrt"],
            minmax="builtins ordered comparison then select; first on ties",
            numeric_literal_forms=compiler["numeric_literals"],
            predicate_literal_forms=compiler["predicate_literals"],
            address_words="one explicit 32-bit base per local/shared space",
            native_control="BRA family; reconvergence work unresolved",
            register_policy="fixed order with explicit predicate word spills",
        ),
        dynamic_work=dict(
            trace_operations=len(lowered["nodes"]),
            trace_opcodes=dict(Counter(n["opcode"] for n in lowered["nodes"])),
            allocation_opcodes=dict(
                Counter(
                    e["opcode"] for e in allocation["events"] if e["opcode"]
                )
            ),
            static_hot_instruction_bytes=None,
            hot_templates_are_not_native_copy_counts=True,
            iterations=regime,
            iteration_values_are_explicit_scenarios=True,
        ),
        native_labels_consumed=False,
        measured_iteration_counts_consumed=False,
        complete_kernel_prediction=False,
        unresolved=[
            "Actual native opcode forms, operand folding and scheduling",
            "Reconvergence, uniform-bank routing and kernel ABI temporaries",
            "Native static loop replication and instruction working set",
            "Memory warp layout, cache reuse and load/store service",
        ],
    )


def main():
    """Read actual JSON source and explicit architecture; write one plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--materialization",
        choices=("promote", "addressable"),
        default="promote",
    )
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text())
    architecture = json.loads(args.architecture.read_text())
    compiler = json.loads(args.compiler.read_text())
    result = make_plan(graph, architecture, compiler, args.materialization)
    result["provenance"]["graph"] = dict(
        path=str(args.graph.resolve()),
        sha256=base.digest(args.graph),
    )
    result["provenance"]["architecture"] = dict(
        path=str(args.architecture.resolve()),
        sha256=base.digest(args.architecture),
    )
    result["provenance"]["compiler_alternative"] = dict(
        path=str(args.compiler.resolve()),
        sha256=base.digest(args.compiler),
    )
    base.write_json(args.out, result)
    print(json.dumps(result["verification"], sort_keys=True))


if __name__ == "__main__":
    main()
