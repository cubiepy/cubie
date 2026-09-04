"""Project proved FP32 operand forms into a conditional NativePlan.

This module leaves the frozen NativePlan and forwarding forecasts intact.
It models exact constant negation, one native FP32 literal operand, and a
register-release schedule as separately checkable compiler alternatives.
"""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from benchmarks.hardware_model import native_plan as base
from benchmarks.hardware_model import native_plan_forwarding as forwarding


SCRIPT = Path(__file__).resolve()
BASE_SHA = (
    "f547ee91e5f3a390d68c8113e8eb438bde03438935ca8d4b294e148fb9480471"
)
FORWARDING_SHA = (
    "d1ea624ae986373831177b26de03ed7b4ecf113d83b2622528cac234b3bd1add"
)
LITERAL_OPCODES = {
    "FADD": (1,),
    "FMUL": (1,),
    "FFMA": (1, 2),
}


def canonical_digest(value):
    """Hash a JSON-domain value with stable separators and key order."""
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(data).hexdigest()


def fp32_from_bits(bits):
    """Return the exact FP32 represented by a little-endian hex word."""
    if not isinstance(bits, str) or len(bits) != 8:
        raise ValueError("FP32 bits must be one four-byte hex word")
    try:
        raw = bytes.fromhex(bits)
    except ValueError as error:
        raise ValueError("FP32 bits are not hexadecimal") from error
    return np.float32(struct.unpack("<f", raw)[0])


def fp32_bits(value):
    """Return the little-endian bit identity of an FP32 value."""
    return struct.pack("<f", np.float32(value)).hex()


def negated_payload(payload):
    """Toggle the sign bit of one finite normalized FP32 constant."""
    if payload.get("dtype") != "float32" or set(payload) != {
        "dtype",
        "bits",
        "value",
    }:
        raise ValueError("Constant negation requires an exact FP32 payload")
    value = fp32_from_bits(payload["bits"])
    if not np.isfinite(value):
        raise ValueError("Nonfinite constant negation is not admitted")
    word = int.from_bytes(bytes.fromhex(payload["bits"]), "little")
    bits = (word ^ 0x80000000).to_bytes(4, "little").hex()
    result = dict(dtype="float32", bits=bits, value=float(fp32_from_bits(bits)))
    if fp32_bits(result["value"]) != bits:
        raise ValueError("Constant sign fold changed FP32 bits")
    return result


def constant_token(payload):
    """Return the semantic token used by the frozen lowering."""
    return "constant:" + json.dumps(payload, sort_keys=True)


def is_literal_candidate(value):
    """Admit a finite FP32 literal other than zero or unit magnitude."""
    if value.get("kind") != "constant" or value.get("dtype") != "float32":
        return False
    payload = value.get("constant", {})
    if set(payload) != {"dtype", "bits", "value"}:
        return False
    scalar = fp32_from_bits(payload["bits"])
    return bool(np.isfinite(scalar) and scalar != 0 and abs(scalar) != 1)


def bypass_predecessors(nodes, removed):
    """Replace dependencies on removed pure nodes by their dependencies."""
    dependencies = {node["id"]: set(node["predecessors"]) for node in nodes}

    def expand(node_id, visiting):
        if node_id not in removed:
            return {node_id}
        if node_id in visiting:
            raise ValueError("Removed-node dependencies contain a cycle")
        result = set()
        for predecessor in dependencies[node_id]:
            result.update(expand(predecessor, visiting | {node_id}))
        return result

    for node in nodes:
        if node["id"] in removed:
            continue
        expanded = set()
        for predecessor in node["predecessors"]:
            expanded.update(expand(predecessor, set()))
        node["predecessors"] = sorted(expanded)


def fold_constant_negations(plan):
    """Fold only exact FNEG operations whose input is an FP32 constant."""
    projected = copy.deepcopy(plan)
    values = {value["id"]: value for value in projected["values"]}
    if len(values) != len(projected["values"]):
        raise ValueError("Lowered values have duplicate identities")
    constants = {
        json.dumps(value["constant"], sort_keys=True): value["id"]
        for value in projected["values"]
        if value["kind"] == "constant"
    }
    next_value = max(values, default=-1) + 1
    replacements = {}
    removed = set()
    witnesses = []
    for node in projected["nodes"]:
        if node["opcode"] != "FNEG" or len(node["inputs"]) != 1:
            continue
        source = values[node["inputs"][0]]
        if source["kind"] != "constant" or source["dtype"] != "float32":
            continue
        if len(node["outputs"]) != 1 or node["memory"] is not None:
            raise ValueError("Constant FNEG has an unsupported result shape")
        payload = negated_payload(source["constant"])
        token = json.dumps(payload, sort_keys=True)
        replacement = constants.get(token)
        if replacement is None:
            replacement = next_value
            next_value += 1
            record = dict(
                id=replacement,
                dtype="float32",
                words=1,
                kind="constant",
                semantic=constant_token(payload),
                constant=payload,
            )
            projected["values"].append(record)
            values[replacement] = record
            constants[token] = replacement
        output = node["outputs"][0]
        replacements[output] = replacement
        removed.add(node["id"])
        witnesses.append(
            dict(
                node=node["id"],
                input=node["inputs"][0],
                output=output,
                replacement=replacement,
                input_bits=source["constant"]["bits"],
                output_bits=payload["bits"],
                rule="toggle_fp32_sign_bit",
            )
        )

    def replaced(value):
        seen = set()
        while value in replacements:
            if value in seen:
                raise ValueError("Constant-fold replacement contains a cycle")
            seen.add(value)
            value = replacements[value]
        return value

    semantic_replacements = {
        values[old]["semantic"]: values[replaced(old)]["semantic"]
        for old in replacements
    }
    for node in projected["nodes"]:
        node["inputs"] = [replaced(value) for value in node["inputs"]]
        if node["memory"] is not None:
            semantic = node["memory"]["expected_semantic"]
            node["memory"]["expected_semantic"] = semantic_replacements.get(
                semantic,
                semantic,
            )
    bypass_predecessors(projected["nodes"], removed)
    projected["nodes"] = [
        node for node in projected["nodes"] if node["id"] not in removed
    ]
    projected["observable_values"] = [
        replaced(value) for value in projected["observable_values"]
    ]
    if len(set(projected["observable_values"])) != len(
        projected["observable_values"]
    ):
        raise ValueError("Constant folding merged distinct observable values")
    for memory in ("initial_memory", "final_memory"):
        projected[memory] = {
            key: semantic_replacements.get(value, value)
            for key, value in projected[memory].items()
        }
    mapping = projected.get("source_node_mapping", {})
    for witness in witnesses:
        for key, native_nodes in list(mapping.items()):
            if witness["node"] in native_nodes:
                mapping[key] = [
                    node for node in native_nodes if node != witness["node"]
                ]
    projected.setdefault("rewrites", []).extend(
        dict(
            source_nodes=next(
                node["source_nodes"]
                for node in plan["nodes"]
                if node["id"] == witness["node"]
            ),
            **witness,
        )
        for witness in witnesses
    )
    return projected, witnesses


def operand_choice(node, values):
    """Choose at most one literal using retained SM89 SASS forms."""
    logical = list(node["inputs"])
    physical = list(range(len(logical)))
    literal = None
    opcode = node["opcode"]
    if opcode in ("FADD", "FMUL") and len(logical) == 2:
        if is_literal_candidate(values[logical[0]]):
            physical = [1, 0]
        candidate = physical[1]
        if is_literal_candidate(values[logical[candidate]]):
            literal = candidate
    elif opcode == "FFMA" and len(logical) == 3:
        if is_literal_candidate(values[logical[0]]):
            physical = [1, 0, 2]
            literal = 0
        elif is_literal_candidate(values[logical[1]]):
            literal = 1
        elif is_literal_candidate(values[logical[2]]):
            literal = 2
    if literal is not None:
        physical_index = physical.index(literal)
        if physical_index not in LITERAL_OPCODES[opcode]:
            raise ValueError("Chosen literal is outside its admitted SASS slot")
    return logical, physical, literal


def project_operand_forms(plan):
    """Return a register-input plan with full logical operand witnesses."""
    projected, folds = fold_constant_negations(plan)
    values = {value["id"]: value for value in projected["values"]}
    counts = Counter()
    commutations = []
    for node in projected["nodes"]:
        logical, order, literal = operand_choice(node, values)
        register_inputs = []
        forms = []
        for physical_index, source_index in enumerate(order):
            value = logical[source_index]
            if source_index == literal:
                record = values[value]
                forms.append(
                    dict(
                        kind="fp32_literal",
                        value=value,
                        source_input_index=source_index,
                        native_operand_index=physical_index + 1,
                        payload=record["constant"],
                    )
                )
                counts[(node["opcode"], physical_index + 1)] += 1
            else:
                forms.append(
                    dict(
                        kind="register",
                        value=value,
                        source_input_index=source_index,
                        native_operand_index=physical_index + 1,
                        register_input_index=len(register_inputs),
                    )
                )
                register_inputs.append(value)
        if order != list(range(len(logical))):
            commutations.append(
                dict(
                    node=node["id"],
                    opcode=node["opcode"],
                    source_input_order=list(range(len(logical))),
                    physical_input_order=order,
                    qualification="finite_fp32_operand_domain",
                )
            )
        node["logical_inputs"] = logical
        node["physical_input_order"] = order
        node["operand_forms"] = forms
        node["inputs"] = register_inputs
    projected["operand_projection"] = dict(
        constant_negation_folds=folds,
        commutations=commutations,
        literal_form_counts=[
            dict(opcode=key[0], native_operand_index=key[1], uses=count)
            for key, count in sorted(counts.items())
        ],
        form_scope=(
            "Saved SM89 whole-entry SASS admits at most one numeric FP32 "
            "literal in FADD/FMUL operand2 or FFMA operand2/3. Source-to-SASS "
            "use selection remains a compiler alternative."
        ),
        excluded_forms=[
            "Zero-register and unit-operand elimination lack exact source-use maps",
            "General CSE, reassociation, and nonconstant sign folding",
            "Literal forms for memory, reciprocal, integer, and predicate operations",
        ],
    )
    projected.setdefault("assumptions", []).extend(
        [
            "Exact constant FNEG toggles only the retained FP32 sign bit",
            "One finite nonzero nonunit FP32 literal occupies an admitted slot",
            "FADD/FMUL or the FFMA product pair commute only for finite inputs",
            "Every other logical operand remains an explicit register input",
            "Literal capability does not claim that the compiler selects every use",
        ]
    )
    verify_operand_projection(plan, projected)
    return projected


def verify_constant_folds(original, projected):
    """Independently verify every removed constant-negation identity."""
    original_nodes = {node["id"]: node for node in original["nodes"]}
    original_values = {value["id"]: value for value in original["values"]}
    values = {value["id"]: value for value in projected["values"]}
    folds = projected["operand_projection"]["constant_negation_folds"]
    removed = set()
    removed_outputs = set()
    replacements = {}
    for witness in folds:
        if set(witness) != {
            "node",
            "input",
            "output",
            "replacement",
            "input_bits",
            "output_bits",
            "rule",
        }:
            raise ValueError("Constant-negation witness schema changed")
        node = original_nodes[witness["node"]]
        source = original_values[witness["input"]]
        replacement = values[witness["replacement"]]
        if (
            node["opcode"] != "FNEG"
            or node["inputs"] != [witness["input"]]
            or node["outputs"] != [witness["output"]]
            or node["memory"] is not None
            or source["kind"] != "constant"
            or replacement["constant"] != negated_payload(source["constant"])
            or witness["input_bits"] != source["constant"]["bits"]
            or witness["output_bits"] != replacement["constant"]["bits"]
            or witness["rule"] != "toggle_fp32_sign_bit"
        ):
            raise ValueError("Constant-negation witness is not exact")
        if node["id"] in removed or node["outputs"][0] in removed_outputs:
            raise ValueError("Constant-negation witness is duplicated")
        removed.add(node["id"])
        removed_outputs.add(node["outputs"][0])
        replacements[node["outputs"][0]] = witness["replacement"]
    expected = {
        node["id"]
        for node in original["nodes"]
        if node["opcode"] == "FNEG"
        and original_values[node["inputs"][0]]["kind"] == "constant"
    }
    if removed != expected or len(folds) != len(expected):
        raise ValueError("Constant-negation fold membership changed")
    return replacements, removed


def verify_operand_projection(original, projected):
    """Prove register projection reconstructs all logical operands."""
    replacements, removed = verify_constant_folds(original, projected)
    original_nodes = {node["id"]: node for node in original["nodes"]}
    original_values = {value["id"]: value for value in original["values"]}
    values = {value["id"]: value for value in projected["values"]}
    nodes = {node["id"]: node for node in projected["nodes"]}
    if set(nodes) != set(original_nodes) - removed:
        raise ValueError("Operand projection changed surviving node membership")

    expected_value_ids = set(original_values) | set(replacements.values())
    if set(values) != expected_value_ids:
        raise ValueError("Operand projection changed value membership")
    for value_id, record in original_values.items():
        if values[value_id] != record:
            raise ValueError("Operand projection changed an original value")

    def replaced(value):
        return replacements.get(value, value)

    semantic_replacements = {
        original_values[old]["semantic"]: values[new]["semantic"]
        for old, new in replacements.items()
    }
    dependency_cache = {}

    def expanded_dependency(node_id, visiting=None):
        if node_id not in removed:
            return {node_id}
        if node_id in dependency_cache:
            return dependency_cache[node_id]
        visiting = set() if visiting is None else visiting
        if node_id in visiting:
            raise ValueError("Original removed dependencies contain a cycle")
        result = set()
        for predecessor in original_nodes[node_id]["predecessors"]:
            result.update(expanded_dependency(predecessor, visiting | {node_id}))
        dependency_cache[node_id] = result
        return result

    expected_observables = [
        replaced(value) for value in original["observable_values"]
    ]
    if projected["observable_values"] != expected_observables:
        raise ValueError("Operand projection changed observable values")
    for field in ("initial_memory", "final_memory"):
        expected_memory = {
            key: semantic_replacements.get(value, value)
            for key, value in original[field].items()
        }
        if projected[field] != expected_memory:
            raise ValueError("Operand projection changed memory boundaries")
    expected_mapping = {
        key: [node for node in mapped if node not in removed]
        for key, mapped in original.get("source_node_mapping", {}).items()
    }
    if projected.get("source_node_mapping", {}) != expected_mapping:
        raise ValueError("Operand projection changed source-node mapping")

    literal_count = 0
    literal_counts = Counter()
    commutations = []
    for node_id, node in nodes.items():
        source = original_nodes[node_id]
        logical = [replaced(value) for value in source["inputs"]]
        if node["logical_inputs"] != logical:
            raise ValueError(
                f"Projected logical operands differ at node {node_id}"
            )
        expected_predecessors = set()
        for predecessor in source["predecessors"]:
            expected_predecessors.update(expanded_dependency(predecessor))
        if node["predecessors"] != sorted(expected_predecessors):
            raise ValueError("Operand projection changed dependencies")
        order = node["physical_input_order"]
        if sorted(order) != list(range(len(logical))):
            raise ValueError("Physical operand order is not a permutation")
        expected_node = dict(opcode=source["opcode"], inputs=logical)
        _, expected_order, expected_literal = operand_choice(
            expected_node,
            values,
        )
        if order != expected_order:
            raise ValueError("Physical operand choice changed")
        forms = node["operand_forms"]
        if len(forms) != len(logical):
            raise ValueError("Operand-form witness has the wrong arity")
        reconstructed = [None] * len(logical)
        registers = []
        literals = []
        for physical_index, form in enumerate(forms):
            source_index = form["source_input_index"]
            if (
                source_index != order[physical_index]
                or form["native_operand_index"] != physical_index + 1
                or form["value"] != logical[source_index]
            ):
                raise ValueError("Physical operand witness changed identity")
            reconstructed[source_index] = form["value"]
            if form["kind"] == "register":
                if form["register_input_index"] != len(registers):
                    raise ValueError("Register-input projection changed order")
                registers.append(form["value"])
            elif form["kind"] == "fp32_literal":
                if set(form) != {
                    "kind",
                    "value",
                    "source_input_index",
                    "native_operand_index",
                    "payload",
                }:
                    raise ValueError("Literal witness schema changed")
                if (
                    node["opcode"] not in LITERAL_OPCODES
                    or physical_index not in LITERAL_OPCODES[node["opcode"]]
                    or not is_literal_candidate(values[form["value"]])
                    or form["payload"] != values[form["value"]]["constant"]
                ):
                    raise ValueError("Literal is outside the admitted form")
                literals.append(form)
                literal_count += 1
                literal_counts[(node["opcode"], physical_index + 1)] += 1
            else:
                raise ValueError("Unknown physical operand form")
        if (
            reconstructed != logical
            or registers != node["inputs"]
            or len(literals) > 1
        ):
            raise ValueError("Logical operands do not reconstruct exactly")
        if order != list(range(len(logical))):
            if node["opcode"] in ("FADD", "FMUL"):
                valid = order == [1, 0]
            elif node["opcode"] == "FFMA":
                valid = order == [1, 0, 2]
            else:
                valid = False
            if not valid:
                raise ValueError("Unproved arithmetic commutation")
            commutations.append(
                dict(
                    node=node_id,
                    opcode=node["opcode"],
                    source_input_order=list(range(len(logical))),
                    physical_input_order=order,
                    qualification="finite_fp32_operand_domain",
                )
            )
        literal_source_indices = {
            form["source_input_index"] for form in literals
        }
        expected_literal_indices = (
            set() if expected_literal is None else {expected_literal}
        )
        if literal_source_indices != expected_literal_indices:
            raise ValueError("Literal operand choice changed")
        expected_memory = copy.deepcopy(source["memory"])
        if expected_memory is not None:
            semantic = expected_memory["expected_semantic"]
            expected_memory["expected_semantic"] = semantic_replacements.get(
                semantic,
                semantic,
            )
        if (
            node["opcode"] != source["opcode"]
            or node["outputs"] != source["outputs"]
            or node["source_nodes"] != source["source_nodes"]
            or node["memory"] != expected_memory
        ):
            raise ValueError("Operand projection changed an instruction")
    expected_literal_counts = [
        dict(opcode=key[0], native_operand_index=key[1], uses=count)
        for key, count in sorted(literal_counts.items())
    ]
    projection = projected["operand_projection"]
    if projection["literal_form_counts"] != expected_literal_counts:
        raise ValueError("Literal-form count summary changed")
    if projection["commutations"] != commutations:
        raise ValueError("Commutation witness summary changed")
    base.native_schedule(projected)
    return dict(
        status="EXACT_LOGICAL_OPERAND_RECONSTRUCTION_PASS",
        surviving_nodes=len(nodes),
        constant_negation_folds=len(removed),
        fp32_literal_uses=literal_count,
    )


def register_release_schedule(plan):
    """Schedule ready nodes by exact dead-word release, then source ID."""
    nodes = {node["id"]: node for node in plan["nodes"]}
    values = {value["id"]: value for value in plan["values"]}
    following = defaultdict(list)
    remaining = {}
    for node_id, node in nodes.items():
        remaining[node_id] = len(node["predecessors"])
        for predecessor in node["predecessors"]:
            if predecessor not in nodes:
                raise ValueError("Schedule dependency is absent")
            following[predecessor].append(node_id)
    uses = Counter(
        value for node in nodes.values() for value in set(node["inputs"])
    )
    observable = set(plan["observable_values"])
    ready = {node_id for node_id, count in remaining.items() if count == 0}
    order = []
    decisions = []
    while ready:
        scores = {}
        for node_id in ready:
            node = nodes[node_id]
            released = sum(
                values[value]["words"]
                for value in set(node["inputs"])
                if uses[value] == 1
                and value not in observable
                and values[value]["kind"] != "constant"
            )
            retained = sum(
                values[value]["words"]
                for value in node["outputs"]
                if value in observable or following[node_id]
            )
            scores[node_id] = released - retained
        selected = min(ready, key=lambda node_id: (-scores[node_id], node_id))
        decisions.append(
            dict(
                position=len(order),
                node=selected,
                ready_nodes=len(ready),
                net_released_words=scores[selected],
            )
        )
        ready.remove(selected)
        order.append(selected)
        for value in set(nodes[selected]["inputs"]):
            uses[value] -= 1
        for successor in following[selected]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.add(successor)
    if len(order) != len(nodes):
        raise ValueError("Register-release schedule did not cover the DAG")
    return order, decisions


def seeded_value(semantic, trial):
    """Map a semantic identity to a bounded, finite test value."""
    raw = hashlib.sha256(f"{trial}:{semantic}".encode()).digest()
    integer = int.from_bytes(raw[:4], "little")
    return np.float32(0.5 + (integer % 4096) / 4096.0)


def evaluate_float_plan(plan, trial):
    """Evaluate the retained FP32 arithmetic for one finite input pattern."""
    values = {value["id"]: value for value in plan["values"]}
    nodes = {node["id"]: node for node in plan["nodes"]}
    state = {}
    for value_id, record in values.items():
        if record["dtype"] != "float32":
            continue
        if record["kind"] == "constant":
            state[value_id] = fp32_from_bits(record["constant"]["bits"])
        elif record["kind"] == "live_in":
            state[value_id] = seeded_value(record["semantic"], trial)
    for node_id in base.native_schedule(plan):
        node = nodes[node_id]
        if node["memory"] is not None:
            if node["memory"]["access"] == "read":
                output = node["outputs"][0]
                state[output] = seeded_value(values[output]["semantic"], trial)
            continue
        inputs = node.get("logical_inputs", node["inputs"])
        operands = [state[value] for value in inputs]
        with np.errstate(all="ignore"):
            if node["opcode"] in ("FADD", "FSUB", "FMUL"):
                function = {
                    "FADD": np.add,
                    "FSUB": np.subtract,
                    "FMUL": np.multiply,
                }[node["opcode"]]
                result = np.float32(function(operands[0], operands[1]))
            elif node["opcode"] == "FNEG":
                result = np.float32(-operands[0])
            elif node["opcode"] == "FFMA":
                result = np.float32(
                    float(operands[0]) * float(operands[1])
                    + float(operands[2])
                )
            elif node["opcode"] == "MUFU.RCP":
                result = np.float32(1.0) / operands[0]
            else:
                raise ValueError("Numeric verifier found an unsupported opcode")
        state[node["outputs"][0]] = result
    return state


def verify_numeric_projection(original, projected, trials=3):
    """Compare every surviving arithmetic value on finite FP32 inputs."""
    original_values = {value["id"]: value for value in original["values"]}
    projected_values = {value["id"]: value for value in projected["values"]}
    surviving = sorted(
        set(original_values)
        & set(projected_values)
        & {
            value
            for node in projected["nodes"]
            for value in node["outputs"]
        }
    )
    comparisons = 0
    for trial in range(trials):
        before = evaluate_float_plan(original, trial)
        after = evaluate_float_plan(projected, trial)
        for value in surviving:
            if original_values[value]["dtype"] != "float32":
                continue
            if fp32_bits(before[value]) != fp32_bits(after[value]):
                raise ValueError("Operand projection changed an FP32 result")
            comparisons += 1
    return dict(
        status="FINITE_FP32_TRACE_EQUIVALENCE_PASS",
        trials=trials,
        compared_values=comparisons,
        domain="deterministic finite normalized inputs",
    )


def validate_saved_prediction(prediction):
    """Revalidate a retained baseline prediction and its source graph."""
    if base.digest(base.SCRIPT) != BASE_SHA:
        raise ValueError("Frozen NativePlan source changed")
    if base.digest(forwarding.SCRIPT) != FORWARDING_SHA:
        raise ValueError("Frozen forwarding source changed")
    kind = prediction.get("kind")
    model_hash = prediction.get("provenance", {}).get("model_source_sha256")
    expected = (
        BASE_SHA
        if kind == "conditional_erk_native_plan"
        else FORWARDING_SHA
        if kind == "conditional_erk_shared_forwarding_plan"
        else None
    )
    if model_hash != expected:
        raise ValueError("Saved prediction does not bind a frozen model")
    graph_record = prediction["provenance"].get("input_graph")
    if not graph_record:
        raise ValueError("Saved prediction lacks an input-graph receipt")
    graph_path = Path(graph_record["path"])
    if base.digest(graph_path) != graph_record["sha256"]:
        raise ValueError("Saved prediction graph bytes changed")
    graph = json.loads(graph_path.read_text())
    base.validate_graph(graph)
    base.validate_construction(graph)
    lowering = prediction["lowering"]
    if prediction["native_schedule"] != base.native_schedule(lowering):
        raise ValueError("Saved baseline schedule changed")
    unlimited = base.Allocation(
        lowering,
        prediction["native_schedule"],
        max(1, len(lowering["values"])),
    ).run()
    base.verify_allocation(lowering, unlimited)
    if unlimited["peak_words"] != prediction["modeled_no_spill_words"]:
        raise ValueError("Saved no-spill frontier changed")
    base.verify_allocation(lowering, prediction["allocation"])
    return graph


def allocation_scenario(plan, order, budget, hardware, block, dynamic, catalog):
    """Allocate one projected schedule under exact SM89 capacities."""
    unlimited = base.Allocation(plan, order, max(1, len(plan["values"]))).run()
    base.verify_allocation(plan, unlimited)
    allocation = base.Allocation(plan, order, budget).run()
    conservation = base.verify_allocation(plan, allocation)
    geometry = base.residency(
        hardware,
        max(1, allocation["peak_words"]),
        block,
        dynamic,
    )
    service = None
    if geometry["feasible"]:
        service = base.service_estimate(allocation["trace"], geometry, catalog)
    return dict(
        schedule=order,
        modeled_no_spill_words=unlimited["peak_words"],
        allocation=allocation,
        conservation=conservation,
        geometry=geometry,
        service=service,
    )


def predict_saved(prediction, register_budget=None, catalog=None):
    """Build source-order and register-release operand alternatives."""
    validate_saved_prediction(prediction)
    original = prediction["lowering"]
    projected = project_operand_forms(original)
    projection = verify_operand_projection(original, projected)
    numeric = verify_numeric_projection(original, projected)
    hardware = prediction["hardware"]
    maximum = hardware["maximum_registers_per_thread"]
    budget = maximum if register_budget is None else base.exact_int(
        register_budget,
        "register_budget",
        1,
    )
    if budget > maximum:
        raise ValueError("Requested budget exceeds the hardware maximum")
    block = prediction["candidate"]["block_size"]
    construction = prediction["provenance"]["construction"]
    dynamic = max(4, construction["shared_stride_bytes"] * block)
    release_order, decisions = register_release_schedule(projected)
    orders = {
        "source_topological": base.native_schedule(projected),
        "register_release": release_order,
    }
    scenarios = {
        name: allocation_scenario(
            projected,
            order,
            budget,
            hardware,
            block,
            dynamic,
            catalog,
        )
        for name, order in orders.items()
    }
    eligible = [
        (name, result)
        for name, result in scenarios.items()
        if result["geometry"]["feasible"]
    ]
    preferred = min(
        eligible,
        key=lambda item: (
            item[1]["allocation"]["spill_bytes"],
            -item[1]["geometry"]["resident_blocks"],
            len(item[1]["allocation"]["trace"]),
            item[0],
        ),
    )[0]
    return dict(
        schema=1,
        kind="conditional_erk_operand_form_plan",
        candidate=prediction["candidate"],
        provenance=dict(
            model_source_sha256=base.digest(SCRIPT),
            frozen_base_source_sha256=BASE_SHA,
            frozen_forwarding_source_sha256=FORWARDING_SHA,
            input_prediction_sha256=canonical_digest(prediction),
            input_graph=prediction["provenance"]["input_graph"],
            construction=construction,
        ),
        hardware=hardware,
        register_budget=budget,
        projected_lowering=projected,
        projection_conservation=projection,
        finite_numeric_check=numeric,
        schedule_scenarios=scenarios,
        register_release_decisions=decisions,
        preferred_schedule_by_resource_equations=preferred,
        selection_order=[
            "fewest exact spill bytes",
            "most capacity-derived resident blocks",
            "fewest retained instruction events",
            "stable schedule name",
        ],
        parameter_policy=(
            "No timing, native register label, or solver winner selects an "
            "operand or schedule. FP32 forms come from retained SASS grammar; "
            "capacity and allocation use hardware words and quanta."
        ),
        status="conditional_compiler_alternatives",
        claim=(
            "Pre-native operand and schedule scenarios; source-use selection, "
            "caller inlining, and installed-compiler allocation remain uncertain"
        ),
    )


def main():
    """Project a frozen saved forecast without importing CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--register-budget", type=int)
    args = parser.parse_args()
    prediction = json.loads(args.prediction.read_text())
    catalog = json.loads(args.catalog.read_text()) if args.catalog else None
    result = predict_saved(prediction, args.register_budget, catalog)
    result["provenance"]["input_prediction"] = dict(
        path=str(args.prediction.resolve()),
        sha256=base.digest(args.prediction),
    )
    if args.catalog:
        result["provenance"]["service_catalog"] = dict(
            path=str(args.catalog.resolve()),
            sha256=base.digest(args.catalog),
        )
    base.write_json(args.output, result)
    summary = {
        name: dict(
            no_spill_words=value["modeled_no_spill_words"],
            spill_bytes=value["allocation"]["spill_bytes"],
            blocks_per_sm=value["geometry"]["resident_blocks"],
        )
        for name, value in result["schedule_scenarios"].items()
    }
    print(
        json.dumps(
            dict(
                output=str(args.output.resolve()),
                preferred=result["preferred_schedule_by_resource_equations"],
                schedules=summary,
            )
        )
    )


if __name__ == "__main__":
    main()
