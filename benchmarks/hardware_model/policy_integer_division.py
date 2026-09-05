"""Cost constant floor division over source-proved positive int32 ranges."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from benchmarks.hardware_model import source_value_graph as source


WORD = 2**32
MAXIMUM = 2**31 - 1


def reciprocal_form(divisor, minimum, maximum):
    """Derive a sufficient exact reciprocal proof without sampling."""
    if any(type(x) is not int for x in (divisor, minimum, maximum)):
        raise ValueError("Division proof requires exact integer bounds")
    if not (1 <= divisor <= MAXIMUM and 0 <= minimum <= maximum <= MAXIMUM):
        raise ValueError("Only nonnegative int32 by positive int32 admitted")
    if divisor == 1:
        return dict(kind="identity", divisor=1, minimum=minimum,
                    maximum=maximum)
    if divisor & (divisor - 1) == 0:
        return dict(kind="unsigned_shift", divisor=divisor, minimum=minimum,
                    maximum=maximum, shift=divisor.bit_length() - 1)
    multiplier = (WORD + divisor - 1) // divisor
    excess = multiplier * divisor - WORD
    if maximum * excess >= WORD:
        raise ValueError("Source range needs a correction form not admitted")
    return dict(kind="bounded_unsigned_mulhi", divisor=divisor,
                minimum=minimum, maximum=maximum, multiplier=multiplier,
                word_bits=32, reciprocal_excess=excess,
                exactness_product=maximum * excess,
                exactness_strict_limit=WORD)


def range_proof(values, loops, numerator):
    """Use the complete source loop, never a selected execution witness."""
    value = values[numerator]
    if (value["dtype"] != "int32"
            or value.get("source_origin") != "runtime_loop_induction"
            or value.get("external_kernel_input") is not False):
        raise ValueError("Division numerator needs an internal int32 range")
    loop_id = value["policy_loop_id"]
    control = loops[loop_id]
    instances = control["structure"]["execution_instances"]
    if not instances:
        raise ValueError("Division range is empty")
    selected = instances[value["execution_position"]]
    if (selected["codegen_constant"]
            or selected["lane"] != value["template_lane"]
            or selected["part"] != value["loop_part"]):
        raise ValueError("Division numerator differs from captured induction")
    start = instances[0]["value"]
    step = instances[1]["value"] - start if len(instances) > 1 else 1
    if any(type(item["value"]) is not int
           or item["position"] != position
           or item["value"] != start + step * position
           for position, item in enumerate(instances)):
        raise ValueError("Division range is not the captured affine range")
    bounds = [start, instances[-1]["value"]]
    return dict(numerator_source_value=numerator, policy_loop_id=loop_id,
                range_start=start, range_step=step, range_count=len(instances),
                minimum=min(bounds), maximum=max(bounds),
                source=deepcopy(control["source"]),
                proof_domain="complete captured source range")


class SourceConstantDivision:
    """Retain exact integer division and a complete induction-domain proof."""

    def binary(self, node, left, right):
        """Fold only actual source constants; cost surviving integer work."""
        if not isinstance(node.op, ast.FloorDiv):
            return super().binary(node, left, right)
        operands = [self.scalar(item, node) for item in (left, right)]
        types = [self.values[item["identity"]]["dtype"] for item in operands]
        if any(dtype not in ("literal_int", "int32") for dtype in types):
            self.unknown(node, "FloorDiv requires source integer operands")
        divisor = operands[1]["raw"]
        if not isinstance(divisor, (int, np.int32)):
            self.unknown(node, "FloorDiv divisor is not a source constant")
        divisor = int(divisor)
        if not 1 <= divisor <= MAXIMUM:
            self.unknown(node, "FloorDiv divisor is outside positive int32")
        numerator = operands[0]["raw"]
        if numerator is not source.UNKNOWN:
            numerator = int(numerator)
            if not -(2**31) <= numerator <= MAXIMUM:
                self.unknown(node, "FloorDiv constant numerator exceeds int32")
            # Python integer // is floor, including negative constants.
            return source.item(np.int32(numerator // divisor))
        proof = range_proof(self.values, self.policy_loop_controls,
                            operands[0]["identity"])
        form = reciprocal_form(divisor, proof["minimum"], proof["maximum"])
        result = self.operation(
            "FloorDiv", operands, node, dtype="int32",
            division_range_proof=proof, division_form=form,
            integer_semantics="exact Python floor on proved source domain",
        )[0]
        trace = self.trace_raw(operands[0])
        if trace is source.UNKNOWN:
            return result
        return self.record_trace(result, np.int32(int(trace) // divisor))


def source_form(graph, node):
    """Reconstruct division's proof and algebraic constants from source."""
    if node["kind"] != "FloorDiv" or len(node["inputs"]) != 2:
        raise ValueError("Expected a binary source FloorDiv")
    proof = range_proof(graph["values"], graph["policy_loops"],
                        node["inputs"][0])
    denominator = graph["values"][node["inputs"][1]]
    if denominator["kind"] != "constant":
        raise ValueError("Division denominator must remain source constant")
    divisor = denominator["constant"]["value"]
    form = reciprocal_form(divisor, proof["minimum"], proof["maximum"])
    if (node["division_range_proof"] != proof
            or node["division_form"] != form):
        raise ValueError("Division proof differs from full source range")
    return proof, form


class PolicyIntegerDivision:
    """Emit a conditional mul-high form with exact producer/liveness edges."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integer_division_forms = []

    def typed_operation(self, kind, inputs, dtype, node):
        """Keep high-word multiplication distinct from low-word IMAD."""
        if kind != "FloorDiv":
            return super().typed_operation(kind, inputs, dtype, node)
        if dtype != "int32" or len(inputs) != 2:
            raise ValueError("FloorDiv requires int32 output and two operands")
        proof, form = source_form(self.graph, node)
        return "POLICY_FLOOR_DIV", dict(
            source_operation=kind, range_proof=proof, form=form)

    def emit(self, opcode, inputs, output, source_ids, memory=None,
             semantics=None):
        """Make every materialized multiplier and actual operand visible."""
        if opcode != "POLICY_FLOOR_DIV":
            return super().emit(opcode, inputs, output, source_ids,
                                memory=memory, semantics=semantics)
        if len(source_ids) != 1 or memory is not None:
            raise ValueError("Division must bind one pure source operation")
        form = semantics["form"]
        operands = [inputs[0]]
        native_opcode = "MOV"
        if form["kind"] == "bounded_unsigned_mulhi":
            operands.append(self.literal(dict(
                dtype="uint32", value=form["multiplier"])))
            native_opcode = "IMAD.HI.U32"
        elif form["kind"] == "unsigned_shift":
            operands.append(self.literal(dict(dtype="uint32",
                                              value=form["shift"])))
            native_opcode = "SHF.R.U32"
        details = dict(
            **semantics, native_form_is_conditional=True,
            division_form_role="exact_nonnegative_quotient",
            logical_operation={
                "MOV": "a",
                "IMAD.HI.U32": "(uint32(a) * uint32(b)) >> 32",
                "SHF.R.U32": "uint32(a) >> b; RZ is the native zero-fill operand",
            }[native_opcode],
            source_denominator_eliminated_by_algebra=True,
            native_zero_operand=(None if native_opcode == "MOV" else "RZ"),
            native_shift_form=(dict(
                opcode="SHF.R.U32.HI",
                operands=["destination", "RZ", "shift", "numerator"],
                qualification="Exact saved PTXAS operand spelling; RZ "
                "is described only as zero-fill",
            ) if native_opcode == "SHF.R.U32" else None),
            result_signedness="nonnegative quotient bit-identical int32",
        )
        identifier = super().emit(
            native_opcode, operands, output, source_ids, semantics=details)
        self.source_nodes[source_ids[0]] = [identifier]
        self.integer_division_forms.append(dict(
            source_node=source_ids[0], typed_nodes=[identifier],
            form=deepcopy(form), range_proof=deepcopy(semantics["range_proof"])))
        return identifier

    def build(self):
        """Bind conditional form source bytes to the allocated plan."""
        result = super().build()
        result["integer_division_forms"] = self.integer_division_forms
        result["integer_division_lowering"] = dict(
            source_path=str(Path(__file__).resolve()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            assumption="Range-aware mul-high compiler alternative; exact "
            "arithmetic, not proof installed backend chooses this form")
        verify_division_forms(self.graph, result)
        return result


def verify_division_forms(graph, lowered):
    """Join exact source bounds to every emitted quotient and its operands."""
    records = {item["source_node"]: item
               for item in lowered["integer_division_forms"]}
    source_nodes = [node for node in graph["nodes"]
                    if node["kind"] == "FloorDiv"]
    if set(records) != {node["id"] for node in source_nodes}:
        raise ValueError("Division form coverage differs from source")
    values = lowered["values"]
    for source_node in source_nodes:
        proof, form = source_form(graph, source_node)
        record = records[source_node["id"]]
        mapped = lowered["source_node_mapping"][source_node["id"]]
        if (record["form"] != form or record["range_proof"] != proof
                or record["typed_nodes"] != mapped or len(mapped) != 1):
            raise ValueError("Division source-to-typed form differs")
        node = lowered["nodes"][mapped[0]]
        expected_opcode = {
            "identity": "MOV", "unsigned_shift": "SHF.R.U32",
            "bounded_unsigned_mulhi": "IMAD.HI.U32",
        }[form["kind"]]
        if (node["opcode"] != expected_opcode
                or node["source_nodes"] != [source_node["id"]]
                or node["semantics"]["form"] != form
                or node["semantics"]["range_proof"] != proof):
            raise ValueError("Division native operation differs")
        if (values[node["inputs"][0]]["semantic"] !=
                f"source:{source_node['inputs'][0]}"
                or values[node["outputs"][0]]["semantic"] !=
                f"source:{source_node['outputs'][0]}"):
            raise ValueError("Division producer or consumer identity differs")
        if form["kind"] != "identity":
            constant = values[node["inputs"][1]]["constant"]
            expected = form.get("multiplier", form.get("shift"))
            if constant != dict(dtype="uint32", value=expected):
                raise ValueError("Division algebraic operand differs")
    return dict(status="SOURCE_DIVISION_FORMS_PASS", count=len(source_nodes))


def division_catalog(catalog):
    """Expose the named physical form-transfer without inventing a latency."""
    result = deepcopy(catalog)
    service = deepcopy(result["instructions"]["IMAD"])
    service["confidence"] = "cross_form_and_architecture_transfer"
    service["conditional_form"] = dict(
        assumption="Transfer plain IMAD dependent latency and integer route "
        "capacity to unsigned high-word IMAD. This is not a measured HI "
        "latency. Retain the published 4/5-cycle alternatives.",
        source="Existing catalog IMAD primary A100/Turing evidence",
        native_opcode="IMAD.HI.U32")
    result["instructions"]["IMAD.HI.U32"] = service
    shift = deepcopy(result["instructions"]["LOP3"])
    shift["confidence"] = "cross_form_and_architecture_transfer"
    shift["conditional_form"] = dict(
        assumption="Transfer 4-cycle integer ALU latency and 64-result "
        "integer route capacity to SHF.R.U32. Explicit zero-fill operand RZ; "
        "not a target SHF microbenchmark.",
        source="Existing catalog Turing integer ALU evidence")
    result["instructions"]["SHF.R.U32"] = shift
    return result
