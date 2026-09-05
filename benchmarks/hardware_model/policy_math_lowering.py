"""Retain source-bound approximate FP32 transcendental instruction forms."""

import hashlib
import json
from pathlib import Path

import numpy as np

from cubie.cuda_backend import CUDA_BACKEND
from benchmarks.hardware_model import native_plan as base
from benchmarks.hardware_model.workload import python_function


MATH_OPERATIONS = {"Exp", "Log", "Log2", "Pow"}
LOG2_E = np.float32(1.4426950216293334961)
LN_2 = np.float32(0.69314718246459960938)
NATIVE_FORM_EVIDENCE = {
    "path": "C:/local_working_projects/cubie-notes/"
    "hardware_unroll_placement/math_forms_e1/receipt.json",
    "sha256": "c2f7d1a1bbae528d8e765f0dc72abdb30028cec4e7d19494d7a89a2814e77a50",
    "scope": "Eleven isolated same-backend FP32 forms, no kernel launches "
    "or timing measurements. Independently disassembled and reviewed.",
}
CALIBRATED_OPTIONS = {
    "fastmath": ["afn", "arcp", "contract", "ftz", "nsz"],
    "lineinfo": False,
    "lto": True,
    "experimental_ast_transforms": True,
}


def math_owners(graph):
    """Join every math operation to its actual captured source callable."""
    calls = {call["context"]: call["function"] for call in graph["calls"]
             if call.get("kind") == "source_call"}
    result = {}
    for node in graph["nodes"]:
        if node["kind"] not in MATH_OPERATIONS:
            continue
        owner = calls[node["source"]["context"]]
        template = node["source"]["hot_template"]["function"]["function"]
        if template != owner:
            raise ValueError("Math call and template owners differ")
        result[str(node["id"])] = owner
    return result


def math_lowering_contract(solver, captured, graph):
    """Bind math forms to actual owning dispatchers and exact JIT flags."""
    bindings = math_owners(graph)
    dispatchers = {captured.identities[id(python_function(dispatcher))]:
                   dispatcher for dispatcher in captured.dispatchers}
    functions = {record["id"]: record for record in captured.functions}
    owners = {}
    for identifier in sorted(set(bindings.values())):
        if identifier not in dispatchers:
            raise ValueError("Math helper has no owning dispatcher flag proof")
        dispatcher = dispatchers[identifier]
        options = dispatcher.targetoptions
        fastmath = options["fastmath"]
        normalized = {
            name: (sorted(getattr(fastmath, "flags", fastmath))
                   if name == "fastmath" else options[name])
            for name in CALIBRATED_OPTIONS
        }
        if normalized != CALIBRATED_OPTIONS:
            raise ValueError("Math owner's JIT flags differ from calibration")
        if (options.get("debug") is not False
                or options.get("opt_level") != 3
                or options.get("ptxas_options") is not None
                or options.get("fast_math") is not None):
            raise ValueError("Math owner has uncalibrated compiler overrides")
        owners[identifier] = dict(
            function=functions[identifier], compiler_options=normalized,
            device=options.get("device"), inline=options.get("inline"),
            debug=False, opt_level=3, ptxas_options=None, fast_math=None,
            native_overloads=len(dispatcher.overloads),
        )
    result = dict(
        backend=CUDA_BACKEND, owners=owners, node_owners=bindings,
        native_form_evidence=dict(NATIVE_FORM_EVIDENCE),
    )
    verify_math_contract(graph, result)
    return result


def verify_math_contract(graph, contract):
    """Reject stale evidence, changed owner joins or uncalibrated flags."""
    evidence = contract["native_form_evidence"]
    if evidence != NATIVE_FORM_EVIDENCE:
        raise ValueError("Math calibration identity differs")
    raw = Path(evidence["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence["sha256"]:
        raise ValueError("Math native-form evidence changed")
    receipt = json.loads(raw)
    if (contract["backend"] != "mlir" or receipt["backend"] != "mlir"
            or receipt["jit_kwargs"] != CALIBRATED_OPTIONS):
        raise ValueError("Math calibration compiler contract differs")
    expected = math_owners(graph)
    if contract["node_owners"] != expected:
        raise ValueError("Math node ownership differs from source calls")
    if set(contract["owners"]) != set(expected.values()):
        raise ValueError("Math dispatcher owner coverage differs")
    functions = {item["id"]: item
                 for item in graph["provenance"]["functions"]}
    for identifier, owner in contract["owners"].items():
        if (owner["function"] != functions[identifier]
                or owner["compiler_options"] != CALIBRATED_OPTIONS
                or owner["native_overloads"] != 0
                or owner["debug"] is not False
                or owner["opt_level"] != 3
                or owner["ptxas_options"] is not None
                or owner["fast_math"] is not None):
            raise ValueError("Math owning-function evidence differs")
    return True


class PolicyMathLowering:
    """Expand calibrated math forms before ordinary fresh allocation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.math_forms = []
        if any(node["kind"] in MATH_OPERATIONS
               for node in self.graph["nodes"]):
            verify_math_contract(self.graph,
                                 self.graph["math_lowering_contract"])

    def typed_operation(self, kind, inputs, dtype, node):
        """Identify only source-typed, explicitly approximate math forms."""
        if kind not in MATH_OPERATIONS:
            return super().typed_operation(kind, inputs, dtype, node)
        count = 2 if kind == "Pow" else 1
        if dtype != "float32" or len(inputs) != count or any(
                self.values[value]["dtype"] != "float32"
                for value in inputs):
            raise ValueError("Math form requires exact FP32 source operands")
        if not self.compiler["fp32_flush_subnormals"]:
            raise ValueError("Math form needs the calibrated MLIR AFN/FTZ path")
        return "POLICY_MATH", dict(source_operation=kind)

    def emit(self, opcode, inputs, output, source_ids, memory=None,
             semantics=None):
        """Emit every native-form operation with actual producer edges."""
        if opcode != "POLICY_MATH":
            return super().emit(opcode, inputs, output, source_ids,
                                memory=memory, semantics=semantics)
        if len(source_ids) != 1 or memory is not None:
            raise ValueError("Math form must bind one pure source operation")
        source_id = source_ids[0]
        kind = semantics["source_operation"]
        emitted = []

        def temporary(label):
            return self.value("float32", "expression",
                              f"math:{source_id}:{label}")

        def literal(value):
            return self.literal(base.constant_payload(value, "float32"))

        def operation(code, operands, result, role, modifiers=None):
            detail = dict(
                source_operation=kind, math_form_role=role,
                approximate=True, refinement=False,
                fp32_flush_subnormals=True,
                native_form_is_conditional=True,
                calibration=self.graph["math_lowering_contract"][
                    "native_form_evidence"],
            )
            if modifiers is not None:
                detail["operand_modifiers"] = modifiers
            identifier = super(PolicyMathLowering, self).emit(
                code, operands, result, source_ids, semantics=detail,
            )
            emitted.append(identifier)
            return result

        if kind == "Exp":
            scaled = operation("FMUL", [inputs[0], literal(LOG2_E)],
                               temporary("log2_scale"), "log2_scale")
            operation("MUFU.EX2", [scaled], output, "base2_exponential")
        elif kind == "Log2":
            operation("MUFU.LG2", inputs, output, "base2_logarithm")
        elif kind == "Log":
            logarithm = operation("MUFU.LG2", inputs,
                                  temporary("log2"), "base2_logarithm")
            operation("FMUL", [logarithm, literal(LN_2)], output,
                      "natural_log_scale")
        elif kind == "Pow":
            logarithm = operation("MUFU.LG2", [inputs[0]],
                                  temporary("log2"), "base2_logarithm")
            exponent = self.values[inputs[1]].get("constant", {})
            exponent = exponent.get("value")
            if exponent == 2:
                scaled = operation(
                    "FADD", [logarithm, logarithm], temporary("scale"),
                    "constant_exponent_two",
                )
            elif exponent == -1:
                scaled = operation(
                    "FADD", [logarithm, literal(np.float32(0))],
                    temporary("scale"), "constant_exponent_minus_one",
                    ["negate", "negate"],
                )
            else:
                scaled = operation(
                    "FMUL", [logarithm, inputs[1]], temporary("scale"),
                    "exponent_scale",
                )
            operation("MUFU.EX2", [scaled], output, "base2_exponential")
        else:
            raise ValueError("Unknown retained math form")
        self.source_nodes[source_id] = emitted
        self.math_forms.append(dict(source_node=source_id, operation=kind,
                                    typed_nodes=emitted))
        return emitted[-1]

    def build(self):
        """Bind the exact form implementation and its native calibration."""
        result = super().build()
        if not self.math_forms:
            return result
        result["math_forms"] = self.math_forms
        result["math_lowering"] = dict(
            source_path=str(Path(__file__).resolve()),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            contract=self.graph.get("math_lowering_contract"),
            constant_operands="Retain the selected materialized-GPR literal "
            "alternative; native kernels can encode these as immediates.",
            domain="Approximate FP32 forms, with IEEE zero/infinity paths "
            "retained by the graph replay contract. NaN paths reject.",
        )
        return result
