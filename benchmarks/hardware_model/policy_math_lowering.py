"""Retain source-bound approximate FP32 transcendental instruction forms."""

import hashlib
from pathlib import Path

import numpy as np

from cubie.cuda_backend import CUDA_BACKEND
from benchmarks.hardware_model import native_plan as base


MATH_OPERATIONS = {"Exp", "Log", "Log2", "Pow"}
LOG2_E = np.float32(1.4426950216293334961)
LN_2 = np.float32(0.69314718246459960938)
NATIVE_FORM_EVIDENCE = {
    "path": "C:/local_working_projects/cubie-notes/"
    "hardware_unroll_placement/math_forms_e1/receipt.json",
    "sha256": "c2f7d1a1bbae528d8e765f0dc72abdb30028cec4e7d19494d7a89a2814e77a50",
    "scope": "Eleven isolated same-backend FP32 forms, no kernel launches "
    "or timing measurements. Independent review pending.",
}


def math_lowering_contract(solver):
    """Capture actual backend and step JIT flags for the form alternative."""
    step = solver.kernel.single_integrator._algo_step
    return dict(backend=CUDA_BACKEND, fastmath=sorted(
        step.jit_kwargs["fastmath"]), native_form_evidence=NATIVE_FORM_EVIDENCE)


class PolicyMathLowering:
    """Expand calibrated math forms before ordinary fresh allocation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.math_forms = []

    def typed_operation(self, kind, inputs, dtype, node):
        """Identify only source-typed, explicitly approximate math forms."""
        if kind not in MATH_OPERATIONS:
            return super().typed_operation(kind, inputs, dtype, node)
        count = 2 if kind == "Pow" else 1
        if dtype != "float32" or len(inputs) != count or any(
                self.values[value]["dtype"] != "float32"
                for value in inputs):
            raise ValueError("Math form requires exact FP32 source operands")
        contract = self.graph["math_lowering_contract"]
        if (contract["backend"] != "mlir"
                or not {"afn", "ftz"}.issubset(contract["fastmath"])
                or not self.compiler["fp32_flush_subnormals"]):
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
            domain="Approximate positive-base real powers and finite FP32 "
            "log/exp paths. Exceptional libdevice paths are not inferred.",
        )
        return result
