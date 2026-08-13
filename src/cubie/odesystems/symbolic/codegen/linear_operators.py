"""Emit CUDA factory code for linear operators and cached JVP helpers.

Published Functions
-------------------
:func:`generate_operator_apply_code`
    Emit a factory that applies ``(I - gamma * h * J) * v`` using
    analytic JVP expressions.

:func:`generate_cached_operator_apply_code`
    Variant that reads precomputed auxiliary values from a cache buffer.

:func:`generate_prepare_jac_code`
    Emit a factory that populates the auxiliary cache buffer used by
    ``generate_cached_operator_apply_code``.

:func:`generate_cached_jvp_code`
    Emit a factory that evaluates the JVP from cached auxiliaries.

:func:`generate_n_stage_linear_operator_code`
    Emit a flattened multi-stage linear operator for FIRK methods.

See Also
--------
:mod:`cubie.odesystems.symbolic.codegen.jacobian`
    Produces the JVP expressions consumed by this module.
:mod:`cubie.odesystems.symbolic.codegen.preconditioners`
    Companion preconditioner code generators.
:mod:`cubie.odesystems.symbolic.codegen._stage_utils`
    Shared FIRK stage metadata helpers.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.adapter import SystemIR, system_ir
from cubie.odesystems.symbolic.engine.assignments import (
    cse_and_stack,
    prune_unused,
    topological_sort,
)
from cubie.odesystems.symbolic.engine.printer import (
    print_cuda_multiple,
)
from cubie.odesystems.symbolic.codegen.jacobian import (
    generate_analytical_jvp,
)
from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations
from cubie.odesystems.symbolic.parsing.parser import (
    IndexedBases,
    ParsedEquations,
)
from cubie.odesystems.symbolic.sym_utils import (
    render_constant_assignments,
)
from cubie.time_logger import default_timelogger

from ._matrix_utils import mass_matrix_ir
from ._stage_utils import build_stage_metadata, prepare_stage_data
from .nonlinear_residuals import build_stage_substitutions

# Register timing events for codegen functions
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
default_timelogger.register_event("codegen_generate_operator_apply_code",
                                   "codegen",
                                   "Codegen time for generate_operator_apply_code")
default_timelogger.register_event(
    "codegen_generate_cached_operator_apply_code", "codegen",
    "Codegen time for generate_cached_operator_apply_code")
default_timelogger.register_event("codegen_generate_prepare_jac_code",
                                   "codegen",
                                   "Codegen time for generate_prepare_jac_code")
default_timelogger.register_event("codegen_generate_cached_jvp_code",
                                   "codegen",
                                   "Codegen time for generate_cached_jvp_code")
default_timelogger.register_event(
    "codegen_generate_n_stage_linear_operator_code", "codegen",
    "Codegen time for generate_n_stage_linear_operator_code")
default_timelogger.register_event(
    "codegen_generate_operator_apply_at_state_code", "codegen",
    "Codegen time for generate_operator_apply_at_state_code")
default_timelogger.register_event(
    "codegen_generate_apply_mass_code", "codegen",
    "Codegen time for generate_apply_mass_code")

CACHED_OPERATOR_APPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, lineinfo=None):\n"
    '    """Auto-generated cached linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * a_ij * h * (J @ v)\n"
    "    using cached auxiliary intermediates.\n"
    "    Returns device function:\n"
    "      operator_apply(\n"
    "          state, parameters, drivers, cached_aux, base_state, t, h, a_ij, v, out\n"
    "      )\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def operator_apply(\n"
    "        state, parameters, drivers, cached_aux, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, v, out\n"
    "    ):\n"
    "{body}\n"
    "    return operator_apply\n"
)


OPERATOR_APPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, lineinfo=None):\n"
    '    """Auto-generated linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * a_ij * h * (J @ v)\n"
    "    Returns device function:\n"
    "      operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out)\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def operator_apply(\n"
    "        state, parameters, drivers, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, v, out,\n"
    "    ):\n"
    "{body}\n"
    "    return operator_apply\n"
)


PREPARE_JAC_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED JACOBIAN PREPARATION FACTORY\n"
    "def {func_name}(constants, precision, lineinfo=None):\n"
    '    """Auto-generated Jacobian auxiliary preparation.\n'
    "    Populates cached_aux with intermediate Jacobian values.\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision,\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def prepare_jac(state, parameters, drivers, t, cached_aux):\n"
    "{body}\n"
    "    return prepare_jac\n"
    "# Store aux_count for retrieval when loading from file cache\n"
    "{func_name}.aux_count = {aux_count}\n"
)


CACHED_JVP_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED JVP FACTORY\n"
    "def {func_name}(constants, precision, lineinfo=None):\n"
    '    """Auto-generated cached Jacobian-vector product.\n'
    "    Computes out = J @ v using cached auxiliaries.\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision,\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def calculate_cached_jvp(\n"
    "        state, parameters, drivers, cached_aux, t, v, out\n"
    "    ):\n"
    "{body}\n"
    "    return calculate_cached_jvp\n"
)


def _partition_cached_assignments(
    equations: JVPEquations,
) -> Tuple[
    List[Tuple[ir.Expr, ir.Expr]],
    List[Tuple[ir.Expr, ir.Expr]],
    List[Tuple[ir.Expr, ir.Expr]],
]:
    """Partition assignments into cached, runtime, and preparation subsets.

    Notes
    -----
    This helper is intended for cached (Rosenbrock-style) code generation.
    It consults the cache planner via :meth:`JVPEquations.cached_partition`.
    Non-cached (DIRK/Newton-Krylov) code paths must not call this function.
    """

    return equations.cached_partition()


def _inline_aux_assignments(
    equations: JVPEquations,
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Return auxiliary expressions in order for inline (non-cached) code
    generation.

    Returns
    -------
    list
        All auxiliary assignments, in order.
    """

    return [
        (lhs, equations.non_jvp_exprs[lhs]) for lhs in equations.non_jvp_order
    ]


def _state_increment_subs(sysir: SystemIR) -> Dict[ir.Expr, ir.Expr]:
    """Map state symbols to ``base_state + a_ij * state`` eval points.

    Used by the non-cached (Newton--Krylov) paths, where the ``state``
    argument is the stage increment.
    """
    a_ij_sym = ir.sym("_cubie_codegen_a_ij")
    subs = {}
    for i, state_sym in enumerate(sysir.state_symbols):
        subs[state_sym] = ir.add(
            ir.arr("base_state", i),
            ir.mul(a_ij_sym, ir.arr("state", i)),
        )
    return subs


def _build_operator_body(
    runtime_assigns: List[Tuple[ir.Expr, ir.Expr]],
    jvp_terms: Dict[int, ir.Expr],
    sysir: SystemIR,
    M: List[List[ir.Expr]],
    use_cached_aux: bool = False,
    cached_assigns: Optional[List[Tuple[ir.Expr, ir.Expr]]] = None,
    prepare_assigns: Optional[List[Tuple[ir.Expr, ir.Expr]]] = None,
    state_is_increment: bool = True,
) -> str:
    """Build the CUDA body computing ``β·M·v − γ·h·J·v``."""

    n_out = len(sysir.dxdt_symbols)
    n_in = len(sysir.state_symbols)
    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    a_ij_sym = ir.sym("_cubie_codegen_a_ij")
    h_sym = ir.sym("_cubie_codegen_h")

    # Newton increments evaluate at base_state + a_ij * state;
    # cached and at-state bodies evaluate at state directly.
    if use_cached_aux or not state_is_increment:
        state_subs = {}
    else:
        state_subs = _state_increment_subs(sysir)
    memo: dict = {}

    mass_assigns: List[Tuple[ir.Expr, ir.Expr]] = []
    out_updates: List[Tuple[ir.Expr, ir.Expr]] = []
    for i in range(n_out):
        mv_terms = []
        for j in range(n_in):
            entry = M[i][j]
            if ir.is_zero(entry):
                continue
            m_sym = ir.sym(f"_cubie_codegen_m_{i}_{j}")
            mass_assigns.append((m_sym, entry))
            mv_terms.append(ir.mul(m_sym, ir.arr("v", j)))
        mv = ir.add(*mv_terms) if mv_terms else ir.ZERO
        jvp_term = jvp_terms.get(i, ir.ZERO)
        if state_subs:
            jvp_term = ir.xreplace(jvp_term, state_subs, memo)
        rhs = ir.sub(
            ir.mul(beta_sym, mv),
            ir.mul(gamma_sym, a_ij_sym, h_sym, jvp_term),
        )
        out_updates.append((ir.arr("out", i), rhs))

    if use_cached_aux:
        aux_assignments = [
            (lhs, ir.arr("cached_aux", idx))
            for idx, (lhs, _) in enumerate(cached_assigns or [])
        ] + runtime_assigns
    else:
        combined = (
            list(prepare_assigns or [])
            + list(cached_assigns or [])
            + runtime_assigns
        )
        seen = set()
        aux_assignments = []
        for lhs, rhs in combined:
            if lhs in seen:
                continue
            seen.add(lhs)
            if state_subs:
                rhs = ir.xreplace(rhs, state_subs, memo)
            aux_assignments.append((lhs, rhs))

    exprs = mass_assigns + aux_assignments + out_updates
    exprs = prune_unused(exprs, output_name="out")

    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


def _build_cached_jvp_body(
    cached_assigns: List[Tuple[ir.Expr, ir.Expr]],
    runtime_assigns: List[Tuple[ir.Expr, ir.Expr]],
    jvp_terms: Dict[int, ir.Expr],
    sysir: SystemIR,
) -> str:
    """Build the CUDA body computing ``J·v`` with cached auxiliaries."""

    n_out = len(sysir.dxdt_symbols)
    aux_assignments = [
        (lhs, ir.arr("cached_aux", idx))
        for idx, (lhs, _) in enumerate(cached_assigns)
    ] + runtime_assigns

    out_updates = [
        (ir.arr("out", i), jvp_terms.get(i, ir.ZERO))
        for i in range(n_out)
    ]

    exprs = aux_assignments + out_updates
    exprs = prune_unused(exprs, output_name="out")

    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


def _build_prepare_body(
    cached_assigns: List[Tuple[ir.Expr, ir.Expr]],
    prepare_assigns: List[Tuple[ir.Expr, ir.Expr]],
    sysir: SystemIR,
) -> str:
    """Build the CUDA body populating the cached Jacobian auxiliaries."""

    cached_slots = {
        lhs: idx for idx, (lhs, _) in enumerate(cached_assigns)
    }
    exprs: List[Tuple[ir.Expr, ir.Expr]] = []
    for lhs, rhs in prepare_assigns:
        exprs.append((lhs, rhs))
        idx = cached_slots.get(lhs)
        if idx is not None:
            exprs.append((ir.arr("cached_aux", idx), lhs))
    exprs = prune_unused(exprs, output_name="cached_aux")

    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)


def generate_operator_apply_code_from_jvp(
    equations: JVPEquations,
    sysir: SystemIR,
    index_map: IndexedBases,
    M: List[List[ir.Expr]],
    func_name: str = "operator_apply_factory",
) -> str:
    """Emit the operator apply factory from precomputed JVP expressions."""

    runtime_aux = _inline_aux_assignments(equations)
    body = _build_operator_body(
        runtime_assigns=runtime_aux,
        jvp_terms=equations.jvp_terms,
        sysir=sysir,
        M=M,
        use_cached_aux=False,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    return OPERATOR_APPLY_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )


def generate_operator_apply_at_state_code_from_jvp(
    equations: JVPEquations,
    sysir: SystemIR,
    index_map: IndexedBases,
    M: List[List[ir.Expr]],
    func_name: str = "operator_apply_at_state_factory",
) -> str:
    """Emit an operator apply factory evaluating J at ``state``.

    Same signature as the non-cached operator; ``base_state`` is
    unused and ``a_ij`` scales the matrix only.
    """

    runtime_aux = _inline_aux_assignments(equations)
    body = _build_operator_body(
        runtime_assigns=runtime_aux,
        jvp_terms=equations.jvp_terms,
        sysir=sysir,
        M=M,
        use_cached_aux=False,
        state_is_increment=False,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    return OPERATOR_APPLY_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )


def generate_cached_operator_apply_code_from_jvp(
    equations: JVPEquations,
    sysir: SystemIR,
    index_map: IndexedBases,
    M: List[List[ir.Expr]],
    func_name: str = "linear_operator_cached",
) -> str:
    """Emit the cached linear operator factory from JVP expressions."""

    cached_aux, runtime_aux, _ = _partition_cached_assignments(equations)
    body = _build_operator_body(
        cached_assigns=cached_aux,
        runtime_assigns=runtime_aux,
        jvp_terms=equations.jvp_terms,
        sysir=sysir,
        M=M,
        use_cached_aux=True,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    return CACHED_OPERATOR_APPLY_TEMPLATE.format(
        func_name=func_name,
        body=body,
        const_lines=const_block,
    )


def generate_prepare_jac_code_from_jvp(
    equations: JVPEquations,
    sysir: SystemIR,
    index_map: IndexedBases,
    func_name: str = "prepare_jac",
) -> Tuple[str, int]:
    """Emit the auxiliary preparation factory from JVP expressions."""

    cached_aux, _, prepare_assigns = _partition_cached_assignments(equations)
    body = _build_prepare_body(cached_aux, prepare_assigns, sysir)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    aux_count = len(cached_aux)
    code = PREPARE_JAC_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block,
        aux_count=aux_count
    )
    return code, aux_count


def generate_cached_jvp_code_from_jvp(
    equations: JVPEquations,
    sysir: SystemIR,
    index_map: IndexedBases,
    func_name: str = "calculate_cached_jvp",
) -> str:
    """Emit the cached JVP factory from precomputed JVP expressions."""

    cached_aux, runtime_aux, _ = _partition_cached_assignments(equations)
    body = _build_cached_jvp_body(
        cached_assigns=cached_aux,
        runtime_assigns=runtime_aux,
        jvp_terms=equations.jvp_terms,
        sysir=sysir,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    code = CACHED_JVP_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    return code


def _resolve_jvp(
    equations: ParsedEquations,
    index_map: IndexedBases,
    cse: bool,
    jvp_equations: Optional[JVPEquations],
    operation_ordering: str = "kahn",
) -> JVPEquations:
    """Return the JVP equations, generating them when not supplied."""
    if jvp_equations is not None:
        return jvp_equations
    return generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
        operation_ordering=operation_ordering,
    )


def generate_operator_apply_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "operator_apply_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> str:
    """Generate the linear operator factory from system equations."""
    default_timelogger.start_event("codegen_generate_operator_apply_code")

    sysir = system_ir(equations, index_map)
    mass = mass_matrix_ir(M, len(sysir.state_symbols))
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    result = generate_operator_apply_code_from_jvp(
        equations=jvp_equations,
        sysir=sysir,
        index_map=index_map,
        M=mass,
        func_name=func_name,
    )
    default_timelogger.stop_event("codegen_generate_operator_apply_code")
    return result


def generate_operator_apply_at_state_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "operator_apply_at_state_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> str:
    """Generate the linear operator factory linearized at ``state``."""
    default_timelogger.start_event(
        "codegen_generate_operator_apply_at_state_code"
    )

    sysir = system_ir(equations, index_map)
    mass = mass_matrix_ir(M, len(sysir.state_symbols))
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    result = generate_operator_apply_at_state_code_from_jvp(
        equations=jvp_equations,
        sysir=sysir,
        index_map=index_map,
        M=mass,
        func_name=func_name,
    )
    default_timelogger.stop_event(
        "codegen_generate_operator_apply_at_state_code"
    )
    return result


def generate_cached_operator_apply_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "linear_operator_cached",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> str:
    """Generate the cached linear operator factory."""
    default_timelogger.start_event("codegen_generate_cached_operator_apply_code")

    sysir = system_ir(equations, index_map)
    mass = mass_matrix_ir(M, len(sysir.state_symbols))
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    result = generate_cached_operator_apply_code_from_jvp(
        equations=jvp_equations,
        sysir=sysir,
        index_map=index_map,
        M=mass,
        func_name=func_name,
    )
    default_timelogger.stop_event("codegen_generate_cached_operator_apply_code")
    return result


def generate_prepare_jac_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "prepare_jac",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> Tuple[str, int]:
    """Generate the cached auxiliary preparation factory."""
    default_timelogger.start_event("codegen_generate_prepare_jac_code")

    sysir = system_ir(equations, index_map)
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    result = generate_prepare_jac_code_from_jvp(
        equations=jvp_equations,
        sysir=sysir,
        index_map=index_map,
        func_name=func_name,
    )
    default_timelogger.stop_event("codegen_generate_prepare_jac_code")
    return result


def generate_cached_jvp_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "calculate_cached_jvp",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> str:
    """Generate the cached Jacobian-vector product factory."""
    default_timelogger.start_event("codegen_generate_cached_jvp_code")

    sysir = system_ir(equations, index_map)
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    result = generate_cached_jvp_code_from_jvp(
        equations=jvp_equations,
        sysir=sysir,
        index_map=index_map,
        func_name=func_name,
    )
    default_timelogger.stop_event("codegen_generate_cached_jvp_code")
    return result


def build_stage_jvp_assignments(
    sysir: SystemIR,
    jvp_equations: JVPEquations,
    stage_idx: int,
    coeff_symbols: List[List[ir.Sym]],
    node_symbols: List[ir.Sym],
    stage_coefficients: List[List[ir.Expr]],
    direction_name: str = "v",
) -> Tuple[List[Tuple[ir.Expr, ir.Expr]], Dict[int, ir.Sym]]:
    """Instantiate the JVP auxiliary chain and terms for one stage.

    The direction vector ``v`` is replaced by the stage coupling
    ``sum_j a[stage][j] * <direction_name>[j*n + i]`` and every
    auxiliary is renamed with a stage suffix so stages coexist in one
    body.

    Returns
    -------
    tuple
        Stage-suffixed assignments (auxiliaries then JVP terms) and a
        mapping from output index to the stage JVP symbol.
    """
    state_count = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)

    subs_map = build_stage_substitutions(
        sysir,
        stage_idx,
        coeff_symbols,
        node_symbols,
        stage_coefficients,
        state_vector_name="state",
    )

    # v[i] -> sum over contributing stages of a_ij * v_flat[j*n + i]
    for comp_idx in range(state_count):
        combo_terms = []
        for contrib_idx in range(stage_count):
            if ir.is_zero(
                stage_coefficients[stage_idx][contrib_idx]
            ):
                continue
            coeff_sym = coeff_symbols[stage_idx][contrib_idx]
            combo_terms.append(
                ir.mul(
                    coeff_sym,
                    ir.arr(
                        direction_name,
                        contrib_idx * state_count + comp_idx,
                    ),
                )
            )
        combo = ir.add(*combo_terms) if combo_terms else ir.ZERO
        subs_map[ir.arr("v", comp_idx)] = combo

    # Stage-rename every JVP auxiliary; dependencies are topologically
    # ordered, so one simultaneous rename map is exact.
    for lhs in jvp_equations.non_jvp_order:
        subs_map[lhs] = ir.sym(
            f"_cubie_codegen_s{stage_idx}_{lhs.name}"
        )

    memo: dict = {}
    assignments: List[Tuple[ir.Expr, ir.Expr]] = []
    for lhs in jvp_equations.non_jvp_order:
        rhs = jvp_equations.non_jvp_exprs[lhs]
        assignments.append(
            (subs_map[lhs], ir.xreplace(rhs, subs_map, memo))
        )

    stage_jvp_symbols: Dict[int, ir.Sym] = {}
    for idx, term in jvp_equations.jvp_terms.items():
        stage_symbol = ir.sym(
            f"_cubie_codegen_jvp_{stage_idx}_{idx}"
        )
        stage_jvp_symbols[idx] = stage_symbol
        assignments.append(
            (stage_symbol, ir.xreplace(term, subs_map, memo))
        )
    return assignments, stage_jvp_symbols


def _build_n_stage_operator_lines(
    sysir: SystemIR,
    M: List[List[ir.Expr]],
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    jvp_equations: JVPEquations,
    cse: bool = True,
    operation_ordering: str = "kahn",
) -> str:
    """Construct CUDA statements for the FIRK n-stage linear operator."""

    metadata_exprs, coeff_symbols, node_symbols = build_stage_metadata(
        stage_coefficients, stage_nodes
    )
    state_count = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)

    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    h_sym = ir.sym("_cubie_codegen_h")

    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = list(metadata_exprs)

    for stage_idx in range(stage_count):
        stage_assignments, stage_jvp_symbols = (
            build_stage_jvp_assignments(
                sysir,
                jvp_equations,
                stage_idx,
                coeff_symbols,
                node_symbols,
                stage_coefficients,
            )
        )
        eval_exprs.extend(stage_assignments)

        stage_offset = stage_idx * state_count
        for comp_idx in range(state_count):
            mv_terms = []
            for col_idx in range(state_count):
                entry = M[comp_idx][col_idx]
                if ir.is_zero(entry):
                    continue
                mv_terms.append(
                    ir.mul(
                        entry,
                        ir.arr("v", stage_offset + col_idx),
                    )
                )
            mv = ir.add(*mv_terms) if mv_terms else ir.ZERO
            jvp_value = stage_jvp_symbols.get(comp_idx, ir.ZERO)
            update_expr = ir.sub(
                ir.mul(beta_sym, mv),
                ir.mul(gamma_sym, h_sym, jvp_value),
            )
            eval_exprs.append(
                (ir.arr("out", stage_offset + comp_idx), update_expr)
            )

    if cse:
        eval_exprs = cse_and_stack(
            eval_exprs,
            operation_ordering=operation_ordering,
        )
    else:
        eval_exprs = topological_sort(
            eval_exprs,
            operation_ordering=operation_ordering,
        )

    eval_exprs = prune_unused(eval_exprs, output_name="out")

    lines = print_cuda_multiple(
        eval_exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


def generate_n_stage_linear_operator_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: Sequence[Sequence[Union[float, object]]],
    stage_nodes: Sequence[Union[float, object]],
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "n_stage_linear_operator",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = "kahn",
) -> str:
    """Generate a flattened n-stage FIRK linear operator factory."""
    default_timelogger.start_event("codegen_generate_n_stage_linear_operator_code")

    coeff_matrix, node_values, stage_count = prepare_stage_data(
        stage_coefficients, stage_nodes
    )
    sysir = system_ir(equations, index_map)
    mass = mass_matrix_ir(M, len(sysir.state_symbols))
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    body = _build_n_stage_operator_lines(
        sysir=sysir,
        M=mass,
        stage_coefficients=coeff_matrix,
        stage_nodes=node_values,
        jvp_equations=jvp_equations,
        cse=cse,
        operation_ordering=operation_ordering,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    result = N_STAGE_OPERATOR_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        metadata_lines="",
        body=body,
        stage_count=stage_count,
    )
    default_timelogger.stop_event("codegen_generate_n_stage_linear_operator_code")
    return result


N_STAGE_OPERATOR_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED N-STAGE LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, lineinfo=None):\n"
    '    """Auto-generated FIRK linear operator for flattened stages.\n'
    "    Handles {stage_count} stages with ``s * n`` unknowns.\n"
    '    """\n'
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "{const_lines}"
    "{metadata_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def operator_apply(\n"
    "        state, parameters, drivers, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, v, out,\n"
    "    ):\n"
    "{body}\n"
    "    return operator_apply\n"
)


APPLY_MASS_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED MASS-MATRIX APPLY FACTORY\n"
    "def {func_name}(constants, precision, lineinfo=None):\n"
    '    """Auto-generated mass-matrix product.\n'
    "    Computes out = M @ v. `out` must not alias `v`.\n"
    '    """\n'
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def apply_mass(v, out):\n"
    "{body}\n"
    "    return apply_mass\n"
)


def _matrix_product_body(matrix, sysir, n: int) -> str:
    """Render ``out = matrix @ v`` as an indented CUDA body."""
    exprs: List[Tuple[ir.Expr, ir.Expr]] = []
    for i in range(n):
        terms = []
        for j in range(n):
            entry = matrix[i][j]
            if ir.is_zero(entry):
                continue
            if ir.is_one(entry):
                terms.append(ir.arr("v", j))
            else:
                terms.append(ir.mul(entry, ir.arr("v", j)))
        row = ir.add(*terms) if terms else ir.ZERO
        exprs.append((ir.arr("out", i), row))

    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


def generate_apply_mass_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "apply_mass_factory",
) -> str:
    """Generate a factory applying the mass matrix to a vector."""
    default_timelogger.start_event("codegen_generate_apply_mass_code")

    sysir = system_ir(equations, index_map)
    n = len(sysir.state_symbols)
    mass = mass_matrix_ir(M, n)

    body = _matrix_product_body(mass, sysir, n)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    result = APPLY_MASS_TEMPLATE.format(
        func_name=func_name, body=body, const_lines=const_block
    )
    default_timelogger.stop_event("codegen_generate_apply_mass_code")
    return result


__all__ = [
    "generate_operator_apply_code",
    "generate_operator_apply_at_state_code",
    "generate_cached_operator_apply_code",
    "generate_prepare_jac_code",
    "generate_cached_jvp_code",
    "generate_n_stage_linear_operator_code",
    "generate_apply_mass_code",
    "build_stage_jvp_assignments",
]
