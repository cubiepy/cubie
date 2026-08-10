"""Emit CUDA factory code for Neumann-series and Jacobi preconditioners.

Published Functions
-------------------
:func:`generate_neumann_preconditioner_code`
    Emit a factory computing a truncated Neumann series approximation
    to ``(I - gamma * h * J)^{-1} * v``.

:func:`generate_neumann_preconditioner_cached_code`
    Variant that reads precomputed auxiliary values from a cache buffer.

:func:`generate_n_stage_neumann_preconditioner_code`
    Emit a flattened multi-stage preconditioner for FIRK methods.

:func:`generate_n_stage_jacobi_preconditioner_code`
    Emit a diagonal Jacobi preconditioner for FIRK methods.

See Also
--------
:mod:`cubie.odesystems.symbolic.codegen.linear_operators`
    Companion linear operator code generators.
:mod:`cubie.odesystems.symbolic.codegen.jacobian`
    Produces the JVP expressions consumed by this module.
:mod:`cubie.odesystems.symbolic.codegen._stage_utils`
    Shared FIRK stage metadata helpers.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    generate_jacobian,
)
from cubie.odesystems.symbolic.codegen.linear_operators import (
    _resolve_jvp,
    _state_increment_subs,
    build_stage_jvp_assignments,
)
from cubie.odesystems.symbolic.codegen.nonlinear_residuals import (
    build_stage_substitutions,
)
from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations
from cubie.odesystems.symbolic.parsing.parser import (
    IndexedBases,
    ParsedEquations,
)
from cubie.odesystems.symbolic.codegen._matrix_utils import (
    mass_matrix_ir,
)
from cubie.odesystems.symbolic.codegen._stage_utils import (
    build_stage_metadata,
    prepare_stage_data,
)
from cubie.odesystems.symbolic.sym_utils import (
    render_constant_assignments,
)
from cubie.time_logger import default_timelogger

# Register timing events for codegen functions
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
default_timelogger.register_event(
    "codegen_generate_neumann_preconditioner_code", "codegen",
    "Codegen time for generate_neumann_preconditioner_code")
default_timelogger.register_event(
    "codegen_generate_neumann_preconditioner_cached_code", "codegen",
    "Codegen time for generate_neumann_preconditioner_cached_code")
default_timelogger.register_event(
    "codegen_generate_neumann_preconditioner_at_state_code", "codegen",
    "Codegen time for generate_neumann_preconditioner_at_state_code")
default_timelogger.register_event(
    "codegen_generate_jacobi_preconditioner_at_state_code", "codegen",
    "Codegen time for generate_jacobi_preconditioner_at_state_code")
default_timelogger.register_event(
    "codegen_generate_n_stage_neumann_preconditioner_code", "codegen",
    "Codegen time for generate_n_stage_neumann_preconditioner_code")
default_timelogger.register_event(
    "codegen_generate_chained_preconditioner_code", "codegen",
    "Codegen time for generate_chained_preconditioner_code")

NEUMANN_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated Neumann preconditioner.\n'
    "    Approximates (beta*I - gamma*a_ij*h*J)^[-1] via a truncated\n"
    "    Neumann series. Returns device function:\n"
    "      preconditioner(state, parameters, drivers, base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch)\n"
    "    where `jvp` is a caller-provided scratch buffer for J*v and\n"
    "    `chain_scratch` is consumed only by chained compositions.\n"
    '    """\n'
    "    _cubie_codegen_n = int32({n_out})\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_order = int32(order)\n"
    "    _cubie_codegen_beta_inv = precision(\n"
    "        1.0 / _cubie_codegen_beta\n"
    "    )\n"
    "    _cubie_codegen_h_eff_factor = precision(\n"
    "        _cubie_codegen_gamma * _cubie_codegen_beta_inv\n"
    "    )\n"
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
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def preconditioner(\n"
    "        state, parameters, drivers, base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch\n"
    "    ):\n"
    "        # Horner form: S[m] = v + T S[m-1], T = ((gamma*a_ij)/beta) * h * J\n"
    "        # Accumulator lives in `out`. Uses caller-provided `jvp` for JVP.\n"
    "        for i in range(_cubie_codegen_n):\n"
    "            out[i] = v[i]\n"
    "        _cubie_codegen_h_eff = (\n"
    "            _cubie_codegen_h\n"
    "            * _cubie_codegen_h_eff_factor\n"
    "            * _cubie_codegen_a_ij\n"
    "        )\n"
    "        for _ in range(_cubie_codegen_order):\n"
    "{jv_body}\n"
    "            for i in range(_cubie_codegen_n):\n"
    "                out[i] = v[i] + _cubie_codegen_h_eff * jvp[i]\n"
    "        for i in range(_cubie_codegen_n):\n"
    "            out[i] = _cubie_codegen_beta_inv * out[i]\n"
    "    return preconditioner\n"
)


NEUMANN_CACHED_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Cached Neumann preconditioner using stored auxiliaries.\n'
    "    Approximates (beta*I - gamma*a_ij*h*J)^[-1] via a truncated\n"
    "    Neumann series with cached auxiliaries. Returns device function:\n"
    "      preconditioner(\n"
    "          state, parameters, drivers, cached_aux, base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch\n"
    "      )\n"
    '    """\n'
    "    _cubie_codegen_n = int32({n_out})\n"
    "    _cubie_codegen_order = int32(order)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_beta_inv = precision(\n"
    "        1.0 / _cubie_codegen_beta\n"
    "    )\n"
    "    _cubie_codegen_h_eff_factor = precision(\n"
    "        _cubie_codegen_gamma * _cubie_codegen_beta_inv\n"
    "    )\n"
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
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def preconditioner(\n"
    "        state, parameters, drivers, cached_aux, base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch\n"
    "    ):\n"
    "        for i in range(_cubie_codegen_n):\n"
    "            out[i] = v[i]\n"
    "        _cubie_codegen_h_eff = (\n"
    "            _cubie_codegen_h\n"
    "            * _cubie_codegen_h_eff_factor\n"
    "            * _cubie_codegen_a_ij\n"
    "        )\n"
    "        for _ in range(_cubie_codegen_order):\n"
    "{jv_body}\n"
    "            for i in range(_cubie_codegen_n):\n"
    "                out[i] = v[i] + _cubie_codegen_h_eff * jvp[i]\n"
    "        for i in range(_cubie_codegen_n):\n"
    "            out[i] = _cubie_codegen_beta_inv * out[i]\n"
    "    return preconditioner\n"
)


N_STAGE_NEUMANN_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED N-STAGE NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated FIRK Neumann preconditioner.\n'
    "    Handles {stage_count} stages with ``s * n`` unknowns.\n"
    "    Approximates the inverse of ``beta * I - gamma * h * (A ⊗ J)`` using\n"
    "    a truncated Neumann series applied to flattened stages.\n"
    "    Returns device function:\n"
    "      preconditioner(state, parameters, drivers, base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch)\n"
    '    """\n'
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "{const_lines}"
    "{metadata_lines}"
    "    _cubie_codegen_total_n = int32({total_states})\n"
    "    _cubie_codegen_order = int32(order)\n"
    "    _cubie_codegen_beta_inv = precision(\n"
    "        1.0 / _cubie_codegen_beta\n"
    "    )\n"
    "    _cubie_codegen_h_eff_factor = precision(\n"
    "        _cubie_codegen_gamma * _cubie_codegen_beta_inv\n"
    "    )\n"
    "    _cubie_codegen_stage_width = int32({state_count})\n"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision,\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1]),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def preconditioner(state, parameters, drivers, base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch):\n"
    "        for i in range(_cubie_codegen_total_n):\n"
    "            out[i] = v[i]\n"
    "        _cubie_codegen_h_eff = (\n"
    "            _cubie_codegen_h * _cubie_codegen_h_eff_factor\n"
    "        )\n"
    "        for _ in range(_cubie_codegen_order):\n"
    "{jv_body}\n"
    "            for i in range(_cubie_codegen_total_n):\n"
    "                out[i] = v[i] + _cubie_codegen_h_eff * jvp[i]\n"
    "        for i in range(_cubie_codegen_total_n):\n"
    "            out[i] = _cubie_codegen_beta_inv * out[i]\n"
    "    return preconditioner\n"
)


def _accumulator_reads(
    assignments: List[Tuple[ir.Expr, ir.Expr]],
    n_states: int,
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Rewrite direction reads ``v[i]`` to the ``out`` accumulator.

    The Neumann loop applies J to the running accumulator stored in
    ``out``; the JVP expressions are built against ``v``, so their
    reads are redirected here (structurally, not by string
    replacement).
    """
    v_to_out = {
        ir.arr("v", i): ir.arr("out", i) for i in range(n_states)
    }
    memo: dict = {}
    return [
        (lhs, ir.xreplace(rhs, v_to_out, memo))
        for lhs, rhs in assignments
    ]


def _build_neumann_body_with_state_subs(
    jvp_equations: JVPEquations,
    sysir: SystemIR,
) -> str:
    """Build non-cached Neumann JVP body with inline state evaluation.

    For Newton-Krylov usage: the ``state`` argument is the stage
    increment, so the Jacobian evaluates at
    ``base_state + a_ij * state``.
    """
    state_subs = _state_increment_subs(sysir)
    memo: dict = {}
    substituted = [
        (lhs, ir.xreplace(rhs, state_subs, memo))
        for lhs, rhs in jvp_equations.ordered_assignments
    ]
    substituted = _accumulator_reads(
        substituted, len(sysir.state_symbols)
    )
    substituted = prune_unused(substituted, output_name="jvp")

    lines = print_cuda_multiple(
        substituted,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    if not lines:
        lines = ["pass"]
    return "\n".join("            " + ln for ln in lines)


def _build_neumann_body_at_state(
    jvp_equations: JVPEquations,
    sysir: SystemIR,
) -> str:
    """Build the Neumann JVP body evaluating J at ``state``."""
    substituted = _accumulator_reads(
        list(jvp_equations.ordered_assignments),
        len(sysir.state_symbols),
    )
    substituted = prune_unused(substituted, output_name="jvp")

    lines = print_cuda_multiple(
        substituted,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    if not lines:
        lines = ["pass"]
    return "\n".join("            " + ln for ln in lines)


def _build_cached_neumann_body(
    equations: JVPEquations,
    sysir: SystemIR,
) -> str:
    """Build the cached Neumann-series Jacobian-vector body.

    For Rosenbrock usage: the ``state`` argument is the actual state;
    auxiliaries come from the ``cached_aux`` buffer.
    """
    cached_aux, runtime_aux, _ = equations.cached_partition()
    jvp_terms = equations.jvp_terms

    aux_assignments = [
        (lhs, ir.arr("cached_aux", idx))
        for idx, (lhs, _) in enumerate(cached_aux)
    ] + runtime_aux

    n_out = len(sysir.dxdt_symbols)
    exprs = list(aux_assignments)
    for i in range(n_out):
        exprs.append((ir.arr("jvp", i), jvp_terms.get(i, ir.ZERO)))

    exprs = _accumulator_reads(exprs, len(sysir.state_symbols))
    exprs = prune_unused(exprs, output_name="jvp")
    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("            " + ln for ln in lines)


def _build_n_stage_neumann_lines(
    sysir: SystemIR,
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    jvp_equations: JVPEquations,
    cse: bool = True,
) -> str:
    """Construct CUDA statements computing J·v for flattened FIRK stages."""

    metadata_exprs, coeff_symbols, node_symbols = build_stage_metadata(
        stage_coefficients, stage_nodes
    )
    state_count = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)

    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = list(metadata_exprs)

    for stage_idx in range(stage_count):
        # The Neumann loop applies (A ⊗ J) to the accumulator in
        # ``out``, so the stage direction combos read ``out``.
        stage_assignments, stage_jvp_symbols = (
            build_stage_jvp_assignments(
                sysir,
                jvp_equations,
                stage_idx,
                coeff_symbols,
                node_symbols,
                stage_coefficients,
                direction_name="out",
            )
        )
        eval_exprs.extend(stage_assignments)

        stage_offset = stage_idx * state_count
        for comp_idx in range(state_count):
            jvp_value = stage_jvp_symbols.get(comp_idx, ir.ZERO)
            eval_exprs.append(
                (ir.arr("jvp", stage_offset + comp_idx), jvp_value)
            )

    if cse:
        eval_exprs = cse_and_stack(eval_exprs)
    else:
        eval_exprs = topological_sort(eval_exprs)

    eval_exprs = prune_unused(eval_exprs, output_name="jvp")

    lines = print_cuda_multiple(
        eval_exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("            " + ln for ln in lines)


def generate_n_stage_neumann_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: Sequence[Sequence[Union[float, object]]],
    stage_nodes: Sequence[Union[float, object]],
    func_name: str = "n_stage_neumann_preconditioner",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate a flattened n-stage FIRK Neumann preconditioner factory."""
    default_timelogger.start_event("codegen_generate_n_stage_neumann_preconditioner_code")

    coeff_matrix, node_values, stage_count = prepare_stage_data(
        stage_coefficients, stage_nodes
    )
    sysir = system_ir(equations, index_map)
    jvp_equations = _resolve_jvp(equations, index_map, cse, jvp_equations)
    body = _build_n_stage_neumann_lines(
        sysir=sysir,
        stage_coefficients=coeff_matrix,
        stage_nodes=node_values,
        jvp_equations=jvp_equations,
        cse=cse,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    total_states = stage_count * len(sysir.state_symbols)
    state_count = len(sysir.state_symbols)
    result = N_STAGE_NEUMANN_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        metadata_lines="",
        jv_body=body,
        stage_count=stage_count,
        total_states=total_states,
        state_count=state_count,
    )
    default_timelogger.stop_event("codegen_generate_n_stage_neumann_preconditioner_code")
    return result


def generate_neumann_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate the Neumann preconditioner factory.

    For Newton-Krylov usage: applies inline state evaluation.
    """
    default_timelogger.start_event("codegen_generate_neumann_preconditioner_code")

    sysir = system_ir(equations, index_map)
    n_out = len(sysir.dxdt_symbols)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    jvp_equations = _resolve_jvp(equations, index_map, cse, jvp_equations)
    jv_body = _build_neumann_body_with_state_subs(jvp_equations, sysir)
    result = NEUMANN_TEMPLATE.format(
        func_name=func_name,
        n_out=n_out,
        jv_body=jv_body,
        const_lines=const_block,
    )
    default_timelogger.stop_event("codegen_generate_neumann_preconditioner_code")
    return result


def generate_neumann_preconditioner_at_state_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_at_state",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate a Neumann preconditioner evaluating J at ``state``.

    ``a_ij`` scales the matrix only.
    """
    default_timelogger.start_event(
        "codegen_generate_neumann_preconditioner_at_state_code"
    )

    sysir = system_ir(equations, index_map)
    n_out = len(sysir.dxdt_symbols)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    jvp_equations = _resolve_jvp(equations, index_map, cse, jvp_equations)
    jv_body = _build_neumann_body_at_state(jvp_equations, sysir)
    result = NEUMANN_TEMPLATE.format(
        func_name=func_name,
        n_out=n_out,
        jv_body=jv_body,
        const_lines=const_block,
    )
    default_timelogger.stop_event(
        "codegen_generate_neumann_preconditioner_at_state_code"
    )
    return result


def generate_neumann_preconditioner_cached_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_cached",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
) -> str:
    """Generate the cached Neumann preconditioner factory.

    For Rosenbrock usage: state param is actual state,
    no inline substitution needed.
    """
    default_timelogger.start_event("codegen_generate_neumann_preconditioner_cached_code")

    sysir = system_ir(equations, index_map)
    n_out = len(sysir.dxdt_symbols)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    jvp_equations = _resolve_jvp(equations, index_map, cse, jvp_equations)
    jv_body = _build_cached_neumann_body(jvp_equations, sysir)
    result = NEUMANN_CACHED_TEMPLATE.format(
        func_name=func_name,
        n_out=n_out,
        jv_body=jv_body,
        const_lines=const_block,
    )
    default_timelogger.stop_event("codegen_generate_neumann_preconditioner_cached_code")
    return result


JACOBI_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED DIAGONAL JACOBI PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated diagonal Jacobi preconditioner.\n'
    "    Computes diagonal of ``beta * M - gamma * a_ij * h * J`` and\n"
    "    applies pointwise inversion: ``out[i] = v[i] / d[i]``.\n"
    "    Returns device function:\n"
    "      preconditioner(state, parameters, drivers, base_state,"
    " t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch)\n"
    '    """\n'
    "    _cubie_codegen_n = int32({n_out})\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def preconditioner("
    "state, parameters, drivers, base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch):\n"
    "{diag_body}\n"
    "    return preconditioner\n"
)


JACOBI_CACHED_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CACHED DIAGONAL JACOBI PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Cached diagonal Jacobi preconditioner using stored auxiliaries.\n'
    "    Computes diagonal of ``beta * M - gamma * a_ij * h * J`` and\n"
    "    applies pointwise inversion: ``out[i] = v[i] / d[i]``.\n"
    "    Returns device function:\n"
    "      preconditioner(\n"
    "          state, parameters, drivers, cached_aux, base_state,"
    " t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch\n"
    "      )\n"
    '    """\n'
    "    _cubie_codegen_n = int32({n_out})\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def preconditioner("
    "state, parameters, drivers, cached_aux, base_state,"
    " t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch):\n"
    "{diag_body}\n"
    "    return preconditioner\n"
)


DIAG_DIVISION_FLOOR = 1e-16
"""Magnitude floor applied to Jacobi diagonals before division.

A diagonal entry ``beta - gamma*h*a_ij*J_ii`` crosses zero when
``h*a_ij*J_ii`` approaches ``beta/gamma``; dividing by it would emit
inf/NaN, and NaN passes untouched through the linear solvers' selp
clamps. Flooring the magnitude keeps the correction large but finite.
"""


def _guarded_diag_division(diag_sym, comp_idx, stage_idx=None):
    """Return a magnitude-floored alias assignment for a diagonal.

    Parameters
    ----------
    diag_sym
        Symbol holding the raw diagonal value.
    comp_idx
        State component index used to name the alias.
    stage_idx
        Optional FIRK stage index prefixed to the alias name.

    Returns
    -------
    tuple of (ir.Sym, ir.Expr)
        ``safe_diag_*`` symbol and the guarded expression. When
        ``|diag| < DIAG_DIVISION_FLOOR`` the floor value is used
        (sign dropped; near zero the sign carries no information).
    """
    if stage_idx is not None:
        suffix = f"{stage_idx}_{comp_idx}"
    else:
        suffix = f"{comp_idx}"
    safe_sym = ir.sym(f"_cubie_codegen_safe_diag_{suffix}")
    floor = ir.num(DIAG_DIVISION_FLOOR)
    guarded = ir.piecewise(
        (diag_sym, ir.rel(">=", ir.call("Abs", diag_sym), floor)),
        (floor, ir.TRUE),
    )
    return (safe_sym, guarded)


def _mass_diag_term(M, comp_idx, beta_sym):
    """Return the ``beta*M_ii`` term of a Jacobi diagonal entry.

    Off-diagonal mass entries are ignored: the Jacobi preconditioner
    approximates only the diagonal of ``beta*M - gamma*h*a_ij*J``.
    """
    entry = M[comp_idx][comp_idx]
    if ir.is_one(entry):
        return beta_sym
    return ir.mul(beta_sym, entry)


def _build_jacobi_body_with_state_subs(
    equations: ParsedEquations,
    index_map: IndexedBases,
    cse: bool = True,
    M: Optional[Union[Sequence, object]] = None,
    state_is_increment: bool = True,
) -> str:
    """Build single-system Jacobi body with inline state evaluation.

    ``state_is_increment`` selects the J_ii point:
    ``base_state + a_ij * state`` (Newton) or ``state`` directly.
    The diagonal is ``beta*M_ii - gamma*h*a_ij*J_ii``; off-diagonal
    mass entries are ignored.
    """
    sysir = system_ir(equations, index_map)
    state_count = len(sysir.state_symbols)

    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
    )

    h_sym = ir.sym("_cubie_codegen_h")
    a_ij_sym = ir.sym("_cubie_codegen_a_ij")
    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")

    # dx/observable outputs become locals; states evaluate at
    # base_state + a_ij * state when state is a Newton increment,
    # at state directly otherwise.
    subs_map = {}
    for idx, dx_sym in enumerate(sysir.dxdt_symbols):
        subs_map[dx_sym] = ir.sym(f"_cubie_codegen_dx_{idx}")
    for idx, obs_sym in enumerate(sysir.observable_symbols):
        subs_map[obs_sym] = ir.sym(f"_cubie_codegen_aux_{idx + 1}")
    if state_is_increment:
        subs_map.update(_state_increment_subs(sysir))

    memo: dict = {}
    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = [
        (
            ir.xreplace(lhs, subs_map, memo),
            ir.xreplace(rhs, subs_map, memo),
        )
        for lhs, rhs in sysir.equations
    ]

    mass = mass_matrix_ir(M, state_count)
    for comp_idx in range(state_count):
        j_ii = ir.xreplace(jac[comp_idx][comp_idx], subs_map, memo)
        diag_sym = ir.sym(f"_cubie_codegen_diag_{comp_idx}")
        diag_val = ir.sub(
            _mass_diag_term(mass, comp_idx, beta_sym),
            ir.mul(gamma_sym, h_sym, a_ij_sym, j_ii),
        )
        eval_exprs.append((diag_sym, diag_val))
        eval_exprs.append(_guarded_diag_division(diag_sym, comp_idx))
        eval_exprs.append(
            (
                ir.arr("out", comp_idx),
                ir.div(
                    ir.arr("v", comp_idx),
                    ir.sym(f"_cubie_codegen_safe_diag_{comp_idx}"),
                ),
            )
        )

    if cse:
        eval_exprs = cse_and_stack(eval_exprs)
    else:
        eval_exprs = topological_sort(eval_exprs)
    eval_exprs = prune_unused(eval_exprs, output_name="out")

    lines = print_cuda_multiple(
        eval_exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


def _build_cached_jacobi_body(
    equations: ParsedEquations,
    index_map: IndexedBases,
    cse: bool = True,
    M: Optional[Union[Sequence, object]] = None,
) -> str:
    """Build cached Jacobi body for Rosenbrock usage.

    ``state`` is the actual state vector — no inline substitution
    needed. Auxiliaries come from the ``cached_aux`` buffer. The
    diagonal is ``beta*M_ii - gamma*h*a_ij*J_ii``; off-diagonal mass
    entries are ignored.
    """
    sysir = system_ir(equations, index_map)
    state_count = len(sysir.state_symbols)

    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
    )

    jvp_equations = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    cached_aux, runtime_aux, _ = jvp_equations.cached_partition()

    h_sym = ir.sym("_cubie_codegen_h")
    a_ij_sym = ir.sym("_cubie_codegen_a_ij")
    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")

    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = []
    eval_exprs.extend(
        (lhs, ir.arr("cached_aux", idx))
        for idx, (lhs, _) in enumerate(cached_aux)
    )
    eval_exprs.extend(runtime_aux)

    # The Jacobian diagonal references auxiliaries by name; cached
    # ones read from the buffer, runtime ones from their expressions.
    aux_subs: Dict[ir.Expr, ir.Expr] = {
        lhs: ir.arr("cached_aux", idx)
        for idx, (lhs, _) in enumerate(cached_aux)
    }
    for lhs, rhs in runtime_aux:
        aux_subs[lhs] = rhs

    # The full Jacobian references observables by their original
    # names, while the JVP pipeline renamed them to aux_<n>; map the
    # originals to the same numbered locals so both agree.
    obs_renames = {
        obs_sym: ir.sym(f"_cubie_codegen_aux_{idx + 1}")
        for idx, obs_sym in enumerate(sysir.observable_symbols)
    }

    memo: dict = {}
    mass = mass_matrix_ir(M, state_count)
    for comp_idx in range(state_count):
        j_ii = ir.xreplace(
            jac[comp_idx][comp_idx], obs_renames, memo
        )
        j_ii = ir.xreplace(j_ii, aux_subs)
        diag_sym = ir.sym(f"_cubie_codegen_diag_{comp_idx}")
        diag_val = ir.sub(
            _mass_diag_term(mass, comp_idx, beta_sym),
            ir.mul(gamma_sym, h_sym, a_ij_sym, j_ii),
        )
        eval_exprs.append((diag_sym, diag_val))
        eval_exprs.append(_guarded_diag_division(diag_sym, comp_idx))
        eval_exprs.append(
            (
                ir.arr("out", comp_idx),
                ir.div(
                    ir.arr("v", comp_idx),
                    ir.sym(f"_cubie_codegen_safe_diag_{comp_idx}"),
                ),
            )
        )

    if cse:
        eval_exprs = cse_and_stack(eval_exprs)
    else:
        eval_exprs = topological_sort(eval_exprs)
    eval_exprs = prune_unused(eval_exprs, output_name="out")

    lines = print_cuda_multiple(
        eval_exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


default_timelogger.register_event(
    "codegen_generate_jacobi_preconditioner_code", "codegen",
    "Codegen time for generate_jacobi_preconditioner_code")
default_timelogger.register_event(
    "codegen_generate_jacobi_preconditioner_cached_code", "codegen",
    "Codegen time for generate_jacobi_preconditioner_cached_code")


def generate_jacobi_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "jacobi_preconditioner_factory",
    cse: bool = True,
    M: Optional[Union[Sequence, object]] = None,
) -> str:
    """Generate a diagonal Jacobi preconditioner for single-system solvers.

    Computes ``diag(beta*M - gamma*h*a_ij*J)`` and applies
    pointwise inversion. For Newton-Krylov usage with inline state
    evaluation. Off-diagonal mass entries are ignored.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination.
    M
        Mass matrix; identity when omitted.

    Returns
    -------
    str
        Generated Python/CUDA factory function code.
    """
    default_timelogger.start_event(
        "codegen_generate_jacobi_preconditioner_code"
    )
    n_out = len(index_map.dxdt.ref_map)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    diag_body = _build_jacobi_body_with_state_subs(
        equations, index_map, cse, M=M
    )
    result = JACOBI_TEMPLATE.format(
        func_name=func_name,
        n_out=n_out,
        const_lines=const_block,
        diag_body=diag_body,
    )
    default_timelogger.stop_event(
        "codegen_generate_jacobi_preconditioner_code"
    )
    return result


def generate_jacobi_preconditioner_at_state_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "jacobi_preconditioner_at_state",
    cse: bool = True,
    M: Optional[Union[Sequence, object]] = None,
) -> str:
    """Generate a diagonal Jacobi preconditioner evaluating J at
    ``state``.

    ``a_ij`` scales the matrix only; off-diagonal mass entries are
    ignored.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination.
    M
        Mass matrix; identity when omitted.

    Returns
    -------
    str
        Generated Python/CUDA factory function code.
    """
    default_timelogger.start_event(
        "codegen_generate_jacobi_preconditioner_at_state_code"
    )
    n_out = len(index_map.dxdt.ref_map)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    diag_body = _build_jacobi_body_with_state_subs(
        equations, index_map, cse, M=M, state_is_increment=False
    )
    result = JACOBI_TEMPLATE.format(
        func_name=func_name,
        n_out=n_out,
        const_lines=const_block,
        diag_body=diag_body,
    )
    default_timelogger.stop_event(
        "codegen_generate_jacobi_preconditioner_at_state_code"
    )
    return result


def generate_jacobi_preconditioner_cached_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "jacobi_preconditioner_cached",
    cse: bool = True,
    M: Optional[Union[Sequence, object]] = None,
) -> str:
    """Generate a cached diagonal Jacobi preconditioner.

    For Rosenbrock usage: state is the actual state, auxiliaries come
    from a cached buffer. Off-diagonal mass entries are ignored.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination.
    M
        Mass matrix; identity when omitted.

    Returns
    -------
    str
        Generated Python/CUDA factory function code.
    """
    default_timelogger.start_event(
        "codegen_generate_jacobi_preconditioner_cached_code"
    )
    n_out = len(index_map.dxdt.ref_map)
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    diag_body = _build_cached_jacobi_body(equations, index_map, cse, M=M)
    result = JACOBI_CACHED_TEMPLATE.format(
        func_name=func_name,
        n_out=n_out,
        const_lines=const_block,
        diag_body=diag_body,
    )
    default_timelogger.stop_event(
        "codegen_generate_jacobi_preconditioner_cached_code"
    )
    return result


N_STAGE_JACOBI_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED N-STAGE DIAGONAL JACOBI PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated FIRK diagonal Jacobi preconditioner.\n'
    "    Handles {stage_count} stages with ``s * n`` unknowns.\n"
    "    Computes diagonal of ``beta * M - gamma * h * (A ⊗ J)`` and\n"
    "    applies pointwise inversion: ``out[k] = v[k] / d[k]``.\n"
    "    Returns device function:\n"
    "      preconditioner(state, parameters, drivers, base_state,"
    " t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch)\n"
    '    """\n'
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "{const_lines}"
    "{metadata_lines}"
    "    _cubie_codegen_total_n = int32({total_states})\n"
    "    _cubie_codegen_stage_width = int32({state_count})\n"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def preconditioner("
    "state, parameters, drivers, base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp, scratch, chain_scratch):\n"
    "        # Evaluate diagonal J_ii at each stage evaluation point,\n"
    "        # form d[s*n+i] = beta - gamma*h*a_ss*J_ii,\n"
    "        # apply out[k] = v[k] / d[k].\n"
    "{diag_body}\n"
    "    return preconditioner\n"
)


def _build_n_stage_jacobi_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    cse: bool = True,
    M: Optional[Union[Sequence, object]] = None,
) -> str:
    """Build diagonal Jacobi preconditioner body for n-stage FIRK.

    Extracts J_ii = df_i/dy_i for each state, evaluates at each
    stage point, forms d = beta*M_ii - gamma*h*a_ss*J_ii (off-diagonal
    mass entries ignored), and applies out[k] = v[k] / d[k].
    """
    sysir = system_ir(equations, index_map)
    metadata_exprs, coeff_symbols, node_symbols = build_stage_metadata(
        stage_coefficients, stage_nodes
    )
    state_count = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)

    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
    )

    h_sym = ir.sym("_cubie_codegen_h")
    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")

    mass = mass_matrix_ir(M, state_count)
    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = list(metadata_exprs)

    for stage_idx in range(stage_count):
        subs_map = build_stage_substitutions(
            sysir,
            stage_idx,
            coeff_symbols,
            node_symbols,
            stage_coefficients,
            state_vector_name="state",
        )
        memo: dict = {}
        # Emit the stage-renamed equation list so every intermediate
        # the diagonal needs is defined; pruning drops the rest.
        substituted_eqs = [
            (
                ir.xreplace(lhs, subs_map, memo),
                ir.xreplace(rhs, subs_map, memo),
            )
            for lhs, rhs in sysir.equations
        ]
        eval_exprs.extend(substituted_eqs)

        diag_coeff = coeff_symbols[stage_idx][stage_idx]
        stage_offset = stage_idx * state_count
        for comp_idx in range(state_count):
            j_ii = ir.xreplace(
                jac[comp_idx][comp_idx], subs_map, memo
            )
            diag_sym = ir.sym(
                f"_cubie_codegen_diag_{stage_idx}_{comp_idx}"
            )
            diag_val = ir.sub(
                _mass_diag_term(mass, comp_idx, beta_sym),
                ir.mul(gamma_sym, h_sym, diag_coeff, j_ii),
            )
            eval_exprs.append((diag_sym, diag_val))
            eval_exprs.append(
                _guarded_diag_division(
                    diag_sym, comp_idx, stage_idx=stage_idx
                )
            )
            eval_exprs.append(
                (
                    ir.arr("out", stage_offset + comp_idx),
                    ir.div(
                        ir.arr("v", stage_offset + comp_idx),
                        ir.sym(
                            "_cubie_codegen_safe_diag_"
                            f"{stage_idx}_{comp_idx}"
                        ),
                    ),
                )
            )

    if cse:
        eval_exprs = cse_and_stack(eval_exprs)
    else:
        eval_exprs = topological_sort(eval_exprs)

    eval_exprs = prune_unused(eval_exprs, output_name="out")

    lines = print_cuda_multiple(
        eval_exprs,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


default_timelogger.register_event(
    "codegen_generate_n_stage_jacobi_preconditioner_code",
    "codegen",
    "Codegen time for generate_n_stage_jacobi_preconditioner_code",
)


def generate_n_stage_jacobi_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: Sequence[Sequence[Union[float, object]]],
    stage_nodes: Sequence[Union[float, object]],
    func_name: str = "n_stage_jacobi_preconditioner",
    cse: bool = True,
    M: Optional[Union[Sequence, object]] = None,
) -> str:
    """Generate a diagonal Jacobi preconditioner for n-stage FIRK.

    Computes ``diag(beta*M - gamma*h*(A_diag x J_diag))`` and
    applies pointwise inversion. Much cheaper than Neumann series
    and handles stiff diagonal entries correctly. Off-diagonal mass
    entries are ignored.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    stage_coefficients
        Butcher tableau A matrix.
    stage_nodes
        Butcher tableau c vector.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination.
    M
        Mass matrix; identity when omitted.

    Returns
    -------
    str
        Generated Python/CUDA factory function code.
    """
    default_timelogger.start_event(
        "codegen_generate_n_stage_jacobi_preconditioner_code"
    )

    coeff_matrix, node_values, stage_count = prepare_stage_data(
        stage_coefficients, stage_nodes
    )
    body = _build_n_stage_jacobi_lines(
        equations=equations,
        index_map=index_map,
        stage_coefficients=coeff_matrix,
        stage_nodes=node_values,
        cse=cse,
        M=M,
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )
    total_states = stage_count * len(index_map.states.index_map)
    state_count = len(index_map.states.index_map)

    metadata_lines = ""
    result = N_STAGE_JACOBI_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        metadata_lines=metadata_lines,
        diag_body=body,
        stage_count=stage_count,
        total_states=total_states,
        state_count=state_count,
    )
    default_timelogger.stop_event(
        "codegen_generate_n_stage_jacobi_preconditioner_code"
    )
    return result


CHAINED_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CHAINED PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated preconditioner composition.\n'
    "    Applies the composed stages in request order (stage 0 first).\n"
    "    Intermediate results ping-pong between `scratch` and\n"
    "    `chain_scratch`; each stage's own scratch slot is a buffer it\n"
    "    neither reads nor writes as data. Returns device function:\n"
    "      preconditioner({device_args})\n"
    '    """\n'
    "{stage_sources}"
    "{stage_bindings}"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def preconditioner(\n"
    "        {device_args}\n"
    "    ):\n"
    "{stage_calls}"
    "    return preconditioner\n"
)


def _indent_factory_source(source: str) -> str:
    """Indent a generated factory source for nesting one level in."""
    lines = []
    for line in source.splitlines():
        lines.append("    " + line if line.strip() else line)
    return "\n".join(lines) + "\n"


def _chain_buffer_plan(stage_count: int) -> list:
    """Return per-stage (source, target, work) buffer assignments.

    Values flow ``v`` -> alternating ``scratch``/``chain_scratch``
    intermediates -> ``out``. Each stage's work slot is a buffer that
    is dead for that stage: ``out`` until the final stage, then
    whichever intermediate buffer is not the final stage's source.
    """
    plan = []
    for index in range(stage_count):
        source = "v" if index == 0 else plan[-1][1]
        if index == stage_count - 1:
            target = "out"
            work = "chain_scratch" if source == "scratch" else "scratch"
        else:
            target = "scratch" if index % 2 == 0 else "chain_scratch"
            work = "out"
        plan.append((source, target, work))
    return plan


def generate_chained_preconditioner_code(
    stage_sources,
    func_name: str = "chained_preconditioner_factory",
    cached: bool = False,
) -> str:
    """Emit a factory composing preconditioners in sequence.

    Parameters
    ----------
    stage_sources
        Generated factory sources for the composed stages in
        application order; stage ``i`` must be emitted with the
        function name ``_cubie_codegen_stage<i>_factory``.
    func_name
        Name of the emitted composed factory.
    cached
        Whether the composed signature carries a ``cached_aux``
        argument (the Rosenbrock-W cached helper family).

    Returns
    -------
    str
        Source code for the composed factory. Every stage factory is
        nested inside it, so one module-level definition serves the
        whole composition and all stages bind the same constants,
        precision, beta, gamma, and order arguments.
    """
    default_timelogger.start_event(
        "codegen_generate_chained_preconditioner_code"
    )
    stage_sources = list(stage_sources)
    common = "state, parameters, drivers, "
    if cached:
        common += "cached_aux, "
    common += "base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij,"
    device_args = (
        common + " v, out, jvp, scratch, chain_scratch"
    )
    bindings = []
    for index in range(len(stage_sources)):
        bindings.append(
            f"    _cubie_codegen_p{index} = "
            f"_cubie_codegen_stage{index}_factory(\n"
            "        constants, precision, beta, gamma, order, "
            "lineinfo\n"
            "    )\n"
        )
    calls = []
    for index, (source, target, work) in enumerate(
        _chain_buffer_plan(len(stage_sources))
    ):
        calls.append(
            f"        _cubie_codegen_p{index}(\n"
            f"            {common}\n"
            f"            {source}, {target}, jvp, {work}, "
            "chain_scratch,\n"
            "        )\n"
        )
    result = CHAINED_TEMPLATE.format(
        func_name=func_name,
        device_args=device_args,
        stage_sources="".join(
            _indent_factory_source(source) for source in stage_sources
        ),
        stage_bindings="".join(bindings),
        stage_calls="".join(calls),
    )
    default_timelogger.stop_event(
        "codegen_generate_chained_preconditioner_code"
    )
    return result


__all__ = [
    "generate_neumann_preconditioner_code",
    "generate_neumann_preconditioner_cached_code",
    "generate_neumann_preconditioner_at_state_code",
    "generate_jacobi_preconditioner_code",
    "generate_jacobi_preconditioner_cached_code",
    "generate_jacobi_preconditioner_at_state_code",
    "generate_n_stage_neumann_preconditioner_code",
    "generate_n_stage_jacobi_preconditioner_code",
    "generate_chained_preconditioner_code",
]
