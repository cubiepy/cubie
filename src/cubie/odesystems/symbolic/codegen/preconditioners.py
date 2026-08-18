"""Emit CUDA factory code for Neumann-series and Jacobi preconditioners.

Published Functions
-------------------
:func:`generate_neumann_preconditioner_code`
    Emit a truncated Neumann series approximating
    ``(beta*I - gamma*a_ij*h*J)^{-1} * v`` for one variant.

:func:`generate_jacobi_preconditioner_code`
    Emit pointwise inversion by ``diag(beta*M - gamma*a_ij*h*J)`` for
    one variant.

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

from cubie.odesystems.solver_helpers import HelperVariant
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
from cubie.odesystems.symbolic.parsing import (
    IndexedBases,
    ParsedEquations,
)
from cubie.odesystems.symbolic.codegen._matrix_utils import (
    mass_diagonal_flags,
)
from cubie._env import operation_ordering_default
from cubie.odesystems.symbolic.codegen._stage_utils import (
    build_stage_metadata,
    prepare_stage_data,
)
from cubie.time_logger import default_timelogger

# Register timing events for codegen functions
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
for _variant in HelperVariant:
    default_timelogger.register_event(
        f"codegen_neumann_preconditioner_{_variant.value}",
        "codegen",
        f"Codegen time for the {_variant.value} Neumann preconditioner",
    )
    default_timelogger.register_event(
        f"codegen_jacobi_preconditioner_{_variant.value}",
        "codegen",
        f"Codegen time for the {_variant.value} Jacobi preconditioner",
    )

NEUMANN_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated Neumann preconditioner.\n'
    "    Approximates (beta*I - gamma*a_ij*h*J)^[-1] via a truncated\n"
    "    Neumann series. Returns device function:\n"
    "      preconditioner(\n"
    "          state, parameters, drivers, {cached_arg}base_state, t, h, a_ij, v, out, jvp\n"
    "      )\n"
    "    where `jvp` is a caller-provided buffer for J*v.\n"
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
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def preconditioner(\n"
    "        state, parameters, drivers, {cached_arg}base_state, t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp\n"
    "    ):\n"
    "        # Horner form: S[m] = v + T S[m-1], T = ((gamma*a_ij)/beta) * h * J\n"
    "        # Accumulator lives in `out`. Uses caller-provided `jvp` for JVP.\n"
    "        for i in range(_cubie_codegen_n):\n"
    "            out[i] = v[i]\n"
    "        _cubie_codegen_h_eff = (\n"
    "            _cubie_codegen_h * _cubie_codegen_h_eff_factor{a_ij_factor}\n"
    "        )\n"
    "        for _ in range(_cubie_codegen_order):\n"
    "{jv_body}\n"
    "            for i in range(_cubie_codegen_n):\n"
    "                out[i] = v[i] + _cubie_codegen_h_eff * jvp[i]\n"
    "        for i in range(_cubie_codegen_n):\n"
    "            out[i] = _cubie_codegen_beta_inv * out[i]\n"
    "    return preconditioner\n"
)


JACOBI_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED DIAGONAL JACOBI PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, order=1, lineinfo=None):\n"
    '    """Auto-generated diagonal Jacobi preconditioner.\n'
    "    Computes diagonal of ``beta * M - gamma * a_ij * h * J`` and\n"
    "    applies pointwise inversion: ``out[i] = v[i] / d[i]``.\n"
    "    Returns device function:\n"
    "      preconditioner(\n"
    "          state, parameters, drivers, {cached_arg}base_state, t, h, a_ij, v, out, jvp\n"
    "      )\n"
    '    """\n'
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "    _cubie_codegen_beta = precision(beta)\n"
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def preconditioner("
    "state, parameters, drivers, {cached_arg}base_state,"
    " t, _cubie_codegen_h, _cubie_codegen_a_ij, v, out, jvp):\n"
    "{diag_body}\n"
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


def _build_neumann_jv_body(
    jvp_equations: JVPEquations,
    sysir: SystemIR,
    use_cached_aux: bool = False,
    state_is_increment: bool = True,
) -> str:
    """Build the Neumann-series Jacobian-vector body for one variant.

    ``state_is_increment`` selects the J evaluation point;
    ``use_cached_aux`` reads auxiliaries from ``cached_aux``.
    """
    if use_cached_aux:
        cached_aux, runtime_aux, _ = jvp_equations.cached_partition()
        exprs: List[Tuple[ir.Expr, ir.Expr]] = [
            (lhs, ir.arr("cached_aux", idx))
            for idx, (lhs, _) in enumerate(cached_aux)
        ] + runtime_aux
        n_out = len(sysir.dxdt_symbols)
        for i in range(n_out):
            exprs.append(
                (ir.arr("jvp", i), jvp_equations.jvp_terms.get(i, ir.ZERO))
            )
    else:
        exprs = list(jvp_equations.ordered_assignments)
        if state_is_increment:
            state_subs = _state_increment_subs(sysir)
            memo: dict = {}
            exprs = [
                (lhs, ir.xreplace(rhs, state_subs, memo))
                for lhs, rhs in exprs
            ]

    exprs = _accumulator_reads(exprs, len(sysir.state_symbols))
    exprs = prune_unused(exprs, output_name="jvp")

    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        function_aliases=sysir.function_aliases,
    )
    if not lines:
        lines = ["pass"]
    return "\n".join("            " + ln for ln in lines)


def _build_n_stage_neumann_lines(
    sysir: SystemIR,
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    jvp_equations: JVPEquations,
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
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
        eval_exprs = cse_and_stack(
            eval_exprs,
            operation_ordering=operation_ordering,
        )
    else:
        eval_exprs = topological_sort(
            eval_exprs,
            operation_ordering=operation_ordering,
        )

    eval_exprs = prune_unused(eval_exprs, output_name="jvp")

    lines = print_cuda_multiple(
        eval_exprs,
        symbol_map=sysir.arrayrefs,
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("            " + ln for ln in lines)


def generate_neumann_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    variant: HelperVariant = HelperVariant.PLAIN,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, object]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, object]]] = None,
    func_name: str = "neumann_preconditioner_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Generate the Neumann preconditioner factory for one variant.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    variant
        Helper variant selecting the evaluation-point and auxiliary
        conventions.
    stage_coefficients
        Butcher tableau A matrix; ``STACKED_STAGES`` only.
    stage_nodes
        Butcher tableau c vector; ``STACKED_STAGES`` only.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination.
    jvp_equations
        Precomputed JVP expressions; generated when omitted.

    Returns
    -------
    str
        Generated Python/CUDA factory function code.
    """
    event = f"codegen_neumann_preconditioner_{variant.value}"
    default_timelogger.start_event(event)

    sysir = system_ir(equations, index_map)
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    if variant.stacked_stages:
        coeff_matrix, node_values, stage_count = prepare_stage_data(
            stage_coefficients, stage_nodes
        )
        jv_body = _build_n_stage_neumann_lines(
            sysir=sysir,
            stage_coefficients=coeff_matrix,
            stage_nodes=node_values,
            jvp_equations=jvp_equations,
            cse=cse,
            operation_ordering=operation_ordering,
        )
        n_out = stage_count * len(sysir.state_symbols)
        # The tableau is baked into the stage coupling, so a_ij does
        # not scale the flattened series.
        a_ij_factor = ""
    else:
        jv_body = _build_neumann_jv_body(
            jvp_equations,
            sysir,
            use_cached_aux=variant.cached,
            state_is_increment=variant is HelperVariant.PLAIN,
        )
        n_out = len(sysir.dxdt_symbols)
        a_ij_factor = " * _cubie_codegen_a_ij"
    result = NEUMANN_TEMPLATE.format(
        func_name=func_name,
        cached_arg="cached_aux, " if variant.cached else "",
        n_out=n_out,
        a_ij_factor=a_ij_factor,
        jv_body=jv_body,
    )
    default_timelogger.stop_event(event)
    return result


DIAG_DIVISION_FLOOR = 1e-16
"""Magnitude floor applied to Jacobi diagonals before division."""


def _diag_row_exprs(
    j_ii: ir.Expr,
    has_mass: bool,
    scale_syms: Tuple[ir.Expr, ...],
    out_idx: int,
    suffix: str,
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Return the diagonal, guard, and division exprs for one row.

    Parameters
    ----------
    j_ii
        Jacobian diagonal expression at the row's evaluation point.
    has_mass
        Whether the row's 0/1 mass diagonal entry is one. An identity
        row contributes ``beta``; a zero (algebraic residual) row
        contributes nothing, leaving the pure Jacobian diagonal.
    scale_syms
        Symbols whose product scales ``j_ii`` in the diagonal.
    out_idx
        Flattened output index of the row.
    suffix
        Name suffix for the diagonal and guard locals.

    Returns
    -------
    list of (ir.Expr, ir.Expr)
        Diagonal assignment, magnitude-floored guard, and the
        ``out[i] = v[i] / d[i]`` division.
    """
    beta_sym = ir.sym("_cubie_codegen_beta")
    diag_sym = ir.sym(f"_cubie_codegen_diag_{suffix}")
    safe_sym = ir.sym(f"_cubie_codegen_safe_diag_{suffix}")
    mass_term = beta_sym if has_mass else ir.ZERO
    diag_val = ir.sub(mass_term, ir.mul(*scale_syms, j_ii))
    floor = ir.num(DIAG_DIVISION_FLOOR)
    guarded = ir.piecewise(
        (diag_sym, ir.rel(">=", ir.call("Abs", diag_sym), floor)),
        (floor, ir.TRUE),
    )
    return [
        (diag_sym, diag_val),
        (safe_sym, guarded),
        (
            ir.arr("out", out_idx),
            ir.div(ir.arr("v", out_idx), safe_sym),
        ),
    ]


def _finalise_diag_body(
    eval_exprs: List[Tuple[ir.Expr, ir.Expr]],
    sysir: SystemIR,
    cse: bool,
    operation_ordering: str,
) -> str:
    """Sort, prune, and print the assembled Jacobi diagonal body."""
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
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


def _build_jacobi_body(
    equations: ParsedEquations,
    index_map: IndexedBases,
    cse: bool = True,
    M: Optional[Union[Sequence, object]] = None,
    use_cached_aux: bool = False,
    state_is_increment: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Build the single-system Jacobi diagonal body for one variant.

    ``state_is_increment`` selects the J_ii point:
    ``base_state + a_ij * state`` (Newton) or ``state`` directly.
    With ``use_cached_aux`` the auxiliaries the diagonal references
    come from the ``cached_aux`` buffer. The diagonal is
    ``beta*M_ii - gamma*h*a_ij*J_ii`` with ``M_ii`` the system's 0/1
    mass diagonal.
    """
    sysir = system_ir(equations, index_map)
    state_count = len(sysir.state_symbols)

    jac = generate_jacobian(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        operation_ordering=operation_ordering,
    )

    h_sym = ir.sym("_cubie_codegen_h")
    a_ij_sym = ir.sym("_cubie_codegen_a_ij")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    scale_syms = (gamma_sym, h_sym, a_ij_sym)

    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = []
    memo: dict = {}
    if use_cached_aux:
        jvp_equations = generate_analytical_jvp(
            equations,
            input_order=index_map.states.index_map,
            output_order=index_map.dxdt.index_map,
            observables=index_map.observable_symbols,
            cse=cse,
            operation_ordering=operation_ordering,
        )
        cached_aux, runtime_aux, _ = jvp_equations.cached_partition()

        eval_exprs.extend(
            (lhs, ir.arr("cached_aux", idx))
            for idx, (lhs, _) in enumerate(cached_aux)
        )
        eval_exprs.extend(runtime_aux)

        # Bind every auxiliary the diagonal can reference.
        cached_slots = {
            lhs: idx for idx, (lhs, _) in enumerate(cached_aux)
        }
        runtime_symbols = {lhs for lhs, _ in runtime_aux}
        subs_memo: dict = {}
        aux_subs: Dict[ir.Expr, ir.Expr] = {}
        for lhs in jvp_equations.non_jvp_order:
            slot = cached_slots.get(lhs)
            if slot is not None:
                aux_subs[lhs] = ir.arr("cached_aux", slot)
            elif lhs in runtime_symbols:
                aux_subs[lhs] = jvp_equations.non_jvp_exprs[lhs]
            else:
                aux_subs[lhs] = ir.xreplace(
                    jvp_equations.non_jvp_exprs[lhs], aux_subs, subs_memo
                )

        # The full Jacobian references observables by their original
        # names, while the JVP pipeline renamed them to aux_<n>; map
        # the originals to the same numbered locals so both agree.
        subs_map = {
            obs_sym: ir.sym(f"_cubie_codegen_aux_{idx + 1}")
            for idx, obs_sym in enumerate(sysir.observable_symbols)
        }

        def _row_j_ii(comp_idx):
            j_ii = ir.xreplace(jac[comp_idx][comp_idx], subs_map, memo)
            return ir.xreplace(j_ii, aux_subs)

    else:
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

        eval_exprs.extend(
            (
                ir.xreplace(lhs, subs_map, memo),
                ir.xreplace(rhs, subs_map, memo),
            )
            for lhs, rhs in sysir.equations
        )

        def _row_j_ii(comp_idx):
            return ir.xreplace(jac[comp_idx][comp_idx], subs_map, memo)

    mass_diag = mass_diagonal_flags(M, state_count)
    for comp_idx in range(state_count):
        eval_exprs.extend(
            _diag_row_exprs(
                j_ii=_row_j_ii(comp_idx),
                has_mass=mass_diag[comp_idx],
                scale_syms=scale_syms,
                out_idx=comp_idx,
                suffix=f"{comp_idx}",
            )
        )

    return _finalise_diag_body(
        eval_exprs, sysir, cse, operation_ordering
    )


def _build_n_stage_jacobi_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    cse: bool = True,
    M: Optional[Union[Sequence, object]] = None,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Build diagonal Jacobi preconditioner body for n-stage FIRK.

    Extracts J_ii = df_i/dy_i for each state, evaluates at each
    stage point, forms d = beta*M_ii - gamma*h*a_ss*J_ii with M_ii
    the system's 0/1 mass diagonal, and applies out[k] = v[k] / d[k].
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
        operation_ordering=operation_ordering,
    )

    h_sym = ir.sym("_cubie_codegen_h")
    gamma_sym = ir.sym("_cubie_codegen_gamma")

    mass_diag = mass_diagonal_flags(M, state_count)
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
            eval_exprs.extend(
                _diag_row_exprs(
                    j_ii=j_ii,
                    has_mass=mass_diag[comp_idx],
                    scale_syms=(gamma_sym, h_sym, diag_coeff),
                    out_idx=stage_offset + comp_idx,
                    suffix=f"{stage_idx}_{comp_idx}",
                )
            )

    return _finalise_diag_body(
        eval_exprs, sysir, cse, operation_ordering
    )


def generate_jacobi_preconditioner_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    variant: HelperVariant = HelperVariant.PLAIN,
    M: Optional[Union[Sequence, object]] = None,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, object]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, object]]] = None,
    func_name: str = "jacobi_preconditioner_factory",
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Generate the diagonal Jacobi preconditioner for one variant.

    Computes ``diag(beta*M - gamma*h*a_ij*J)`` and applies pointwise
    inversion.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    variant
        Helper variant selecting the evaluation-point and auxiliary
        conventions.
    M
        0/1 diagonal mass matrix; identity when omitted.
    stage_coefficients
        Butcher tableau A matrix; ``STACKED_STAGES`` only.
    stage_nodes
        Butcher tableau c vector; ``STACKED_STAGES`` only.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination.

    Returns
    -------
    str
        Generated Python/CUDA factory function code.
    """
    event = f"codegen_jacobi_preconditioner_{variant.value}"
    default_timelogger.start_event(event)

    if variant.stacked_stages:
        coeff_matrix, node_values, _ = prepare_stage_data(
            stage_coefficients, stage_nodes
        )
        diag_body = _build_n_stage_jacobi_lines(
            equations=equations,
            index_map=index_map,
            stage_coefficients=coeff_matrix,
            stage_nodes=node_values,
            cse=cse,
            M=M,
            operation_ordering=operation_ordering,
        )
    else:
        diag_body = _build_jacobi_body(
            equations,
            index_map,
            cse,
            M=M,
            use_cached_aux=variant.cached,
            state_is_increment=variant is HelperVariant.PLAIN,
            operation_ordering=operation_ordering,
        )
    result = JACOBI_TEMPLATE.format(
        func_name=func_name,
        cached_arg="cached_aux, " if variant.cached else "",
        diag_body=diag_body,
    )
    default_timelogger.stop_event(event)
    return result


__all__ = [
    "generate_neumann_preconditioner_code",
    "generate_jacobi_preconditioner_code",
]
