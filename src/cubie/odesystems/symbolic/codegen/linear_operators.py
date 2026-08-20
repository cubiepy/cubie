"""Emit CUDA factory code for linear operators and Jacobian caching.

Published Functions
-------------------
:func:`generate_linear_operator_code`
    Emit ``beta * (M @ v) - gamma * a_ij * h * (J @ v)`` for one
    :class:`~cubie.odesystems.solver_helpers.HelperVariant`.

:func:`generate_init_operator_code`
    Emit the consistent-initialisation operator.

:func:`generate_prepare_jac_code`
    Emit the factory filling the auxiliary cache buffer.

:func:`generate_apply_mass_code`
    Emit the 0/1 diagonal mass-matrix product.

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

from cubie.odesystems.solver_helpers import HelperVariant
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.adapter import SystemIR, system_ir
from cubie.odesystems.symbolic.engine.assignments import (
    cse_and_stack,
    prune_unused,
    topological_sort,
)
from cubie.odesystems.symbolic.engine.printer import (
    indent_lines,
    print_cuda_multiple,
)
from cubie.odesystems.symbolic.codegen.jacobian import (
    generate_analytical_jvp,
)
from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations
from cubie._env import operation_ordering_default
from cubie.odesystems.symbolic.parsing import (
    IndexedBases,
    ParsedEquations,
)
from cubie.time_logger import default_timelogger

from ._matrix_utils import mass_diagonal_flags
from ._stage_utils import build_stage_metadata, prepare_stage_data
from .nonlinear_residuals import build_stage_substitutions

# Register timing events for codegen functions
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
for _variant in HelperVariant:
    default_timelogger.register_event(
        f"codegen_linear_operator_{_variant.value}",
        "codegen",
        f"Codegen time for the {_variant.value} linear operator",
    )
default_timelogger.register_event(
    "codegen_prepare_jac", "codegen",
    "Codegen time for generate_prepare_jac_code")
default_timelogger.register_event(
    "codegen_init_operator", "codegen",
    "Codegen time for generate_init_operator_code")
default_timelogger.register_event(
    "codegen_apply_mass", "codegen",
    "Codegen time for generate_apply_mass_code")


OPERATOR_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(precision, lineinfo=None):\n"
    '    """Auto-generated linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * a_ij * h * (J @ v)\n"
    "    with beta and gamma baked in as numeric literals.\n"
    "    Returns device function:\n"
    "      operator_apply(\n"
    "          state, parameters, drivers, cached_aux, base_state, t, h, "
    "a_ij, v, out\n"
    "      )\n"
    '    """\n'
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def operator_apply(\n"
    "        state, parameters, drivers, cached_aux, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, v, out,\n"
    "    ):\n"
    "{body}\n"
    "    return operator_apply\n"
    "# Buffer sizes read by the helper registry\n"
    "{func_name}.aux_count = None\n"
    "{func_name}.lu_nnz = None\n"
)


PREPARE_JAC_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED JACOBIAN PREPARATION FACTORY\n"
    "def {func_name}(precision, lineinfo=None):\n"
    '    """Auto-generated Jacobian auxiliary preparation.\n'
    "    Populates cached_aux with intermediate Jacobian values.\n"
    "    Signature (state, parameters, drivers, t, h, cached_aux)\n"
    "    -> int32 status; always returns int32(0).\n"
    '    """\n'
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def prepare_jac(state, parameters, drivers, t, h, cached_aux):\n"
    "{body}\n"
    "        return int32(0)\n"
    "    return prepare_jac\n"
    "# Buffer sizes read by the helper registry\n"
    "{func_name}.aux_count = {aux_count}\n"
    "{func_name}.lu_nnz = None\n"
)


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


def hoisted_scale(
    name: str,
    *factors: ir.Expr,
) -> Tuple[List[Tuple[ir.Expr, ir.Expr]], ir.Expr]:
    """Name the product of ``factors`` so callers can reuse it.

    Returns ``(assignments, scale)``: one assignment binding the
    product to ``name``, and the named symbol to use in its place.
    When the product simplifies to a single number, symbol, or
    array element there is nothing to reuse, so no assignment is
    made and that value is returned directly as ``scale``.
    """
    value = ir.mul(*factors)
    if isinstance(value, (ir.Num, ir.Sym, ir.Local, ir.Arr)):
        return [], value
    scale_sym = ir.sym(name)
    return [(scale_sym, value)], scale_sym


def _state_increment_subs(
    sysir: SystemIR,
    a_ij_expr: Optional[ir.Expr] = None,
) -> Dict[ir.Expr, ir.Expr]:
    """Map state symbols to ``base_state + a_ij * state`` eval points.

    Plain-variant only: there ``state`` is the stage increment;
    cached and at-state bodies read the eval state from ``state``.
    """
    if a_ij_expr is None:
        a_ij_expr = ir.sym("_cubie_codegen_a_ij")
    subs = {}
    for i, state_sym in enumerate(sysir.state_symbols):
        subs[state_sym] = ir.add(
            ir.arr("base_state", i),
            ir.mul(a_ij_expr, ir.arr("state", i)),
        )
    return subs


def _build_operator_body(
    aux_assignments: List[Tuple[ir.Expr, ir.Expr]],
    jvp_terms: Dict[int, ir.Expr],
    sysir: SystemIR,
    mass_diag: Tuple[bool, ...],
    beta: float,
    gamma: float,
    state_is_increment: bool = True,
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
    a_ij: Optional[float] = None,
) -> str:
    """Build the CUDA body computing ``β·M·v − γ·h·J·v``.

    A zero mass-diagonal row drops the ``beta * v[i]`` term.
    """

    n_out = len(sysir.dxdt_symbols)
    beta_num = ir.num(beta)
    gamma_num = ir.num(gamma)
    if a_ij is None:
        a_ij_expr = ir.sym("_cubie_codegen_a_ij")
    else:
        a_ij_expr = ir.num(a_ij)
    h_sym = ir.sym("_cubie_codegen_h")

    # Newton increments evaluate at base_state + a_ij * state;
    # cached and at-state bodies evaluate at state directly.
    if state_is_increment:
        state_subs = _state_increment_subs(sysir, a_ij_expr)
    else:
        state_subs = {}
    memo: dict = {}

    scale_assigns, scale = hoisted_scale(
        "_cubie_codegen_jac_scale", gamma_num, a_ij_expr, h_sym
    )
    out_updates: List[Tuple[ir.Expr, ir.Expr]] = list(scale_assigns)
    for i in range(n_out):
        mv = ir.arr("v", i) if mass_diag[i] else ir.ZERO
        jvp_term = jvp_terms.get(i, ir.ZERO)
        if state_subs:
            jvp_term = ir.xreplace(jvp_term, state_subs, memo)
        rhs = ir.sub(
            ir.mul(beta_num, mv),
            ir.mul(scale, jvp_term),
        )
        out_updates.append((ir.arr("out", i), rhs))

    if state_subs:
        aux_assignments = [
            (lhs, ir.xreplace(rhs, state_subs, memo))
            for lhs, rhs in aux_assignments
        ]

    exprs = list(aux_assignments) + out_updates
    if cse:
        exprs = cse_and_stack(
            exprs, operation_ordering=operation_ordering
        )
    else:
        exprs = topological_sort(
            exprs, operation_ordering=operation_ordering
        )
    exprs = prune_unused(exprs, output_name="out")

    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        function_aliases=sysir.function_aliases,
    )
    return indent_lines(lines, 8)


def _build_prepare_body(
    jvp_equations: JVPEquations,
    sysir: SystemIR,
) -> str:
    """Build the CUDA body populating the cached Jacobian auxiliaries."""

    exprs = jvp_equations.prepare_fill_assignments()
    exprs = prune_unused(exprs, output_name="cached_aux")

    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        function_aliases=sysir.function_aliases,
    )
    if not lines:
        return "        pass"
    return indent_lines(lines, 8)


def _resolve_jvp(
    equations: ParsedEquations,
    index_map: IndexedBases,
    cse: bool,
    jvp_equations: Optional[JVPEquations],
    operation_ordering: str = operation_ordering_default(),
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


def generate_linear_operator_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    variant: HelperVariant = HelperVariant.PLAIN,
    M: Optional[Union[Iterable, object]] = None,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, object]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, object]]] = None,
    func_name: str = "operator_apply_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = operation_ordering_default(),
    beta: float = 1.0,
    gamma: float = 1.0,
    a_ij: Optional[float] = None,
) -> str:
    """Generate the linear operator factory for one variant.

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
    jvp_equations
        Precomputed JVP expressions; generated when omitted.
    beta
        Mass-matrix shift scaling, folded in as a numeric literal.
    gamma
        Jacobian-term weight, folded in as a numeric literal.
    a_ij
        Stage diagonal baked as a literal; ``None`` keeps it runtime.

    Returns
    -------
    str
        Generated Python/CUDA factory function code.
    """
    event = f"codegen_linear_operator_{variant.value}"
    default_timelogger.start_event(event)

    sysir = system_ir(equations, index_map)
    mass_diag = mass_diagonal_flags(M, len(sysir.state_symbols))
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    if variant is HelperVariant.CACHED_STACKED:
        coeff_matrix, node_values, _ = prepare_stage_data(
            stage_coefficients, stage_nodes
        )
        body = _build_cached_stacked_operator_lines(
            sysir=sysir,
            mass_diag=mass_diag,
            stage_coefficients=coeff_matrix,
            stage_nodes=node_values,
            jvp_equations=jvp_equations,
            beta=beta,
            gamma=gamma,
            cse=cse,
            operation_ordering=operation_ordering,
        )
    elif variant.stacked_stages:
        coeff_matrix, node_values, _ = prepare_stage_data(
            stage_coefficients, stage_nodes
        )
        body = _build_n_stage_operator_lines(
            sysir=sysir,
            mass_diag=mass_diag,
            stage_coefficients=coeff_matrix,
            stage_nodes=node_values,
            jvp_equations=jvp_equations,
            beta=beta,
            gamma=gamma,
            cse=cse,
            operation_ordering=operation_ordering,
        )
    elif variant.cached:
        body = _build_operator_body(
            aux_assignments=jvp_equations.cached_runtime_assignments(),
            jvp_terms=jvp_equations.jvp_terms,
            sysir=sysir,
            mass_diag=mass_diag,
            beta=beta,
            gamma=gamma,
            state_is_increment=False,
            cse=cse,
            operation_ordering=operation_ordering,
            a_ij=a_ij,
        )
    else:
        body = _build_operator_body(
            aux_assignments=_inline_aux_assignments(jvp_equations),
            jvp_terms=jvp_equations.jvp_terms,
            sysir=sysir,
            mass_diag=mass_diag,
            beta=beta,
            gamma=gamma,
            state_is_increment=variant is HelperVariant.PLAIN,
            cse=cse,
            operation_ordering=operation_ordering,
            a_ij=a_ij,
        )
    result = OPERATOR_TEMPLATE.format(
        func_name=func_name,
        body=body,
    )
    default_timelogger.stop_event(event)
    return result


def generate_prepare_jac_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "prepare_jac",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = operation_ordering_default(),
) -> Tuple[str, int]:
    """Generate the cached auxiliary preparation factory."""
    default_timelogger.start_event("codegen_prepare_jac")

    sysir = system_ir(equations, index_map)
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    body = _build_prepare_body(jvp_equations, sysir)
    aux_count = len(jvp_equations.cached_slot_order)
    code = PREPARE_JAC_TEMPLATE.format(
        func_name=func_name, body=body, aux_count=aux_count
    )
    default_timelogger.stop_event("codegen_prepare_jac")
    return code, aux_count


def build_stage_cached_jvp_assignments(
    sysir: SystemIR,
    jvp_equations: JVPEquations,
    stage_idx: int,
    coeff_symbols: List[List[ir.Sym]],
    stage_coefficients: List[List[ir.Expr]],
    direction_name: str = "v",
) -> Tuple[List[Tuple[ir.Expr, ir.Expr]], Dict[int, ir.Sym]]:
    """Instantiate one stage's JVP terms on the shared frozen chain.

    Stage-renames only the v-dependent assignments, with ``v``
    replaced by ``sum_j a[stage][j] * <direction_name>[j*n + i]``.

    Returns
    -------
    tuple
        Stage-suffixed assignments plus JVP terms, and a mapping
        from output index to the stage JVP symbol.
    """
    state_count = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)
    v_dependent = jvp_equations.v_dependent_nodes

    subs_map: Dict[ir.Expr, ir.Expr] = {}
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

    for lhs in jvp_equations.non_jvp_order:
        if lhs in v_dependent:
            subs_map[lhs] = ir.sym(
                f"_cubie_codegen_s{stage_idx}_{lhs.name}"
            )

    memo: dict = {}
    assignments: List[Tuple[ir.Expr, ir.Expr]] = []
    for lhs in jvp_equations.non_jvp_order:
        if lhs not in v_dependent:
            continue
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


def cached_shared_assignments(
    jvp_equations: JVPEquations,
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Return the v-independent canonical chain with slots bound.

    Raises
    ------
    ValueError
        If the cache plan selected a v-dependent leaf.
    """
    v_dependent = jvp_equations.v_dependent_nodes
    for lhs in jvp_equations.cached_slot_order:
        if lhs in v_dependent:
            raise ValueError(
                f"Cached auxiliary {lhs} depends on the direction "
                "vector; the cache plan must only select "
                "prepare-computable values."
            )
    return [
        (lhs, rhs)
        for lhs, rhs in jvp_equations.cached_runtime_assignments()
        if lhs not in v_dependent
    ]


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


def _build_cached_stacked_operator_lines(
    sysir: SystemIR,
    mass_diag: Tuple[bool, ...],
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    jvp_equations: JVPEquations,
    beta: float,
    gamma: float,
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Construct the frozen-Jacobian FIRK operator body."""
    metadata_exprs, coeff_symbols, _ = build_stage_metadata(
        stage_coefficients, stage_nodes
    )
    state_count = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)

    beta_num = ir.num(beta)
    gamma_num = ir.num(gamma)
    h_sym = ir.sym("_cubie_codegen_h")

    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = list(metadata_exprs)
    scale_assigns, scale = hoisted_scale(
        "_cubie_codegen_jac_scale", gamma_num, h_sym
    )
    eval_exprs.extend(scale_assigns)
    eval_exprs.extend(cached_shared_assignments(jvp_equations))

    for stage_idx in range(stage_count):
        stage_assignments, stage_jvp_symbols = (
            build_stage_cached_jvp_assignments(
                sysir,
                jvp_equations,
                stage_idx,
                coeff_symbols,
                stage_coefficients,
            )
        )
        eval_exprs.extend(stage_assignments)

        stage_offset = stage_idx * state_count
        for comp_idx in range(state_count):
            if mass_diag[comp_idx]:
                mv = ir.arr("v", stage_offset + comp_idx)
            else:
                mv = ir.ZERO
            jvp_value = stage_jvp_symbols.get(comp_idx, ir.ZERO)
            update_expr = ir.sub(
                ir.mul(beta_num, mv),
                ir.mul(scale, jvp_value),
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
        function_aliases=sysir.function_aliases,
    )
    return "\n".join("        " + ln for ln in lines)


def _build_n_stage_operator_lines(
    sysir: SystemIR,
    mass_diag: Tuple[bool, ...],
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    jvp_equations: JVPEquations,
    beta: float,
    gamma: float,
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Construct CUDA statements for the FIRK n-stage linear operator."""

    metadata_exprs, coeff_symbols, node_symbols = build_stage_metadata(
        stage_coefficients, stage_nodes
    )
    state_count = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)

    beta_num = ir.num(beta)
    gamma_num = ir.num(gamma)
    h_sym = ir.sym("_cubie_codegen_h")

    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = list(metadata_exprs)
    scale_assigns, scale = hoisted_scale(
        "_cubie_codegen_jac_scale", gamma_num, h_sym
    )
    eval_exprs.extend(scale_assigns)

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
            if mass_diag[comp_idx]:
                mv = ir.arr("v", stage_offset + comp_idx)
            else:
                mv = ir.ZERO
            jvp_value = stage_jvp_symbols.get(comp_idx, ir.ZERO)
            update_expr = ir.sub(
                ir.mul(beta_num, mv),
                ir.mul(scale, jvp_value),
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
        function_aliases=sysir.function_aliases,
    )
    return indent_lines(lines, 8)


INIT_OPERATOR_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CONSISTENT-INITIALISATION OPERATOR FACTORY\n"
    "def {func_name}(precision, lineinfo=None):\n"
    '    """Auto-generated consistent-initialisation operator.\n'
    "    Differential rows are the identity (out[i] = v[i]);\n"
    "    algebraic rows apply the negated Jacobian of the constraint\n"
    "    (out[i] = -(J @ v)[i]) evaluated at base_state + state.\n"
    "    h and a_ij are unused.\n"
    "    Returns device function:\n"
    "      operator_apply(\n"
    "          state, parameters, drivers, cached_aux, base_state, t, h, "
    "a_ij, v, out\n"
    "      )\n"
    '    """\n'
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def operator_apply(\n"
    "        state, parameters, drivers, cached_aux, base_state, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, v, out,\n"
    "    ):\n"
    "{body}\n"
    "    return operator_apply\n"
    "# Buffer sizes read by the helper registry\n"
    "{func_name}.aux_count = None\n"
    "{func_name}.lu_nnz = None\n"
)


def _build_init_operator_body(
    aux_assignments: List[Tuple[ir.Expr, ir.Expr]],
    jvp_terms: Dict[int, ir.Expr],
    sysir: SystemIR,
    mass_diag: Tuple[bool, ...],
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Emit ``v[i]`` on identity-mass rows, ``-(J @ v)[i]`` on zero rows."""
    n_out = len(sysir.dxdt_symbols)
    state_subs = _state_increment_subs(sysir, ir.num(1.0))
    memo: dict = {}

    out_updates: List[Tuple[ir.Expr, ir.Expr]] = []
    for i in range(n_out):
        if mass_diag[i]:
            rhs = ir.arr("v", i)
        else:
            jvp_term = jvp_terms.get(i, ir.ZERO)
            jvp_term = ir.xreplace(jvp_term, state_subs, memo)
            rhs = ir.sub(ir.ZERO, jvp_term)
        out_updates.append((ir.arr("out", i), rhs))

    aux_assignments = [
        (lhs, ir.xreplace(rhs, state_subs, memo))
        for lhs, rhs in aux_assignments
    ]

    exprs = list(aux_assignments) + out_updates
    if cse:
        exprs = cse_and_stack(
            exprs, operation_ordering=operation_ordering
        )
    else:
        exprs = topological_sort(
            exprs, operation_ordering=operation_ordering
        )
    exprs = prune_unused(exprs, output_name="out")

    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        function_aliases=sysir.function_aliases,
    )
    return indent_lines(lines, 8)


def generate_init_operator_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "init_operator_factory",
    cse: bool = True,
    jvp_equations: Optional[JVPEquations] = None,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Generate the consistent-initialisation operator factory.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    M
        0/1 diagonal mass matrix; identity when omitted.
    func_name
        Name for the generated factory function.
    cse
        Whether to apply common-subexpression elimination.
    jvp_equations
        Precomputed JVP expressions; generated when omitted.
    operation_ordering
        Statement-ordering policy for the emitted body.

    Returns
    -------
    str
        Generated factory source.
    """
    default_timelogger.start_event("codegen_init_operator")

    sysir = system_ir(equations, index_map)
    mass_diag = mass_diagonal_flags(M, len(sysir.state_symbols))
    jvp_equations = _resolve_jvp(
        equations,
        index_map,
        cse,
        jvp_equations,
        operation_ordering,
    )
    body = _build_init_operator_body(
        aux_assignments=_inline_aux_assignments(jvp_equations),
        jvp_terms=jvp_equations.jvp_terms,
        sysir=sysir,
        mass_diag=mass_diag,
        cse=cse,
        operation_ordering=operation_ordering,
    )
    result = INIT_OPERATOR_TEMPLATE.format(
        func_name=func_name,
        body=body,
    )
    default_timelogger.stop_event("codegen_init_operator")
    return result


APPLY_MASS_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED MASS-MATRIX APPLY FACTORY\n"
    "def {func_name}(precision, lineinfo=None):\n"
    '    """Auto-generated mass-matrix product.\n'
    "    Computes out = M @ v. `out` must not alias `v`.\n"
    '    """\n'
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def apply_mass(v, out):\n"
    "{body}\n"
    "    return apply_mass\n"
    "# Buffer sizes read by the helper registry\n"
    "{func_name}.aux_count = None\n"
    "{func_name}.lu_nnz = None\n"
)


def _mass_apply_body(mass_diag, sysir, n: int) -> str:
    """Render ``out = M @ v`` for a 0/1 diagonal mass as a CUDA body."""
    exprs: List[Tuple[ir.Expr, ir.Expr]] = [
        (ir.arr("out", i), ir.arr("v", i) if mass_diag[i] else ir.ZERO)
        for i in range(n)
    ]
    lines = print_cuda_multiple(
        exprs,
        symbol_map=sysir.arrayrefs,
        function_aliases=sysir.function_aliases,
    )
    return indent_lines(lines, 8)


def generate_apply_mass_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "apply_mass_factory",
) -> str:
    """Generate a factory applying the mass matrix to a vector."""
    default_timelogger.start_event("codegen_apply_mass")

    sysir = system_ir(equations, index_map)
    n = len(sysir.state_symbols)
    mass_diag = mass_diagonal_flags(M, n)

    body = _mass_apply_body(mass_diag, sysir, n)
    result = APPLY_MASS_TEMPLATE.format(
        func_name=func_name, body=body
    )
    default_timelogger.stop_event("codegen_apply_mass")
    return result


__all__ = [
    "generate_linear_operator_code",
    "generate_init_operator_code",
    "generate_prepare_jac_code",
    "generate_apply_mass_code",
    "build_stage_jvp_assignments",
    "build_stage_cached_jvp_assignments",
    "cached_shared_assignments",
]
