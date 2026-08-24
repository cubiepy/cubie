"""Emit CUDA factory code for nonlinear stage residual functions.

Published Functions
-------------------
:func:`generate_residual_code`
    Emit ``beta * M * u - gamma * h * f(t, base_state + a_ij * u)``,
    single-stage or flattened all-stages per the variant.

:func:`generate_init_residual_code`
    Emit the consistent-initialisation residual.

See Also
--------
:mod:`cubie.odesystems.symbolic.codegen.linear_operators`
    Companion linear operator code generators.
:mod:`cubie.odesystems.symbolic.codegen.preconditioners`
    Companion preconditioner code generators.
:mod:`cubie.odesystems.symbolic.codegen._stage_utils`
    Shared FIRK stage metadata helpers.
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union

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
from cubie._env import operation_ordering_default
from cubie.odesystems.symbolic.parsing import (
    IndexedBases,
    ParsedEquations,
)
from cubie.time_logger import default_timelogger

from ._stage_utils import build_stage_metadata, prepare_stage_data
from ._matrix_utils import mass_diagonal_flags

# Register timing events for codegen functions
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
for _variant in HelperVariant:
    default_timelogger.register_event(
        f"codegen_residual_{_variant.value}",
        "codegen",
        f"Codegen time for the {_variant.value} residual",
    )
default_timelogger.register_event(
    "codegen_init_residual", "codegen",
    "Codegen time for generate_init_residual_code")

RESIDUAL_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NONLINEAR RESIDUAL FACTORY\n"
    "def {func_name}(precision, lineinfo=None):\n"
    '    """Auto-generated nonlinear residual for implicit updates.\n'
    "    Computes beta * M * u - gamma * h * f(t, base_state + a_ij * u)\n"
    "    with beta and gamma baked in as numeric literals.\n"
    '    """\n'
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def residual(\n"
    "        u, parameters, drivers, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, base_state, out,\n"
    "    ):\n"
    "{body}\n"
    "    return residual\n"
    "# Buffer sizes read by the helper registry\n"
    "{func_name}.aux_count = None\n"
    "{func_name}.lu_nnz = None\n"
)


def _residual_row_expr(
    mv: ir.Expr,
    dx_sym: ir.Expr,
    beta: float,
    gamma: float,
) -> ir.Expr:
    """Return ``beta*M*u - gamma*h*dx`` for one residual row."""
    return ir.sub(
        ir.mul(ir.num(beta), mv),
        ir.mul(
            ir.num(gamma),
            ir.sym("_cubie_codegen_h"),
            dx_sym,
        ),
    )


def _sorted_pruned_lines(
    eval_exprs: List[Tuple[ir.Expr, ir.Expr]],
    sysir: SystemIR,
    cse: bool,
    operation_ordering: str,
) -> str:
    """Sort, prune, and print the assembled residual body."""
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
        nonfloat_functions=sysir.nonfloat_functions,
    )
    assert lines, "internal error: codegen produced an empty body"
    return "\n".join("        " + ln for ln in lines)


def _build_residual_lines(
    sysir: SystemIR,
    mass_diag: Tuple[bool, ...],
    beta: float,
    gamma: float,
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
    a_ij: Optional[float] = None,
) -> str:
    """Construct CUDA code lines for the stage-increment residual.

    A zero mass-diagonal row drops the ``beta * u[i]`` term.
    """

    n = len(sysir.state_symbols)
    if a_ij is None:
        aij_sym = ir.sym("_cubie_codegen_a_ij")
    else:
        aij_sym = ir.num(a_ij)

    # dx/observable outputs become stage locals; states evaluate at
    # base_state + a_ij * u. Domains and images are disjoint, so one
    # simultaneous substitution covers all of it. Internal locals use
    # the reserved _cubie_codegen_ namespace so they can never merge
    # with a same-named user symbol in the hash-consed IR.
    subs_map = {}
    for i, dx_sym in enumerate(sysir.dxdt_symbols):
        subs_map[dx_sym] = ir.sym(f"_cubie_codegen_dx_{i}")
    for position, obs_sym in enumerate(sysir.observable_symbols):
        subs_map[obs_sym] = ir.sym(
            f"_cubie_codegen_aux_{position + 1}"
        )
    for i, state_sym in enumerate(sysir.state_symbols):
        subs_map[state_sym] = ir.add(
            ir.arr("base_state", i),
            ir.mul(aij_sym, ir.arr("u", i)),
        )

    memo: dict = {}
    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = [
        (
            ir.xreplace(lhs, subs_map, memo),
            ir.xreplace(rhs, subs_map, memo),
        )
        for lhs, rhs in sysir.equations
    ]

    for i in range(n):
        mv = ir.arr("u", i) if mass_diag[i] else ir.ZERO
        dx_sym = ir.sym(f"_cubie_codegen_dx_{i}")
        eval_exprs.append(
            (
                ir.arr("out", i),
                _residual_row_expr(mv, dx_sym, beta, gamma),
            )
        )

    return _sorted_pruned_lines(
        eval_exprs, sysir, cse, operation_ordering
    )


def build_stage_substitutions(
    sysir: SystemIR,
    stage_idx: int,
    coeff_symbols: List[List[ir.Sym]],
    node_symbols: List[ir.Sym],
    stage_coefficients: List[List[ir.Expr]],
    state_vector_name: str,
) -> dict:
    """Build the per-stage substitution map for FIRK builders.

    Replaces dx/observable outputs with stage-suffixed locals, the
    time symbol with the stage evaluation time, drivers with their
    stage-flattened slots, and state symbols with
    ``base_state + sum(a_ij * <vec>[j*n + i])``.

    Parameters
    ----------
    sysir
        IR system bundle.
    stage_idx
        Stage being instantiated.
    coeff_symbols
        Coefficient symbols from :func:`build_stage_metadata`.
    node_symbols
        Node symbols from :func:`build_stage_metadata`.
    stage_coefficients
        IR tableau entries (used only for zero-skipping).
    state_vector_name
        Name of the flattened unknown vector (``"u"`` or ``"state"``).

    Returns
    -------
    dict
        Node-for-node substitution map.
    """
    state_count = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)
    h_sym = ir.sym("_cubie_codegen_h")
    time_arg = ir.sym("t")

    subs_map = {}
    for idx, dx_sym in enumerate(sysir.dxdt_symbols):
        subs_map[dx_sym] = ir.sym(
            f"_cubie_codegen_dx_{stage_idx}_{idx}"
        )
    for idx, obs_sym in enumerate(sysir.observable_symbols):
        subs_map[obs_sym] = ir.sym(
            f"_cubie_codegen_aux_{stage_idx}_{idx + 1}"
        )
    # Anonymous auxiliaries must be stage-renamed too: repeated
    # left-hand sides across stages would otherwise collapse to a
    # single assignment during topological sorting, leaving early
    # stages reading another stage's values.
    named = set(subs_map)
    for lhs, _ in sysir.equations:
        if isinstance(lhs, ir.Sym) and lhs not in named:
            subs_map[lhs] = ir.sym(
                f"_cubie_codegen_s{stage_idx}_{lhs.name}"
            )
    subs_map[sysir.time_symbol] = ir.add(
        time_arg, ir.mul(h_sym, node_symbols[stage_idx])
    )

    driver_count = len(sysir.driver_symbols)
    if driver_count:
        stage_driver_offset = stage_idx * driver_count
        for driver_idx, driver_sym in enumerate(sysir.driver_symbols):
            subs_map[driver_sym] = ir.arr(
                "drivers", stage_driver_offset + driver_idx
            )

    for state_idx, state_sym in enumerate(sysir.state_symbols):
        terms: List[ir.Expr] = [ir.arr("base_state", state_idx)]
        for contrib_idx in range(stage_count):
            if ir.is_zero(
                stage_coefficients[stage_idx][contrib_idx]
            ):
                continue
            coeff_sym = coeff_symbols[stage_idx][contrib_idx]
            terms.append(
                ir.mul(
                    coeff_sym,
                    ir.arr(
                        state_vector_name,
                        contrib_idx * state_count + state_idx,
                    ),
                )
            )
        subs_map[state_sym] = ir.add(*terms)
    return subs_map


def _build_n_stage_residual_lines(
    sysir: SystemIR,
    mass_diag: Tuple[bool, ...],
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    beta: float,
    gamma: float,
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Construct CUDA statements for the FIRK n-stage residual."""

    metadata_exprs, coeff_symbols, node_symbols = build_stage_metadata(
        stage_coefficients, stage_nodes
    )
    state_count = len(sysir.state_symbols)
    stage_count = len(stage_coefficients)

    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = list(metadata_exprs)

    for stage_idx in range(stage_count):
        subs_map = build_stage_substitutions(
            sysir,
            stage_idx,
            coeff_symbols,
            node_symbols,
            stage_coefficients,
            state_vector_name="u",
        )
        memo: dict = {}
        substituted = [
            (
                ir.xreplace(lhs, subs_map, memo),
                ir.xreplace(rhs, subs_map, memo),
            )
            for lhs, rhs in sysir.equations
        ]
        eval_exprs.extend(substituted)

        stage_offset = stage_idx * state_count
        for comp_idx in range(state_count):
            if mass_diag[comp_idx]:
                mv = ir.arr("u", stage_offset + comp_idx)
            else:
                mv = ir.ZERO
            dx_symbol = ir.sym(
                f"_cubie_codegen_dx_{stage_idx}_{comp_idx}"
            )
            eval_exprs.append(
                (
                    ir.arr("out", stage_offset + comp_idx),
                    _residual_row_expr(mv, dx_symbol, beta, gamma),
                )
            )

    return _sorted_pruned_lines(
        eval_exprs, sysir, cse, operation_ordering
    )


def generate_residual_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    variant: HelperVariant = HelperVariant.PLAIN,
    M: Optional[Union[Iterable, object]] = None,
    stage_coefficients: Optional[
        Sequence[Sequence[Union[float, object]]]
    ] = None,
    stage_nodes: Optional[Sequence[Union[float, object]]] = None,
    func_name: str = "residual_factory",
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
    beta: float = 1.0,
    gamma: float = 1.0,
    a_ij: Optional[float] = None,
) -> str:
    """Generate the stage-increment residual factory for one variant.

    Parameters
    ----------
    equations
        Parsed ODE equations.
    index_map
        Symbol-to-array mapping for states, parameters, etc.
    variant
        Helper variant: single-stage or flattened all-stages.
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
    event = f"codegen_residual_{variant.value}"
    default_timelogger.start_event(event)

    sysir = system_ir(equations, index_map)
    mass_diag = mass_diagonal_flags(M, len(sysir.state_symbols))
    if variant.stacked_stages:
        coeff_matrix, node_values, _ = prepare_stage_data(
            stage_coefficients, stage_nodes
        )
        body = _build_n_stage_residual_lines(
            sysir=sysir,
            mass_diag=mass_diag,
            stage_coefficients=coeff_matrix,
            stage_nodes=node_values,
            beta=beta,
            gamma=gamma,
            cse=cse,
            operation_ordering=operation_ordering,
        )
    else:
        body = _build_residual_lines(
            sysir=sysir,
            mass_diag=mass_diag,
            beta=beta,
            gamma=gamma,
            cse=cse,
            operation_ordering=operation_ordering,
            a_ij=a_ij,
        )
    result = RESIDUAL_TEMPLATE.format(
        func_name=func_name,
        body=body,
    )
    default_timelogger.stop_event(event)
    return result


INIT_RESIDUAL_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED CONSISTENT-INITIALISATION RESIDUAL FACTORY\n"
    "def {func_name}(precision, lineinfo=None):\n"
    '    """Auto-generated consistent-initialisation residual.\n'
    "    Differential rows pin the increment (out[i] = u[i]);\n"
    "    algebraic rows keep the unscaled constraint\n"
    "    (out[i] = -f_i(t, base_state + u)). h and a_ij are unused.\n"
    '    """\n'
    "    @cuda.jit(\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def residual(\n"
    "        u, parameters, drivers, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, base_state, out,\n"
    "    ):\n"
    "{body}\n"
    "    return residual\n"
    "# Buffer sizes read by the helper registry\n"
    "{func_name}.aux_count = None\n"
    "{func_name}.lu_nnz = None\n"
)


def _init_state_substitutions(sysir: SystemIR) -> dict:
    """Map outputs to locals and states to ``base_state + u``."""
    subs_map = {}
    for i, dx_sym in enumerate(sysir.dxdt_symbols):
        subs_map[dx_sym] = ir.sym(f"_cubie_codegen_dx_{i}")
    for position, obs_sym in enumerate(sysir.observable_symbols):
        subs_map[obs_sym] = ir.sym(
            f"_cubie_codegen_aux_{position + 1}"
        )
    for i, state_sym in enumerate(sysir.state_symbols):
        subs_map[state_sym] = ir.add(
            ir.arr("base_state", i), ir.arr("u", i)
        )
    return subs_map


def _build_init_residual_lines(
    sysir: SystemIR,
    mass_diag: Tuple[bool, ...],
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Emit ``u[i]`` on identity-mass rows, ``-f_i`` on zero rows."""
    n = len(sysir.state_symbols)
    subs_map = _init_state_substitutions(sysir)

    memo: dict = {}
    eval_exprs: List[Tuple[ir.Expr, ir.Expr]] = [
        (
            ir.xreplace(lhs, subs_map, memo),
            ir.xreplace(rhs, subs_map, memo),
        )
        for lhs, rhs in sysir.equations
    ]

    for i in range(n):
        if mass_diag[i]:
            row = ir.arr("u", i)
        else:
            dx_sym = ir.sym(f"_cubie_codegen_dx_{i}")
            row = ir.sub(ir.ZERO, dx_sym)
        eval_exprs.append((ir.arr("out", i), row))

    return _sorted_pruned_lines(
        eval_exprs, sysir, cse, operation_ordering
    )


def generate_init_residual_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "init_residual_factory",
    cse: bool = True,
    operation_ordering: str = operation_ordering_default(),
) -> str:
    """Generate the consistent-initialisation residual factory.

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
    operation_ordering
        Statement-ordering policy for the emitted body.

    Returns
    -------
    str
        Generated factory source.
    """
    default_timelogger.start_event("codegen_init_residual")

    sysir = system_ir(equations, index_map)
    mass_diag = mass_diagonal_flags(M, len(sysir.state_symbols))
    body = _build_init_residual_lines(
        sysir=sysir,
        mass_diag=mass_diag,
        cse=cse,
        operation_ordering=operation_ordering,
    )
    result = INIT_RESIDUAL_TEMPLATE.format(
        func_name=func_name,
        body=body,
    )
    default_timelogger.stop_event("codegen_init_residual")
    return result


__all__ = [
    "generate_residual_code",
    "generate_init_residual_code",
    "build_stage_substitutions",
]
