"""Emit CUDA factory code for nonlinear stage residual functions.

Published Functions
-------------------
:func:`generate_residual_code`
    Emit a factory computing the nonlinear residual
    ``F(Y) = Y - y_n - h * sum(a_ij * f(Y_j))``.

:func:`generate_stage_residual_code`
    Emit a single-stage residual factory for SDIRK/ESDIRK methods.

:func:`generate_n_stage_residual_code`
    Emit a flattened multi-stage residual factory for FIRK methods.

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
from cubie.odesystems.symbolic.parsing.parser import (
    IndexedBases,
    ParsedEquations,
)
from cubie.odesystems.symbolic.sym_utils import (
    render_constant_assignments,
)
from cubie.time_logger import default_timelogger

from ._stage_utils import build_stage_metadata, prepare_stage_data
from ._matrix_utils import mass_matrix_ir

# Register timing events for codegen functions
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
default_timelogger.register_event("codegen_generate_stage_residual_code",
                                   "codegen",
                                   "Codegen time for generate_stage_residual_code")
default_timelogger.register_event("codegen_generate_n_stage_residual_code",
                                   "codegen",
                                   "Codegen time for generate_n_stage_residual_code")

RESIDUAL_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NONLINEAR RESIDUAL FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, lineinfo=None):\n"
    '    """Auto-generated nonlinear residual for implicit updates.\n'
    "    Computes beta * M * u - gamma * h * f(t, base_state + a_ij * u).\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
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
    "    def residual(\n"
    "        u, parameters, drivers, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, base_state, out,\n"
    "    ):\n"
    "{res_lines}\n"
    "    return residual\n"
)


N_STAGE_RESIDUAL_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED N-STAGE RESIDUAL FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, lineinfo=None):\n"
    '    """Auto-generated FIRK residual for flattened stage increments.\n'
    "    Handles {stage_count} stages with ``s * n`` unknowns.\n"
    '    """\n'
    "    _cubie_codegen_beta = precision(beta)\n"
    "    _cubie_codegen_gamma = precision(gamma)\n"
    "{const_lines}"
    "{metadata_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
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
    "    def residual(\n"
    "        u, parameters, drivers, t,\n"
    "        _cubie_codegen_h, _cubie_codegen_a_ij, base_state, out,\n"
    "    ):\n"
    "{body}\n"
    "    return residual\n"
)


def _build_residual_lines(
    sysir: SystemIR,
    M: List[List[ir.Expr]],
    cse: bool = True,
) -> str:
    """Construct CUDA code lines for the stage-increment residual."""

    n = len(sysir.state_symbols)
    beta_sym = ir.sym("_cubie_codegen_beta")
    gamma_sym = ir.sym("_cubie_codegen_gamma")
    h_sym = ir.sym("_cubie_codegen_h")
    aij_sym = ir.sym("_cubie_codegen_a_ij")

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
        mv_terms = []
        for j in range(n):
            entry = M[i][j]
            if ir.is_zero(entry):
                continue
            mv_terms.append(ir.mul(entry, ir.arr("u", j)))
        mv = ir.add(*mv_terms) if mv_terms else ir.ZERO
        dx_sym = ir.sym(f"_cubie_codegen_dx_{i}")
        residual_expr = ir.sub(
            ir.mul(beta_sym, mv),
            ir.mul(gamma_sym, h_sym, dx_sym),
        )
        eval_exprs.append((ir.arr("out", i), residual_expr))

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
    assert lines, "internal error: codegen produced an empty body"
    return "\n".join("        " + ln for ln in lines)


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
    M: List[List[ir.Expr]],
    stage_coefficients: List[List[ir.Expr]],
    stage_nodes: Tuple[ir.Expr, ...],
    cse: bool = True,
) -> str:
    """Construct CUDA statements for the FIRK n-stage residual."""

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
            mv_terms = []
            for col_idx in range(state_count):
                entry = M[comp_idx][col_idx]
                if ir.is_zero(entry):
                    continue
                mv_terms.append(
                    ir.mul(entry, ir.arr("u", stage_offset + col_idx))
                )
            mv = ir.add(*mv_terms) if mv_terms else ir.ZERO
            dx_symbol = ir.sym(
                f"_cubie_codegen_dx_{stage_idx}_{comp_idx}"
            )
            update_expr = ir.sub(
                ir.mul(beta_sym, mv),
                ir.mul(gamma_sym, h_sym, dx_symbol),
            )
            eval_exprs.append(
                (ir.arr("out", stage_offset + comp_idx), update_expr)
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
    assert lines, "internal error: codegen produced an empty body"
    return "\n".join("        " + ln for ln in lines)


def generate_residual_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "residual_factory",
    cse: bool = True,
) -> str:
    """Emit the stage-increment residual factory for Newton--Krylov integration."""

    sysir = system_ir(equations, index_map)
    n = len(sysir.state_symbols)
    mass = mass_matrix_ir(M, n)

    res_lines = _build_residual_lines(
        sysir=sysir,
        M=mass,
        cse=cse,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)

    return RESIDUAL_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        res_lines=res_lines,
    )


def generate_stage_residual_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "stage_residual",
    cse: bool = True,
) -> str:
    """Generate the stage residual factory."""
    default_timelogger.start_event("codegen_generate_stage_residual_code")

    result = generate_residual_code(
        equations=equations,
        index_map=index_map,
        M=M,
        func_name=func_name,
        cse=cse,
    )
    default_timelogger.stop_event("codegen_generate_stage_residual_code")
    return result


def generate_n_stage_residual_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    stage_coefficients: Sequence[Sequence[Union[float, object]]],
    stage_nodes: Sequence[Union[float, object]],
    M: Optional[Union[Iterable, object]] = None,
    func_name: str = "n_stage_residual",
    cse: bool = True,
) -> str:
    """Generate a flattened n-stage FIRK residual factory."""
    default_timelogger.start_event("codegen_generate_n_stage_residual_code")

    coeff_matrix, node_values, stage_count = prepare_stage_data(
        stage_coefficients, stage_nodes
    )
    sysir = system_ir(equations, index_map)
    mass = mass_matrix_ir(M, len(sysir.state_symbols))
    body = _build_n_stage_residual_lines(
        sysir=sysir,
        M=mass,
        stage_coefficients=coeff_matrix,
        stage_nodes=node_values,
        cse=cse,
    )
    const_block = render_constant_assignments(index_map.constants.symbol_map)
    result = N_STAGE_RESIDUAL_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        metadata_lines="",
        body=body,
        stage_count=stage_count,
    )
    default_timelogger.stop_event("codegen_generate_n_stage_residual_code")
    return result


__all__ = [
    "generate_residual_code",
    "generate_stage_residual_code",
    "generate_n_stage_residual_code",
    "build_stage_substitutions",
]
