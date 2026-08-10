"""Emit CUDA factory code for explicit time derivatives of the RHS.

Published Functions
-------------------
:func:`generate_time_derivative_fac_code`
    Return a string containing the ``time_derivative_rhs`` factory
    function definition, which computes the partial derivative of the
    RHS with respect to time.

See Also
--------
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`
    Serves this helper for a ``SolverHelperRequest`` of kind
    ``time_derivative_rhs`` through ``get_solver_helper``.
:mod:`cubie.odesystems.symbolic.codegen.dxdt`
    Companion module generating the primary ``dxdt`` factory.
:mod:`cubie.odesystems.symbolic.engine`
    Expression engine used for differentiation and printing.
"""

from typing import Dict, List, Tuple

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
from cubie.odesystems.symbolic.parsing import (
    IndexedBases,
    ParsedEquations,
)
from cubie.odesystems.symbolic.sym_utils import (
    render_constant_assignments,
)
from cubie.time_logger import default_timelogger


# Register timing event for codegen function
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
default_timelogger.register_event("codegen_generate_time_derivative_fac_code",
                                   "codegen",
                                   "Codegen time for generate_time_derivative_fac_code")


TIME_DERIVATIVE_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED TIME-DERIVATIVE FACTORY\n"
    "def {func_name}(constants, precision, lineinfo=None):\n"
    '    """Auto-generated time-derivative factory."""\n'
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def time_derivative_rhs(\n"
    "        state, parameters, drivers, driver_dt, observables, out, t\n"
    "    ):\n"
    "{body}\n"
    "\n"
    "    return time_derivative_rhs\n"
)


def _build_time_derivative_assignments(
    sysir: SystemIR,
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Build IR assignments for time-derivative evaluation.

    Returns
    -------
    list of tuple
        Original equations, their total time derivatives, and the
        final ``out[i]`` assignments.
    """
    sorted_equations = topological_sort(
        sysir.non_observable_equations()
    )
    driver_symbols = list(sysir.driver_symbols)
    time_symbol = sysir.time_symbol
    derivative_names = sysir.derivative_names

    symbol_derivatives: Dict[ir.Expr, ir.Expr] = {}
    derivative_symbols: Dict[ir.Expr, ir.Sym] = {}

    assignments: List[Tuple[ir.Expr, ir.Expr]] = list(sorted_equations)
    derivative_assignments: List[Tuple[ir.Expr, ir.Expr]] = []

    time_memo: Dict = {}
    driver_memos: Dict[ir.Sym, Dict] = {
        drv: {} for drv in driver_symbols
    }

    processed: set = set()
    for lhs, rhs in sorted_equations:
        processed.add(lhs)
        direct_time = ir.diff(
            rhs,
            time_symbol,
            memo=time_memo,
            derivative_names=derivative_names,
        )

        driver_terms: List[ir.Expr] = []
        rhs_atoms = ir.free_atoms(rhs)
        for driver in driver_symbols:
            if driver in rhs_atoms:
                partial = ir.diff(
                    rhs,
                    driver,
                    memo=driver_memos[driver],
                    derivative_names=derivative_names,
                )
                driver_terms.append(
                    ir.mul(
                        partial,
                        ir.arr(
                            "driver_dt",
                            sysir.driver_index[driver],
                        ),
                    )
                )

        chain_terms: List[ir.Expr] = []
        for dep in sorted(
            rhs_atoms & processed, key=lambda node: node.sort_key
        ):
            derivative = symbol_derivatives.get(dep)
            if derivative is None:
                continue
            partial = ir.diff(
                rhs, dep, derivative_names=derivative_names
            )
            chain_terms.append(ir.mul(partial, derivative))

        total = ir.add(direct_time, *driver_terms, *chain_terms)
        deriv_symbol = ir.sym(f"_cubie_codegen_time_{lhs.name}")
        symbol_derivatives[lhs] = total
        derivative_symbols[lhs] = deriv_symbol
        derivative_assignments.append((deriv_symbol, total))

    assignments.extend(derivative_assignments)

    for position, dx_sym in enumerate(sysir.dxdt_symbols):
        assignments.append(
            (ir.arr("out", position), derivative_symbols[dx_sym])
        )
    return assignments


def generate_time_derivative_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    cse: bool = True,
) -> List[str]:
    """Generate CUDA source lines for time-derivative computation.

    Parameters
    ----------
    equations
        Parsed equations describing the ODE system.
    index_map
        Indexed bases mapping symbols to CUDA array references.
    cse
        Whether to apply common subexpression elimination.

    Returns
    -------
    list[str]
        CUDA source lines computing the explicit time derivative.
    """
    sysir = system_ir(equations, index_map)
    assignments = _build_time_derivative_assignments(sysir)

    if cse:
        processed = cse_and_stack(assignments)
    else:
        processed = topological_sort(assignments)

    processed = prune_unused(processed, output_name="out")

    lines = print_cuda_multiple(
        processed,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    assert lines, "internal error: codegen produced an empty body"
    return lines


def generate_time_derivative_fac_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "time_derivative_rhs_factory",
    cse: bool = True,
) -> str:
    """Emit Python source for a time-derivative CUDA factory.

    Parameters
    ----------
    equations
        Parsed equations describing the ODE system.
    index_map
        Indexed bases providing symbol references and constants.
    func_name
        Name of the generated factory function.
    cse
        Whether to apply common subexpression elimination.

    Returns
    -------
    str
        Python source code implementing the factory function.
    """
    default_timelogger.start_event("codegen_generate_time_derivative_fac_code")

    body_lines = generate_time_derivative_lines(
        equations, index_map=index_map, cse=cse
    )
    body = "\n".join(f"        {line}" for line in body_lines)
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )
    result = TIME_DERIVATIVE_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        body=body,
    )
    default_timelogger.stop_event("codegen_generate_time_derivative_fac_code")
    return result
