"""Emit CUDA ``dxdt`` and observables factory code from parsed equations.

Published Functions
-------------------
:func:`generate_dxdt_fac_code`
    Return a string containing the ``dxdt_factory`` function definition
    ready for disk caching and import.

:func:`generate_observables_fac_code`
    Return a string containing the ``observables_factory`` function
    definition.

:func:`generate_evaluate_inv_mass_f_code`
    Return a factory computing ``out = M**-1 @ f(state, t)``.

See Also
--------
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`
    Calls these generators inside :meth:`SymbolicODE.build`.
:mod:`cubie.odesystems.symbolic.engine`
    Expression engine used for manipulation and printing.
:class:`~cubie.odesystems.symbolic.odefile.ODEFile`
    Disk cache that stores and imports the generated code.
"""

from typing import Optional

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.adapter import system_ir
from cubie.odesystems.symbolic.engine.assignments import (
    cse_and_stack,
    inline_cheap_assignments,
    prune_unused,
    topological_sort,
)
from cubie.odesystems.symbolic.engine.printer import (
    print_cuda_multiple,
)
from cubie.odesystems.symbolic.parsing import IndexedBases, ParsedEquations
from cubie.odesystems.symbolic.sym_utils import (
    render_constant_assignments,
)
from cubie.time_logger import default_timelogger

from ._matrix_utils import (
    mass_matrix_inverse_ir,
    mass_matrix_is_identity,
)

# Register timing events for codegen functions
# Module-level registration required since codegen functions return code
# strings rather than cacheable objects that could auto-register
default_timelogger.register_event("codegen_generate_dxdt_fac_code", "codegen",
                                   "Codegen time for generate_dxdt_fac_code")
default_timelogger.register_event("codegen_generate_observables_fac_code",
                                   "codegen",
                                   "Codegen time for generate_observables_fac_code")
default_timelogger.register_event(
    "codegen_generate_evaluate_inv_mass_f_code", "codegen",
    "Codegen time for generate_evaluate_inv_mass_f_code")

DXDT_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED DXDT FACTORY\n"
    "def {func_name}(constants, precision, lineinfo=None):\n"
    '    """Auto-generated dxdt factory."""\n'
    "{const_lines}"
    "    \n"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def dxdt(state, parameters, drivers, observables, out, t):\n"
    "    {body}\n"
    "    \n"
    "    return dxdt\n"
)

OBSERVABLES_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED OBSERVABLES FACTORY\n"
    "def {func_name}(constants, precision, lineinfo=None):\n"
    '    """Auto-generated observables factory."""\n'
    "{const_lines}"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def get_observables(state, parameters, drivers, observables, t):\n"
    "    {body}\n"
    "    \n"
    "    return get_observables\n"
)


def generate_dxdt_lines(
    equations: ParsedEquations,
    index_map: Optional[IndexedBases] = None,
    cse: bool = True,
) -> list[str]:
    """Generate CUDA assignment statements for ``dx/dt`` updates.

    Parameters
    ----------
    equations
        Parsed equations describing ``dx/dt`` assignments.
    index_map
        Indexed bases that supply CUDA array references for each symbol.
    cse
        Whether to apply common subexpression elimination before emission.

    Returns
    -------
    list of str
        CUDA source lines that evaluate the ``dx/dt`` equations.
    """
    sysir = system_ir(equations, index_map)
    working_equations = sysir.non_observable_equations()

    if cse:
        processed = cse_and_stack(working_equations)
    else:
        processed = topological_sort(working_equations)

    observable_symbols = sysir.observable_set
    processed = [
        (lhs, rhs)
        for lhs, rhs in processed
        if lhs not in observable_symbols
    ]
    processed = prune_unused(
        processed, output_symbols=sysir.dxdt_symbols
    )
    processed = inline_cheap_assignments(
        processed, protect=sysir.dxdt_symbols
    )

    dxdt_lines = print_cuda_multiple(
        processed,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    if not dxdt_lines:
        dxdt_lines = ["pass"]
    return dxdt_lines


def generate_observables_lines(
    equations: ParsedEquations,
    index_map: IndexedBases,
    cse: bool = True,
) -> list[str]:
    """Generate CUDA source for observable calculations.

    Parameters
    ----------
    equations
        Parsed equations describing observable assignments.
    index_map
        Indexed bases used to substitute CUDA array references.
    cse
        Whether to apply common subexpression elimination before emission.

    Returns
    -------
    list of str
        CUDA source lines that compute the observables.
    """
    # Early return if no observables
    if not index_map.observables.ref_map:
        return ["pass"]

    sysir = system_ir(equations, index_map)
    working_equations = list(sysir.equations)

    if cse:
        processed = cse_and_stack(working_equations)
    else:
        processed = topological_sort(working_equations)

    # dx/dt outputs are not written by the observables kernel; route
    # them to throwaway locals instead of the out array.
    out_subs = {
        dx_sym: ir.sym(f"_cubie_codegen_dxout_{position + 1}")
        for position, dx_sym in enumerate(sysir.dxdt_symbols)
    }
    memo: dict = {}
    substituted = [
        (
            ir.xreplace(lhs, out_subs, memo),
            ir.xreplace(rhs, out_subs, memo),
        )
        for lhs, rhs in processed
    ]

    observable_targets = [
        sysir.arrayrefs[obs.name] for obs in sysir.observable_symbols
    ]
    substituted = prune_unused(
        substituted,
        output_symbols=list(sysir.observable_symbols)
        + observable_targets,
    )
    substituted = inline_cheap_assignments(
        substituted,
        protect=list(sysir.observable_symbols) + observable_targets,
    )
    obs_lines = print_cuda_multiple(
        substituted,
        symbol_map=sysir.arrayrefs,
        constant_names=sysir.constant_names,
        function_aliases=sysir.function_aliases,
    )
    if not obs_lines:
        obs_lines = ["pass"]
    return obs_lines


def generate_dxdt_fac_code(
    equations: ParsedEquations,
    index_map: Optional[IndexedBases] = None,
    func_name: str = "dxdt_factory",
    cse: bool = True,
) -> str:
    """Emit Python source for a ``dx/dt`` CUDA factory.

    Parameters
    ----------
    equations
        Parsed equations describing ``dx/dt`` assignments.
    index_map
        Indexed bases that provide both symbol references and constants.
    func_name
        Name of the generated factory function.
    cse
        Whether to apply common subexpression elimination before emission.

    Returns
    -------
    str
        Python source code implementing the requested factory.

    Notes
    -----
    The generated factory expects ``func(constants, precision)`` and returns a
    CUDA device function compiled with :func:`numba.cuda.jit`.
    """
    default_timelogger.start_event("codegen_generate_dxdt_fac_code")
    dxdt_lines = generate_dxdt_lines(
        equations, index_map=index_map, cse=cse
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )

    code = DXDT_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        body="    " + "\n        ".join(dxdt_lines),
    )
    default_timelogger.stop_event("codegen_generate_dxdt_fac_code")
    return code


INV_MASS_F_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED INVERSE-MASS RHS FACTORY\n"
    "def {func_name}(constants, precision, lineinfo=None):\n"
    '    """Auto-generated effective-derivative factory.\n'
    "    Computes out = M**-1 @ f(state, t).\n"
    '    """\n'
    "{const_lines}"
    "    \n"
    "    @cuda.jit(\n"
    "        # (precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision[::1],\n"
    "        #  precision),\n"
    "        device=True,\n"
    "        inline=True,\n"
    "        **get_jit_kwargs(lineinfo))\n"
    "    def evaluate_inv_mass_f("
    "state, parameters, drivers, observables, out, t):\n"
    "    {body}\n"
    "    \n"
    "    return evaluate_inv_mass_f\n"
)


def generate_evaluate_inv_mass_f_code(
    equations: ParsedEquations,
    index_map: Optional[IndexedBases] = None,
    M=None,
    func_name: str = "evaluate_inv_mass_f_factory",
    cse: bool = True,
) -> str:
    """Emit a factory computing ``out = M**-1 @ f(state, t)``.

    An identity mass matrix emits the plain ``dx/dt`` body.
    """
    default_timelogger.start_event(
        "codegen_generate_evaluate_inv_mass_f_code"
    )
    if mass_matrix_is_identity(M):
        lines = generate_dxdt_lines(
            equations, index_map=index_map, cse=cse
        )
    else:
        sysir = system_ir(equations, index_map)
        working_equations = sysir.non_observable_equations()
        if cse:
            processed = cse_and_stack(working_equations)
        else:
            processed = topological_sort(working_equations)
        observable_symbols = sysir.observable_set
        processed = [
            (lhs, rhs)
            for lhs, rhs in processed
            if lhs not in observable_symbols
        ]

        # Route raw derivatives to locals; out gets M**-1 @ f.
        f_subs = {
            dx_sym: ir.sym(f"_cubie_codegen_f_{position}")
            for position, dx_sym in enumerate(sysir.dxdt_symbols)
        }
        memo: dict = {}
        exprs = [
            (
                ir.xreplace(lhs, f_subs, memo),
                ir.xreplace(rhs, f_subs, memo),
            )
            for lhs, rhs in processed
        ]

        n = len(sysir.dxdt_symbols)
        inverse = mass_matrix_inverse_ir(M, n)
        for i, dx_sym in enumerate(sysir.dxdt_symbols):
            terms = []
            for j, source_sym in enumerate(sysir.dxdt_symbols):
                entry = inverse[i][j]
                if ir.is_zero(entry):
                    continue
                term = f_subs[source_sym]
                if not ir.is_one(entry):
                    term = ir.mul(entry, term)
                terms.append(term)
            row = ir.add(*terms) if terms else ir.ZERO
            exprs.append((dx_sym, row))

        exprs = prune_unused(
            exprs, output_symbols=sysir.dxdt_symbols
        )
        lines = print_cuda_multiple(
            exprs,
            symbol_map=sysir.arrayrefs,
            constant_names=sysir.constant_names,
            function_aliases=sysir.function_aliases,
        )
        if not lines:
            lines = ["pass"]

    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )
    code = INV_MASS_F_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        body="    " + "\n        ".join(lines),
    )
    default_timelogger.stop_event(
        "codegen_generate_evaluate_inv_mass_f_code"
    )
    return code


def generate_observables_fac_code(
    equations: ParsedEquations,
    index_map: IndexedBases,
    func_name: str = "observables",
    cse: bool = True,
) -> str:
    """Emit Python source for an observables CUDA factory.

    Parameters
    ----------
    equations
        Parsed equations describing observable assignments.
    index_map
        Indexed bases that provide symbol and constant lookups.
    func_name
        Name of the generated factory function.
    cse
        Whether to apply common subexpression elimination before emission.

    Returns
    -------
    str
        Python source code implementing the requested factory.
    """
    default_timelogger.start_event("codegen_generate_observables_fac_code")

    obs_lines = generate_observables_lines(
        equations, index_map=index_map, cse=cse
    )
    const_block = render_constant_assignments(
        index_map.constants.symbol_map
    )

    code = OBSERVABLES_TEMPLATE.format(
        func_name=func_name,
        const_lines=const_block,
        body="    " + "\n        ".join(obs_lines),
    )
    default_timelogger.stop_event("codegen_generate_observables_fac_code")
    return code
