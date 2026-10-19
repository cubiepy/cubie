"""Build the IR data used by source generators."""

from functools import lru_cache
from types import MappingProxyType
from typing import Dict, List, Mapping, Tuple

import attrs

from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.from_sympy import (
    convert_assignments,
    derivative_name_map,
)

__all__ = ["BufferMaps", "SystemIR", "buffer_maps", "system_ir"]

STATE_BUFFER = "state"
PARAMETER_BUFFER = "parameters"
OBSERVABLE_BUFFER = "observables"
DRIVER_BUFFER = "drivers"
DXDT_BUFFER = "out"


@attrs.frozen(eq=False)
class BufferMaps:
    """The positions and array elements we read in generated code.

    Parameters
    ----------
    state_order, dxdt_order, driver_order
        Position of each state, derivative and driver symbol in its
        buffer.
    observable_symbols
        Observable symbols, in buffer order.
    arrayrefs
        The array element we read for each name.
    """

    state_order: Mapping[ir.Sym, int]
    dxdt_order: Mapping[ir.Sym, int]
    driver_order: Mapping[ir.Sym, int]
    observable_symbols: Tuple[ir.Sym, ...]
    arrayrefs: Mapping[str, ir.Expr]


def _order(names: Tuple[str, ...]) -> Mapping[ir.Sym, int]:
    """Map each name's IR symbol to its position."""
    return MappingProxyType(
        {ir.sym(name): index for index, name in enumerate(names)}
    )


@lru_cache(maxsize=64)
def buffer_maps(layout: SystemLayout) -> BufferMaps:
    """Return the positions and array elements of ``layout``.

    We build them once for each layout, because every generator reads
    them and a layout never changes.

    Parameters
    ----------
    layout
        The buffer layout.

    Returns
    -------
    BufferMaps
        The read-only maps.
    """
    refs = {}
    for buffer, names in (
        (STATE_BUFFER, layout.state_names),
        (PARAMETER_BUFFER, layout.swept_names),
        (OBSERVABLE_BUFFER, layout.observable_names),
        (DRIVER_BUFFER, layout.driver_names),
        (DXDT_BUFFER, layout.dxdt_names),
    ):
        for index, name in enumerate(names):
            refs[name] = ir.arr(buffer, index)
    return BufferMaps(
        state_order=_order(layout.state_names),
        dxdt_order=_order(layout.dxdt_names),
        driver_order=_order(layout.driver_names),
        observable_symbols=tuple(
            ir.sym(name) for name in layout.observable_names
        ),
        arrayrefs=MappingProxyType(refs),
    )


@attrs.frozen(eq=False)
class SystemIR:
    """IR view of a parsed system.

    Parameters
    ----------
    equations
        All equations in evaluation order as IR pairs.
    state_symbols
        State symbols ordered by state index.
    dxdt_symbols
        Derivative-output symbols ordered by output index.
    observable_symbols
        Observable symbols ordered by observable index.
    driver_symbols
        Drivers-buffer symbols in order: drivers, then their
        derivatives.
    state_index, dxdt_index, driver_index
        Symbol-to-position lookups for the ordered collections.
    arrayrefs
        Printer symbol map: scalar name to :class:`~.expr.Arr`.
    function_aliases
        Renamed user-function name to printable original name.
    nonfloat_functions
        IR call names printed without a precision cast.
    derivative_names
        User-function name to derivative placeholder print name.
    time_symbol
        The IR time symbol (``t``).
    """

    equations: Tuple[Tuple[ir.Expr, ir.Expr], ...]
    state_symbols: Tuple[ir.Sym, ...]
    dxdt_symbols: Tuple[ir.Sym, ...]
    observable_symbols: Tuple[ir.Sym, ...]
    driver_symbols: Tuple[ir.Sym, ...]
    state_index: Dict[ir.Sym, int]
    dxdt_index: Dict[ir.Sym, int]
    driver_index: Dict[ir.Sym, int]
    arrayrefs: Dict[str, ir.Expr]
    function_aliases: Dict[str, str]
    nonfloat_functions: frozenset
    derivative_names: Dict[str, str]
    time_symbol: ir.Sym

    @property
    def observable_set(self) -> frozenset:
        """Return the observable symbols as a frozenset."""
        return frozenset(self.observable_symbols)

    def non_observable_equations(
        self,
    ) -> List[Tuple[ir.Expr, ir.Expr]]:
        """Return equations whose outputs are not observables."""
        observables = self.observable_set
        return [
            (lhs, rhs)
            for lhs, rhs in self.equations
            if lhs not in observables
        ]


def system_ir(equations, layout: SystemLayout) -> SystemIR:
    """Return the current IR view of a parsed system.

    Parameters
    ----------
    equations
        ``ParsedEquations`` from the parser.
    layout
        The buffer layout of the same system.

    Returns
    -------
    SystemIR
        Converted equations and symbol tables.
    """
    memo: Dict = {}
    eq_list = equations.to_equation_list()
    if eq_list and isinstance(eq_list[0][0], ir.Expr):
        # Parser output is already IR; the pairs pass through and
        # derivative placeholder names ride on the container.
        eq_pairs = tuple(eq_list)
        derivative_names = dict(
            getattr(equations, "derivative_names", None) or {}
        )
    else:
        eq_pairs = tuple(convert_assignments(eq_list, memo))
        derivative_names = derivative_name_map(eq_list)

    maps = buffer_maps(layout)
    states = tuple(maps.state_order)
    state_index = dict(maps.state_order)
    dxdt = tuple(maps.dxdt_order)
    dxdt_index = dict(maps.dxdt_order)
    # Place driver derivatives after the driver symbols.
    driver_derivatives = tuple(
        getattr(equations, "driver_derivatives", None) or {}
    )
    drivers = tuple(maps.driver_order) + driver_derivatives
    driver_index = {symbol: pos for pos, symbol in enumerate(drivers)}
    observables = maps.observable_symbols

    arrayrefs: Dict[str, ir.Expr] = dict(maps.arrayrefs)
    aliases = dict(getattr(equations, "function_aliases", None) or {})
    for symbol in driver_derivatives:
        arrayrefs[symbol.name] = ir.arr("drivers", driver_index[symbol])
    for derivative_name in derivative_names.values():
        aliases.setdefault(derivative_name, derivative_name)

    return SystemIR(
        equations=eq_pairs,
        state_symbols=states,
        dxdt_symbols=dxdt,
        observable_symbols=observables,
        driver_symbols=drivers,
        state_index=state_index,
        dxdt_index=dxdt_index,
        driver_index=driver_index,
        arrayrefs=arrayrefs,
        function_aliases=aliases,
        nonfloat_functions=frozenset(
            getattr(equations, "nonfloat_functions", None) or ()
        ),
        derivative_names=derivative_names,
        time_symbol=ir.sym("t"),
    )
