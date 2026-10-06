"""Dependent derivative rows rewritten as algebraic constraints.

An equation whose derivative terms are an exact combination of other
equations' derivative terms is replaced by the derivative-free
equation that combination implies.

Published Functions
-------------------
:func:`eliminate_singular_derivative_blocks`
    Replace dependent derivative rows by the constraint they imply.
"""

from typing import Dict, List, Tuple

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.structural.symbolics import (
    linear_dependencies,
    linear_expansion,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)


def _derivative_rows(
    state: StructuralState,
) -> List[Tuple[int, Dict[int, ir.Expr], ir.Expr]]:
    """Return ``(equation, {derivative: coefficient}, remainder)`` for
    each equation linear in its derivatives with known coefficients."""

    graph = state.graph
    unknowns = set(state.var2idx) | {state.time_symbol}
    rows = []
    for ieq in range(graph.nsrcs()):
        dvars = [
            v
            for v in graph.s_neighbors(ieq)
            if state.primal_of(v) is not None
        ]
        if not dvars:
            continue
        lhs, rhs = state.eqs[ieq]
        term = ir.sub(rhs, lhs)
        coeffs = {}
        known = True
        for v in dvars:
            a, b, islinear = linear_expansion(term, state.fullvars[v])
            if not islinear or ir.free_atoms(a) & unknowns:
                known = False
                break
            if not ir.is_zero(a):
                coeffs[v] = ir.rationalize(a)
            term = b
        if not known or not coeffs:
            continue
        rows.append((ieq, coeffs, term))
    # Rows reading fewest algebraic unknowns become pivots.
    rows.sort(key=lambda row: _algebraic_count(state, row[2]))
    return rows


def _algebraic_count(state: StructuralState, term: ir.Expr) -> int:
    """Count the unknowns in ``term`` that have no derivative."""
    count = 0
    for atom in ir.free_atoms(term):
        index = state.var2idx.get(atom)
        if (
            index is not None
            and state.derivative_of(index) is None
            and state.primal_of(index) is None
        ):
            count += 1
    return count


def eliminate_singular_derivative_blocks(
    state: StructuralState,
    allow_symbolic: bool = False,
    allow_parameter: bool = True,
    **_ignored,
) -> List[int]:
    """Rewrite dependent derivative rows as constraints; returns indices."""

    rows = _derivative_rows(state)
    if len(rows) < 2:
        return []
    equations = [ieq for ieq, _, _ in rows]
    remainders = {ieq: remainder for ieq, _, remainder in rows}
    graph = state.graph
    rewritten = []

    def pivot_ok(entry: ir.Expr) -> bool:
        return state.division_permitted(
            entry, allow_symbolic, allow_parameter
        )

    dependent = linear_dependencies(
        [coeffs for _, coeffs, _ in rows], pivot_ok
    )
    for position, multipliers in dependent:
        ieq = equations[position]
        terms = [
            ir.mul(weight, remainders[equations[source]])
            for source, weight in sorted(multipliers.items())
        ]
        constraint = (ir.ZERO, ir.add(*terms))
        state.eqs[ieq] = constraint
        state.original_eqs[ieq] = constraint
        graph.set_neighbors(ieq, state.incidence(constraint))
        rewritten.append(ieq)
    return rewritten
