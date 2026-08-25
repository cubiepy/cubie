"""Dependent derivative rows rewritten as algebraic constraints.

Published Functions
-------------------
:func:`eliminate_singular_derivative_blocks`
    Replace each equation whose derivative terms are a combination of
    earlier equations' derivative terms by the derivative-free
    equation that combination implies.
"""

from typing import Dict, List, Tuple

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.structural.symbolics import (
    linear_dependencies,
    linear_expansion,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
    StructuralState,
)


def _derivative_rows(
    state: StructuralState,
) -> List[Tuple[int, Dict[int, ir.Expr], ir.Expr]]:
    """Return ``(equation, {derivative: coefficient}, remainder)`` for
    each equation linear in its derivatives with unknown-free
    coefficients; coefficients carry exact rational literals."""

    structure = state.structure
    graph = structure.graph
    unknowns = set(state.var2idx) | {state.time_symbol}
    rows = []
    for ieq in range(graph.nsrcs()):
        dvars = [
            v for v in graph.s_neighbors(ieq) if structure.isdervar(v)
        ]
        if not dvars:
            continue
        term = state.eqs[ieq].residual()
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
    return rows


def eliminate_singular_derivative_blocks(state: StructuralState) -> List[int]:
    """Replace dependent derivative rows by the constraint
    ``0 ~ sum(lambda_e * remainder_e)``; returns their indices."""

    rows = _derivative_rows(state)
    if len(rows) < 2:
        return []
    equations = [ieq for ieq, _, _ in rows]
    remainders = {ieq: remainder for ieq, _, remainder in rows}
    graph = state.structure.graph
    rewritten = []
    dependent = linear_dependencies([coeffs for _, coeffs, _ in rows])
    for position, multipliers in dependent:
        ieq = equations[position]
        terms = [
            ir.mul(weight, remainders[equations[source]])
            for source, weight in sorted(multipliers.items())
        ]
        constraint = Equation(ir.ZERO, ir.add(*terms))
        state.eqs[ieq] = constraint
        state.original_eqs[ieq] = constraint
        incidence = [
            state.var2idx[symbol]
            for symbol in constraint.free_symbols()
            if symbol in state.var2idx
        ]
        graph.set_neighbors(ieq, incidence)
        rewritten.append(ieq)
    return rewritten
