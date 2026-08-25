"""Exact removal of numerically singular derivative blocks.

Structural analysis sees only incidence, so a block of equations
whose derivative coefficients are dependent (a singular non-diagonal
mass matrix such as ``-c*D(y1) + c*D(y2) ~ f1`` paired with
``c*D(y1) - c*D(y2) ~ f2``) looks full rank and is matched as index
1. This pass detects the dependence exactly and rewrites each
dependent row as the derivative-free combination it implies, so the
constraint is visible to Pantelides and gets differentiated.

Coefficients are handled as rational combinations of symbolic
monomials: numeric parts use exact ``Fraction`` arithmetic and
parameter parts cancel structurally through the IR's power folding.

Published Functions
-------------------
:func:`eliminate_singular_derivative_blocks`
    Replace dependent derivative rows by their implied algebraic
    constraints.
"""

from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.structural.symbolics import (
    linear_expansion,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
    StructuralState,
)

Combination = Dict[ir.Expr, Fraction]
"""Exact linear combination of monomials, keyed by monomial node."""


def _exact(value) -> Fraction:
    """Return the exact rational value of a numeric payload."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(float(value))


def _split_term(term: ir.Expr) -> Tuple[Fraction, ir.Expr]:
    """Split ``term`` into its exact numeric weight and monomial."""

    if isinstance(term, ir.Num):
        return _exact(term.value), ir.ONE
    if isinstance(term, ir.Mul) and isinstance(term.args[0], ir.Num):
        return _exact(term.args[0].value), ir.mul(*term.args[1:])
    return Fraction(1), term


def _as_combination(expr: ir.Expr) -> Combination:
    """Decompose ``expr`` into weighted monomials."""

    terms = expr.args if isinstance(expr, ir.Add) else (expr,)
    combination = {}
    for term in terms:
        weight, monomial = _split_term(term)
        _accumulate(combination, monomial, weight)
    return combination


def _accumulate(
    combination: Combination, monomial: ir.Expr, weight: Fraction
) -> None:
    """Add ``weight * monomial`` into ``combination`` in place."""

    updated = combination.get(monomial, Fraction(0)) + weight
    if updated == 0:
        combination.pop(monomial, None)
    else:
        combination[monomial] = updated


def _product(a: Combination, b: Combination) -> Combination:
    """Return the exact product of two combinations."""

    result = {}
    for monomial_a, weight_a in a.items():
        for monomial_b, weight_b in b.items():
            weight, monomial = _split_term(ir.mul(monomial_a, monomial_b))
            _accumulate(result, monomial, weight * weight_a * weight_b)
    return result


def _quotient(entry: Combination, pivot: Combination) -> Combination:
    """Return ``entry / pivot`` for a single-monomial ``pivot``."""

    ((pivot_monomial, pivot_weight),) = pivot.items()
    result = {}
    for monomial, weight in entry.items():
        scale, quotient = _split_term(ir.div(monomial, pivot_monomial))
        _accumulate(result, quotient, scale * weight / pivot_weight)
    return result


def _subtract_scaled(
    target: Combination, factor: Combination, source: Combination
) -> None:
    """Update ``target -= factor * source`` in place."""

    for monomial, weight in _product(factor, source).items():
        _accumulate(target, monomial, -weight)


def _to_expr(combination: Combination) -> ir.Expr:
    """Rebuild a combination as an IR expression with float weights."""

    return ir.add(
        *[
            ir.mul(ir.num(_payload(weight)), monomial)
            for monomial, weight in sorted(
                combination.items(), key=lambda item: item[0].sort_key
            )
        ]
    )


def _payload(weight: Fraction):
    """Return an integral weight as ``int``, otherwise a float."""

    if weight.denominator == 1:
        return int(weight)
    return float(weight)


def _derivative_rows(
    state: StructuralState,
) -> List[Tuple[int, Dict[int, Combination], ir.Expr]]:
    """Collect equations linear in their derivatives with known
    coefficients as ``(equation, {derivative: coefficient},
    remainder)`` triples in equation order."""

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
            coefficient = _as_combination(a)
            if coefficient:
                coeffs[v] = coefficient
            term = b
        if not known or not coeffs:
            continue
        rows.append((ieq, coeffs, term))
    return rows


def _pivot_column(row: Dict[int, Combination]) -> Optional[int]:
    """Return the lowest column holding a single-monomial entry."""

    for column in sorted(row):
        if len(row[column]) == 1:
            return column
    return None


def _dependent_rows(
    rows: List[Tuple[int, Dict[int, Combination], ir.Expr]],
) -> List[Tuple[int, Dict[int, Combination]]]:
    """Return ``(equation, multipliers)`` for each row whose derivative
    coefficients are an exact combination of earlier rows.

    ``multipliers`` maps equation indices to the factors whose
    weighted sum of derivative rows is exactly zero."""

    basis = []
    dependent = []
    for ieq, coeffs, _ in rows:
        reduced = {
            column: dict(entry) for column, entry in coeffs.items()
        }
        multipliers = {ieq: {ir.ONE: Fraction(1)}}
        for pivot, basis_row, basis_multipliers in basis:
            entry = reduced.get(pivot)
            if entry is None:
                continue
            factor = _quotient(entry, basis_row[pivot])
            for column, value in basis_row.items():
                target = reduced.setdefault(column, {})
                _subtract_scaled(target, factor, value)
                if not target:
                    del reduced[column]
            for source, value in basis_multipliers.items():
                target = multipliers.setdefault(source, {})
                _subtract_scaled(target, factor, value)
                if not target:
                    del multipliers[source]
        if not reduced:
            dependent.append((ieq, multipliers))
            continue
        pivot = _pivot_column(reduced)
        if pivot is not None:
            basis.append((pivot, reduced, multipliers))
    return dependent


def eliminate_singular_derivative_blocks(state: StructuralState) -> List[int]:
    """Rewrite dependent derivative rows as algebraic constraints.

    Every equation that is linear in its derivative variables with
    coefficients free of unknowns contributes a row of the derivative
    coefficient matrix. Rows that are exact combinations of earlier
    rows have their derivative terms cancelled by that combination;
    the equation is replaced in place by the resulting derivative-free
    constraint ``0 ~ sum(lambda_e * remainder_e)``.

    Returns the indices of the rewritten equations.
    """

    rows = _derivative_rows(state)
    if len(rows) < 2:
        return []
    remainders = {ieq: remainder for ieq, _, remainder in rows}
    rewritten = []
    graph = state.structure.graph
    for ieq, multipliers in _dependent_rows(rows):
        terms = [
            ir.mul(_to_expr(multiplier), remainders[source])
            for source, multiplier in sorted(multipliers.items())
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
