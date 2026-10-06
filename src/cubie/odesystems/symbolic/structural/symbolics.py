"""Engine-IR primitives backing the structural simplification passes.

Structural linear expansion, fixpoint substitution,
total time derivatives over derivative-symbol maps, linear dependencies
among rows of a symbolic matrix, and the derivative-symbol registry
that stands in for ``Differential`` terms (cubie states are plain IR
symbols, so derivatives are represented by registered companion
symbols).

All expressions are engine IR nodes
(:mod:`cubie.odesystems.symbolic.engine`); SymPy input converts to IR
at the parse boundary before any of these primitives run.

Published Classes
-----------------
:class:`DerivativeRegistry`
    Creates and tracks derivative symbols (``x`` -> ``x_t`` ->
    ``x_tt`` ...) with collision-safe naming; the record of every
    variable's derivative chain.

Published Functions
-------------------
:func:`linear_expansion`
    Decompose ``expr`` as ``a*var + b`` with ``a``, ``b`` free of
    ``var``, or report nonlinearity.

:func:`fixpoint_sub`
    Structural substitution applied until a fixed point.

:func:`total_derivative`
    Total time derivative of an expression under a derivative map.

:func:`linear_dependencies`
    Rows of a sparse symbolic matrix that the pivot rows span.
"""

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from cubie.odesystems.symbolic.engine import expr as ir

ZERO = ir.ZERO
ONE = ir.ONE


def linear_expansion(
    expr: ir.Expr, var: ir.Sym
) -> Tuple[ir.Expr, ir.Expr, bool]:
    """Decompose ``expr`` into ``a*var + b`` when possible.

    Returns ``(a, b, islinear)``. When ``islinear`` is true, ``a`` and
    ``b`` contain no occurrence of ``var`` and
    ``expr == a*var + b`` holds structurally. Mirrors
    ``Symbolics.linear_expansion``: expressions where ``var`` appears
    inside a nonlinear or non-polynomial construct report
    ``islinear = False``.
    """

    if expr is var:
        return (ONE, ZERO, True)
    if var not in ir.free_atoms(expr):
        return (ZERO, expr, True)
    if isinstance(expr, ir.Add):
        a_terms: List[ir.Expr] = []
        b_terms: List[ir.Expr] = []
        for arg in expr.args:
            a, b, islinear = linear_expansion(arg, var)
            if not islinear:
                return (ZERO, ZERO, False)
            a_terms.append(a)
            b_terms.append(b)
        return (ir.add(*a_terms), ir.add(*b_terms), True)
    if isinstance(expr, ir.Mul):
        var_factor = None
        rest: List[ir.Expr] = []
        for arg in expr.args:
            if var in ir.free_atoms(arg):
                if var_factor is not None:
                    return (ZERO, ZERO, False)
                var_factor = arg
            else:
                rest.append(arg)
        a, b, islinear = linear_expansion(var_factor, var)
        if not islinear:
            return (ZERO, ZERO, False)
        rest_prod = ir.mul(*rest)
        return (ir.mul(a, rest_prod), ir.mul(b, rest_prod), True)
    # Pow, calls, piecewise, ... containing var: nonlinear.
    return (ZERO, ZERO, False)


def fixpoint_sub(
    expr: ir.Expr,
    sub_map: Dict[ir.Expr, ir.Expr],
    memo: Optional[Dict[ir.Expr, ir.Expr]] = None,
) -> ir.Expr:
    """Apply structural substitution until the expression stabilises.

    Substitution keys are plain symbols, so :func:`~.expr.xreplace`
    (exact-node replacement) is applied repeatedly until no key
    remains reachable. ``memo`` caches single-pass replacements under
    ``sub_map`` across calls.

    Raises
    ------
    ValueError
        When the substitutions reach ``expr`` through a cycle.
    """

    if memo is None:
        memo = {}
    for _ in range(len(sub_map) + 1):
        new_expr = ir.xreplace(expr, sub_map, memo)
        if new_expr is expr:
            return expr
        expr = new_expr
    raise ValueError("the substitutions contain a cycle")


def total_derivative(
    expr: ir.Expr,
    deriv_map: Dict[ir.Sym, ir.Sym],
    time_symbol: ir.Sym,
    derivative_names: Optional[Dict[str, str]] = None,
) -> ir.Expr:
    """Total time derivative of ``expr``.

    Parameters
    ----------
    expr
        Expression to differentiate.
    deriv_map
        Map from unknown symbols to their derivative symbols. Symbols
        absent from the map differentiate to zero, matching MTK's
        default for time-dependent parameters.
    time_symbol
        The independent variable; explicit dependence differentiates
        through :func:`~.expr.diff`.
    derivative_names
        Name of each user function, and of each derivative helper, to
        the name of the helper for its next derivative.

    Notes
    -----
    Computed as ``sum(diff(expr, v) * dv) + diff(expr, t)`` over the
    mapped symbols occurring in ``expr``.
    """

    terms: List[ir.Expr] = [
        ir.diff(expr, time_symbol, derivative_names=derivative_names)
    ]
    atoms = sorted(ir.free_atoms(expr), key=lambda a: a.sort_key)
    for atom in atoms:
        if atom is time_symbol:
            continue
        dsym = deriv_map.get(atom)
        if dsym is not None:
            partial = ir.diff(
                expr, atom, derivative_names=derivative_names
            )
            terms.append(ir.mul(partial, dsym))
    return ir.add(*terms)


def _combine(
    target: Dict[int, ir.Expr],
    pivot: ir.Expr,
    entry: ir.Expr,
    source: Dict[int, ir.Expr],
    expand: bool,
) -> None:
    """Update ``target = pivot*target - entry*source``, dropping zeros."""

    for column in set(target) | set(source):
        value = ir.sub(
            ir.mul(pivot, target.get(column, ZERO)),
            ir.mul(entry, source.get(column, ZERO)),
        )
        if ir.is_zero(ir.expand(value) if expand else value):
            target.pop(column, None)
        else:
            target[column] = value


def _inverse(node: ir.Expr) -> ir.Expr:
    """Return ``1/node`` factor by factor so shared factors cancel."""

    if isinstance(node, ir.Mul):
        return ir.mul(*(ir.pow_(arg, -1) for arg in node.args))
    return ir.pow_(node, -1)


def linear_dependencies(
    rows: List[Dict[int, ir.Expr]],
    pivot_ok: Callable[[ir.Expr], bool],
) -> List[Tuple[int, Dict[int, ir.Expr]]]:
    """Return ``(row, {row: weight})`` for each row the pivot rows span."""

    reduced = []
    for row in rows:
        entries = {}
        for column, entry in row.items():
            if not ir.is_zero(ir.expand(entry)):
                entries[column] = entry
        reduced.append(entries)
    multipliers = [{index: ONE} for index in range(len(rows))]
    free = list(range(len(rows)))
    for column in sorted({c for row in reduced for c in row}):
        candidates = [i for i in free if column in reduced[i]]
        pivot_row = next(
            (i for i in candidates if isinstance(reduced[i][column], ir.Num)),
            None,
        )
        if pivot_row is None:
            pivot_row = next(
                (i for i in candidates if pivot_ok(reduced[i][column])),
                None,
            )
        if pivot_row is None:
            continue
        free.remove(pivot_row)
        pivot = reduced[pivot_row][column]
        for other in free:
            entry = reduced[other].get(column)
            if entry is None:
                continue
            _combine(reduced[other], pivot, entry, reduced[pivot_row], True)
            _combine(
                multipliers[other],
                pivot,
                entry,
                multipliers[pivot_row],
                False,
            )
    dependent = []
    for index in free:
        if reduced[index]:
            continue
        own = _inverse(multipliers[index][index])
        weights = {
            source: ir.mul(weight, own)
            for source, weight in multipliers[index].items()
        }
        dependent.append((index, weights))
    return dependent


class DerivativeRegistry:
    """Factory and index for derivative symbols.

    Derivative symbols stand in for MTK's ``Differential`` terms and
    are plain IR symbols named ``x_t``, ``x_tt``, ... after their
    base, with an underscore appended until the name is free in
    ``reserved``. The registry records each symbol's lower and higher
    order, and is the record of every variable's derivative chain.

    Parameters
    ----------
    reserved_names
        Names that generated symbols must not collide with (every
        name in the user's input).
    """

    def __init__(self, reserved_names: Iterable[str]) -> None:
        self.reserved = set(reserved_names)
        self._to_base = {}
        self._to_derivative = {}

    def derivative(self, var: ir.Sym) -> ir.Sym:
        """Return (creating if needed) the derivative symbol of ``var``."""

        existing = self._to_derivative.get(var)
        if existing is not None:
            return existing
        base, order = self.base_and_order(var)
        name = f"{base.name}_{'t' * (order + 1)}"
        while name in self.reserved:
            name = name + "_"
        dsym = ir.sym(name)
        self.reserved.add(name)
        self._to_derivative[var] = dsym
        self._to_base[dsym] = var
        return dsym

    def lower_order(self, var: ir.Sym) -> Optional[ir.Sym]:
        """Return the symbol ``var`` is the derivative of, if any."""

        return self._to_base.get(var)

    def higher_order(self, var: ir.Sym) -> Optional[ir.Sym]:
        """Return the derivative symbol of ``var``, if registered."""

        return self._to_derivative.get(var)

    def base_and_order(self, var: ir.Sym) -> Tuple[ir.Sym, int]:
        """Return the underived base symbol and derivative order."""

        order = 0
        base = var
        while True:
            lower = self._to_base.get(base)
            if lower is None:
                return (base, order)
            base = lower
            order += 1

    def is_derivative(self, var: ir.Expr) -> bool:
        """Whether ``var`` is a registered derivative symbol."""

        return var in self._to_base

    def copy(self) -> "DerivativeRegistry":
        """Return an independent copy of the registry."""

        duplicate = DerivativeRegistry(self.reserved)
        duplicate._to_base = dict(self._to_base)
        duplicate._to_derivative = dict(self._to_derivative)
        return duplicate

    def cut(self, var: ir.Sym) -> None:
        """Make the derivative symbol ``var`` an ordinary chain root.

        ``var`` keeps its name and its own higher derivatives; the
        symbol it derived has no derivative afterwards (MTK's
        ``diff2term``).
        """

        lower = self._to_base.pop(var)
        del self._to_derivative[lower]

    def move_derivative(self, source: ir.Sym, target: ir.Sym) -> None:
        """Make the derivative of ``source`` the derivative of ``target``.

        ``source`` is left without a derivative, and any previous
        derivative of ``target`` without a base.
        """

        previous = self._to_derivative.pop(target, None)
        if previous is not None:
            del self._to_base[previous]
        upper = self._to_derivative.pop(source, None)
        if upper is not None:
            self._to_derivative[target] = upper
            self._to_base[upper] = target
