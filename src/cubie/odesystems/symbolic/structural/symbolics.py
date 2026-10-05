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
    ``x_tt`` ...) with collision-safe naming.

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

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

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
    maxiters: Optional[int] = None,
) -> ir.Expr:
    """Apply structural substitution until the expression stabilises.

    Substitution keys are plain symbols so :func:`~.expr.xreplace`
    (exact-node replacement) is sufficient; it is applied repeatedly
    because substituted expressions may themselves contain keys.
    """

    if not sub_map:
        return expr
    if maxiters is None:
        maxiters = len(sub_map) + 10
    for _ in range(maxiters):
        new_expr = ir.xreplace(expr, sub_map)
        if new_expr is expr:
            return new_expr
        expr = new_expr
    raise ValueError(
        "fixpoint substitution failed to converge; the substitution "
        "map is likely cyclic"
    )


def total_derivative(
    expr: ir.Expr,
    deriv_map: Dict[ir.Sym, ir.Sym],
    time_symbol: ir.Sym,
    known_derivative_map: Optional[Dict[ir.Sym, ir.Expr]] = None,
) -> ir.Expr:
    """Total time derivative of ``expr``.

    Parameters
    ----------
    expr
        Expression to differentiate.
    deriv_map
        Map from unknown symbols to their derivative symbols. Unknowns
        absent from the map cannot be differentiated and raise.
    time_symbol
        The independent variable; explicit dependence differentiates
        through :func:`~.expr.diff`.
    known_derivative_map
        Derivative expressions for known time-dependent quantities
        (drivers). Known symbols absent from this map differentiate
        to zero, matching MTK's default for time-dependent
        parameters.

    Notes
    -----
    Computed as ``sum(diff(expr, v) * dv) + diff(expr, t)`` over the
    mapped symbols occurring in ``expr``.
    """

    if known_derivative_map is None:
        known_derivative_map = {}
    terms: List[ir.Expr] = [ir.diff(expr, time_symbol)]
    atoms = sorted(ir.free_atoms(expr), key=lambda a: a.sort_key)
    for atom in atoms:
        if atom is time_symbol:
            continue
        dsym = deriv_map.get(atom)
        if dsym is not None:
            terms.append(ir.mul(ir.diff(expr, atom), dsym))
            continue
        known = known_derivative_map.get(atom)
        if known is not None:
            terms.append(ir.mul(ir.diff(expr, atom), known))
        # Symbols that are neither unknowns nor known time-dependent
        # quantities are constants/parameters: derivative zero.
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


def lower_varname(
    base_name: str, order: int, reserved: Set[str]
) -> str:
    """User-visible name for a dummy-derivative variable.

    Produces ``x_t``, ``x_tt``, ... (the plain-symbol analogue of
    MTK's ``xˍt`` naming), appending underscores until the name is
    free in ``reserved``. The chosen name is added to ``reserved``.
    """

    name = f"{base_name}_{'t' * order}"
    while name in reserved:
        name = name + "_"
    reserved.add(name)
    return name


class DerivativeRegistry:
    """Factory and index for derivative symbols.

    Derivative symbols stand in for MTK's ``Differential`` terms and
    are plain IR symbols with mangled internal names
    (``_cubie_D<order>_<base>``); they are renamed to user-visible
    ``x_t`` forms during reassembly when state selection turns them
    into ordinary algebraic variables. The registry records
    base/order relations so higher-order chains can be walked
    symbolically as well as through the integer
    :class:`~cubie.odesystems.symbolic.structural.diffgraph.DiffGraph`.

    Parameters
    ----------
    reserved_names
        Names that generated symbols must not collide with (all user
        symbols in the system).
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
        name = f"_cubie_D{order + 1}_{base.name}"
        while name in self.reserved:
            name = name + "_"
        dsym = ir.sym(name)
        self.reserved.add(name)
        self._to_derivative[var] = dsym
        self._to_base[dsym] = var
        return dsym

    def lower_order(self, var: ir.Sym) -> Optional[ir.Sym]:
        """Return the symbol ``var`` is the derivative of, if known."""

        return self._to_base.get(var)

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

    def rename(self, old: ir.Sym, new: ir.Sym) -> None:
        """Rebind a registered derivative symbol to a new symbol.

        The renamed variable becomes an ordinary chain root (its link
        to the variable it derived is cut, mirroring MTK's
        ``diff2term``); a higher derivative of ``old``, if any, is
        rebased onto ``new``.
        """

        lower = self._to_base.pop(old, None)
        if lower is not None and self._to_derivative.get(lower) is old:
            del self._to_derivative[lower]
        upper = self._to_derivative.pop(old, None)
        if upper is not None:
            self._to_derivative[new] = upper
            self._to_base[upper] = new
        self.reserved.add(new.name)
