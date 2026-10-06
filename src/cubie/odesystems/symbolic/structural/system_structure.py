"""Structural state of a DAE system under simplification.

The equations and variables of the system, its bipartite incidence
graph, solvability analysis via linear expansion, the integer-linear
subsystem matrix, and the symbolic differentiation hooks used by
Pantelides. Equations are ``(lhs, rhs)`` pairs of engine IR
expressions.

Ported from ModelingToolkit.jl (commit c4177c335):
``StructuralState`` construction (algebraic-equation canonicalisation
and the equation order) from ``TearingState`` in
``src/systems/systemstructure.jl``, except the variable order;
``StructuralState.var_derivative`` and ``StructuralState.eq_derivative``
from ``src/structural_transformation/symbolics_tearing.jl``
(``var_derivative!``, ``eq_derivative!``);
``StructuralState._build_state_priorities`` from the
``state_priority`` closure of ``dummy_derivative`` in the same file;
``StructuralState.find_eq_solvables`` with
``StructuralState.division_permitted``,
``StructuralState.linear_subsys_adjmat`` and
``StructuralState.n_concrete_eqs`` from
``src/structural_transformation/utils.jl``;
``StructuralState.rm_eqs_vars`` and ``_old_to_new_indices`` from the
equation renumbering and graph rebuild in ``alias_elimination!``
(``src/systems/alias_elimination.jl``).
``StructuralState.is_unused_var`` combines the empty-incidence test of
ModelingToolkit.jl c4177c335 (``utils.jl``) with the
``always_present`` marking of ModelingToolkit.jl (commit a2b6dc56,
``src/systems/alias_elimination.jl``).

Published Classes
-----------------
:class:`StructuralState`
    Full transformation state: the symbolic equations, variables,
    derivative registry, incidence and solvability graphs, and the
    bookkeeping updated by the passes.

Published Functions
-------------------
:func:`variable_ranks`
    Rank of each variable by base name, then derivative order.
"""

from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from cubie.odesystems.symbolic.engine import expr as ir

from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    DST,
    SRC,
)
from cubie.odesystems.symbolic.structural.clil import SparseMatrixCLIL
from cubie.odesystems.symbolic.structural.diffgraph import DiffGraph
from cubie.odesystems.symbolic.structural.symbolics import (
    DerivativeRegistry,
    linear_expansion,
    total_derivative,
)

# Not ported: largest coefficient magnitude in the integer matrix.
MAX_INTEGER_COEFFICIENT = 127

# Per-variable lists, aligned with ``StructuralState.fullvars``.
_VARIABLE_FIELDS = (
    "fullvars",
    "state_priorities",
    "canonical_ranks",
    "always_present",
)


def _old_to_new_indices(n: int, dels: List[int]) -> Tuple[List[int], int]:
    """Map ``n`` old indices past the sorted ``dels``.

    Returns the new index of each old index (``-1`` for a deleted
    one) and the number of indices kept.
    """

    old_to_new = [0] * n
    idx = 0
    cursor = 0
    ndels = len(dels)
    for i in range(n):
        if cursor < ndels and i == dels[cursor]:
            cursor += 1
            old_to_new[i] = -1
            continue
        old_to_new[i] = idx
        idx += 1
    return old_to_new, idx


def _rank_key(var: ir.Sym, registry: DerivativeRegistry) -> Tuple[str, int]:
    """Base name and derivative order of ``var``."""

    base, order = registry.base_and_order(var)
    return base.name, order


def variable_ranks(
    variables: Sequence[ir.Sym], registry: DerivativeRegistry
) -> List[int]:
    """Rank of each variable by base name, then derivative order.

    Not ported.

    Parameters
    ----------
    variables
        The unknowns, base symbols and derivative symbols alike.
    registry
        Registry resolving each derivative symbol's base and order.

    Returns
    -------
    list[int]
        The position of each variable in ``variables`` sorted by
        (base name, derivative order).
    """

    ordered = sorted(
        range(len(variables)),
        key=lambda i: _rank_key(variables[i], registry),
    )
    ranks = [0] * len(variables)
    for rank, i in enumerate(ordered):
        ranks[i] = rank
    return ranks


def _free_symbols(equation: Tuple[ir.Expr, ir.Expr]) -> frozenset:
    """Free symbols of both sides of ``equation``."""

    lhs, rhs = equation
    return ir.free_atoms(lhs) | ir.free_atoms(rhs)


class StructuralState:
    """Symbolic and structural state of a system being simplified.

    Parameters
    ----------
    equations
        The system equations as ``(lhs, rhs)`` pairs. Derivatives
        must already appear as symbols registered in ``registry``.
    registry
        Derivative-symbol registry covering every derivative symbol
        appearing in ``equations``; it records each variable's
        derivative chain.
    known_symbols
        Symbols with externally supplied values (parameters,
        constants, drivers, and the time symbol). Every other symbol
        of ``equations`` is an unknown.
    time_symbol
        The independent variable.
    derivative_names
        User-function name to its derivative helper's name.
    state_priorities
        Optional per-symbol state-selection priorities.
    irreducibles
        Symbols that may not be eliminated from the unknowns.

    Attributes
    ----------
    graph
        Bipartite incidence graph (equations x variables).
    solvable_graph
        Subgraph of ``graph`` restricted to (equation, variable) pairs
        the equation can be explicitly solved for, or ``None`` before
        solvability analysis.
    eq_to_diff
        Maps equation indices to their differentiated equations.
    state_priorities
        Per-variable state-selection priority (higher is more likely
        to stay a state).
    canonical_ranks
        Per-variable rank (see :func:`variable_ranks`), breaking ties
        between equal state priorities.
    """

    def __init__(
        self,
        equations: Sequence[Tuple[ir.Expr, ir.Expr]],
        registry: DerivativeRegistry,
        known_symbols: Iterable[ir.Sym],
        time_symbol: ir.Sym,
        derivative_names: Optional[Dict[str, str]] = None,
        state_priorities: Optional[Dict[ir.Sym, float]] = None,
        irreducibles: Optional[Iterable[ir.Sym]] = None,
    ) -> None:
        self.registry = registry
        self.time_symbol = time_symbol
        self.derivative_names = dict(derivative_names or {})
        self.known_symbols = set(known_symbols) | {time_symbol}
        self.irreducibles = set(irreducibles or ())
        self.mm = None
        self.additional_observed = []
        self.solvable_graph = None

        eqs = [(lhs, rhs) for lhs, rhs in equations]
        original_eqs = list(eqs)

        self.fullvars = self._ordered_variables(eqs)
        self.var2idx = {v: i for i, v in enumerate(self.fullvars)}
        nvars = len(self.fullvars)
        self.canonical_ranks = variable_ranks(self.fullvars, registry)
        self.state_priorities = self._build_state_priorities(
            state_priorities or {}
        )
        self.always_present = [False] * nvars

        # Canonicalize algebraic equations to 0 ~ rhs - lhs. An
        # equation is algebraic when it is incident on no derivative
        # symbol.
        for i, (lhs, rhs) in enumerate(eqs):
            isalgeq = all(
                not registry.is_derivative(v)
                for v in _free_symbols((lhs, rhs))
                if v in self.var2idx
            )
            if isalgeq and not ir.is_zero(lhs):
                eqs[i] = (ir.ZERO, ir.sub(rhs, lhs))

        # Order equations by their printed form.
        sortidxs = sorted(
            range(len(eqs)), key=lambda i: f"{eqs[i][0]} ~ {eqs[i][1]}"
        )
        self.eqs = [eqs[i] for i in sortidxs]
        self.original_eqs = [original_eqs[i] for i in sortidxs]

        self.graph = BipartiteGraph(len(self.eqs), nvars)
        for ie, eq in enumerate(self.eqs):
            self.graph.set_neighbors(ie, self.incidence(eq))
        self.eq_to_diff = DiffGraph(len(self.eqs))

    def _ordered_variables(
        self, eqs: Sequence[Tuple[ir.Expr, ir.Expr]]
    ) -> List[ir.Sym]:
        """Variables of ``eqs`` in index order.

        Not ported. The derivative symbols occurring in ``eqs`` come
        first, sorted by base name, then derivative order; then the other
        members of their chains down to the base unknowns, sorted by base
        name, then derivative order descending; then the remaining
        occurring unknowns, sorted by base name.
        """

        registry = self.registry
        occurring = set().union(*(_free_symbols(eq) for eq in eqs))
        occurring -= self.known_symbols

        derivatives = {s for s in occurring if registry.is_derivative(s)}
        lower_orders = set()
        for sym in derivatives:
            sym = registry.lower_order(sym)
            while sym is not None:
                lower_orders.add(sym)
                sym = registry.lower_order(sym)
        lower_orders -= derivatives

        def rank(sym: ir.Sym) -> Tuple[str, int]:
            return _rank_key(sym, registry)

        def rank_descending(sym: ir.Sym) -> Tuple[str, int]:
            name, order = _rank_key(sym, registry)
            return name, -order

        return (
            sorted(derivatives, key=rank)
            + sorted(lower_orders, key=rank_descending)
            + sorted(occurring - derivatives - lower_orders, key=rank)
        )

    def _build_state_priorities(
        self, priority_map: Dict[ir.Sym, float]
    ) -> List[float]:
        """Give each variable the priority of its derivative chain.

        A chain's priority is the largest user priority of any of its
        members, and never less than zero.
        """

        priorities = [0.0] * len(self.fullvars)
        for i in range(len(self.fullvars)):
            if self.primal_of(i) is not None:
                continue
            chain = [i]
            while self.derivative_of(chain[-1]) is not None:
                chain.append(self.derivative_of(chain[-1]))
            p = 0.0
            for var in chain:
                p = max(p, float(priority_map.get(self.fullvars[var], 0)))
            for var in chain:
                priorities[var] = p
        return priorities

    # -- Derivative chains -------------------------------------------

    def derivative_of(self, var: int) -> Optional[int]:
        """Index of the derivative of variable ``var``, if one exists."""

        derivative = self.registry.higher_order(self.fullvars[var])
        if derivative is None:
            return None
        return self.var2idx.get(derivative)

    def primal_of(self, var: int) -> Optional[int]:
        """Index of the variable ``var`` is the derivative of, if any."""

        lower = self.registry.lower_order(self.fullvars[var])
        if lower is None:
            return None
        return self.var2idx.get(lower)

    def is_algebraic(self, var: int) -> bool:
        """Whether variable ``var`` has no derivative relations at all."""

        return self.derivative_of(var) is None and self.primal_of(var) is None

    # -- Transformation-state interface ------------------------------

    def incidence(self, equation: Tuple[ir.Expr, ir.Expr]) -> List[int]:
        """Sorted indices of the variables occurring in ``equation``."""

        return sorted(
            self.var2idx[symbol]
            for symbol in _free_symbols(equation)
            if symbol in self.var2idx
        )

    def is_unused_var(self, var: int) -> bool:
        """Whether ``var`` occurs in no equation and is removable."""

        return not self.always_present[var] and not (
            self.graph.d_neighbors(var)
        )

    def add_variable(self, symbol: ir.Sym, like: int) -> int:
        """Append variable ``symbol``; return its index.

        The new variable takes the state priority and rank of
        variable ``like`` and occurs in no equation yet.
        """

        var = len(self.fullvars)
        self.fullvars.append(symbol)
        self.var2idx[symbol] = var
        self.state_priorities.append(self.state_priorities[like])
        self.canonical_ranks.append(self.canonical_ranks[like])
        self.always_present.append(False)
        self.graph.add_vertex(DST)
        if self.solvable_graph is not None:
            self.solvable_graph.add_vertex(DST)
        if self.mm is not None:
            self.mm.ncols += 1
        return var

    def add_equation(
        self,
        equation: Tuple[ir.Expr, ir.Expr],
        incidence: Iterable[int],
    ) -> int:
        """Append ``equation`` incident on ``incidence``; return its index.

        The new equation has no solvable edges and no derivative.
        """

        ieq = len(self.eqs)
        self.eqs.append(equation)
        self.original_eqs.append(equation)
        self.graph.add_vertex(SRC)
        self.graph.set_neighbors(ieq, incidence)
        self.solvable_graph.add_vertex(SRC)
        self.eq_to_diff.add_vertex()
        self.mm.nparentrows += 1
        return ieq

    def var_derivative(self, v: int) -> int:
        """Introduce the derivative variable of ``v``; return its index."""

        return self.add_variable(
            self.registry.derivative(self.fullvars[v]), v
        )

    def eq_derivative(self, ieq: int, **kwargs) -> int:
        """Differentiate equation ``ieq``; return the new equation index."""

        lhs, rhs = self.eqs[ieq]
        deriv_map = {}
        for v in _free_symbols((lhs, rhs)):
            j = self.var2idx.get(v)
            if j is not None:
                dv = self.derivative_of(j)
                if dv is not None:
                    deriv_map[v] = self.fullvars[dv]
        new_rhs = total_derivative(
            ir.sub(rhs, lhs),
            deriv_map,
            self.time_symbol,
            derivative_names=self.derivative_names,
        )

        # Superset incidence: previous incidence plus derivatives;
        # find_eq_solvables prunes false entries.
        neighbors = self.graph.s_neighbors(ieq)
        eq_diff = self.add_equation(
            (ir.ZERO, new_rhs),
            list(neighbors) + [self.derivative_of(v) for v in neighbors],
        )
        self.eq_to_diff[ieq] = eq_diff

        coeffs = []
        solv_kwargs = {"allow_symbolic": False}
        solv_kwargs.update(kwargs)
        all_int_vars, rem = self.find_eq_solvables(
            eq_diff, coeffs=coeffs, **solv_kwargs
        )
        if all_int_vars and ir.is_zero(rem):
            # Not ported: an integer-linear derivative joins mm.
            self.mm.nzrows.append(eq_diff)
            self.mm.row_cols.append(list(self.graph.s_neighbors(eq_diff)))
            self.mm.row_vals.append(coeffs)
        return eq_diff

    def division_permitted(
        self,
        denom: ir.Expr,
        allow_symbolic: bool,
        allow_parameter: bool,
    ) -> bool:
        """Whether dividing by ``denom`` is permitted."""

        if allow_symbolic:
            return True
        if not allow_parameter:
            return isinstance(denom, ir.Num)
        return not any(v in self.var2idx for v in ir.free_atoms(denom))

    def find_eq_solvables(
        self,
        ieq: int,
        coeffs: Optional[List[int]] = None,
        allow_symbolic: bool = False,
        allow_parameter: bool = True,
        conservative: bool = False,
        **_ignored,
    ) -> Tuple[bool, ir.Expr]:
        """Recompute the solvable edges of equation ``ieq``.

        Incident variables whose coefficient is zero lose their
        incidence edge.

        Parameters
        ----------
        ieq
            Equation index.
        coeffs
            Filled with the integer coefficients of the incident
            variables, aligned with the equation's incidence after
            zero-coefficient removal.
        allow_symbolic, allow_parameter
            Division policy for symbolic coefficients.
        conservative
            Admit only coefficients of magnitude one, both into the
            integer row and as solvable edges.

        Returns
        -------
        tuple
            ``(all_int_vars, remainder)``: whether every unknown
            enters linearly with an integer coefficient of magnitude
            at most 127, and the residual after peeling the terms
            with numeric coefficients (zero for a homogeneous
            integer-linear equation).
        """

        if coeffs is not None:
            coeffs.clear()
        graph = self.graph
        solvable_graph = self.solvable_graph
        solvable_graph.set_neighbors(ieq, ())
        lhs, rhs = self.eqs[ieq]
        term = ir.sub(rhs, lhs)
        all_int_vars = True
        to_rm = []

        for j in list(graph.s_neighbors(ieq)):
            var = self.fullvars[j]
            if var in self.irreducibles:
                all_int_vars = False
                continue
            a, b, islinear = linear_expansion(term, var)
            if not islinear:
                all_int_vars = False
                continue
            if not isinstance(a, ir.Num):
                all_int_vars = False
                if not self.division_permitted(
                    a, allow_symbolic, allow_parameter
                ):
                    continue
                solvable_graph.add_edge(ieq, j)
                continue
            term = b
            # The IR keeps cancelling terms apart, so a zero
            # coefficient means the variable is absent.
            if ir.is_zero(a):
                to_rm.append(j)
                continue
            a_int = ir.int_value(a)
            if a_int is not None and abs(a_int) > MAX_INTEGER_COEFFICIENT:
                a_int = None
            if conservative and a_int not in (-1, 1):
                all_int_vars = False
                continue
            if a_int is None:
                all_int_vars = False
            elif coeffs is not None:
                coeffs.append(a_int)
            solvable_graph.add_edge(ieq, j)
        for j in to_rm:
            graph.rem_edge(ieq, j)
        return all_int_vars, term

    def rewrite_from_row(
        self, ieq: int, cols: List[int], vals: List[int], **kwargs
    ) -> None:
        """Rewrite equation ``ieq`` as ``0 ~ sum(vals * variables)``.

        The equation becomes incident on ``cols`` and its solvable
        edges are recomputed under the solvability options
        ``kwargs`` of :meth:`find_eq_solvables`.
        """

        rhs = ir.add(*[c * self.fullvars[v] for c, v in zip(vals, cols)])
        self.eqs[ieq] = (ir.ZERO, rhs)
        self.graph.set_neighbors(ieq, cols)
        self.find_eq_solvables(ieq, **kwargs)

    def linear_subsys_adjmat(self, **kwargs) -> SparseMatrixCLIL:
        """Identify integer-coefficient homogeneous linear equations.

        Returns the :class:`SparseMatrixCLIL` of rows of the form
        ``sum(c_i * v_i) == 0`` with small integer ``c_i``.
        """

        graph = self.graph
        self.solvable_graph = BipartiteGraph(graph.nsrcs(), graph.ndsts())
        linear_equations = []
        eadj = []
        cadj = []
        for i in range(len(self.eqs)):
            coeffs = []
            all_int_vars, rem = self.find_eq_solvables(
                i, coeffs=coeffs, **kwargs
            )
            if all_int_vars and ir.is_zero(rem):
                linear_equations.append(i)
                eadj.append(list(graph.s_neighbors(i)))
                cadj.append(list(coeffs))
        return SparseMatrixCLIL(
            graph.nsrcs(),
            graph.ndsts(),
            linear_equations,
            eadj,
            cadj,
        )

    def rm_eqs_vars(
        self, eqs_to_rm: List[int], vars_to_rm: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Delete equations and variables, renumbering the rest.

        Parameters
        ----------
        eqs_to_rm
            Equation indices to delete, in any order.
        vars_to_rm
            Variable indices to delete, in any order, possibly
            repeated.

        Returns
        -------
        tuple
            ``(old_to_new_eq, old_to_new_var)``: the new index of
            each old equation and variable, ``-1`` for a deleted one.
        """

        old_to_new_eq, n_new_eqs = _old_to_new_indices(
            self.graph.nsrcs(), sorted(set(eqs_to_rm))
        )
        old_to_new_var, n_new_vars = _old_to_new_indices(
            self.graph.ndsts(), sorted(set(vars_to_rm))
        )

        def renumbered(graph: BipartiteGraph) -> BipartiteGraph:
            new_graph = BipartiteGraph(n_new_eqs, n_new_vars)
            for e, ne in enumerate(old_to_new_eq):
                if ne < 0:
                    continue
                new_graph.set_neighbors(
                    ne,
                    [
                        old_to_new_var[v]
                        for v in graph.s_neighbors(e)
                        if old_to_new_var[v] >= 0
                    ],
                )
            return new_graph

        self.graph = renumbered(self.graph)
        if self.solvable_graph is not None:
            self.solvable_graph = renumbered(self.solvable_graph)

        kept_eqs = [e for e, ie in enumerate(old_to_new_eq) if ie >= 0]
        kept_vars = [v for v, iv in enumerate(old_to_new_var) if iv >= 0]
        self.eqs[:] = [self.eqs[e] for e in kept_eqs]
        self.original_eqs[:] = [self.original_eqs[e] for e in kept_eqs]
        for name in _VARIABLE_FIELDS:
            values = getattr(self, name)
            values[:] = [values[v] for v in kept_vars]
        self.var2idx = {v: i for i, v in enumerate(self.fullvars)}
        # No equation is differentiated before index reduction.
        self.eq_to_diff = DiffGraph(n_new_eqs)
        return old_to_new_eq, old_to_new_var

    def n_concrete_eqs(self) -> int:
        """Number of equations with at least one incident variable."""

        graph = self.graph
        return sum(1 for e in range(graph.nsrcs()) if graph.s_neighbors(e))
