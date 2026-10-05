"""Structural state of a DAE system under simplification.

The bipartite incidence graph, derivative chains, solvability analysis
via linear expansion, the integer-linear subsystem matrix, and the
symbolic differentiation hooks used by Pantelides.

Ported from ModelingToolkit.jl (commit c4177c335):
``SystemStructure`` from ``src/systems/systemstructure.jl``;
``StructuralState`` construction (algebraic-equation canonicalisation
and the equation order) from ``TearingState`` in the same file, except
the variable order;
``StructuralState.var_derivative`` and ``StructuralState.eq_derivative``
from ``src/structural_transformation/symbolics_tearing.jl``
(``var_derivative!``, ``eq_derivative!``);
``StructuralState._build_state_priorities`` from the
``state_priority`` closure of ``dummy_derivative`` in the same file;
``StructuralState.find_eq_solvables`` with
``StructuralState.division_permitted``,
``StructuralState.find_solvables``,
``StructuralState.linear_subsys_adjmat`` and
``StructuralState.n_concrete_eqs`` from
``src/structural_transformation/utils.jl``;
``StructuralState.rm_eqs_vars`` and ``_old_to_new_indices`` from the
equation renumbering and graph rebuild in ``alias_elimination!``
(``src/systems/alias_elimination.jl``).
``SystemStructure.eq_derivative_graph`` and
``SystemStructure.var_derivative_graph`` are ported from
StateSelection.jl (commit 74df007e, ``src/interface.jl``,
``eq_derivative_graph!``, ``var_derivative_graph!``).
``StructuralState.is_unused_var`` combines the empty-incidence test of
ModelingToolkit.jl c4177c335 (``utils.jl``) with the
``always_present`` marking of ModelingToolkit.jl (commit a2b6dc56,
``src/systems/alias_elimination.jl``).

Published Classes
-----------------
:class:`Equation`
    Immutable ``lhs ~ rhs`` pair of engine IR expressions.

:class:`SystemStructure`
    Integer-graph view of the system (incidence, solvability,
    derivative chains, priorities, ranks).

:class:`StructuralState`
    Full transformation state: structure plus the symbolic equations,
    variables, derivative registry, and bookkeeping updated by the
    passes.

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


class Equation:
    """An equation ``lhs ~ rhs`` of engine IR expressions."""

    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs: ir.Expr, rhs: ir.Expr) -> None:
        self.lhs = self._coerce(lhs)
        self.rhs = self._coerce(rhs)

    @staticmethod
    def _coerce(value) -> ir.Expr:
        """Coerce an operand to an engine IR expression.

        Plain Python ``int``/``float`` values are wrapped as numeric
        literals. Any other non-IR value (in particular a SymPy
        expression) raises ``TypeError``: conversion from SymPy
        belongs at the parse boundary, before an ``Equation`` is
        constructed.
        """

        if isinstance(value, ir.Expr):
            return value
        if isinstance(value, (int, float)):
            return ir.num(value)
        raise TypeError(
            "Equation operands must be engine IR expressions "
            "(cubie.odesystems.symbolic.engine.expr.Expr) or plain "
            f"Python int/float; got {type(value).__name__}. Convert "
            "SymPy expressions to IR at the parse boundary before "
            "constructing an Equation."
        )

    def __repr__(self) -> str:
        return f"{self.lhs} ~ {self.rhs}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Equation)
            and self.lhs is other.lhs
            and self.rhs is other.rhs
        )

    def __hash__(self) -> int:
        return hash((self.lhs, self.rhs))

    def residual(self) -> ir.Expr:
        """Return ``rhs - lhs``."""

        return ir.sub(self.rhs, self.lhs)

    def free_symbols(self) -> frozenset:
        """Free symbols of both sides."""

        return ir.free_atoms(self.lhs) | ir.free_atoms(self.rhs)

    def xreplace(self, rules: Dict[ir.Expr, ir.Expr]) -> "Equation":
        """Return a copy with ``rules`` structurally substituted."""

        return Equation(
            ir.xreplace(self.lhs, rules), ir.xreplace(self.rhs, rules)
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


class SystemStructure:
    """Integer-graph structural information about a DAE.

    Parameters
    ----------
    var_to_diff
        Maps variable indices to their derivative variable indices.
    eq_to_diff
        Maps equation indices to their differentiated equations.
    graph
        Bipartite incidence graph (equations x variables).
    solvable_graph
        Subgraph of ``graph`` restricted to (equation, variable) pairs
        the equation can be explicitly solved for, or ``None`` before
        solvability analysis.
    state_priorities
        Per-variable state-selection priority (higher is more likely
        to stay a state).
    canonical_ranks
        Per-variable rank (see :func:`variable_ranks`), breaking ties
        between equal state priorities.
    """

    def __init__(
        self,
        var_to_diff: DiffGraph,
        eq_to_diff: DiffGraph,
        graph: BipartiteGraph,
        solvable_graph: Optional[BipartiteGraph],
        state_priorities: List[float],
        canonical_ranks: List[int],
    ) -> None:
        self.var_to_diff = var_to_diff
        self.eq_to_diff = eq_to_diff
        self.graph = graph
        self.solvable_graph = solvable_graph
        self.state_priorities = state_priorities
        self.canonical_ranks = canonical_ranks

    def copy(self) -> "SystemStructure":
        """Return a deep copy."""

        return SystemStructure(
            self.var_to_diff.copy(),
            self.eq_to_diff.copy(),
            self.graph.copy(),
            None
            if self.solvable_graph is None
            else self.solvable_graph.copy(),
            list(self.state_priorities),
            list(self.canonical_ranks),
        )

    def complete(self) -> "SystemStructure":
        """Complete all member graphs (inverse/backward adjacency)."""

        self.var_to_diff.complete()
        self.eq_to_diff.complete()
        self.graph.complete()
        if self.solvable_graph is not None:
            self.solvable_graph.complete()
        return self

    def isdervar(self, i: int) -> bool:
        """Whether variable ``i`` is the derivative of another."""

        self.var_to_diff.require_complete()
        return self.var_to_diff.diff_to_primal[i] is not None

    def eq_derivative_graph(self, eq: int) -> int:
        """Add the graph vertices for the derivative of equation ``eq``."""

        self.graph.add_vertex(SRC)
        if self.solvable_graph is not None:
            self.solvable_graph.add_vertex(SRC)
        eq_diff = self.eq_to_diff.add_vertex()
        self.eq_to_diff.add_edge(eq, eq_diff)
        return eq_diff

    def var_derivative_graph(self, v: int) -> int:
        """Add the graph vertices for the derivative of variable ``v``."""

        g = self.graph.add_vertex(DST)
        var_diff = self.var_to_diff.add_vertex()
        self.var_to_diff.add_edge(v, var_diff)
        if self.solvable_graph is not None:
            sg = self.solvable_graph.add_vertex(DST)
            if sg != g:
                raise AssertionError("graph vertex counts diverged")
        if g != var_diff:
            raise AssertionError("graph vertex counts diverged")
        return var_diff


class StructuralState:
    """Symbolic and structural state of a system being simplified.

    Parameters
    ----------
    equations
        The system equations. Derivatives must already appear as
        symbols registered in ``registry``.
    unknowns
        Declared unknown symbols (differential or algebraic; the
        pipeline decides which become states).
    registry
        Derivative-symbol registry covering every derivative symbol
        appearing in ``equations``.
    known_symbols
        Symbols with externally supplied values (parameters,
        constants, drivers, and the time symbol).
    time_symbol
        The independent variable.
    known_derivative_map
        Time derivatives of known time-dependent symbols (drivers).
        Knowns absent from the map differentiate to zero.
    state_priorities
        Optional per-symbol state-selection priorities.
    irreducibles
        Symbols that may not be eliminated from the unknowns.
    sort_eqs
        Whether to sort equations before analysis.
    """

    def __init__(
        self,
        equations: Sequence[Equation],
        unknowns: Sequence[ir.Sym],
        registry: DerivativeRegistry,
        known_symbols: Iterable[ir.Sym],
        time_symbol: ir.Sym,
        known_derivative_map: Optional[Dict[ir.Sym, ir.Expr]] = None,
        state_priorities: Optional[Dict[ir.Sym, float]] = None,
        irreducibles: Optional[Iterable[ir.Sym]] = None,
        sort_eqs: bool = True,
    ) -> None:
        self.registry = registry
        self.time_symbol = time_symbol
        self.known_symbols = set(known_symbols) | {time_symbol}
        self.known_derivative_map = dict(known_derivative_map or {})
        self.irreducibles = set(irreducibles or ())
        self.mm = None
        self.additional_observed = []

        eqs = [Equation(eq.lhs, eq.rhs) for eq in equations]
        original_eqs = list(eqs)

        fullvars = self._ordered_variables(eqs, unknowns)
        seen = set(fullvars)
        self.fullvars = fullvars
        self.var2idx = {v: i for i, v in enumerate(fullvars)}

        # var_to_diff from the registry chains.
        nvars = len(fullvars)
        var_to_diff = DiffGraph(nvars, with_badj=True)
        for i, v in enumerate(fullvars):
            lower = registry.lower_order(v)
            if lower is not None and lower in self.var2idx:
                var_to_diff[self.var2idx[lower]] = i

        canonical_ranks = variable_ranks(fullvars, registry)
        priorities = self._build_state_priorities(
            state_priorities or {}, var_to_diff
        )

        # Canonicalize algebraic equations to 0 ~ rhs - lhs. An
        # equation is algebraic when it is incident on no derivative
        # symbol.
        for i, eq in enumerate(eqs):
            incidence = eq.free_symbols() & seen
            isalgeq = all(
                not registry.is_derivative(v) for v in incidence
            )
            if isalgeq and not ir.is_zero(eq.lhs):
                eqs[i] = Equation(ir.ZERO, eq.residual())

        if sort_eqs:
            # Order equations by their printed form.
            sortidxs = sorted(range(len(eqs)), key=lambda i: str(eqs[i]))
            eqs = [eqs[i] for i in sortidxs]
            original_eqs = [original_eqs[i] for i in sortidxs]

        self.eqs = eqs
        self.original_eqs = original_eqs

        graph = BipartiteGraph(len(eqs), nvars, with_badj=False)
        for ie, eq in enumerate(eqs):
            for v in eq.free_symbols():
                j = self.var2idx.get(v)
                if j is not None:
                    graph.add_edge(ie, j)

        eq_to_diff = DiffGraph(len(eqs))
        self.structure = SystemStructure(
            var_to_diff.complete(),
            eq_to_diff.complete(),
            graph.complete(),
            None,
            priorities,
            canonical_ranks,
        )
        self.always_present = [False] * nvars

    def _ordered_variables(
        self, eqs: Sequence[Equation], unknowns: Sequence[ir.Sym]
    ) -> List[ir.Sym]:
        """Variables of ``eqs`` in index order.

        The derivative symbols occurring in ``eqs`` come first, sorted
        by base name, then derivative order; then the other members of
        their chains down to the base unknowns, sorted by base name,
        then derivative order descending; then the remaining occurring
        unknowns, sorted by base name. Declared unknowns absent from
        ``eqs`` are left out.
        """

        registry = self.registry
        unknown_set = set(unknowns)
        occurring = set().union(*(eq.free_symbols() for eq in eqs))
        occurring -= self.known_symbols

        for sym in sorted(occurring, key=lambda s: s.name):
            base, _ = registry.base_and_order(sym)
            if base not in unknown_set:
                raise ValueError(
                    f"{sym} is present in the system but {base} is not "
                    "an unknown."
                )

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
        self,
        priority_map: Dict[ir.Sym, float],
        var_to_diff: DiffGraph,
    ) -> List[float]:
        """Give each variable the priority of its derivative chain.

        A chain's priority is the largest user priority of any of its
        members, and never less than zero.
        """

        priorities = [0.0] * len(self.fullvars)
        var_to_diff.complete()
        for i in range(len(self.fullvars)):
            if var_to_diff.diff_to_primal[i] is not None:
                continue
            chain = [i]
            while var_to_diff[chain[-1]] is not None:
                chain.append(var_to_diff[chain[-1]])
            p = 0.0
            for var in chain:
                p = max(p, float(priority_map.get(self.fullvars[var], 0)))
            for var in chain:
                priorities[var] = p
        return priorities

    # -- Transformation-state interface ------------------------------

    def is_unused_var(self, var: int) -> bool:
        """Whether ``var`` occurs in no equation and is removable."""

        return not self.always_present[var] and not (
            self.structure.graph.d_neighbors(var)
        )

    def var_derivative(self, v: int) -> int:
        """Introduce the derivative variable of ``v``; return its index."""

        s = self.structure
        var_diff = s.var_derivative_graph(v)
        dsym = self.registry.derivative(self.fullvars[v])
        self.fullvars.append(dsym)
        self.var2idx[dsym] = var_diff
        s.state_priorities.append(s.state_priorities[v])
        s.canonical_ranks.append(s.canonical_ranks[v])
        self.always_present.append(False)
        if self.mm is not None:
            self.mm.ncols += 1
        return var_diff

    def eq_derivative(self, ieq: int, **kwargs) -> int:
        """Differentiate equation ``ieq``; return the new equation index."""

        s = self.structure
        eq_diff = s.eq_derivative_graph(ieq)

        deriv_map = {}
        for v in self.eqs[ieq].free_symbols():
            j = self.var2idx.get(v)
            if j is not None:
                dv = s.var_to_diff[j]
                if dv is not None:
                    deriv_map[v] = self.fullvars[dv]
        new_rhs = total_derivative(
            self.eqs[ieq].residual(),
            deriv_map,
            self.time_symbol,
            self.known_derivative_map,
        )
        new_eq = Equation(ir.ZERO, new_rhs)
        self.eqs.append(new_eq)
        self.original_eqs.append(new_eq)
        if len(self.eqs) != eq_diff + 1:
            raise AssertionError("equation count diverged from graph")

        # Superset incidence: previous incidence plus derivatives;
        # find_eq_solvables prunes false entries.
        for var in list(s.graph.s_neighbors(ieq)):
            s.graph.add_edge(eq_diff, var)
            s.graph.add_edge(eq_diff, s.var_to_diff[var])

        if self.mm is not None:
            self.mm.nparentrows += 1
        if s.solvable_graph is not None:
            to_rm = []
            coeffs = []
            solv_kwargs = {"allow_symbolic": False}
            solv_kwargs.update(kwargs)
            all_int_vars, rem = self.find_eq_solvables(
                eq_diff, to_rm, coeffs, **solv_kwargs
            )
            if self.mm is not None and all_int_vars and ir.is_zero(rem):
                # Not ported: an integer-linear derivative joins mm.
                self.mm.nzrows.append(eq_diff)
                self.mm.row_cols.append(list(s.graph.s_neighbors(eq_diff)))
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
        to_rm: Optional[List[int]] = None,
        coeffs: Optional[List[int]] = None,
        allow_symbolic: bool = False,
        allow_parameter: bool = True,
        conservative: bool = False,
        **_ignored,
    ) -> Tuple[bool, ir.Expr]:
        """Populate the solvable graph for equation ``ieq``.

        Parameters
        ----------
        ieq
            Equation index.
        to_rm
            Filled with the incident variables whose coefficient is
            zero; their incidence edges are removed.
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

        if to_rm is None:
            to_rm = []
        else:
            to_rm.clear()
        if coeffs is not None:
            coeffs.clear()
        s = self.structure
        graph = s.graph
        solvable_graph = s.solvable_graph
        term = self.eqs[ieq].residual()
        all_int_vars = True

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

    def find_solvables(self, **kwargs) -> None:
        """Populate the solvable graph for every equation."""

        if self.structure.solvable_graph is not None:
            raise AssertionError("solvable graph already populated")
        graph = self.structure.graph
        self.structure.solvable_graph = BipartiteGraph(
            graph.nsrcs(), graph.ndsts()
        )
        for ieq in range(graph.nsrcs()):
            self.find_eq_solvables(ieq, **kwargs)

    def rewrite_from_row(
        self, ieq: int, cols: List[int], vals: List[int]
    ) -> None:
        """Rewrite equation ``ieq`` as ``0 ~ sum(vals * variables)``.

        The equation becomes incident on, and solvable for, ``cols``.
        """

        rhs = ir.add(*[c * self.fullvars[v] for c, v in zip(vals, cols)])
        self.eqs[ieq] = Equation(ir.ZERO, rhs)
        self.structure.graph.set_neighbors(ieq, cols)
        self.structure.solvable_graph.set_neighbors(ieq, cols)

    def linear_subsys_adjmat(self, **kwargs) -> SparseMatrixCLIL:
        """Identify integer-coefficient homogeneous linear equations.

        Returns the :class:`SparseMatrixCLIL` of rows of the form
        ``sum(c_i * v_i) == 0`` with small integer ``c_i``.
        """

        graph = self.structure.graph
        if self.structure.solvable_graph is None:
            self.structure.solvable_graph = BipartiteGraph(
                graph.nsrcs(), graph.ndsts()
            )
        linear_equations = []
        eadj = []
        cadj = []
        to_rm = []
        for i in range(len(self.eqs)):
            coeffs = []
            all_int_vars, rem = self.find_eq_solvables(
                i, to_rm, coeffs, **kwargs
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

        s = self.structure
        old_to_new_eq, n_new_eqs = _old_to_new_indices(
            s.graph.nsrcs(), sorted(set(eqs_to_rm))
        )
        old_to_new_var, n_new_vars = _old_to_new_indices(
            s.graph.ndsts(), sorted(set(vars_to_rm))
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

        new_eq_to_diff = DiffGraph(n_new_eqs, with_badj=True)
        for i, ieq in enumerate(old_to_new_eq):
            if ieq < 0:
                continue
            deq = s.eq_to_diff[i]
            if deq is not None and old_to_new_eq[deq] >= 0:
                new_eq_to_diff[ieq] = old_to_new_eq[deq]

        new_var_to_diff = DiffGraph(n_new_vars, with_badj=True)
        for iv, i in enumerate(old_to_new_var):
            if i < 0:
                continue
            dv = s.var_to_diff[iv]
            if dv is not None and old_to_new_var[dv] >= 0:
                new_var_to_diff[i] = old_to_new_var[dv]

        kept_eqs = [e for e, ie in enumerate(old_to_new_eq) if ie >= 0]
        kept_vars = [v for v, iv in enumerate(old_to_new_var) if iv >= 0]
        self.eqs[:] = [self.eqs[e] for e in kept_eqs]
        self.original_eqs[:] = [self.original_eqs[e] for e in kept_eqs]
        self.fullvars[:] = [self.fullvars[v] for v in kept_vars]
        self.var2idx = {v: i for i, v in enumerate(self.fullvars)}
        self.always_present[:] = [
            self.always_present[v] for v in kept_vars
        ]
        s.state_priorities[:] = [s.state_priorities[v] for v in kept_vars]
        s.canonical_ranks[:] = [s.canonical_ranks[v] for v in kept_vars]

        s.graph = renumbered(s.graph)
        if s.solvable_graph is not None:
            s.solvable_graph = renumbered(s.solvable_graph)
        s.eq_to_diff = new_eq_to_diff
        s.var_to_diff = new_var_to_diff
        return old_to_new_eq, old_to_new_var

    def n_concrete_eqs(self) -> int:
        """Number of equations with at least one incident variable."""

        graph = self.structure.graph
        return sum(
            1
            for e in range(graph.nsrcs())
            if graph.s_neighbors(e)
        )
