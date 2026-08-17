"""Structural state of a DAE system under simplification.

Port of ModelingToolkitTearing's ``TearingState``/``SystemStructure``
pair and its StateSelection interface implementation: the bipartite
incidence graph, derivative chains, solvability analysis via linear
expansion, the integer-linear subsystem matrix, and the symbolic
differentiation hooks used by Pantelides.

Published Classes
-----------------
:class:`Equation`
    Immutable ``lhs ~ rhs`` pair of engine IR expressions.

:class:`SystemStructure`
    Integer-graph view of the system (incidence, solvability,
    derivative chains, priorities, deterministic ranks).

:class:`StructuralState`
    Full transformation state: structure plus the symbolic equations,
    variables, derivative registry, and bookkeeping updated by the
    passes.
"""

import warnings
from math import inf
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.expr import _children

from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    DST,
    SRC,
)
from cubie.odesystems.symbolic.structural.clil import SparseMatrixCLIL
from cubie.odesystems.symbolic.structural.diffgraph import DiffGraph
from cubie.odesystems.symbolic.structural.symbolics import (
    DerivativeRegistry,
    as_small_int,
    linear_expansion,
    total_derivative,
)


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
        Deterministic per-variable rank used as a tie-break so results
        do not depend on declaration or equation order.
    """

    def __init__(
        self,
        var_to_diff: DiffGraph,
        eq_to_diff: DiffGraph,
        graph: BipartiteGraph,
        solvable_graph: Optional[BipartiteGraph],
        state_priorities: List[int],
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

    def isalgvar(self, i: int) -> bool:
        """Whether variable ``i`` has no derivative relations."""

        return (
            self.var_to_diff[i] is None
            and self.var_to_diff.diff_to_primal[i] is None
        )

    def isdiffvar(self, i: int) -> bool:
        """Whether ``i`` is differentiated but not itself a derivative."""

        return (
            self.var_to_diff[i] is not None
            and self.var_to_diff.diff_to_primal[i] is None
        )

    def algeqs(self) -> Set[int]:
        """Equations incident on no derivative variables."""

        return {
            eq
            for eq in range(self.graph.nsrcs())
            if all(
                not self.isdervar(v) for v in self.graph.s_neighbors(eq)
            )
        }

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


def get_old_to_new_idxs(
    n_before: int, dels: List[int]
) -> Tuple[List[int], int]:
    """Index remapping after deleting sorted indices ``dels``.

    Deleted entries map to ``-1``; the second element is the new
    length.
    """

    old_to_new = [0] * n_before
    idx = 0
    cursor = 0
    ndels = len(dels)
    for i in range(n_before):
        if cursor < ndels and i == dels[cursor]:
            cursor += 1
            old_to_new[i] = -1
            continue
        old_to_new[i] = idx
        idx += 1
    return old_to_new, idx


def default_rm_eqs_vars(
    structure: SystemStructure,
    eqs_to_rm: List[int],
    vars_to_rm: List[int],
    eqs_sorted_and_uniqued: bool = False,
    vars_sorted_and_uniqued: bool = False,
) -> Tuple[List[int], List[int]]:
    """Remove equations and variables from ``structure``.

    Returns ``(old_to_new_eq, old_to_new_var)`` index maps with
    deleted entries mapped to ``-1``.
    """

    graph = structure.graph
    solvable_graph = structure.solvable_graph

    if not eqs_sorted_and_uniqued:
        eqs_to_rm[:] = sorted(set(eqs_to_rm))
    if not vars_sorted_and_uniqued:
        vars_to_rm[:] = sorted(set(vars_to_rm))

    old_to_new_eq, n_new_eqs = get_old_to_new_idxs(
        graph.nsrcs(), eqs_to_rm
    )
    old_to_new_var, n_new_vars = get_old_to_new_idxs(
        graph.ndsts(), vars_to_rm
    )

    new_graph = BipartiteGraph(n_new_eqs, n_new_vars)
    new_solvable_graph = None
    if solvable_graph is not None:
        new_solvable_graph = BipartiteGraph(n_new_eqs, n_new_vars)
    new_eq_to_diff = DiffGraph(n_new_eqs)
    for i, ieq in enumerate(old_to_new_eq):
        if ieq < 0:
            continue
        new_nbors = [
            old_to_new_var[v]
            for v in graph.s_neighbors(i)
            if old_to_new_var[v] >= 0
        ]
        new_graph.set_neighbors(ieq, new_nbors)
        if solvable_graph is not None:
            new_nbors = [
                old_to_new_var[v]
                for v in solvable_graph.s_neighbors(i)
                if old_to_new_var[v] >= 0
            ]
            new_solvable_graph.set_neighbors(ieq, new_nbors)
        ediff = structure.eq_to_diff[i]
        if ediff is not None:
            ediff = old_to_new_eq[ediff]
            if ediff < 0:
                raise AssertionError(
                    "removed an equation whose derivative is retained"
                )
        new_eq_to_diff[ieq] = ediff

    new_var_to_diff = DiffGraph(n_new_vars)
    for v, newv in enumerate(old_to_new_var):
        if newv < 0:
            continue
        vdiff = structure.var_to_diff[v]
        if vdiff is not None:
            vdiff = old_to_new_var[vdiff]
            if vdiff < 0:
                continue
        new_var_to_diff[newv] = vdiff
    structure.graph = new_graph
    if solvable_graph is not None:
        structure.solvable_graph = new_solvable_graph
    structure.eq_to_diff = new_eq_to_diff
    structure.var_to_diff = new_var_to_diff
    return old_to_new_eq, old_to_new_var


def _canonical_sort_key(
    var: ir.Sym, registry: DerivativeRegistry
) -> Tuple[Tuple[int, ...], str]:
    """Deterministic ordering key: derivative chain, then base name."""

    base, order = registry.base_and_order(var)
    return ((1,) * order, base.name)


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
        Whether to sort equations by a deterministic structural key
        before analysis.
    """

    def __init__(
        self,
        equations: Sequence[Equation],
        unknowns: Sequence[ir.Sym],
        registry: DerivativeRegistry,
        known_symbols: Iterable[ir.Sym],
        time_symbol: ir.Sym,
        known_derivative_map: Optional[Dict[ir.Sym, ir.Expr]] = None,
        state_priorities: Optional[Dict[ir.Sym, int]] = None,
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

        # Collect fullvars: declared unknowns that occur, plus every
        # derivative symbol occurring in the equations, plus the
        # intermediate orders of any higher-order chain.
        unknown_set = set(unknowns)
        occurring = set()
        for eq in eqs:
            occurring |= eq.free_symbols()
        occurring -= self.known_symbols

        for v in sorted(occurring, key=lambda s: s.name):
            base, _ = registry.base_and_order(v)
            if base not in unknown_set:
                raise ValueError(
                    f"{v} is present in the system but {base} is not "
                    "an unknown."
                )

        fullvars = []
        seen = set()

        def addvar(sym: ir.Sym) -> None:
            if sym not in seen:
                seen.add(sym)
                fullvars.append(sym)

        # Derivative symbols and their chains first.
        dervars = [v for v in occurring if registry.is_derivative(v)]
        dervars.sort(key=lambda v: _canonical_sort_key(v, registry))
        for v in dervars:
            addvar(v)
        for v in dervars:
            chain = v
            while True:
                lower = registry.lower_order(chain)
                if lower is None:
                    break
                addvar(lower)
                chain = lower
        for v in sorted(
            occurring, key=lambda v: _canonical_sort_key(v, registry)
        ):
            addvar(v)
        # Declared unknowns that do not occur are dropped (mirrors
        # MTK: variables not present in the equations are removed).

        self.fullvars = fullvars
        self.var2idx = {v: i for i, v in enumerate(fullvars)}

        # var_to_diff from the registry chains.
        nvars = len(fullvars)
        var_to_diff = DiffGraph(nvars, with_badj=True)
        for i, v in enumerate(fullvars):
            lower = registry.lower_order(v)
            if lower is not None and lower in self.var2idx:
                var_to_diff[self.var2idx[lower]] = i

        canonical_ranks = self._build_canonical_ranks()
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
            keys = [
                _equation_sort_key(
                    eq, self.var2idx, canonical_ranks
                )
                for eq in eqs
            ]
            order = sorted(range(len(eqs)), key=lambda i: keys[i])
            eqs = [eqs[i] for i in order]
            original_eqs = [original_eqs[i] for i in order]

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

    def _build_canonical_ranks(self) -> List[int]:
        keys = [
            _canonical_sort_key(v, self.registry) for v in self.fullvars
        ]
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        ranks = [0] * len(keys)
        for rank, i in enumerate(order):
            ranks[i] = (rank + 1) * 100
        return ranks

    def _build_state_priorities(
        self,
        priority_map: Dict[ir.Sym, int],
        var_to_diff: DiffGraph,
    ) -> List[int]:
        priorities = [
            int(round(priority_map.get(v, 0))) for v in self.fullvars
        ]
        # Propagate up derivative chains: each variable's priority is
        # the running maximum from the lowest-order variable upward.
        var_to_diff.complete()
        for i in range(len(self.fullvars)):
            if var_to_diff.diff_to_primal[i] is not None:
                continue
            p = priorities[i]
            var = i
            while True:
                p = max(p, priorities[var])
                priorities[var] = p
                nxt = var_to_diff[var]
                if nxt is None:
                    break
                var = nxt
        return priorities

    # -- StateSelection interface ------------------------------------

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
        s.canonical_ranks.append(s.canonical_ranks[v] + 1)
        self.always_present.append(self.always_present[v])
        if self.mm is not None:
            self.mm.ncols += 1
        return var_diff

    def eq_derivative(self, ieq: int, **kwargs) -> int:
        """Differentiate equation ``ieq``; return the new equation index."""

        s = self.structure
        eq_diff = s.eq_derivative_graph(ieq)

        if self.mm is not None:
            self.mm.nparentrows += 1
            try:
                idx = self.mm.nzrows.index(ieq)
            except ValueError:
                idx = None
            if idx is not None:
                return self._eq_derivative_mm(ieq, eq_diff, idx)

        deriv_map = {}
        for v in self.eqs[ieq].free_symbols():
            j = self.var2idx.get(v)
            if j is not None:
                dv = s.var_to_diff[j]
                if dv is not None:
                    deriv_map[v] = self.fullvars[dv]
        residual = self.eqs[ieq].residual()
        for v in ir.free_atoms(residual):
            if (
                v in self.var2idx
                and v not in deriv_map
                and v is not self.time_symbol
            ):
                raise ValueError(
                    f"Cannot differentiate equation {self.eqs[ieq]}: "
                    f"variable {v} has no derivative variable."
                )
        new_rhs = total_derivative(
            residual,
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
            dvar = s.var_to_diff[var]
            if dvar is not None:
                s.graph.add_edge(eq_diff, dvar)

        if s.solvable_graph is not None:
            to_rm = []
            coeffs = []
            solv_kwargs = {
                "may_be_zero": True,
                "allow_symbolic": False,
            }
            solv_kwargs.update(kwargs)
            all_int_vars, rem = self.find_eq_solvables(
                eq_diff, to_rm, coeffs, **solv_kwargs
            )
            if self.mm is not None and all_int_vars and ir.is_zero(rem):
                if self.mm.nzrows and eq_diff <= self.mm.nzrows[-1]:
                    raise AssertionError("mm rows out of order")
                self.mm.nzrows.append(eq_diff)
                self.mm.row_cols.append(
                    list(s.solvable_graph.s_neighbors(eq_diff))
                )
                self.mm.row_vals.append(coeffs)
        return eq_diff

    def _eq_derivative_mm(self, ieq: int, eq_diff: int, idx: int) -> int:
        """Differentiate an integer-linear equation through ``mm``."""

        s = self.structure
        mm = self.mm
        rcol = list(mm.row_cols[idx])
        rval = list(mm.row_vals[idx])
        new_cols = []
        for c in rcol:
            dc = s.var_to_diff[c]
            if dc is None:
                raise AssertionError(
                    "differentiating an mm row whose variable has no "
                    "derivative"
                )
            new_cols.append(dc)
        pairs = sorted(zip(new_cols, rval))
        rcol = [p[0] for p in pairs]
        rval = [p[1] for p in pairs]
        if mm.nzrows and eq_diff <= mm.nzrows[-1]:
            raise AssertionError("mm rows out of order")
        mm.nzrows.append(eq_diff)
        mm.row_cols.append(rcol)
        mm.row_vals.append(rval)

        rhs = ir.add(
            *[
                ir.mul(coeff, self.fullvars[c])
                for c, coeff in zip(rcol, rval)
            ]
        )
        new_eq = Equation(ir.ZERO, rhs)
        self.eqs.append(new_eq)
        self.original_eqs.append(new_eq)
        if len(self.eqs) != eq_diff + 1:
            raise AssertionError("equation count diverged from graph")
        for v in rcol:
            s.graph.add_edge(eq_diff, v)
            if s.solvable_graph is not None:
                s.solvable_graph.add_edge(eq_diff, v)
        return eq_diff

    def _check_allow_symbolic_parameter(
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
        # Parameter-only denominators allowed; anything containing an
        # unknown is rejected.
        for v in ir.free_atoms(denom):
            if v in self.var2idx:
                return False
            base, _ = self.registry.base_and_order(v)
            if base in self.var2idx:
                return False
        return True

    def find_eq_solvables(
        self,
        ieq: int,
        to_rm: Optional[List[int]] = None,
        coeffs: Optional[List[int]] = None,
        may_be_zero: bool = True,
        allow_symbolic: bool = False,
        allow_parameter: bool = True,
        conservative: bool = False,
        **_ignored,
    ) -> Tuple[bool, ir.Expr]:
        """Populate the solvable graph for equation ``ieq``.

        Returns ``(all_int_vars, remainder)`` where ``all_int_vars``
        reports whether every unknown enters linearly with a small
        integer coefficient and ``remainder`` is the residual after
        peeling those terms (zero for a homogeneous integer-linear
        equation).
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
        eq = self.eqs[ieq]
        term = eq.residual()
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
                if not self._check_allow_symbolic_parameter(
                    a, allow_symbolic, allow_parameter
                ):
                    continue
                solvable_graph.add_edge(ieq, j)
                continue
            term = b
            a_int = as_small_int(a)
            if a_int is None:
                all_int_vars = False
                if conservative:
                    continue
                a_int = 0 if ir.is_zero(a) else None
                if a_int is None:
                    # Non-small nonzero numeric coefficient: solvable,
                    # but not part of the integer subsystem.
                    solvable_graph.add_edge(ieq, j)
                    continue
            elif conservative and abs(a_int) > 1:
                # Conservative mode admits only unit coefficients:
                # the variable is not solvable here and the equation
                # leaves the integer subsystem (a partial coeffs row
                # would desync from the incidence columns).
                all_int_vars = False
                continue
            if coeffs is not None and (a_int != 0 or not may_be_zero):
                coeffs.append(a_int)
            if a_int != 0:
                solvable_graph.add_edge(ieq, j)
                continue
            if may_be_zero:
                to_rm.append(j)
            else:
                warnings.warn(
                    f"Internal error: variable {var} was marked as "
                    f"being in {eq}, but was actually zero"
                )
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
        self,
        eqs_to_rm: List[int],
        vars_to_rm: List[int],
        eqs_sorted_and_uniqued: bool = False,
        vars_sorted_and_uniqued: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """Remove equations and variables from the state.

        Does not update ``mm``; callers combine the returned index
        maps with alias information via ``get_new_mm``.
        """

        if not eqs_sorted_and_uniqued:
            eqs_to_rm = sorted(set(eqs_to_rm))
        if not vars_sorted_and_uniqued:
            vars_to_rm = sorted(set(vars_to_rm))
        structure = self.structure
        old_to_new_eq, old_to_new_var = default_rm_eqs_vars(
            structure,
            eqs_to_rm,
            vars_to_rm,
            eqs_sorted_and_uniqued=True,
            vars_sorted_and_uniqued=True,
        )
        for v in reversed(vars_to_rm):
            del structure.canonical_ranks[v]
            del structure.state_priorities[v]
            del self.fullvars[v]
            del self.always_present[v]
        self.var2idx = {v: i for i, v in enumerate(self.fullvars)}
        for e in reversed(eqs_to_rm):
            del self.eqs[e]
            del self.original_eqs[e]
        return old_to_new_eq, old_to_new_var

    def possibly_explicit_equations(self) -> List[Tuple[int, int]]:
        """Candidates for trivial tearing.

        Returns ``(equation_index, variable_index)`` pairs where the
        original user equation assigns a single unknown explicitly and
        the assignment is not self-referential.
        """

        result = []
        for i, oeq in enumerate(self.original_eqs):
            lhs = oeq.lhs
            if not isinstance(lhs, ir.Sym):
                continue
            vidx = self.var2idx.get(lhs)
            if vidx is None:
                continue
            if lhs in self.irreducibles:
                continue
            sys_eq = self.eqs[i]
            if ir.is_zero(sys_eq.lhs) and ir.is_zero(sys_eq.rhs):
                continue
            if lhs in ir.free_atoms(oeq.rhs):
                continue
            result.append((i, vidx))
        return result

    def trivial_tearing_postprocess(
        self, torn_eqs: List[int], torn_vars: List[int]
    ) -> None:
        """Record preemptively torn equations as observed."""

        self.additional_observed.extend(
            self.original_eqs[e] for e in torn_eqs
        )

    def n_concrete_eqs(self) -> int:
        """Number of equations with at least one incident variable."""

        graph = self.structure.graph
        return sum(
            1
            for e in range(graph.nsrcs())
            if graph.s_neighbors(e)
        )


def _num_float(node: ir.Expr) -> float:
    """Return ``float(node.value)``; raise for a non-numeric node."""

    if not isinstance(node, ir.Num):
        raise TypeError(f"{node!r} is not a numeric IR node")
    return float(node.value)


def _ieee_pow(base: float, exponent: float) -> float:
    """Float power with IEEE semantics: zero or overflow gives inf."""
    try:
        return base**exponent
    except (ZeroDivisionError, OverflowError):
        return inf


def _expression_sort_key(
    expr: ir.Expr,
    var2idx: Dict[ir.Sym, int],
    canonical_ranks: List[int],
    cache: Dict,
) -> List[Tuple[int, float, float]]:
    """Structural sort key for deterministic equation ordering.

    Port of ModelingToolkitTearing's equation sort key: a list of
    ``(variable_rank, coefficient, exponent)`` tuples derived from the
    expression tree, capped at 100 entries.
    """

    cached = cache.get(expr)
    if cached is not None:
        return cached
    result = __expression_sort_key(expr, var2idx, canonical_ranks, cache)
    cache[expr] = result
    return result


def __expression_sort_key(
    expr: ir.Expr,
    var2idx: Dict[ir.Sym, int],
    canonical_ranks: List[int],
    cache: Dict,
) -> List[Tuple[int, float, float]]:
    if isinstance(expr, ir.Num):
        try:
            return [(0, float(expr.value), 1.0)]
        except (TypeError, ValueError):
            return []
    if isinstance(expr, ir.Sym):
        idx = var2idx.get(expr)
        if idx is None:
            return []
        return [(canonical_ranks[idx], 1.0, 1.0)]
    if isinstance(expr, ir.Add):
        result = []
        # Add.args are already stored in sort_key order.
        for term in expr.args:
            if isinstance(term, ir.Mul) and isinstance(
                term.args[0], ir.Num
            ):
                coeff = term.args[0].value
                rest = ir.mul(*term.args[1:])
            elif isinstance(term, ir.Num):
                coeff = term.value
                rest = ir.ONE
            else:
                coeff = 1
                rest = term
            sub = _expression_sort_key(
                rest, var2idx, canonical_ranks, cache
            )
            try:
                cf = float(coeff)
            except (TypeError, ValueError):
                result.extend(sub)
                continue
            for rank, c, e in sub:
                result.append((rank, c * cf, e))
            if len(result) > 100:
                break
        return result
    if isinstance(expr, ir.Mul):
        if isinstance(expr.args[0], ir.Num):
            coeff = expr.args[0].value
            rest_factors = expr.args[1:]
        else:
            coeff = 1
            rest_factors = expr.args
        try:
            cf = abs(float(coeff))
        except (TypeError, ValueError):
            cf = 1.0
        result = []
        # The remaining factors stay in sort_key order.
        for factor in rest_factors:
            if isinstance(factor, ir.Pow):
                base, exponent = factor.base, factor.exp
            else:
                base, exponent = factor, ir.ONE
            sub = _expression_sort_key(
                base, var2idx, canonical_ranks, cache
            )
            try:
                ev = _num_float(exponent)
            except (TypeError, ValueError):
                result.extend(sub)
                continue
            for rank, c, e in sub:
                result.append((rank, _ieee_pow(abs(c), ev) * cf, e + ev))
            if len(result) > 100:
                break
        return result
    if isinstance(expr, ir.Pow):
        base, exponent = expr.base, expr.exp
        base_key = _expression_sort_key(
            base, var2idx, canonical_ranks, cache
        )
        try:
            ev = _num_float(exponent)
        except (TypeError, ValueError):
            if len(base_key) > 100:
                return base_key
            return base_key + _expression_sort_key(
                exponent, var2idx, canonical_ranks, cache
            )
        return [
            (rank, _ieee_pow(abs(c), ev), e + ev)
            for rank, c, e in base_key
        ]
    result = []
    for arg in _children(expr):
        result.extend(
            _expression_sort_key(arg, var2idx, canonical_ranks, cache)
        )
        if len(result) > 100:
            break
    return result


def _equation_sort_key(
    eq: Equation,
    var2idx: Dict[ir.Sym, int],
    canonical_ranks: List[int],
) -> List[Tuple[int, float, float]]:
    """Sort key of an equation (over its RHS, matching MTK)."""

    return _expression_sort_key(eq.rhs, var2idx, canonical_ranks, {})
