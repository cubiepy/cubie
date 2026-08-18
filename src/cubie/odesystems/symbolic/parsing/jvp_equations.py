"""Store Jacobian-vector product assignments and cache metadata."""
from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)


if TYPE_CHECKING:
    from cubie.odesystems.symbolic.parsing.auxiliary_caching import CacheSelection

from cubie.odesystems.symbolic.engine import expr as ir


import attrs


_ENTRY_PREFIX = "_cubie_codegen_j_"


def _entry_position(name: str) -> Optional[Tuple[int, int]]:
    """Parse ``(row, col)`` from an entry symbol name, else None."""
    if not name.startswith(_ENTRY_PREFIX):
        return None
    parts = name[len(_ENTRY_PREFIX):].split("_")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1])


@attrs.define
class JVPEquations:
    """Capture ordered auxiliary and JVP assignments with dependency metadata.

    Parameters
    ----------
    assignments
        Topologically ordered sequence of symbolic assignments defining the
        Jacobian-vector product. Entries include auxiliary intermediates and the
        ``jvp[<idx>]`` outputs produced by the Jacobian generator.
    max_cached_terms
        Optional upper bound on the number of auxiliary expressions that may be
        cached. Defaults to twice the number of JVP outputs when omitted.
    read_price
        Device-weighted cost charged per cached slot read in each
        runtime operator evaluation.
    entry_symbols
        Mapping from ``(row, col)`` to the graph symbol holding that
        Jacobian entry; derived from the reserved entry names when
        omitted.
    """

    assignments = attrs.field()
    max_cached_terms = attrs.field(default=None)
    read_price = attrs.field(default=8)
    entry_symbols = attrs.field(default=None)

    _ordered_assignments = attrs.field(init=False, repr=False)
    _non_jvp_order = attrs.field(init=False, repr=False)
    _non_jvp_exprs = attrs.field(init=False, repr=False)
    _jvp_terms = attrs.field(init=False, repr=False)
    _jvp_symbols = attrs.field(init=False, repr=False)
    _dependencies = attrs.field(init=False, repr=False)
    _dependents = attrs.field(init=False, repr=False)
    _ops_cost = attrs.field(init=False, repr=False)
    _jvp_usage = attrs.field(init=False, repr=False)
    _jvp_closure_usage = attrs.field(init=False, repr=False)
    _cache_slot_limit = attrs.field(init=False, repr=False)
    _reference_counts = attrs.field(init=False, repr=False)
    _order_index = attrs.field(init=False, repr=False)
    _v_dependent = attrs.field(init=False, repr=False)
    _total_ops_cost = attrs.field(init=False, repr=False)
    _entry_index = attrs.field(init=False, repr=False)
    _cache_selection = attrs.field(init=False, default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        ordered = tuple(self.assignments)
        self._ordered_assignments = ordered
        non_jvp_order = []
        non_jvp_exprs = {}
        jvp_terms = {}
        jvp_symbols = {}
        for lhs, rhs in ordered:
            if isinstance(lhs, ir.Arr) and lhs.name == "jvp":
                jvp_terms[lhs.index] = rhs
                jvp_symbols[lhs.index] = lhs
            else:
                non_jvp_order.append(lhs)
                non_jvp_exprs[lhs] = rhs
        self._non_jvp_order = tuple(non_jvp_order)
        self._non_jvp_exprs = non_jvp_exprs
        self._jvp_terms = jvp_terms
        self._jvp_symbols = jvp_symbols
        if self.max_cached_terms is None:
            self._cache_slot_limit = 2 * len(jvp_terms)
        else:
            self._cache_slot_limit = self.max_cached_terms
        if self.entry_symbols is None:
            derived: Dict[Tuple[int, int], ir.Expr] = {}
            for lhs in self._non_jvp_order:
                if not isinstance(lhs, ir.Sym):
                    continue
                position = _entry_position(lhs.name)
                if position is not None:
                    derived[position] = lhs
            self._entry_index = derived
        else:
            self._entry_index = dict(self.entry_symbols)
        self._initialise_expression_metadata()

    def _initialise_expression_metadata(self) -> None:
        dependencies = {}
        dependents = {sym: set() for sym in self._non_jvp_order}
        ops_cost = {}
        ops_memo = {}
        assigned_symbols = set(sym for sym, _ in self._ordered_assignments)
        v_dependent = set()
        for lhs in self._non_jvp_order:
            rhs = self._non_jvp_exprs[lhs]
            ops_cost[lhs] = ir.count_device_ops(rhs, ops_memo)
            atoms = ir.free_atoms(rhs)
            deps = {
                sym
                for sym in atoms
                if sym in assigned_symbols
                and not (
                    isinstance(sym, ir.Arr) and sym.name == "jvp"
                )
            }
            dependencies[lhs] = deps
            reads_v = any(
                isinstance(sym, ir.Arr) and sym.name == "v"
                for sym in atoms
            )
            if reads_v or any(dep in v_dependent for dep in deps):
                v_dependent.add(lhs)
            for dep in deps:
                if dep in dependents:
                    dependents[dep].add(lhs)
        jvp_usage = {}
        jvp_closure = {}
        for rhs in self._jvp_terms.values():
            direct = [
                sym for sym in ir.free_atoms(rhs) if sym in dependents
            ]
            seen_direct = set()
            for sym in direct:
                seen_direct.add(sym)
                jvp_usage[sym] = jvp_usage.get(sym, 0) + 1
            stack = list(seen_direct)
            seen_closure = set()
            while stack:
                sym = stack.pop()
                if sym in seen_closure:
                    continue
                seen_closure.add(sym)
                jvp_closure[sym] = jvp_closure.get(sym, 0) + 1
                for dep in dependencies.get(sym, set()):
                    if dep in dependents:
                        stack.append(dep)
        memo_total = {}

        def total_cost(symbol):
            if symbol in memo_total:
                return memo_total[symbol]
            cost = ops_cost.get(symbol, 0)
            for dep in dependencies.get(symbol, set()):
                cost += total_cost(dep)
            memo_total[symbol] = cost
            return cost

        total_ops_cost = {}
        for sym in self._non_jvp_order:
            total_ops_cost[sym] = total_cost(sym)
        for index, expr in self._jvp_terms.items():
            lhs = self._jvp_symbols.get(index)
            if lhs is None:
                continue
            cost = ir.count_device_ops(expr, ops_memo)
            for dep in ir.free_atoms(expr):
                cost += total_cost(dep)
            total_ops_cost[lhs] = cost
        self._dependencies = dependencies
        self._dependents = dependents
        self._ops_cost = ops_cost
        self._jvp_usage = jvp_usage
        self._jvp_closure_usage = jvp_closure
        reference_counts = {
            sym: len(dependents[sym]) + jvp_usage.get(sym, 0)
            for sym in self._non_jvp_order
        }
        self._reference_counts = reference_counts
        self._order_index = {
            sym: idx for idx, sym in enumerate(self._non_jvp_order)
        }
        self._v_dependent = frozenset(v_dependent)
        self._total_ops_cost = total_ops_cost

    @property
    def ordered_assignments(self) -> Tuple[Tuple[ir.Expr, ir.Expr], ...]:
        """Return the canonical ordered assignments."""

        return self._ordered_assignments

    @property
    def non_jvp_order(self) -> Tuple[ir.Expr, ...]:
        """Return auxiliary assignment order excluding JVP outputs."""

        return self._non_jvp_order

    @property
    def non_jvp_exprs(self) -> Mapping[ir.Expr, ir.Expr]:
        """Return mapping from auxiliary symbols to their expressions."""

        return self._non_jvp_exprs

    @property
    def jvp_terms(self) -> Mapping[int, ir.Expr]:
        """Return mapping from output indices to JVP expressions."""

        return self._jvp_terms

    @property
    def dependencies(self) -> Mapping[ir.Expr, Set[ir.Expr]]:
        """Return dependency graph for auxiliary assignments."""

        return self._dependencies

    @property
    def dependents(self) -> Mapping[ir.Expr, Set[ir.Expr]]:
        """Return reverse dependency graph for auxiliary assignments."""

        return self._dependents

    @property
    def ops_cost(self) -> Mapping[ir.Expr, int]:
        """Return per-assignment device-weighted operation counts."""

        return self._ops_cost

    @property
    def jvp_usage(self) -> Mapping[ir.Expr, int]:
        """Return direct JVP usage counts for auxiliary symbols."""

        return self._jvp_usage

    @property
    def jvp_closure_usage(self) -> Mapping[ir.Expr, int]:
        """Return transitive JVP usage counts across dependency chains."""

        return self._jvp_closure_usage

    @property
    def cache_slot_limit(self) -> int:
        """Return the maximum number of cached auxiliary leaves permitted."""

        return self._cache_slot_limit

    @property
    def reference_counts(self) -> Mapping[ir.Expr, int]:
        """Return base reference counts including JVP usage."""

        return self._reference_counts

    @property
    def order_index(self) -> Mapping[ir.Expr, int]:
        """Return evaluation order lookup for auxiliary assignments."""

        return self._order_index

    @property
    def v_dependent_nodes(self) -> frozenset:
        """Return auxiliary symbols reading ``v`` directly or transitively."""

        return self._v_dependent

    @property
    def total_ops_cost(self) -> Mapping[ir.Expr, int]:
        """Return cumulative operation counts for auxiliaries and JVP outputs."""

        return self._total_ops_cost

    def update_cache_selection(self, selection: "CacheSelection") -> None:
        """Persist the cache selection for reuse by solver helpers.

        Parameters
        ----------
        selection
            Computed cache selection to store.
        """
        self._cache_selection = selection

    def ensure_cache_selection(self) -> None:
        """Ensure a cache selection has been computed."""

        if self._cache_selection is None:
            from cubie.odesystems.symbolic.parsing.auxiliary_caching import (
                plan_auxiliary_cache,
            )

            self._cache_selection = plan_auxiliary_cache(self)

    @property
    def cache_selection(self) -> "CacheSelection":
        """Return the cached auxiliary selection."""

        self.ensure_cache_selection()
        assert self._cache_selection is not None
        return self._cache_selection

    def cached_partition(
        self,
    ) -> Tuple[
        List[Tuple[ir.Expr, ir.Expr]],
        List[Tuple[ir.Expr, ir.Expr]],
        List[Tuple[ir.Expr, ir.Expr]],
    ]:
        """Return cached, runtime, and preparation assignments from selection.

        Returns
        -------
        tuple of list, list, list
            Cached assignments, runtime assignments, and preparation assignments
            derived from the stored cache selection.
        """

        selection = self.cache_selection
        cached_symbols = set(selection.cached_leaf_order)
        runtime_symbols = set(selection.runtime_nodes)
        prepare_symbols = set(selection.prepare_nodes)
        cached_assigns = []
        runtime_assigns = []
        prepare_assigns = []
        for lhs in self._non_jvp_order:
            rhs = self._non_jvp_exprs[lhs]
            if lhs in prepare_symbols:
                prepare_assigns.append((lhs, rhs))
            if lhs in cached_symbols:
                cached_assigns.append((lhs, rhs))
            elif lhs in runtime_symbols:
                runtime_assigns.append((lhs, rhs))
        return cached_assigns, runtime_assigns, prepare_assigns

    @property
    def jacobian_entry_symbols(self) -> Mapping[Tuple[int, int], ir.Expr]:
        """Return the ``(row, col) -> symbol`` Jacobian entry index."""

        return self._entry_index

    def jacobian_entry(self, row: int, col: int) -> ir.Expr:
        """Return the graph's entry symbol, or ``ZERO`` when absent.

        Raises
        ------
        KeyError
            If the entry symbol has no defining assignment in the
            graph.
        """
        symbol = self._entry_index.get((row, col))
        if symbol is None:
            return ir.ZERO
        if symbol not in self._non_jvp_exprs:
            raise KeyError(
                f"Jacobian entry symbol {symbol} has no defining "
                "assignment in the JVP graph; entry symbols must be "
                "pinned as pruning outputs when the graph is built."
            )
        return symbol

    @property
    def cached_slot_order(self) -> Tuple[ir.Expr, ...]:
        """Return the cached symbols in slot order."""
        cached_assigns, _, _ = self.cached_partition()
        return tuple(lhs for lhs, _ in cached_assigns)

    def cached_runtime_assignments(self) -> List[Tuple[ir.Expr, ir.Expr]]:
        """Return every non-JVP assignment with cached slots bound.

        Cached symbols read their ``cached_aux`` slot; all other
        symbols keep their defining expressions, in canonical order.
        """
        slots = {
            lhs: idx for idx, lhs in enumerate(self.cached_slot_order)
        }
        assignments: List[Tuple[ir.Expr, ir.Expr]] = []
        for lhs in self._non_jvp_order:
            slot = slots.get(lhs)
            if slot is not None:
                assignments.append((lhs, ir.arr("cached_aux", slot)))
            else:
                assignments.append((lhs, self._non_jvp_exprs[lhs]))
        return assignments

    def prepare_fill_assignments(self) -> List[Tuple[ir.Expr, ir.Expr]]:
        """Return the prepare chain with a slot store after each
        cached symbol's definition."""
        cached_assigns, _, prepare_assigns = self.cached_partition()
        slots = {
            lhs: idx for idx, (lhs, _) in enumerate(cached_assigns)
        }
        assignments: List[Tuple[ir.Expr, ir.Expr]] = []
        for lhs, rhs in prepare_assigns:
            assignments.append((lhs, rhs))
            slot = slots.get(lhs)
            if slot is not None:
                assignments.append((ir.arr("cached_aux", slot), lhs))
        return assignments
