"""Auxiliary caching planner for JVP solver helpers.

Selects v-independent assignments in the JVP graph (named
auxiliaries, Jacobian entries, ``_cse`` locals) for once-per-step
caching. A cached value is read from a ``cached_aux`` buffer slot
in the runtime operator body and computed in the ``prepare_jac``
fill. Selection is a maximum-weight closure problem solved by
min-cut: a removed node earns its device-weighted cost, each
cached slot charges ``read_price`` per operator evaluation, and a
removed node stays uncached only when every consumer is removed.
Over-cap plans are re-solved at bisected higher prices and trimmed
to ``cache_slot_limit``.

Published Classes
-----------------
:class:`CacheSelection`
    Frozen attrs container capturing the final cache plan: which
    leaves to cache, which nodes to remove from runtime, and the
    estimated costs.

Published Functions
-------------------
:func:`plan_auxiliary_cache`
    Analyse a :class:`~.jvp_equations.JVPEquations` instance and persist
    the computed cache plan.

See Also
--------
:class:`~cubie.odesystems.symbolic.parsing.jvp_equations.JVPEquations`
    Owns the dependency metadata consumed by this module.
:mod:`cubie.odesystems.symbolic.codegen.linear_operators`
    Generates cached linear operator code using the cache plan.
"""

from collections import deque
from typing import List, Sequence, Set, Tuple

import attrs

from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations


@attrs.frozen
class CacheSelection:
    """Capture the final auxiliary cache plan.

    Parameters
    ----------
    cached_leaf_order
        Cached leaves in evaluation order. Slot indices bind
        positionally to this order across the cached helper family.
    removal_nodes
        Symbols removed from runtime evaluation.
    runtime_nodes
        Symbols that remain in runtime evaluation.
    prepare_nodes
        Symbols evaluated when populating the cache.
    saved
        Total device-weighted operations removed from each runtime
        operator evaluation.
    fill_cost
        Device-weighted operations required to populate the cache
        once per step.
    duplicate_cost
        Device-weighted operations computed in both the fill and
        the runtime body (shared producers of cached leaves).
    read_price
        Per-slot read price the plan was solved at.
    """

    cached_leaf_order = attrs.field(converter=tuple)
    removal_nodes = attrs.field(converter=tuple)
    runtime_nodes = attrs.field(converter=tuple)
    prepare_nodes = attrs.field(converter=tuple)
    saved = attrs.field()
    fill_cost = attrs.field()
    duplicate_cost = attrs.field()
    read_price = attrs.field()


class _Network:
    """Dinic max-flow network; paired edges share ``index ^ 1``."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.adjacency: List[List[int]] = [[] for _ in range(size)]
        self.targets: List[int] = []
        self.residual: List[int] = []

    def add_edge(self, tail: int, head: int, capacity: int) -> None:
        """Add a directed edge and its zero-capacity reverse."""
        self.adjacency[tail].append(len(self.targets))
        self.targets.append(head)
        self.residual.append(capacity)
        self.adjacency[head].append(len(self.targets))
        self.targets.append(tail)
        self.residual.append(0)

    def max_flow(self, source: int, sink: int) -> int:
        """Return the maximum flow from ``source`` to ``sink``."""
        adjacency = self.adjacency
        targets = self.targets
        residual = self.residual
        flow = 0
        while True:
            level = [-1] * self.size
            level[source] = 0
            queue = deque((source,))
            while queue:
                node = queue.popleft()
                for edge in adjacency[node]:
                    head = targets[edge]
                    if residual[edge] > 0 and level[head] < 0:
                        level[head] = level[node] + 1
                        queue.append(head)
            if level[sink] < 0:
                return flow
            pointers = [0] * self.size
            path: List[int] = []
            node = source
            while True:
                if node == sink:
                    bottleneck = min(residual[e] for e in path)
                    for edge in path:
                        residual[edge] -= bottleneck
                        residual[edge ^ 1] += bottleneck
                    flow += bottleneck
                    for position, edge in enumerate(path):
                        if residual[edge] == 0:
                            del path[position:]
                            break
                    node = targets[path[-1]] if path else source
                    continue
                advanced = False
                while pointers[node] < len(adjacency[node]):
                    edge = adjacency[node][pointers[node]]
                    head = targets[edge]
                    if residual[edge] > 0 and level[head] == level[node] + 1:
                        path.append(edge)
                        node = head
                        advanced = True
                        break
                    pointers[node] += 1
                if advanced:
                    continue
                level[node] = -1
                if node == source:
                    break
                edge = path.pop()
                node = targets[edge ^ 1]
                pointers[node] += 1

    def source_side(self, source: int) -> List[bool]:
        """Return residual reachability from ``source`` after max-flow."""
        seen = [False] * self.size
        seen[source] = True
        stack = [source]
        while stack:
            node = stack.pop()
            for edge in self.adjacency[node]:
                head = self.targets[edge]
                if self.residual[edge] > 0 and not seen[head]:
                    seen[head] = True
                    stack.append(head)
        return seen


def _candidate_symbols(equations: JVPEquations) -> list:
    """Return cacheable symbols in evaluation order.

    Every v-independent auxiliary feeding the JVP outputs qualifies.
    """
    v_dependent = equations.v_dependent_nodes
    return [
        symbol
        for symbol in equations.non_jvp_order
        if symbol not in v_dependent
        and equations.jvp_closure_usage.get(symbol, 0) > 0
    ]


def _uncachable_symbols(equations: JVPEquations, candidate_set: Set) -> Set:
    """Return candidates readable only through a cached slot binding."""
    uncachable = set()
    for symbol in candidate_set:
        if equations.jvp_usage.get(symbol, 0) > 0:
            uncachable.add(symbol)
            continue
        for consumer in equations.dependents.get(symbol, ()):
            if consumer not in candidate_set:
                uncachable.add(symbol)
                break
    return uncachable


def _solve_cut(
    equations: JVPEquations,
    candidates: Sequence,
    candidate_set: Set,
    uncachable: Set,
    read_price: int,
) -> Tuple[List, List]:
    """Solve the closure problem at ``read_price`` via min-cut.

    Returns
    -------
    tuple of list, list
        ``(removed, cached)`` in evaluation order.
    """
    ops_cost = equations.ops_cost
    dependents = equations.dependents
    count = len(candidates)
    y_base = 2
    u_base = 2 + count
    index = {symbol: i for i, symbol in enumerate(candidates)}
    infinite = (
        sum(ops_cost.get(symbol, 0) for symbol in candidates)
        + read_price * count
        + 1
    )
    network = _Network(2 + 2 * count)
    for i, symbol in enumerate(candidates):
        weight = ops_cost.get(symbol, 0) - read_price
        if weight > 0:
            network.add_edge(0, y_base + i, weight)
        elif weight < 0:
            network.add_edge(y_base + i, 1, -weight)
        if symbol in uncachable:
            continue
        network.add_edge(0, u_base + i, read_price)
        network.add_edge(u_base + i, y_base + i, infinite)
        consumers = sorted(
            dependents.get(symbol, ()), key=lambda s: s.name
        )
        for consumer in consumers:
            network.add_edge(
                u_base + i, y_base + index[consumer], infinite
            )
    network.max_flow(0, 1)
    side = network.source_side(0)
    removed = [
        symbol
        for i, symbol in enumerate(candidates)
        if side[y_base + i]
    ]
    uncached = {
        symbol
        for i, symbol in enumerate(candidates)
        if side[u_base + i]
    }
    cached = [symbol for symbol in removed if symbol not in uncached]
    return removed, cached


def _removed_closure(
    equations: JVPEquations,
    candidate_set: Set,
    uncachable: Set,
    cached: Sequence,
) -> List:
    """Return the maximal removed set for a fixed cached frontier."""
    removed = set(cached)
    dependents = equations.dependents
    jvp_usage = equations.jvp_usage
    changed = True
    while changed:
        changed = False
        for symbol in reversed(equations.non_jvp_order):
            if (
                symbol in removed
                or symbol not in candidate_set
                or symbol in uncachable
                or jvp_usage.get(symbol, 0) > 0
            ):
                continue
            consumers = dependents.get(symbol, ())
            if consumers and all(c in removed for c in consumers):
                removed.add(symbol)
                changed = True
    return [
        symbol
        for symbol in equations.non_jvp_order
        if symbol in removed
    ]


def _trim_to_cap(
    equations: JVPEquations,
    candidate_set: Set,
    uncachable: Set,
    cached: Sequence,
    slot_limit: int,
) -> Tuple[List, List]:
    """Shrink an over-cap frontier by cheapest-loss slot removal."""
    ops_cost = equations.ops_cost
    cached = list(cached)
    removed = _removed_closure(
        equations, candidate_set, uncachable, cached
    )
    while len(cached) > slot_limit:
        best = None
        for leaf in cached:
            trial = [c for c in cached if c is not leaf]
            trial_removed = _removed_closure(
                equations, candidate_set, uncachable, trial
            )
            trial_saved = sum(
                ops_cost.get(symbol, 0) for symbol in trial_removed
            )
            key = (-trial_saved, leaf.name)
            if best is None or key < best[0]:
                best = (key, trial, trial_removed)
        cached = best[1]
        removed = best[2]
    return removed, cached


def _prepare_closure(equations: JVPEquations, leaves: Sequence) -> Set:
    """Return the dependency closure evaluated to fill the cache."""
    dependencies = equations.dependencies
    prepare: Set = set()
    stack = list(leaves)
    while stack:
        node = stack.pop()
        if node in prepare:
            continue
        prepare.add(node)
        stack.extend(dependencies.get(node, ()))
    return prepare


def _empty_selection(equations: JVPEquations) -> CacheSelection:
    """Return the no-caching plan."""
    return CacheSelection(
        cached_leaf_order=tuple(),
        removal_nodes=tuple(),
        runtime_nodes=tuple(equations.non_jvp_order),
        prepare_nodes=tuple(),
        saved=0,
        fill_cost=0,
        duplicate_cost=0,
        read_price=equations.read_price,
    )


def plan_auxiliary_cache(equations: JVPEquations) -> CacheSelection:
    """Compute and persist the auxiliary cache plan for ``equations``.

    Solves for the removed region and cached frontier maximising
    device-weighted runtime savings net of ``read_price`` per slot,
    within ``cache_slot_limit`` slots.

    Parameters
    ----------
    equations
        JVP equations to optimize with caching.

    Returns
    -------
    CacheSelection
        The computed cache plan, also stored in ``equations``.
    """
    slot_limit = equations.cache_slot_limit
    ops_cost = equations.ops_cost

    if slot_limit <= 0:
        selection = _empty_selection(equations)
        equations.update_cache_selection(selection)
        return selection

    candidates = _candidate_symbols(equations)
    if not candidates:
        selection = _empty_selection(equations)
        equations.update_cache_selection(selection)
        return selection

    candidate_set = set(candidates)
    uncachable = _uncachable_symbols(equations, candidate_set)
    price = equations.read_price
    removed, cached = _solve_cut(
        equations, candidates, candidate_set, uncachable, price
    )
    base_price = price
    if len(cached) > slot_limit:
        low = price
        high = (
            sum(ops_cost.get(symbol, 0) for symbol in candidates) + 1
        )
        feasible = None
        infeasible = (removed, cached)
        while low + 1 < high:
            mid = (low + high) // 2
            trial_removed, trial_cached = _solve_cut(
                equations, candidates, candidate_set, uncachable, mid
            )
            if len(trial_cached) <= slot_limit:
                high = mid
                feasible = (trial_removed, trial_cached, mid)
            else:
                low = mid
                infeasible = (trial_removed, trial_cached)
        trimmed_removed, trimmed_cached = _trim_to_cap(
            equations,
            candidate_set,
            uncachable,
            infeasible[1],
            slot_limit,
        )

        def net(removed_nodes, cached_nodes):
            return sum(
                ops_cost.get(symbol, 0) for symbol in removed_nodes
            ) - base_price * len(cached_nodes)

        removed = trimmed_removed
        cached = trimmed_cached
        price = base_price
        if feasible is not None:
            swept_removed, swept_cached, swept_price = feasible
            if net(swept_removed, swept_cached) >= net(removed, cached):
                removed = swept_removed
                cached = swept_cached
                price = swept_price
        if net(removed, cached) <= 0:
            cached = []

    if not cached:
        selection = _empty_selection(equations)
        equations.update_cache_selection(selection)
        return selection

    removed_set = set(removed)
    prepare = _prepare_closure(equations, cached)
    runtime_nodes = tuple(
        symbol
        for symbol in equations.non_jvp_order
        if symbol not in removed_set
    )
    duplicate = prepare.intersection(runtime_nodes)
    order_idx = equations.order_index
    selection = CacheSelection(
        cached_leaf_order=tuple(cached),
        removal_nodes=tuple(removed),
        runtime_nodes=runtime_nodes,
        prepare_nodes=tuple(sorted(prepare, key=order_idx.get)),
        saved=sum(ops_cost.get(symbol, 0) for symbol in removed),
        fill_cost=sum(ops_cost.get(symbol, 0) for symbol in prepare),
        duplicate_cost=sum(
            ops_cost.get(symbol, 0) for symbol in duplicate
        ),
        read_price=price,
    )
    equations.update_cache_selection(selection)
    return selection
