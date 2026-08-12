"""Auxiliary caching planner for JVP solver helpers.

Selects v-independent assignments in the JVP graph (named
auxiliaries, Jacobian entries, ``_cse`` locals) for once-per-step
caching. A cached value is read from a ``cached_aux`` buffer slot
in the runtime operator body and computed in the ``prepare_jac``
fill. The planner greedily adds the candidate removing the most
device-weighted runtime work per operator evaluation, cascading
removal into dependencies left without live consumers, until the
slot limit is reached or no candidate saves ``min_ops_threshold``
weighted operations. Ties prefer larger runtime-body removals,
then the earlier assignment.

Published Classes
-----------------
:class:`CacheGroup`
    Frozen attrs container describing one greedy addition: the leaf
    cached and the marginal savings it contributed.

:class:`CacheSelection`
    Frozen attrs container capturing the final cache plan: which
    leaves to cache, which nodes to remove from runtime, and the
    estimated savings.

Published Functions
-------------------
:func:`plan_auxiliary_cache`
    Analyse a :class:`~cubie.odesystems.symbolic.parsing.jvp_equations.JVPEquations`
    instance and persist the computed cache plan.

See Also
--------
:class:`~cubie.odesystems.symbolic.parsing.jvp_equations.JVPEquations`
    Owns the dependency metadata consumed by this module.
:mod:`cubie.odesystems.symbolic.codegen.linear_operators`
    Generates cached linear operator code using the cache plan.
"""

from typing import Dict, List, Sequence, Set, Tuple

import attrs

from cubie.odesystems.symbolic.parsing.jvp_equations import JVPEquations

# Candidates examined per greedy round are capped at this multiple of
# the slot limit (ranked by cumulative cost) so planning stays fast on
# very large systems.
_CANDIDATE_CAP_FACTOR = 8


@attrs.frozen
class CacheGroup:
    """Describe one greedy addition to the cache plan.

    Parameters
    ----------
    seed
        The auxiliary symbol cached by this addition.
    leaves
        Leaves cached so far, including this addition.
    removal
        Symbols removed from runtime evaluation after this addition.
    prepare
        Symbols evaluated when populating the cache after this
        addition.
    saved
        Marginal device-weighted runtime operations removed by this
        addition.
    fill_cost
        Total cache-fill cost after this addition.
    """

    seed = attrs.field()
    leaves = attrs.field(converter=tuple)
    removal = attrs.field(converter=tuple)
    prepare = attrs.field(converter=tuple)
    saved = attrs.field()
    fill_cost = attrs.field()


@attrs.frozen
class CacheSelection:
    """Capture the final auxiliary cache plan.

    Parameters
    ----------
    groups
        Greedy additions in selection order.
    cached_leaves
        Auxiliary symbols whose values are cached.
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
    """

    groups = attrs.field(converter=tuple)
    cached_leaves = attrs.field(converter=tuple)
    cached_leaf_order = attrs.field(converter=tuple)
    removal_nodes = attrs.field(converter=tuple)
    runtime_nodes = attrs.field(converter=tuple)
    prepare_nodes = attrs.field(converter=tuple)
    saved = attrs.field()
    fill_cost = attrs.field()


def _candidate_symbols(equations: JVPEquations) -> list:
    """Return cache candidates ranked by descending cumulative cost.

    Every v-independent auxiliary feeding the JVP outputs qualifies;
    the list is capped at a multiple of the slot limit.
    """
    total_cost = equations.total_ops_cost
    order_idx = equations.order_index
    v_dependent = equations.v_dependent_nodes
    candidates = [
        symbol
        for symbol in equations.non_jvp_order
        if symbol not in v_dependent
        and equations.jvp_closure_usage.get(symbol, 0) > 0
    ]
    candidates.sort(
        key=lambda symbol: (
            -total_cost.get(symbol, 0),
            order_idx.get(symbol, len(order_idx)),
        )
    )
    cap = max(1, _CANDIDATE_CAP_FACTOR * equations.cache_slot_limit)
    return candidates[:cap]


def _cascade_removal(
    leaf,
    ref_counts: Dict,
    removed: Set,
    ops_cost,
    dependencies,
) -> Tuple[int, List, List]:
    """Remove ``leaf``'s runtime computation and cascade into dead deps.

    Mutates ``ref_counts`` and ``removed`` in place; consumers of
    ``leaf`` stay live.

    Returns
    -------
    tuple
        ``(saved, added, touched)`` — weighted operations removed,
        nodes newly removed, and dependencies decremented (the undo
        log for :func:`_undo_removal`).
    """
    saved = 0
    added: List = []
    touched: List = []
    stack = [leaf]
    while stack:
        node = stack.pop()
        if node in removed:
            continue
        removed.add(node)
        added.append(node)
        saved += ops_cost.get(node, 0)
        for dep in dependencies.get(node, ()):
            ref_counts[dep] -= 1
            touched.append(dep)
            if ref_counts[dep] == 0 and dep not in removed:
                stack.append(dep)
    return saved, added, touched


def _undo_removal(
    added: Sequence,
    touched: Sequence,
    ref_counts: Dict,
    removed: Set,
) -> None:
    """Reverse a :func:`_cascade_removal` trial."""
    for dep in touched:
        ref_counts[dep] += 1
    for node in added:
        removed.discard(node)


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
        groups=tuple(),
        cached_leaves=tuple(),
        cached_leaf_order=tuple(),
        removal_nodes=tuple(),
        runtime_nodes=tuple(equations.non_jvp_order),
        prepare_nodes=tuple(),
        saved=0,
        fill_cost=0,
    )


def plan_auxiliary_cache(equations: JVPEquations) -> CacheSelection:
    """Compute and persist the auxiliary cache plan for ``equations``.

    Greedily grows the cached-leaf set by the candidate with the
    largest marginal device-weighted runtime saving until the slot
    limit is reached or no candidate saves at least
    ``min_ops_threshold`` per operator evaluation.

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
    min_ops = equations.min_ops_threshold
    order_idx = equations.order_index
    ops_cost = equations.ops_cost
    dependencies = equations.dependencies

    selection = _empty_selection(equations)
    if slot_limit <= 0:
        equations.update_cache_selection(selection)
        return selection

    candidates = _candidate_symbols(equations)
    if not candidates:
        equations.update_cache_selection(selection)
        return selection

    ref_counts = dict(equations.reference_counts)
    removed: Set = set()
    chosen: List = []
    groups: List = []
    total_saved = 0

    while len(chosen) < slot_limit:
        best_symbol = None
        best_key = None
        best_saved = 0
        for symbol in candidates:
            if symbol in removed:
                # Already cached, or already removed as a dead
                # dependency of the cached set.
                continue
            saved, added, touched = _cascade_removal(
                symbol, ref_counts, removed, ops_cost, dependencies
            )
            key = (
                -saved,
                -len(added),
                order_idx.get(symbol, len(order_idx)),
            )
            _undo_removal(added, touched, ref_counts, removed)
            if saved < min_ops:
                continue
            if best_key is None or key < best_key:
                best_key = key
                best_symbol = symbol
                best_saved = saved
        if best_symbol is None:
            break
        _cascade_removal(
            best_symbol, ref_counts, removed, ops_cost, dependencies
        )
        chosen.append(best_symbol)
        total_saved += best_saved
        prepare = _prepare_closure(equations, chosen)
        fill_cost = sum(ops_cost.get(node, 0) for node in prepare)
        groups.append(
            CacheGroup(
                seed=best_symbol,
                leaves=tuple(chosen),
                removal=tuple(sorted(removed, key=order_idx.get)),
                prepare=tuple(sorted(prepare, key=order_idx.get)),
                saved=best_saved,
                fill_cost=fill_cost,
            )
        )

    if not chosen:
        equations.update_cache_selection(selection)
        return selection

    prepare = _prepare_closure(equations, chosen)
    fill_cost = sum(ops_cost.get(node, 0) for node in prepare)
    cached_order = tuple(sorted(chosen, key=order_idx.get))
    runtime_nodes = tuple(
        symbol
        for symbol in equations.non_jvp_order
        if symbol not in removed
    )
    selection = CacheSelection(
        groups=tuple(groups),
        cached_leaves=cached_order,
        cached_leaf_order=cached_order,
        removal_nodes=tuple(sorted(removed, key=order_idx.get)),
        runtime_nodes=runtime_nodes,
        prepare_nodes=tuple(sorted(prepare, key=order_idx.get)),
        saved=total_saved,
        fill_cost=fill_cost,
    )
    equations.update_cache_selection(selection)
    return selection
