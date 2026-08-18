"""Order, prune, and deduplicate IR assignments."""

from heapq import heapify, heappop, heappush
from typing import Dict, Iterable, List, Optional, Set, Tuple

from cubie._env import operation_ordering_default
from cubie.odesystems.symbolic.engine.expr import (
    DEVICE_WEIGHT_DIVIDE,
    DEVICE_WEIGHT_TRANSCENDENTAL,
    NEG_ONE,
    Add as AddNode,
    Arr,
    BoolConst,
    Call,
    Expr,
    Local,
    Mul as MulNode,
    Num,
    Pow as PowNode,
    Sym,
    _children,
    _rebuild,
    add,
    free_atoms,
    local,
    mul,
    num,
    pow_,
    xreplace,
)

__all__ = [
    "topological_sort",
    "prune_unused",
    "cse_and_stack",
]

Assignment = Tuple[Expr, Expr]

# Minimum breadth-first liveness peak before rescheduling engages.
_RESCHEDULE_PEAK_THRESHOLD = 64
_OPERATION_ORDERINGS = (
    "kahn",
    "greedy",
    "dfs",
    "liveness_auto",
)


def _kahn_order(
    pairs: List[Assignment],
    dep_map: Dict[Expr, List[Expr]],
    consumers: Dict[Expr, List[Expr]],
    order_index: Dict[Expr, int],
) -> List[Expr]:
    """Return the stable breadth-first emission order.

    Raises
    ------
    ValueError
        When a dependency cycle prevents ordering.
    """
    incoming = {lhs: len(dep_map[lhs]) for lhs, _ in pairs}
    ready = [lhs for lhs, _ in pairs if incoming[lhs] == 0]
    order: List[Expr] = []
    cursor = 0
    while cursor < len(ready):
        current = ready[cursor]
        cursor += 1
        order.append(current)
        released = []
        for waiter in consumers.get(current, ()):
            incoming[waiter] -= 1
            if incoming[waiter] == 0:
                released.append(waiter)
        released.sort(key=order_index.__getitem__)
        ready.extend(released)

    if len(order) != len(pairs):
        remaining = {lhs for lhs, _ in pairs} - set(order)
        names = sorted(str(node) for node in remaining)
        raise ValueError(
            f"Circular dependency detected. Remaining symbols: {names}"
        )
    return order


def _dfs_order(
    pairs: List[Assignment],
    dep_map: Dict[Expr, List[Expr]],
    consumers: Dict[Expr, List[Expr]],
) -> List[Expr]:
    """Return the roots-first depth-first emission order."""
    roots = [lhs for lhs, _ in pairs if not consumers.get(lhs)]
    order: List[Expr] = []
    emitted: Set[Expr] = set()
    for root in roots:
        stack: List[Tuple[Expr, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                emitted.add(node)
                order.append(node)
                continue
            if node in emitted:
                continue
            stack.append((node, True))
            for dep in reversed(dep_map[node]):
                if dep not in emitted:
                    stack.append((dep, False))
    return order


def _greedy_order(
    pairs: List[Assignment],
    dep_map: Dict[Expr, List[Expr]],
    consumers: Dict[Expr, List[Expr]],
    order_index: Dict[Expr, int],
) -> List[Expr]:
    """Return an emission order that greedily minimises live values.

    Among ready assignments, pick the one leaving the fewest scalar
    temporaries live; ties break toward retiring more, then input
    order.
    """
    scalar = {
        lhs: not isinstance(lhs, Arr) for lhs, _ in pairs
    }
    opens = {
        lhs: int(scalar[lhs] and bool(consumers.get(lhs)))
        for lhs, _ in pairs
    }
    remaining_uses = {
        lhs: len(consumers.get(lhs, ())) for lhs, _ in pairs
    }
    closes = {
        lhs: sum(
            1
            for dep in dep_map[lhs]
            if scalar[dep] and remaining_uses[dep] == 1
        )
        for lhs, _ in pairs
    }
    unmet = {lhs: len(dep_map[lhs]) for lhs, _ in pairs}

    def key(node: Expr) -> Tuple[int, int, int]:
        # Net change in live count if emitted next.
        return (
            opens[node] - closes[node],
            -closes[node],
            order_index[node],
        )

    heap: List[Tuple[Tuple[int, int, int], Expr]] = [
        (key(lhs), lhs) for lhs, _ in pairs if unmet[lhs] == 0
    ]
    heapify(heap)
    emitted: Set[Expr] = set()
    order: List[Expr] = []
    while heap:
        entry_key, node = heappop(heap)
        # Skip stale heap entries.
        if node in emitted or entry_key != key(node):
            continue
        emitted.add(node)
        order.append(node)
        for dep in dep_map[node]:
            remaining_uses[dep] -= 1
            if remaining_uses[dep] == 1 and scalar[dep]:
                for waiter in consumers[dep]:
                    if waiter in emitted:
                        continue
                    closes[waiter] += 1
                    if unmet[waiter] == 0:
                        heappush(heap, (key(waiter), waiter))
        for waiter in consumers.get(node, ()):
            unmet[waiter] -= 1
            if unmet[waiter] == 0:
                heappush(heap, (key(waiter), waiter))
    return order


def _liveness_cost(
    order: List[Expr],
    dep_map: Dict[Expr, List[Expr]],
    consumers: Dict[Expr, List[Expr]],
) -> Tuple[int, int]:
    """Score an order by scalar liveness.

    A consumed non-array left-hand side is live from definition
    through its final consumer.

    Returns
    -------
    tuple of int
        Peak simultaneous live count and total live-range area.
    """
    position = {node: i for i, node in enumerate(order)}
    last_use: Dict[int, List[Expr]] = {}
    for node in order:
        if isinstance(node, Arr) or not consumers.get(node):
            continue
        final = max(position[c] for c in consumers[node])
        last_use.setdefault(final, []).append(node)

    live = 0
    peak = 0
    area = 0
    for index, node in enumerate(order):
        if not isinstance(node, Arr) and consumers.get(node):
            live += 1
        if live > peak:
            peak = live
        live -= len(last_use.get(index, ()))
        area += live
    return peak, area


def topological_sort(
    assignments: Iterable[Assignment],
    operation_ordering: str = operation_ordering_default(),
) -> List[Assignment]:
    """Order assignments according to the requested dependency policy.

    ``"kahn"`` preserves the stable breadth-first order. ``"greedy"``
    always uses remaining-use greedy ordering, while ``"dfs"`` always
    uses roots-first depth-first ordering. ``"liveness_auto"`` ranks
    the greedy and DFS alternatives by peak live count then live-range
    area. Above ``_RESCHEDULE_PEAK_THRESHOLD``, the best alternative
    replaces Kahn only when it strictly lowers Kahn's peak. Only whole
    assignments move; expression trees are untouched.

    Parameters
    ----------
    assignments
        ``(lhs, rhs)`` pairs; each ``lhs`` is a :class:`Sym` or
        :class:`Arr` node.
    operation_ordering
        Dependency ordering policy: ``"liveness_auto"``, ``"kahn"``,
        ``"greedy"``, or ``"dfs"``. Defaults to
        ``CUBIE_OPERATION_ORDERING``.

    Returns
    -------
    list of tuple
        Dependency-ordered assignments. Ties break by input order,
        never by hash order.

    Raises
    ------
    ValueError
        When the ordering policy is invalid or a dependency cycle
        prevents ordering.
    """
    if operation_ordering not in _OPERATION_ORDERINGS:
        raise ValueError(
            "operation_ordering must be one of "
            f"{_OPERATION_ORDERINGS}; got {operation_ordering!r}"
        )
    pairs = list(assignments)
    sym_map: Dict[Expr, Expr] = {lhs: rhs for lhs, rhs in pairs}
    if len(sym_map) != len(pairs):
        seen: Dict[Expr, int] = {}
        for lhs, _ in pairs:
            seen[lhs] = seen.get(lhs, 0) + 1
        duplicates = sorted(
            str(lhs) for lhs, count in seen.items() if count > 1
        )
        raise ValueError(
            f"Duplicate assignment targets: {duplicates}"
        )
    order_index = {lhs: i for i, (lhs, _) in enumerate(pairs)}

    dep_map: Dict[Expr, List[Expr]] = {}
    consumers: Dict[Expr, List[Expr]] = {}
    for lhs, rhs in pairs:
        deps = free_atoms(rhs) & sym_map.keys()
        dep_map[lhs] = sorted(deps, key=order_index.__getitem__)
        for dep in dep_map[lhs]:
            consumers.setdefault(dep, []).append(lhs)

    # The breadth-first pass runs first and owns cycle detection.
    kahn = _kahn_order(pairs, dep_map, consumers, order_index)
    if operation_ordering == "greedy":
        chosen = _greedy_order(
            pairs, dep_map, consumers, order_index
        )
    elif operation_ordering == "dfs":
        chosen = _dfs_order(pairs, dep_map, consumers)
    else:
        chosen = kahn
    if operation_ordering == operation_ordering_default():
        kahn_peak, _ = _liveness_cost(kahn, dep_map, consumers)
        if kahn_peak > _RESCHEDULE_PEAK_THRESHOLD:
            alternatives = [
                _greedy_order(pairs, dep_map, consumers, order_index),
                _dfs_order(pairs, dep_map, consumers),
            ]
            best = min(
                alternatives,
                key=lambda order: _liveness_cost(
                    order, dep_map, consumers
                ),
            )
            best_peak, _ = _liveness_cost(best, dep_map, consumers)
            if best_peak < kahn_peak:
                chosen = best
    return [(lhs, sym_map[lhs]) for lhs in chosen]


def prune_unused(
    assignments: Iterable[Assignment],
    output_name: Optional[str] = None,
    output_symbols: Optional[Iterable[Expr]] = None,
) -> List[Assignment]:
    """Drop assignments that do not feed the requested outputs.

    Parameters
    ----------
    assignments
        Topologically ordered ``(lhs, rhs)`` pairs.
    output_name
        Array name identifying outputs: every ``Arr(output_name, i)``
        left-hand side is an output. Ignored when ``output_symbols``
        is given.
    output_symbols
        Explicit output left-hand sides to retain.

    Returns
    -------
    list of tuple
        The assignments transitively required by the outputs, in
        their original relative order. Returned unchanged when no
        output matches.
    """
    pairs = list(assignments)
    if not pairs:
        return pairs
    all_lhs = {lhs for lhs, _ in pairs}
    if output_symbols is not None:
        outputs = set(output_symbols) & all_lhs
    else:
        outputs = {
            lhs
            for lhs in all_lhs
            if isinstance(lhs, Arr) and lhs.name == output_name
        }
    if not outputs:
        return pairs

    used: Set[Expr] = set(outputs)
    kept: List[Assignment] = []
    for lhs, rhs in reversed(pairs):
        if lhs in used:
            kept.append((lhs, rhs))
            used.update(free_atoms(rhs) & all_lhs)
    kept.reverse()
    return kept


def _inline_atomic_assignments(
    pairs: List[Assignment],
) -> List[Assignment]:
    """Substitute literal- and alias-valued targets into later RHS.

    A scalar target assigned a numeric or boolean literal, symbol,
    or local propagates by value into every later right-hand side,
    so constant chains fold through the expression constructors and
    pure renames drop out of downstream expressions. Every
    assignment is kept — :func:`prune_unused` removes the ones this
    pass leaves unreferenced. Pairs are walked in list order, which
    the assignment contract requires to be evaluation order; the
    shared memo stays valid because a target only enters the
    mapping before any of its uses are walked.
    """
    mapping: Dict[Expr, Expr] = {}
    memo: Dict[Expr, Expr] = {}
    out: List[Assignment] = []
    for lhs, rhs in pairs:
        rhs = xreplace(rhs, mapping, memo)
        if isinstance(lhs, (Sym, Local)) and isinstance(
            rhs, (Num, Sym, Local, BoolConst)
        ):
            mapping[lhs] = rhs
        out.append((lhs, rhs))
    return out


# Rewrite cost of each derived-power form, in device weights; a
# derived power replaces one transcendental ``powf`` call.
_POW_RELATION_COSTS: Dict[str, int] = {
    "mul": 1,
    "div": DEVICE_WEIGHT_DIVIDE,
    "square": 1,
    "square_mul": 2,
    "square_div": 1 + DEVICE_WEIGHT_DIVIDE,
}


def _match_pow_relation(p: float, q: float) -> Optional[str]:
    """Return how ``base**q`` derives from ``base**p``, if it does.

    Exponent families come from differentiation (``p - 1``) and
    same-base product folding (``2p``, ``2p - 1``, ``2p + 1``), so
    matching is by exact float equality against each derivation's
    possible roundings.
    """
    if q == p + 1.0:
        return "mul"
    if q == p - 1.0:
        return "div"
    if q == 2.0 * p:
        return "square"
    if q == 2.0 * p + 1.0 or q == p + (p + 1.0):
        return "square_mul"
    if q == 2.0 * p - 1.0 or q == p + (p - 1.0):
        return "square_div"
    return None


def _build_pow_replacement(
    tag: str, primal: Expr, base: Expr
) -> Expr:
    """Build the derived-power expression for ``tag``."""
    if tag == "mul":
        return mul(primal, base)
    if tag == "div":
        return mul(primal, pow_(base, NEG_ONE))
    if tag == "square":
        return pow_(primal, num(2))
    if tag == "square_mul":
        return mul(pow_(primal, num(2)), base)
    return mul(pow_(primal, num(2)), pow_(base, NEG_ONE))


def _is_costly_pow(node: Expr) -> bool:
    """Return whether ``node`` prints as a transcendental power."""
    if not isinstance(node, PowNode):
        return False
    exp = node.exp
    if not isinstance(exp, Num):
        return False
    value = exp.value
    if not isinstance(value, float):
        return False
    if value.is_integer() or abs(value) == 0.5:
        return False
    return True


def _collect_pow_families(
    pairs: List[Assignment],
) -> Tuple[Dict[Expr, Dict[float, Expr]], List[Expr]]:
    """Group costly power nodes by base in first-appearance order.

    Bases that are themselves ``Pow`` nodes are skipped: the
    reciprocal in a derived form would fold back into a
    transcendental power of the inner base.
    """
    families: Dict[Expr, Dict[float, Expr]] = {}
    base_order: List[Expr] = []
    seen: Set[Expr] = set()

    def walk(node: Expr) -> None:
        if node in seen:
            return
        seen.add(node)
        if _is_costly_pow(node) and not isinstance(
            node.base, PowNode
        ):
            members = families.get(node.base)
            if members is None:
                members = {}
                families[node.base] = members
                base_order.append(node.base)
            members[node.exp.value] = node
        for child in _children(node):
            walk(child)

    for _, rhs in pairs:
        walk(rhs)
    return families, base_order


def _reduce_pow_families(
    pairs: List[Assignment],
    allocate_name,
) -> List[Assignment]:
    """Derive related non-integer powers from one named primal.

    Differentiation and product folding emit ``x**(p - 1)``,
    ``x**(2p)``, and ``x**(2p ± 1)`` alongside ``x**p``. Each is a
    full transcendental power; naming the primal as a local turns
    every derived member into a multiply or divide of that local.
    The primal with the largest device-weight saving wins each
    round, so multi-primal families reduce fully.

    Parameters
    ----------
    pairs
        Assignment pairs to rewrite.
    allocate_name
        Zero-argument callable returning a fresh local name.

    Returns
    -------
    list of tuple
        Rewritten pairs with primal-power assignments appended;
        :func:`topological_sort` orders them before their uses.
    """
    families, base_order = _collect_pow_families(pairs)
    # First whole-RHS owner of each candidate power node: a primal
    # with an owner reuses that assignment's target instead of
    # gaining a fresh local and leaving a pure rename behind.
    owners: Dict[Expr, int] = {}
    for index, (lhs, rhs) in enumerate(pairs):
        if (
            isinstance(rhs, PowNode)
            and isinstance(lhs, (Sym, Local))
            and rhs not in owners
        ):
            owners[rhs] = index
    replacements: Dict[Expr, Expr] = {}
    primal_pairs: List[Assignment] = []
    preserved: Dict[int, Expr] = {}
    for base in base_order:
        available = dict(families[base])
        while len(available) >= 2:
            best = None
            for p in sorted(available):
                saving = 0
                derived: List[Tuple[float, str]] = []
                for q in sorted(available):
                    if q == p:
                        continue
                    tag = _match_pow_relation(p, q)
                    if tag is None:
                        continue
                    saving += (
                        DEVICE_WEIGHT_TRANSCENDENTAL
                        - _POW_RELATION_COSTS[tag]
                    )
                    derived.append((q, tag))
                if derived and (best is None or saving > best[0]):
                    best = (saving, p, derived)
            if best is None:
                break
            _, p, derived = best
            node = available[p]
            owner = owners.get(node)
            if owner is None:
                primal = local(allocate_name())
                primal_pairs.append((primal, node))
            else:
                primal = pairs[owner][0]
                preserved[owner] = node
            replacements[node] = primal
            del available[p]
            for q, tag in derived:
                replacements[available[q]] = (
                    _build_pow_replacement(tag, primal, base)
                )
                del available[q]
    if not replacements:
        return pairs
    # A primal's base may itself contain a replaced power from
    # another family; rebuild the primal assignment from the
    # replaced base so it shares that family's local too. The
    # primal node itself maps to its own target, so only children
    # are walked.
    memo: Dict[Expr, Expr] = {}

    def resolve_primal(node: Expr) -> Expr:
        return pow_(
            xreplace(node.base, replacements, memo), node.exp
        )

    rewritten: List[Assignment] = []
    for index, (lhs, rhs) in enumerate(pairs):
        node = preserved.get(index)
        if node is not None:
            rewritten.append((lhs, resolve_primal(node)))
        else:
            rewritten.append(
                (lhs, xreplace(rhs, replacements, memo))
            )
    resolved_primals = [
        (lhs, resolve_primal(rhs)) for lhs, rhs in primal_pairs
    ]
    return rewritten + resolved_primals


def _is_extractable(node: Expr) -> bool:
    """Return whether a shared node is worth naming as a CSE local."""
    if isinstance(node, (Sym, Local, Arr, Num, BoolConst)):
        return False
    if isinstance(node, Call) and not node.args:
        return False
    return True


# Args appearing in more than this many products/sums are too generic
# to seed subset matching (think ``h`` multiplying every JVP term);
# pairing them would cost O(n^2) for near-zero sharing value.
_SUBSET_PAIR_CAP = 100


def _find_partial_subsets(
    nodes: List[Expr],
    raw_build,
) -> Dict[Expr, Tuple[Expr, Tuple[Expr, ...]]]:
    """Match shared argument subsets across n-ary Add/Mul nodes.

    Flattening destroys nested sharing: ``2*e*a`` interns as
    ``Mul(2, e, a)``, which does not contain the shared ``Mul(e, a)``
    as a child. This pass finds argument subsets (size >= 2, numeric
    coefficients excluded) common to at least two nodes and assigns
    each node its largest such subset.

    Parameters
    ----------
    nodes
        Distinct Add or Mul nodes in first-appearance order.
    raw_build
        Constructor building the interned subset node from a tuple of
        two-or-more args.

    Returns
    -------
    dict
        Mapping from node to ``(subset_node, remaining_args)``.
    """
    arg_sets: Dict[Expr, frozenset] = {}
    by_arg: Dict[Expr, List[int]] = {}
    for position, node in enumerate(nodes):
        significant = frozenset(
            a for a in node.args if not isinstance(a, Num)
        )
        arg_sets[node] = significant
        for argument in significant:
            by_arg.setdefault(argument, []).append(position)

    # Candidate subsets from pairwise intersections, discovered via
    # the shared-argument index so unrelated nodes never pair up.
    best: Dict[int, frozenset] = {}
    seen_pairs: Set[Tuple[int, int]] = set()
    for argument, positions in by_arg.items():
        if len(positions) < 2 or len(positions) > _SUBSET_PAIR_CAP:
            continue
        for i, pos_a in enumerate(positions):
            set_a = arg_sets[nodes[pos_a]]
            for pos_b in positions[i + 1:]:
                pair = (pos_a, pos_b)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                common = set_a & arg_sets[nodes[pos_b]]
                if len(common) < 2:
                    continue
                for position in pair:
                    node_args = arg_sets[nodes[position]]
                    if common == node_args and len(
                        nodes[position].args
                    ) == len(common):
                        # The node IS the subset; plain counting
                        # already handles whole-node sharing.
                        continue
                    current = best.get(position)
                    if current is None or len(common) > len(current):
                        best[position] = common
                    elif current is not None and len(common) == len(
                        current
                    ) and common is not current:
                        # Deterministic tie-break by sort key.
                        chosen = min(
                            (
                                tuple(
                                    sorted(
                                        a.sort_key for a in common
                                    )
                                ),
                                common,
                            ),
                            (
                                tuple(
                                    sorted(
                                        a.sort_key for a in current
                                    )
                                ),
                                current,
                            ),
                        )[1]
                        best[position] = chosen

    adopted: Dict[Expr, Tuple[Expr, Tuple[Expr, ...]]] = {}
    for position, subset in best.items():
        node = nodes[position]
        subset_args = tuple(
            sorted(subset, key=lambda a: a.sort_key)
        )
        subset_node = raw_build(subset_args)
        if subset_node is node:
            continue
        remaining = tuple(
            a for a in node.args if a not in subset
        )
        adopted[node] = (subset_node, remaining)
    return adopted


def cse_and_stack(
    assignments: Iterable[Assignment],
    symbol: Optional[str] = None,
    operation_ordering: str = operation_ordering_default(),
) -> List[Assignment]:
    """Extract shared subexpressions and return ordered assignments.

    Parameters
    ----------
    assignments
        ``(lhs, rhs)`` pairs defining the computation.
    symbol
        Prefix for generated locals. Defaults to ``"_cse"``.
        Numbering continues after any existing ``<symbol><n>``
        left-hand sides.
    operation_ordering
        Dependency ordering policy forwarded to
        :func:`topological_sort`.

    Returns
    -------
    list of tuple
        Dependency-ordered assignments in which every IR node that is
        referenced more than once (and is worth naming) is assigned
        to a ``<symbol><n>`` local and reused by name.

    Notes
    -----
    Piecewise conditions are extracted like any other shared node;
    booleans are valid locals in the generated source. Before
    extraction, literal- and alias-valued targets inline into later
    right-hand sides so constant chains fold; after extraction, the
    inlining repeats to collapse extraction-created renames and
    related non-integer powers reduce to one named primal per
    family. Assignments those passes leave unreferenced are dropped
    by the callers' :func:`prune_unused`.
    """
    if symbol is None:
        symbol = "_cse"
    # The inline pass needs evaluation order to see a target's
    # definition before its uses; incoming equation lists are not
    # guaranteed to provide it.
    pairs = topological_sort(list(assignments), "kahn")
    pairs = _inline_atomic_assignments(pairs)

    used_names: Set[str] = set()
    visited_names: Set[Expr] = set()

    def record_names(node: Expr) -> None:
        if node in visited_names:
            return
        visited_names.add(node)
        if isinstance(node, (Sym, Local, Arr, Call)):
            used_names.add(node.name)
        for child in _children(node):
            record_names(child)

    for lhs, rhs in pairs:
        record_names(lhs)
        record_names(rhs)

    suffixes = [
        int(name[len(symbol):])
        for name in used_names
        if name.startswith(symbol) and name[len(symbol):].isdigit()
    ]
    next_index = max(suffixes, default=-1) + 1

    def allocate_name() -> str:
        nonlocal next_index
        name = f"{symbol}{next_index}"
        while name in used_names:
            next_index += 1
            name = f"{symbol}{next_index}"
        used_names.add(name)
        next_index += 1
        return name

    # Count references of every composite node across all RHS roots,
    # and record distinct Add/Mul nodes for subset matching.
    counts: Dict[Expr, int] = {}
    visited_roots: Set[Expr] = set()
    add_nodes: List[Expr] = []
    mul_nodes: List[Expr] = []

    def count(node: Expr) -> None:
        current = counts.get(node, 0)
        counts[node] = current + 1
        if current == 0:
            if type(node) is AddNode:
                add_nodes.append(node)
            elif type(node) is MulNode:
                mul_nodes.append(node)
            for child in _children(node):
                count(child)

    for _, rhs in pairs:
        # Each root counts once per assignment that uses it, so a
        # full RHS repeated across assignments is extracted too.
        if rhs in visited_roots:
            counts[rhs] += 1
            continue
        visited_roots.add(rhs)
        count(rhs)

    # Partial sharing: n-ary flattening hides subset reuse (e.g.
    # ``2*e*a`` vs ``e*a``); match subsets and count them as virtual
    # occurrences so they qualify for extraction.
    adopted: Dict[Expr, Tuple[Expr, Tuple[Expr, ...]]] = {}
    adopted.update(
        _find_partial_subsets(mul_nodes, lambda args: mul(*args))
    )
    adopted.update(
        _find_partial_subsets(add_nodes, lambda args: add(*args))
    )
    for node, (subset_node, _) in adopted.items():
        counts[subset_node] = (
            counts.get(subset_node, 0) + counts.get(node, 1)
        )

    shared = [
        node
        for node, n_refs in counts.items()
        if n_refs > 1 and _is_extractable(node)
    ]
    if not shared:
        combined = _reduce_pow_families(pairs, allocate_name)
        return topological_sort(combined, operation_ordering)
    shared_set = set(shared)

    # Drop adoptions whose subset did not end up shared, so the
    # rewrite phase does not restructure products for nothing.
    adopted = {
        node: (subset_node, remaining)
        for node, (subset_node, remaining) in adopted.items()
        if subset_node in shared_set
    }

    # Assign names in first-appearance order for deterministic and
    # readable output: walk the RHS roots in order, depth-first.
    name_order: List[Expr] = []
    seen: Set[Expr] = set()

    def collect(node: Expr) -> None:
        if node in seen:
            return
        seen.add(node)
        for child in _children(node):
            collect(child)
        adoption = adopted.get(node)
        if adoption is not None and adoption[0] not in seen:
            collect(adoption[0])
        if node in shared_set:
            name_order.append(node)

    for _, rhs in pairs:
        collect(rhs)

    replacements: Dict[Expr, Expr] = {}
    cse_assignments: List[Assignment] = []
    for node in name_order:
        temporary = local(allocate_name())
        replacements[node] = temporary

    def rewrite(node: Expr, memo: Dict[Expr, Expr]) -> Expr:
        cached = memo.get(node)
        if cached is not None:
            return cached
        adoption = adopted.get(node)
        if adoption is not None:
            subset_node, remaining = adoption
            local = replacements[subset_node]
            rebuilt_rest = tuple(
                _lookup(child, memo) for child in remaining
            )
            if type(node) is MulNode:
                result = mul(local, *rebuilt_rest)
            else:
                result = add(local, *rebuilt_rest)
        else:
            children = _children(node)
            if children:
                new_children = tuple(
                    _lookup(child, memo) for child in children
                )
                if new_children != children:
                    result = _rebuild(node, new_children)
                else:
                    result = node
            else:
                result = node
        memo[node] = result
        return result

    def _lookup(node: Expr, memo: Dict[Expr, Expr]) -> Expr:
        replacement = replacements.get(node)
        if replacement is not None:
            return replacement
        return rewrite(node, memo)

    memo: Dict[Expr, Expr] = {}
    for node in name_order:
        # Rewrite each extracted node's body in terms of previously
        # extracted locals (children first, so bodies nest properly).
        body = rewrite(node, memo)
        cse_assignments.append((replacements[node], body))

    rewritten_pairs: List[Assignment] = [
        (lhs, _lookup(rhs, memo)) for lhs, rhs in pairs
    ]

    combined = topological_sort(
        rewritten_pairs + cse_assignments, "kahn"
    )
    combined = _inline_atomic_assignments(combined)
    combined = _reduce_pow_families(combined, allocate_name)
    return topological_sort(combined, operation_ordering)
