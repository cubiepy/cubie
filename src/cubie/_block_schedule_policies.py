"""Ordering policies for the typed-IR block scheduler.

Pure graph computations over :class:`ScheduleNode` records — no CUDA
backend imports — so the policies are unit-testable everywhere. The
backend-facing pass in :mod:`cubie._typed_block_scheduler` builds the
dependency DAG from typed Numba IR and delegates ordering here.

A node's ``statement_kind`` is the IR statement class name; ``"Del"``
nodes are lifetime pins that splice back after their last emitted
predecessor rather than participating in the postorder walks.
"""

import heapq

__all__ = [
    "BLOCK_SCHEDULE_POLICIES",
    "ScheduleNode",
    "modeled_peak",
    "order_nodes",
]

BLOCK_SCHEDULE_POLICIES = (
    "source",
    "dfs",
    "anchor_dfs",
    "liveness",
    "longlived_dfs",
    "inject",
)


class ScheduleNode:
    """One schedulable statement in a block's dependency DAG.

    Attributes
    ----------
    index
        Position in the movable statement list.
    statement_kind
        IR statement class name (``"Assign"``, ``"Del"``, ...).
    defs, uses
        Names the statement defines and reads.
    successors, predecessors
        Dependency edges as node indices.
    memory
        ``(kind, root, index)`` classification for memory-touching
        statements, or ``None``.
    """

    __slots__ = (
        "index",
        "statement_kind",
        "defs",
        "uses",
        "successors",
        "predecessors",
        "memory",
    )

    def __init__(self, index, statement_kind, defs, uses):
        self.index = index
        self.statement_kind = statement_kind
        self.defs = defs
        self.uses = uses
        self.successors = set()
        self.predecessors = set()
        self.memory = None

    def add_edge_to(self, after):
        """Add a dependency edge from this node to ``after``."""
        if self is after:
            return
        if after.index not in self.successors:
            self.successors.add(after.index)
            after.predecessors.add(self.index)

    @property
    def is_del(self):
        """Whether this node is a Del lifetime pin."""
        return self.statement_kind == "Del"


def modeled_peak(nodes, order, scalar_names, live_out):
    """Peak live scalar count over an order; live-out names never die."""

    order = list(order)
    defined = set()
    for node in nodes:
        defined.update(
            name for name in node.defs if name in scalar_names
        )
    last_use = {}
    for position, index in enumerate(order):
        node = nodes[index]
        if node.is_del:
            continue
        for name in node.uses:
            if name in defined:
                last_use[name] = position
    live = 0
    peak = 0
    for position, index in enumerate(order):
        node = nodes[index]
        for name in node.defs:
            if name in defined:
                live += 1
        if not node.is_del:
            for name in node.uses:
                if (
                    name in defined
                    and name not in live_out
                    and last_use.get(name) == position
                ):
                    live -= 1
        peak = max(peak, live)
    return peak


def order_nodes(nodes, policy, live_out):
    """Order a block's dependency DAG under the selected policy.

    Returns the node-index order, or ``None`` when the DAG cannot be
    fully scheduled (a defensive signal; the caller keeps source
    order).
    """
    if policy == "liveness":
        return _order_liveness(nodes, live_out)
    return _order_dfs(nodes, live_out, policy)


def _order_dfs(nodes, live_out, policy):
    """Predecessor postorder; Dels splice after their predecessor.

    ``anchor_dfs`` roots every store, barrier, and terminal in source
    order so per-address store chains stay contiguous; ``dfs`` roots
    terminals only; ``longlived_dfs`` visits live-out-defining roots
    first.
    """

    def is_del(index):
        return nodes[index].is_del

    def is_terminal(node):
        return not any(
            not is_del(successor) for successor in node.successors
        )

    if policy == "anchor_dfs":
        root_nodes = [
            node
            for node in nodes
            if not node.is_del
            and (
                (
                    node.memory is not None
                    and node.memory[0] in ("store", "barrier")
                )
                or is_terminal(node)
            )
        ]
    else:
        root_nodes = [
            node
            for node in nodes
            if not node.is_del and is_terminal(node)
        ]
    if policy == "longlived_dfs":

        def root_key(node):
            defines_live_out = bool(node.defs & live_out)
            return (0 if defines_live_out else 1, node.index)

        root_nodes.sort(key=root_key)
    scheduled = [False] * len(nodes)
    order = []

    for root in root_nodes:
        if scheduled[root.index]:
            continue
        stack = [(root, None)]
        while stack:
            node, iterator = stack.pop()
            if iterator is None:
                if scheduled[node.index]:
                    continue
                iterator = iter(sorted(node.predecessors))
            advanced = False
            for predecessor_index in iterator:
                if not scheduled[predecessor_index] and not is_del(
                    predecessor_index
                ):
                    stack.append((node, iterator))
                    stack.append((nodes[predecessor_index], None))
                    advanced = True
                    break
            if not advanced and not scheduled[node.index]:
                scheduled[node.index] = True
                order.append(node.index)

    return _splice_dels(nodes, order)


def _splice_dels(nodes, order):
    """Insert each Del directly after its last emitted predecessor."""

    position = {index: pos for pos, index in enumerate(order)}
    insertions = {}
    for node in nodes:
        if not node.is_del:
            continue
        anchor = max(
            (
                position[predecessor]
                for predecessor in node.predecessors
                if predecessor in position
            ),
            default=-1,
        )
        insertions.setdefault(anchor, []).append(node.index)
    final = list(insertions.pop(-1, ()))
    for pos, index in enumerate(order):
        final.append(index)
        final.extend(insertions.pop(pos, ()))
    if insertions or len(final) != len(nodes):
        return None
    return final


def _order_liveness(nodes, live_out):
    """Greedy list schedule minimising the live-value count.

    Ready statements run best ``opened - closed`` balance first,
    original position as tie-break; stale heap entries re-score on
    pop.
    """

    # Count only non-Del uses; Dels chase their last use.
    remaining_uses = {}
    for node in nodes:
        if node.is_del:
            continue
        for name in node.uses:
            remaining_uses[name] = remaining_uses.get(name, 0) + 1

    def score(node):
        if node.is_del:
            return -len(nodes)
        closes = sum(
            1
            for name in node.uses
            if remaining_uses.get(name, 0) == 1
            and name not in live_out
        )
        opens = sum(
            1 for name in node.defs if name not in live_out
        )
        return opens - closes

    pending = {
        node.index: len(node.predecessors) for node in nodes
    }
    ready = [
        (score(node), node.index)
        for node in nodes
        if pending[node.index] == 0
    ]
    heapq.heapify(ready)
    order = []
    scheduled = [False] * len(nodes)
    while ready:
        stale_score, index = heapq.heappop(ready)
        if scheduled[index]:
            continue
        node = nodes[index]
        current_score = score(node)
        if current_score != stale_score:
            heapq.heappush(ready, (current_score, index))
            continue
        scheduled[index] = True
        order.append(index)
        if not node.is_del:
            for name in node.uses:
                remaining_uses[name] -= 1
        for successor_index in node.successors:
            pending[successor_index] -= 1
            if pending[successor_index] == 0:
                successor = nodes[successor_index]
                heapq.heappush(
                    ready, (score(successor), successor_index)
                )
    if len(order) != len(nodes):
        return None
    return order
