"""Unit tests for the pure block-schedule ordering policies."""

import pytest

from cubie._block_schedule_policies import (
    BLOCK_SCHEDULE_POLICIES,
    ScheduleNode,
    modeled_peak,
    order_nodes,
)
from cubie._env import (
    active_block_schedule,
    set_active_block_schedule,
)


def build_nodes(spec):
    """Build ScheduleNode records from (kind, defs, uses) triples."""
    nodes = [
        ScheduleNode(index, kind, set(defs), set(uses))
        for index, (kind, defs, uses) in enumerate(spec)
    ]
    def_site = {}
    for node in nodes:
        for name in node.defs:
            def_site[name] = node
    for node in nodes:
        for name in node.uses:
            site = def_site.get(name)
            if site is not None and site.index < node.index:
                site.add_edge_to(node)
        if node.statement_kind == "Del":
            for other in nodes[: node.index]:
                if node.uses & (other.defs | other.uses):
                    other.add_edge_to(node)
    return nodes


DIAMOND = [
    ("Assign", ["a"], ["x"]),
    ("Assign", ["b"], ["a"]),
    ("Assign", ["c"], ["a"]),
    ("Assign", ["d"], ["b", "c"]),
    ("Del", [], ["a"]),
]


def _assert_valid_topological(nodes, order):
    assert order is not None
    assert sorted(order) == list(range(len(nodes)))
    position = {index: pos for pos, index in enumerate(order)}
    for node in nodes:
        for successor in node.successors:
            assert position[successor] > position[node.index]


@pytest.mark.parametrize(
    "policy", ["dfs", "anchor_dfs", "liveness", "longlived_dfs"]
)
def test_policies_emit_valid_topological_orders(policy):
    """Every policy emits a dependency-respecting permutation."""
    nodes = build_nodes(DIAMOND)
    order = order_nodes(nodes, policy, frozenset())
    _assert_valid_topological(nodes, order)


def test_del_splices_after_last_predecessor():
    """A Del lands immediately after its final referencing statement."""
    nodes = build_nodes(DIAMOND)
    order = order_nodes(nodes, "dfs", frozenset())
    del_pos = order.index(4)
    last_ref = max(order.index(1), order.index(2))
    assert del_pos == last_ref + 1


def test_anchor_dfs_roots_stores_in_source_order():
    """Store anchors pull their cones in source order."""
    spec = [
        ("Assign", ["a"], []),
        ("Assign", ["b"], []),
        ("SetItem", [], ["a"]),
        ("SetItem", [], ["b"]),
    ]
    nodes = build_nodes(spec)
    nodes[2].memory = ("store", "r", "0")
    nodes[3].memory = ("store", "r", "1")
    nodes[2].add_edge_to(nodes[3])
    order = order_nodes(nodes, "anchor_dfs", frozenset())
    _assert_valid_topological(nodes, order)
    # The first store's cone (a, store-a) precedes the second's.
    assert order.index(0) < order.index(1)
    assert order.index(2) < order.index(3)


def test_liveness_reduces_modeled_peak_on_interleavable_chains():
    """The greedy schedule retires chains instead of batching defs."""
    spec = []
    for chain in range(4):
        spec.append(("Assign", [f"v{chain}"], []))
    for chain in range(4):
        spec.append(("SetItem", [], [f"v{chain}"]))
    nodes = build_nodes(spec)
    scalars = {f"v{chain}" for chain in range(4)}
    source_peak = modeled_peak(
        nodes, range(len(nodes)), scalars, frozenset()
    )
    order = order_nodes(nodes, "liveness", frozenset())
    _assert_valid_topological(nodes, order)
    scheduled_peak = modeled_peak(nodes, order, scalars, frozenset())
    assert source_peak == 4
    assert scheduled_peak == 1


def test_modeled_peak_live_out_names_never_die():
    """Live-out names stay live through the whole block."""
    spec = [
        ("Assign", ["a"], []),
        ("Assign", ["b"], ["a"]),
        ("SetItem", [], ["b"]),
    ]
    nodes = build_nodes(spec)
    scalars = {"a", "b"}
    # Uses retire before the peak is taken at each position.
    assert modeled_peak(
        nodes, range(3), scalars, frozenset()
    ) == 1
    # Live-out "a" survives its last use.
    assert modeled_peak(
        nodes, range(3), scalars, frozenset({"a"})
    ) == 2
    order = order_nodes(nodes, "liveness", frozenset())
    _assert_valid_topological(nodes, order)


def test_policy_names_are_stable():
    """The registered policy vocabulary is fixed."""
    assert BLOCK_SCHEDULE_POLICIES == (
        "source",
        "dfs",
        "anchor_dfs",
        "liveness",
        "longlived_dfs",
        "inject",
    )


def test_active_block_schedule_folds_into_cache_fingerprint():
    """The cache fingerprint carries the active schedule policy."""
    from cubie.cubie_cache import _abi_fingerprint_entries

    previous = active_block_schedule()
    try:
        set_active_block_schedule("anchor_dfs")
        entries = _abi_fingerprint_entries()
        assert "block-schedule=anchor_dfs" in entries
        set_active_block_schedule("source")
        entries = _abi_fingerprint_entries()
        assert "block-schedule=source" in entries
    finally:
        set_active_block_schedule(previous)
