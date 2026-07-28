"""Graph-layer tests: bipartite graphs, matchings, digraph views."""


from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    DST,
    Matching,
    SRC,
    UNASSIGNED,
    maximal_matching,
)
from cubie.odesystems.symbolic.structural.digraph import (
    DiCMOBiGraphF,
    DiCMOBiGraphT,
    IncrementalCycleTracker,
    find_var_sccs,
    neighborhood_in,
    tarjan_scc,
    toposort_equations,
)


def build_graph(nsrcs, ndsts, edges):
    graph = BipartiteGraph(nsrcs, ndsts)
    for e, v in edges:
        graph.add_edge(e, v)
    return graph


class TestBipartiteGraph:
    def test_edges_sorted_and_counted(self):
        graph = build_graph(2, 3, [(0, 2), (0, 0), (1, 1)])
        assert graph.ne == 3
        assert graph.s_neighbors(0) == [0, 2]
        assert graph.d_neighbors(1) == [1]
        assert graph.has_edge(0, 2)
        assert not graph.has_edge(1, 2)

    def test_duplicate_edge_not_added(self):
        graph = build_graph(1, 1, [(0, 0)])
        assert not graph.add_edge(0, 0)
        assert graph.ne == 1

    def test_rem_edge_updates_both_sides(self):
        graph = build_graph(2, 2, [(0, 0), (0, 1), (1, 0)])
        graph.rem_edge(0, 0)
        assert graph.s_neighbors(0) == [1]
        assert graph.d_neighbors(0) == [1]
        assert graph.ne == 2

    def test_set_neighbors_syncs_backward(self):
        graph = build_graph(2, 3, [(0, 0), (0, 1)])
        graph.set_neighbors(0, [2, 1])
        assert graph.s_neighbors(0) == [1, 2]
        assert graph.d_neighbors(0) == []
        assert graph.d_neighbors(2) == [0]

    def test_invview_shares_edges(self):
        graph = build_graph(2, 2, [(0, 1)])
        inv = graph.invview()
        assert inv.s_neighbors(1) == [0]
        inv.add_edge(0, 1)
        assert graph.has_edge(1, 0)
        assert graph.ne == 2

    def test_add_vertex(self):
        graph = build_graph(1, 1, [(0, 0)])
        assert graph.add_vertex(SRC) == 1
        assert graph.add_vertex(DST) == 1
        assert graph.nsrcs() == 2
        assert graph.ndsts() == 2

    def test_delete_srcs_renumbers(self):
        graph = build_graph(3, 2, [(0, 0), (1, 1), (2, 0)])
        graph.delete_srcs([1], rm_verts=True)
        assert graph.nsrcs() == 2
        assert graph.d_neighbors(0) == [0, 1]
        assert graph.d_neighbors(1) == []


class TestMatching:
    def test_setitem_maintains_inverse(self):
        matching = Matching(3).complete(3)
        matching[0] = 2
        matching[1] = 2
        assert matching[0] is UNASSIGNED
        assert matching[1] == 2
        assert matching.inv_match[2] == 1

    def test_invview_roundtrip(self):
        matching = Matching(2).complete(2)
        matching[1] = 0
        inv = matching.invview()
        assert inv[0] == 1
        inv[0] = 0
        assert matching[0] == 0
        assert matching[1] is UNASSIGNED


class TestMaximalMatching:
    def test_perfect_matching_found(self):
        # Bipartite graph with a known perfect matching.
        graph = build_graph(
            3, 3, [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
        )
        matching = maximal_matching(graph)
        matched = [m for m in matching if isinstance(m, int)]
        assert len(matched) == 3

    def test_augmenting_path_reassigns(self):
        # eq0 can take v0 or v1; eq1 only v0. Matching eq0 first to
        # v0 must be undone through an augmenting path.
        graph = build_graph(2, 2, [(0, 0), (0, 1), (1, 0)])
        matching = maximal_matching(graph)
        assert matching[0] == 1
        assert matching[1] == 0

    def test_filters_respected(self):
        graph = build_graph(2, 2, [(0, 0), (1, 1)])
        matching = maximal_matching(
            graph, dstfilter=lambda v: v != 1
        )
        assert matching[0] == 0
        assert matching[1] is UNASSIGNED

    def test_deficient_side_leaves_unassigned(self):
        graph = build_graph(3, 1, [(0, 0), (1, 0), (2, 0)])
        matching = maximal_matching(graph)
        matched = [m for m in matching if isinstance(m, int)]
        assert len(matched) == 1


class TestTarjanAndSCCs:
    def test_tarjan_reverse_topological(self):
        # 0 -> 1 -> 2, 2 -> 1 (cycle {1, 2}), 3 isolated.
        adj = {0: [1], 1: [2], 2: [1], 3: []}
        sccs = tarjan_scc(4, lambda v: adj[v])
        as_sets = [set(c) for c in sccs]
        assert {1, 2} in as_sets
        # Reverse topological: {1,2} appears before {0}.
        assert as_sets.index({1, 2}) < as_sets.index({0})

    def test_find_var_sccs_blt_order(self):
        # eq0: v0 = f(v1); eq1: v1 = g(v1) -- v1's block first.
        graph = build_graph(2, 2, [(0, 0), (0, 1), (1, 1)])
        matching = Matching([0, 1]).complete(2)
        sccs = find_var_sccs(graph, matching)
        assert sccs == [[1], [0]]

    def test_coupled_block_grouped(self):
        # eq0: v0 <-> v1 coupling, eq1: v1 <-> v0: one 2-SCC.
        graph = build_graph(
            2, 2, [(0, 0), (0, 1), (1, 0), (1, 1)]
        )
        matching = Matching([0, 1]).complete(2)
        sccs = find_var_sccs(graph, matching)
        assert sccs == [[0, 1]]


class TestDigraphViews:
    def test_transposed_neighbors(self):
        # eq0 solves v0 and touches v1: edge v1 -> v0.
        graph = build_graph(1, 2, [(0, 0), (0, 1)])
        matching = Matching([0, UNASSIGNED]).complete(1)
        dig = DiCMOBiGraphT(graph, matching)
        assert list(dig.inneighbors(0)) == [1]
        assert list(dig.outneighbors(1)) == [0]

    def test_untransposed_equation_order(self):
        # eq1 solves v0; eq0 uses v0: edge eq0 -> eq1.
        graph = build_graph(2, 2, [(0, 0), (0, 1), (1, 0)])
        matching = Matching([1, 0]).complete(2)
        dig = DiCMOBiGraphF(graph, matching)
        assert list(dig.outneighbors(0)) == [1]
        order = toposort_equations(dig, [0, 1])
        # Evaluation order is the reverse of the toposort.
        assert list(reversed(order)) == [1, 0]

    def test_neighborhood_in(self):
        graph = build_graph(2, 3, [(0, 0), (0, 1), (1, 1), (1, 2)])
        matching = Matching([0, 1, UNASSIGNED]).complete(2)
        dig = DiCMOBiGraphT(graph, matching)
        assert neighborhood_in(dig, 0) == [0, 1, 2]


class TestIncrementalCycleTracker:
    def test_rejects_cycle(self):
        graph = build_graph(2, 2, [(0, 0), (0, 1), (1, 0), (1, 1)])
        matching = Matching(2).complete(2)
        dig = DiCMOBiGraphT(graph, matching)
        ict = IncrementalCycleTracker(dig)

        def assign(eq, vj):
            def apply(g):
                g.matching[vj] = eq

            srcs = [
                v for v in graph.s_neighbors(eq) if v != vj
            ]
            return ict.add_edge_checked(apply, srcs, vj)

        # v0 solved by eq0 (needs v1); v1 solved by eq1 (needs v0)
        # closes a cycle and must be rejected.
        assert assign(0, 0)
        assert not assign(1, 1)
        assert dig.matching[1] is UNASSIGNED

    def test_self_loop_reject_reverts_levels(self):
        # A batch whose srcs include dst must be rejected without
        # leaking level bumps applied for earlier srcs in the batch.
        graph = build_graph(2, 2, [(0, 0), (0, 1), (1, 1)])
        matching = Matching(2).complete(2)
        dig = DiCMOBiGraphT(graph, matching)
        ict = IncrementalCycleTracker(dig)

        before = list(ict.levels.values)
        accepted = ict.add_edge_checked(
            lambda g: None, [1, 0], 0
        )
        assert not accepted
        assert ict.levels.values == before
        assert ict.levels.log == []

    def test_accepts_chain(self):
        graph = build_graph(2, 2, [(0, 0), (1, 0), (1, 1)])
        matching = Matching(2).complete(2)
        dig = DiCMOBiGraphT(graph, matching)
        ict = IncrementalCycleTracker(dig)

        def assign(eq, vj):
            def apply(g):
                g.matching[vj] = eq

            srcs = [
                v for v in graph.s_neighbors(eq) if v != vj
            ]
            return ict.add_edge_checked(apply, srcs, vj)

        assert assign(0, 0)
        assert assign(1, 1)
        assert dig.matching[0] == 0
        assert dig.matching[1] == 1
