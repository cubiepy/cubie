"""Contraction of eliminated variables out of the incidence graph."""

from typing import List

from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    Matching,
)
from cubie.odesystems.symbolic.structural.digraph import (
    DiCMOBiGraphT,
    neighborhood_in,
)
from cubie.odesystems.symbolic.structural.singularity_removal import (
    RestrictedBareissContext,
)


def contract_variables(
    graph: BipartiteGraph,
    var_eq_matching: Matching,
    var_rename: List[int],
    eq_rename: List[int],
    nelim_eq: int,
    nelim_var: int,
) -> BipartiteGraph:
    """Contract eliminated variables out of the incidence graph.

    Every incidence on an eliminated variable is replaced by
    incidences on the retained variables it (transitively) depends on
    through the matching-induced digraph. ``var_rename``/``eq_rename``
    map old indices to new 0-based indices with ``-1`` marking
    eliminated entries.
    """

    dig = DiCMOBiGraphT(graph, var_eq_matching)
    var_deps = [
        [
            var_rename[v2]
            for v2 in neighborhood_in(dig, v)
            if var_rename[v2] != -1
        ]
        for v in range(graph.ndsts())
    ]

    newgraph = BipartiteGraph(
        graph.nsrcs() - nelim_eq, graph.ndsts() - nelim_var
    )
    for e in range(graph.nsrcs()):
        ne = eq_rename[e]
        if ne == -1:
            continue
        for v in graph.s_neighbors(e):
            newvar = var_rename[v]
            if newvar != -1:
                newgraph.add_edge(ne, newvar)
            else:
                for nv in var_deps[v]:
                    newgraph.add_edge(ne, nv)
    return newgraph
