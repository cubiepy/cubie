"""Directed views of bipartite graphs and supporting digraph algorithms.

Matching-oriented directed views of a bipartite graph, strongly
connected components and their topological order, and the incremental
cycle tracker that tearing uses to keep solved-equation dependency
graphs acyclic.

``DiCMOBiGraphT`` and ``DiCMOBiGraphF`` are ported from
BipartiteGraphs.jl (commit 647b6a42, v0.1.14, ``src/dicmobigraph.jl``,
``DiCMOBiGraph``). Ported from Graphs.jl (commit dffc7a64, v1.15.0):
``tarjan_scc`` from ``src/connectivity.jl``
(``strongly_connected_components_tarjan``);
``IncrementalCycleTracker`` and ``_TransactionalList`` from
``src/cycles/incremental.jl`` (``DenseGraphICT_BFGT_N``,
``TransactionalVector``). ``find_var_sccs`` and
``toposort_equations`` are ported from ModelingToolkit.jl (commit
c4177c335, ``src/structural_transformation/utils.jl``,
``find_var_sccs``, and
``src/structural_transformation/symbolics_tearing.jl``,
``get_sorted_scc``), ordered with the engine's depth-first
:func:`~cubie.odesystems.symbolic.engine.assignments.dfs_order`.

BipartiteGraphs.jl: Copyright (c) 2022 Aayush Sabharwal; MIT.
Graphs.jl: Copyright (c) 2015 Seth Bromberger and other contributors;
BSD-2-Clause.

Published Classes
-----------------
:class:`DiCMOBiGraphT`
    Transposed matching-oriented view: vertices are variables.

:class:`DiCMOBiGraphF`
    Untransposed matching-oriented view: vertices are equations.

:class:`IncrementalCycleTracker`
    Incremental cycle detection with transactional level rollback.

Published Functions
-------------------
:func:`find_var_sccs`
    Topologically sorted variable SCCs induced by a matching.

:func:`toposort_equations`
    Evaluation order of a set of equations in a
    :class:`DiCMOBiGraphF`.
"""

from typing import Callable, Dict, Hashable, Iterable, Iterator, List
from typing import Optional, Sequence

from cubie.odesystems.symbolic.engine.assignments import dfs_order
from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    Matching,
)


class DiCMOBiGraphT:
    """Directed, contracted, matching-oriented view (transposed).

    Vertices are the destination (variable) vertices of ``graph``. An
    edge ``u -> v`` exists when ``matching[v]`` is an equation that is
    incident on ``u`` (with ``u != v``): solving the matched equation
    for ``v`` requires ``u``.

    Parameters
    ----------
    graph
        Complete bipartite incidence graph.
    matching
        Destination-to-source matching orienting the graph. Must be
        complete for :meth:`outneighbors`.
    """

    def __init__(
        self, graph: BipartiteGraph, matching: Optional[Matching] = None
    ) -> None:
        self.graph = graph
        if matching is None:
            matching = Matching(graph.ndsts())
        self.matching = matching

    def nv(self) -> int:
        """Number of vertices (variables) in the view."""

        return self.graph.ndsts()

    def inneighbors(self, v: int) -> Iterator[int]:
        """Variables the matched equation of ``v`` also touches."""

        eq = self.matching[v]
        if not isinstance(eq, int):
            return
        for w in self.graph.s_neighbors(eq):
            if w != v:
                yield w

    def outneighbors(self, v: int) -> Iterator[int]:
        """Variables whose matched equation is incident on ``v``."""

        inv = self.matching.inv_match
        for eq in self.graph.d_neighbors(v):
            if eq < len(inv):
                w = inv[eq]
                if isinstance(w, int) and w != v:
                    yield w


class DiCMOBiGraphF:
    """Directed, contracted, matching-oriented view (untransposed).

    Vertices are the source (equation) vertices of ``graph``. An edge
    ``e -> e2`` exists when ``e`` is incident on a variable matched to
    ``e2``: evaluating ``e`` requires the variable solved by ``e2``.
    """

    def __init__(self, graph: BipartiteGraph, matching: Matching) -> None:
        self.graph = graph
        self.matching = matching

    def outneighbors(self, e: int) -> Iterator[int]:
        """Equations solving a variable that ``e`` is incident on."""

        for v in self.graph.s_neighbors(e):
            e2 = self.matching[v]
            if isinstance(e2, int) and e2 != e:
                yield e2


def tarjan_scc(
    n: int, outneighbors: Callable[[int], Iterable[int]]
) -> List[List[int]]:
    """Strongly connected components via iterative Tarjan.

    Parameters
    ----------
    n
        Number of vertices, labelled ``0..n-1``.
    outneighbors
        Callable returning an iterable of a vertex's out-neighbors.

    Returns
    -------
    list[list[int]]
        Components in reverse topological order (every edge points
        from a later component to an earlier one or stays within a
        component), matching Graphs.jl's output convention.
    """

    index = [-1] * n
    lowlink = [0] * n
    on_stack = [False] * n
    stack = []
    sccs = []
    counter = [0]

    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, iter(outneighbors(root)))]
        index[root] = lowlink[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            v, nbr_iter = work[-1]
            advanced = False
            for w in nbr_iter:
                if index[w] == -1:
                    index[w] = lowlink[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, iter(outneighbors(w))))
                    advanced = True
                    break
                if on_stack[w]:
                    lowlink[v] = min(lowlink[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[v])
            if lowlink[v] == index[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                sccs.append(component)
    return sccs


def _dependencies_first(
    nodes: Sequence[Hashable],
    dependencies: Dict[Hashable, List[Hashable]],
) -> List[Hashable]:
    """Order ``nodes`` with each after its ``dependencies``.

    The roots (nodes nothing depends on) are visited in ``nodes``
    order by the engine's depth-first
    :func:`~cubie.odesystems.symbolic.engine.assignments.dfs_order`.
    """

    consumers = {}
    for node in nodes:
        for dependency in dependencies[node]:
            consumers.setdefault(dependency, []).append(node)
    return dfs_order(nodes, dependencies, consumers)


def find_var_sccs(
    graph: BipartiteGraph, assign: Optional[Matching] = None
) -> List[List[int]]:
    """Variable SCCs of the matching-induced digraph, in BLT order.

    Parameters
    ----------
    graph
        Complete bipartite incidence graph.
    assign
        Matching orienting the graph. When ``None``, variable ``i`` is
        assumed matched to equation ``i``.

    Returns
    -------
    list[list[int]]
        SCCs sorted topologically (each SCC only depends on variables
        in previous SCCs), each sorted ascending.
    """

    if assign is None:
        matching = Matching(list(range(graph.nsrcs())))
        matching = matching.complete(graph.nsrcs())
    else:
        matching = assign
    cmog = DiCMOBiGraphT(graph, matching)
    sccs = tarjan_scc(cmog.nv(), cmog.outneighbors)

    assignment = [0] * cmog.nv()
    for i, component in enumerate(sccs):
        for v in component:
            assignment[v] = i
    dependencies = {
        i: sorted(
            {
                assignment[u]
                for v in component
                for u in cmog.inneighbors(v)
            }
            - {i}
        )
        for i, component in enumerate(sccs)
    }
    order = _dependencies_first(range(len(sccs)), dependencies)
    return [sorted(sccs[i]) for i in order]


def toposort_equations(
    dig: DiCMOBiGraphF, eqs: List[int]
) -> List[int]:
    """Order ``eqs`` for evaluation within the induced subgraph of ``dig``.

    An edge ``e -> e2`` in ``dig`` means ``e`` needs the variable
    solved by ``e2``; the returned order places every equation after
    the equations it needs. The induced subgraph must be acyclic.
    """

    eq_set = set(eqs)
    dependencies = {
        e: sorted({e2 for e2 in dig.outneighbors(e) if e2 in eq_set})
        for e in eqs
    }
    return _dependencies_first(eqs, dependencies)


class _TransactionalList:
    """List with a single revert checkpoint (Graphs.jl port)."""

    def __init__(self, values: List[int]) -> None:
        self.values = values
        self.log = []

    def __getitem__(self, i: int) -> int:
        return self.values[i]

    def __setitem__(self, i: int, val: int) -> None:
        self.log.append((i, self.values[i]))
        self.values[i] = val

    def commit(self) -> None:
        self.log.clear()

    def revert(self) -> None:
        for i, val in reversed(self.log):
            self.values[i] = val
        self.log.clear()


class IncrementalCycleTracker:
    """Incremental cycle detection for a growing DAG (BFGT Algorithm N).

    Wraps a :class:`DiCMOBiGraphT` whose edges are induced by its
    matching. :meth:`add_edge_checked` tests whether a batch of edges
    sharing a destination can be added without creating a cycle; if
    so, an update callback mutates the underlying graph (typically by
    assigning the matching) and the tracker's topological levels are
    committed, otherwise the levels are rolled back and the graph is
    untouched.

    Parameters
    ----------
    graph
        The matching-oriented digraph being tracked.

    Notes
    -----
    Only the ``dir = :in`` orientation used by the tearing algorithms
    is implemented: batches share a common destination vertex and
    cycle validation walks in-neighbors.
    """

    def __init__(self, graph: DiCMOBiGraphT) -> None:
        self.graph = graph
        self.levels = _TransactionalList([0] * graph.nv())

    def add_edge_checked(
        self,
        apply_fn: Callable[[DiCMOBiGraphT], None],
        srcs: Iterable[int],
        dst: int,
    ) -> bool:
        """Try to add edges ``src -> dst`` for every ``src`` in ``srcs``.

        Returns ``True`` and calls ``apply_fn(graph)`` when no cycle
        would be created; returns ``False`` leaving all state intact
        otherwise.
        """

        g = self.graph
        levels = self.levels
        worklist = []
        # In the :in orientation the level inequality is checked with
        # the roles of source and destination swapped.
        for w in srcs:
            v = dst
            if levels[v] < levels[w]:
                continue
            levels[w] = levels[v] + 1
            worklist.append((v, w))
        idx = 0
        while idx < len(worklist):
            x, y = worklist[idx]
            idx += 1
            xlevel = levels[x]
            ylevel = levels[y]
            if xlevel >= ylevel:
                ylevel = xlevel + 1
                levels[y] = ylevel
            elif ylevel > xlevel + 1:
                continue
            for z in g.inneighbors(y):
                if z == dst:
                    levels.revert()
                    return False
                if ylevel >= levels[z]:
                    levels[z] = ylevel + 1
                    worklist.append((y, z))
        levels.commit()
        apply_fn(g)
        return True
