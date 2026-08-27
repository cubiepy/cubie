"""Directed views of bipartite graphs and supporting digraph algorithms.

Ports the ``DiCMOBiGraph`` adapter from BipartiteGraphs.jl, Tarjan's
strongly-connected-components algorithm together with the condensation
topological sort used by ``find_var_sccs``, and the
Bender-Fineman-Gilbert-Tarjan (Algorithm N) incremental cycle tracker
from Graphs.jl that tearing uses to keep solved-equation dependency
graphs acyclic.

BipartiteGraphs.jl: Copyright (c) 2022 Aayush Sabharwal; MIT.

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

:func:`neighborhood_in`
    All vertices reachable through in-edges (inclusive BFS).

:func:`toposort_equations`
    Topological sort of a set of equations in a
    :class:`DiCMOBiGraphF`.
"""

from typing import Callable, Iterable, Iterator, List, Optional

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
        self.ne = 0

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
        if inv is None:
            self.matching.require_complete()
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
        self.ne = 0

    def nv(self) -> int:
        """Number of vertices (equations) in the view."""

        return self.graph.nsrcs()

    def outneighbors(self, e: int) -> Iterator[int]:
        """Equations solving a variable that ``e`` is incident on."""

        for v in self.graph.s_neighbors(e):
            e2 = self.matching[v]
            if isinstance(e2, int) and e2 != e:
                yield e2

    def inneighbors(self, e: int) -> Iterator[int]:
        """Equations incident on the variable matched to ``e``."""

        inv = self.matching.inv_match
        if inv is None:
            self.matching.require_complete()
        if e >= len(inv):
            return
        v = inv[e]
        if not isinstance(v, int):
            return
        for e2 in self.graph.d_neighbors(v):
            if e2 != e:
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


def _kahn_toposort(
    n: int, out_edges: List[set]
) -> List[int]:
    """Deterministic Kahn topological sort with ascending tie-break."""

    import heapq

    indegree = [0] * n
    for src in range(n):
        for dst in out_edges[src]:
            indegree[dst] += 1
    heap = [v for v in range(n) if indegree[v] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        v = heapq.heappop(heap)
        order.append(v)
        for w in out_edges[v]:
            indegree[w] -= 1
            if indegree[w] == 0:
                heapq.heappush(heap, w)
    if len(order) != n:
        raise ValueError("Graph is not a DAG")
    return order


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

    n_scc = len(sccs)
    assignment = [0] * cmog.nv()
    for i, component in enumerate(sccs):
        for v in component:
            assignment[v] = i
    out_edges = [set() for _ in range(n_scc)]
    for i, component in enumerate(sccs):
        for v in component:
            for w in cmog.outneighbors(v):
                j = assignment[w]
                if j != i:
                    out_edges[i].add(j)
    order = _kahn_toposort(n_scc, out_edges)
    return [sorted(sccs[i]) for i in order]


def neighborhood_in(dig: DiCMOBiGraphT, v: int) -> List[int]:
    """Vertices reachable from ``v`` through in-edges, including ``v``."""

    seen = {v}
    queue = [v]
    while queue:
        u = queue.pop()
        for w in dig.inneighbors(u):
            if w not in seen:
                seen.add(w)
                queue.append(w)
    return sorted(seen)


def toposort_equations(
    dig: DiCMOBiGraphF, eqs: List[int]
) -> List[int]:
    """Topologically sort ``eqs`` within the induced subgraph of ``dig``.

    An edge ``e -> e2`` in ``dig`` means ``e`` needs the variable
    solved by ``e2``; the returned order places ``e`` before its
    out-neighbors (callers reverse it to get evaluation order),
    mirroring ``Graphs.topological_sort`` on the induced subgraph.
    """

    eq_set = set(eqs)
    local_idx = {e: i for i, e in enumerate(eqs)}
    out_edges = [set() for _ in eqs]
    for e in eqs:
        for e2 in dig.outneighbors(e):
            if e2 in eq_set and e2 != e:
                out_edges[local_idx[e]].add(local_idx[e2])
    order = _kahn_toposort(len(eqs), out_edges)
    return [eqs[i] for i in order]


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

    def __len__(self) -> int:
        return len(self.values)

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
            if v == w:
                levels.revert()
                return False
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

    def add_vertex(self) -> None:
        """Track one more vertex at level zero."""

        self.levels.values.append(0)
