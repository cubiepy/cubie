"""Bipartite incidence graphs and matchings.

Port of BipartiteGraphs.jl (SciML), the graph substrate of the
structural simplification pipeline. Source vertices are equations and
destination vertices are variables throughout the pipeline. All indices
are 0-based.

BipartiteGraphs.jl: Copyright (c) 2022 Aayush Sabharwal; MIT.

Published Classes
-----------------
:class:`BipartiteGraph`
    Undirected bipartite graph stored as sorted adjacency lists, with
    an optional backward adjacency for destination-side lookup.

:class:`Matching`
    Destination-to-source matching with an optional inverse view.

Published Functions
-------------------
:func:`maximal_matching`
    Augmenting-path maximal matching over a bipartite graph.

:func:`construct_augmenting_path`
    Single-source augmenting path search that updates a matching.
"""

from bisect import bisect_left, insort
from typing import Callable, Iterable, Iterator, List, Optional, Union


class Unassigned:
    """Sentinel for an unmatched vertex in a :class:`Matching`."""

    _instance = None

    def __new__(cls) -> "Unassigned":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "u"


class SelectedState:
    """Sentinel marking a variable selected as a differential state."""

    _instance = None

    def __new__(cls) -> "SelectedState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SelectedState"


UNASSIGNED = Unassigned()
SELECTED_STATE = SelectedState()

MatchEntry = Union[int, Unassigned, SelectedState]

SRC = "src"
DST = "dst"


class BipartiteGraph:
    """Bipartite graph between equations (sources) and variables.

    Parameters
    ----------
    nsrcs
        Number of source vertices.
    ndsts
        Number of destination vertices.
    with_badj
        Whether to maintain the backward adjacency list eagerly.
        Without it, destination-side lookups require :meth:`complete`.

    Notes
    -----
    Adjacency lists are kept sorted, so edge insertion is
    logarithmic-search plus list insertion, mirroring the Julia
    implementation.
    """

    def __init__(
        self, nsrcs: int, ndsts: int, with_badj: bool = True
    ) -> None:
        self._ne = [0]
        self.fadjlist = [[] for _ in range(nsrcs)]
        if with_badj:
            self.badjlist = [[] for _ in range(ndsts)]
        else:
            self.badjlist = ndsts

    @classmethod
    def _from_parts(
        cls,
        ne_box: List[int],
        fadjlist: List[List[int]],
        badjlist: Union[List[List[int]], int],
    ) -> "BipartiteGraph":
        graph = cls.__new__(cls)
        graph._ne = ne_box
        graph.fadjlist = fadjlist
        graph.badjlist = badjlist
        return graph

    @property
    def ne(self) -> int:
        """Number of edges in the graph."""

        return self._ne[0]

    def nsrcs(self) -> int:
        """Number of source (equation) vertices."""

        return len(self.fadjlist)

    def ndsts(self) -> int:
        """Number of destination (variable) vertices."""

        if isinstance(self.badjlist, int):
            return self.badjlist
        return len(self.badjlist)

    def require_complete(self) -> None:
        """Raise unless the backward adjacency list is stored."""

        if isinstance(self.badjlist, int):
            raise ValueError(
                "The graph has no back edges. Use `complete`."
            )

    def complete(self) -> "BipartiteGraph":
        """Populate the backward adjacency list if absent."""

        if not isinstance(self.badjlist, int):
            return self
        badjlist = [[] for _ in range(self.badjlist)]
        for s, dsts in enumerate(self.fadjlist):
            for d in dsts:
                badjlist[d].append(s)
        self.badjlist = badjlist
        return self

    def invview(self) -> "BipartiteGraph":
        """Return a view with source and destination vertices swapped.

        The returned graph aliases this graph's adjacency lists and
        edge count, so mutations through either are shared.
        """

        self.require_complete()
        return BipartiteGraph._from_parts(
            self._ne, self.badjlist, self.fadjlist
        )

    def copy(self) -> "BipartiteGraph":
        """Return a deep copy of the graph."""

        badj = self.badjlist
        if not isinstance(badj, int):
            badj = [list(row) for row in badj]
        return BipartiteGraph._from_parts(
            [self._ne[0]],
            [list(row) for row in self.fadjlist],
            badj,
        )

    def s_neighbors(self, i: int) -> List[int]:
        """Return the destinations adjacent to source ``i``."""

        return self.fadjlist[i]

    def d_neighbors(self, j: int) -> List[int]:
        """Return the sources adjacent to destination ``j``."""

        self.require_complete()
        return self.badjlist[j]

    def has_edge(self, i: int, j: int) -> bool:
        """Return whether the edge ``i -> j`` exists."""

        lst = self.fadjlist[i]
        idx = bisect_left(lst, j)
        return idx < len(lst) and lst[idx] == j

    def add_edge(self, i: int, j: int) -> bool:
        """Add the edge from source ``i`` to destination ``j``.

        Returns ``False`` when the edge already exists.
        """

        lst = self.fadjlist[i]
        idx = bisect_left(lst, j)
        if idx < len(lst) and lst[idx] == j:
            return False
        lst.insert(idx, j)
        self._ne[0] += 1
        if not isinstance(self.badjlist, int):
            insort(self.badjlist[j], i)
        return True

    def rem_edge(self, i: int, j: int) -> bool:
        """Remove the edge from source ``i`` to destination ``j``."""

        lst = self.fadjlist[i]
        idx = bisect_left(lst, j)
        if idx >= len(lst) or lst[idx] != j:
            raise ValueError(f"graph does not have edge {i} -> {j}")
        del lst[idx]
        self._ne[0] -= 1
        if not isinstance(self.badjlist, int):
            blst = self.badjlist[j]
            bidx = bisect_left(blst, i)
            del blst[bidx]
        return True

    def add_vertex(self, vert_type: str) -> int:
        """Append a vertex of ``vert_type`` and return its index."""

        if vert_type == DST:
            if isinstance(self.badjlist, int):
                self.badjlist += 1
                return self.badjlist - 1
            self.badjlist.append([])
            return len(self.badjlist) - 1
        if vert_type == SRC:
            self.fadjlist.append([])
            return len(self.fadjlist) - 1
        raise ValueError(f"type ({vert_type}) must be SRC or DST")

    def set_neighbors(self, i: int, new_neighbors: Iterable[int]) -> None:
        """Replace the neighbor set of source ``i``."""

        new_sorted = sorted(set(new_neighbors))
        old_neighbors = self.fadjlist[i]
        self._ne[0] += len(new_sorted) - len(old_neighbors)
        if not isinstance(self.badjlist, int):
            for n in old_neighbors:
                blst = self.badjlist[n]
                idx = bisect_left(blst, i)
                if idx < len(blst) and blst[idx] == i:
                    del blst[idx]
            for n in new_sorted:
                blst = self.badjlist[n]
                idx = bisect_left(blst, i)
                if not (idx < len(blst) and blst[idx] == i):
                    blst.insert(idx, i)
        old_neighbors[:] = new_sorted

    def delete_srcs(
        self, srcs: Iterable[int], rm_verts: bool = False
    ) -> "BipartiteGraph":
        """Remove all edges incident on the given source vertices.

        When ``rm_verts`` is true the vertices themselves are removed,
        renumbering the remaining source vertices.
        """

        srcs = list(srcs)
        for s in srcs:
            self.set_neighbors(s, ())
        if rm_verts:
            old_to_new = list(range(self.nsrcs()))
            for s in srcs:
                old_to_new[s] = -1
            offset = 0
            for i in range(len(old_to_new)):
                if old_to_new[i] == -1:
                    offset += 1
                    continue
                old_to_new[i] -= offset
            if not isinstance(self.badjlist, int):
                for j in range(self.ndsts()):
                    row = self.badjlist[j]
                    row[:] = [
                        old_to_new[s] for s in row if old_to_new[s] != -1
                    ]
            for s in sorted(srcs, reverse=True):
                del self.fadjlist[s]
        return self

    def delete_dsts(
        self, dsts: Iterable[int], rm_verts: bool = False
    ) -> "BipartiteGraph":
        """Destination-side analogue of :meth:`delete_srcs`."""

        self.invview().delete_srcs(dsts, rm_verts=rm_verts)
        return self

    def edges(self) -> Iterator[tuple]:
        """Iterate over ``(src, dst)`` edges, ordered by source."""

        for s, dsts in enumerate(self.fadjlist):
            for d in dsts:
                yield (s, d)


class Matching:
    """Matching of destination vertices to source vertices.

    Parameters
    ----------
    match
        Per-destination entries: a source index, :data:`UNASSIGNED`,
        or :data:`SELECTED_STATE`.
    inv_match
        Optional inverse (source-to-destination) entries.

    Notes
    -----
    ``matching[i] = v`` maintains the invariant
    ``inv_match[match[i]] == i`` when the inverse is stored, including
    unassigning any destination previously matched to ``v``.
    """

    def __init__(
        self,
        match: Union[int, List[MatchEntry]],
        inv_match: Optional[List[MatchEntry]] = None,
    ) -> None:
        if isinstance(match, int):
            match = [UNASSIGNED] * match
        self.match = match
        self.inv_match = inv_match

    def __len__(self) -> int:
        return len(self.match)

    def __iter__(self) -> Iterator[MatchEntry]:
        return iter(self.match)

    def __getitem__(self, i: int) -> MatchEntry:
        return self.match[i]

    def __setitem__(self, i: int, v: MatchEntry) -> None:
        if self.inv_match is not None:
            oldv = self.match[i]
            if isinstance(v, int) and 0 <= v < len(self.inv_match):
                iv = self.inv_match[v]
                if isinstance(iv, int):
                    self.match[iv] = UNASSIGNED
            if isinstance(oldv, int):
                if self.inv_match[oldv] != i:
                    raise AssertionError(
                        "matching inverse invariant violated"
                    )
                self.inv_match[oldv] = UNASSIGNED
            if isinstance(v, int):
                while len(self.inv_match) < v + 1:
                    self.inv_match.append(UNASSIGNED)
                self.inv_match[v] = i
        self.match[i] = v

    def push(self, v: MatchEntry) -> None:
        """Append a destination entry matched to ``v``."""

        self.match.append(v)
        if isinstance(v, int) and self.inv_match is not None:
            while len(self.inv_match) < v + 1:
                self.inv_match.append(UNASSIGNED)
            self.inv_match[v] = len(self.match) - 1

    def copy(self) -> "Matching":
        """Return a copy sharing no mutable state."""

        inv = None if self.inv_match is None else list(self.inv_match)
        return Matching(list(self.match), inv)

    def complete(self, n: Optional[int] = None) -> "Matching":
        """Populate the inverse matching if absent.

        Parameters
        ----------
        n
            Size of the inverse vector. Defaults to one more than the
            largest matched source index.
        """

        if self.inv_match is not None:
            return self
        if n is None:
            n = 0
            for entry in self.match:
                if isinstance(entry, int):
                    n = max(n, entry + 1)
        inv_match = [UNASSIGNED] * n
        for i, entry in enumerate(self.match):
            if isinstance(entry, int):
                inv_match[entry] = i
        return Matching(list(self.match), inv_match)

    def require_complete(self) -> None:
        """Raise unless the inverse matching is stored."""

        if self.inv_match is None:
            raise ValueError(
                "Backwards matching not defined. `complete` the "
                "matching first."
            )

    def invview(self) -> "Matching":
        """Return a view with forward and inverse entries swapped.

        The view aliases this matching's storage.
        """

        self.require_complete()
        return Matching(self.inv_match, self.match)


def construct_augmenting_path(
    matching: Matching,
    graph: BipartiteGraph,
    vsrc: int,
    dstfilter: Callable[[int], bool],
    dcolor: Optional[List[bool]] = None,
    scolor: Optional[List[bool]] = None,
) -> bool:
    """Search for an augmenting path from source ``vsrc``.

    If one is found the matching is updated along the path and ``True``
    is returned. ``dcolor``/``scolor`` are visited flags for
    destination and source vertices; ``scolor`` may be ``None`` when
    the caller does not need equation coloring.

    Notes
    -----
    This is an explicit-stack rendition of the recursive Julia
    implementation, preserving its two-phase visit order: each source
    first scans for an unassigned destination, then explores colored
    alternatives depth-first.
    """

    if dcolor is None:
        dcolor = [False] * graph.ndsts()
    if scolor is not None:
        scolor[vsrc] = True

    for vdst in graph.s_neighbors(vsrc):
        if dstfilter(vdst) and matching[vdst] is UNASSIGNED:
            matching[vdst] = vsrc
            return True

    # Depth-first exploration of already-matched destinations. Each
    # stack frame is (vsrc, neighbor_iterator, vdst_pending) where
    # vdst_pending is the destination whose reassignment depends on
    # the child frame's success.
    stack = [(vsrc, iter(graph.s_neighbors(vsrc)), None)]
    while stack:
        cur_src, nbr_iter, _ = stack[-1]
        advanced = False
        for vdst in nbr_iter:
            if not (dstfilter(vdst) and not dcolor[vdst]):
                continue
            dcolor[vdst] = True
            matched_src = matching[vdst]
            if not isinstance(matched_src, int):
                continue
            if scolor is not None:
                scolor[matched_src] = True
            # Phase 1 for the child: unassigned destination?
            found = False
            for child_dst in graph.s_neighbors(matched_src):
                if (
                    dstfilter(child_dst)
                    and matching[child_dst] is UNASSIGNED
                ):
                    matching[child_dst] = matched_src
                    found = True
                    break
            if found:
                matching[vdst] = cur_src
                # Unwind: propagate success up the stack.
                stack.pop()
                while stack:
                    parent_src, _, parent_dst = stack.pop()
                    matching[parent_dst] = parent_src
                return True
            stack[-1] = (cur_src, nbr_iter, vdst)
            stack.append(
                (matched_src, iter(graph.s_neighbors(matched_src)), None)
            )
            advanced = True
            break
        if not advanced:
            stack.pop()
            if stack:
                cur = stack[-1]
                stack[-1] = (cur[0], cur[1], None)
    return False


def _always_true(_x: int) -> bool:
    return True


def maximal_matching(
    graph: BipartiteGraph,
    srcfilter: Callable[[int], bool] = _always_true,
    dstfilter: Callable[[int], bool] = _always_true,
) -> Matching:
    """Construct a maximal matching of destinations to sources.

    Vertices rejected by ``srcfilter``/``dstfilter`` do not
    participate. The matching has ``max(nsrcs, ndsts)`` entries,
    mirroring the Julia implementation.
    """

    matching = Matching(max(graph.nsrcs(), graph.ndsts()))
    dcolor = [False] * graph.ndsts()
    for vsrc in range(graph.nsrcs()):
        if not srcfilter(vsrc):
            continue
        for i in range(len(dcolor)):
            dcolor[i] = False
        construct_augmenting_path(matching, graph, vsrc, dstfilter, dcolor)
    return matching
