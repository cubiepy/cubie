"""Differentiation chains between equations.

``DiffGraph`` is a partial map from each vertex to the vertex
representing its time derivative, with its inverse map. Indices are
0-based; absent edges are ``None``.

Ported from StateSelection.jl (commit 74df007e,
``src/graph/diff.jl``, ``DiffGraph``).
"""

from typing import Iterator, List, Optional


class DiffGraph:
    """Maps each vertex to its derivative vertex, if any.

    Parameters
    ----------
    n
        Number of vertices.
    """

    def __init__(self, n: int) -> None:
        self.primal_to_diff = [None] * n
        self.diff_to_primal = [None] * n

    @classmethod
    def _from_parts(
        cls,
        primal_to_diff: List[Optional[int]],
        diff_to_primal: List[Optional[int]],
    ) -> "DiffGraph":
        graph = cls.__new__(cls)
        graph.primal_to_diff = primal_to_diff
        graph.diff_to_primal = diff_to_primal
        return graph

    def __iter__(self) -> Iterator[Optional[int]]:
        return iter(self.primal_to_diff)

    def __getitem__(self, var: int) -> Optional[int]:
        return self.primal_to_diff[var]

    def __setitem__(self, var: int, val: Optional[int]) -> None:
        old_pd = self.primal_to_diff[var]
        if old_pd is not None:
            self.diff_to_primal[old_pd] = None
        if val is not None:
            self.diff_to_primal[val] = var
        self.primal_to_diff[var] = val

    def add_vertex(self) -> int:
        """Append a vertex with no derivative edge; return its index."""

        self.primal_to_diff.append(None)
        self.diff_to_primal.append(None)
        return len(self.primal_to_diff) - 1

    def invview(self) -> "DiffGraph":
        """Return a view with the maps swapped (aliases storage)."""

        return DiffGraph._from_parts(
            self.diff_to_primal, self.primal_to_diff
        )
