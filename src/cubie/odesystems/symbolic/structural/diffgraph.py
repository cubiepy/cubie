"""Differentiation chains between variables (and between equations).

``DiffGraph`` is a partial map from each vertex to the vertex
representing its time derivative, with an optional inverse map.
Indices are 0-based; absent edges are ``None``.

Ported from StateSelection.jl (commit 74df007e,
``src/graph/diff.jl``, ``DiffGraph``).
"""

from typing import Iterator, List, Optional, Tuple


class DiffGraph:
    """Maps each vertex to its derivative vertex, if any.

    Parameters
    ----------
    n
        Number of vertices.
    with_badj
        Whether to maintain the inverse (derivative-to-primal) map
        eagerly.
    """

    def __init__(self, n: int, with_badj: bool = False) -> None:
        self.primal_to_diff = [None] * n
        self.diff_to_primal = [None] * n if with_badj else None

    @classmethod
    def _from_parts(
        cls,
        primal_to_diff: List[Optional[int]],
        diff_to_primal: Optional[List[Optional[int]]],
    ) -> "DiffGraph":
        graph = cls.__new__(cls)
        graph.primal_to_diff = primal_to_diff
        graph.diff_to_primal = diff_to_primal
        return graph

    def __len__(self) -> int:
        return len(self.primal_to_diff)

    def __iter__(self) -> Iterator[Optional[int]]:
        return iter(self.primal_to_diff)

    def __getitem__(self, var: int) -> Optional[int]:
        return self.primal_to_diff[var]

    def __setitem__(self, var: int, val: Optional[int]) -> None:
        if self.diff_to_primal is not None:
            old_pd = self.primal_to_diff[var]
            if old_pd is not None:
                self.diff_to_primal[old_pd] = None
            if val is not None:
                self.diff_to_primal[val] = var
        self.primal_to_diff[var] = val

    def add_vertex(self) -> int:
        """Append a vertex with no derivative edge; return its index."""

        self.primal_to_diff.append(None)
        if self.diff_to_primal is not None:
            self.diff_to_primal.append(None)
        return len(self.primal_to_diff) - 1

    def add_edge(self, var: int, diff: int) -> None:
        """Record that vertex ``diff`` is the derivative of ``var``."""

        self[var] = diff

    def edges(self) -> Iterator[Tuple[int, int]]:
        """Iterate over ``(primal, derivative)`` pairs."""

        for i, v in enumerate(self.primal_to_diff):
            if v is not None:
                yield (i, v)

    def complete(self) -> "DiffGraph":
        """Populate the inverse map if absent."""

        if self.diff_to_primal is not None:
            return self
        diff_to_primal = [None] * len(self.primal_to_diff)
        for var, diff in self.edges():
            diff_to_primal[diff] = var
        self.diff_to_primal = diff_to_primal
        return self

    def invview(self) -> "DiffGraph":
        """Return a view with the maps swapped (aliases storage)."""

        return DiffGraph._from_parts(
            self.diff_to_primal, self.primal_to_diff
        )
