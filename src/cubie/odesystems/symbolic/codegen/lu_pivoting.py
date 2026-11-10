"""Choose static pivot orders for the direct sparse LU solves.

Published Functions
-------------------
:func:`markowitz_ordering`
    Pick each pivot by the smallest Markowitz count.

:func:`min_fill_ordering`
    Pick each pivot by the least fill it creates.

:func:`natural_ordering`
    Pivot the columns in index order.

:func:`amd_ordering`
    Pivot the columns in approximate-minimum-degree order.

Each ordering takes the structural pattern of a square matrix and
returns the row and column permutations, the ``L + U`` pattern in
permuted coordinates and the predicted flop count. It picks each
pivot from the structural nonzeros of the active submatrix, fill
included. It takes ``strong`` entries first and entries inside a
``block_size``-wide diagonal block ahead of entries that couple two
blocks, and applies its own rule among the entries of the best tier
left. It raises ValueError on a structurally singular pattern.
"""

from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    Matching,
    construct_augmenting_path,
)

Pattern = Set[Tuple[int, int]]
OrderingResult = Tuple[List[int], List[int], Set[Tuple[int, int]], int]


def _matched_rows(pattern: Pattern, n: int) -> List[int]:
    """Return a structurally nonzero row for every column.

    Keep every structural diagonal entry and augment the rest.

    Raises
    ------
    ValueError
        If no matching covers every column.
    """
    graph = BipartiteGraph(n, n)
    for i, j in pattern:
        graph.add_edge(i, j)
    matching = Matching(n)
    unmatched_rows = []
    for i in range(n):
        if (i, i) in pattern:
            matching[i] = i
        else:
            unmatched_rows.append(i)
    # An augmenting path keeps every matched row matched, so each
    # row without a diagonal needs exactly one search.
    colour = [False] * n
    for i in unmatched_rows:
        for k in range(n):
            colour[k] = False
        if not construct_augmenting_path(
            matching, graph, i, lambda column: True, colour
        ):
            raise ValueError(
                "Direct LU generation failed: the matrix pattern "
                "is structurally singular (no assignment of rows "
                "to columns covers every variable). The system's "
                "constraints do not determine its variables."
            )
    return list(matching.match)


class _Elimination:
    """Track the active submatrix of a symbolic elimination.

    The elimination stores the active columns of each active row in
    ``rows`` and the active rows of each active column in ``cols``,
    fill included. :meth:`pivot` removes a pivot's row and column and
    adds the fill its update creates; :meth:`result` returns the
    ordering output once every row has been pivoted.
    """

    __slots__ = (
        "rows",
        "cols",
        "active_rows",
        "strong_cols",
        "block_size",
        "steps",
        "flops",
    )

    def __init__(
        self,
        pattern: Pattern,
        n: int,
        strong: Set[Tuple[int, int]],
        block_size: Optional[int],
    ) -> None:
        rows: List[Set[int]] = [set() for _ in range(n)]
        cols: List[Set[int]] = [set() for _ in range(n)]
        for i, j in pattern:
            rows[i].add(j)
            cols[j].add(i)
        strong_cols: List[FrozenSet[int]] = [frozenset()] * n
        by_row: Dict[int, Set[int]] = {}
        for i, j in strong:
            by_row.setdefault(i, set()).add(j)
        for i, js in by_row.items():
            strong_cols[i] = frozenset(js)
        self.rows = rows
        self.cols = cols
        self.active_rows = set(range(n))
        self.strong_cols = strong_cols
        self.block_size = block_size
        self.steps: List[Tuple[int, int, Set[int], Set[int]]] = []
        self.flops = 0

    def tier(self, i: int, j: int) -> int:
        """Return the pivot tier of entry ``(i, j)``, lowest first.

        Return 0 for a strong entry inside a diagonal block and 1 for
        a strong entry coupling two blocks, and 2 and 3 for the same
        two cases of any other entry.
        """
        weak = 0 if j in self.strong_cols[i] else 2
        block_size = self.block_size
        if block_size is not None and i // block_size != j // block_size:
            return weak + 1
        return weak

    def pivot(self, pivot_row: int, pivot_col: int) -> None:
        """Eliminate the pivot and add the fill of its update."""
        rows = self.rows
        cols = self.cols
        targets = rows[pivot_row]
        sources = cols[pivot_col]
        targets.discard(pivot_col)
        sources.discard(pivot_row)
        rows[pivot_row] = None
        cols[pivot_col] = None
        self.active_rows.discard(pivot_row)
        for j in targets:
            cols[j].discard(pivot_row)
        for i in sources:
            row = rows[i]
            row.discard(pivot_col)
            fill = targets - row
            if fill:
                row |= fill
                for j in fill:
                    cols[j].add(i)
        self.flops += len(sources) * (1 + 2 * len(targets))
        self.steps.append((pivot_row, pivot_col, sources, targets))

    def result(self) -> OrderingResult:
        """Return the permutations, ``L + U`` pattern and flops."""
        row_perm = [step[0] for step in self.steps]
        col_perm = [step[1] for step in self.steps]
        row_position = {orig: k for k, orig in enumerate(row_perm)}
        col_position = {orig: k for k, orig in enumerate(col_perm)}
        lu_pattern: Set[Tuple[int, int]] = set()
        for k, (_, _, sources, targets) in enumerate(self.steps):
            lu_pattern.add((k, k))
            lu_pattern.update((row_position[i], k) for i in sources)
            lu_pattern.update((k, col_position[j]) for j in targets)
        return row_perm, col_perm, lu_pattern, self.flops


def markowitz_ordering(
    pattern: Pattern,
    n: int,
    strong: Set[Tuple[int, int]] = frozenset(),
    block_size: Optional[int] = None,
) -> OrderingResult:
    """Order pivots by the Markowitz count.

    Pick each pivot within its tier by the smallest
    ``(row_count - 1) * (col_count - 1)`` over the active submatrix,
    breaking ties by row then column index.

    Parameters
    ----------
    pattern
        Structural nonzeros ``(row, column)`` of the matrix.
    n
        Matrix width.
    strong
        Entries preferred as pivots over every other entry.
    block_size
        Width of the diagonal blocks whose entries are preferred
        over entries coupling two blocks; ``None`` for one block.

    Returns
    -------
    tuple
        Row permutation, column permutation, ``L + U`` pattern in
        permuted coordinates, and predicted flop count.

    Raises
    ------
    ValueError
        If the pattern is structurally singular.
    """
    _matched_rows(pattern, n)
    state = _Elimination(pattern, n, strong, block_size)
    rows = state.rows
    cols = state.cols
    for _ in range(n):
        best = None
        best_key = None
        for i in state.active_rows:
            row = rows[i]
            row_count = len(row) - 1
            for j in row:
                key = (
                    state.tier(i, j),
                    row_count * (len(cols[j]) - 1),
                    i,
                    j,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (i, j)
        state.pivot(*best)
    return state.result()


def min_fill_ordering(
    pattern: Pattern,
    n: int,
    strong: Set[Tuple[int, int]] = frozenset(),
    block_size: Optional[int] = None,
) -> OrderingResult:
    """Order pivots by the exact fill each creates.

    Pick each pivot within its tier by the fewest new structural
    nonzeros its update adds, breaking ties by Markowitz count, then
    row, then column index.

    Parameters
    ----------
    pattern
        Structural nonzeros ``(row, column)`` of the matrix.
    n
        Matrix width.
    strong
        Entries preferred as pivots over every other entry.
    block_size
        Width of the diagonal blocks whose entries are preferred
        over entries coupling two blocks; ``None`` for one block.

    Returns
    -------
    tuple
        Row permutation, column permutation, ``L + U`` pattern in
        permuted coordinates, and predicted flop count.

    Raises
    ------
    ValueError
        If the pattern is structurally singular.
    """
    _matched_rows(pattern, n)
    state = _Elimination(pattern, n, strong, block_size)
    rows = state.rows
    cols = state.cols
    for _ in range(n):
        best = None
        best_key = None
        for i in state.active_rows:
            row = rows[i]
            row_count = len(row) - 1
            for j in row:
                tier = state.tier(i, j)
                if best_key is not None and tier > best_key[0]:
                    continue
                # Stop counting once this candidate can no longer
                # beat the best fill of its tier.
                bound = (
                    best_key[1]
                    if best_key is not None and tier == best_key[0]
                    else None
                )
                fill = 0
                for source in cols[j]:
                    if source != i:
                        fill += len(row.difference(rows[source]))
                        if bound is not None and fill > bound:
                            break
                key = (
                    tier,
                    fill,
                    row_count * (len(cols[j]) - 1),
                    i,
                    j,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (i, j)
        state.pivot(*best)
    return state.result()


def _ranked_ordering(
    pattern: Pattern,
    n: int,
    strong: Set[Tuple[int, int]],
    block_size: Optional[int],
    col_rank: Sequence[int],
    row_rank: Sequence[int],
) -> OrderingResult:
    """Pivot the active entry of lowest ``(tier, col, row)`` rank."""
    state = _Elimination(pattern, n, strong, block_size)
    rows = state.rows
    for _ in range(n):
        best = None
        best_key = None
        for i in state.active_rows:
            rank_i = row_rank[i]
            for j in rows[i]:
                key = (state.tier(i, j), col_rank[j], rank_i)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (i, j)
        state.pivot(*best)
    return state.result()


def natural_ordering(
    pattern: Pattern,
    n: int,
    strong: Set[Tuple[int, int]] = frozenset(),
    block_size: Optional[int] = None,
) -> OrderingResult:
    """Order pivots by column index.

    Pick each pivot within its tier from the lowest-index active
    column, taking its lowest-index active row.

    Parameters
    ----------
    pattern
        Structural nonzeros ``(row, column)`` of the matrix.
    n
        Matrix width.
    strong
        Entries preferred as pivots over every other entry.
    block_size
        Width of the diagonal blocks whose entries are preferred
        over entries coupling two blocks; ``None`` for one block.

    Returns
    -------
    tuple
        Row permutation, column permutation, ``L + U`` pattern in
        permuted coordinates, and predicted flop count.

    Raises
    ------
    ValueError
        If the pattern is structurally singular.
    """
    _matched_rows(pattern, n)
    identity = range(n)
    return _ranked_ordering(
        pattern, n, strong, block_size, identity, identity
    )


def _approximate_minimum_degree(adjacency: List[Set[int]]) -> List[int]:
    """Return a symmetric elimination order by approximate degree.

    Eliminate the variable of least approximate external degree on
    the quotient graph, breaking ties by index. Absorb every element
    that the newest element covers, and merge variables with
    identical adjacency into one supervariable, listing its members
    consecutively. Pass the off-diagonal adjacency of a symmetric
    pattern; the function consumes ``adjacency``.
    """
    n = len(adjacency)
    variables = adjacency
    elements: List[Set[int]] = [set() for _ in range(n)]
    members: Dict[int, Set[int]] = {}
    element_weight: Dict[int, int] = {}
    weight = [1] * n
    groups: List[List[int]] = [[v] for v in range(n)]
    degree = [len(adjacency[v]) for v in range(n)]
    alive = set(range(n))
    remaining = n
    order: List[int] = []
    while alive:
        p = min(alive, key=lambda v: (degree[v], v))
        alive.discard(p)
        order.extend(sorted(groups[p]))
        remaining -= weight[p]

        absorbed = elements[p]
        reach = variables[p]
        for e in absorbed:
            reach |= members.pop(e)
            del element_weight[e]
        reach.discard(p)
        variables[p] = None
        elements[p] = None
        members[p] = reach
        reach_weight = 0
        for i in reach:
            reach_weight += weight[i]
        element_weight[p] = reach_weight

        # Count each adjacent element's weight outside the new one.
        outside: Dict[int, int] = {}
        for i in reach:
            adjacent = elements[i]
            adjacent -= absorbed
            for e in adjacent:
                outside[e] = (
                    outside.get(e, element_weight[e]) - weight[i]
                )
            adjacent.add(p)
            neighbours = variables[i]
            neighbours -= reach
            neighbours.discard(p)
        for e, size in outside.items():
            if size == 0:
                for i in members.pop(e):
                    elements[i].discard(e)
                del element_weight[e]

        for i in reach:
            external = reach_weight - weight[i]
            bound = external
            for k in variables[i]:
                bound += weight[k]
            for e in elements[i]:
                if e != p:
                    bound += outside[e]
            degree[i] = min(
                remaining - weight[i],
                degree[i] + external,
                bound,
            )

        candidates: Dict[Tuple[int, ...], List[int]] = {}
        for i in sorted(reach):
            signature = (
                sum(variables[i]),
                sum(elements[i]),
                len(variables[i]),
                len(elements[i]),
            )
            bucket = candidates.setdefault(signature, [])
            for survivor in bucket:
                if (
                    variables[survivor] == variables[i]
                    and elements[survivor] == elements[i]
                ):
                    weight[survivor] += weight[i]
                    degree[survivor] -= weight[i]
                    groups[survivor].extend(groups[i])
                    alive.discard(i)
                    reach.discard(i)
                    for e in elements[i]:
                        members[e].discard(i)
                    for k in variables[i]:
                        variables[k].discard(i)
                    variables[i] = None
                    elements[i] = None
                    break
            else:
                bucket.append(i)
    return order


def amd_ordering(
    pattern: Pattern,
    n: int,
    strong: Set[Tuple[int, int]] = frozenset(),
    block_size: Optional[int] = None,
) -> OrderingResult:
    """Order pivots by approximate minimum degree.

    Match every column to a structurally nonzero row, keeping the
    structural diagonal, then order the columns by approximate
    minimum degree on the symmetrised pattern of the matched matrix.
    Pick each pivot within its tier from the earliest-ordered active
    column, taking the active row matched earliest.

    Parameters
    ----------
    pattern
        Structural nonzeros ``(row, column)`` of the matrix.
    n
        Matrix width.
    strong
        Entries preferred as pivots over every other entry.
    block_size
        Width of the diagonal blocks whose entries are preferred
        over entries coupling two blocks; ``None`` for one block.

    Returns
    -------
    tuple
        Row permutation, column permutation, ``L + U`` pattern in
        permuted coordinates, and predicted flop count.

    Raises
    ------
    ValueError
        If the pattern is structurally singular.
    """
    row_of = _matched_rows(pattern, n)
    position_of_row = [0] * n
    for column, row in enumerate(row_of):
        position_of_row[row] = column
    adjacency: List[Set[int]] = [set() for _ in range(n)]
    for i, j in pattern:
        position = position_of_row[i]
        if position != j:
            adjacency[position].add(j)
            adjacency[j].add(position)
    order = _approximate_minimum_degree(adjacency)
    col_rank = [0] * n
    row_rank = [0] * n
    for rank, column in enumerate(order):
        col_rank[column] = rank
        row_rank[row_of[column]] = rank
    return _ranked_ordering(
        pattern, n, strong, block_size, col_rank, row_rank
    )


__all__ = [
    "markowitz_ordering",
    "min_fill_ordering",
    "natural_ordering",
    "amd_ordering",
]
