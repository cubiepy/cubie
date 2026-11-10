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

:func:`best_ordering`
    Keep the ordering whose factor has the fewest nonzeros.

Each ordering takes the structural pattern of a square matrix and
returns the row and column permutations, the ``L + U`` pattern in
permuted coordinates and the predicted flop count. It picks each
pivot from the structural nonzeros of the active submatrix, fill
included. It takes ``strong`` entries first and entries inside a
``block_size``-wide diagonal block ahead of entries that couple two
blocks, and applies its own rule among the entries of the best tier
left. It raises ValueError on a structurally singular pattern.
"""

from heapq import heapify, heappop, heappush
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from cubie.odesystems.symbolic.structural.bipartite import (
    UNASSIGNED,
    BipartiteGraph,
    maximal_matching,
)

Pattern = Set[Tuple[int, int]]
OrderingResult = Tuple[List[int], List[int], Set[Tuple[int, int]], int]


def _matched_rows(pattern: Pattern, n: int) -> List[int]:
    """Return a structurally nonzero row for every column.

    Raises
    ------
    ValueError
        If the pattern is structurally singular.
    """
    graph = BipartiteGraph(n, n)
    for i, j in pattern:
        graph.add_edge(i, j)
    rows = list(maximal_matching(graph))
    if any(row is UNASSIGNED for row in rows):
        raise ValueError(
            "Direct LU generation failed: the matrix pattern "
            "is structurally singular (no assignment of rows "
            "to columns covers every variable). The system's "
            "constraints do not determine its variables."
        )
    return rows


class _Elimination:
    """Track the active submatrix of a symbolic elimination.

    The elimination stores the active columns of each active row in
    ``rows`` and the active rows of each active column in ``cols``,
    fill included. :meth:`pivot` removes a pivot's row and column and
    adds the fill its update creates, appending each row's new
    columns to ``fill_log`` when the caller sets a list there;
    :meth:`result` returns the ordering output once every row has
    been pivoted.
    """

    __slots__ = (
        "rows",
        "cols",
        "active_rows",
        "strong_cols",
        "steps",
        "flops",
        "fill_log",
    )

    def __init__(
        self,
        pattern: Pattern,
        n: int,
        strong: Set[Tuple[int, int]],
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
        self.steps: List[Tuple[int, int, Set[int], Set[int]]] = []
        self.flops = 0
        self.fill_log: Optional[List[Tuple[int, Set[int]]]] = None

    def pivot(self, pivot_row: int, pivot_col: int) -> None:
        """Eliminate the pivot and add the fill of its update."""
        rows = self.rows
        cols = self.cols
        fill_log = self.fill_log
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
                if fill_log is not None:
                    fill_log.append((i, fill))
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


def _row_block(i: int, block_size: Optional[int]) -> int:
    """Return the diagonal block of row ``i``, or -1 for one block."""
    return -1 if block_size is None else i // block_size


def _tier(
    j: int,
    strong_row: FrozenSet[int],
    row_block: int,
    block_size: Optional[int],
) -> int:
    """Return the pivot tier of column ``j`` in one row, lowest first.

    Return 0 for a strong entry inside a diagonal block and 1 for a
    strong entry coupling two blocks, and 2 and 3 for the same two
    cases of any other entry. Pass the row's strong columns and the
    block :func:`_row_block` returns for it.
    """
    tier = 0 if j in strong_row else 2
    if row_block >= 0 and j // block_size != row_block:
        tier += 1
    return tier


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
    state = _Elimination(pattern, n, strong)
    rows = state.rows
    cols = state.cols
    strong_cols = state.strong_cols
    # Pack (tier, count, row, column) into one integer so we compare
    # candidates lexicographically in a single integer comparison.
    span = n * n
    unset = 4 * span * span
    for _ in range(n):
        best = unset
        for i in state.active_rows:
            row = rows[i]
            row_count = len(row) - 1
            strong_row = strong_cols[i]
            row_block = _row_block(i, block_size)
            row_base = i * n
            for j in row:
                tier = 0 if j in strong_row else 2
                if row_block >= 0 and j // block_size != row_block:
                    tier += 1
                count = row_count * (len(cols[j]) - 1)
                key = (tier * span + count) * span + row_base + j
                if key < best:
                    best = key
        state.pivot((best // n) % n, best % n)
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
    state = _Elimination(pattern, n, strong)
    rows = state.rows
    cols = state.cols
    strong_cols = state.strong_cols
    # Pack (tier, fill) into a prefix and (count, row, column) into a
    # suffix, so we compare candidates in two integer comparisons.
    span = n * n
    for _ in range(n):
        best_prefix = 4 * span
        best_suffix = 0
        for i in state.active_rows:
            row = rows[i]
            row_count = len(row) - 1
            strong_row = strong_cols[i]
            row_block = _row_block(i, block_size)
            row_base = i * n
            for j in row:
                tier = 0 if j in strong_row else 2
                if row_block >= 0 and j // block_size != row_block:
                    tier += 1
                # Stop counting once this fill exceeds the best
                # candidate's. We skip a candidate of a worse tier,
                # whose bound is negative.
                bound = best_prefix - tier * span
                if bound < 0:
                    continue
                fill = 0
                for source in cols[j]:
                    if source != i:
                        fill += len(row.difference(rows[source]))
                        if fill > bound:
                            break
                if fill > bound:
                    continue
                prefix = tier * span + fill
                count = row_count * (len(cols[j]) - 1)
                suffix = count * span + row_base + j
                if prefix < best_prefix or suffix < best_suffix:
                    best_prefix = prefix
                    best_suffix = suffix
        state.pivot((best_suffix // n) % n, best_suffix % n)
    return state.result()


def _ranked_ordering(
    pattern: Pattern,
    n: int,
    strong: Set[Tuple[int, int]],
    block_size: Optional[int],
    col_rank: Sequence[int],
    row_rank: Sequence[int],
) -> OrderingResult:
    """Pivot the active entry of lowest ``(tier, col, row)`` rank.

    Pass ``col_rank`` and ``row_rank`` as permutations of
    ``range(n)``.
    """
    state = _Elimination(pattern, n, strong)
    rows = state.rows
    strong_cols = state.strong_cols
    col_of_rank = [0] * n
    row_of_rank = [0] * n
    for index in range(n):
        col_of_rank[col_rank[index]] = index
        row_of_rank[row_rank[index]] = index
    span = n * n

    def entry_key(i: int, j: int) -> int:
        tier = _tier(
            j, strong_cols[i], _row_block(i, block_size), block_size
        )
        return tier * span + col_rank[j] * n + row_rank[i]

    # An entry's key never changes, so we keep every entry, fill
    # included, in one heap and skip entries already eliminated.
    heap = [entry_key(i, j) for i, j in pattern]
    heapify(heap)
    fill_log: List[Tuple[int, Set[int]]] = []
    state.fill_log = fill_log
    for _ in range(n):
        while True:
            key = heappop(heap)
            i = row_of_rank[key % n]
            j = col_of_rank[(key // n) % n]
            row = rows[i]
            if row is not None and j in row:
                break
        state.pivot(i, j)
        for source, fill in fill_log:
            for target in fill:
                heappush(heap, entry_key(source, target))
        fill_log.clear()
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


class _AssemblyTree:
    """Record the elements of one minimum-degree elimination.

    The tree stores the elements in elimination order in
    ``sequence``. For each element it stores in ``fronts`` the
    variables the element pivots, its own supervariable first and
    then the variables eliminated together with it, and in
    ``front_size`` that pivot count plus the element's external
    degree. It stores the absorbing element of each absorbed element
    in ``parent``.
    """

    __slots__ = ("sequence", "fronts", "parent", "front_size")

    def __init__(self) -> None:
        self.sequence: List[int] = []
        self.fronts: Dict[int, List[int]] = {}
        self.parent: Dict[int, int] = {}
        self.front_size: Dict[int, int] = {}

    def elimination_order(self) -> List[int]:
        """Return the variables in elimination order.

        List each element's supervariable by index, then each
        variable group eliminated with it by index.
        """
        order: List[int] = []
        for element in self.sequence:
            order.extend(self.fronts[element])
        return order

    def postorder(self) -> List[int]:
        """Return the variables in a postorder of the assembly tree.

        Visit children before their parent, children by index with
        the largest front last, and list each element's variables by
        index with the element's own variable last.
        """
        children: Dict[int, List[int]] = {e: [] for e in self.sequence}
        for element in sorted(self.parent):
            children[self.parent[element]].append(element)
        front_size = self.front_size
        for kids in children.values():
            if len(kids) > 1:
                largest = max(
                    range(len(kids)),
                    key=lambda k: (front_size[kids[k]], k),
                )
                kids.append(kids.pop(largest))
        order: List[int] = []
        roots = sorted(e for e in self.sequence if e not in self.parent)
        for root in roots:
            # Walk the tree with an explicit stack of child iterators.
            stack = [(root, iter(children[root]))]
            while stack:
                element, pending = stack[-1]
                child = next(pending, None)
                if child is not None:
                    stack.append((child, iter(children[child])))
                    continue
                stack.pop()
                front = self.fronts[element]
                order.extend(sorted(v for v in front if v != element))
                order.append(element)
        return order


def _approximate_minimum_degree(
    adjacency: List[Set[int]],
    recent_ties: bool,
    highest_principal: bool,
) -> _AssemblyTree:
    """Eliminate a symmetric pattern by approximate minimum degree.

    Eliminate the variable of least approximate external degree on
    the quotient graph. Absorb every element that the newest element
    covers, eliminate with each pivot the variables adjacent only to
    its element, and merge variables with identical adjacency into
    one supervariable. Pass the off-diagonal adjacency of a
    symmetric pattern; the function consumes ``adjacency``.

    Parameters
    ----------
    adjacency
        Off-diagonal neighbours of every variable.
    recent_ties
        Break degree ties towards the variable whose degree changed
        most recently, and between variables first given a degree
        together towards the highest index; ``False`` breaks them
        towards the lowest index.
    highest_principal
        Name each new supervariable after its highest-index member;
        ``False`` names it after the lowest.

    Returns
    -------
    _AssemblyTree
        The elements of the elimination.
    """
    n = len(adjacency)
    variables = adjacency
    elements: List[Set[int]] = [set() for _ in range(n)]
    members: Dict[int, Set[int]] = {}
    element_weight: Dict[int, int] = {}
    weight = [1] * n
    groups: List[List[int]] = [[v] for v in range(n)]
    degree = [len(adjacency[v]) for v in range(n)]
    # Rank degree ties by priority, lowest first.
    priority = [-v for v in range(n)] if recent_ties else list(range(n))
    clock = -n
    alive = set(range(n))
    remaining = n
    tree = _AssemblyTree()
    while alive:
        p = min(alive, key=lambda v: (degree[v], priority[v]))
        alive.discard(p)
        tree.sequence.append(p)
        front = sorted(groups[p])
        remaining -= weight[p]

        absorbed = elements[p]
        reach = variables[p]
        for e in absorbed:
            reach |= members.pop(e)
            del element_weight[e]
            tree.parent[e] = p
        reach.discard(p)
        variables[p] = None
        elements[p] = None
        members[p] = reach
        reach_weight = 0
        for i in reach:
            reach_weight += weight[i]

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
                tree.parent[e] = p

        # We would pick a variable adjacent only to the new element
        # next anyway, so we eliminate it together with p.
        for i in sorted(reach):
            if not variables[i] and len(elements[i]) == 1:
                front.extend(sorted(groups[i]))
                remaining -= weight[i]
                reach_weight -= weight[i]
                alive.discard(i)
                reach.discard(i)
                variables[i] = None
                elements[i] = None
        element_weight[p] = reach_weight
        tree.fronts[p] = front
        tree.front_size[p] = len(front) + reach_weight

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
        for i in sorted(reach, reverse=highest_principal):
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
        if recent_ties:
            for i in sorted(reach):
                clock -= 1
                priority[i] = clock
    return tree


def _better(candidate: OrderingResult, best: OrderingResult) -> bool:
    """Return whether ``candidate`` has fewer nonzeros, then flops."""
    return (len(candidate[2]), candidate[3]) < (len(best[2]), best[3])


def amd_ordering(
    pattern: Pattern,
    n: int,
    strong: Set[Tuple[int, int]] = frozenset(),
    block_size: Optional[int] = None,
) -> OrderingResult:
    """Order pivots by approximate minimum degree.

    Match every column to a structurally nonzero row, then order the
    columns by approximate minimum degree on the symmetrised pattern
    of the matched matrix. Run the ordering with each tie-break and
    supervariable naming, read each run in elimination order and in
    assembly-tree postorder, and keep the order whose factor has the
    fewest nonzeros, then the fewest flops. Pick each pivot within
    its tier from the earliest-ordered active column, taking the
    active row matched earliest.

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
    orders: List[List[int]] = []
    for recent_ties in (False, True):
        for highest_principal in (False, True):
            tree = _approximate_minimum_degree(
                [set(neighbours) for neighbours in adjacency],
                recent_ties,
                highest_principal,
            )
            for order in (tree.elimination_order(), tree.postorder()):
                if order not in orders:
                    orders.append(order)
    best = None
    col_rank = [0] * n
    row_rank = [0] * n
    for order in orders:
        for rank, column in enumerate(order):
            col_rank[column] = rank
            row_rank[row_of[column]] = rank
        result = _ranked_ordering(
            pattern, n, strong, block_size, col_rank, row_rank
        )
        if best is None or _better(result, best):
            best = result
    return best


def best_ordering(
    pattern: Pattern,
    n: int,
    strong: Set[Tuple[int, int]] = frozenset(),
    block_size: Optional[int] = None,
) -> OrderingResult:
    """Return the ordering whose factor has the fewest nonzeros.

    Run :func:`markowitz_ordering`, :func:`min_fill_ordering`,
    :func:`natural_ordering` and :func:`amd_ordering`, and keep the
    result with the fewest ``L + U`` nonzeros, then the fewest flops,
    then the earliest in that list.

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
    best = None
    for ordering in (
        markowitz_ordering,
        min_fill_ordering,
        natural_ordering,
        amd_ordering,
    ):
        result = ordering(pattern, n, strong, block_size)
        if best is None or _better(result, best):
            best = result
    return best


__all__ = [
    "markowitz_ordering",
    "min_fill_ordering",
    "natural_ordering",
    "amd_ordering",
    "best_ordering",
]
