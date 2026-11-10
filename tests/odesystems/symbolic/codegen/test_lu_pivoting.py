"""Tests for the static LU pivot orderings."""

import itertools

import pytest

from cubie.odesystems.symbolic.codegen.lu_pivoting import (
    _approximate_minimum_degree,
    _matched_rows,
    _ranked_ordering,
    amd_ordering,
    best_ordering,
    markowitz_ordering,
    min_fill_ordering,
    natural_ordering,
)

CANDIDATES = [
    markowitz_ordering,
    min_fill_ordering,
    natural_ordering,
    amd_ordering,
]
ORDERINGS = CANDIDATES + [best_ordering]

# Model a torn system: rows 0-3 have mass on their diagonal, and rows
# 4 and 5 are constraints without a diagonal entry.
TORN_PATTERN = {
    (0, 0), (0, 4),
    (1, 1), (1, 0), (1, 5),
    (2, 2), (2, 1), (2, 3),
    (3, 3), (3, 2),
    (4, 0), (4, 2),
    (5, 1), (5, 3),
}
TORN_STRONG = frozenset((i, i) for i in range(4))


def _oscillator_ring(oscillators, reach):
    """Return the shifted-Jacobian pattern of coupled oscillators.

    Interleave each oscillator's position and velocity, and couple
    every velocity to the positions within ``reach`` ring steps.
    """
    pattern = set()
    for k in range(oscillators):
        x, v = 2 * k, 2 * k + 1
        pattern |= {(x, x), (x, v), (v, v), (v, x)}
        for step in range(1, reach + 1):
            for other in (k + step, k - step):
                pattern.add((v, 2 * (other % oscillators)))
    return pattern


ARROW = (
    {(i, i) for i in range(5)}
    | {(0, j) for j in range(5)}
    | {(i, 0) for i in range(5)}
)
# On the first ring min-fill beats Markowitz, and on the second natural
# order beats both, so each rule picks different pivots on some case.
RULE_CASES = [
    (TORN_PATTERN, 6, TORN_STRONG),
    (ARROW, 5, frozenset((i, i) for i in range(5))),
    (_oscillator_ring(5, 2), 10, frozenset((i, i) for i in range(10))),
    (_oscillator_ring(6, 1), 12, frozenset((i, i) for i in range(12))),
]


def _tier(entry, strong, block_size):
    i, j = entry
    tier = 0 if entry in strong else 2
    if block_size is not None and i // block_size != j // block_size:
        tier += 1
    return tier


def _eliminate(pattern, n, row_perm, col_perm):
    """Eliminate along the given pivots on a dense boolean matrix.

    Return the filled pattern in original coordinates, the flops, and
    the active pattern before each pivot.
    """
    active = set(pattern)
    filled = set(pattern)
    rows = set(range(n))
    cols = set(range(n))
    flops = 0
    before = []
    for pivot in zip(row_perm, col_perm):
        before.append(active)
        rows.discard(pivot[0])
        cols.discard(pivot[1])
        targets = [j for j in cols if (pivot[0], j) in active]
        sources = [i for i in rows if (i, pivot[1]) in active]
        flops += len(sources) * (1 + 2 * len(targets))
        fill = {(i, j) for i in sources for j in targets}
        active = {
            (i, j) for (i, j) in active | fill if i in rows and j in cols
        }
        filled |= fill
    return filled, flops, before


def _best_tier(active, strong, block_size):
    """Return the active entries of the lowest pivot tier."""
    best = min(_tier(entry, strong, block_size) for entry in active)
    return [e for e in active if _tier(e, strong, block_size) == best]


def _markowitz_count(active, entry):
    i, j = entry
    row = sum(1 for a, _ in active if a == i) - 1
    col = sum(1 for _, b in active if b == j) - 1
    return row * col


def _fill_count(active, entry):
    i, j = entry
    sources = [a for a, b in active if b == j and a != i]
    targets = [b for a, b in active if a == i and b != j]
    return sum(
        1 for a in sources for b in targets if (a, b) not in active
    )


# Each rule ranks the best-tier candidates of one elimination step.
RULES = [
    (
        markowitz_ordering,
        lambda active, e: (_markowitz_count(active, e), e),
    ),
    (
        min_fill_ordering,
        lambda active, e: (
            _fill_count(active, e), _markowitz_count(active, e), e
        ),
    ),
    (natural_ordering, lambda active, e: (e[1], e[0])),
]


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize(
    "pattern, n, strong, block_size",
    [
        (TORN_PATTERN, 6, TORN_STRONG, None),
        (TORN_PATTERN, 6, TORN_STRONG, 3),
        (ARROW, 5, frozenset((i, i) for i in range(5)), None),
    ],
)
def test_ordering_returns_the_symbolic_factor_of_its_pivots(
    ordering, pattern, n, strong, block_size
):
    """Check the pattern and flops against a replay of the pivots."""
    row_perm, col_perm, lu_pattern, flops = ordering(
        set(pattern), n, strong, block_size
    )
    assert sorted(row_perm) == list(range(n))
    assert sorted(col_perm) == list(range(n))
    filled, replay_flops, before = _eliminate(
        pattern, n, row_perm, col_perm
    )
    for active, pivot in zip(before, zip(row_perm, col_perm)):
        assert pivot in _best_tier(active, strong, block_size)
    row_position = {orig: k for k, orig in enumerate(row_perm)}
    col_position = {orig: k for k, orig in enumerate(col_perm)}
    assert lu_pattern == {
        (row_position[i], col_position[j]) for i, j in filled
    }
    assert flops == replay_flops


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_block_size_keeps_pivots_inside_diagonal_blocks(ordering):
    """Check that a block size keeps every pivot inside its block."""
    pattern = {
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1),
        (2, 0), (2, 3),
        (3, 3),
    }
    row_perm, col_perm, _, _ = markowitz_ordering(set(pattern), 4)
    assert (row_perm[0], col_perm[0]) == (0, 2)
    row_perm, col_perm, _, _ = ordering(
        set(pattern), 4, block_size=2
    )
    assert all(
        i // 2 == j // 2 for i, j in zip(row_perm, col_perm)
    )


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize(
    "pattern, strong",
    [
        ({(0, 0), (1, 0)}, frozenset()),
        # Rows 1 and 2 have an entry in column 0 only. Eliminating
        # the strong pivot (0, 0) fills both rows in columns 1 and 2,
        # so the elimination alone finds a pivot for every row.
        (
            {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
            frozenset({(0, 0)}),
        ),
    ],
)
def test_structurally_singular_pattern_refused(ordering, pattern, strong):
    """Check that a structurally singular pattern raises."""
    n = 1 + max(max(entry) for entry in pattern)
    with pytest.raises(ValueError, match="structurally singular"):
        ordering(set(pattern), n, strong)


@pytest.mark.parametrize("block_size", [None, 3])
def test_best_ordering_keeps_the_fewest_nonzeros(block_size):
    """Check that the best ordering matches the sparsest candidate."""
    results = [
        ordering(set(TORN_PATTERN), 6, TORN_STRONG, block_size)
        for ordering in CANDIDATES
    ]
    best = best_ordering(set(TORN_PATTERN), 6, TORN_STRONG, block_size)
    fewest = min((len(r[2]), r[3]) for r in results)
    assert (len(best[2]), best[3]) == fewest
    assert best in results


@pytest.mark.parametrize("ordering, rank", RULES)
@pytest.mark.parametrize("pattern, n, strong", RULE_CASES)
def test_ordering_pivots_by_its_rule(ordering, rank, pattern, n, strong):
    """Check that every pivot ranks first in its tier by the rule."""
    row_perm, col_perm, _, _ = ordering(set(pattern), n, strong)
    _, _, before = _eliminate(pattern, n, row_perm, col_perm)
    for active, pivot in zip(before, zip(row_perm, col_perm)):
        candidates = _best_tier(active, strong, None)
        assert pivot == min(candidates, key=lambda e: rank(active, e))


def _adjacency(pattern, n):
    adjacency = [set() for _ in range(n)]
    for i, j in pattern:
        if i != j:
            adjacency[i].add(j)
            adjacency[j].add(i)
    return adjacency


AMD_FLAGS = list(itertools.product((False, True), repeat=2))
STAR = {(0, j) for j in range(1, 6)} | {(j, 0) for j in range(1, 6)}
# Variables 0 and 1 share the neighbours 2, 3 and 4.
BIPARTITE = {(a, b) for a in (0, 1) for b in (2, 3, 4)}


@pytest.mark.parametrize("flags", AMD_FLAGS)
def test_minimum_degree_pivots_on_the_hub_last(flags):
    """Check that a star's hub is the last pivot."""
    tree = _approximate_minimum_degree(_adjacency(STAR, 6), *flags)
    assert tree.sequence[-1] == 0


@pytest.mark.parametrize("flags", AMD_FLAGS)
def test_minimum_degree_merges_identical_variables(flags):
    """Check that variables with one adjacency share an element."""
    tree = _approximate_minimum_degree(_adjacency(BIPARTITE, 5), *flags)
    assert any({0, 1} <= set(front) for front in tree.fronts.values())


@pytest.mark.parametrize("flags", AMD_FLAGS)
@pytest.mark.parametrize(
    "pattern, n",
    [(STAR, 6), (BIPARTITE, 5), (_oscillator_ring(5, 2), 10)],
)
def test_assembly_tree_of_a_connected_graph_has_one_root(
    flags, pattern, n
):
    """Check the tree's root, parent order and both read orders."""
    tree = _approximate_minimum_degree(_adjacency(pattern, n), *flags)
    position = {e: k for k, e in enumerate(tree.sequence)}
    roots = [e for e in tree.sequence if e not in tree.parent]
    assert len(roots) == 1
    for child, parent in tree.parent.items():
        assert position[parent] > position[child]
    postorder = tree.postorder()
    assert sorted(tree.elimination_order()) == list(range(n))
    assert sorted(postorder) == list(range(n))
    placed = {v: k for k, v in enumerate(postorder)}
    for child, parent in tree.parent.items():
        assert placed[child] < placed[parent]


@pytest.mark.parametrize("pattern, n, strong", RULE_CASES)
def test_amd_ordering_keeps_its_sparsest_variant(pattern, n, strong):
    """Check that AMD returns the sparsest of its variant orders."""
    row_of = _matched_rows(set(pattern), n)
    position_of_row = {row: column for column, row in enumerate(row_of)}
    matched = {(position_of_row[i], j) for i, j in pattern}
    results = []
    for flags in AMD_FLAGS:
        tree = _approximate_minimum_degree(_adjacency(matched, n), *flags)
        for order in (tree.elimination_order(), tree.postorder()):
            col_rank = [0] * n
            row_rank = [0] * n
            for rank, column in enumerate(order):
                col_rank[column] = rank
                row_rank[row_of[column]] = rank
            results.append(
                _ranked_ordering(
                    set(pattern), n, strong, None, col_rank, row_rank
                )
            )
    chosen = amd_ordering(set(pattern), n, strong)
    assert (len(chosen[2]), chosen[3]) == min(
        (len(r[2]), r[3]) for r in results
    )
