"""Tests for the static LU pivot orderings."""

import pytest

from cubie.odesystems.symbolic.codegen.lu_pivoting import (
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


def _tier(entry, strong, block_size):
    i, j = entry
    tier = 0 if entry in strong else 2
    if block_size is not None and i // block_size != j // block_size:
        tier += 1
    return tier


def _replay(pattern, n, strong, block_size, row_perm, col_perm):
    """Eliminate along the given pivots on a dense boolean matrix.

    Return the filled pattern in original coordinates and the flops,
    asserting that each pivot is a structural nonzero of the best tier
    in the active submatrix.
    """
    active = set(pattern)
    filled = set(pattern)
    rows = set(range(n))
    cols = set(range(n))
    flops = 0
    for pivot in zip(row_perm, col_perm):
        assert pivot in active
        best = min(_tier(entry, strong, block_size) for entry in active)
        assert _tier(pivot, strong, block_size) == best
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
    return filled, flops


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize(
    "pattern, n, strong, block_size",
    [
        (TORN_PATTERN, 6, TORN_STRONG, None),
        (TORN_PATTERN, 6, TORN_STRONG, 3),
        (
            {(i, i) for i in range(5)}
            | {(0, j) for j in range(5)}
            | {(i, 0) for i in range(5)},
            5,
            frozenset((i, i) for i in range(5)),
            None,
        ),
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
    filled, replay_flops = _replay(
        pattern, n, strong, block_size, row_perm, col_perm
    )
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
