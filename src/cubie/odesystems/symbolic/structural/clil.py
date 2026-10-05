"""Exact integer linear algebra for structural simplification.

Ports StateSelection.jl's ``SparseMatrixCLIL`` (a row-dense,
column-sparse integer matrix synced with the incidence graph) and the
fraction-free Bareiss elimination used for integer-linear singularity
removal and dummy-derivative rank checks.

Python integers are arbitrary precision, so the overflow-checked
arithmetic paths of the Julia implementation are unnecessary here; the
elimination arithmetic is otherwise identical. The nullspace helper
returns the rank and pivot column order only — the nullspace basis
matrix computed by the Julia version is never consumed by the
pipeline, so its reduced-echelon construction is not ported.

Published Classes
-----------------
:class:`SparseMatrixCLIL`
    Row-dense, column-sparse integer matrix.

Published Functions
-------------------
:func:`bareiss`
    Generic fraction-free row reduction with pluggable pivoting.

:func:`bareiss_update_virtual_colswap_clil`
    CLIL-specialised elimination step with virtual column swaps.

:func:`nullspace_rank`
    Rank and pivot column order of a dense integer matrix.

:func:`exactdiv`
    Integer division asserting a zero remainder.
"""

from typing import Callable, List, Optional, Tuple


def exactdiv(a: int, b: int) -> int:
    """Divide ``a`` by ``b`` asserting the division is exact."""

    d, r = divmod(a, b)
    if r != 0:
        raise AssertionError(f"inexact division {a} / {b}")
    return d


class SparseMatrixCLIL:
    """Sparse integer matrix stored as compressed lists of lists.

    Represents the integer-linear equation subsystem: each stored row
    is one equation, ``nzrows[i]`` records which (parent) equation it
    is, ``row_cols[i]`` the sorted variable indices with nonzero
    coefficients and ``row_vals[i]`` the matching coefficients.

    Parameters
    ----------
    nparentrows
        Number of rows of the full (parent) system.
    ncols
        Number of columns (variables).
    nzrows
        Parent row index of each stored row.
    row_cols
        Sorted column indices per stored row.
    row_vals
        Coefficients per stored row, aligned with ``row_cols``.
    """

    def __init__(
        self,
        nparentrows: int,
        ncols: int,
        nzrows: List[int],
        row_cols: List[List[int]],
        row_vals: List[List[int]],
    ) -> None:
        self.nparentrows = nparentrows
        self.ncols = ncols
        self.nzrows = nzrows
        self.row_cols = row_cols
        self.row_vals = row_vals

    def size(self) -> Tuple[int, int]:
        """Return ``(stored_rows, ncols)``."""

        return (len(self.nzrows), self.ncols)

    def copy(self) -> "SparseMatrixCLIL":
        """Return a deep copy."""

        return SparseMatrixCLIL(
            self.nparentrows,
            self.ncols,
            list(self.nzrows),
            [list(r) for r in self.row_cols],
            [list(r) for r in self.row_vals],
        )

    def swaprows(self, i: int, j: int) -> None:
        """Swap stored rows ``i`` and ``j``."""

        if i == j:
            return
        self.nzrows[i], self.nzrows[j] = self.nzrows[j], self.nzrows[i]
        self.row_cols[i], self.row_cols[j] = (
            self.row_cols[j],
            self.row_cols[i],
        )
        self.row_vals[i], self.row_vals[j] = (
            self.row_vals[j],
            self.row_vals[i],
        )

    def dropzeros(self) -> "SparseMatrixCLIL":
        """Remove explicitly stored zero coefficients in place."""

        for r in range(len(self.row_vals)):
            cols = self.row_cols[r]
            vals = self.row_vals[r]
            keep = 0
            for k in range(len(vals)):
                if vals[k] == 0:
                    continue
                cols[keep] = cols[k]
                vals[keep] = vals[k]
                keep += 1
            del cols[keep:]
            del vals[keep:]
        return self


def bareiss_update_virtual_colswap_clil(
    matrix: SparseMatrixCLIL,
    k: int,
    pivot_col: int,
    pivot: int,
    last_pivot: int,
    pivot_equal_optimization: bool = True,
) -> None:
    """One Bareiss elimination step on a CLIL matrix.

    Eliminates column ``pivot_col`` from every stored row below ``k``
    using row ``k`` as the pivot row, keeping the matrix fraction
    free. Column swaps are virtual: the pivot column keeps its index.

    Notes
    -----
    When ``|pivot| == |last_pivot|`` rows without an entry in the
    pivot column are left untouched (they would only be scaled by
    ``±1``), which is the MTK-specific micro-optimisation. For
    ``pivot == -last_pivot`` the skipped scaling is a global sign
    flip of the row — harmless because stored rows are homogeneous
    equations (``row = 0``), so the solution set, rank, and sparsity
    are unchanged.
    """

    eadj = matrix.row_cols
    old_cadj = matrix.row_vals
    pivot_equal = (
        pivot_equal_optimization and abs(pivot) == abs(last_pivot)
    )
    nrows = len(matrix.nzrows)
    kvars = eadj[k]
    kcoeffs = old_cadj[k]
    for ei in range(k + 1, nrows):
        ivars = eadj[ei]
        icoeffs = old_cadj[ei]
        coeff = 0
        for idx, col in enumerate(ivars):
            if col == pivot_col:
                coeff = icoeffs[idx]
                break
        if coeff == 0 and pivot_equal:
            continue

        tmp_cols = []
        tmp_vals = []
        ki = 0
        ii = 0
        nk = len(kvars)
        ni = len(ivars)
        while ki < nk or ii < ni:
            if ki < nk and (ii >= ni or kvars[ki] < ivars[ii]):
                v = kvars[ki]
                if v != pivot_col:
                    ci = exactdiv(-coeff * kcoeffs[ki], last_pivot)
                    if ci != 0:
                        tmp_cols.append(v)
                        tmp_vals.append(ci)
                ki += 1
            elif ii < ni and (ki >= nk or ivars[ii] < kvars[ki]):
                v = ivars[ii]
                if v != pivot_col:
                    ci = exactdiv(pivot * icoeffs[ii], last_pivot)
                    if ci != 0:
                        tmp_cols.append(v)
                        tmp_vals.append(ci)
                ii += 1
            else:
                v = kvars[ki]
                if v != pivot_col:
                    ci = exactdiv(
                        pivot * icoeffs[ii] - coeff * kcoeffs[ki],
                        last_pivot,
                    )
                    if ci != 0:
                        tmp_cols.append(v)
                        tmp_vals.append(ci)
                ki += 1
                ii += 1
        eadj[ei] = tmp_cols
        old_cadj[ei] = tmp_vals


def bareiss(
    matrix,
    find_pivot: Callable,
    swapcols: Optional[Callable] = None,
    swaprows: Optional[Callable] = None,
    update: Optional[Callable] = None,
    column_pivots: Optional[List[int]] = None,
) -> Tuple[int, int, bool]:
    """Fraction-free Bareiss row reduction with pluggable operations.

    Parameters
    ----------
    matrix
        The matrix being reduced (mutated in place).
    find_pivot
        ``find_pivot(matrix, k)`` returning ``((row, col), value)`` or
        ``None`` when no pivot remains.
    swapcols, swaprows, update
        Operation callbacks; ``update(matrix, k, (row, col), pivot,
        last_pivot)`` performs the elimination step and any
        pivot-column zeroing.
    column_pivots
        Optional record of column swaps for pivot-order recovery.

    Returns
    -------
    tuple
        ``(rank, last_pivot, column_permuted)``.
    """

    prev = 1
    n = matrix.size()[0] if hasattr(matrix, "size") else len(matrix)
    pivot = 1
    column_permuted = False
    for k in range(n):
        r = find_pivot(matrix, k)
        if r is None:
            return (k, pivot, column_permuted)
        (row, col), pivot = r
        if column_pivots is not None and k != col:
            column_pivots[k] = col
            column_permuted = True
        if (row, col) != (k, k):
            if swapcols is not None:
                swapcols(matrix, k, col)
            if swaprows is not None:
                swaprows(matrix, k, row)
        update(matrix, k, (row, col), pivot, prev)
        prev = pivot
    return (n, pivot, column_permuted)


def _dense_find_pivot_any(matrix: List[List[int]], k: int):
    nrows = len(matrix)
    ncols = len(matrix[0]) if nrows else 0
    for j in range(k, ncols):
        for i in range(k, nrows):
            if matrix[i][j] != 0:
                return ((i, j), matrix[i][j])
    return None


def _dense_swapcols(matrix: List[List[int]], i: int, j: int) -> None:
    if i == j:
        return
    for row in matrix:
        row[i], row[j] = row[j], row[i]


def _dense_swaprows(matrix: List[List[int]], i: int, j: int) -> None:
    if i == j:
        return
    matrix[i], matrix[j] = matrix[j], matrix[i]


def _dense_update(
    matrix: List[List[int]],
    k: int,
    _swapto: Tuple[int, int],
    pivot: int,
    prev_pivot: int,
) -> None:
    # _swapto (the pivot's original position) is unused: the swap
    # callbacks have already moved the pivot to (k, k).
    nrows = len(matrix)
    ncols = len(matrix[0]) if nrows else 0
    for i in range(k + 1, ncols):
        mki = matrix[k][i]
        for j in range(k + 1, nrows):
            matrix[j][i] = exactdiv(
                matrix[j][i] * pivot - matrix[j][k] * mki, prev_pivot
            )
    for j in range(k + 1, nrows):
        matrix[j][k] = 0


def nullspace_rank(
    matrix: List[List[int]],
    col_order: Optional[List[int]] = None,
) -> int:
    """Rank of a dense integer matrix via exact Bareiss elimination.

    Parameters
    ----------
    matrix
        Dense integer matrix as a list of row lists. Not mutated.
    col_order
        Optional output list; when provided it is filled so that its
        first ``rank`` entries are the pivot columns in elimination
        order (the columns proven linearly independent) and the
        remainder are the free columns, matching the Julia
        ``bareiss.nullspace`` ``col_order`` contract.

    Returns
    -------
    int
        The rank of the matrix.
    """

    work = [list(row) for row in matrix]
    n = len(work[0]) if work else 0
    column_pivots = list(range(n))
    rank, _, _ = bareiss(
        work,
        _dense_find_pivot_any,
        swapcols=_dense_swapcols,
        swaprows=_dense_swaprows,
        update=_dense_update,
        column_pivots=column_pivots,
    )
    if col_order is not None:
        col_order[:] = list(range(n))
        for i, cp in enumerate(column_pivots):
            col_order[i], col_order[cp] = col_order[cp], col_order[i]
    return rank
