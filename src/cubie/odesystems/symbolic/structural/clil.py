"""Exact integer linear algebra for structural simplification.

Ports StateSelection.jl's ``SparseMatrixCLIL`` (a row-dense,
column-sparse integer matrix synced with the incidence graph) and the
fraction-free Bareiss elimination used for integer-linear singularity
removal and dummy-derivative rank checks. ``bareiss`` combines the
elimination loop of ``bareiss!`` with the staged masked pivot search
of ``do_bareiss!`` (``find_masked_pivot``).

Python integers are arbitrary precision, so the overflow-checked
arithmetic paths of the Julia implementation are unnecessary here; the
elimination arithmetic is otherwise identical.

Published Classes
-----------------
:class:`SparseMatrixCLIL`
    Row-dense, column-sparse integer matrix.

Published Functions
-------------------
:func:`bareiss`
    Fraction-free row reduction pivoting through staged column masks.

:func:`bareiss_update_virtual_colswap_clil`
    CLIL-specialised elimination step with virtual column swaps.

:func:`find_masked_pivot`
    Pivot search restricted to the columns of a mask.

:func:`exactdiv`
    Integer division asserting a zero remainder.
"""

from typing import Callable, List, Optional, Sequence, Tuple


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


def find_first_linear_variable(
    matrix: SparseMatrixCLIL,
    row_range: range,
    mask: Optional[List[bool]],
    constraint: Callable[[int], bool],
) -> Optional[Tuple[Tuple[int, int], int]]:
    """Find the first allowed entry in a row whose length passes.

    Parameters
    ----------
    matrix
        The matrix searched.
    row_range
        Stored rows searched, in order.
    mask
        Per-variable flags of the columns a pivot may take, or
        ``None`` to allow every column.
    constraint
        Predicate on a row's number of nonzeros.

    Returns
    -------
    tuple or None
        ``((row, col), value)`` of the first row passing
        ``constraint`` that holds an allowed column, taking that
        row's first allowed column; ``None`` when there is none.
    """

    eadj = matrix.row_cols
    for i in row_range:
        vertices = eadj[i]
        if constraint(len(vertices)):
            for j, v in enumerate(vertices):
                if mask is None or mask[v]:
                    return ((i, v), matrix.row_vals[i][j])
    return None


def find_masked_pivot(
    variables: Optional[List[bool]], matrix: SparseMatrixCLIL, k: int
) -> Optional[Tuple[Tuple[int, int], int]]:
    """Pivot from rows ``k`` on: one nonzero, then two, then any.

    Parameters
    ----------
    variables
        Per-variable flags of the columns a pivot may take, or
        ``None`` to allow every column.
    matrix
        The matrix being reduced.
    k
        First stored row searched.

    Returns
    -------
    tuple or None
        ``((row, col), value)`` of the pivot, or ``None``.
    """

    rows = range(k, matrix.size()[0])
    r = find_first_linear_variable(
        matrix, rows, variables, lambda n: n == 1
    )
    if r is not None:
        return r
    r = find_first_linear_variable(
        matrix, rows, variables, lambda n: n == 2
    )
    if r is not None:
        return r
    return find_first_linear_variable(
        matrix, rows, variables, lambda n: True
    )


def bareiss(
    matrix: SparseMatrixCLIL,
    pivot_masks: Sequence[Optional[List[bool]]],
) -> List[int]:
    """Fraction-free Bareiss row reduction of ``matrix`` in place.

    Each step pivots, through :func:`find_masked_pivot`, on a column
    of the first mask in ``pivot_masks`` that still offers one
    (``None`` allows every column); a mask that offers none is not
    tried again. Elimination stops when no mask offers a pivot. Row
    ``k`` of the result holds its pivot column and no earlier pivot
    column.

    Parameters
    ----------
    matrix
        The matrix reduced in place; its rows are swapped into pivot
        order.
    pivot_masks
        Per-variable column flags, tried in order.

    Returns
    -------
    list[int]
        The pivot column of each step; its length is the rank.
    """

    pivots = []
    stage = 0
    last_pivot = 1
    for k in range(matrix.size()[0]):
        found = None
        while stage < len(pivot_masks):
            found = find_masked_pivot(pivot_masks[stage], matrix, k)
            if found is not None:
                break
            stage += 1
        if found is None:
            break
        (row, col), pivot = found
        matrix.swaprows(k, row)
        bareiss_update_virtual_colswap_clil(
            matrix, k, col, pivot, last_pivot
        )
        last_pivot = pivot
        pivots.append(col)
    return pivots
