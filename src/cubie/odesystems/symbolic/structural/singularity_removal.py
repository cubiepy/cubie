"""Integer-linear singularity removal.

``find_first_linear_variable``, ``find_masked_pivot``, ``aag_bareiss``
and ``do_bareiss`` are ported from StateSelection.jl (commit 74df007e,
``src/singularity_removal.jl``, functions of the same names with the
trailing ``!`` dropped).

Published Functions
-------------------
:func:`structural_singularity_removal`
    The pass entry point; returns the reduced integer subsystem.

:func:`aag_bareiss`
    Bareiss factorisation of the integer-linear subsystem.
"""

from typing import Callable, List, Optional, Tuple

from cubie.odesystems.symbolic.structural.clil import (
    SparseMatrixCLIL,
    bareiss,
    bareiss_update_virtual_colswap_clil,
)
from cubie.odesystems.symbolic.structural.diffgraph import DiffGraph
from cubie.odesystems.symbolic.structural.pantelides import (
    computed_highest_diff_variables,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
    SystemStructure,
)


def is_algebraic(var_to_diff: DiffGraph, v: int) -> bool:
    """Whether variable ``v`` has no derivative relations at all."""

    return (
        var_to_diff[v] is None
        and var_to_diff.diff_to_primal[v] is None
    )


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


def aag_bareiss(
    structure: SystemStructure, mm_orig: SparseMatrixCLIL
) -> Tuple[SparseMatrixCLIL, List[int], Tuple[int, int, int, List[int]]]:
    """Bareiss-factorise a copy of the integer-linear subsystem.

    Parameters
    ----------
    structure
        Structure the matrix rows belong to.
    mm_orig
        The integer-linear subsystem. Its rows are permuted in step
        with the factorised copy.

    Returns
    -------
    tuple
        ``(mm, solvable_variables, (rank1, rank2, rank3, pivots))``:
        the factorised matrix, the algebraic variables that occur
        only in linear algebraic equations, and the ranks after the
        linear-variable, highest-derivative and unrestricted pivot
        stages with the pivot columns in elimination order.
    """

    graph = structure.graph
    var_to_diff = structure.var_to_diff
    mm = mm_orig.copy()
    linear_equations_set = set(mm_orig.nzrows)

    is_linear_variables = [
        is_algebraic(var_to_diff, v) for v in range(len(var_to_diff))
    ]
    is_highest_diff = computed_highest_diff_variables(structure)
    for i in range(graph.nsrcs()):
        # Only linear algebraic equations keep their variables linear.
        if i in linear_equations_set and all(
            is_algebraic(var_to_diff, v) for v in graph.s_neighbors(i)
        ):
            continue
        for j in graph.s_neighbors(i):
            is_linear_variables[j] = False
    solvable_variables = [
        v for v, linear in enumerate(is_linear_variables) if linear
    ]

    bar = do_bareiss(mm, mm_orig, is_linear_variables, is_highest_diff)
    return mm, solvable_variables, bar


def do_bareiss(
    matrix: SparseMatrixCLIL,
    mold: Optional[SparseMatrixCLIL],
    is_linear_variables: List[bool],
    is_highest_diff: List[bool],
) -> Tuple[int, int, int, List[int]]:
    """Run Bareiss elimination with staged pivot restrictions.

    Pivots are taken on linear variables while any remain, then on
    highest-differentiated variables, then on any variable.

    Parameters
    ----------
    matrix
        The matrix reduced in place.
    mold
        A matrix whose rows are swapped with ``matrix``'s, or
        ``None``.
    is_linear_variables
        Per-variable flags of the first pivot stage.
    is_highest_diff
        Per-variable flags of the second pivot stage.

    Returns
    -------
    tuple
        ``(rank1, rank2, rank3, pivots)``.
    """

    rank1r = None
    rank2r = None

    def find_pivot(m, k):
        nonlocal rank1r, rank2r
        if rank1r is None:
            r = find_masked_pivot(is_linear_variables, m, k)
            if r is not None:
                return r
            rank1r = k
        if rank2r is None:
            r = find_masked_pivot(is_highest_diff, m, k)
            if r is not None:
                return r
            rank2r = k
        return find_masked_pivot(None, m, k)

    pivots = []

    def find_and_record_pivot(m, k):
        r = find_pivot(m, k)
        if r is None:
            return None
        pivots.append(r[0][1])
        return r

    def myswaprows(m, i, j):
        if mold is not None:
            mold.swaprows(i, j)
        m.swaprows(i, j)

    def update(m, k, swapto, pivot, last_pivot):
        bareiss_update_virtual_colswap_clil(
            m, k, swapto[1], pivot, last_pivot
        )

    rank3, _, _ = bareiss(
        matrix,
        find_and_record_pivot,
        swaprows=myswaprows,
        update=update,
    )
    rank2 = rank3 if rank2r is None else rank2r
    rank1 = rank2 if rank1r is None else rank1r
    return (rank1, rank2, rank3, pivots)


def force_var_to_zero(
    structure: SystemStructure, ils: SparseMatrixCLIL, v: int
) -> SparseMatrixCLIL:
    """Append the equation ``v == 0`` for an underconstrained variable."""

    from cubie.odesystems.symbolic.structural.bipartite import SRC

    ils.nparentrows += 1
    ils.nzrows.append(ils.nparentrows - 1)
    ils.row_cols.append([v])
    ils.row_vals.append([1])
    structure.graph.add_vertex(SRC)
    if structure.solvable_graph is not None:
        structure.solvable_graph.add_vertex(SRC)
    structure.graph.add_edge(ils.nparentrows - 1, v)
    if structure.solvable_graph is not None:
        structure.solvable_graph.add_edge(ils.nparentrows - 1, v)
    structure.eq_to_diff.add_vertex()
    return ils


class IgnoreUnderconstrainedVariable:
    """Record underconstrained variables without altering the system."""

    def __init__(self) -> None:
        self.underconstrained = []

    def __call__(
        self,
        structure: SystemStructure,
        ils: SparseMatrixCLIL,
        v: int,
    ) -> SparseMatrixCLIL:
        self.underconstrained.append(v)
        return ils


def structural_singularity_removal(
    state: StructuralState,
    variable_underconstrained: Optional[Callable] = None,
    **kwargs,
) -> SparseMatrixCLIL:
    """Run the integer-linear singularity removal pass.

    Factorises the integer-linear subsystem exactly, applies the
    underconstrained-variable hook to purely-linear variables that
    were not pivoted, and updates the incidence and solvable graphs
    with the reduced row contents.

    Returns the reduced :class:`SparseMatrixCLIL`.
    """

    if variable_underconstrained is None:
        variable_underconstrained = force_var_to_zero
    mm = state.linear_subsys_adjmat(**kwargs)
    if len(mm.nzrows) == 0:
        return mm

    structure = state.structure
    ils, solvable_variables, (rank1, rank2, _rank3, pivots) = (
        aag_bareiss(structure, mm)
    )
    rk1vars = set(pivots[:rank1])
    for v in solvable_variables:
        if v in rk1vars:
            continue
        ils = variable_underconstrained(structure, ils, v)

    for ei, e in enumerate(ils.nzrows):
        structure.graph.set_neighbors(e, ils.row_cols[ei])
    if structure.solvable_graph is not None:
        for ei, e in enumerate(ils.nzrows):
            structure.solvable_graph.set_neighbors(e, ils.row_cols[ei])

    return ils
