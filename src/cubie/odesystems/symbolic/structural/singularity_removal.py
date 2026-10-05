"""Integer-linear singularity removal.

``aag_bareiss`` is ported from StateSelection.jl (commit 74df007e,
``src/singularity_removal.jl``, ``aag_bareiss!``).

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


def aag_bareiss(
    structure: SystemStructure, mm: SparseMatrixCLIL
) -> Tuple[List[int], List[int]]:
    """Bareiss-factorise the integer-linear subsystem in place.

    Pivots are taken first on algebraic variables that occur only in
    linear algebraic equations, then on highest-differentiated
    variables, then on any variable.

    Parameters
    ----------
    structure
        Structure the matrix rows belong to.
    mm
        The integer-linear subsystem, reduced in place.

    Returns
    -------
    tuple
        ``(solvable_variables, pivots)``: the algebraic variables
        that occur only in linear algebraic equations, and the pivot
        columns in elimination order.
    """

    graph = structure.graph
    var_to_diff = structure.var_to_diff
    linear_equations_set = set(mm.nzrows)

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

    pivots = bareiss(mm, [is_linear_variables, is_highest_diff, None])
    return solvable_variables, pivots


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
    ils = mm
    solvable_variables, pivots = aag_bareiss(structure, ils)
    # Pivots on linear variables are all taken before any other.
    rk1vars = {v for v in pivots if v in solvable_variables}
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
