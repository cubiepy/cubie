"""Integer-linear singularity removal.

Published Functions
-------------------
:func:`structural_singularity_removal`
    The pass entry point; returns the reduced integer subsystem.
"""

from typing import Callable, Optional

from cubie.odesystems.symbolic.structural.clil import SparseMatrixCLIL
from cubie.odesystems.symbolic.structural.diffgraph import DiffGraph
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
