"""Integer-linear singularity removal.

``structural_singularity_removal`` and ``aag_bareiss`` are ported from
StateSelection.jl (commit 74df007e, ``src/singularity_removal.jl``,
functions of the same names with the trailing ``!`` dropped).
``get_new_mm`` follows the integer-matrix rebuild in
ModelingToolkit.jl (commit c4177c335,
``src/systems/alias_elimination.jl``, ``alias_elimination!``).

Published Functions
-------------------
:func:`structural_singularity_removal`
    The pass entry point; returns the reduced integer subsystem.

:func:`aag_bareiss`
    Bareiss factorisation of the integer-linear subsystem.

:func:`get_new_mm`
    Rebase the integer-linear subsystem after equation and variable
    deletion.
"""

from typing import List

from cubie.odesystems.symbolic.structural.clil import (
    SparseMatrixCLIL,
    bareiss,
)
from cubie.odesystems.symbolic.structural.pantelides import (
    computed_highest_diff_variables,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)


def aag_bareiss(state: StructuralState, mm: SparseMatrixCLIL) -> None:
    """Bareiss-factorise the integer-linear subsystem in place.

    Parameters
    ----------
    state
        State the matrix rows belong to.
    mm
        The integer-linear subsystem. Pivots are taken first on
        algebraic variables that occur only in linear algebraic
        equations, then on highest-differentiated variables, then on
        any variable.
    """

    graph = state.graph
    linear_equations_set = set(mm.nzrows)

    is_linear_variables = [
        state.is_algebraic(v) for v in range(graph.ndsts())
    ]
    is_highest_diff = computed_highest_diff_variables(state)
    for i in range(graph.nsrcs()):
        # Only linear algebraic equations keep their variables linear.
        if i in linear_equations_set and all(
            state.is_algebraic(v) for v in graph.s_neighbors(i)
        ):
            continue
        for j in graph.s_neighbors(i):
            is_linear_variables[j] = False

    bareiss(mm, [is_linear_variables, is_highest_diff, None])


def get_new_mm(
    old_to_new_eq: List[int],
    old_to_new_var: List[int],
    mm: SparseMatrixCLIL,
) -> SparseMatrixCLIL:
    """Rebase ``mm`` onto renumbered equations and variables.

    Parameters
    ----------
    old_to_new_eq
        New index of each old equation, ``-1`` for a deleted one.
    old_to_new_var
        New index of each old variable, ``-1`` for a deleted one.
    mm
        The integer-linear subsystem on the old indices.

    Returns
    -------
    SparseMatrixCLIL
        The rows of kept equations that hold only kept variables, on
        the new indices.
    """

    new_row_cols = []
    new_row_vals = []
    new_nzrows = []
    for i, eq in enumerate(mm.nzrows):
        if old_to_new_eq[eq] < 0:
            continue
        cols = mm.row_cols[i]
        if any(old_to_new_var[v] < 0 for v in cols):
            continue
        new_row_cols.append([old_to_new_var[v] for v in cols])
        new_row_vals.append(mm.row_vals[i])
        new_nzrows.append(old_to_new_eq[eq])
    return SparseMatrixCLIL(
        sum(1 for ieq in old_to_new_eq if ieq >= 0),
        sum(1 for iv in old_to_new_var if iv >= 0),
        new_nzrows,
        new_row_cols,
        new_row_vals,
    )


def structural_singularity_removal(
    state: StructuralState, **kwargs
) -> SparseMatrixCLIL:
    """Run the integer-linear singularity removal pass.

    Factorises the integer-linear subsystem exactly under the
    solvability options ``kwargs`` of
    :meth:`StructuralState.find_eq_solvables`.

    Returns the reduced :class:`SparseMatrixCLIL`.
    """

    mm = state.linear_subsys_adjmat(**kwargs)
    if len(mm.nzrows) == 0:
        return mm

    aag_bareiss(state, mm)
    return mm
