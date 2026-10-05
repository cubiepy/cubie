"""Exact matching of integer-linear algebraic loops.

Not ported. A strongly connected component of homogeneous
integer-linear equations is reduced with fraction-free elimination
into equations that solve their variables explicitly one after
another, so tearing never has to iterate on it.

Published Functions
-------------------
:func:`match_linear_sccs`
    Reduce each nonsingular integer-linear SCC to an explicit solve
    sequence.
"""

import warnings
from typing import Callable

from cubie.odesystems.symbolic.structural.clil import (
    SparseMatrixCLIL,
    bareiss,
)
from cubie.odesystems.symbolic.structural.digraph import find_var_sccs
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)
from cubie.odesystems.symbolic.structural.tearing import (
    build_var_eq_matching,
)


def match_linear_sccs(
    state: StructuralState,
    isder: Callable[[int], bool],
    varfilter: Callable[[int], bool],
    **kwargs,
) -> None:
    """Reduce integer-linear SCCs to explicit solve sequences.

    The SCCs are those of a maximal matching of the variables passing
    ``varfilter``. An SCC qualifies when it has at least two
    variables, every variable is matched, and every matched equation
    is a row of ``state.mm`` whose columns equal the equation's
    incidence. Its rows are reduced exactly over the SCC's variables.
    When the reduction reaches full rank, each equation is replaced
    by a reduced row that holds one SCC variable not held by any
    later row; the equations, ``state.mm`` rows and incidence graph
    are rewritten to the reduced rows, and the solvable edges of each
    rewritten equation are recomputed with
    :meth:`StructuralState.find_eq_solvables` under ``kwargs``. A
    singular SCC warns and is left unchanged.

    Parameters
    ----------
    state
        The structural state, mutated in place.
    isder
        Predicate marking differentiated variables, which are chosen
        as pivots before other variables.
    varfilter
        Predicate selecting the variables that may be solved for.
    **kwargs
        Solvability options of
        :meth:`StructuralState.find_eq_solvables`.
    """

    graph = state.structure.graph
    mm = state.mm
    mm_rows = {eq: i for i, eq in enumerate(mm.nzrows)}
    var_eq_matching, _ = build_var_eq_matching(graph, varfilter)

    for scc in find_var_sccs(graph, var_eq_matching):
        eqs = [var_eq_matching[v] for v in scc]
        if len(scc) < 2 or not all(isinstance(e, int) for e in eqs):
            continue
        rows = [mm_rows.get(e) for e in eqs]
        if any(
            r is None or mm.row_cols[r] != graph.s_neighbors(e)
            for r, e in zip(rows, eqs)
        ):
            continue

        block = SparseMatrixCLIL(
            len(eqs),
            mm.ncols,
            list(eqs),
            [list(mm.row_cols[r]) for r in rows],
            [list(mm.row_vals[r]) for r in rows],
        )
        scc_cols = [False] * mm.ncols
        for v in scc:
            scc_cols[v] = True
        derivative_cols = [
            in_scc and isder(v) for v, in_scc in enumerate(scc_cols)
        ]
        if len(bareiss(block, [derivative_cols, scc_cols])) < len(eqs):
            names = ", ".join(state.fullvars[v].name for v in scc)
            warnings.warn(
                "Integer-linear equations are singular in the "
                f"variables they solve ({names}); tearing them "
                "instead."
            )
            continue

        for k, eq in enumerate(block.nzrows):
            cols = block.row_cols[k]
            vals = block.row_vals[k]
            row = mm_rows[eq]
            mm.row_cols[row] = list(cols)
            mm.row_vals[row] = list(vals)
            state.rewrite_from_row(eq, cols, vals, **kwargs)
            state.original_eqs[eq] = state.eqs[eq]
