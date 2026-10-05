"""Exact matching of integer-linear algebraic loops.

Cubie's own design. A strongly connected component of homogeneous
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
from typing import Callable, List, Optional, Tuple

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.structural.bipartite import _always_true
from cubie.odesystems.symbolic.structural.clil import (
    SparseMatrixCLIL,
    bareiss,
    bareiss_update_virtual_colswap_clil,
)
from cubie.odesystems.symbolic.structural.digraph import find_var_sccs
from cubie.odesystems.symbolic.structural.singularity_removal import (
    find_masked_pivot,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
    StructuralState,
)
from cubie.odesystems.symbolic.structural.tearing import (
    build_var_eq_matching,
)


def _display_name(state: StructuralState, var: ir.Sym) -> str:
    """Name of ``var`` with one prime per derivative order."""

    base, order = state.registry.base_and_order(var)
    return base.name + "'" * order


def _reduce_block(
    block: SparseMatrixCLIL,
    scc_cols: List[bool],
    derivative_cols: List[bool],
) -> int:
    """Bareiss-reduce ``block`` on its SCC columns; return the rank.

    Each step pivots on a derivative column when a remaining row holds
    one, and on any SCC column otherwise. Row ``k`` of the result holds
    its pivot column and no earlier pivot column.
    """

    def find_pivot(
        matrix: SparseMatrixCLIL, k: int
    ) -> Optional[Tuple[Tuple[int, int], int]]:
        pivot = find_masked_pivot(derivative_cols, matrix, k)
        if pivot is None:
            pivot = find_masked_pivot(scc_cols, matrix, k)
        return pivot

    def swaprows(matrix: SparseMatrixCLIL, i: int, j: int) -> None:
        matrix.swaprows(i, j)

    def update(matrix, k, swapto, pivot, last_pivot) -> None:
        bareiss_update_virtual_colswap_clil(
            matrix, k, swapto[1], pivot, last_pivot
        )

    rank, _, _ = bareiss(
        block, find_pivot, swaprows=swaprows, update=update
    )
    return rank


def match_linear_sccs(
    state: StructuralState,
    isder: Callable[[int], bool],
    varfilter: Callable[[int], bool],
) -> None:
    """Reduce integer-linear SCCs to explicit solve sequences.

    The SCCs are those of a maximal matching of the variables passing
    ``varfilter``. An SCC qualifies when it has at least two
    variables, every variable is matched, and every matched equation
    is a row of ``state.mm`` whose columns equal the equation's
    incidence. Its rows are reduced exactly over the SCC's variables.
    When the reduction reaches full rank, each equation is replaced
    by a reduced row that holds one SCC variable not held by any
    later row, and the equations, ``state.mm`` rows, incidence graph
    and solvable graph are rewritten to the reduced rows. A singular
    SCC warns and is left unchanged.

    Parameters
    ----------
    state
        The structural state, mutated in place.
    isder
        Predicate marking differentiated variables, which are chosen
        as pivots before other variables.
    varfilter
        Predicate selecting the variables that may be solved for.
    """

    structure = state.structure
    graph = structure.graph
    solvable_graph = structure.solvable_graph
    mm = state.mm
    mm_rows = {eq: i for i, eq in enumerate(mm.nzrows)}
    var_eq_matching, _ = build_var_eq_matching(
        structure, varfilter, _always_true
    )

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
        if _reduce_block(block, scc_cols, derivative_cols) < len(eqs):
            names = ", ".join(
                _display_name(state, state.fullvars[v]) for v in scc
            )
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
            rhs = ir.add(
                *[c * state.fullvars[v] for c, v in zip(vals, cols)]
            )
            state.eqs[eq] = Equation(ir.ZERO, rhs)
            state.original_eqs[eq] = state.eqs[eq]
            graph.set_neighbors(eq, cols)
            solvable_graph.set_neighbors(eq, cols)
