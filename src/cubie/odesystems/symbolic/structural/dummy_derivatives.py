"""Dummy-derivative state selection.

After Pantelides index reduction, choose which differentiated
variables become algebraic ("dummy derivatives", Mattsson-Soederlind)
so the remaining system is index 1, then tear. The per-SCC rank
decisions use the exact integer Jacobian (Bareiss elimination) when
available and an augmenting-path structural rank otherwise.

Ported from ModelingToolkit.jl (commit c4177c335,
``src/structural_transformation/partial_state_selection.jl``):
``dummy_derivative_graph`` and ``_dummy_derivative_graph``
(``dummy_derivative_graph!``), ``is_present``, ``is_some_diff``,
``isdiffed`` and ``_tear_with_dummies`` (``DummyDerivativeTearing``).

Published Functions
-------------------
:func:`dummy_derivative_graph`
    Run Pantelides, select dummy derivatives, and tear. Returns a
    :class:`~cubie.odesystems.symbolic.structural.tearing.TearingResult`.
"""

import warnings
from typing import Callable, List, Optional

from cubie.odesystems.symbolic.structural.bipartite import (
    Matching,
    SELECTED_STATE,
    UNASSIGNED,
    construct_augmenting_path,
)
from cubie.odesystems.symbolic.structural.clil import (
    SparseMatrixCLIL,
    bareiss,
)
from cubie.odesystems.symbolic.structural.digraph import find_var_sccs
from cubie.odesystems.symbolic.structural.exact_matching import (
    match_linear_sccs,
)
from cubie.odesystems.symbolic.structural.pantelides import pantelides
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)
from cubie.odesystems.symbolic.structural.tearing import (
    ModiaTearing,
    TearingResult,
)


def is_present(state: StructuralState, v: int) -> bool:
    """Whether ``v`` or any of its higher derivatives occurs."""

    graph = state.graph
    while True:
        if graph.d_neighbors(v):
            return True
        v = state.derivative_of(v)
        if v is None:
            return False


def is_some_diff(
    state: StructuralState, dummy_derivatives: set, v: int
) -> bool:
    """Whether ``v`` is a live (non-dummy, present) derivative."""

    return v not in dummy_derivatives and is_present(state, v)


def isdiffed(
    state: StructuralState, dummy_derivatives: set, v: int
) -> bool:
    """Whether ``v`` is an actually differentiated variable.

    Tearing must not produce ``y_t ~ D(y)``, so equations solving for
    real derivative variables are treated specially.
    """

    return state.primal_of(v) is not None and is_some_diff(
        state, dummy_derivatives, v
    )


def _independent_columns(matrix: List[List[int]]) -> List[int]:
    """Columns of ``matrix`` independent of the columns before them.

    Bareiss elimination pivots on one column at a time, in column
    order; the number of columns returned is the rank.
    """

    ncols = len(matrix[0]) if matrix else 0
    row_cols = [[c for c in range(ncols) if row[c]] for row in matrix]
    clil = SparseMatrixCLIL(
        len(matrix),
        ncols,
        list(range(len(matrix))),
        row_cols,
        [[row[c] for c in cols] for row, cols in zip(matrix, row_cols)],
    )
    return bareiss(
        clil, [[c == j for c in range(ncols)] for j in range(ncols)]
    )


def dummy_derivative_graph(
    state: StructuralState,
    jac: Optional[Callable] = None,
    state_priority: Optional[Callable[[int], float]] = None,
    **kwargs,
) -> TearingResult:
    """Pantelides + dummy-derivative selection + tearing.

    Parameters
    ----------
    state
        The structural state (mutated).
    jac
        ``jac(eqs, vars)`` returning the integer Jacobian of the
        given equations with respect to the given variables, or
        ``None`` when it is not all-integer.
    state_priority
        Priority of each variable's derivative chain, the same for
        every member of a chain; higher-priority variables are more
        likely to remain states.
    """

    var_eq_matching = pantelides(state, **kwargs).complete(
        state.graph.nsrcs()
    )
    return _dummy_derivative_graph(
        state, var_eq_matching, jac, state_priority, **kwargs
    )


def _dummy_derivative_graph(
    state: StructuralState,
    var_eq_matching: Matching,
    jac: Optional[Callable],
    state_priority: Optional[Callable[[int], float]],
    **kwargs,
) -> TearingResult:
    graph = state.graph
    diff_to_eq = state.eq_to_diff.invview()
    invgraph = graph.invview()
    cranks = state.canonical_ranks

    var_sccs = find_var_sccs(graph, var_eq_matching)
    dummy_derivatives = []
    neqs = graph.nsrcs()
    nvars = graph.ndsts()

    for scc_vars in var_sccs:
        eqs = []
        variables = []
        for var in scc_vars:
            eq = var_eq_matching[var]
            if not isinstance(eq, int):
                continue
            if diff_to_eq[eq] is not None:
                eqs.append(eq)
            if state.primal_of(var) is not None and is_present(
                state, var
            ):
                variables.append(var)
        if not eqs:
            continue

        rank_matching = Matching(max(nvars, neqs))
        isfirst = True
        J = None
        if jac is not None:
            J = jac(eqs, variables)
        next_eq_idxs = []
        next_var_idxs = []
        while True:
            nrows = len(eqs)
            if nrows == 0:
                break

            if state_priority is not None and isfirst:
                sp_vals = [state_priority(v) for v in variables]
                var_perm = sorted(
                    range(len(sp_vals)),
                    key=lambda i: (sp_vals[i], cranks[variables[i]]),
                )
                variables = [variables[i] for i in var_perm]
                # Keep Jacobian columns aligned with the permuted
                # variable order.
                if J is not None:
                    J = [
                        [row[i] for i in var_perm] for row in J
                    ]

            if J is not None:
                if not isfirst:
                    J = [
                        [J[i][j] for j in next_var_idxs]
                        for i in next_eq_idxs
                    ]
                columns = _independent_columns(J)
                rank = len(columns)
                for column in columns:
                    dummy_derivatives.append(variables[column])
            else:
                eqs_set = set(eqs)
                rank = 0
                eqcolor = [False] * graph.nsrcs()
                for var in variables:
                    for i in range(len(eqcolor)):
                        eqcolor[i] = False
                    # Match from variables to equations, hence the
                    # inverse graph.
                    pathfound = construct_augmenting_path(
                        rank_matching,
                        invgraph,
                        var,
                        lambda e: e in eqs_set,
                        eqcolor,
                    )
                    if not pathfound:
                        continue
                    dummy_derivatives.append(var)
                    rank += 1
                    if rank == nrows:
                        break
                for i in range(len(rank_matching)):
                    rank_matching[i] = UNASSIGNED
            if rank != nrows:
                warnings.warn("The DAE system is singular!")

            next_eq_idxs = []
            next_var_idxs = []
            new_eqs = []
            new_vars = []
            for i, eq in enumerate(eqs):
                int_eq = diff_to_eq[eq]
                if diff_to_eq[int_eq] is None:
                    continue
                if J is not None:
                    next_eq_idxs.append(i)
                new_eqs.append(int_eq)
            for i, var in enumerate(variables):
                int_var = state.primal_of(var)
                if state.primal_of(int_var) is None:
                    continue
                if J is not None:
                    next_var_idxs.append(i)
                new_vars.append(int_var)
            eqs = new_eqs
            variables = new_vars
            isfirst = False

    n_diff_eqs = sum(
        1 for e in diff_to_eq if e is not None
    )
    n_dummys = len(dummy_derivatives)
    if n_diff_eqs != n_dummys:
        warnings.warn(
            f"The number of dummy derivatives ({n_dummys}) does not "
            f"match the number of differentiated equations "
            f"({n_diff_eqs})."
        )

    return _tear_with_dummies(state, set(dummy_derivatives), **kwargs)


def _tear_with_dummies(
    state: StructuralState,
    dummy_derivatives: set,
    **kwargs,
) -> TearingResult:
    """Tear after dummy-derivative selection.

    Integer-linear SCCs are first reduced to explicit solve sequences
    by :func:`match_linear_sccs` under the solvability options
    ``kwargs``; Modia tearing then tears the rest.
    """

    nvars = state.graph.ndsts()
    can_eliminate = [False] * nvars
    for v in range(nvars):
        dv = state.derivative_of(v)
        if dv is None or not is_some_diff(state, dummy_derivatives, dv):
            can_eliminate[v] = True

    def isder(v: int) -> bool:
        return isdiffed(state, dummy_derivatives, v)

    def varfilter(v: int) -> bool:
        return can_eliminate[v]

    # Not ported: exact matching runs before Modia tearing.
    match_linear_sccs(state, isder, varfilter, **kwargs)
    modia_tearing = ModiaTearing(isder=isder, varfilter=varfilter)
    tearing_result = modia_tearing(state.graph, state.solvable_graph)

    for v in range(state.graph.ndsts()):
        if not is_present(state, v):
            continue
        dv = state.derivative_of(v)
        if dv is None or not is_some_diff(state, dummy_derivatives, dv):
            continue
        tearing_result.var_eq_matching[v] = SELECTED_STATE

    return tearing_result
