"""Dummy-derivative state selection.

Port of StateSelection.jl's ``partial_state_selection.jl``: after
Pantelides index reduction, choose which differentiated variables
become algebraic ("dummy derivatives", Mattsson-Soederlind) so the
remaining system is index 1, then tear. The per-SCC rank decisions
use the exact integer Jacobian (Bareiss nullspace) when available and
an augmenting-path structural rank otherwise.

Published Functions
-------------------
:func:`dummy_derivative_graph`
    Run Pantelides, select dummy derivatives, and tear. Returns a
    :class:`~cubie.odesystems.symbolic.structural.tearing.TearingResult`
    and a dict of extra data.
"""

import warnings
from typing import Callable, Dict, List, Optional, Tuple

from cubie.odesystems.symbolic.structural.bipartite import (
    Matching,
    SELECTED_STATE,
    UNASSIGNED,
    construct_augmenting_path,
)
from cubie.odesystems.symbolic.structural.clil import nullspace_rank
from cubie.odesystems.symbolic.structural.digraph import find_var_sccs
from cubie.odesystems.symbolic.structural.pantelides import pantelides
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
    SystemStructure,
)
from cubie.odesystems.symbolic.structural.tearing import TearingResult


def is_present(structure: SystemStructure, v: int) -> bool:
    """Whether ``v`` or any of its higher derivatives occurs."""

    var_to_diff = structure.var_to_diff
    graph = structure.graph
    while True:
        if graph.d_neighbors(v):
            return True
        v = var_to_diff[v]
        if v is None:
            return False


def is_some_diff(
    structure: SystemStructure, dummy_derivatives: set, v: int
) -> bool:
    """Whether ``v`` is a live (non-dummy, present) derivative."""

    return v not in dummy_derivatives and is_present(structure, v)


def isdiffed(
    structure: SystemStructure, dummy_derivatives: set, v: int
) -> bool:
    """Whether ``v`` is an actually differentiated variable.

    Tearing must not produce ``y_t ~ D(y)``, so equations solving for
    real derivative variables are treated specially.
    """

    var_to_diff = structure.var_to_diff
    return var_to_diff.diff_to_primal[v] is not None and is_some_diff(
        structure, dummy_derivatives, v
    )


class DummyDerivativeSummary:
    """Per-SCC variable ordering and priorities used for selection."""

    def __init__(
        self,
        var_sccs: List[List[int]],
        state_priority: List[List[float]],
    ) -> None:
        self.var_sccs = var_sccs
        self.state_priority = state_priority


def dummy_derivative_graph(
    state: StructuralState,
    jac: Optional[Callable] = None,
    state_priority: Optional[Callable[[int], float]] = None,
    tearing_alg: Optional[Callable] = None,
    **kwargs,
) -> Tuple[TearingResult, Dict]:
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
        Per-variable priority function; higher-priority variables are
        more likely to remain states.
    tearing_alg
        The tearing algorithm applied after selection, seeded with the
        dummy-derivative filters.
    """

    if state.structure.solvable_graph is None:
        state.find_solvables(**kwargs)
    state.structure.complete()
    var_eq_matching = pantelides(state, **kwargs).complete(
        state.structure.graph.nsrcs()
    )
    # `mm` must be queried after Pantelides, which extends the linear
    # subsystem matrix with differentiated rows.
    return _dummy_derivative_graph(
        state.structure,
        var_eq_matching,
        jac,
        state_priority,
        tearing_alg=tearing_alg,
        mm=state.mm,
    )


def _dummy_derivative_graph(
    structure: SystemStructure,
    var_eq_matching: Matching,
    jac: Optional[Callable],
    state_priority: Optional[Callable[[int], float]],
    tearing_alg: Optional[Callable] = None,
    mm=None,
) -> Tuple[TearingResult, Dict]:
    eq_to_diff = structure.eq_to_diff
    var_to_diff = structure.var_to_diff
    graph = structure.graph
    diff_to_eq = eq_to_diff.invview()
    diff_to_var = var_to_diff.invview()
    invgraph = graph.invview()
    cranks = structure.canonical_ranks

    def extended_sp(var: int) -> float:
        """Priority of a variable's whole derivative chain."""

        min_p = 0.0
        max_p = 0.0
        v = var
        while var_to_diff[v] is not None:
            v = var_to_diff[v]
        while True:
            p = state_priority(v)
            max_p = max(max_p, p)
            min_p = min(min_p, p)
            v = diff_to_var[v]
            if v is None:
                break
        return min_p if min_p < 0 else max_p

    var_sccs = find_var_sccs(graph, var_eq_matching)
    var_dummy_scc = []
    var_state_priority = []
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
            if var_to_diff[var] is not None:
                raise AssertionError("Invalid SCC")
            if diff_to_var[var] is not None and is_present(
                structure, var
            ):
                variables.append(var)
        if not eqs:
            continue

        rank_matching = Matching(max(nvars, neqs))
        isfirst = True
        J = None
        if jac is not None:
            J = jac(eqs, variables)
        if J is not None:
            for row in J:
                for x in row:
                    if not (
                        isinstance(x, int) and -128 <= x <= 127
                    ):
                        J = None
                        break
                if J is None:
                    break
        next_eq_idxs = []
        next_var_idxs = []
        while True:
            nrows = len(eqs)
            if nrows == 0:
                break

            if state_priority is not None and isfirst:
                sp_vals = [extended_sp(v) for v in variables]
                if cranks is None:
                    var_perm = sorted(
                        range(len(sp_vals)),
                        key=lambda i: sp_vals[i],
                    )
                else:
                    var_perm = sorted(
                        range(len(sp_vals)),
                        key=lambda i: (
                            sp_vals[i],
                            cranks[variables[i]],
                        ),
                    )
                variables = [variables[i] for i in var_perm]
                sp_vals = [sp_vals[i] for i in var_perm]
                # Keep Jacobian columns aligned with the permuted
                # variable order.
                if J is not None:
                    J = [
                        [row[i] for i in var_perm] for row in J
                    ]
                var_dummy_scc.append(list(variables))
                var_state_priority.append(sp_vals)

            if J is not None:
                if not isfirst:
                    J = [
                        [J[i][j] for j in next_var_idxs]
                        for i in next_eq_idxs
                    ]
                col_order = []
                rank = nullspace_rank(J, col_order)
                for i in range(rank):
                    dummy_derivatives.append(
                        variables[col_order[i]]
                    )
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
                if int_eq is None:
                    continue
                if diff_to_eq[int_eq] is None:
                    continue
                if J is not None:
                    next_eq_idxs.append(i)
                new_eqs.append(int_eq)
            for i, var in enumerate(variables):
                int_var = diff_to_var[var]
                if int_var is None:
                    continue
                if diff_to_var[int_var] is None:
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

    dummy_set = set(dummy_derivatives)
    tearing_result, extra = _tear_with_dummies(
        structure, dummy_set, tearing_alg, mm
    )
    extra = dict(extra)
    extra["ddsummary"] = DummyDerivativeSummary(
        var_dummy_scc, var_state_priority
    )
    return tearing_result, extra


def _tear_with_dummies(
    structure: SystemStructure,
    dummy_derivatives: set,
    tearing_alg: Optional[Callable],
    mm,
) -> Tuple[TearingResult, Dict]:
    """Tear after dummy-derivative selection."""

    var_to_diff = structure.var_to_diff
    can_eliminate = [False] * len(var_to_diff)
    for v in range(len(var_to_diff)):
        dv = var_to_diff[v]
        if dv is None or not is_some_diff(
            structure, dummy_derivatives, dv
        ):
            can_eliminate[v] = True

    def isder(v: int) -> bool:
        return isdiffed(structure, dummy_derivatives, v)

    def varfilter(v: int) -> bool:
        return can_eliminate[v]

    alg = tearing_alg(isder=isder, varfilter=varfilter)
    tearing_result, inner_extra = alg(structure)

    for v in range(structure.graph.ndsts()):
        if not is_present(structure, v):
            continue
        dv = var_to_diff[v]
        if dv is None or not is_some_diff(
            structure, dummy_derivatives, dv
        ):
            continue
        tearing_result.var_eq_matching[v] = SELECTED_STATE

    extra = dict(inner_extra)
    extra["can_eliminate"] = can_eliminate
    return tearing_result, extra
