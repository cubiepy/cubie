"""System consistency checking.

Verifies the system is balanced (as many equations as highest-order
present variables) and structurally nonsingular, with best-effort
identification of the offending equations or variables.

``singular_check`` and ``check_consistency`` with its error reporting
are ported from ModelingToolkit.jl (commit c4177c335,
``src/structural_transformation/utils.jl``, ``singular_check``,
``check_consistency`` and ``error_reporting``) and StateSelection.jl
(commit 74df007e, ``src/utils.jl``, ``check_consistency`` and
``error_reporting``).
"""

from typing import List

from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    UNASSIGNED,
    maximal_matching,
)
from cubie.odesystems.symbolic.structural.errors import (
    ExtraEquationsSystemError,
    ExtraVariablesSystemError,
    InvalidSystemError,
)
from cubie.odesystems.symbolic.structural.pantelides import (
    computed_highest_diff_variables,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)


def singular_check(state: StructuralState) -> List:
    """Return variables unmatched in the Pantelides-extended graph.

    Extends the incidence graph with the derivative edges (equation
    (15) of the Pantelides paper) and reports used variables that a
    maximal matching leaves unassigned.
    """

    graph = state.structure.graph
    var_to_diff = state.structure.var_to_diff
    extended = BipartiteGraph(
        graph.nsrcs() + sum(1 for _ in var_to_diff.edges()),
        graph.ndsts(),
    )
    for e in range(graph.nsrcs()):
        extended.set_neighbors(e, graph.s_neighbors(e))
    idx = graph.nsrcs()
    for var, diff in var_to_diff.edges():
        extended.set_neighbors(idx, [var, diff])
        idx += 1
    extended_matching = maximal_matching(extended)

    nvars = graph.ndsts()
    unassigned_vars = []
    for vj in range(min(len(extended_matching), nvars)):
        if extended_matching[vj] is UNASSIGNED and not (
            state.is_unused_var(vj)
        ):
            unassigned_vars.append(state.fullvars[vj])
    return unassigned_vars


def check_consistency(state: StructuralState) -> None:
    """Check that ``state`` is balanced and structurally nonsingular.

    Raises
    ------
    ExtraEquationsSystemError, ExtraVariablesSystemError
        When the system is unbalanced.
    InvalidSystemError
        When the system is structurally singular.
    """

    neqs = state.n_concrete_eqs()
    structure = state.structure.complete()
    graph = structure.graph
    highest_vars = computed_highest_diff_variables(structure)
    n_highest_vars = 0
    for v, h in enumerate(highest_vars):
        if not h:
            continue
        if state.is_unused_var(v):
            continue
        n_highest_vars += 1
    is_balanced = n_highest_vars == neqs

    if neqs > 0 and not is_balanced:
        varwhitelist = [
            d is None for d in structure.var_to_diff
        ]
        var_eq_matching = maximal_matching(
            graph, dstfilter=lambda v: varwhitelist[v]
        )
        if n_highest_vars < neqs:
            eq_var_matching = var_eq_matching.complete(
                graph.nsrcs()
            ).invview()
            bad_eqs = [
                state.eqs[e]
                for e in range(graph.nsrcs())
                if eq_var_matching[e] is UNASSIGNED
            ]
            raise ExtraEquationsSystemError(
                "The system is unbalanced. There are "
                f"{n_highest_vars} highest order derivative "
                f"variables and {neqs} equations.\n"
                "More equations than variables, here are the "
                "potential extra equation(s):\n"
                + "\n".join(str(e) for e in bad_eqs)
            )
        bad_vars = [
            state.fullvars[v]
            for v in range(graph.ndsts())
            if v < len(var_eq_matching)
            and var_eq_matching[v] is UNASSIGNED
        ]
        raise ExtraVariablesSystemError(
            "The system is unbalanced. There are "
            f"{n_highest_vars} highest order derivative variables "
            f"and {neqs} equations.\n"
            "More variables than equations, here are the potential "
            "extra variable(s):\n"
            + "\n".join(str(v) for v in bad_vars)
        )

    unassigned_vars = singular_check(state)

    if unassigned_vars or not is_balanced:
        raise InvalidSystemError(
            "The system is structurally singular! Here are the "
            "problematic variables:\n"
            + "\n".join(str(v) for v in unassigned_vars)
        )
