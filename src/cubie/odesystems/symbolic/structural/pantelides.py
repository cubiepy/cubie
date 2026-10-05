"""Pantelides algorithm for DAE index reduction.

Finds a maximal matching on highest-differentiated variables,
differentiating equations and introducing derivative variables until
an augmenting path exists for every equation.

Ported from StateSelection.jl (commit 74df007e, ``src/pantelides.jl``,
``computed_highest_diff_variables`` and ``pantelides!``), with the
keyword forwarding to the equation derivative of ModelingToolkit.jl
(commit c4177c335, ``src/structural_transformation/pantelides.jl``,
``pantelides!``).

Published Functions
-------------------
:func:`pantelides`
    Run the algorithm, mutating the state, and return the resulting
    variable-equation matching.

:func:`computed_highest_diff_variables`
    Boolean mask of the highest-differentiated variables that occur
    in the system.
"""

from typing import List

from cubie.odesystems.symbolic.structural.bipartite import (
    Matching,
    UNASSIGNED,
    construct_augmenting_path,
)
from cubie.odesystems.symbolic.structural.errors import InvalidSystemError
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
    SystemStructure,
)


def computed_highest_diff_variables(
    structure: SystemStructure,
) -> List[bool]:
    """Mask of highest-differentiated variables present in the system.

    A structurally highest-differentiated variable that occurs in no
    equation is replaced by the highest differentiated form of its
    chain that does occur. Variables with a whitelisted higher
    derivative are excluded.
    """

    graph = structure.graph
    var_to_diff = structure.var_to_diff
    nvars = len(var_to_diff)
    varwhitelist = [False] * nvars
    for var in range(nvars):
        if var_to_diff[var] is None and not varwhitelist[var]:
            while not graph.d_neighbors(var):
                var_lower = var_to_diff.diff_to_primal[var]
                if var_lower is None:
                    break
                var = var_lower
            varwhitelist[var] = True

    for var in range(nvars):
        if not varwhitelist[var]:
            continue
        var2 = var
        while True:
            var2 = var_to_diff[var2]
            if var2 is None:
                break
            if varwhitelist[var2]:
                varwhitelist[var] = False
                break
    return varwhitelist


_MAXITERS = 8000


def pantelides(state: StructuralState, **kwargs) -> Matching:
    """Perform the Pantelides index-reduction algorithm.

    Repeatedly attempts to match each undifferentiated equation to a
    highest-differentiated variable; on failure, every visited
    variable and equation is differentiated and the search moves to
    the differentiated equation. Raises
    :class:`~cubie.odesystems.symbolic.structural.errors.InvalidSystemError`
    for structurally singular systems.

    Returns the variable-equation :class:`Matching`, with matches on
    non-highest-differentiated variables cleared.
    """

    structure = state.structure
    graph = structure.graph
    var_to_diff = structure.var_to_diff
    eq_to_diff = structure.eq_to_diff
    neqs = graph.nsrcs()
    nvars = len(var_to_diff)
    vcolor = [False] * nvars
    ecolor = [False] * neqs
    var_eq_matching = Matching(nvars)
    neqs_orig = neqs
    nnonemptyeqs = sum(
        1
        for eq in range(neqs_orig)
        if graph.s_neighbors(eq) and eq_to_diff[eq] is None
    )

    varwhitelist = computed_highest_diff_variables(structure)

    if nnonemptyeqs > sum(varwhitelist):
        raise InvalidSystemError("System is structurally singular")

    for k in range(neqs_orig):
        eq_prime = k
        if eq_to_diff[eq_prime] is not None:
            continue
        if not graph.s_neighbors(eq_prime):
            continue
        pathfound = False
        for _ in range(_MAXITERS):
            # Match on highest-differentiated variables only.
            nvars = len(var_to_diff)
            neqs = graph.nsrcs()
            vcolor = [False] * nvars
            ecolor = [False] * neqs
            pathfound = construct_augmenting_path(
                var_eq_matching,
                graph,
                eq_prime,
                lambda v: varwhitelist[v],
                vcolor,
                ecolor,
            )
            if pathfound:
                break
            for var in range(len(vcolor)):
                if not vcolor[var]:
                    continue
                if var_to_diff[var] is None:
                    # Introduce a new (derivative) variable.
                    state.var_derivative(var)
                    var_eq_matching.push(UNASSIGNED)
                    varwhitelist.append(False)
                varwhitelist[var] = False
                varwhitelist[var_to_diff[var]] = True

            for eq in range(len(ecolor)):
                if not ecolor[eq]:
                    continue
                state.eq_derivative(eq, **kwargs)

            for var in range(len(vcolor)):
                if not vcolor[var]:
                    continue
                # Newly introduced variables and equations inherit
                # the assignment.
                var_eq_matching[var_to_diff[var]] = eq_to_diff[
                    var_eq_matching[var]
                ]
            eq_prime = eq_to_diff[eq_prime]
        if not pathfound:
            raise InvalidSystemError(
                f"maxiters={_MAXITERS} reached in Pantelides. File a "
                "bug report if your system has a reasonable index "
                "(<100)."
            )

    for var in range(state.structure.graph.ndsts()):
        if varwhitelist[var]:
            continue
        var_eq_matching[var] = UNASSIGNED
    return var_eq_matching
