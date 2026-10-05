"""Modia tearing and contraction of eliminated variables.

Ported from ModelingToolkit.jl c4177c335: ``contract_variables``,
``free_equations`` and ``TearingResult`` from
``src/structural_transformation/tearing.jl``; ``try_assign_eq!``,
``tearEquations!``, ``tear_graph_block_modia!``,
``build_var_eq_matching`` and ``ModiaTearing`` from
``src/structural_transformation/bipartite_tearing/modia_tearing.jl``
(derived from Modia.jl).

Published Classes
-----------------
:class:`TearingResult`
    Torn matching, pre-tearing matching and variable SCCs.

:class:`ModiaTearing`
    Modia tearing of each variable SCC of a system structure.

Published Functions
-------------------
:func:`contract_variables`
    Contract eliminated variables out of an incidence graph.

:func:`free_equations`
    Equations not matched to any filtered variable.
"""

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from cubie.odesystems.symbolic.structural.bipartite import (
    UNASSIGNED,
    BipartiteGraph,
    Matching,
    _always_true,
    maximal_matching,
)
from cubie.odesystems.symbolic.structural.digraph import (
    DiCMOBiGraphT,
    IncrementalCycleTracker,
    find_var_sccs,
    neighborhood_in,
)
from cubie.odesystems.symbolic.structural.singularity_removal import (
    RestrictedBareissContext,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    SystemStructure,
)


def contract_variables(
    graph: BipartiteGraph,
    var_eq_matching: Matching,
    var_rename: List[int],
    eq_rename: List[int],
    nelim_eq: int,
    nelim_var: int,
) -> BipartiteGraph:
    """Contract eliminated variables out of the incidence graph.

    Every incidence on an eliminated variable is replaced by
    incidences on the retained variables it (transitively) depends on
    through the matching-induced digraph. ``var_rename``/``eq_rename``
    map old indices to new 0-based indices with ``-1`` marking
    eliminated entries.
    """

    dig = DiCMOBiGraphT(graph, var_eq_matching)
    var_deps = [
        [
            var_rename[v2]
            for v2 in neighborhood_in(dig, v)
            if var_rename[v2] != -1
        ]
        for v in range(graph.ndsts())
    ]

    newgraph = BipartiteGraph(
        graph.nsrcs() - nelim_eq, graph.ndsts() - nelim_var
    )
    for e in range(graph.nsrcs()):
        ne = eq_rename[e]
        if ne == -1:
            continue
        for v in graph.s_neighbors(e):
            newvar = var_rename[v]
            if newvar != -1:
                newgraph.add_edge(ne, newvar)
            else:
                for nv in var_deps[v]:
                    newgraph.add_edge(ne, nv)
    return newgraph


def free_equations(
    graph: BipartiteGraph,
    vars_scc: List[List[int]],
    var_eq_matching: Matching,
    varfilter: Callable[[int], bool],
) -> List[int]:
    """Equations not matched to any variable that passes ``varfilter``.

    Parameters
    ----------
    graph
        Incidence graph of equations and variables.
    vars_scc
        Variable SCCs.
    var_eq_matching
        Variable-to-equation matching.
    varfilter
        Predicate selecting the variables whose matches count.

    Returns
    -------
    list[int]
        Unmatched equation indices in ascending order.
    """

    seen_eqs = [False] * graph.nsrcs()
    for scc in vars_scc:
        for var in scc:
            if not varfilter(var):
                continue
            ieq = var_eq_matching[var]
            if isinstance(ieq, int):
                seen_eqs[ieq] = True
    return [eq for eq, seen in enumerate(seen_eqs) if not seen]


class TearingResult:
    """Result of tearing a system structure.

    Parameters
    ----------
    var_eq_matching
        Torn matching: differential variables are matched to
        ``SELECTED_STATE``, solved variables to the equation that
        solves them and torn algebraic variables to ``UNASSIGNED``.
    full_var_eq_matching
        Maximal matching before tearing, used to compute
        ``var_sccs``.
    var_sccs
        Variable SCCs in dependency order.
    """

    def __init__(
        self,
        var_eq_matching: Matching,
        full_var_eq_matching: Matching,
        var_sccs: List[List[int]],
    ) -> None:
        self.var_eq_matching = var_eq_matching
        self.full_var_eq_matching = full_var_eq_matching
        self.var_sccs = var_sccs


def try_assign_eq(ict: IncrementalCycleTracker, vj: int, eq: int) -> bool:
    """Match ``vj`` to ``eq`` unless that closes a cycle.

    Returns
    -------
    bool
        Whether the assignment was made.
    """

    graph = ict.graph

    def assign(g: DiCMOBiGraphT) -> None:
        g.matching[vj] = eq
        g.ne += len(g.graph.s_neighbors(eq)) - 1

    return ict.add_edge_checked(
        assign,
        (v for v in graph.graph.s_neighbors(eq) if v != vj),
        vj,
    )


def try_assign_eq_vars(
    ict: IncrementalCycleTracker,
    variables: Iterable[int],
    v_active: Set[int],
    eq: int,
    condition: Callable[[int], bool] = _always_true,
) -> bool:
    """Match ``eq`` to the first eligible variable that stays acyclic.

    A variable is eligible when it is active, unmatched and passes
    ``condition``.

    Returns
    -------
    bool
        Whether an assignment was made.
    """

    graph = ict.graph
    for vj in variables:
        if not (
            vj in v_active
            and graph.matching[vj] is UNASSIGNED
            and condition(vj)
        ):
            continue
        if try_assign_eq(ict, vj, eq):
            return True
    return False


def tear_equations(
    ict: IncrementalCycleTracker,
    solvable_adjacency: List[List[int]],
    eqs: List[int],
    v_active: Set[int],
    isder: Optional[Callable[[int], bool]],
) -> IncrementalCycleTracker:
    """Greedily match equations to the variables they can solve for.

    Equations with a single solvable variable are assigned first. When
    ``isder`` is given, an equation with an eligible differentiated
    variable is only matched to differentiated variables.

    Parameters
    ----------
    ict
        Cycle tracker over the variable digraph being matched.
    solvable_adjacency
        Solvable variables of each equation.
    eqs
        Equations to assign.
    v_active
        Variables that may be assigned.
    isder
        Predicate marking differentiated variables, or ``None``.

    Returns
    -------
    IncrementalCycleTracker
        ``ict``, with its matching extended.
    """

    check_der = isder is not None
    has_der = [False]

    def isder_seen(v: int) -> bool:
        r = isder(v)
        has_der[0] |= r
        return r

    for only_single_solvable in (True, False):
        for eq in eqs:
            vs = solvable_adjacency[eq]
            if (len(vs) == 1) != only_single_solvable:
                continue
            if check_der:
                # Only consider differentiated variables when the
                # equation has any.
                try_assign_eq_vars(ict, vs, v_active, eq, isder_seen)
                if has_der[0]:
                    has_der[0] = False
                    continue
            try_assign_eq_vars(ict, vs, v_active, eq)
    return ict


def tear_graph_block_modia(
    var_eq_matching: Matching,
    ict: IncrementalCycleTracker,
    solvable_graph: BipartiteGraph,
    eqs: List[int],
    variables: Set[int],
    isder: Optional[Callable[[int], bool]],
) -> None:
    """Tear one block and copy its variables' matches.

    Parameters
    ----------
    var_eq_matching
        Matching receiving the torn assignment of ``variables``.
    ict
        Cycle tracker over the variable digraph being matched.
    solvable_graph
        Solvability graph of the system.
    eqs
        Equations of the block.
    variables
        Variables of the block.
    isder
        Predicate marking differentiated variables, or ``None``.
    """

    tear_equations(ict, solvable_graph.fadjlist, eqs, variables, isder)
    for var in sorted(variables):
        var_eq_matching[var] = ict.graph.matching[var]


def build_var_eq_matching(
    structure: SystemStructure,
    varfilter: Callable[[int], bool],
    eqfilter: Callable[[int], bool],
) -> Tuple[Matching, int]:
    """Maximal matching of filtered variables to filtered equations.

    Returns
    -------
    tuple[Matching, int]
        The completed matching and its length.
    """

    var_eq_matching = maximal_matching(structure.graph, eqfilter, varfilter)
    matching_len = max(
        len(var_eq_matching),
        max(
            (x + 1 for x in var_eq_matching if isinstance(x, int)),
            default=0,
        ),
    )
    return var_eq_matching.complete(matching_len), matching_len


class ModiaTearing:
    """Modia tearing of each variable SCC.

    Parameters
    ----------
    isder
        Predicate marking differentiated variables, or ``None``. An
        equation with a solvable differentiated variable is only
        solved for differentiated variables.
    varfilter
        Predicate selecting the variables that may be solved for.
    eqfilter
        Predicate selecting the equations that may be matched.
    """

    def __init__(
        self,
        isder: Optional[Callable[[int], bool]] = None,
        varfilter: Callable[[int], bool] = _always_true,
        eqfilter: Callable[[int], bool] = _always_true,
    ) -> None:
        self.isder = isder
        self.varfilter = varfilter
        self.eqfilter = eqfilter

    def __call__(
        self, structure: SystemStructure
    ) -> Tuple[TearingResult, Dict]:
        """Tear ``structure``.

        Returns
        -------
        tuple[TearingResult, dict]
            The tearing result and an empty dict of extra data.
        """

        isder = self.isder
        varfilter = self.varfilter
        graph = structure.graph
        solvable_graph = structure.solvable_graph
        var_eq_matching, matching_len = build_var_eq_matching(
            structure, varfilter, self.eqfilter
        )
        full_var_eq_matching = var_eq_matching.copy()
        var_sccs = find_var_sccs(graph, var_eq_matching)
        vargraph = DiCMOBiGraphT(graph, Matching(matching_len))
        ict = IncrementalCycleTracker(vargraph)

        ieqs = []
        filtered_vars = set()
        free_eqs = free_equations(graph, var_sccs, var_eq_matching, varfilter)
        is_overdetermined = len(free_eqs) > 0
        for scc in var_sccs:
            for var in scc:
                if varfilter(var):
                    filtered_vars.add(var)
                    if var_eq_matching[var] is not UNASSIGNED:
                        ieqs.append(var_eq_matching[var])
                var_eq_matching[var] = UNASSIGNED
            tear_graph_block_modia(
                var_eq_matching,
                ict,
                solvable_graph,
                ieqs,
                filtered_vars,
                isder,
            )
            # Keep the block's assignments when free equations may
            # close loops through it.
            if not is_overdetermined:
                vargraph.ne = 0
                for var in scc:
                    vargraph.matching[var] = UNASSIGNED
            ieqs = []
            filtered_vars = set()
        if is_overdetermined:
            free_vars = {
                v
                for v, x in enumerate(var_eq_matching)
                if not isinstance(x, int)
            }
            tear_graph_block_modia(
                var_eq_matching,
                ict,
                solvable_graph,
                free_eqs,
                free_vars,
                isder,
            )

        return (
            TearingResult(var_eq_matching, full_var_eq_matching, var_sccs),
            {},
        )
