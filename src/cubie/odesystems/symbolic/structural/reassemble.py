"""Reassembly of the simplified system from tearing decisions.

Cuts dummy derivatives from their base variables, lowers higher-order
derivatives to first order, solves each matched equation for its
variable (producing differential equations and observed equations),
leaves torn equations as algebraic residuals, and collects the
simplified system.

Ported from ModelingToolkit.jl (commit c4177c335,
``src/structural_transformation/symbolics_tearing.jl``):
``default_reassemble`` (``DefaultReassembleAlgorithm`` with
``update_simplified_system!``), ``substitute_derivatives_algevars``,
``generate_derivative_variables``, ``find_duplicate_dd``,
``_insert_sccs``, ``get_sorted_scc``, ``EquationGenerator``,
``get_extra_eqs_vars`` (``_extra_vars``) and
``generate_system_equations``, each from the function of the same
name less any leading underscore or trailing ``!``.

Published Classes
-----------------
:class:`SimplifiedSystem`
    The simplification result consumed by cubie's parser/codegen.

Published Functions
-------------------
:func:`default_reassemble`
    Run the default reassembly on a tearing result.
"""

from typing import Dict, List, Optional, Set, Tuple

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.assignments import (
    topological_sort,
)
from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    Matching,
    SELECTED_STATE,
    UNASSIGNED,
)
from cubie.odesystems.symbolic.structural.digraph import (
    DiCMOBiGraphF,
    toposort_equations,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    fixpoint_sub,
    linear_expansion,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)
from cubie.odesystems.symbolic.structural.tearing import TearingResult


class SimplifiedSystem:
    """Result of structural simplification.

    Parameters
    ----------
    differential_states
        States integrated through their solved derivatives, in BLT
        order.
    algebraic_states
        The torn (iteration) variables constrained by ``residuals``,
        in BLT order.
    dxdt
        Map from each differential state to its explicit derivative
        expression.
    residuals
        Algebraic residual expressions (each constrained to zero).
        Empty for fully torn systems.
    observed
        Topologically sorted ``(symbol, expression)`` assignments for
        eliminated variables.
    """

    def __init__(
        self,
        differential_states: List[ir.Sym],
        algebraic_states: List[ir.Sym],
        dxdt: Dict[ir.Sym, ir.Expr],
        residuals: List[ir.Expr],
        observed: List[Tuple[ir.Sym, ir.Expr]],
    ) -> None:
        self.differential_states = differential_states
        self.algebraic_states = algebraic_states
        self.dxdt = dxdt
        self.residuals = residuals
        self.observed = observed


def substitute_derivatives_algevars(
    state: StructuralState, var_eq_matching: Matching
) -> None:
    """Cut the derivatives of non-selected variables from their base.

    State selection may determine that some differential variables
    are algebraic variables in disguise; their derivative variables
    (``x_t``) become ordinary algebraic variables. After this pass,
    ``SelectedState`` information is no longer needed.
    """

    for var in range(len(state.fullvars)):
        dv = state.derivative_of(var)
        if dv is None:
            continue
        if var_eq_matching[var] is SELECTED_STATE:
            continue
        state.registry.cut(state.fullvars[dv])


def find_duplicate_dd(
    state: StructuralState, dv: int, linear_eqs: Dict[int, int]
) -> Optional[Tuple[int, int]]:
    """Find a pre-existing ``0 ~ D(x) - y`` equation for ``dv``."""

    mm = state.mm
    for eq in state.solvable_graph.d_neighbors(dv):
        mi = linear_eqs.get(eq)
        if mi is None:
            continue
        rvs = mm.row_cols[mi]
        nzs = mm.row_vals[mi]
        if (
            len(nzs) == 2
            and abs(nzs[0]) == 1
            and nzs[0] == -nzs[1]
        ):
            v_t = rvs[1] if rvs[0] == dv else rvs[0]
            if state.primal_of(v_t) is None:
                return eq, v_t
    return None


def _insert_sccs(
    var_sccs: List[List[int]],
    sccs_to_insert: List[Tuple[int, List[int]]],
) -> List[List[int]]:
    """Insert singleton SCCs at the requested indices."""

    old_idx = 0
    insert_idx = 0
    new_sccs = []
    total = len(var_sccs) + len(sccs_to_insert)
    for _ in range(total):
        if (
            insert_idx < len(sccs_to_insert)
            and sccs_to_insert[insert_idx][0] == old_idx
        ):
            new_sccs.append(sccs_to_insert[insert_idx][1])
            insert_idx += 1
        else:
            new_sccs.append(list(var_sccs[old_idx]))
            old_idx += 1
    return [scc for scc in new_sccs if scc]


def generate_derivative_variables(
    state: StructuralState,
    var_eq_matching: Matching,
    full_var_eq_matching: Matching,
    var_sccs: List[List[int]],
) -> List[List[int]]:
    """Lower the system to first order.

    For every differentiated variable ``x`` whose derivative ``x_t``
    is solved from no equation, ``x_t`` becomes a state of its own:
    ``x`` takes a new derivative variable ``D(x)`` matched to the new
    equation ``0 ~ D(x) - x_t``. When an equation ``0 ~ x_t - y``
    already exists, ``x_t`` is matched to it instead and ``y`` takes
    over the derivative of ``x_t``. Each equation solving ``D(x)``
    forms a singleton SCC before the SCC of ``x_t``. Returns the new
    SCC list.
    """

    graph = state.graph
    registry = state.registry
    linear_eqs = {e: i for i, e in enumerate(state.mm.nzrows)}

    v_to_scc = [None] * graph.ndsts()
    for i, scc in enumerate(var_sccs):
        for j, v in enumerate(scc):
            v_to_scc[v] = (i, j)

    # (unsolved derivative, its variable as a state, the variable
    # matched to the equation solving the derivative)
    lowered = []

    for v in range(graph.ndsts()):
        dv = state.derivative_of(v)
        if dv is None:
            continue
        if isinstance(var_eq_matching[dv], int):
            continue

        duplicate = find_duplicate_dd(state, dv, linear_eqs)
        if duplicate is None:
            x_t = state.fullvars[dv]
            registry.cut(x_t)
            dx = state.add_variable(
                registry.derivative(state.fullvars[v]), dv
            )
            dummy_eq = state.add_equation(
                (ir.ZERO, ir.sub(state.fullvars[dx], x_t)), [dx, dv]
            )
            state.solvable_graph.add_edge(dummy_eq, dx)
            var_eq_matching.push(UNASSIGNED)
            full_var_eq_matching.push(UNASSIGNED)
            var_eq_matching[dx] = dummy_eq
            var_eq_matching[dv] = UNASSIGNED
            full_var_eq_matching[dx] = dummy_eq
            lowered.append((dv, dv, dx))
            continue
        dummy_eq, v_t = duplicate
        registry.move_derivative(state.fullvars[dv], state.fullvars[v_t])
        old_matched_eq = full_var_eq_matching[dv]
        var_eq_matching[dv] = dummy_eq
        full_var_eq_matching[dv] = dummy_eq
        full_var_eq_matching[v_t] = old_matched_eq
        lowered.append((dv, v_t, dv))

    sccs_to_insert = []
    idxs_to_remove = {}
    for dv, v_t, dx in lowered:
        i, j = v_to_scc[dv]
        if v_t != dv:
            var_sccs[i][j] = v_t
            i2, j2 = v_to_scc[v_t]
            idxs_to_remove.setdefault(i2, []).append(j2)
        # Emit D(x) first, so later equations read its solution.
        sccs_to_insert.append((i, [dx]))
    sccs_to_insert.sort(key=lambda pair: pair[0])

    for i, idxs in idxs_to_remove.items():
        for j in sorted(idxs, reverse=True):
            del var_sccs[i][j]
    return _insert_sccs(var_sccs, sccs_to_insert)


def get_sorted_scc(
    digraph: DiCMOBiGraphF,
    full_var_eq_matching: Matching,
    var_eq_matching: Matching,
    scc: List[int],
) -> Tuple[List[int], List[int]]:
    """Sort one SCC's variables and equations into solve order."""

    eq_var_matching = var_eq_matching.invview()
    scc_eqs = []
    scc_solved_eqs = []
    for v in scc:
        e = full_var_eq_matching[v]
        if isinstance(e, int):
            scc_eqs.append(e)
        e = var_eq_matching[v]
        if isinstance(e, int):
            scc_solved_eqs.append(e)
    sorted_solved = toposort_equations(digraph, scc_solved_eqs)
    solved_set = set(scc_solved_eqs)
    scc_eqs_sorted = sorted_solved + [
        e for e in scc_eqs if e not in solved_set
    ]
    scc_vars = []
    for e in scc_eqs_sorted:
        v = eq_var_matching[e] if e < len(eq_var_matching.match) else (
            UNASSIGNED
        )
        if isinstance(v, int):
            scc_vars.append(v)
    var_set = set(scc_vars)
    scc_vars.extend(v for v in scc if v not in var_set)
    return scc_vars, scc_eqs_sorted


def _solve_for(
    equation: Tuple[ir.Expr, ir.Expr], var: ir.Sym
) -> ir.Expr:
    lhs, rhs = equation
    a, b, islinear = linear_expansion(ir.sub(lhs, rhs), var)
    if not islinear or ir.is_zero(a):
        raise ValueError(
            f"equation {lhs} ~ {rhs} is not solvable for {var} despite "
            "a solvable-graph edge"
        )
    return ir.div(ir.neg(b), a)


class EquationGenerator:
    """Accumulates generated equations and their orderings."""

    def __init__(self, state: StructuralState) -> None:
        self.state = state
        self.total_sub = {}
        self.differential_states = []
        self.dxdt = {}
        self.residuals = []
        self.eq_ordering = []
        self.var_ordering = []
        self.solved_eqs = []
        self.solved_vars = []

    def codegen_equation(self, ieq: int, iv) -> None:
        """Generate the output form of equation ``ieq``.

        Solvable equations of derivative variables become
        differential equations of the variable they derive; solvable
        equations of algebraic variables become observed equations;
        everything else stays as an algebraic residual.
        """

        state = self.state
        graph = state.graph
        total_sub = self.total_sub
        lhs, rhs = state.eqs[ieq]

        issolvable = isinstance(iv, int) and state.solvable_graph.has_edge(
            ieq, iv
        )
        primal = state.primal_of(iv) if issolvable else None
        if primal is not None:
            var = state.fullvars[iv]
            solved = fixpoint_sub(_solve_for((lhs, rhs), var), total_sub)
            # Any equation incident on `iv` will have it substituted:
            # rewire incidence through this equation's variables.
            for e in list(graph.d_neighbors(iv)):
                if e == ieq:
                    continue
                for v in graph.s_neighbors(ieq):
                    graph.add_edge(e, v)
                graph.rem_edge(e, iv)
            total_sub[var] = solved
            self.differential_states.append(state.fullvars[primal])
            self.dxdt[state.fullvars[primal]] = solved
            self.eq_ordering.append(ieq)
            self.var_ordering.append(primal)
        elif issolvable:
            var = state.fullvars[iv]
            solved = fixpoint_sub(_solve_for((lhs, rhs), var), total_sub)
            self.solved_eqs.append((var, solved))
            self.solved_vars.append(iv)
        else:
            self.residuals.append(fixpoint_sub(ir.sub(rhs, lhs), total_sub))
            self.eq_ordering.append(ieq)
            self.var_ordering.append(-1)


def _extra_vars(
    state: StructuralState,
    var_eq_matching: Matching,
    full_var_eq_matching: Matching,
) -> List[int]:
    """Variables neither matched before tearing nor selected or solved."""

    return [
        v
        for v in range(state.graph.ndsts())
        if not isinstance(full_var_eq_matching[v], int)
        and var_eq_matching[v] is UNASSIGNED
    ]


def torn_partner(
    eq: int,
    var_eq_matching: Matching,
    full_eq_var_matching: Matching,
) -> Optional[int]:
    """Torn variable reached from residual ``eq`` by alternating paths.

    Follows the pre-tearing matching from ``eq`` to a variable and,
    while that variable is solved by tearing, on through its solving
    equation's pre-tearing partner. The torn variable reached enters
    ``eq`` directly or through the solved variables passed on the
    way, and distinct residuals reach distinct torn variables.
    Returns ``None`` when the path ends without a torn variable.
    """

    def partner(e: int):
        if e < len(full_eq_var_matching.match):
            return full_eq_var_matching[e]
        return UNASSIGNED

    v = partner(eq)
    while isinstance(v, int):
        solving_eq = var_eq_matching[v]
        if solving_eq is UNASSIGNED:
            return v
        if not isinstance(solving_eq, int):
            return None
        v = partner(solving_eq)
    return None


def generate_system_equations(
    state: StructuralState,
    var_eq_matching: Matching,
    full_var_eq_matching: Matching,
    var_sccs: List[List[int]],
    extra_eqs: List[int],
    extra_vars: List[int],
) -> Tuple[EquationGenerator, List[int]]:
    """Solve matched equations and order the system into BLT form.

    Returns the generator holding the generated equations and the
    variable ordering: the variable of each differential equation
    and residual in generation order, then every other unsolved
    variable by index.
    """

    graph = state.graph
    eq_var_matching = var_eq_matching.invview()

    gen = EquationGenerator(state)

    # Solve extra (overdetermined) equations first to respect
    # topological order.
    for eq in extra_eqs:
        var = (
            eq_var_matching[eq]
            if eq < len(eq_var_matching.match)
            else UNASSIGNED
        )
        if not isinstance(var, int):
            continue
        gen.codegen_equation(eq, var)

    digraph = DiCMOBiGraphF(graph, var_eq_matching)
    for i, scc in enumerate(var_sccs):
        vscc, escc = get_sorted_scc(
            digraph, full_var_eq_matching, var_eq_matching, scc
        )
        var_sccs[i] = vscc
        if len(escc) != len(vscc):
            if not escc:
                continue
            escc = [e for e in escc if e not in set(extra_eqs)]
            if not escc:
                continue
            vscc = [v for v in vscc if v not in set(extra_vars)]
            if not vscc:
                continue

        for ieq in escc:
            iv = (
                eq_var_matching[ieq]
                if ieq < len(eq_var_matching.match)
                else UNASSIGNED
            )
            gen.codegen_equation(ieq, iv)

    for eq in extra_eqs:
        var = (
            eq_var_matching[eq]
            if eq < len(eq_var_matching.match)
            else UNASSIGNED
        )
        if isinstance(var, int):
            continue
        gen.codegen_equation(eq, var)

    var_ordering = gen.var_ordering
    solved_vars_set = set(gen.solved_vars)

    # Each residual takes the torn variable its matching reaches.
    full_eq_var_matching = full_var_eq_matching.invview()
    for i, v in enumerate(var_ordering):
        if v >= 0:
            continue
        paired = torn_partner(
            gen.eq_ordering[i], var_eq_matching, full_eq_var_matching
        )
        if paired is not None:
            var_ordering[i] = paired

    def ispresent(i: int) -> bool:
        if graph.d_neighbors(i):
            return True
        dvi = state.derivative_of(i)
        return dvi is not None and bool(graph.d_neighbors(dvi))

    # Fill unpaired algebraic (torn) variable slots.
    paired_vars = {v for v in var_ordering if v >= 0}
    offset = 0
    for i, v in enumerate(var_ordering):
        if v >= 0:
            continue
        index = None
        for j in range(offset, graph.ndsts()):
            if (
                j not in paired_vars
                and j not in solved_vars_set
                and state.primal_of(j) is None
                and ispresent(j)
            ):
                index = j
                break
        if index is None:
            break
        var_ordering[i] = index
        offset = index + 1
    var_ordering = [v for v in var_ordering if v >= 0]
    used = set(var_ordering) | solved_vars_set
    var_ordering = var_ordering + [
        v for v in range(graph.ndsts()) if v not in used
    ]
    return gen, var_ordering


def _matched_closure(
    graph: BipartiteGraph, var_eq_matching: Matching, var: int
) -> Set[int]:
    """``var`` and every variable its matched equations read, recursively."""

    seen = {var}
    stack = [var]
    while stack:
        eq = var_eq_matching[stack.pop()]
        if not isinstance(eq, int):
            continue
        for w in graph.s_neighbors(eq):
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def _unknowns(
    state: StructuralState,
    gen: EquationGenerator,
    var_eq_matching: Matching,
    var_ordering: List[int],
) -> List[ir.Sym]:
    """Unknowns of the reassembled system, in ``var_ordering`` order.

    A variable is reached when a differential equation or residual
    reads it, directly or through the matched equations of the solved
    variables it reads. An unknown is an unsolved variable that is
    reached or whose derivative is reached, and that is not the
    derivative of another unsolved variable.
    """

    graph = state.graph
    solved = set(gen.solved_vars)
    reached = set()
    for e in gen.eq_ordering:
        for v in graph.s_neighbors(e):
            if v in solved:
                reached |= _matched_closure(graph, var_eq_matching, v)
            else:
                reached.add(v)
    reached -= solved

    def is_unknown(v: int) -> bool:
        primal = state.primal_of(v)
        if primal is not None and primal not in solved:
            return False
        return v in reached or state.derivative_of(v) in reached

    return [state.fullvars[v] for v in var_ordering if is_unknown(v)]


def default_reassemble(
    state: StructuralState,
    tearing_result: TearingResult,
    fully_determined: bool = True,
) -> SimplifiedSystem:
    """Reassemble the simplified system from a tearing result."""

    var_eq_matching = tearing_result.var_eq_matching
    full_var_eq_matching = tearing_result.full_var_eq_matching
    var_sccs = [list(s) for s in tearing_result.var_sccs]

    if fully_determined:
        extra_eqs, extra_vars = [], []
    else:
        extra_eqs = tearing_result.free_eqs
        extra_vars = _extra_vars(
            state, var_eq_matching, full_var_eq_matching
        )
    extra_unknowns = [state.fullvars[v] for v in extra_vars]

    substitute_derivatives_algevars(state, var_eq_matching)
    var_sccs = generate_derivative_variables(
        state, var_eq_matching, full_var_eq_matching, var_sccs
    )
    gen, var_ordering = generate_system_equations(
        state,
        var_eq_matching,
        full_var_eq_matching,
        var_sccs,
        extra_eqs,
        extra_vars,
    )

    unknowns = _unknowns(state, gen, var_eq_matching, var_ordering)
    for extra in extra_unknowns:
        if extra not in unknowns:
            unknowns.append(extra)
    diff_set = set(gen.differential_states)

    observed = list(gen.solved_eqs)
    observed.extend(
        (lhs, ir.xreplace(rhs, gen.total_sub))
        for lhs, rhs in state.additional_observed
    )

    return SimplifiedSystem(
        gen.differential_states,
        [s for s in unknowns if s not in diff_set],
        gen.dxdt,
        gen.residuals,
        topological_sort(observed),
    )
