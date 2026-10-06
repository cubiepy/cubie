"""Alias elimination.

Perfect-alias elimination via a sign-tracking union-find, trivial
tearing of explicit equations, and the integer-linear alias pass built
on singularity removal.

``eliminate_perfect_aliases``, ``alias_elimination`` and their helpers
are ported from ModelingToolkit.jl (commit a2b6dc56,
``src/systems/alias_elimination.jl``: ``eliminate_perfect_aliases!``,
``find_perfect_aliases!``, ``union_with_sign!``, ``pick_alias_target``,
``build_expr_from_coeffs_vars!`` and ``alias_elimination!``).
``trivial_tearing`` is ported from ModelingToolkit.jl (commit
c4177c335, ``src/systems/systemstructure.jl``, ``trivial_tearing!``).

Published Functions
-------------------
:func:`eliminate_perfect_aliases`
    Remove ``v ~ w`` / ``v ~ -w`` equations, substituting a chosen
    target through the system.

:func:`trivial_tearing`
    Remove explicit equations whose solved variable occurs nowhere
    else, recording them as observed.

:func:`alias_elimination`
    Integer-linear alias pass: singularity removal followed by
    rewriting the reduced rows back into the symbolic equations.
"""

import warnings
from typing import Dict, List, Tuple

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.structural.clil import SparseMatrixCLIL
from cubie.odesystems.symbolic.structural.singularity_removal import (
    IgnoreUnderconstrainedVariable,
    get_new_mm,
    structural_singularity_removal,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    Equation,
    StructuralState,
)


def _term_coeff_rest(term: ir.Expr) -> Tuple[object, ir.Expr]:
    """Split ``term`` into a numeric coefficient and the remainder.

    A ``Mul`` with a leading ``Num`` factor splits into that factor's
    value and the product of the remaining factors; a bare ``Num``
    splits into its value and
    :data:`~cubie.odesystems.symbolic.engine.expr.ONE`; anything else
    has coefficient ``1``.
    """

    if isinstance(term, ir.Mul) and isinstance(term.args[0], ir.Num):
        return term.args[0].value, ir.mul(*term.args[1:])
    if isinstance(term, ir.Num):
        return term.value, ir.ONE
    return 1, term


def _add_coeffs_dict(expr: ir.Add) -> Dict[ir.Expr, object]:
    """Map each term's symbolic remainder to its summed coefficient.

    Built from the terms of ``expr``; a constant term is keyed by
    :data:`~cubie.odesystems.symbolic.engine.expr.ONE`.
    """

    coeffs = {}
    for term in expr.args:
        coeff, rest = _term_coeff_rest(term)
        coeffs[rest] = coeffs.get(rest, 0) + coeff
    return coeffs


def _union_with_sign(
    parent: Dict[int, int],
    parity: Dict[int, int],
    members: Dict[int, List[int]],
    v1: int,
    v2: int,
    edge_sign: int,
) -> None:
    """Merge alias components under ``v1 ~ edge_sign * v2``.

    Weighted union-find with sign tracking: ``parent[v]`` is always
    the current root and ``parity[v]`` the sign of ``v`` relative to
    it (0 marks a contradictory component whose members are forced to
    zero). Merges are smaller-into-larger.
    """

    for v in (v1, v2):
        if v not in parent:
            parent[v] = v
            parity[v] = 1
            members[v] = [v]
    r1 = parent[v1]
    s1 = parity[v1]
    r2 = parent[v2]
    s2 = parity[v2]
    if r1 == r2:
        if s1 != edge_sign * s2:
            for m in members[r1]:
                parity[m] = 0
        return
    if len(members[r1]) < len(members[r2]):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    r2_to_r1 = s1 * edge_sign * s2
    r2_members = members[r2]
    for m in r2_members:
        parent[m] = r1
        parity[m] = parity[m] * r2_to_r1
    if r2_to_r1 == 0:
        for m in members[r1]:
            parity[m] = 0
    members[r1].extend(r2_members)
    del members[r2]


def _pick_alias_target(
    state: StructuralState, group_vars: List[int]
) -> int:
    """Choose the surviving variable of an alias group.

    Irreducible variables win outright; otherwise the highest
    state-priority variable, tie-broken by incidence degree and
    canonical rank.
    """

    structure = state.structure
    graph = structure.graph
    priorities = structure.state_priorities
    for v in group_vars:
        if state.fullvars[v] in state.irreducibles:
            return v
    max_priority = max(priorities[v] for v in group_vars)
    candidates = [
        v for v in group_vars if priorities[v] == max_priority
    ]
    if len(candidates) > 1 and max_priority > 0:
        if max_priority >= 100:
            tied_names = [state.fullvars[v] for v in candidates]
            warnings.warn(
                "Multiple variables in an alias group share the "
                f"highest state_priority ({max_priority}); choosing "
                "alias target by equation count. Tied variables: "
                f"{tied_names}"
            )
        max_degree = max(
            len(graph.d_neighbors(v)) for v in candidates
        )
        candidates = [
            v
            for v in candidates
            if len(graph.d_neighbors(v)) == max_degree
        ]
        candidates.sort(key=lambda v: structure.canonical_ranks[v])
    return candidates[0]


def _find_perfect_aliases(
    state: StructuralState,
    eqs_to_rm: List[int],
    vars_to_rm: List[int],
) -> Dict[int, int]:
    """Identify and rewrite perfect alias equations.

    Appends removable equations/variables to the given buffers and
    returns ``aliases``: removed variable index to target variable
    index.
    """

    structure = state.structure
    graph = structure.graph
    solvable_graph = structure.solvable_graph
    var_to_diff = structure.var_to_diff
    fullvars = state.fullvars
    eqs = state.eqs
    original_eqs = state.original_eqs

    aliases = {}
    subs = {}
    parent = {}
    parity = {}
    members = {}
    # (eq_index, v1, v2, edge_sign) with edge_sign encoding
    # v1 ~ edge_sign * v2.
    candidate_eqs = []

    for ieq in range(graph.nsrcs()):
        snbors = graph.s_neighbors(ieq)
        if len(snbors) != 2:
            continue
        if var_to_diff.diff_to_primal[snbors[0]] is not None:
            continue
        if var_to_diff.diff_to_primal[snbors[1]] is not None:
            continue
        eq = eqs[ieq]
        if not ir.is_zero(eq.lhs):
            continue
        v1_sym = fullvars[snbors[0]]
        v2_sym = fullvars[snbors[1]]
        rhs = eq.rhs
        if not isinstance(rhs, ir.Add):
            continue
        coeffs = _add_coeffs_dict(rhs)
        if coeffs.get(ir.ONE, 0) != 0:
            continue
        if len(coeffs) != 2:
            continue
        c1 = coeffs.get(v1_sym)
        c2 = coeffs.get(v2_sym)
        if c1 is None or c2 is None:
            continue
        if c1 + c2 == 0:
            edge_sign = 1
        elif c1 - c2 == 0:
            edge_sign = -1
        else:
            continue
        candidate_eqs.append((ieq, snbors[0], snbors[1], edge_sign))
        _union_with_sign(
            parent, parity, members, snbors[0], snbors[1], edge_sign
        )

    group_target = {}
    eqs_to_substitute = []
    irrs_by_root = {}
    zero = ir.ZERO

    def is_irreducible_v(v: int) -> bool:
        return fullvars[v] in state.irreducibles

    for root, group_vars in list(members.items()):
        if parity[root] == 0:
            # Conflict group: all non-irreducible members forced to 0.
            irrs_by_root[root] = [
                v for v in group_vars if is_irreducible_v(v)
            ]
            for v in group_vars:
                if is_irreducible_v(v):
                    state.always_present[v] = True
                    continue
                vars_to_rm.append(v)
                subs[fullvars[v]] = zero
                state.additional_observed.append(
                    Equation(fullvars[v], zero)
                )
                for e in list(graph.d_neighbors(v)):
                    eqs_to_substitute.append(e)
                graph.invview().set_neighbors(v, ())
                if solvable_graph is not None:
                    solvable_graph.invview().set_neighbors(v, ())
                dv = var_to_diff[v]
                while dv is not None:
                    vars_to_rm.append(dv)
                    subs[fullvars[dv]] = zero
                    for e in list(graph.d_neighbors(dv)):
                        eqs_to_substitute.append(e)
                    graph.invview().set_neighbors(dv, ())
                    if solvable_graph is not None:
                        solvable_graph.invview().set_neighbors(dv, ())
                    dv = var_to_diff[dv]
            continue

        target = _pick_alias_target(state, group_vars)
        group_target[root] = target
        target_p = parity[target]
        for v in group_vars:
            if is_irreducible_v(v) or v == target:
                state.always_present[v] = True
                continue
            s = parity[v] * target_p
            vars_to_rm.append(v)
            rhs_sym = (
                fullvars[target]
                if s == 1
                else ir.neg(fullvars[target])
            )
            subs[fullvars[v]] = rhs_sym
            state.additional_observed.append(
                Equation(fullvars[v], rhs_sym)
            )
            aliases[v] = target

            for e in list(graph.d_neighbors(v)):
                eqs_to_substitute.append(e)
                graph.rem_edge(e, v)
                graph.add_edge(e, target)
                if (
                    solvable_graph is not None
                    and solvable_graph.has_edge(e, v)
                ):
                    solvable_graph.rem_edge(e, v)
                    solvable_graph.add_edge(e, target)

            dv = var_to_diff[v]
            # One differentiation level below dtarget.
            prev_dtarget = target
            dtarget = var_to_diff[target]
            while dv is not None:
                if dtarget is None:
                    dtarget = state.var_derivative(prev_dtarget)
                vars_to_rm.append(dv)
                dsub = (
                    fullvars[dtarget]
                    if s == 1
                    else ir.neg(fullvars[dtarget])
                )
                subs[fullvars[dv]] = dsub
                aliases[dv] = dtarget
                for e in list(graph.d_neighbors(dv)):
                    eqs_to_substitute.append(e)
                    graph.rem_edge(e, dv)
                    graph.add_edge(e, dtarget)
                    if (
                        solvable_graph is not None
                        and solvable_graph.has_edge(e, dv)
                    ):
                        solvable_graph.rem_edge(e, dv)
                        solvable_graph.add_edge(e, dtarget)
                dv = var_to_diff[dv]
                prev_dtarget = dtarget
                dtarget = var_to_diff[dtarget]

    # Per-equation cleanup of the candidate alias equations.
    for ieq, v1, v2, _sign in candidate_eqs:
        if parity.get(v1, 1) == 0:
            irrs = irrs_by_root[parent[v1]]
            if not irrs:
                eqs_to_rm.append(ieq)
            else:
                v_pin = irrs.pop()
                graph.set_neighbors(ieq, [v_pin])
                if solvable_graph is not None:
                    solvable_graph.set_neighbors(ieq, [v_pin])
                eqs[ieq] = Equation(fullvars[v_pin], zero)
                original_eqs[ieq] = Equation(fullvars[v_pin], zero)
        else:
            target = group_target[parent[v1]]
            c1 = v1 if is_irreducible_v(v1) else target
            c2 = v2 if is_irreducible_v(v2) else target
            if c1 == c2:
                eqs_to_rm.append(ieq)

    for e in set(eqs_to_substitute):
        # Substitute twice: an alias substitution may cancel the
        # target, and zero substitution annihilates cofactors.
        eqs[e] = eqs[e].xreplace(subs).xreplace(subs)
        original_eqs[e] = (
            original_eqs[e].xreplace(subs).xreplace(subs)
        )
        new_vars = eqs[e].free_symbols()
        new_row = [
            v_idx
            for v_idx in graph.s_neighbors(e)
            if fullvars[v_idx] in new_vars
        ]
        graph.set_neighbors(e, new_row)
        if solvable_graph is not None:
            new_row = [
                v
                for v in new_row
                if solvable_graph.has_edge(e, v)
            ]
            solvable_graph.set_neighbors(e, new_row)

    # Remove duplicate structural aliases produced by redirection.
    seen = set()
    eqs_rm_set = set(eqs_to_rm)
    for ieq, _v1, _v2, _sign in candidate_eqs:
        if ieq in eqs_rm_set:
            continue
        snbors = graph.s_neighbors(ieq)
        if len(snbors) != 2:
            continue
        pair = (min(snbors), max(snbors))
        if pair in seen:
            eqs_to_rm.append(ieq)
        else:
            seen.add(pair)
    eqs_to_rm.sort()
    return aliases


def eliminate_perfect_aliases(
    state: StructuralState,
) -> Tuple[List[int], List[int], Dict[int, int]]:
    """Remove perfect alias equations from ``state``.

    Returns ``(old_to_new_eq, old_to_new_var, aliases)`` where
    ``aliases`` maps removed (old) variable indices to their (old)
    target variable indices.
    """

    state.structure.complete()
    eqs_to_rm = []
    vars_to_rm = []
    aliases = _find_perfect_aliases(state, eqs_to_rm, vars_to_rm)
    old_to_new_eq, old_to_new_var = state.rm_eqs_vars(
        eqs_to_rm, vars_to_rm
    )
    return old_to_new_eq, old_to_new_var, aliases


def trivial_tearing(state: StructuralState) -> None:
    """Tear explicit observed equations before simplification.

    An equation ``x ~ f(...)`` is torn when ``x`` is an unknown that
    is neither irreducible nor part of a derivative chain, occurs in
    no other equation that is not already torn, does not occur in
    ``f``, and every other variable of the equation occurs in some
    other equation that is not torn. Torn equations and their
    variables are removed from ``state`` and the equations appended
    to ``state.additional_observed``.

    Parameters
    ----------
    state
        The structural state, mutated in place.
    """

    trivial_idxs = set()
    blacklist = set()
    torn_eqs = []
    matched_vars = set()
    var_to_idx = state.var2idx
    sys_eqs = state.eqs

    state.structure.complete()
    var_to_diff = state.structure.var_to_diff
    graph = state.structure.graph
    while True:
        added_equation = False
        for i, eq in enumerate(state.original_eqs):
            if i in trivial_idxs or i in blacklist:
                continue
            vari = var_to_idx.get(eq.lhs)
            if vari is None:
                continue
            if eq.lhs in state.irreducibles:
                blacklist.add(i)
                continue
            # A var ~ var equation is stored as 0 ~ 0.
            sys_eq = sys_eqs[i]
            if ir.is_zero(sys_eq.lhs) and ir.is_zero(sys_eq.rhs):
                continue
            if var_to_diff[vari] is not None:
                continue
            if var_to_diff.diff_to_primal[vari] is not None:
                continue
            eqidxs = [
                e for e in graph.d_neighbors(vari) if e not in trivial_idxs
            ]
            if len(eqidxs) != 1:
                continue
            eqi = eqidxs[0]

            isvalid = True
            for v in graph.s_neighbors(eqi):
                if v == vari or v in matched_vars:
                    continue
                n_untorn = sum(
                    1 for e in graph.d_neighbors(v) if e not in trivial_idxs
                )
                # One of the counted equations is eqi itself.
                isvalid = n_untorn > 1
                if not isvalid:
                    break
            if not isvalid:
                continue
            if eq.lhs in ir.free_atoms(eq.rhs):
                blacklist.add(i)
                continue

            added_equation = True
            trivial_idxs.add(eqi)
            torn_eqs.append(eq)
            matched_vars.add(vari)

        if not added_equation:
            break

    state.rm_eqs_vars(sorted(trivial_idxs), sorted(matched_vars))
    state.additional_observed.extend(torn_eqs)


def alias_elimination(
    state: StructuralState,
    print_underconstrained_variables: bool = False,
    **kwargs,
) -> SparseMatrixCLIL:
    """Integer-linear alias elimination pass.

    Runs singularity removal, rewrites the reduced rows back into the
    symbolic equations, removes equations that reduced to ``0 ~ 0``,
    and returns the updated integer subsystem matrix.
    """

    state.structure.complete()
    eqs_to_rm = []
    vars_to_rm = []
    aliases = {}

    underconstrained_hook = IgnoreUnderconstrainedVariable()
    mm = structural_singularity_removal(
        state,
        variable_underconstrained=underconstrained_hook,
        **kwargs,
    )
    if print_underconstrained_variables:
        names = [
            state.fullvars[v]
            for v in underconstrained_hook.underconstrained
        ]
        warnings.warn(
            f"Found underconstrained variables in the system: {names}"
        )

    fullvars_to_idx = state.var2idx
    eqs = state.eqs
    original_eqs = state.original_eqs

    for ieq, eq in enumerate(mm.nzrows):
        rcol = mm.row_cols[ieq]
        rval = mm.row_vals[ieq]
        if not rcol:
            eqs_to_rm.append(eq)
            continue
        state.rewrite_from_row(eq, rcol, rval)
        rhs = eqs[eq].rhs
        oeq = original_eqs[eq]
        lhs = oeq.lhs
        idx = fullvars_to_idx.get(lhs)
        colidx = None
        if idx is not None:
            for c_i, c in enumerate(rcol):
                if c == idx:
                    colidx = c_i
                    break
        if (
            idx is not None
            and colidx is not None
            and rval[colidx] == -1
        ):
            original_eqs[eq] = Equation(oeq.lhs, rhs + oeq.lhs)
        else:
            original_eqs[eq] = eqs[eq]

    old_to_new_eq, old_to_new_var = state.rm_eqs_vars(
        eqs_to_rm, vars_to_rm
    )
    return get_new_mm(aliases, old_to_new_eq, old_to_new_var, mm)
