"""Alias elimination.

Perfect-alias elimination via a sign-tracking union-find, trivial
tearing of explicit equations, and the integer-linear alias pass built
on singularity removal.

``eliminate_perfect_aliases``, ``alias_elimination`` and their helpers
are ported from ModelingToolkit.jl (commit a2b6dc56,
``src/systems/alias_elimination.jl``: ``eliminate_perfect_aliases!``,
``find_perfect_aliases!``, ``union_with_sign!``, ``pick_alias_target``,
``alias_elimination!``).
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
from typing import Dict, List, Optional, Tuple

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.structural.clil import SparseMatrixCLIL
from cubie.odesystems.symbolic.structural.singularity_removal import (
    get_new_mm,
    structural_singularity_removal,
)
from cubie.odesystems.symbolic.structural.symbolics import (
    linear_expansion,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    StructuralState,
)


def _alias_sign(
    rhs: ir.Expr, v1: ir.Sym, v2: ir.Sym
) -> Optional[int]:
    """Sign ``s`` with ``0 ~ rhs`` meaning ``v1 ~ s * v2``.

    ``rhs`` must be a sum of numeric multiples of ``v1`` and ``v2``
    alone, the multiples equal in magnitude; otherwise ``None``.
    """

    if not isinstance(rhs, ir.Add):
        return None
    c1, rest, linear = linear_expansion(rhs, v1)
    if not linear or not isinstance(c1, ir.Num):
        return None
    c2, rest, linear = linear_expansion(rest, v2)
    if not linear or not isinstance(c2, ir.Num) or not ir.is_zero(rest):
        return None
    if c1.value + c2.value == 0:
        return 1
    if c1.value - c2.value == 0:
        return -1
    return None


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

    graph = state.graph
    priorities = state.state_priorities
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
        candidates.sort(key=lambda v: state.canonical_ranks[v])
    return candidates[0]


def _find_perfect_aliases(
    state: StructuralState,
    eqs_to_rm: List[int],
    vars_to_rm: List[int],
    **kwargs,
) -> None:
    """Identify and rewrite perfect alias equations.

    Appends removable equations/variables to the given buffers.
    Solvable edges of every rewritten equation are recomputed with
    :meth:`StructuralState.find_eq_solvables` under ``kwargs``.
    """

    graph = state.graph
    fullvars = state.fullvars
    eqs = state.eqs
    original_eqs = state.original_eqs

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
        if state.primal_of(snbors[0]) is not None:
            continue
        if state.primal_of(snbors[1]) is not None:
            continue
        edge_sign = _alias_sign(
            eqs[ieq][1], fullvars[snbors[0]], fullvars[snbors[1]]
        )
        if edge_sign is None:
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
                state.additional_observed.append((fullvars[v], zero))
                for e in list(graph.d_neighbors(v)):
                    eqs_to_substitute.append(e)
                graph.invview().set_neighbors(v, ())
                dv = state.derivative_of(v)
                while dv is not None:
                    vars_to_rm.append(dv)
                    subs[fullvars[dv]] = zero
                    for e in list(graph.d_neighbors(dv)):
                        eqs_to_substitute.append(e)
                    graph.invview().set_neighbors(dv, ())
                    dv = state.derivative_of(dv)
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
            state.additional_observed.append((fullvars[v], rhs_sym))

            for e in list(graph.d_neighbors(v)):
                eqs_to_substitute.append(e)
                graph.rem_edge(e, v)
                graph.add_edge(e, target)

            dv = state.derivative_of(v)
            # One differentiation level below dtarget.
            prev_dtarget = target
            dtarget = state.derivative_of(target)
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
                for e in list(graph.d_neighbors(dv)):
                    eqs_to_substitute.append(e)
                    graph.rem_edge(e, dv)
                    graph.add_edge(e, dtarget)
                dv = state.derivative_of(dv)
                prev_dtarget = dtarget
                dtarget = state.derivative_of(dtarget)

    # Per-equation cleanup of the candidate alias equations.
    for ieq, v1, v2, _sign in candidate_eqs:
        if parity.get(v1, 1) == 0:
            irrs = irrs_by_root[parent[v1]]
            if not irrs:
                eqs_to_rm.append(ieq)
            else:
                v_pin = irrs.pop()
                graph.set_neighbors(ieq, [v_pin])
                eqs[ieq] = (fullvars[v_pin], zero)
                original_eqs[ieq] = (fullvars[v_pin], zero)
                eqs_to_substitute.append(ieq)
        else:
            target = group_target[parent[v1]]
            c1 = v1 if is_irreducible_v(v1) else target
            c2 = v2 if is_irreducible_v(v2) else target
            if c1 == c2:
                eqs_to_rm.append(ieq)

    for e in dict.fromkeys(eqs_to_substitute):
        for equations in (eqs, original_eqs):
            lhs, rhs = equations[e]
            equations[e] = (ir.xreplace(lhs, subs), ir.xreplace(rhs, subs))
        # Substitution can cancel the target or annihilate cofactors.
        graph.set_neighbors(e, state.incidence(eqs[e]))
        # Substitution can make a variable enter nonlinearly.
        if state.solvable_graph is not None:
            state.find_eq_solvables(e, **kwargs)

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


def eliminate_perfect_aliases(
    state: StructuralState, **kwargs
) -> Tuple[List[int], List[int]]:
    """Remove perfect alias equations from ``state``.

    ``kwargs`` are the solvability options of
    :meth:`StructuralState.find_eq_solvables`.

    Returns ``(old_to_new_eq, old_to_new_var)``.
    """

    eqs_to_rm = []
    vars_to_rm = []
    _find_perfect_aliases(state, eqs_to_rm, vars_to_rm, **kwargs)
    return state.rm_eqs_vars(eqs_to_rm, vars_to_rm)


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
    graph = state.graph
    while True:
        added_equation = False
        for i, (lhs, rhs) in enumerate(state.original_eqs):
            if i in trivial_idxs or i in blacklist:
                continue
            vari = var_to_idx.get(lhs)
            if vari is None:
                continue
            if lhs in state.irreducibles:
                blacklist.add(i)
                continue
            # A var ~ var equation is stored as 0 ~ 0.
            sys_lhs, sys_rhs = sys_eqs[i]
            if ir.is_zero(sys_lhs) and ir.is_zero(sys_rhs):
                continue
            if not state.is_algebraic(vari):
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
            if lhs in ir.free_atoms(rhs):
                blacklist.add(i)
                continue

            added_equation = True
            trivial_idxs.add(eqi)
            torn_eqs.append((lhs, rhs))
            matched_vars.add(vari)

        if not added_equation:
            break

    state.rm_eqs_vars(sorted(trivial_idxs), sorted(matched_vars))
    state.additional_observed.extend(torn_eqs)


def alias_elimination(
    state: StructuralState, **kwargs
) -> SparseMatrixCLIL:
    """Integer-linear alias elimination pass.

    Runs singularity removal, rewrites the reduced rows back into the
    symbolic equations, removes equations that reduced to ``0 ~ 0``,
    and returns the updated integer subsystem matrix.
    """

    eqs_to_rm = []

    mm = structural_singularity_removal(state, **kwargs)

    fullvars_to_idx = state.var2idx
    eqs = state.eqs
    original_eqs = state.original_eqs

    for ieq, eq in enumerate(mm.nzrows):
        rcol = mm.row_cols[ieq]
        rval = mm.row_vals[ieq]
        if not rcol:
            eqs_to_rm.append(eq)
            continue
        state.rewrite_from_row(eq, rcol, rval, **kwargs)
        rhs = eqs[eq][1]
        lhs = original_eqs[eq][0]
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
            original_eqs[eq] = (lhs, rhs + lhs)
        else:
            original_eqs[eq] = eqs[eq]

    old_to_new_eq, old_to_new_var = state.rm_eqs_vars(eqs_to_rm, [])
    return get_new_mm(old_to_new_eq, old_to_new_var, mm)
