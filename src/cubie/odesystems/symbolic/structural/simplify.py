"""Structural simplification pipeline driver.

Singular derivative-block removal, perfect-alias elimination, trivial
tearing, integer-linear alias elimination (singularity removal),
consistency checking, Pantelides index reduction with
dummy-derivative state selection, tearing, and reassembly into an
explicit (or semi-explicit mass-matrix) system.

``structural_simplify`` follows the continuous-system pipeline of
ModelingToolkit.jl (commit a2b6dc56, ``src/systems/systemstructure.jl``,
``_mtkcompile!`` and ``_mtkcompile_worker!``);
``_pantelides_reassemble_state`` is ported from the same commit
(``src/structural_transformation/pantelides.jl``,
``pantelides_reassemble``) and ``_integer_jacobian`` from the integer
Jacobian of ``dummy_derivative`` in the same commit
(``src/structural_transformation/symbolics_tearing.jl``).

Published Functions
-------------------
:func:`structural_simplify`
    Run the full pipeline on a
    :class:`~cubie.odesystems.symbolic.structural.system_structure.StructuralState`.
"""

import warnings
from typing import Dict, List, Optional

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.structural.alias_elimination import (
    alias_elimination,
    eliminate_perfect_aliases,
    trivial_tearing,
)
from cubie.odesystems.symbolic.structural.bipartite import (
    BipartiteGraph,
    Matching,
    maximal_matching,
)
from cubie.odesystems.symbolic.structural.consistency import (
    check_consistency,
)
from cubie.odesystems.symbolic.structural.derivative_block import (
    eliminate_singular_derivative_blocks,
)
from cubie.odesystems.symbolic.structural.dummy_derivatives import (
    _tear_with_dummies,
    dummy_derivative_graph,
)
from cubie.odesystems.symbolic.structural.errors import raise_unmatched
from cubie.odesystems.symbolic.structural.pantelides import pantelides
from cubie.odesystems.symbolic.structural.reassemble import (
    SimplifiedSystem,
    default_reassemble,
)
from cubie.odesystems.symbolic.structural.singularity_removal import (
    get_new_mm,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    MAX_INTEGER_COEFFICIENT,
    StructuralState,
)


def _integer_jacobian(state: StructuralState):
    """Integer Jacobian closure for dummy-derivative rank checks."""

    def jac(
        eq_idxs: List[int], var_idxs: List[int]
    ) -> Optional[List[List[int]]]:
        rows = []
        for e in eq_idxs:
            rhs = state.eqs[e][1]
            row = []
            for v in var_idxs:
                entry = ir.diff(
                    rhs,
                    state.fullvars[v],
                    derivative_names=state.derivative_names,
                )
                value = ir.int_value(entry)
                if value is None or abs(value) > MAX_INTEGER_COEFFICIENT:
                    return None
                row.append(value)
            rows.append(row)
        return rows

    return jac


def _pantelides_reassemble_state(
    state: StructuralState, var_eq_matching: Matching
) -> StructuralState:
    """Rebuild a first-analysis state after bare index reduction.

    Keeps, for each matched highest-differentiated variable, the
    (differentiated) equation it is matched to, and rebuilds a fresh
    structural state from those equations.
    """

    matched_eqs = sorted(
        {
            e
            for e in var_eq_matching
            if isinstance(e, int)
        }
    )
    new_eqs = [state.eqs[e] for e in matched_eqs]
    priorities = {
        state.fullvars[i]: state.state_priorities[i]
        for i in range(len(state.fullvars))
    }
    return StructuralState(
        new_eqs,
        state.registry,
        state.known_symbols - {state.time_symbol},
        state.time_symbol,
        derivative_names=state.derivative_names,
        state_priorities=priorities,
        irreducibles=state.irreducibles,
    )


def _check_algebraic_block(result: SimplifiedSystem) -> None:
    """Require a residual row determining each algebraic state.

    The solvers take one residual row per algebraic state under a
    singular mass matrix, so the residuals must match the algebraic
    states one to one through the states each residual reaches,
    directly or through the observed assignments.

    Raises
    ------
    ExtraEquationsSystemError
        When residuals are left that determine no algebraic state.
    ExtraVariablesSystemError
        When algebraic states are left with no residual.
    InvalidSystemError
        When both are left.
    """

    residuals = result.residuals
    algebraic_states = result.algebraic_states
    state_index = {s: i for i, s in enumerate(algebraic_states)}
    reached = {}
    for lhs, rhs in result.observed:
        reached[lhs] = _reached_states(rhs, state_index, reached)
    graph = BipartiteGraph(len(residuals), len(algebraic_states))
    for i, residual in enumerate(residuals):
        graph.set_neighbors(
            i, _reached_states(residual, state_index, reached)
        )
    matching = maximal_matching(graph)
    matched_rows = {r for r in matching if isinstance(r, int)}
    extra_rows = [
        f"0 ~ {residual}"
        for i, residual in enumerate(residuals)
        if i not in matched_rows
    ]
    extra_states = [
        str(s)
        for i, s in enumerate(algebraic_states)
        if not isinstance(matching[i], int)
    ]
    counts = []
    if extra_rows:
        counts.append(
            f"{len(extra_rows)} residual equations determine no "
            "algebraic state"
        )
    if extra_states:
        counts.append(
            f"{len(extra_states)} algebraic states have no residual "
            "equation"
        )
    raise_unmatched(
        "The algebraic equations do not pair one to one: "
        + " and ".join(counts)
        + ".",
        extra_rows,
        extra_states,
    )


def _reached_states(
    expr: ir.Expr,
    state_index: Dict[ir.Sym, int],
    reached: Dict[ir.Sym, List[int]],
) -> List[int]:
    """Algebraic-state indices ``expr`` reads, through ``reached``."""

    indices = set()
    for atom in ir.free_atoms(expr):
        if atom in state_index:
            indices.add(state_index[atom])
        else:
            indices.update(reached.get(atom, ()))
    return sorted(indices)


def structural_simplify(
    state: StructuralState,
    fully_determined: bool = True,
    dummy_derivative: bool = True,
    consistency_check: bool = True,
    conservative: bool = False,
    allow_symbolic: bool = False,
    allow_parameter: bool = True,
) -> SimplifiedSystem:
    """Run the full structural simplification pipeline.

    Parameters
    ----------
    state
        The structural state to simplify (mutated).
    fully_determined
        Whether the system must have matching equation and unknown
        counts; disables the consistency check and index reduction
        when false (tearing only). The simplified result must still
        pair each algebraic state with a residual that determines
        it; ``InvalidSystemError`` (or its extra-equations or
        extra-variables subclass) names what is left unpaired.
    dummy_derivative
        Use dummy-derivative state selection (the default MTK path).
        When false, bare Pantelides index reduction runs first and
        the resulting system is re-analysed.
    consistency_check
        Verify balance/nonsingularity before state selection.
    conservative
        Restrict tearing to coefficients with absolute value one.
    allow_symbolic, allow_parameter
        Solvability limits on symbolic pivots (division safety).
    """

    solve_kwargs = {
        "allow_symbolic": allow_symbolic,
        "allow_parameter": allow_parameter,
        "conservative": conservative,
    }

    eliminate_singular_derivative_blocks(state, **solve_kwargs)
    # Two-phase alias elimination (MTK pattern): the first call
    # clears obvious aliases before the integer-linear pass and its
    # return maps are not needed; the second call catches aliases
    # newly exposed by alias_elimination, and its maps rebase mm.
    eliminate_perfect_aliases(state, **solve_kwargs)
    trivial_tearing(state)
    mm = alias_elimination(state, **solve_kwargs)
    old_to_new_eq, old_to_new_var = eliminate_perfect_aliases(
        state, **solve_kwargs
    )
    mm = get_new_mm(old_to_new_eq, old_to_new_var, mm)
    state.mm = mm

    if consistency_check and fully_determined:
        check_consistency(state)

    if fully_determined and not dummy_derivative:
        # Alias elimination rewrites integer-linear differential
        # equations to 0 ~ f, so they cannot be told apart from
        # their derivatives; only highest-order matches are kept.
        var_eq_matching = pantelides(state, **solve_kwargs)
        state = _pantelides_reassemble_state(state, var_eq_matching)
        state.mm = alias_elimination(state, **solve_kwargs)

    if fully_determined:
        tearing_result = dummy_derivative_graph(
            state,
            _integer_jacobian(state),
            state_priority=lambda v: state.state_priorities[v],
            **solve_kwargs,
        )
    else:
        tearing_result = _tear_with_dummies(state, set(), **solve_kwargs)
    result = default_reassemble(state, tearing_result, fully_determined)

    if not fully_determined:
        # No balance check ran, so the result can be unbalanced.
        _check_algebraic_block(result)
    elif len(result.residuals) != len(result.algebraic_states):
        # A singular tearing solve leaves its variable unsolved.
        warnings.warn(
            f"{len(result.residuals)} residual equations for "
            f"{len(result.algebraic_states)} algebraic states; the "
            "system is not fully determined"
        )
    return result
