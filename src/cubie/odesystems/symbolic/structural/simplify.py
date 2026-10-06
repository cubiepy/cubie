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

Published Classes
-----------------
:class:`SimplifiedSystem`
    The simplification result consumed by cubie's parser/codegen.

Published Functions
-------------------
:func:`structural_simplify`
    Run the full pipeline on a
    :class:`~cubie.odesystems.symbolic.structural.system_structure.StructuralState`.
"""

import warnings
from typing import Dict, List, Optional, Tuple

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.assignments import (
    topological_sort,
)
from cubie.odesystems.symbolic.structural.alias_elimination import (
    alias_elimination,
    eliminate_perfect_aliases,
    trivial_tearing,
)
from cubie.odesystems.symbolic.structural.bipartite import (
    Matching,
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
from cubie.odesystems.symbolic.structural.pantelides import pantelides
from cubie.odesystems.symbolic.structural.reassemble import (
    ReassembledSystem,
    default_reassemble,
)
from cubie.odesystems.symbolic.structural.singularity_removal import (
    get_new_mm,
)
from cubie.odesystems.symbolic.structural.system_structure import (
    MAX_INTEGER_COEFFICIENT,
    StructuralState,
)


class SimplifiedSystem:
    """Result of structural simplification.

    Parameters
    ----------
    states
        Final solver unknowns in BLT order: differential states
        first, then torn algebraic variables.
    differential_states
        The subset of ``states`` integrated through their solved
        derivatives.
    algebraic_states
        The torn (iteration) variables constrained by ``residuals``.
    dxdt
        Map from each differential state to its explicit derivative
        expression.
    residuals
        Algebraic residual expressions (each constrained to zero).
        Empty for fully torn systems.
    observed
        Topologically sorted ``(symbol, expression)`` assignments for
        eliminated variables.
    mass_matrix
        ``None`` when there are no residuals; otherwise the singular
        diagonal mass matrix, as a nested list of floats (identity
        for differential states, zero rows for algebraic
        constraints).
    """

    def __init__(
        self,
        states: List[ir.Sym],
        differential_states: List[ir.Sym],
        algebraic_states: List[ir.Sym],
        dxdt: Dict[ir.Sym, ir.Expr],
        residuals: List[ir.Expr],
        observed: List[Tuple[ir.Sym, ir.Expr]],
        mass_matrix: Optional[List[List[float]]],
    ) -> None:
        self.states = states
        self.differential_states = differential_states
        self.algebraic_states = algebraic_states
        self.dxdt = dxdt
        self.residuals = residuals
        self.observed = observed
        self.mass_matrix = mass_matrix


def _integer_jacobian(state: StructuralState):
    """Integer Jacobian closure for dummy-derivative rank checks."""

    def jac(
        eq_idxs: List[int], var_idxs: List[int]
    ) -> Optional[List[List[int]]]:
        rows = []
        for e in eq_idxs:
            rhs = state.eqs[e].rhs
            row = []
            for v in var_idxs:
                entry = ir.diff(rhs, state.fullvars[v])
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
        state.fullvars[i]: state.structure.state_priorities[i]
        for i in range(len(state.fullvars))
    }
    return StructuralState(
        new_eqs,
        state.registry,
        state.known_symbols - {state.time_symbol},
        state.time_symbol,
        state_priorities=priorities,
        irreducibles=state.irreducibles,
    )


def _assemble_result(
    reassembled: ReassembledSystem,
) -> SimplifiedSystem:
    """Convert a reassembled system into the cubie-facing result."""

    dxdt = {}
    differential_states = []
    residuals = []
    for eq, diff_state in zip(
        reassembled.neweqs, reassembled.diff_eq_states
    ):
        if diff_state is None:
            residuals.append(eq.rhs)
            continue
        differential_states.append(diff_state)
        dxdt[diff_state] = eq.rhs

    diff_set = set(differential_states)
    algebraic_states = [
        s for s in reassembled.unknowns if s not in diff_set
    ]
    states = differential_states + algebraic_states

    if len(residuals) != len(algebraic_states):
        # Balanced systems always pair up; unbalanced systems (run
        # with fully_determined=False) may not.
        warnings.warn(
            f"{len(residuals)} residual equations for "
            f"{len(algebraic_states)} algebraic states; the system "
            "is not fully determined"
        )

    n = len(states)
    if residuals:
        mass = [[0.0] * n for _ in range(n)]
        for i in range(len(differential_states)):
            mass[i][i] = 1.0
    else:
        mass = None

    observed = _topsort_observed(
        [(eq.lhs, eq.rhs) for eq in reassembled.observed]
    )

    return SimplifiedSystem(
        states,
        differential_states,
        algebraic_states,
        dxdt,
        residuals,
        observed,
        mass,
    )


def _topsort_observed(
    observed: List[Tuple[ir.Expr, ir.Expr]],
) -> List[Tuple[ir.Sym, ir.Expr]]:
    """Topologically sort observed assignments by dependency."""

    return topological_sort(list(observed))


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
        when false (tearing only).
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

    reassemble_kwargs = {
        "fully_determined": fully_determined,
    }

    if fully_determined and dummy_derivative:
        tearing_result = dummy_derivative_graph(
            state,
            _integer_jacobian(state),
            state_priority=lambda v: (
                state.structure.state_priorities[v]
            ),
            **solve_kwargs,
        )
        reassembled = default_reassemble(
            state, tearing_result, state.mm, **reassemble_kwargs
        )
    elif fully_determined:
        # Bare index reduction, then re-analyse and select states.
        state.structure.complete()
        # Alias elimination rewrites integer-linear differential
        # equations to 0 ~ f, so they cannot be told apart from
        # their derivatives; only highest-order matches are kept.
        var_eq_matching = pantelides(state, **solve_kwargs)
        state = _pantelides_reassemble_state(state, var_eq_matching)
        mm = alias_elimination(state, **solve_kwargs)
        state.mm = mm
        tearing_result = dummy_derivative_graph(
            state,
            _integer_jacobian(state),
            state_priority=lambda v: (
                state.structure.state_priorities[v]
            ),
            **solve_kwargs,
        )
        reassembled = default_reassemble(
            state, tearing_result, state.mm, **reassemble_kwargs
        )
    else:
        state.structure.complete()
        tearing_result = _tear_with_dummies(
            state, set(), **solve_kwargs
        )
        reassembled = default_reassemble(
            state, tearing_result, state.mm, **reassemble_kwargs
        )

    return _assemble_result(reassembled)
