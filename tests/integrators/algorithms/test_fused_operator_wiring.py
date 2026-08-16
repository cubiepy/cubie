"""fuse_operator_preconditioner reaches the linear solver and a
fused solve agrees with the unfused solve.
"""

import numpy as np
import pytest

from cubie import Solver
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


@pytest.fixture(scope="session")
def fused_wiring_system(precision):
    """Nonlinear three-state system for fused wiring checks."""
    dxdt = [
        "aux = x0 * x1 + p0",
        "dx0 = -p0 * x0 + x1 * x2 + aux",
        "dx1 = p1 * (x0 - x1) - aux",
        "dx2 = x0 * x1 - c0 * x2",
    ]
    return create_ODE_system(
        dxdt,
        states={"x0": 1.0, "x1": 0.5, "x2": 2.0},
        parameters={"p0": 0.3, "p1": 1.7},
        constants={"c0": 2.66},
        name="fused_wiring_system",
        precision=precision,
    )


def _solve_states(system, fused, algorithm):
    solver = Solver(
        system,
        algorithm=algorithm,
        preconditioner_type="jacobi",
        fuse_operator_preconditioner=fused,
        atol=1e-6,
        rtol=1e-6,
        output_types=["state"],
    )
    inits, params = solver.build_grid(
        {"x0": np.array([1.0, 1.1], dtype=system.precision)}, None
    )
    result = solver.solve(inits, params, duration=0.05)
    return (
        solver,
        np.asarray(result.state),
        np.asarray(result.status_codes),
    )


@pytest.mark.parametrize("algorithm", ["radau", "dirk"])
def test_fused_flag_reaches_linear_solver(
    fused_wiring_system, algorithm
):
    """The flag injects (and clears) the fused device function."""
    solver_on, _, _ = _solve_states(
        fused_wiring_system, True, algorithm
    )
    algo_step = (
        solver_on.kernel.single_integrator._algo_step
    )
    linear = algo_step.linear_solver
    assert (
        linear.compile_settings.fused_operator_apply is not None
    )

    solver_off, _, _ = _solve_states(
        fused_wiring_system, False, algorithm
    )
    algo_step = (
        solver_off.kernel.single_integrator._algo_step
    )
    linear = algo_step.linear_solver
    assert linear.compile_settings.fused_operator_apply is None


@pytest.mark.parametrize("algorithm", ["radau", "dirk"])
def test_fused_solve_matches_unfused(
    fused_wiring_system, precision, algorithm
):
    """Fused and unfused solves agree on the same problem."""
    _, state_off, codes_off = _solve_states(
        fused_wiring_system, False, algorithm
    )
    _, state_on, codes_on = _solve_states(
        fused_wiring_system, True, algorithm
    )
    assert np.all(codes_off == 0)
    assert np.all(codes_on == 0)
    if precision == np.float64:
        tolerances = {"rtol": 1e-9, "atol": 1e-10}
    else:
        tolerances = {"rtol": 5e-4, "atol": 5e-5}
    np.testing.assert_allclose(state_on, state_off, **tolerances)
