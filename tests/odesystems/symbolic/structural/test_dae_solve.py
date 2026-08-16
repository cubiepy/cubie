"""Numerical solve of a structurally simplified torn DAE.

Integrates ``dx/dt = -z`` with the implicit constraint
``z**5 + z = x`` (not explicitly solvable, so ``z`` is a torn
algebraic state under a singular mass matrix) and compares against a
high-accuracy reference computed on the reduced ODE with a per-step
Newton solve for ``z``.
"""

import numpy as np
import pytest

from cubie import Solver, solve_ivp
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


def test_torn_system_rejects_explicit_algorithm(torn_dae_system):
    # The singular mass matrix cannot be consumed by an explicit
    # step; silently ignoring it would integrate the constraint
    # residuals as derivatives.
    with pytest.raises(ValueError, match="implicit algorithm"):
        Solver(torn_dae_system, algorithm="euler")


def test_implicit_runs_leave_plain_system_massless():
    # The mass matrix belongs to the system; building implicit
    # solvers (whose helpers read it) must not mark a plain system
    # as a DAE, and explicit solvers must still build afterwards.
    ode = create_ODE_system(
        dxdt="dx = -x",
        states={"x": 1.0},
        precision=np.float64,
        name="dae_guard_explicit",
    )
    implicit = Solver(ode, algorithm="backwards_euler")
    implicit.update({"algorithm": "crank_nicolson"})
    assert ode.mass is None
    assert ode.compile_settings.mass is None
    Solver(ode, algorithm="euler")


def test_hot_swap_to_explicit_rejected(torn_dae_system):
    # Swapping algorithms after construction re-runs the mass
    # guard: a torn system cannot swap to an explicit step.
    solver = Solver(torn_dae_system, algorithm="backwards_euler")
    with pytest.raises(ValueError, match="implicit algorithm"):
        solver.update({"algorithm": "euler"})


def test_mass_is_not_an_algorithm_setting(torn_dae_system):
    # The mass matrix is part of the system definition; 'M' is
    # rejected as an algorithm setting on any system.
    with pytest.raises(ValueError, match="not an algorithm setting"):
        Solver(
            torn_dae_system,
            algorithm="backwards_euler",
            algorithm_settings={"M": np.eye(2)},
        )


def test_user_mass_cannot_override_structural():
    # Structural simplification derives the mass matrix; a user
    # supplied matrix is rejected at system construction.
    with pytest.raises(ValueError, match="cannot override"):
        create_ODE_system(
            dxdt="""
            dx = -z
            0 = z**5 + z - x
            """,
            states={"x": 2.0, "z": 1.0},
            precision=np.float64,
            simplify=True,
            mass=np.eye(2),
            name="dae_guard_user_mass",
        )


def test_hand_formulated_mass_requires_implicit():
    # A user-supplied singular mass matrix at system construction
    # behaves like a structural one: implicit algorithms build,
    # explicit algorithms are rejected.
    ode = create_ODE_system(
        dxdt="""
        dx = -z
        dz = z**5 + z - x
        """,
        states={"x": 2.0, "z": 1.0},
        precision=np.float64,
        mass=np.diag([1.0, 0.0]),
        name="dae_guard_hand_mass",
    )
    assert ode.mass is not None
    Solver(ode, algorithm="backwards_euler")
    with pytest.raises(ValueError, match="implicit algorithm"):
        Solver(ode, algorithm="euler")


def test_singular_mass_defaults_solver_stack(torn_dae_system):
    # Mass-matrix systems default to jacobi + bicgstab + scaled cap.
    solver = Solver(torn_dae_system, algorithm="backwards_euler")
    step = solver.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "jacobi"
    assert step.linear_correction_type == "bicgstab"
    width = int(step.compile_settings.solver_width)
    cap = step.solver.linear_solver.compile_settings.max_iters
    assert cap == max(50, 4 * width)


def test_singular_mass_radau_cap_scales_with_width(
    ring_modulator_index2_system,
):
    solver = Solver(ring_modulator_index2_system, algorithm="radau_iia_5")
    step = solver.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "jacobi"
    assert step.linear_correction_type == "bicgstab"
    width = int(step.compile_settings.solver_width)
    assert width == 3 * 14
    cap = step.solver.linear_solver.compile_settings.max_iters
    assert cap == 4 * width


def test_singular_mass_explicit_stack_preserved(torn_dae_system):
    # User-chosen stack settings survive construction and hot-swap.
    solver = Solver(
        torn_dae_system,
        algorithm="backwards_euler",
        preconditioner_type="neumann",
        linear_correction_type="minimal_residual",
        krylov_max_iters=37,
    )
    step = solver.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "neumann"
    assert step.linear_correction_type == "minimal_residual"
    assert step.solver.linear_solver.compile_settings.max_iters == 37

    solver.update({"algorithm": "radau_iia_5"})
    step = solver.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "neumann"
    assert step.linear_correction_type == "minimal_residual"
    assert step.solver.linear_solver.compile_settings.max_iters == 37


def test_plain_system_keeps_default_stack():
    # Massless systems keep the neumann + minimal_residual defaults.
    ode = create_ODE_system(
        dxdt="dx = -x",
        states={"x": 1.0},
        precision=np.float64,
        name="dae_guard_default_stack",
    )
    solver = Solver(ode, algorithm="backwards_euler")
    step = solver.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "neumann"
    assert step.linear_correction_type == "minimal_residual"
    assert step.solver.linear_solver.compile_settings.max_iters == 50


def _ring_constraint_residuals(values):
    """Evaluate the four diode constraints from named final values."""

    gamma = 40.67286402e-9
    delta = 17.7493332
    ud4 = -(values["UD1"] + values["UD2"] + values["UD3"])
    charges = [
        gamma * (np.exp(delta * ud) - 1.0)
        for ud in (values["UD1"], values["UD2"], values["UD3"], ud4)
    ]
    i3 = -(values["I4"] + values["I5"] + values["I6"])
    return (
        i3 - charges[0] + charges[3],
        -values["I4"] + charges[1] - charges[2],
        values["I5"] + charges[0] - charges[2],
        -values["I6"] - charges[1] + charges[3],
    )


def _solve_ring(system, method):
    y0 = {
        str(sym): np.array([0.0])
        for sym in system.indices.states.index_map
    }
    result = solve_ivp(
        system,
        y0=y0,
        method=method,
        duration=2e-6,
        dt=1e-7,
        save_every=1e-6,
    )
    legend = {
        label: idx for idx, label in result.time_domain_legend.items()
    }
    trajectory = result.time_domain_array
    assert np.isfinite(trajectory).all()
    finals = {
        name: float(trajectory[-1, legend[name], 0])
        for name in ("I4", "I5", "I6", "UD1", "UD2", "UD3")
    }
    # A flat trajectory would satisfy the constraints trivially.
    assert max(abs(finals[k]) for k in ("UD1", "UD2", "UD3")) > 0.1
    for residual in _ring_constraint_residuals(finals):
        assert residual == pytest.approx(0.0, abs=1e-5)


@pytest.mark.slow
def test_ring_modulator_index2_backwards_euler(
    ring_modulator_index2_system,
):
    _solve_ring(ring_modulator_index2_system, "backwards_euler")


@pytest.mark.nocudasim
@pytest.mark.slow
def test_ring_modulator_index2_radau(ring_modulator_index2_system):
    _solve_ring(ring_modulator_index2_system, "radau_iia_5")


def z_of_x(x):
    """Solve z**5 + z = x by Newton iteration."""

    z = x / 2.0
    for _ in range(60):
        f = z**5 + z - x
        z = z - f / (5.0 * z**4 + 1.0)
    return z


def reference_solution(x0, t_end, n_steps):
    """RK4 on the reduced ODE dx/dt = -z(x)."""

    dt = t_end / n_steps
    x = x0
    for _ in range(n_steps):
        k1 = -z_of_x(x)
        k2 = -z_of_x(x + 0.5 * dt * k1)
        k3 = -z_of_x(x + 0.5 * dt * k2)
        k4 = -z_of_x(x + dt * k3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return x


@pytest.mark.slow
def test_torn_dae_solution_matches_reference(torn_dae_system):
    t_end = 0.2
    result = solve_ivp(
        torn_dae_system,
        y0={"x": np.array([2.0]), "z": np.array([1.0])},
        method="backwards_euler",
        duration=t_end,
        dt=1e-3,
        save_every=0.1,
        newton_atol=1e-10,
        newton_rtol=1e-10,
        # The torn system's algebraic row leaves the operator
        # ill-scaled, so its attainable weighted residual sits above
        # the derived sqrt(eps) floor in float64. A floor of 1e-6
        # weighted (1e-12 absolute at the default weights) keeps the
        # linear error two decades below the Newton target.
        krylov_residual_floor=1e-6,
    )
    legend = {
        label: idx
        for idx, label in result.time_domain_legend.items()
    }
    trajectory = result.time_domain_array
    x_final = float(trajectory[-1, legend["x"], 0])
    z_final = float(trajectory[-1, legend["z"], 0])

    x_ref = reference_solution(2.0, t_end, 4000)
    z_ref = z_of_x(x_ref)

    # Backward Euler global error is O(dt); dt=1e-3 over t=0.2 with
    # |dz/dt| < 1 keeps it well inside 2e-3.
    assert x_final == pytest.approx(x_ref, abs=2e-3)
    # The algebraic constraint must hold at the solution.
    assert z_final**5 + z_final - x_final == pytest.approx(
        0.0, abs=1e-5
    )
    assert z_final == pytest.approx(z_ref, abs=2e-3)
