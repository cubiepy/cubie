"""Numerical solve of a structurally simplified torn DAE.

Integrates ``dx/dt = -z`` with the implicit constraint
``z**5 + z = x`` (not explicitly solvable, so ``z`` is a torn
algebraic state under a singular mass matrix) and compares against a
high-accuracy reference computed on the reduced ODE with a per-step
Newton solve for ``z``.
"""

import warnings

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


# None overrides unset the spine's explicit linear solve values.
UNSET_LINEAR_SOLVE = {
    "linear_correction_type": None,
    "krylov_max_iters": None,
    "preconditioner_type": None,
}

# torn_driver has no observables to save.
NO_OBSERVABLES = {
    "output_types": ["state", "time"],
    "saved_observable_indices": [],
    "summarised_observable_indices": [],
}

TORN_SYSTEM_DEFAULTS = {
    "system_type": "torn_driver",
    "precision": np.float64,
    "algorithm": "backwards_euler",
    **NO_OBSERVABLES,
    **UNSET_LINEAR_SOLVE,
}

MASSLESS_DEFAULTS = {
    "algorithm": "backwards_euler",
    **UNSET_LINEAR_SOLVE,
}

TORN_SYSTEM_EXPLICIT = {
    "system_type": "torn_driver",
    "precision": np.float64,
    "algorithm": "backwards_euler",
    "preconditioner_type": "jacobi",
    "linear_correction_type": "minimal_residual",
    "krylov_max_iters": 37,
    **NO_OBSERVABLES,
}

# Coupling terms reach 1e8, so every series term amplifies here.
RING_SOLVE_COMMON = {
    "system_type": "ring_modulator_index2",
    "precision": np.float64,
    "step_controller": "fixed",
    "dt": 1e-7,
    "save_every": 1e-6,
    "output_types": ["state", "time"],
    "saved_state_indices": list(range(14)),
    "preconditioner_order": 0,
    **UNSET_LINEAR_SOLVE,
}

RING_BACKWARDS_EULER = {
    **RING_SOLVE_COMMON,
    "algorithm": "backwards_euler",
}

RING_RADAU = {**RING_SOLVE_COMMON, "algorithm": "radau_iia_5"}


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_SYSTEM_DEFAULTS], indirect=True
)
def test_singular_mass_defaults_linear_solve_params(solver):
    # Two-state backwards Euler sits under the 50-iteration floor.
    step = solver.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "jacobi"
    assert step.linear_correction_type == "bicgstab"
    width = int(step.compile_settings.solver_width)
    assert 4 * width < 50
    cap = step.solver.linear_solver.compile_settings.max_iters
    assert cap == 50


@pytest.mark.parametrize(
    "solver_settings_override", [RING_RADAU], indirect=True
)
def test_singular_mass_cap_scales_with_width(solver, system):
    # Three-stage radau cap is four times the stacked width.
    step = solver.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "jacobi"
    assert step.linear_correction_type == "bicgstab"
    width = int(step.compile_settings.solver_width)
    assert width == 3 * system.sizes.states
    cap = step.solver.linear_solver.compile_settings.max_iters
    assert cap == 4 * width


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_SYSTEM_EXPLICIT], indirect=True
)
def test_singular_mass_explicit_params_preserved(solver_mutable):
    # User-set linear solve params survive hot-swap.
    step = solver_mutable.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "jacobi"
    assert step.linear_correction_type == "minimal_residual"
    assert step.solver.linear_solver.compile_settings.max_iters == 37

    solver_mutable.update({"algorithm": "radau_iia_5"})
    step = solver_mutable.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "jacobi"
    assert step.linear_correction_type == "minimal_residual"
    assert step.solver.linear_solver.compile_settings.max_iters == 37


def test_singular_mass_defaults_to_the_diagonal_solve(torn_dae_system):
    # Unset order re-resolves after the DAE path swaps the type.
    solver = Solver(torn_dae_system, algorithm="backwards_euler")
    step = solver.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "jacobi"
    assert step.preconditioner_order == 0


def test_neumann_rejected_on_torn_system(torn_dae_system):
    # Explicit neumann on a torn system is rejected at construction.
    with pytest.raises(ValueError, match="identity mass"):
        Solver(
            torn_dae_system,
            algorithm="backwards_euler",
            preconditioner_type="neumann",
        )


@pytest.mark.parametrize(
    "solver_settings_override", [RING_BACKWARDS_EULER], indirect=True
)
def test_singular_mass_default_cap_rederived_on_swap(
    solver_mutable, system
):
    # Defaulted cap re-derives for the new width on hot-swap.
    step = solver_mutable.kernel.single_integrator._algo_step
    n = system.sizes.states
    assert step.solver.linear_solver.compile_settings.max_iters == max(
        50, 4 * n
    )
    solver_mutable.update({"algorithm": "radau_iia_5"})
    step = solver_mutable.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "jacobi"
    assert step.linear_correction_type == "bicgstab"
    cap = step.solver.linear_solver.compile_settings.max_iters
    assert cap == 4 * 3 * n


@pytest.mark.parametrize(
    "solver_settings_override", [MASSLESS_DEFAULTS], indirect=True
)
def test_plain_system_keeps_default_linear_solve(solver):
    # Massless systems keep the algorithm-family defaults.
    step = solver.kernel.single_integrator._algo_step
    assert step.preconditioner_type == "jacobi"
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


def _solve_ring(solver, system):
    inits = np.zeros((system.sizes.states, 1), dtype=np.float64)
    params = np.full((1, 1), 0.5, dtype=np.float64)
    result = solver.solve(inits, params, duration=2e-6)
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


@pytest.mark.parametrize(
    "solver_settings_override", [RING_BACKWARDS_EULER], indirect=True
)
def test_ring_modulator_index2_backwards_euler(solver, system):
    # Simplification leaves index-1; backwards Euler handles it.
    _solve_ring(solver, system)


@pytest.mark.parametrize(
    "solver_settings_override", [RING_RADAU], indirect=True
)
def test_ring_modulator_index2_radau(solver, system):
    _solve_ring(solver, system)


RING_SCALED_BACKWARDS_EULER = {
    **RING_SOLVE_COMMON,
    "system_type": "ring_modulator_index2_scaled",
    "algorithm": "backwards_euler",
}


@pytest.mark.parametrize(
    "solver_settings_override",
    [RING_SCALED_BACKWARDS_EULER],
    indirect=True,
)
def test_scaled_ring_modulator_solves(solver, system):
    """The ``Cs*dU = ...`` form with Cs = 0 integrates correctly."""
    _solve_ring(solver, system)


SCALED_CS_FLIP = {
    "system_type": "scaled_cs",
    "precision": np.float64,
    "algorithm": "backwards_euler",
    "step_controller": "fixed",
    "dt": 1e-3,
    "save_every": 0.05,
    "output_types": ["state", "time"],
    "saved_state_indices": None,
    "saved_observable_indices": None,
    "summarised_state_indices": None,
    "summarised_observable_indices": None,
    "preconditioner_type": "jacobi",
    "linear_correction_type": "bicgstab",
    **UNSET_LINEAR_SOLVE,
}


@pytest.mark.parametrize(
    "solver_settings_override", [SCALED_CS_FLIP], indirect=True
)
def test_structural_flip_solves_after_constant_change(
    solver_mutable, system_restored
):
    """A constant change that restructures the system still solves."""
    system = system_restored
    assert system.mass is not None
    algebraic_names = list(system.initial_values.values_dict)
    y0 = {
        name: np.array([float(value)])
        for name, value in system.initial_values.values_dict.items()
    }
    result = solver_mutable.solve(y0, {}, duration=0.1)
    assert np.isfinite(result.time_domain_array).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver_mutable.update(
            {
                "Cs": 2e-2,
                "algorithm": "euler",
                "dt": 1e-5,
                "save_every": 0.005,
            }
        )
    assert system.mass is None
    explicit_names = list(system.initial_values.values_dict)
    assert explicit_names != algebraic_names
    y0 = {
        name: np.array([float(value)])
        for name, value in system.initial_values.values_dict.items()
    }
    result = solver_mutable.solve(y0, {}, duration=0.01)
    assert np.isfinite(result.time_domain_array).all()


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


# torn_time's shared init settings; the initialiser always runs an
# exact LU solve regardless of the stage solver's correction type.
TORN_INIT_COMMON = {
    "system_type": "torn_time",
    "precision": np.float64,
    "algorithm": "backwards_euler",
    "step_controller": "fixed",
    "dt": 1e-3,
    "save_every": 0.025,
    "newton_atol": 1e-10,
    "newton_rtol": 1e-10,
    # Stage-solver budget; the initialiser keeps its own fixed cap.
    "newton_max_iters": 12,
    **NO_OBSERVABLES,
    **UNSET_LINEAR_SOLVE,
}

TORN_INIT_SHAMPINE = {
    **TORN_INIT_COMMON,
    "dae_initialisation": "shampine",
}
TORN_INIT_NONE = {**TORN_INIT_COMMON, "dae_initialisation": "none"}
UNSOLVABLE_INIT = {
    **TORN_INIT_COMMON,
    "system_type": "torn_unsolvable",
}


def _solve_torn(solver, x0, x1, **solve_kwargs):
    """Solve from (x0, x1) and return the result and t0-saved pair."""
    y0 = {"x0": np.array([x0]), "x1": np.array([x1])}
    result = solver.solve(y0, {}, duration=0.05, **solve_kwargs)
    legend = {
        label: idx for idx, label in result.time_domain_legend.items()
    }
    trajectory = result.time_domain_array
    return result, (
        float(trajectory[0, legend["x0"], 0]),
        float(trajectory[0, legend["x1"], 0]),
    )


def _torn_constraint_residual(x0, x1):
    """Evaluate torn_time's algebraic row c*x0**2 + d*x1 + x1**5."""
    return -0.7 * x0 * x0 + 0.9 * x1 + x1**5


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_COMMON], indirect=True
)
def test_brown_init_corrects_inconsistent_algebraic_start(solver):
    # Brown solves the constraint at t0 and holds x0 exactly.
    result, (x0, x1) = _solve_torn(solver, 2.0, 0.0)
    assert x0 == 2.0
    assert _torn_constraint_residual(x0, x1) == pytest.approx(
        0.0, abs=1e-8
    )
    assert result.status_messages == {}


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_NONE], indirect=True
)
def test_init_none_saves_the_raw_start(solver):
    # 'none' saves the raw inconsistent x1 at t0.
    _, (x0, x1) = _solve_torn(solver, 2.0, 0.9)
    assert x0 == 2.0
    assert x1 == 0.9


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_SHAMPINE], indirect=True
)
def test_shampine_init_lands_on_the_constraint(solver):
    # Shampine moves every component onto the constraint.
    result, (x0, x1) = _solve_torn(solver, 2.0, 0.0)
    assert _torn_constraint_residual(x0, x1) == pytest.approx(
        0.0, abs=1e-8
    )
    # The differential state moves by O(dt * |dx0/dt|) = O(3e-3).
    assert x0 != 2.0
    assert x0 == pytest.approx(2.0, abs=5e-3)
    assert result.status_messages == {}


@pytest.mark.parametrize(
    "solver_settings_override", [UNSOLVABLE_INIT], indirect=True
)
def test_failed_init_ends_run_with_status(solver):
    # No real root of the constraint at x0 = 0: the t0 solve cannot
    # converge, the raw values reach the t0 save, and the run ends.
    result, (x0, x1) = _solve_torn(
        solver, 0.0, 0.5, nan_error_trajectories=False
    )
    assert "DAE_INITIALISATION_FAILED" in result.status_messages[0]
    assert x0 == 0.0
    assert x1 == 0.5


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_COMMON], indirect=True
)
def test_initialiser_defaults_and_decoupled_budget(solver):
    # Torn systems default to a brown initialiser over an exact LU
    # solve; the stage solver's Newton budget stays its own.
    initialiser = solver.kernel.single_integrator._dae_initialiser
    assert not initialiser.is_noop
    assert initialiser.dae_initialisation == "brown"
    assert initialiser.solver.newton_max_iters == 50
    assert (
        initialiser.solver.linear_solver.linear_correction_type
        == "lu"
    )
    step = solver.kernel.single_integrator._algo_step
    assert step.newton_max_iters == 12


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_NONE], indirect=True
)
def test_initialiser_noop_when_disabled(solver):
    initialiser = solver.kernel.single_integrator._dae_initialiser
    assert initialiser.is_noop
    assert initialiser.dae_initialisation == "none"


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_COMMON], indirect=True
)
def test_initialiser_mode_switches_on_update(solver_mutable):
    single = solver_mutable.kernel.single_integrator
    solver_mutable.update({"dae_initialisation": "shampine"})
    assert single._dae_initialiser.dae_initialisation == "shampine"
    solver_mutable.update({"dae_initialisation": "none"})
    assert single._dae_initialiser.is_noop
    solver_mutable.update({"dae_initialisation": "brown"})
    initialiser = single._dae_initialiser
    assert initialiser.dae_initialisation == "brown"
    assert not initialiser.is_noop


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_COMMON], indirect=True
)
def test_invalid_init_mode_rejected(solver_mutable):
    with pytest.raises(ValueError, match="dae_initialisation"):
        solver_mutable.update({"dae_initialisation": "bogus"})


def test_init_mode_warns_on_non_dae_system(system):
    # The default spine system is massless; an explicit mode warns
    # and the initialiser compiles to a no-op.
    with pytest.warns(UserWarning, match="no effect"):
        solver = Solver(
            system,
            algorithm="backwards_euler",
            dae_initialisation="brown",
        )
    assert solver.kernel.single_integrator._dae_initialiser.is_noop


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
