"""Numerical solve of a structurally simplified torn DAE.

Integrates ``dx/dt = -z`` with the implicit constraint
``z**5 + z = x`` (not explicitly solvable, so ``z`` is a torn
algebraic state under a singular mass matrix) and compares against a
high-accuracy reference computed on the reduced ODE with a per-step
Newton solve for ``z``.
"""

import math
import warnings

import numpy as np
import pytest

from cubie import Solver, solve_ivp
from cubie.integrators.algorithms.ode_implicitstep import (
    DAE_SOLVER_DEFAULTS,
    DEFAULT_LINEAR_CORRECTION_TYPE,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from tests.system_fixtures import (
    TRANSAMP_CONSTANTS,
    TRANSAMP_DC_STATES,
)
from tests._utils import (
    TORN_INIT_COMMON,
    TORN_NO_OBSERVABLES,
    UNSET_LINEAR_SOLVE,
)


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


TORN_SYSTEM_DEFAULTS = {
    "system_type": "torn_driver",
    "precision": np.float64,
    "algorithm": "backwards_euler",
    **TORN_NO_OBSERVABLES,
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
    **TORN_NO_OBSERVABLES,
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
    # Unset keys fill from the DAE overlay.
    step = solver.kernel.single_integrator._algo_step
    for key, value in DAE_SOLVER_DEFAULTS.items():
        assert getattr(step, key) == value


RING_RADAU_UNSET_VARIANTS = {
    **RING_RADAU,
    "inexact_newton": None,
    "prefactored": None,
}


@pytest.mark.parametrize(
    "solver_settings_override",
    [RING_RADAU_UNSET_VARIANTS],
    indirect=True,
)
def test_singular_mass_keeps_family_solver_variants(solver):
    # The DAE rule leaves the family's Newton-variant keys alone.
    step = solver.kernel.single_integrator._algo_step
    assert step.linear_correction_type == (
        DAE_SOLVER_DEFAULTS["linear_correction_type"]
    )
    declared = step.step_default_settings
    config = step.compile_settings
    assert config.inexact_newton == declared["inexact_newton"]
    assert config.prefactored == declared["prefactored"]


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
    "solver_settings_override", [RING_RADAU], indirect=True
)
def test_singular_mass_defaults_reapplied_on_swap(solver_mutable):
    # The DAE overlay re-applies on algorithm swap.
    solver_mutable.update({"algorithm": "backwards_euler"})
    step = solver_mutable.kernel.single_integrator._algo_step
    for key, value in DAE_SOLVER_DEFAULTS.items():
        assert getattr(step, key) == value


@pytest.mark.parametrize(
    "solver_settings_override", [MASSLESS_DEFAULTS], indirect=True
)
def test_plain_system_keeps_default_linear_solve(solver):
    # Massless systems keep the algorithm-family defaults.
    step = solver.kernel.single_integrator._algo_step
    declared = step.step_default_settings
    assert step.preconditioner_type == declared["preconditioner_type"]
    assert (
        step.linear_correction_type == DEFAULT_LINEAR_CORRECTION_TYPE
    )
    width = int(step.compile_settings.solver_width)
    cap = step.solver.linear_solver.compile_settings.max_iters
    assert cap == math.ceil(1.5 * width)


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


TORN_INIT_COUNTERS = {
    **TORN_INIT_COMMON,
    "output_types": ["state", "time", "iteration_counters"],
}


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_COUNTERS], indirect=True
)
def test_init_iterations_land_in_the_t0_counter_row(solver):
    # The t0 counter row records the init solve's iterations.
    result, _ = _solve_torn(solver, 2.0, 0.0)
    counters = np.asarray(result.iteration_counters)
    assert counters[0, 0, 0] >= 1
    assert counters[0, 1, 0] >= 1
    assert counters[0, 2, 0] == 0
    assert counters[1, 2, 0] >= 1


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_INIT_COMMON], indirect=True
)
def test_initialiser_defaults_and_decoupled_budget(solver):
    # Torn systems default to a brown initialiser over an exact LU
    # solve; the stage solver's Newton budget stays its own.
    initialiser = solver.kernel.single_integrator._dae_initialiser
    assert not initialiser.is_noop
    assert initialiser.dae_initialisation == "brown"
    assert initialiser.newton_max_iters == 50
    assert (
        initialiser.linear_solver.linear_correction_type == "lu"
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


def test_init_mode_noop_on_non_dae_system(system):
    # A massless system compiles the initialiser to a no-op.
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


# None unsets spine pins; the runs take production defaults.
DIODE_LINE_DIRK = {
    "system_type": "diode_line",
    "precision": np.float32,
    "algorithm": "l_stable_dirk_3",
    "linear_correction_type": "lu",
    "preconditioner_type": None,
    "step_controller": None,
    "rtol": 1e-4,
    "atol": 1e-6,
    "saved_state_indices": list(range(16)),
    "save_every": 0.1,
    **TORN_NO_OBSERVABLES,
}

DIODE_LINE_RADAU = {
    **DIODE_LINE_DIRK,
    "algorithm": "radau_iia_5",
}

# y1, y4, y7 are observables.
TRANSAMP_OUTPUTS = {
    "output_types": ["state", "observables", "time"],
    "saved_state_indices": list(range(8)),
    "saved_observable_indices": list(range(3)),
    "summarised_observable_indices": [],
}

TRANSAMP_DIRK = {
    "system_type": "transistor_amplifier",
    "precision": np.float32,
    "algorithm": "l_stable_dirk_3",
    "linear_correction_type": "lu",
    "preconditioner_type": None,
    "dae_initialisation": "brown",
    "step_controller": "fixed",
    "dt": 0.01,
    "inexact_newton": False,
    "newton_atol": 1e-2,
    "newton_rtol": 1e-2,
    "rtol": 1e-4,
    "atol": 1e-6,
    "save_every": 2.5e-4,
    **TRANSAMP_OUTPUTS,
}

# Adaptive radau from the Test Set initial state under brown init.
TRANSAMP_RADAU_ADAPTIVE = {
    "system_type": "transistor_amplifier",
    "precision": np.float32,
    "algorithm": "radau_iia_5",
    "linear_correction_type": "lu",
    "preconditioner_type": None,
    "dae_initialisation": "brown",
    "step_controller": None,
    "dt": 1e-4,
    "dt_min": 1e-8,
    "dt_max": 1e-2,
    "inexact_newton": False,
    "newton_atol": 1e-2,
    "newton_rtol": 1e-2,
    "rtol": 1e-4,
    "atol": 1e-4,
    "save_every": 0.1,
    **TRANSAMP_OUTPUTS,
}

# IVP Test Set transamp reference solution at t = 0.2.
TRANSAMP_REFERENCE = {
    "y1": -0.5562145012262709e-2,
    "y2": 3.006522471903042,
    "y3": 2.849958788608128,
    "y4": 2.926422536206241,
    "y5": 2.704617865010554,
    "y6": 2.761837778393145,
    "y7": 4.770927631616772,
    "y8": 1.236995868091548,
}


def _transamp_consistent_derivatives():
    """Solve y'(0) from the differentiated node constraints."""
    k = TRANSAMP_CONSTANTS
    y = TRANSAMP_DC_STATES
    drive_rate = 0.1 * 628.3185307179587

    def conductance(a, b):
        return k["bb"] / k["uf"] * np.exp((a - b) / k["uf"])

    dy3 = -(y["y3"] / k["r3"]) / k["c2"]
    dy6 = -(y["y6"] / k["r7"]) / k["c4"]
    g23 = conductance(y["y2"], y["y3"])
    g56 = conductance(y["y5"], y["y6"])
    rates = {}

    def solve_linear(residual):
        return -residual(0.0) / (residual(1.0) - residual(0.0))

    rates["y2_t"] = solve_linear(
        lambda v: -drive_rate / k["r0"]
        + v / k["r0"]
        + v * (1.0 / k["r1"] + 1.0 / k["r2"])
        - (k["alfa"] - 1.0) * g23 * (v - dy3)
    )
    rates["y5_t"] = solve_linear(
        lambda v: v / k["r4"]
        + k["alfa"] * g23 * (rates["y2_t"] - dy3)
        + v * (1.0 / k["r5"] + 1.0 / k["r6"])
        - (k["alfa"] - 1.0) * g56 * (v - dy6)
    )
    rates["y7_t"] = solve_linear(
        lambda v: v / k["r8"]
        + k["alfa"] * g56 * (rates["y5_t"] - dy6)
        + v / k["r9"]
    )
    return rates


def _transamp_trajectory(solver, system, duration):
    inits = {
        name: np.array([float(value)])
        for name, value in system.initial_values.values_dict.items()
    }
    result = solver.solve(inits, {}, duration=duration)
    legend = {
        label: idx for idx, label in result.time_domain_legend.items()
    }
    return result, legend, result.time_domain_array


def _diode_constraint_residuals(finals, t_end, amp):
    """Evaluate the eight ladder constraints from named finals."""
    gs, a, c = 0.1, 3.0, 0.5
    drive = amp * np.sin(2.0 * np.pi * t_end)
    residuals = []
    for i in range(1, 9):
        w_i = finals[f"w{i}"]
        upstream = finals[f"w{i + 1}"] if i < 8 else drive
        residuals.append(
            (finals[f"v{i}"] - w_i)
            - gs * (np.exp(a * w_i) - np.exp(-a * w_i))
            + c * (upstream - w_i)
        )
    return residuals


@pytest.mark.parametrize(
    "solver_settings_override",
    [DIODE_LINE_DIRK, DIODE_LINE_RADAU],
    ids=["dirk-lu", "radau-prefactored-lu"],
    indirect=True,
)
def test_diode_line_solves(solver, system):
    """The mid-size semi-explicit DAE integrates cleanly."""
    t_end = 0.3
    inits = np.zeros((system.sizes.states, 1), dtype=np.float32)
    params = np.full((1, 1), 1.0, dtype=np.float32)
    result = solver.solve(inits, params, duration=t_end)
    legend = {
        label: idx for idx, label in result.time_domain_legend.items()
    }
    trajectory = result.time_domain_array
    assert np.isfinite(trajectory).all()
    finals = {
        name: float(trajectory[-1, legend[name], 0])
        for name in legend
        if name != "time"
    }
    # A flat trajectory cannot satisfy the driven boundary row.
    assert abs(finals["w8"]) > 0.05
    for residual in _diode_constraint_residuals(finals, t_end, 1.0):
        assert residual == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize(
    "solver_settings_override", [DIODE_LINE_DIRK], indirect=True
)
def test_brown_init_corrects_diode_line_algebraic_start(
    solver, system
):
    """Brown init from the Solver corrects the algebraic start."""
    initialiser = solver.kernel.single_integrator._dae_initialiser
    assert initialiser.dae_initialisation == "brown"
    inits = {
        f"v{i}": np.array([0.0]) for i in range(1, 9)
    }
    inits.update(
        {f"w{i}": np.array([0.25]) for i in range(1, 9)}
    )
    result = solver.solve(
        inits, {"amp": np.array([1.0])}, duration=0.2
    )
    assert result.status_messages == {}
    legend = {
        label: idx for idx, label in result.time_domain_legend.items()
    }
    trajectory = result.time_domain_array
    start = {
        name: float(trajectory[0, legend[name], 0])
        for name in legend
        if name != "time"
    }
    for i in range(1, 9):
        assert start[f"v{i}"] == 0.0
    for residual in _diode_constraint_residuals(start, 0.0, 1.0):
        assert residual == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "solver_settings_override", [TRANSAMP_DIRK], indirect=True
)
def test_transistor_amplifier_advances_from_t0(solver, system):
    """The singular capacitor blocks integrate past the first step."""
    assert system.mass_diagonal_flags == (
        True, False, True, True, False, True, False, True
    )
    result, legend, trajectory = _transamp_trajectory(
        solver, system, 1e-3
    )
    assert result.status_messages == {}
    assert np.isfinite(trajectory).all()
    assert set(system.initial_values.values_dict) <= set(legend)
    assert {"y1", "y4", "y7"} <= set(legend)
    y1_final = float(trajectory[-1, legend["y1"], 0])
    # The input node tracks the 0.1 V drive through R0*C1 = 1 ms.
    assert abs(y1_final) > 5e-3


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override", [TRANSAMP_RADAU_ADAPTIVE], indirect=True
)
def test_transistor_amplifier_init_and_reference(solver, system):
    """Consistent derivative states at t0 and the reference at 0.2."""
    initialiser = solver.kernel.single_integrator._dae_initialiser
    assert initialiser.dae_initialisation == "brown"
    assert solver.kernel.single_integrator._step_controller.is_adaptive
    result, legend, trajectory = _transamp_trajectory(
        solver, system, 0.2
    )
    assert result.status_messages == {}
    assert np.isfinite(trajectory).all()
    expected = _transamp_consistent_derivatives()
    algebraic = {
        name
        for name, flag in zip(
            system.initial_values.values_dict, system.mass_diagonal_flags
        )
        if not flag
    }
    assert set(expected) == algebraic
    for name, value in expected.items():
        assert float(trajectory[0, legend[name], 0]) == pytest.approx(
            value, rel=1e-4
        )
    for name in ("y2", "y3", "y5", "y6", "y8"):
        assert float(trajectory[0, legend[name], 0]) == (
            TRANSAMP_DC_STATES[name]
        )
    for name, value in TRANSAMP_REFERENCE.items():
        assert float(trajectory[-1, legend[name], 0]) == pytest.approx(
            value, abs=2e-3
        )
