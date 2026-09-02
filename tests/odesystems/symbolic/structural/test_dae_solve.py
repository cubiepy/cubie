"""Numerical solve of a structurally simplified torn DAE.

Integrates ``dx/dt = -z`` with the implicit constraint
``z**5 + z = x`` (not explicitly solvable, so ``z`` is a torn
algebraic state under a singular mass matrix) and compares against a
high-accuracy reference computed on the reduced ODE with a per-step
Newton solve for ``z``.
"""

import math

import numpy as np
import pytest

from cubie import Solver, solve_ivp
from cubie.cuda_simsafe import cuda
from cubie.integrators.algorithms.generic_firk_tableaus import (
    RADAU_IIA_5_TABLEAU,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    DAE_SOLVER_DEFAULTS,
    DEFAULT_LINEAR_CORRECTION_TYPE,
)
from cubie.memory import default_memmgr
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

TORN_RADAU = {**TORN_SYSTEM_DEFAULTS, "algorithm": "radau_iia_5"}


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_SYSTEM_DEFAULTS], indirect=True
)
def test_singular_mass_defaults_linear_solve_params(solver):
    # Unset keys fill from the DAE overlay.
    step = solver.kernel.single_integrator._algo_step
    for key, value in DAE_SOLVER_DEFAULTS.items():
        assert getattr(step, key) == value


TORN_RADAU_UNSET_VARIANTS = {
    **TORN_RADAU,
    "inexact_newton": None,
    "prefactored": None,
}


@pytest.mark.parametrize(
    "solver_settings_override",
    [TORN_RADAU_UNSET_VARIANTS],
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
    "solver_settings_override", [TORN_RADAU], indirect=True
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
    # Brown holds x0, lands x1 on the constraint, and counts at t0.
    result, (x0, x1) = _solve_torn(solver, 2.0, 0.0)
    assert x0 == 2.0
    assert _torn_constraint_residual(x0, x1) == pytest.approx(
        0.0, abs=1e-8
    )
    assert result.status_messages == {}
    counters = np.asarray(result.iteration_counters)
    assert counters[0, 0, 0] >= 1
    assert counters[0, 1, 0] >= 1
    assert counters[0, 2, 0] == 0
    assert counters[1, 2, 0] >= 1


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

# y1, y4, y7 are observables.
TRANSAMP_OUTPUTS = {
    "output_types": ["state", "observables", "time"],
    "saved_state_indices": list(range(8)),
    "saved_observable_indices": list(range(3)),
    "summarised_observable_indices": [],
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
    [DIODE_LINE_DIRK],
    ids=["dirk-lu"],
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


def _dense_at_state_operator(device_fn, state, params, drivers, t, h):
    """Return the dense ``M - h*J`` from basis-vector applies."""
    precision = state.dtype.type
    n = state.shape[0]
    stream = default_memmgr.get_group_stream()
    state_dev = cuda.to_device(state, stream=stream)
    params_dev = cuda.to_device(params, stream=stream)
    drivers_dev = cuda.to_device(drivers, stream=stream)

    @cuda.jit
    def kernel(state_in, params_in, drivers_in, vec, out):
        cached_aux = cuda.local.array(1, precision)
        device_fn(
            state_in, params_in, drivers_in, cached_aux, state_in,
            precision(t), precision(h), precision(1.0), vec, out,
        )

    columns = []
    for column in range(n):
        vec = np.zeros(n, dtype=precision)
        vec[column] = 1.0
        vec_dev = cuda.to_device(vec, stream=stream)
        out_dev = cuda.to_device(
            np.zeros(n, dtype=precision), stream=stream
        )
        kernel[1, 1, stream](
            state_dev, params_dev, drivers_dev, vec_dev, out_dev
        )
        columns.append(out_dev.copy_to_host(stream=stream))
    stream.synchronize()
    return np.column_stack(columns).astype(np.float64)


@pytest.mark.parametrize(
    "solver_settings_override", [DIODE_LINE_DIRK], indirect=True
)
def test_diode_line_stacked_prefactored_lu_matches_dense(
    system, precision
):
    """Static structural pivots factorise the empty boundary diagonal."""
    tableau = RADAU_IIA_5_TABLEAU
    a_matrix = np.asarray(tableau.stage_coefficients, dtype=np.float64)
    s = tableau.stage_count
    n = system.sizes.states
    h, t = 0.01, 0.3
    state = np.full(n, 0.1, dtype=precision)
    params = np.ones(1, dtype=precision)
    drivers = np.zeros(1, dtype=precision)

    operator = system.get_solver_helper(
        role="linear_operator", jacobian_at="state"
    ).device_function
    mass = np.asarray(system.mass, dtype=np.float64)
    jacobian = (
        mass
        - _dense_at_state_operator(
            operator, state, params, drivers, t, h
        )
    ) / h

    member = system.get_solver_helper(
        "lu_solve",
        jacobian_at="step",
        prefactored=True,
        stacked=True,
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    lu_solve = member.device_function
    prepare = member.prepare_jac
    assert member.lu_nnz == 0
    assert member.cached_auxiliary_count > 0

    rng = np.random.default_rng(16)
    rhs = rng.normal(size=s * n).astype(precision)
    stream = default_memmgr.get_group_stream()
    state_dev = cuda.to_device(state, stream=stream)
    params_dev = cuda.to_device(params, stream=stream)
    drivers_dev = cuda.to_device(drivers, stream=stream)
    cached = cuda.to_device(
        np.zeros(member.cached_auxiliary_count, dtype=precision),
        stream=stream,
    )
    rhs_dev = cuda.to_device(rhs, stream=stream)
    x_dev = cuda.to_device(np.zeros(s * n, dtype=precision), stream=stream)
    factor = cuda.to_device(np.zeros(1, dtype=precision), stream=stream)
    flag = cuda.to_device(np.zeros(2, dtype=np.int32), stream=stream)

    @cuda.jit
    def kernel(state_in, params_in, drivers_in, cached_aux, rhs_io, x,
               factor_buf, flags):
        flags[0] = prepare(
            state_in, params_in, drivers_in, precision(t), precision(h),
            cached_aux,
        )
        flags[1] = lu_solve(
            state_in, params_in, drivers_in, cached_aux, state_in,
            precision(t), precision(h), precision(0.0), rhs_io, x,
            factor_buf,
        )

    kernel[1, 1, stream](
        state_dev, params_dev, drivers_dev, cached, rhs_dev, x_dev,
        factor, flag,
    )
    flags = flag.copy_to_host(stream=stream)
    x_out = x_dev.copy_to_host(stream=stream)
    stream.synchronize()
    assert flags[0] == 0
    assert flags[1] == 0

    coupled = np.kron(np.eye(s), mass) - h * np.kron(a_matrix, jacobian)
    expected = np.linalg.solve(coupled, rhs.astype(np.float64))
    np.testing.assert_allclose(x_out, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override", [TRANSAMP_RADAU_ADAPTIVE], indirect=True
)
def test_transistor_amplifier_init_and_reference(solver, system):
    """Consistent derivative states at t0 and the reference at 0.2."""
    assert system.mass_diagonal_flags == (
        True, False, True, True, False, True, False, True
    )
    initialiser = solver.kernel.single_integrator._dae_initialiser
    assert initialiser.dae_initialisation == "brown"
    assert solver.kernel.single_integrator._step_controller.is_adaptive
    result, legend, trajectory = _transamp_trajectory(
        solver, system, 0.2
    )
    assert result.status_messages == {}
    assert np.isfinite(trajectory).all()
    assert set(system.initial_values.values_dict) <= set(legend)
    assert {"y1", "y4", "y7"} <= set(legend)
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
