"""Tests for the starting-step estimator."""

import numpy as np
import pytest
from scipy.integrate._ivp.common import select_initial_step

from tests._utils import (
    ALGORITHM_CHAIN_SETS,
    IRRECOVERABLE,
    TORN_NO_OBSERVABLES,
    UNSET_LINEAR_SOLVE,
    run_starting_step_device,
    swept_values,
)

ERK_CHAIN = ALGORITHM_CHAIN_SETS["erk"]

# Test the torn system for its algebraic row, which starting_step_fn
# holds over the Euler step and leaves out of the norms.
TORN_RADAU = {
    "system_type": "torn_driver",
    "algorithm": "radau",
    "step_controller": "pi",
    **TORN_NO_OBSERVABLES,
    **UNSET_LINEAR_SOLVE,
}


@pytest.fixture(scope="function")
def device_estimate(
    single_integrator_run,
    system,
    precision,
    cpu_system,
    cpu_driver_evaluator,
):
    """Return a callable giving the device estimate at a state."""

    def estimate(state):
        t0 = precision(0.0)
        parameters = system.parameters.defaults_array(precision)
        drivers = cpu_driver_evaluator(t0)
        return run_starting_step_device(
            single_integrator_run._starting_step,
            precision,
            state,
            swept_values(system),
            cpu_driver_evaluator.coefficients,
            drivers,
            cpu_system.observables(state, parameters, drivers, t0),
            t0,
        )

    return estimate


@pytest.mark.parametrize(
    "solver_settings_override, start_state",
    [
        pytest.param(ERK_CHAIN, np.copy, id="erk-system-state"),
        # Start at ones, off the torn system's constraint, to give
        # the algebraic row a residual to hold.
        pytest.param(TORN_RADAU, np.ones_like, id="torn-off-constraint"),
    ],
    indirect=["solver_settings_override"],
)
def test_starting_step_matches_scipy(
    device_estimate,
    single_integrator_run,
    start_state,
    system,
    precision,
    initial_state,
    cpu_system,
    cpu_driver_evaluator,
    tolerance,
):
    """Check the device estimate against SciPy's starting step."""
    controller = single_integrator_run._step_controller
    state = start_state(initial_state)
    parameters = system.parameters.defaults_array(precision)
    differential = np.asarray(system.mass_diagonal_flags, dtype=bool)

    def differential_rhs(time, differential_state):
        # Evaluate the differential rows with the algebraic rows held
        # at their starting values.
        full_state = state.copy()
        full_state[differential] = differential_state
        time = precision(time)
        drivers = cpu_driver_evaluator(time)
        observables = cpu_system.observables(
            full_state, parameters, drivers, time
        )
        derivative, _ = cpu_system.rhs(
            full_state, parameters, drivers, observables, time
        )
        return np.asarray(derivative, dtype=np.float64)[differential]

    start = state[differential].astype(np.float64)
    expected = select_initial_step(
        differential_rhs,
        0.0,
        start,
        np.inf,
        float(controller.dt_max),
        differential_rhs(0.0, start),
        1.0,
        controller.algorithm_order,
        np.asarray(controller.rtol, dtype=np.float64)[differential],
        np.asarray(controller.atol, dtype=np.float64)[differential],
    )

    np.testing.assert_allclose(
        device_estimate(state), expected, rtol=tolerance.rel_loose, atol=0.0
    )


@pytest.mark.parametrize(
    "solver_settings_override", [ERK_CHAIN], indirect=True
)
def test_zero_state_estimate_grows_from_the_fallback(
    device_estimate, single_integrator_run, initial_state
):
    """Check that the fallback caps a zero state's estimate."""
    controller = single_integrator_run._step_controller
    settings = single_integrator_run._starting_step.compile_settings
    estimate = device_estimate(np.zeros_like(initial_state))
    assert estimate <= (
        settings.starting_step_max_growth * settings.fallback_step
    )
    assert controller.dt_min <= estimate <= controller.dt_max


@pytest.mark.parametrize(
    "solver_settings_override", [IRRECOVERABLE], indirect=True
)
def test_estimate_stays_within_narrow_step_bounds(
    device_estimate, single_integrator_run, initial_state
):
    """Check that the step bounds clamp the estimate."""
    controller = single_integrator_run._step_controller
    estimate = device_estimate(initial_state)
    assert controller.dt_min <= estimate <= controller.dt_max


@pytest.mark.parametrize(
    "solver_settings_override", [ERK_CHAIN], indirect=True
)
def test_given_dt_replaces_the_estimate(
    single_integrator_run_mutable, precision
):
    """Check that the loop starts at a given dt, else estimates."""
    run = single_integrator_run_mutable
    estimator = run._starting_step
    loop_settings = run._loop.compile_settings
    assert estimator.estimates_dt is True
    assert run.dt is None
    assert loop_settings.starting_step_fn is estimator.device_function

    given = precision(0.01)
    run.update(dt=given)
    assert estimator.estimates_dt is False
    assert run.dt == given
    assert run._loop.compile_settings.starting_step_fn is None

    run.update(dt=None)
    assert estimator.estimates_dt is True
    assert run.dt is None
    assert run._loop.compile_settings.starting_step_fn is (
        estimator.device_function
    )


def test_fixed_controller_starts_at_dt(single_integrator_run):
    """Check that a fixed controller's run starts at its dt."""
    run = single_integrator_run
    assert run._starting_step.estimates_dt is False
    assert run.dt == run._step_controller.dt
    assert run._loop.compile_settings.starting_step_fn is None


@pytest.mark.parametrize(
    "solver_settings_override", [ERK_CHAIN], indirect=True
)
def test_initial_derivative_location_moves_the_buffer(
    single_integrator_run_mutable, system
):
    """Check that the location setting moves the derivative buffer."""
    run = single_integrator_run_mutable
    estimator = run._starting_step
    assert estimator.local_buffer_size == system.sizes.states
    recognised = run.update(initial_derivative_location="shared")
    assert "initial_derivative_location" in recognised
    assert estimator.shared_buffer_size == system.sizes.states


@pytest.mark.parametrize(
    "solver_settings_override", [ERK_CHAIN], indirect=True
)
@pytest.mark.parametrize(
    "setting",
    [
        "starting_step_state_fraction",
        "starting_step_zero_norm",
        "starting_step_target",
        "starting_step_max_growth",
    ],
)
def test_estimate_settings_rebuild_the_estimate(
    single_integrator_run_mutable, setting
):
    """Check that each estimate setting reaches the estimator."""
    run = single_integrator_run_mutable
    estimator = run._starting_step
    before = estimator.device_function
    given = 2.0 * getattr(estimator.compile_settings, setting)
    recognised = run.update({setting: given})
    assert setting in recognised
    assert getattr(estimator.compile_settings, setting) == given
    assert estimator.device_function is not before
