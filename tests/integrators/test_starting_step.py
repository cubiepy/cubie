"""Tests for the starting-step estimator."""

import numpy as np
import pytest

from tests._utils import (
    ALGORITHM_CHAIN_SETS,
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


@pytest.mark.parametrize(
    "solver_settings_override, start_state",
    [
        pytest.param(ERK_CHAIN, np.copy, id="erk-system-state"),
        # Start at zero to take the fallback first guess.
        pytest.param(ERK_CHAIN, np.zeros_like, id="erk-zero-state"),
        # Start at ones, off the torn system's constraint, to give
        # the algebraic row a residual to hold.
        pytest.param(TORN_RADAU, np.ones_like, id="torn-off-constraint"),
    ],
    indirect=["solver_settings_override"],
)
def test_starting_step_matches_reference(
    single_integrator_run,
    start_state,
    system,
    precision,
    initial_state,
    cpu_system,
    cpu_step_controller,
    cpu_driver_evaluator,
    tolerance,
):
    """Check the device estimate against the CPU reference."""
    state = start_state(initial_state)
    parameters = system.parameters.defaults_array(precision)
    t0 = precision(0.0)
    drivers = cpu_driver_evaluator(t0)
    observables = cpu_system.observables(state, parameters, drivers, t0)

    estimate = run_starting_step_device(
        single_integrator_run._starting_step,
        precision,
        state,
        swept_values(system),
        cpu_driver_evaluator.coefficients,
        drivers,
        observables,
        t0,
        single_integrator_run._step_controller.dt,
    )
    reference = cpu_step_controller.starting_step(
        cpu_system,
        cpu_driver_evaluator,
        state,
        parameters,
        drivers,
        observables,
        t0,
    )

    np.testing.assert_allclose(
        estimate, reference, rtol=tolerance.rel_tight, atol=0.0
    )


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
