"""Tests for the starting-step estimator."""

import numpy as np
import pytest
from numba_cuda_mlir.numba_cuda.np.numpy_support import from_dtype

from cubie._cudasim_extensions import cuda
from cubie.backend.jit import compile_kwargs
from cubie.memory import default_memmgr
from tests._utils import (
    ESTIMATED_DT_CHAIN,
    TORN_NO_OBSERVABLES,
    UNSET_LINEAR_SOLVE,
    swept_values,
)

# We test the torn system for its algebraic row, which
# starting_step_fn holds over the Euler step and leaves out of the
# norms.
TORN_ESTIMATED_DT = {
    "system_type": "torn_driver",
    "algorithm": "radau",
    "step_controller": "pi",
    **TORN_NO_OBSERVABLES,
    **UNSET_LINEAR_SOLVE,
    "dt": None,
}


@pytest.mark.parametrize(
    "solver_settings_override",
    [ESTIMATED_DT_CHAIN, TORN_ESTIMATED_DT],
    ids=["erk", "torn-radau"],
    indirect=True,
)
def test_starting_step_matches_reference(
    single_integrator_run,
    system,
    precision,
    initial_state,
    driver_array,
    cpu_system,
    cpu_step_controller,
    cpu_driver_evaluator,
    tolerance,
):
    """Check each run's device estimate against the CPU reference."""
    estimator = single_integrator_run._starting_step
    starting_step_fn = estimator.device_function
    drivers_fn = None if driver_array is None else driver_array.drivers_fn
    observables_fn = system.observables_fn
    numba_precision = from_dtype(precision)

    # Start the second run at zero, where starting_step_fn uses its
    # fixed first guess. Start the third off the torn system's
    # constraint, so that starting_step_fn has a residual to hold.
    # Start the fourth at NaN, where every norm is NaN.
    states = np.stack(
        [
            initial_state,
            np.zeros_like(initial_state),
            np.ones_like(initial_state),
            np.full_like(initial_state, np.nan),
        ]
    )
    n_runs, n_states = states.shape
    # Give each local array at least one element.
    n_drivers = max(1, system.sizes.drivers)
    n_observables = max(1, system.sizes.observables)
    shared_elements = max(1, estimator.shared_buffer_size)
    persistent_elements = max(1, estimator.persistent_local_buffer_size)
    if driver_array is None:
        coefficients = np.zeros((1, 1, 1), dtype=precision)
    else:
        coefficients = np.asarray(driver_array.coefficients, dtype=precision)

    @cuda.jit(**compile_kwargs)
    def kernel(run_states, parameters, driver_coefficients, estimates):
        run = cuda.grid(1)
        if run >= run_states.shape[0]:
            return
        zero = numba_precision(0.0)
        state = run_states[run]
        drivers = cuda.local.array(n_drivers, dtype=numba_precision)
        observables = cuda.local.array(n_observables, dtype=numba_precision)
        proposed_state = cuda.local.array(n_states, dtype=numba_precision)
        proposed_drivers = cuda.local.array(n_drivers, dtype=numba_precision)
        proposed_observables = cuda.local.array(
            n_observables, dtype=numba_precision
        )
        error = cuda.local.array(n_states, dtype=numba_precision)
        shared = cuda.local.array(shared_elements, dtype=numba_precision)
        persistent = cuda.local.array(
            persistent_elements, dtype=numba_precision
        )
        if drivers_fn is not None:
            drivers_fn(zero, driver_coefficients, drivers)
        observables_fn(state, parameters, drivers, observables, zero)
        estimates[run] = starting_step_fn(
            state,
            proposed_state,
            parameters,
            driver_coefficients,
            drivers,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            zero,
            shared,
            persistent,
        )

    stream = default_memmgr.get_group_stream()
    d_states = cuda.to_device(states, stream=stream)
    d_parameters = cuda.to_device(
        swept_values(system).astype(precision), stream=stream
    )
    d_coefficients = cuda.to_device(coefficients, stream=stream)
    d_estimates = cuda.to_device(
        np.zeros(n_runs, dtype=precision), stream=stream
    )
    kernel[1, n_runs, stream](
        d_states, d_parameters, d_coefficients, d_estimates
    )
    estimates = d_estimates.copy_to_host(stream=stream)
    stream.synchronize()

    parameters = system.parameters.defaults_array(precision)
    t0 = precision(0.0)
    drivers = cpu_driver_evaluator(t0)
    reference = np.empty(n_runs, dtype=precision)
    for run, state in enumerate(states):
        observables = cpu_system.observables(state, parameters, drivers, t0)
        reference[run] = cpu_step_controller.starting_step(
            cpu_system,
            cpu_driver_evaluator,
            state,
            parameters,
            drivers,
            observables,
            t0,
        )

    np.testing.assert_allclose(
        estimates, reference, rtol=tolerance.rel_tight, atol=0.0
    )


@pytest.mark.parametrize(
    "solver_settings_override", [ESTIMATED_DT_CHAIN], indirect=True
)
def test_given_dt_replaces_the_estimate(single_integrator_run_mutable):
    """Check that giving dt removes the estimate from the loop."""
    run = single_integrator_run_mutable
    estimator = run._starting_step
    assert estimator.estimates_dt is True
    assert run._loop.compile_settings.starting_step_fn is (
        estimator.device_function
    )
    run.update(dt=0.01)
    assert estimator.estimates_dt is False
    assert run._loop.compile_settings.starting_step_fn is None


@pytest.mark.parametrize(
    "solver_settings_override", [ESTIMATED_DT_CHAIN], indirect=True
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
