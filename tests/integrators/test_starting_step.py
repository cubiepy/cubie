"""Tests for the starting-step estimate device function."""

import numpy as np
import pytest
from numba_cuda_mlir.numba_cuda.np.numpy_support import from_dtype

from cubie._cudasim_extensions import cuda
from cubie.backend.jit import compile_kwargs
from cubie.memory import default_memmgr
from tests._utils import STARTING_STEP_CASES, swept_values


def _device_starting_step(
    single_integrator_run, system, precision, initial_state, driver_array
):
    """Return the device estimate at ``initial_state`` and ``t = 0``."""
    starting_step_fn = single_integrator_run._starting_step.device_function
    drivers_fn = None if driver_array is None else driver_array.drivers_fn
    observables_fn = system.observables_fn
    numba_precision = from_dtype(precision)

    n_states = system.sizes.states
    # Give each empty buffer one spare element, because we cannot copy
    # a zero-length array to the device.
    n_drivers = max(1, system.sizes.drivers)
    n_observables = max(1, system.sizes.observables)
    if driver_array is None:
        coefficients = np.zeros((1, 1, 1), dtype=precision)
    else:
        coefficients = np.asarray(driver_array.coefficients, dtype=precision)

    stream = default_memmgr.get_group_stream()
    d_state = cuda.to_device(
        np.asarray(initial_state, dtype=precision), stream=stream
    )
    d_params = cuda.to_device(
        swept_values(system).astype(precision), stream=stream
    )
    d_coefficients = cuda.to_device(coefficients, stream=stream)
    d_drivers = cuda.to_device(
        np.zeros(n_drivers, dtype=precision), stream=stream
    )
    d_observables = cuda.to_device(
        np.zeros(n_observables, dtype=precision), stream=stream
    )
    d_probe_state = cuda.to_device(
        np.zeros(n_states, dtype=precision), stream=stream
    )
    d_probe_drivers = cuda.to_device(
        np.zeros(n_drivers, dtype=precision), stream=stream
    )
    d_probe_observables = cuda.to_device(
        np.zeros(n_observables, dtype=precision), stream=stream
    )
    d_probe_derivative = cuda.to_device(
        np.zeros(n_states, dtype=precision), stream=stream
    )
    d_out = cuda.to_device(np.zeros(1, dtype=precision), stream=stream)

    @cuda.jit(**compile_kwargs)
    def kernel(
        state,
        parameters,
        driver_coefficients,
        drivers,
        observables,
        probe_state,
        probe_drivers,
        probe_observables,
        probe_derivative,
        out,
    ):
        if cuda.grid(1) > 0:
            return
        zero = numba_precision(0.0)
        scratch = cuda.local.array(1, dtype=numba_precision)
        if drivers_fn is not None:
            drivers_fn(zero, driver_coefficients, drivers)
        observables_fn(state, parameters, drivers, observables, zero)
        out[0] = starting_step_fn(
            state,
            parameters,
            driver_coefficients,
            drivers,
            observables,
            probe_state,
            probe_drivers,
            probe_observables,
            probe_derivative,
            zero,
            scratch,
            scratch,
        )

    kernel[1, 1, stream](
        d_state,
        d_params,
        d_coefficients,
        d_drivers,
        d_observables,
        d_probe_state,
        d_probe_drivers,
        d_probe_observables,
        d_probe_derivative,
        d_out,
    )
    stream.synchronize()
    return d_out.copy_to_host()[0]


@pytest.mark.parametrize(
    "solver_settings_override",
    list(STARTING_STEP_CASES.values()),
    ids=list(STARTING_STEP_CASES),
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
    """Check the device estimate against the CPU reference."""
    device_estimate = _device_starting_step(
        single_integrator_run, system, precision, initial_state, driver_array
    )

    state = np.asarray(initial_state, dtype=precision)
    params = system.parameters.defaults_array(precision)
    t0 = precision(0.0)
    drivers = cpu_driver_evaluator(t0)
    observables = cpu_system.observables(state, params, drivers, t0)
    reference = cpu_step_controller.starting_step(
        cpu_system,
        cpu_driver_evaluator,
        state,
        params,
        drivers,
        observables,
        t0,
    )

    # We compare at the loose tolerance because the device evaluates
    # sqrt and pow with approximate-function fastmath.
    np.testing.assert_allclose(
        device_estimate, reference, rtol=tolerance.rel_loose, atol=0.0
    )
