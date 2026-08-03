"""Numerical correctness tests for single-step integration algorithms.

Compares CUDA device step results against CPU reference implementations
for all algorithm families across two consecutive integration steps.
"""

import attrs

import numpy as np
import pytest
from cubie.cuda_simsafe import cuda, numba_from_dtype as from_dtype, int32
from cubie.memory import default_memmgr
from numpy.testing import assert_allclose

from tests.integrators.cpu_reference import (
    CPUODESystem,
    get_ref_stepper,
)
from tests._utils import (
    BICGSTAB_STEP_CASES,
    ALGORITHM_PARAM_SETS,
    _get_algorithm_tableau,
)

Array = np.ndarray
STATUS_MASK = 0xFFFF


@attrs.define
class DualStepResult:
    """Container recording back-to-back step executions."""

    first_state: Array
    second_state: Array
    first_observables: Array
    second_observables: Array
    first_error: Array
    second_error: Array
    statuses: tuple[int, int]


@pytest.fixture(scope="session")
def step_inputs(
    system,
    precision,
    initial_state,
    solver_settings,
    cpu_driver_evaluator,
) -> dict[str, Array]:
    """State, parameters, and drivers for a single step execution."""
    width = system.num_drivers
    driver_coefficients = np.array(
        cpu_driver_evaluator.coefficients, dtype=precision, copy=True
    )
    return {
        "state": initial_state,
        "parameters": system.parameters.values_array.astype(precision),
        "drivers": np.zeros(width, dtype=precision),
        "driver_coefficients": driver_coefficients,
    }


def _execute_step_twice(
    step_object,
    solver_settings,
    precision,
    step_inputs,
    system,
    driver_array,
) -> DualStepResult:
    """Run the compiled step twice without clearing shared memory."""

    shared_elems = step_object.shared_buffer_size

    step_function = step_object.step_function
    evaluate_driver_at_t = (
        driver_array.evaluation_function
        if driver_array is not None
        else None
    )
    evaluate_observables = system.evaluate_observables

    params = step_inputs["parameters"]
    state = np.asarray(step_inputs["state"], dtype=precision)
    driver_coefficients = step_inputs["driver_coefficients"]

    n_states = system.sizes.states
    n_drivers = system.sizes.drivers
    n_observables = system.sizes.observables

    proposed_state_first = np.zeros_like(state)
    proposed_state_second = np.zeros_like(state)

    error_first = np.zeros(n_states, dtype=precision)
    error_second = np.zeros(n_states, dtype=precision)

    drivers_current = np.zeros(n_drivers, dtype=precision)
    proposed_drivers_first = np.zeros(n_drivers, dtype=precision)
    proposed_drivers_second = np.zeros(n_drivers, dtype=precision)

    observables_current = np.zeros(n_observables, dtype=precision)
    proposed_observables_first = np.zeros(
        n_observables, dtype=precision
    )
    proposed_observables_second = np.zeros(
        n_observables, dtype=precision
    )

    status = np.zeros(2, dtype=np.int32)
    counters_first = np.zeros(2, dtype=np.int32)
    counters_second = np.zeros(2, dtype=np.int32)

    shared_bytes = precision(0).itemsize * shared_elems
    persistent_len = max(1, step_object.persistent_local_buffer_size)
    numba_precision = from_dtype(precision)
    dt_value = precision(solver_settings["dt"])

    d_state = cuda.to_device(state)
    d_params = cuda.to_device(params)
    d_driver_coeffs = cuda.to_device(driver_coefficients)

    d_proposed_first = cuda.to_device(proposed_state_first)
    d_proposed_second = cuda.to_device(proposed_state_second)

    d_drivers_current = cuda.to_device(drivers_current)
    d_proposed_drivers_first = cuda.to_device(proposed_drivers_first)
    d_proposed_drivers_second = cuda.to_device(proposed_drivers_second)

    d_observables_current = cuda.to_device(observables_current)
    d_proposed_observables_first = cuda.to_device(
        proposed_observables_first
    )
    d_proposed_observables_second = cuda.to_device(
        proposed_observables_second
    )

    d_error_first = cuda.to_device(error_first)
    d_error_second = cuda.to_device(error_second)

    d_status = cuda.to_device(status)
    d_counters_first = cuda.to_device(counters_first)
    d_counters_second = cuda.to_device(counters_second)

    state_len = int(n_states)
    driver_len = int(n_drivers)
    observable_len = int(n_observables)

    @cuda.jit()
    def kernel(
        state_vec,
        params_vec,
        driver_coeffs_vec,
        drivers_current_vec,
        proposed_drivers_vec_first,
        proposed_drivers_vec_second,
        observables_current_vec,
        proposed_observables_vec_first,
        proposed_observables_vec_second,
        proposed_vec_first,
        proposed_vec_second,
        error_vec_first,
        error_vec_second,
        status_vec,
        counters_vec_first,
        counters_vec_second,
        dt_scalar,
    ) -> None:
        idx = cuda.grid(1)
        if idx > 0:
            return
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(
            persistent_len, dtype=numba_precision
        )

        zero = numba_precision(0.0)

        for cache_idx in range(shared_elems):
            shared[cache_idx] = zero
        for pers_idx in range(persistent_len):
            persistent[pers_idx] = zero

        if evaluate_driver_at_t is not None:
            evaluate_driver_at_t(
                zero, driver_coeffs_vec, drivers_current_vec
            )
        evaluate_observables(
            state_vec,
            params_vec,
            drivers_current_vec,
            observables_current_vec,
            zero,
        )

        first_status = step_function(
            state_vec,
            proposed_vec_first,
            params_vec,
            driver_coeffs_vec,
            drivers_current_vec,
            proposed_drivers_vec_first,
            observables_current_vec,
            proposed_observables_vec_first,
            error_vec_first,
            dt_scalar,
            zero,
            int32(1),
            int32(1),
            shared,
            persistent,
            counters_vec_first,
        )
        status_vec[0] = first_status

        for elem in range(state_len):
            state_vec[elem] = proposed_vec_first[elem]
        for drv_idx in range(driver_len):
            drivers_current_vec[drv_idx] = (
                proposed_drivers_vec_first[drv_idx]
            )
        for obs_idx in range(observable_len):
            observables_current_vec[obs_idx] = (
                proposed_observables_vec_first[obs_idx]
            )

        second_status = step_function(
            state_vec,
            proposed_vec_second,
            params_vec,
            driver_coeffs_vec,
            drivers_current_vec,
            proposed_drivers_vec_second,
            observables_current_vec,
            proposed_observables_vec_second,
            error_vec_second,
            dt_scalar,
            dt_scalar,
            int32(0),
            int32(1),
            shared,
            persistent,
            counters_vec_second,
        )
        status_vec[1] = second_status

    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream, shared_bytes](
        d_state,
        d_params,
        d_driver_coeffs,
        d_drivers_current,
        d_proposed_drivers_first,
        d_proposed_drivers_second,
        d_observables_current,
        d_proposed_observables_first,
        d_proposed_observables_second,
        d_proposed_first,
        d_proposed_second,
        d_error_first,
        d_error_second,
        d_status,
        d_counters_first,
        d_counters_second,
        dt_value,
    )
    stream.synchronize()
    status_host = d_status.copy_to_host()

    first_state = d_proposed_first.copy_to_host()
    second_state = d_proposed_second.copy_to_host()

    first_observables = d_proposed_observables_first.copy_to_host()
    second_observables = d_proposed_observables_second.copy_to_host()

    first_error = d_error_first.copy_to_host()
    second_error = d_error_second.copy_to_host()

    statuses = (
        int(status_host[0]) & STATUS_MASK,
        int(status_host[1]) & STATUS_MASK,
    )

    return DualStepResult(
        first_state=first_state,
        second_state=second_state,
        first_observables=first_observables,
        second_observables=second_observables,
        first_error=first_error,
        second_error=second_error,
        statuses=statuses,
    )


def _execute_cpu_step_twice(
    solver_settings,
    step_inputs,
    cpu_system: CPUODESystem,
    cpu_driver_evaluator,
    tableau,
) -> DualStepResult:
    """Run the CPU reference step twice with shared cache reuse."""

    dt = solver_settings["dt"]
    precision = cpu_system.precision

    state = np.asarray(step_inputs["state"], dtype=precision)
    params = np.asarray(step_inputs["parameters"], dtype=precision)

    if cpu_system.system.num_drivers > 0:
        driver_evaluator = cpu_driver_evaluator.with_coefficients(
            step_inputs["driver_coefficients"]
        )
    else:
        driver_evaluator = cpu_driver_evaluator

    stepper = get_ref_stepper(
        cpu_system,
        driver_evaluator,
        solver_settings["algorithm"],
        newton_tol=solver_settings["newton_atol"],
        newton_max_iters=solver_settings["newton_max_iters"],
        linear_tol=solver_settings["krylov_atol"],
        linear_max_iters=solver_settings["krylov_max_iters"],
        linear_correction_type=solver_settings[
            "linear_correction_type"
        ],
        preconditioner_order=solver_settings["preconditioner_order"],
        tableau=tableau,
    )

    first_result = stepper.step(
        state=state,
        params=params,
        dt=dt,
        time=0.0,
    )

    second_result = stepper.step(
        state=first_result.state.astype(precision, copy=True),
        params=params,
        dt=dt,
        time=dt,
    )

    return DualStepResult(
        first_state=first_result.state.astype(
            precision, copy=True
        ),
        second_state=second_result.state.astype(
            precision, copy=True
        ),
        first_observables=first_result.observables.astype(
            precision, copy=True
        ),
        second_observables=second_result.observables.astype(
            precision, copy=True
        ),
        first_error=first_result.error.astype(
            precision, copy=True
        ),
        second_error=second_result.error.astype(
            precision, copy=True
        ),
        statuses=(
            first_result.status & STATUS_MASK,
            second_result.status & STATUS_MASK,
        ),
    )


# ── Two-step device-vs-CPU comparison ─────────────────────── #

def _run_two_step_comparison(
    solver_settings,
    step_object,
    precision,
    step_inputs,
    system,
    driver_array,
    cpu_system,
    cpu_driver_evaluator,
):
    """Compare two device steps against the CPU reference stepper."""

    gpu_result = _execute_step_twice(
        step_object=step_object,
        solver_settings=solver_settings,
        precision=precision,
        step_inputs=step_inputs,
        system=system,
        driver_array=driver_array,
    )

    cpu_result = _execute_cpu_step_twice(
        solver_settings=solver_settings,
        step_inputs=step_inputs,
        cpu_system=cpu_system,
        cpu_driver_evaluator=cpu_driver_evaluator,
        tableau=getattr(step_object, "tableau", None),
    )

    assert all(status == 0 for status in gpu_result.statuses)
    assert all(status == 0 for status in cpu_result.statuses)
    assert gpu_result.statuses == cpu_result.statuses

    tol = {"rtol": 5e-7, "atol": 1e-7}

    assert_allclose(
        gpu_result.first_state,
        cpu_result.first_state,
        **tol,
    )
    assert_allclose(
        gpu_result.second_state,
        cpu_result.second_state,
        **tol,
    )
    assert_allclose(
        gpu_result.first_error,
        cpu_result.first_error,
        **tol,
    )
    assert_allclose(
        gpu_result.second_error,
        cpu_result.second_error,
        **tol,
    )
    assert_allclose(
        gpu_result.first_observables,
        cpu_result.first_observables,
        **tol,
    )
    assert_allclose(
        gpu_result.second_observables,
        cpu_result.second_observables,
        **tol,
    )

    delta = np.abs(
        gpu_result.second_state - gpu_result.first_state
    )
    assert np.any(delta > precision(1e-10))

    # Coarse check against the independent euler reference.
    euler_result = _execute_cpu_step_twice(
        solver_settings=dict(solver_settings, algorithm="euler"),
        step_inputs=step_inputs,
        cpu_system=cpu_system,
        cpu_driver_evaluator=cpu_driver_evaluator,
        tableau=_get_algorithm_tableau("euler"),
    )
    euler_tol = {"rtol": 1e-3, "atol": 1e-3}
    assert_allclose(
        gpu_result.first_state,
        euler_result.first_state,
        **euler_tol,
    )
    assert_allclose(
        gpu_result.second_state,
        euler_result.second_state,
        **euler_tol,
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    ALGORITHM_PARAM_SETS,
    indirect=True,
)
def test_two_steps(
    solver_settings,
    step_object,
    precision,
    step_inputs,
    system,
    driver_array,
    cpu_system,
    cpu_driver_evaluator,
):
    """Ensure shared-cache reuse yields consistent results
    across devices."""

    _run_two_step_comparison(
        solver_settings,
        step_object,
        precision,
        step_inputs,
        system,
        driver_array,
        cpu_system,
        cpu_driver_evaluator,
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    BICGSTAB_STEP_CASES,
    indirect=True,
)
def test_two_steps_bicgstab_jacobi(
    solver_settings,
    step_object,
    precision,
    step_inputs,
    system,
    driver_array,
    cpu_system,
    cpu_driver_evaluator,
):
    """Implicit steps solved with BiCGSTAB / Jacobi match the CPU
    reference through the full step machinery."""

    _run_two_step_comparison(
        solver_settings,
        step_object,
        precision,
        step_inputs,
        system,
        driver_array,
        cpu_system,
        cpu_driver_evaluator,
    )


