"""Numerical correctness tests for step controllers.

Compares device controller outputs against CPU reference for single
steps, multi-step sequences, and rejection bookkeeping.
"""

import numpy as np
import pytest

from tests._utils import (
    CONTROLLER_TOLERANCE_SETS,
    StepResult,
    run_controller_device_step,
)
from cubie.integrators.step_control.adaptive_PI_controller import (
    AdaptivePIController,
)
from cubie.integrators.step_control.adaptive_PID_controller import (
    AdaptivePIDController,
)
from cubie.integrators.step_control.gustafsson_controller import (
    GustafssonController,
)


class ControllerTrace:
    """Capture outputs from sequential controller executions."""

    def __init__(self):
        self.accepted = []
        self.dt = []
        self.local_memory = []


def _sequence_inputs(
    *,
    n_states,
    precision,
    base_state,
    low_error,
    high_error,
    n_steps,
):
    """Generate deterministic state and error sequences for tests."""

    states_prev = []
    states_new = []
    errors = []
    niters = []
    for idx in range(n_steps):
        offset = precision(0.05 * idx)
        delta = precision(0.01 + 0.002 * idx)
        state_prev = base_state + precision(offset)
        state_new = state_prev + precision(delta)
        if idx % 2 == 0:
            err_value = precision(low_error)
        else:
            err_value = precision(high_error)
        error_vec = np.full(n_states, err_value, dtype=precision)
        states_prev.append(state_prev)
        states_new.append(state_new)
        errors.append(error_vec)
        niters.append(1 + (idx % 3))
    return states_prev, states_new, errors, niters


# ── Single-step CPU-vs-device comparisons ─────────────────── #
# (extracted from test_controllers.py)


@pytest.fixture(scope='function')
def cpu_step_results(cpu_step_controller, precision, step_setup):
    """CPU analogue for one controller step (niters=1)."""
    controller = cpu_step_controller
    kind = controller.kind.lower()
    controller.dt = step_setup['dt0']
    state = np.asarray(step_setup['state'], dtype=precision)
    state_prev = np.asarray(
        step_setup['state_prev'], dtype=precision
    )
    error_vec = np.asarray(step_setup['error'], dtype=precision)
    provided_local = np.asarray(
        step_setup['local_mem'], dtype=precision
    )

    if kind == 'pi':
        controller._prev_nrm2 = float(provided_local[0])
    elif kind == 'pid':
        controller._prev_nrm2 = float(provided_local[0])
        controller._prev_prev_nrm2 = float(provided_local[1])
    elif kind == 'sciml_pi':
        controller._qold = precision(provided_local[0])
    elif kind == 'gustafsson':
        controller._prev_dt = float(provided_local[0])
        controller._prev_nrm2 = float(provided_local[1])

    accept = controller.propose_dt(
        prev_state=state_prev,
        new_state=state,
        error_vector=error_vec,
        niters=1,
    )

    if kind == 'i':
        out_local = np.zeros(0, dtype=precision)
    elif kind == 'pi':
        out_local = np.array([controller._prev_nrm2], dtype=precision)
    elif kind == 'pid':
        out_local = np.array([
            controller._prev_nrm2,
            controller._prev_prev_nrm2,
        ], dtype=precision)
    elif kind == 'sciml_pi':
        out_local = np.array([controller._qold], dtype=precision)
    elif kind == 'gustafsson':
        out_local = np.array([
            controller._prev_dt,
            controller._prev_nrm2,
        ], dtype=precision)
    else:
        out_local = np.zeros(0, dtype=precision)

    return StepResult(controller.dt, int(accept), out_local)


@pytest.mark.parametrize(
    "solver_settings_override",
    list(CONTROLLER_TOLERANCE_SETS.values()),
    ids=tuple(CONTROLLER_TOLERANCE_SETS),
    indirect=True,
)
class TestControllerNumerical:
    """Numerical CPU-vs-device tests for step controllers."""

    @pytest.mark.parametrize(
        'step_setup',
        (
            {
                'dt0': 0.005,
                'error': np.asarray([5e-4, 5e-4, 5e-4]),
            },
            {
                'dt0': 0.005,
                'error': np.asarray([5e-3, 5e-3, 5e-3]),
            },
            {
                'dt0': 0.005,
                'error': np.asarray([5e-4, 5e-4, 5e-4]),
                'local_mem': np.asarray([0.005, 0.8]),
            },
            {
                'dt0': 0.005,
                'error': np.asarray([5e-3, 5e-3, 5e-3]),
                'local_mem': np.asarray([0.005, 0.8]),
            },
        ),
        ids=(
            "low_err",
            "high_err",
            "low_err_with_mem",
            "high_err_with_mem",
        ),
        indirect=True,
    )
    def test_matches_cpu(
        self,
        step_controller,
        step_controller_settings,
        step_setup,
        cpu_step_results,
        device_step_results,
        tolerance,
    ):
        assert device_step_results.dt == pytest.approx(
            cpu_step_results.dt,
            rel=tolerance.rel_tight,
            abs=tolerance.abs_tight,
        )
        valid_localmem = step_controller.persistent_local_buffer_size
        assert np.allclose(
            device_step_results.local_mem[:valid_localmem],
            cpu_step_results.local_mem[:valid_localmem],
            rtol=tolerance.rel_tight,
            atol=tolerance.abs_tight,
        )

    def test_cpu_gpu_sequence_agree(
        self,
        step_controller,
        cpu_step_controller,
        precision,
        system,
        tolerance,
    ):
        n_states = system.sizes.states
        dtype = precision
        cpu_controller = cpu_step_controller
        current_dt_cpu = float(step_controller.dt)
        current_dt_gpu = float(step_controller.dt)
        local_mem = np.zeros(
            step_controller.persistent_local_buffer_size,
            dtype=dtype,
        )
        base_state = np.linspace(
            0.5,
            0.5 + 0.1 * (n_states - 1),
            n_states,
            dtype=dtype,
        )

        error_values = (
            dtype(0.0),
            dtype(5.0e-4),
            dtype(1.4e-3),
            dtype(6.0e-4),
            dtype(1.6e-3),
        )
        delta_values = (
            dtype(2.0e-2),
            dtype(1.5e-2),
            dtype(1.0e-2),
            dtype(2.5e-2),
            dtype(1.5e-2),
        )
        niters_values = (1, 2, 1, 3, 2)

        current_state = base_state.copy()

        for i, (error_val, delta_val, niters) in enumerate(zip(
            error_values,
            delta_values,
            niters_values,
        )):
            state_prev = current_state.copy()
            state = state_prev + np.full(
                n_states, delta_val, dtype=dtype
            )
            error_vec = np.full(
                n_states, error_val, dtype=dtype
            )

            cpu_controller.dt = dtype(current_dt_cpu)
            accept_cpu = cpu_controller.propose_dt(
                prev_state=state_prev,
                new_state=state,
                error_vector=error_vec,
                niters=niters,
            )
            dt_cpu = float(cpu_controller.dt)

            device_result = run_controller_device_step(
                step_controller.device_function,
                dtype,
                dtype(current_dt_gpu),
                error_vec,
                state=state,
                state_prev=state_prev,
                local_mem=local_mem,
                niters=niters,
            )
            local_mem = device_result.local_mem.copy()
            current_dt_gpu = device_result.dt

            assert device_result.accepted == int(accept_cpu), (
                f"Step {i} accept mismatch"
            )
            assert current_dt_gpu == pytest.approx(
                dt_cpu,
                rel=tolerance.rel_tight,
                abs=tolerance.abs_tight,
            ), f"Step {i} dt mismatch"

            if accept_cpu:
                current_state = state

            current_dt_cpu = dt_cpu


@pytest.mark.parametrize(
    "solver_settings_override",
    # Unique set: the order-wiring assertion needs an algorithm whose
    # order exceeds the default euler's 1 on the shared pi tolerances.
    [{**CONTROLLER_TOLERANCE_SETS["pi"], "algorithm": "crank_nicolson"}],
    indirect=True,
)
def test_pi_controller_uses_tableau_order(
    step_controller,
    cpu_step_controller,
    step_controller_settings,
    step_object,
    device_step_results,
    cpu_step_results,
    tolerance,
):
    """Adaptive controller gains should follow the algorithm
    tableau order."""

    expected_order = step_object.order
    assert expected_order > 1
    assert step_controller_settings["algorithm_order"] == (
        expected_order
    )
    assert step_controller.algorithm_order == expected_order
    assert device_step_results.dt == pytest.approx(
        cpu_step_results.dt,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )


# ── Multi-step equivalence sequences ─────────────────────── #
# (moved from test_controller_equivalence_sequences.py)


@pytest.mark.parametrize(
    "solver_settings_override",
    list(CONTROLLER_TOLERANCE_SETS.values()),
    ids=tuple(CONTROLLER_TOLERANCE_SETS),
    indirect=True,
)
class TestControllerEquivalence:
    """Step controller regression tests for CPU and device."""

    def test_sequential_acceptance_matches(
        self,
        step_controller,
        cpu_step_controller,
        precision,
        system,
    ):
        """Check dt updates and acceptance match through
        rejections."""

        dtype = precision
        n_states = system.sizes.states
        local_mem = np.zeros(
            step_controller.persistent_local_buffer_size,
            dtype=dtype,
        )
        base_state = (
            system.initial_values.values_array.astype(dtype)
        )
        low_error = dtype(1e-5)
        high_error = dtype(5.0)
        sequence = _sequence_inputs(
            n_states=n_states,
            precision=dtype,
            base_state=base_state,
            low_error=low_error,
            high_error=high_error,
            n_steps=6,
        )
        states_prev, states_new, errors, niters = sequence
        gpu_trace = ControllerTrace()
        cpu_trace = ControllerTrace()

        current_dt_device = dtype(step_controller.dt)
        current_dt_cpu = dtype(step_controller.dt)
        # The device trace starts from a zeroed persistent buffer, so
        # the CPU reference must start from cleared history too.
        cpu_step_controller._prev_dt = dtype(0)
        cpu_step_controller._prev_nrm2 = dtype(0)
        cpu_step_controller._prev_prev_nrm2 = dtype(0)
        cpu_step_controller.dt = dtype(current_dt_cpu)

        for idx, (prev_state, new_state, err_vec, niter) in (
            enumerate(zip(
                states_prev, states_new, errors, niters
            ))
        ):
            cpu_step_controller.dt = dtype(current_dt_cpu)
            accept_cpu = cpu_step_controller.propose_dt(
                error_vector=err_vec,
                prev_state=prev_state,
                new_state=new_state,
                niters=niter,
            )
            result_cpu = StepResult(
                precision(cpu_step_controller.dt),
                int(accept_cpu),
                np.array([], dtype=dtype),
            )
            cpu_trace.dt.append(result_cpu.dt)
            cpu_trace.accepted.append(result_cpu.accepted)
            if step_controller.persistent_local_buffer_size:
                if isinstance(
                    step_controller, AdaptivePIController
                ):
                    mem_cpu = np.array(
                        [cpu_step_controller._prev_nrm2],
                        dtype=dtype,
                    )
                elif isinstance(
                    step_controller, AdaptivePIDController
                ):
                    mem_cpu = np.array(
                        [
                            cpu_step_controller._prev_nrm2,
                            cpu_step_controller._prev_prev_nrm2,
                        ],
                        dtype=dtype,
                    )
                elif isinstance(
                    step_controller, GustafssonController
                ):
                    prev_norm = cpu_step_controller._prev_nrm2
                    mem_cpu = np.array(
                        [
                            cpu_step_controller.prev_dt,
                            prev_norm,
                        ],
                        dtype=dtype,
                    )
                else:
                    mem_cpu = np.zeros(0, dtype=dtype)
            else:
                mem_cpu = np.zeros(0, dtype=dtype)
            cpu_trace.local_memory.append(mem_cpu)

            device_result = run_controller_device_step(
                step_controller.device_function,
                dtype,
                dtype(current_dt_device),
                err_vec,
                state=new_state,
                state_prev=prev_state,
                local_mem=local_mem,
                niters=niter,
            )
            gpu_trace.dt.append(device_result.dt)
            gpu_trace.accepted.append(device_result.accepted)
            gpu_trace.local_memory.append(
                device_result.local_mem.copy()
            )
            if step_controller.persistent_local_buffer_size:
                local_mem = device_result.local_mem.copy()
            current_dt_device = device_result.dt
            current_dt_cpu = result_cpu.dt

        assert gpu_trace.accepted == cpu_trace.accepted
        assert np.allclose(
            np.array(gpu_trace.dt),
            np.array(cpu_trace.dt),
            rtol=1e-7,
            atol=1e-7,
        )
        if step_controller.persistent_local_buffer_size:
            for i, (gpu_mem, cpu_mem) in enumerate(zip(
                gpu_trace.local_memory,
                cpu_trace.local_memory,
            )):
                np.testing.assert_allclose(
                    gpu_mem[: cpu_mem.size],
                    cpu_mem,
                    rtol=1e-7,
                    atol=1e-7,
                    err_msg=(
                        f"local memory mismatch at step {i}"
                    ),
                )

    def test_rejection_retains_previous_state(
        self,
        step_controller_mutable,
        cpu_step_controller,
        precision,
        system,
    ):
        """Ensure both controllers agree on rejection
        bookkeeping."""

        dtype = precision
        n_states = system.sizes.states
        prev_state = (
            system.initial_values.values_array.astype(dtype)
        )
        new_state = prev_state + dtype(0.02)
        error_vec = np.full(n_states, dtype(10.0), dtype=dtype)
        local_mem = np.zeros(
            step_controller_mutable.persistent_local_buffer_size,
            dtype=dtype,
        )
        cpu_step_controller._prev_dt = dtype(0)
        cpu_step_controller._prev_nrm2 = dtype(0)
        cpu_step_controller._prev_prev_nrm2 = dtype(0)
        cpu_step_controller.dt = dtype(
            step_controller_mutable.dt
        )
        accept_cpu = cpu_step_controller.propose_dt(
            error_vector=error_vec,
            prev_state=prev_state,
            new_state=new_state,
            niters=3,
        )
        device_result = run_controller_device_step(
            step_controller_mutable.device_function,
            dtype,
            dtype(step_controller_mutable.dt),
            error_vec,
            state=new_state,
            state_prev=prev_state,
            local_mem=local_mem,
            niters=3,
        )

        assert int(accept_cpu) == device_result.accepted
        assert device_result.dt == pytest.approx(
            dtype(cpu_step_controller.dt),
            rel=1e-7,
            abs=1e-7,
        )
