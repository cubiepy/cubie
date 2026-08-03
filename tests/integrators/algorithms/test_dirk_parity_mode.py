"""Device-vs-CPU tests for smooth_error and the residual Newton stop."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests._utils import LORENZ_DIRK, run_device_step_schedule
from tests.integrators.cpu_reference.algorithms import CPUDIRKStep

KVAERNO3_SMOOTH = {
    **LORENZ_DIRK,
    "algorithm": "kvaerno3",
    "smooth_error": True,
}

KVAERNO3_RESIDUAL = {
    **LORENZ_DIRK,
    "algorithm": "kvaerno3",
    "newton_stop_criterion": "residual",
    "newton_residual_atol": 1e-8,
    "newton_max_iters": 30,
}


def _run_cpu_schedule(cpu_step, precision, state, params, schedule):
    """Advance the CPU reference through an all-accepted schedule."""
    cpu_state = np.asarray(state, dtype=precision).copy()
    time_value = 0.0
    result = None
    for dt_value in schedule:
        result = cpu_step.step(
            state=cpu_state,
            params=params,
            dt=float(dt_value),
            time=time_value,
            prev_accepted=True,
        )
        cpu_state = np.asarray(result.state, dtype=precision)
        time_value += float(dt_value)
    return cpu_state, result


@pytest.mark.parametrize(
    "solver_settings_override", [KVAERNO3_SMOOTH], indirect=True
)
def test_smooth_error_matches_cpu_reference(
    step_object,
    system,
    precision,
    solver_settings,
    initial_state,
    cpu_system,
    cpu_driver_evaluator,
):
    """Smoothed error matches the CPU reference; state is untouched."""
    assert step_object.smooth_error is True
    params = np.asarray(system.parameters.values_array, dtype=precision)
    state = np.asarray(initial_state, dtype=precision)
    base_dt = precision(0.005)
    schedule = [base_dt] * 3

    device_state, _, device_error = run_device_step_schedule(
        step_object, system, precision, state, params, schedule,
        return_error=True,
    )

    cpu_kwargs = dict(
        newton_tol=float(solver_settings["newton_atol"]),
        newton_rtol=float(solver_settings["newton_rtol"]),
        newton_max_iters=int(solver_settings["newton_max_iters"]),
        linear_tol=float(solver_settings["krylov_atol"]),
        linear_rtol=float(solver_settings["krylov_rtol"]),
        linear_max_iters=int(solver_settings["krylov_max_iters"]),
        tableau=step_object.tableau,
    )
    cpu_smooth = CPUDIRKStep(
        cpu_system, cpu_driver_evaluator, smooth_error=True, **cpu_kwargs
    )
    cpu_raw = CPUDIRKStep(
        cpu_system, cpu_driver_evaluator, smooth_error=False, **cpu_kwargs
    )
    smooth_state, smooth_result = _run_cpu_schedule(
        cpu_smooth, precision, state, params, schedule
    )
    raw_state, raw_result = _run_cpu_schedule(
        cpu_raw, precision, state, params, schedule
    )

    assert_allclose(device_state, smooth_state, rtol=1e-6, atol=1e-9)
    assert_allclose(
        device_error, smooth_result.error, rtol=1e-5, atol=1e-12
    )
    # Smoothing changed the estimate but never the state.
    assert_allclose(smooth_state, raw_state, rtol=1e-12, atol=1e-14)
    assert not np.allclose(
        smooth_result.error, raw_result.error, rtol=1e-3, atol=1e-14
    )


@pytest.mark.parametrize(
    "solver_settings_override", [KVAERNO3_RESIDUAL], indirect=True
)
def test_residual_stop_matches_cpu_reference(
    step_object,
    system,
    precision,
    solver_settings,
    initial_state,
    cpu_system,
    cpu_driver_evaluator,
):
    """The residual-stop Newton mode agrees between device and CPU."""
    assert (
        step_object.solver.compile_settings.stop_criterion == "residual"
    )
    params = np.asarray(system.parameters.values_array, dtype=precision)
    state = np.asarray(initial_state, dtype=precision)
    base_dt = precision(0.005)
    schedule = [base_dt] * 3

    device_state, device_iters = run_device_step_schedule(
        step_object, system, precision, state, params, schedule,
    )

    cpu_step = CPUDIRKStep(
        cpu_system,
        cpu_driver_evaluator,
        newton_tol=float(solver_settings["newton_atol"]),
        newton_rtol=float(solver_settings["newton_rtol"]),
        newton_max_iters=int(solver_settings["newton_max_iters"]),
        linear_tol=float(solver_settings["krylov_atol"]),
        linear_rtol=float(solver_settings["krylov_rtol"]),
        linear_max_iters=int(solver_settings["krylov_max_iters"]),
        tableau=step_object.tableau,
        newton_stop_criterion="residual",
        newton_residual_atol=float(
            solver_settings["newton_residual_atol"]
        ),
    )
    cpu_state, _ = _run_cpu_schedule(
        cpu_step, precision, state, params, schedule
    )

    assert device_iters > 0
    assert_allclose(device_state, cpu_state, rtol=1e-6, atol=1e-9)
