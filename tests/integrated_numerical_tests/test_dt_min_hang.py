"""Save-schedule tests for float32 rounding effects.

Every save must land within one ulp of the schedule the device
accumulates in the run precision, capped at the end time, and every
allocated output row must be written.
"""

import numpy as np
import pytest
from tests._utils import (
    DRIFTED_GRID,
    ROUNDED_DOWN_COUNT,
    SAVE_DRIFT,
    STEP_SIZED_SAVES,
)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _expected_schedule(precision, save_every, n_saves, t_end):
    """Accumulate the save schedule as the loop does, capped at t_end."""
    schedule = [precision(0.0)]
    for _ in range(n_saves - 1):
        schedule.append(precision(schedule[-1] + precision(save_every)))
    return np.minimum(np.array(schedule, dtype=precision), t_end)


def _assert_on_schedule(times, expected):
    """Every save time sits within one ulp of its scheduled time."""
    off = np.abs(times - expected[:, None])
    tolerance = np.spacing(expected)[:, None]
    late = np.argwhere(off > tolerance)
    assert late.size == 0, (
        f"{late.shape[0]} saves off schedule; first at row {late[0][0]} "
        f"run {late[0][1]}: recorded {times[late[0][0], late[0][1]]!r}, "
        f"scheduled {expected[late[0][0]]!r}"
    )


def _saved_times(result, solver_settings):
    """Return the saved times and the schedule they must land on."""
    times = np.asarray(result.time)
    times = times.reshape(times.shape[0], -1)
    precision = solver_settings["precision"]
    expected = _expected_schedule(
        precision,
        solver_settings["save_every"],
        times.shape[0],
        precision(solver_settings["duration"]),
    )
    return times, expected


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "solver_settings_override",
    [SAVE_DRIFT],
    indirect=True,
)
def test_f32_save_drift_does_not_hang(
    solver, solver_settings, precision
):
    """A 100-save float32 schedule completes with every save on it."""
    n = 1
    result = solver.solve(
        initial_values={
            "x1": np.ones(n, dtype=precision),
            "v1": np.zeros(n, dtype=precision),
            "x2": np.full(n, -0.5, dtype=precision),
            "v2": np.zeros(n, dtype=precision),
        },
        parameters={
            "k": np.full(n, 3.0, dtype=precision),
            "c_couple": np.full(n, 0.3, dtype=precision),
            "omega": np.full(n, 2.5, dtype=precision),
        },
        duration=float(solver_settings["duration"]),
    )
    assert np.all(np.asarray(result.status_codes) == 0), (
        result.status_messages
    )
    times, expected = _saved_times(result, solver_settings)
    assert times.shape[0] == 101
    _assert_on_schedule(times, expected)


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        pytest.param(DRIFTED_GRID, id="drifted_schedule"),
        pytest.param(ROUNDED_DOWN_COUNT, id="rounded_down_count"),
    ],
    indirect=True,
)
def test_all_save_slots_written_on_inexact_grid(
    solver, solver_settings, batch_input_arrays, driver_settings
):
    """Ten saves on an inexact float32 grid fill eleven rows on schedule."""
    initial_values, parameters = batch_input_arrays
    duration = float(solver_settings["duration"])
    result = solver.solve(
        initial_values=initial_values,
        parameters=parameters,
        drivers=driver_settings,
        duration=duration,
    )

    status_codes = np.asarray(result.status_codes)
    assert np.all(status_codes == 0), result.status_messages
    times, expected = _saved_times(result, solver_settings)
    assert times.shape[0] == 11
    assert np.all(np.diff(times, axis=0) > 0.0), (
        f"saved times are not strictly increasing: {times}"
    )
    _assert_on_schedule(times, expected)
    assert np.isfinite(result.time_domain_array).all()


def test_fixed_step_saves_land_on_schedule(
    solver, solver_settings, batch_input_arrays, driver_settings
):
    """Fixed steps of half the save interval land every save on schedule."""
    initial_values, parameters = batch_input_arrays
    result = solver.solve(
        initial_values=initial_values,
        parameters=parameters,
        drivers=driver_settings,
        duration=float(solver_settings["duration"]),
    )
    assert np.all(np.asarray(result.status_codes) == 0), (
        result.status_messages
    )
    times, expected = _saved_times(result, solver_settings)
    _assert_on_schedule(times, expected)


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override",
    [STEP_SIZED_SAVES],
    indirect=True,
)
def test_adaptive_saves_land_on_schedule(solver, solver_settings):
    """Adaptive steps capped at the save interval stay on schedule."""
    # rho values whose steps end within rounding of a save boundary.
    rhos = [
        0.7184750733137829,
        1.129032258064516,
        1.375366568914956,
        1.6011730205278591,
        1.7243401759530792,
        1.909090909090909,
        2.196480938416422,
        2.2375366568914954,
    ]
    inits, params = solver.build_grid(parameters={"rho": rhos})
    result = solver.solve(
        inits, params, duration=float(solver_settings["duration"])
    )
    assert np.all(np.asarray(result.status_codes) == 0), (
        result.status_messages
    )
    times, expected = _saved_times(result, solver_settings)
    _assert_on_schedule(times, expected)
