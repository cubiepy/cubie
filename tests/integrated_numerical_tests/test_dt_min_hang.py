"""Save-schedule tests for float32 rounding effects.

The device schedule accumulates ``next_save += save_every`` in the
run precision and caps the result at the end time. When ``save_every``
is not exactly representable in float32 (0.1, for example), each
addition lands slightly off the exact grid, so the scheduled save
times fall slightly before or after the times the user asked for.
These tests check that every save lands on that accumulated schedule
and that rounding in either direction neither hangs the loop nor
changes how many samples are saved:

- a save is due when the landing of the unclamped step,
  ``narrow(t + dt)`` from the float64 time, reaches the scheduled
  time; a float32 estimate of the landing could judge a step short of
  a boundary the step then reached, after which the schedule ran a
  full step late for the rest of the run;
- a due step is shortened to the boundary and lands on it exactly, so
  the schedule can never fall behind the committed time;
- every allocated output row is written, in increasing time order,
  whether the schedule reaches the final save slightly after the end
  time or the duration/save_every division rounds just below a whole
  number: the host allocation and the device's event count come from
  the same arithmetic, so they cannot disagree.
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
    """A long float32 schedule completes with every save on it.

    After about 80 additions of float32(0.1), the accumulated save
    schedule sits about 5 microseconds off the exact grid. k=3.0 with
    radau pushes the adaptive solver into small steps near the save
    boundaries around t=8; the run must still complete with a full
    set of saves, each landing on the accumulated schedule.
    """
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
    """Every allocated save row is written on an inexact float32 grid.

    Both parameter sets request ten regular saves using values that
    are not exactly representable in float32. In the first, the
    accumulated device schedule reaches the final save slightly
    after the end time. In the second, dividing duration by
    save_every in float32 gives 9.9999993 rather than 10. Either
    way the host must allocate eleven rows (the initial state plus
    ten saves) and the device must fill all of them, each on the
    accumulated schedule capped at the requested duration.
    """
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
    """Fixed steps judged short of a boundary never land past it.

    On the default chain (euler, dt 0.01, save_every 0.02) the third
    save falls where the float32 estimate ``t_prec + dt`` sits one
    ulp short of the scheduled 0.06 while the float64 step lands on
    it; judged on the landing, the step is shortened and the save
    fires at 0.06 rather than a full step later.
    """
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
    """Adaptive steps sized like the save interval stay on schedule.

    With ``dt_max`` equal to ``save_every`` the controller saturates
    at the interval, so the unclamped landing of most steps sits
    within rounding of the next save and a float32 estimate judges
    some of them short; the persistent one-step lag that follows shows
    in a few dozen of the swept rho values.
    """
    inits, params = solver.build_grid(
        parameters={"rho": np.linspace(0.0, 21.0, 1024)}
    )
    result = solver.solve(
        inits, params, duration=float(solver_settings["duration"])
    )
    assert np.all(np.asarray(result.status_codes) == 0), (
        result.status_messages
    )
    times, expected = _saved_times(result, solver_settings)
    _assert_on_schedule(times, expected)
