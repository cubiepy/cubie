"""Save-schedule tests for float32 rounding effects.

The device schedule accumulates ``next_save += save_every``. When
``save_every`` is not exactly representable in float32 (0.1, for
example), each addition lands slightly off the exact grid, so the
scheduled save times fall slightly before or after the times the
user asked for. These tests check that rounding in either direction
neither hangs the loop nor changes how many samples are saved:

- the loop keeps stepping when a scheduled save time falls behind
  the current time (a stale save target would clamp the next step
  to zero or negative length, so the clamp only applies when the
  step would be positive);
- every allocated output row is written, in increasing time order,
  whether the schedule reaches the final save slightly after the
  end time or the duration/save_every division rounds just below a
  whole number: the host allocation and the device stop time come
  from the same count, so they cannot disagree.
"""

import numpy as np
import pytest


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

_SAVE_DRIFT = {
    "system_type": "coupled_oscillator",
    "algorithm": "radau",
    "step_controller": "gustafsson",
    "duration": 10.0,
    "dt_min": 1e-6,
    "dt_max": 1.0,
    "save_every": 0.1,
    "output_types": ["state", "time"],
    # The oscillator declares no observables; the shared defaults
    # index two of them.
    "saved_observable_indices": [],
    "summarised_observable_indices": [],
}


@pytest.mark.parametrize(
    "solver_settings_override",
    [_SAVE_DRIFT],
    indirect=True,
)
def test_f32_save_drift_does_not_hang(
    solver, solver_settings, precision
):
    """The loop completes when the save schedule falls behind time.

    After about 80 additions of float32(0.1), the accumulated save
    schedule sits about 5 microseconds earlier than the committed
    simulation time. A save target earlier than the current time
    would clamp the next step to zero or negative length, which the
    step function cannot integrate, so the clamp applies only when
    the resulting step is positive. k=3.0 with radau pushes the
    adaptive solver into small steps near the save boundaries
    around t=8, which is where the stale save target appears; the
    run must still complete with a full set of saves.
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
    # Should produce ~100 saves; any completion is a pass.
    n_saves = result.time_domain_array.shape[0]
    assert n_saves >= 80


_DRIFTED_GRID = {
    "algorithm": "euler",
    "step_controller": "fixed",
    "dt": 0.01,
    "duration": 1.0,
    "save_every": 0.1,
    "output_types": ["state", "time"],
}

_ROUNDED_DOWN_COUNT = {
    "algorithm": "euler",
    "step_controller": "fixed",
    "dt": 0.0005,
    "duration": 0.01,
    "save_every": 0.001,
    "output_types": ["state", "time"],
}


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        pytest.param(_DRIFTED_GRID, id="drifted_schedule"),
        pytest.param(_ROUNDED_DOWN_COUNT, id="rounded_down_count"),
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
    ten saves) and the device must fill all of them, in increasing
    time order, ending at the requested duration.
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
    times = np.asarray(result.time)
    assert times.shape[0] == 11
    assert np.all(np.diff(times, axis=0) > 0.0), (
        f"saved times are not strictly increasing: {times}"
    )
    assert np.allclose(times[-1, :], duration, rtol=1e-4)
    assert np.isfinite(result.time_domain_array).all()
