"""Saves land within one ulp of ``min(next_save, t_end)``."""

import numpy as np
import pytest


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


def _solve_times(solver, solver_settings, driver_settings, parameters):
    """Solve and return (times, expected schedule)."""
    inits, params = solver.build_grid(parameters=parameters)
    duration = solver_settings["duration"]
    result = solver.solve(
        inits, params, drivers=driver_settings, duration=duration
    )
    times = np.array(result.time)
    times = times.reshape(times.shape[0], -1)
    precision = solver_settings["precision"]
    expected = _expected_schedule(
        precision,
        solver_settings["save_every"],
        times.shape[0],
        precision(duration),
    )
    return times, expected


# One float32 ulp under half the default save interval of 0.02.
_HALF_INTERVAL_BELOW = float(
    np.nextafter(np.float32(0.01), np.float32(0.0), dtype=np.float32)
)


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"dt": _HALF_INTERVAL_BELOW}],
    ids=["dt_half_interval_below"],
    indirect=True,
)
def test_fixed_step_saves_land_on_schedule(
    solver, solver_settings, system, driver_settings
):
    """A step judged short of a boundary never lands on it."""
    name = list(system.parameters.names)[0]
    parameters = {name: [float(system.parameters.values_dict[name])]}
    times, expected = _solve_times(
        solver, solver_settings, driver_settings, parameters
    )
    _assert_on_schedule(times, expected)


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "system_type": "lorenz_julia",
            "algorithm": "dormand-prince-54",
            "step_controller": "i",
            "integral_gain": 1.2,
            "dt": 3.1622776601683795e-05,
            "dt_min": 1e-12,
            "dt_max": 1e3,
            "min_step_shrink": 0.2,
            "max_step_growth": 10.0,
            "safety": 0.9,
            "deadband_min": 1.0,
            "deadband_max": 1.0,
            "save_every": 0.05,
            "duration": 16.0,
            "output_types": ["state", "time"],
            "summarise_every": None,
            "sample_summaries_every": None,
            "saved_observable_indices": [],
            "summarised_observable_indices": [],
        }
    ],
    ids=["lorenz_dp54_sweep"],
    indirect=True,
)
def test_adaptive_saves_land_on_schedule(
    solver, solver_settings, driver_settings
):
    """Saves stay on schedule when dt exceeds save_every."""
    times, expected = _solve_times(
        solver,
        solver_settings,
        driver_settings,
        {"rho": np.linspace(0.0, 21.0, 4096)},
    )
    _assert_on_schedule(times, expected)
