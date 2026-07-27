"""Tests for the run status word's transient-failure semantics.

The integration loop accumulates step/controller status bits into an
iteration-scoped temporary that is cleared whenever a step is
accepted, and commits the accumulated bits into the persistent status
word only when the run ends irrecoverably.  A step that fails
transiently (for example an inner linear solve exhausting its
iteration budget) and is recovered at a smaller ``dt`` leaves a clean
status, so a completed run's trajectory survives the default
``nan_error_trajectories=True`` masking, while the flags of a fatal
iteration are preserved for diagnosis.
"""

import numpy as np

import pytest

from cubie import CUBIE_RESULT_CODES


STEP_TOO_SMALL = int(CUBIE_RESULT_CODES.STEP_TOO_SMALL)
MAX_LINEAR = int(CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED)


_RECOVERED_TRANSIENT = {
    "system_type": "staining_stiff",
    "precision": np.float64,
    "algorithm": "rodas3p",
    "step_controller": "pid",
    "duration": 1.0,
    "dt": 1.0,
    "dt_min": 1e-9,
    "dt_max": 1.0,
    "atol": 1e-6,
    "rtol": 1e-3,
    "save_every": 0.1,
    "krylov_max_iters": 2,
    "krylov_residual_reduction": 1e-12,
    "kp": 0.6,
    "ki": -0.4,
    "deadband_min": 1.0,
    "deadband_max": 1.1,
    "min_gain": 0.5,
    "max_gain": 2.0,
    "output_types": ["state", "time"],
    # The stiff two-state system declares no observables; the shared
    # defaults index two of them.
    "saved_observable_indices": [],
    "summarised_observable_indices": [],
}

_IRRECOVERABLE = {
    "system_type": "stiff",
    "precision": np.float64,
    "algorithm": "rodas3p",
    "step_controller": "gustafsson",
    "deadband_min": 1.0,
    "deadband_max": 1.2,
    "min_gain": 0.2,
    "max_gain": 8.0,
    "duration": 1.0,
    "dt": 0.5,
    "dt_min": 0.4,
    "dt_max": 0.5,
    "atol": 1e-13,
    "rtol": 1e-13,
    "save_every": 0.1,
    "output_types": ["state", "time"],
}


@pytest.mark.parametrize(
    "solver_settings_override",
    [_RECOVERED_TRANSIENT],
    indirect=True,
)
def test_recovered_transient_failure_reports_success(
    solver, solver_settings, driver_settings
):
    """A run that fails transiently then recovers ends with status 0.

    ``rodas3p`` with a deliberately large initial ``dt`` and a tight
    ``krylov_max_iters`` budget forces the first Rosenbrock stage's inner
    linear solve to exhaust its iterations (``MAX_LINEAR_ITERATIONS_EXCEEDED``)
    before the adaptive controller reduces ``dt`` and the integration
    proceeds to completion with a finite trajectory.  The delivered status
    must be ``0`` and the trajectory must be returned unmasked under the
    default ``nan_error_trajectories=True``.

    The controller is pinned to a gently clamped PID (0.5--2.0 gain
    limits) rather than the algorithm default: the recovery window
    must stay open long enough for the reduced steps to converge, and
    a springier controller can walk ``dt`` through ``dt_min`` before
    the inner solve first succeeds.

    ``krylov_residual_reduction`` is pinned tight so the starved
    two-iteration budget genuinely fails at the oversized step.
    """
    result = solver.solve(
        initial_values={
            "x0": np.array([1.0], dtype=np.float64),
            "x1": np.array([0.0], dtype=np.float64),
        },
        parameters={"k": np.array([500.0], dtype=np.float64)},
        drivers=driver_settings,
        duration=float(solver_settings["duration"]),
    )

    status_codes = result.status_codes
    assert status_codes is not None
    assert int(status_codes[0]) == 0, (
        f"recovered run reported {result.status_messages}"
    )

    # The full trajectory must survive the default NaN-masking.
    tda = result.time_domain_array
    assert tda.size > 0
    assert np.isfinite(tda).all(), "recovered trajectory was NaN-masked"


@pytest.mark.parametrize(
    "solver_settings_override",
    [_IRRECOVERABLE],
    indirect=True,
)
def test_irrecoverable_failure_preserves_fatal_flags(
    system, solver, solver_settings, driver_settings
):
    """A run driven to ``dt_min`` reports the fatal iteration's flags.

    With ``dt_min`` pinned just below ``dt_max`` and tolerances too tight to
    satisfy, the adaptive controller cannot shrink the step far enough and
    signals ``STEP_TOO_SMALL``, ending the run irrecoverably.  The persistent
    status word must carry ``STEP_TOO_SMALL`` together with the step-status
    bit of the fatal iteration (``MAX_LINEAR_ITERATIONS_EXCEEDED``),
    demonstrating that the accumulated bits are committed on an
    irrecoverable end rather than discarded.
    """
    initial_values = {
        name: np.array([value], dtype=np.float64)
        for name, value in zip(
            system.initial_values.names,
            system.initial_values.values_array,
        )
    }
    result = solver.solve(
        initial_values=initial_values,
        parameters={},
        drivers=driver_settings,
        duration=float(solver_settings["duration"]),
        nan_error_trajectories=False,
    )

    status_codes = result.status_codes
    assert status_codes is not None
    fatal = int(status_codes[0])
    assert fatal != 0, "irrecoverable run reported success"
    assert fatal & STEP_TOO_SMALL == STEP_TOO_SMALL, (
        f"STEP_TOO_SMALL missing from {result.status_messages}"
    )
    assert fatal & MAX_LINEAR == MAX_LINEAR, (
        "fatal iteration's step-status bit was not preserved: "
        f"{result.status_messages}"
    )
