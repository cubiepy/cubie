"""Liveness tests for persistently rejecting adaptive runs.

A run whose error estimate can never satisfy the controller
tolerance must walk dt down to ``dt_min`` and terminate with
``STEP_TOO_SMALL`` rather than spin: every rejected step shrinks dt
by at least the controller safety factor.
"""

import numpy as np
import pytest
from tests._utils import (
    IMPOSSIBLE_TOLERANCE,
)


@pytest.mark.parametrize(
    "solver_settings_override", [IMPOSSIBLE_TOLERANCE], indirect=True
)
def test_persistent_rejection_fails_with_step_too_small(
    solver, solver_settings, batch_input_arrays, driver_settings
):
    """Unreachable tolerances terminate quickly with STEP_TOO_SMALL.

    Stage-solve residuals sit far above a 1e-13 tolerance, so every
    step rejects; dt must reach ``dt_min`` and fail decodably.
    """
    initial_values, parameters = batch_input_arrays
    result = solver.solve(
        initial_values=initial_values,
        parameters=parameters,
        drivers=driver_settings,
        duration=float(solver_settings["duration"]),
    )
    status_codes = np.asarray(result.status_codes)
    assert np.all(status_codes != 0)
    for flags in result.status_messages.values():
        assert "STEP_TOO_SMALL" in flags
