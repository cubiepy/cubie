"""Per-step iteration counters sum over every solve in the step."""

import numpy as np
import pytest

from cubie.integrators.algorithms.generic_firk import FIRKStep

# Constant derivatives with an exactly representable fixed step: every
# stage increment is the same dt * c, so from the second solve onwards
# the carried guess is bit-exact, its correction is zero and the Newton
# loop exits after one iteration. The first solve ever starts from a
# zero guess and takes two. The LU solver reports one Krylov iteration
# per call, so the Krylov count is the number of linear solves.
CONSTANT_DERIVATIVE_LU = {
    "system_type": "constant_deriv",
    "output_types": ["state", "iteration_counters"],
    "saved_state_indices": [0, 1, 2],
    "saved_observable_indices": [],
    "summarised_state_indices": [],
    "summarised_observable_indices": [],
    "summarise_every": None,
    "sample_summaries_every": None,
    "linear_correction_type": "lu",
    "attempt_dense_prediction": False,
    "step_controller": "fixed",
    "dt": 0.03125,
    "duration": 0.125,
    "save_every": 0.125,
}
FIRST_SOLVE_EXTRA_ITERATIONS = 1

COUNTER_CASES = [
    pytest.param(
        {**CONSTANT_DERIVATIVE_LU, "algorithm": "kvaerno3"},
        id="dirk-kvaerno3",
    ),
    pytest.param(
        {**CONSTANT_DERIVATIVE_LU, "algorithm": "radau_iia_3"},
        id="firk-radau_iia_3",
    ),
    pytest.param(
        {**CONSTANT_DERIVATIVE_LU, "algorithm": "rosenbrock23"},
        id="ros-rosenbrock23",
    ),
]


def _solves_per_step(step):
    """Newton solves and linear solves one step performs."""
    tableau = step.tableau
    smoothed = 1 if step.smooth_error else 0
    if step.is_linear:
        return 0, tableau.stage_count + smoothed
    if isinstance(step, FIRKStep):
        newton_solves = 1
    else:
        newton_solves = sum(
            1 for i in range(tableau.stage_count) if tableau.a[i][i] != 0.0
        )
    return newton_solves, newton_solves + smoothed


@pytest.mark.parametrize(
    "solver_settings_override", COUNTER_CASES, indirect=True,
)
def test_counters_sum_over_every_solve_in_the_step(
    solver, system, precision
):
    inits = np.array(system.initial_values.values_array, dtype=precision)
    params = np.array(system.parameters.values_array, dtype=precision)
    result = solver.solve(
        inits[:, None],
        params[:, None],
        grid_type="verbatim",
        duration=CONSTANT_DERIVATIVE_LU["duration"],
        save_every=CONSTANT_DERIVATIVE_LU["save_every"],
    )
    assert not np.any(result.status_codes)
    counters = np.asarray(result.iteration_counters)[:, :, 0]
    newton, krylov, attempted = counters[:, 0], counters[:, 1], counters[:, 2]
    steps = int(
        CONSTANT_DERIVATIVE_LU["duration"] / CONSTANT_DERIVATIVE_LU["dt"]
    )
    np.testing.assert_array_equal(attempted, [0, steps])

    step = solver.kernel.single_integrator._algo_step
    newton_solves, linear_solves = _solves_per_step(step)
    extra = FIRST_SOLVE_EXTRA_ITERATIONS if newton_solves else 0
    np.testing.assert_array_equal(
        newton, [0, newton_solves * steps + extra]
    )
    np.testing.assert_array_equal(
        krylov, [0, linear_solves * steps + extra]
    )
