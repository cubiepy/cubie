"""Tests for DIRK dense-prediction ownership and guess sourcing."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DIRK_TABLEAU_REGISTRY,
    DIRKTableau,
    IMPLICIT_MIDPOINT_TABLEAU,
    KVAERNO3_TABLEAU,
    L_STABLE_DIRK3_TABLEAU,
)
from tests._utils import run_device_step_schedule
from tests.integrators.cpu_reference.algorithms import CPUDIRKStep
from tests._utils import LORENZ_DIRK


NON_ADJACENT_REPEAT_TABLEAU = DIRKTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0),
        (0.25, 0.25, 0.0, 0.0),
        (0.25, 0.5, 0.25, 0.0),
        (0.125, 0.125, 0.0, 0.25),
    ),
    b=(0.25, 0.25, 0.25, 0.25),
    c=(0.0, 0.5, 1.0, 0.5),
    order=2,
    dense_prediction_ratio_float32=4.0,
    dense_prediction_ratio_float64=4.0,
)


@pytest.mark.parametrize(
    "solver_settings_override", [LORENZ_DIRK], indirect=True
)
@pytest.mark.parametrize(
    "tableau_name",
    ["eldirk32_euler", "eldirk32_trapezoidal", "eldirk32_ellsiepen"],
)
def test_eldirk_tableaus_match_cpu_reference(
    tableau_name,
    step_object_mutable,
    system,
    precision,
    solver_settings,
    initial_state,
    cpu_system,
    cpu_driver_evaluator,
):
    """Trailing explicit stages integrate identically on device and CPU."""
    tableau = DIRK_TABLEAU_REGISTRY[tableau_name]
    step_object_mutable.update(tableau=tableau)
    assert step_object_mutable.tableau.explicit_last_stage
    params = np.asarray(
        system.parameters.values_array, dtype=precision
    )
    state = np.asarray(initial_state, dtype=precision)
    base_dt = precision(0.005)
    schedule = [base_dt, base_dt, precision(1.5) * base_dt]
    device_state, _ = run_device_step_schedule(
        step_object_mutable, system, precision, state, params,
        schedule,
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
        tableau=tableau,
    )
    cpu_state = state.copy()
    time_value = 0.0
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

    assert_allclose(device_state, cpu_state, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "solver_settings_override", [LORENZ_DIRK], indirect=True
)
def test_update_rederives_predict_first_stage(step_object_mutable):
    """Tableau updates in both directions refresh the predictor's
    first-stage row through the single update path."""
    predictor = step_object_mutable.dense_predictor
    assert predictor.compile_settings.predict_first_stage is True
    step_object_mutable.update(tableau=KVAERNO3_TABLEAU)
    assert predictor.compile_settings.predict_first_stage is False
    assert (
        predictor.compile_settings.stage_count
        == KVAERNO3_TABLEAU.stage_count
    )
    step_object_mutable.update(tableau=L_STABLE_DIRK3_TABLEAU)
    assert predictor.compile_settings.predict_first_stage is True
    assert (
        step_object_mutable.compile_settings.tableau.stage_count
        == L_STABLE_DIRK3_TABLEAU.stage_count
    )


@pytest.mark.parametrize(
    "solver_settings_override", [LORENZ_DIRK], indirect=True
)
def test_previous_step_size_owned_by_algorithm(step_object_mutable):
    """The previous-step-size scalar lives on the DIRK config; the
    predictor has no such setting."""
    settings = step_object_mutable.compile_settings
    assert settings.previous_step_size_location == "local"
    step_object_mutable.update(previous_step_size_location="shared")
    settings = step_object_mutable.compile_settings
    assert settings.previous_step_size_location == "shared"
    assert not hasattr(
        step_object_mutable.dense_predictor.compile_settings,
        "previous_step_size_location",
    )


@pytest.mark.parametrize(
    "solver_settings_override", [LORENZ_DIRK], indirect=True
)
def test_single_stage_midpoint_predicts_its_stage(step_object_mutable):
    """Implicit midpoint's sole implicit stage keeps its predicted
    row."""
    step_object_mutable.update(tableau=IMPLICIT_MIDPOINT_TABLEAU)
    assert step_object_mutable.dense_prediction
    predictor_settings = (
        step_object_mutable.dense_predictor.compile_settings
    )
    assert predictor_settings.predict_first_stage is True


@pytest.mark.parametrize(
    "solver_settings_override", [LORENZ_DIRK], indirect=True
)
def test_non_adjacent_repeat_matches_cpu_reference(
    step_object_mutable,
    system,
    precision,
    solver_settings,
    initial_state,
    cpu_system,
    cpu_driver_evaluator,
):
    """A non-adjacent repeated node integrates identically on device
    and CPU reference.

    The repeated source (stage 1), the target's own predicted row
    (stage 3), and the intervening live increment (stage 2) all
    differ, so an adjacent-only carry would diverge even though each
    solve still converges.
    """
    step_object_mutable.update(tableau=NON_ADJACENT_REPEAT_TABLEAU)
    assert step_object_mutable.dense_prediction
    params = np.asarray(
        system.parameters.values_array, dtype=precision
    )
    state = np.asarray(initial_state, dtype=precision)
    base_dt = precision(0.005)
    schedule = [
        base_dt,
        base_dt,
        precision(1.5) * base_dt,
        precision(0.75) * base_dt,
    ]
    device_state, _ = run_device_step_schedule(
        step_object_mutable, system, precision, state, params,
        schedule,
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
        tableau=NON_ADJACENT_REPEAT_TABLEAU,
    )
    cpu_state = state.copy()
    time_value = 0.0
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

    assert_allclose(device_state, cpu_state, rtol=1e-6, atol=1e-9)
