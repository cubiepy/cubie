"""Tests for FIRK dense-prediction state ownership."""

import attrs
import numpy as np

from cubie.integrators.algorithms.generic_firk import FIRKStep
from cubie.integrators.algorithms.generic_firk_tableaus import (
    GAUSS_LEGENDRE_2_TABLEAU,
    RADAU_IIA_5_TABLEAU,
)


def opened(tableau, ceiling=8.0):
    """Return the tableau with sweep-open ratio ceilings."""

    return attrs.evolve(
        tableau,
        dense_prediction_ratio_float32=ceiling,
        dense_prediction_ratio_float64=ceiling,
    )


def test_previous_step_size_owned_by_algorithm():
    """The previous-step-size scalar lives on the FIRK config; the
    predictor has no such setting."""
    step = FIRKStep(
        precision=np.float64,
        n_states=2,
        tableau=opened(RADAU_IIA_5_TABLEAU),
    )
    assert (
        step.compile_settings.previous_step_size_location == "local"
    )
    step.update(previous_step_size_location="shared")
    assert (
        step.compile_settings.previous_step_size_location == "shared"
    )
    assert not hasattr(
        step.dense_predictor.compile_settings,
        "previous_step_size_location",
    )


def test_update_carries_typed_ceiling():
    """Tableau and precision updates leave the step holding the
    matching typed ratio ceiling."""
    step = FIRKStep(
        precision=np.float64,
        n_states=2,
        tableau=opened(RADAU_IIA_5_TABLEAU, ceiling=4.0),
    )
    settings = step.compile_settings
    limit = settings.tableau.dense_prediction_ratio_limit(
        settings.precision
    )
    assert isinstance(limit, np.float64)
    assert float(limit) == 4.0

    step.update(
        tableau=opened(GAUSS_LEGENDRE_2_TABLEAU, ceiling=2.0),
        precision=np.float32,
    )
    settings = step.compile_settings
    limit = settings.tableau.dense_prediction_ratio_limit(
        settings.precision
    )
    assert isinstance(limit, np.float32)
    assert float(limit) == 2.0
    assert step.dense_prediction
