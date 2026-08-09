"""Tests for the filtered (smoothed) embedded error estimate."""

import numpy as np
import pytest

from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.crank_nicolson import CrankNicolsonStep
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    KVAERNO3_TABLEAU,
)
from cubie.integrators.algorithms.generic_firk import FIRKStep
from cubie.integrators.algorithms.generic_firk_tableaus import (
    GAUSS_LEGENDRE_2_TABLEAU,
    RADAU_IIA_5_TABLEAU,
)

# gamma0 and DD from Hairer & Wanner's radau5.f.
RADAU5_GAMMA0 = 0.27488882959567734
RADAU5_DD = (
    -(13.0 + 7.0 * np.sqrt(6.0)) / 3.0,
    (-13.0 + 7.0 * np.sqrt(6.0)) / 3.0,
    -1.0 / 3.0,
)


def test_radau_smoothed_weights_match_radau5():
    """The derived estimator reproduces Hairer & Wanner's RADAU5."""

    tableau = RADAU_IIA_5_TABLEAU
    assert tableau.smoothing_gamma == pytest.approx(RADAU5_GAMMA0)

    expected = -RADAU5_GAMMA0 * (
        np.asarray(RADAU5_DD) @ np.asarray(tableau.a)
    )
    weights = np.asarray(tableau.smoothed_error_weights(np.float64))
    assert weights == pytest.approx(expected, abs=1e-15)

    # The f(y_n) weight is -gamma, so the estimator is consistent.
    assert weights.sum() == pytest.approx(RADAU5_GAMMA0)


def test_gauss_legendre_has_no_smoothing_operator():
    """A tableau without a sole real eigenvalue of inv(a) opts out."""

    assert not GAUSS_LEGENDRE_2_TABLEAU.supports_smoothed_error
    assert RADAU_IIA_5_TABLEAU.supports_smoothed_error


def test_dirk_smoothing_gamma_is_the_last_diagonal():
    """DIRK smooths against the matrix its last stage already solves."""

    assert KVAERNO3_TABLEAU.supports_smoothed_error
    assert KVAERNO3_TABLEAU.smoothing_gamma == pytest.approx(
        KVAERNO3_TABLEAU.a[-1][-1]
    )


def test_unsupported_request_warns_and_stays_off():
    """A step without tableau support warns and leaves smoothing off."""

    with pytest.warns(UserWarning, match="use_smoothed_error"):
        step = CrankNicolsonStep(
            precision=np.float64, n=2, use_smoothed_error=True
        )
    assert not step.smooth_error

    step = CrankNicolsonStep(precision=np.float64, n=2)
    with pytest.warns(UserWarning, match="use_smoothed_error"):
        step.update(use_smoothed_error=True)
    assert not step.smooth_error


@pytest.mark.parametrize("enabled", [False, True])
def test_firk_error_solver_costs_nothing_when_disabled(enabled):
    """The smoothing solver enters the step's buffer group only when
    the toggle is on."""

    step = FIRKStep(
        precision=np.float64,
        n=3,
        tableau=RADAU_IIA_5_TABLEAU,
        use_smoothed_error=enabled,
        stage_state_location="shared",
    )
    registered = buffer_registry._groups[step].entries
    assert ("error_solver_shared" in registered) is enabled
    assert step.smooth_error is enabled


def test_firk_error_solver_aliases_the_coupled_solver_window():
    """The smoothing scratch overlaps the coupled solve's shared
    window rather than adding to it."""

    shared_locations = {
        "stage_increment_location": "shared",
        "preconditioned_vec_location": "shared",
        "temp_location": "shared",
        "delta_location": "shared",
        "residual_location": "shared",
    }
    baseline = FIRKStep(
        precision=np.float64,
        n=3,
        tableau=RADAU_IIA_5_TABLEAU,
        **shared_locations,
    )
    smoothed = FIRKStep(
        precision=np.float64,
        n=3,
        tableau=RADAU_IIA_5_TABLEAU,
        use_smoothed_error=True,
        **shared_locations,
    )
    assert buffer_registry.shared_buffer_size(
        smoothed
    ) == buffer_registry.shared_buffer_size(baseline)


@pytest.mark.parametrize(
    "step_class, tableau",
    [(FIRKStep, RADAU_IIA_5_TABLEAU), (DIRKStep, KVAERNO3_TABLEAU)],
)
def test_toggle_survives_update(step_class, tableau):
    """``update`` turns smoothing on and registers its buffers."""

    step = step_class(precision=np.float64, n=2, tableau=tableau)
    assert not step.smooth_error
    step.update(use_smoothed_error=True)
    assert step.smooth_error
    entries = buffer_registry._groups[step].entries
    assert entries["error_solve_iters"].size == 1
