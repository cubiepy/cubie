"""Tests for the BDF tableaus and the step history device function."""

import numpy as np
import pytest

from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms import algorithm_facts
from cubie.integrators.algorithms.generic_bdf import BDFStep
from cubie.integrators.algorithms.generic_bdf_tableaus import (
    BDF_TABLEAU_REGISTRY,
    BDFTableau,
    MAX_BDF_ORDER,
    _parasitic_radius,
    zero_stable_ratio_limit,
)
from cubie.integrators.step_history import StepHistory
from tests._utils import run_step_history


@pytest.mark.parametrize("order", range(2, MAX_BDF_ORDER + 1))
def test_ratio_limit_bounds_zero_stability(order):
    """The limit separates stable from unstable constant step ratios."""
    limit = zero_stable_ratio_limit(order)
    assert _parasitic_radius(order, limit * (1.0 - 1e-6)) < 1.0
    assert _parasitic_radius(order, limit * (1.0 + 1e-6)) >= 1.0


@pytest.mark.parametrize("alias", list(BDF_TABLEAU_REGISTRY))
def test_tableau_ratio_limits_follow_orders(alias):
    """Order q's limit is BDF-max(q, 2)'s zero-stability limit."""
    tableau = BDF_TABLEAU_REGISTRY[alias]
    assert len(tableau.ratio_limits) == tableau.order
    for index, limit in enumerate(tableau.ratio_limits):
        assert limit == zero_stable_ratio_limit(max(index + 1, 2))


@pytest.mark.parametrize("order", [0, MAX_BDF_ORDER + 1])
def test_tableau_rejects_unsupported_order(order):
    """Orders outside the zero-stable range raise."""
    with pytest.raises(ValueError):
        BDFTableau(order=order)


def test_bdf_facts_are_adaptive_and_implicit():
    """A BDF choice reports an estimate and an implicit Newton step."""
    facts = algorithm_facts("bdf")
    assert facts.step_class is BDFStep
    assert facts.has_error_estimate
    assert facts.is_implicit
    assert not facts.is_linear


def _polynomial(times):
    """Return a cubic in each of two states at ``times``."""
    times = np.asarray(times, dtype=np.float64)
    return np.stack(
        (
            1.0 + 2.0 * times - 3.0 * times**2 + 0.5 * times**3,
            -0.5 + times**3,
        ),
        axis=-1,
    )


def _polynomial_slope(times):
    """Return the time derivative of :func:`_polynomial`."""
    times = np.asarray(times, dtype=np.float64)
    return np.stack(
        (2.0 - 6.0 * times + 1.5 * times**2, 3.0 * times**2),
        axis=-1,
    )


def _history_run(steps, accepted, starts):
    """Run a BDF3 history on cubic states starting at ``starts``."""
    precision = np.float32
    history = StepHistory(
        precision=precision,
        n_states=2,
        tableau=BDF_TABLEAU_REGISTRY["bdf3"],
    )
    persistent_len = buffer_registry.persistent_local_buffer_size(history)
    states = _polynomial(starts)
    return run_step_history(
        history.device_function,
        states,
        steps,
        accepted,
        precision,
        persistent_len,
    )


def test_history_is_exact_for_cubics_across_a_rejection():
    """A cubic stays exact at full order across a rejected attempt."""
    steps = [0.10, 0.12, 0.11, 0.13, 0.20, 0.12, 0.125]
    accepted = [1, 1, 1, 1, 1, 0, 1]
    starts = [0.0, 0.10, 0.22, 0.33, 0.46, 0.46, 0.58]
    base, prediction, scalars = _history_run(steps, accepted, starts)

    ends = np.asarray(starts) + np.asarray(steps)
    exact = _polynomial(ends)
    slope = _polynomial_slope(ends)
    for call in (4, 5, 6):
        corrector_step = np.float64(scalars[call, 0])
        np.testing.assert_allclose(
            base[call] + corrector_step * slope[call],
            exact[call],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            prediction[call], exact[call], rtol=1e-5, atol=1e-5
        )
        assert scalars[call, 2] == 0.0


def test_history_restarts_on_first_step_and_ratio_jump():
    """No qualifying order restarts from the start state."""
    steps = [0.10, 0.10, 0.50]
    accepted = [1, 1, 1]
    starts = [0.0, 0.10, 0.20]
    base, prediction, scalars = _history_run(steps, accepted, starts)
    states = _polynomial(starts).astype(np.float32)

    for call in (0, 2):
        assert scalars[call, 2] == 1.0
        assert scalars[call, 1] == np.float32(0.5)
        assert scalars[call, 0] == np.float32(steps[call])
        np.testing.assert_array_equal(prediction[call], states[call])
        np.testing.assert_array_equal(base[call], states[call])
    assert scalars[1, 2] == 0.0


def test_short_step_merges_and_keeps_full_order():
    """A step clamped short joins the step before it at full order."""
    steps = [0.10, 0.10, 0.10, 0.01, 0.10]
    accepted = [1, 1, 1, 1, 1]
    starts = [0.0, 0.10, 0.20, 0.30, 0.31]
    base, prediction, scalars = _history_run(steps, accepted, starts)

    end = starts[-1] + steps[-1]
    exact = _polynomial([end])[0]
    slope = _polynomial_slope([end])[0]
    corrector_step = np.float64(scalars[-1, 0])
    np.testing.assert_allclose(
        base[-1] + corrector_step * slope, exact, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(prediction[-1], exact, rtol=1e-5, atol=1e-5)
    assert scalars[-1, 2] == 0.0
