"""Tests for cubie.integrators.step_control.__init__."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.integrators.step_control import (
    _CONTROLLER_REGISTRY,
    AdaptiveIController,
    AdaptivePIController,
    AdaptivePIDController,
    FixedStepController,
    GustafssonController,
    SciMLPIController,
    get_controller,
)


# ── _CONTROLLER_REGISTRY (items 1-2) ─────────────────────── #

_EXPECTED_REGISTRY = {
    "fixed": FixedStepController,
    "i": AdaptiveIController,
    "pi": AdaptivePIController,
    "pid": AdaptivePIDController,
    "sciml_pi": SciMLPIController,
    "gustafsson": GustafssonController,
}


@pytest.mark.parametrize(
    "key, expected_class",
    list(_EXPECTED_REGISTRY.items()),
)
def test_controller_registry_entries(key, expected_class):
    """_CONTROLLER_REGISTRY maps each key to the correct class."""
    assert _CONTROLLER_REGISTRY[key] is expected_class


def test_controller_registry_size():
    """_CONTROLLER_REGISTRY contains exactly 6 entries."""
    assert len(_CONTROLLER_REGISTRY) == 6


# ── get_controller (items 3-10) ──────────────────────────── #

def test_get_controller_missing_step_controller():
    """get_controller raises ValueError when step_controller missing."""
    with pytest.raises(ValueError, match="No step controller"):
        get_controller(np.float32, settings={})


def test_get_controller_unknown_type():
    """get_controller raises ValueError for unknown controller type."""
    with pytest.raises(ValueError, match="Unknown controller type"):
        get_controller(
            np.float32,
            settings={"step_controller": "bogus"},
        )


def test_get_controller_lowercases():
    """get_controller lowercases the step_controller value."""
    ctrl = get_controller(
        np.float32,
        settings={"step_controller": "FIXED", "dt": 0.01},
    )
    assert ctrl.compile_settings.precision == np.float32


def test_get_controller_kwargs_override():
    """kwargs override settings dict entries."""
    ctrl = get_controller(
        np.float32,
        settings={"step_controller": "fixed", "dt": 0.01},
        dt=0.05,
    )
    assert ctrl.compile_settings._dt == pytest.approx(0.05)


def test_get_controller_injects_precision():
    """get_controller injects precision into the controller."""
    ctrl = get_controller(
        np.float64,
        settings={"step_controller": "fixed", "dt": 0.01},
    )
    assert ctrl.compile_settings.precision == np.float64


def test_get_controller_returns_correct_type(step_controller):
    """get_controller returns correct controller from fixture."""
    # Default fixture uses 'fixed' controller
    assert step_controller.compile_settings.precision == np.float32
