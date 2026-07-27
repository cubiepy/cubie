"""Tests for the Gustafsson controller configuration surface."""

import numpy as np
import pytest

from cubie.integrators.step_control.adaptive_step_controller import (
    AdaptiveStepControlConfig,
)
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
)
from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
)
from cubie.integrators.step_control.gustafsson_controller import (
    GustafssonStepControlConfig,
)


# ── GustafssonStepControlConfig: defaults and validation ─────────── #


def test_config_defaults():
    """Gustafsson-specific fields default correctly."""
    # Inline construction permitted per Rule 9: __init__ test.
    cfg = GustafssonStepControlConfig(precision=np.float64)
    assert cfg.newton_target_iters == 20


def test_config_newton_target_iters_returns_int():
    """newton_target_iters is exposed as a plain int."""
    cfg = GustafssonStepControlConfig(
        precision=np.float64, newton_target_iters=7
    )
    assert cfg.newton_target_iters == 7
    assert isinstance(cfg.newton_target_iters, int)


def test_config_newton_target_iters_validates_nonnegative():
    """newton_target_iters rejects negative values."""
    with pytest.raises((ValueError, TypeError)):
        GustafssonStepControlConfig(
            precision=np.float64, newton_target_iters=-1
        )


def test_config_settings_dict_extends_parent():
    """settings_dict adds the controller's Newton-work target."""
    cfg = GustafssonStepControlConfig(
        precision=np.float64, safety=0.85, newton_target_iters=11
    )
    settings = cfg.settings_dict
    assert settings["safety"] == pytest.approx(0.85)
    assert settings["newton_target_iters"] == 11
    assert "gamma" not in settings
    assert "newton_max_iters" not in settings
    parent_keys = set(
        AdaptiveStepControlConfig(precision=np.float64).settings_dict
    )
    assert parent_keys <= set(settings)


def test_algorithm_and_controller_names_only_share_identical_values():
    """Flat settings share only system precision and state count."""
    shared = ALL_ALGORITHM_STEP_PARAMETERS & ALL_STEP_CONTROLLER_PARAMETERS
    assert shared == {"precision", "n"}
    assert "gamma" in ALL_ALGORITHM_STEP_PARAMETERS
    assert "newton_max_iters" in ALL_ALGORITHM_STEP_PARAMETERS
    assert "newton_target_iters" in ALL_STEP_CONTROLLER_PARAMETERS


# ── GustafssonController: property forwarding ─────────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"step_controller": "gustafsson", "atol": 1e-3, "rtol": 0.0}],
    ids=["gustafsson"],
    indirect=True,
)
def test_controller_forwards_config_properties(step_controller):
    """Controller Newton-work target mirrors compile_settings."""
    assert (
        step_controller.newton_target_iters
        == step_controller.compile_settings.newton_target_iters
    )
    assert isinstance(step_controller.newton_target_iters, int)
