"""Tests for cubie.integrators.step_control.fixed_step_controller."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.integrators.step_control.fixed_step_controller import (
    FixedStepControlConfig,
    FixedStepController,
)
from cubie.integrators.step_control.base_step_controller import (
    ControllerCache,
)


# ── FixedStepControlConfig __init__ / attrs ─────────────────── #


def test_config_dt_default():
    """Default _dt is 1e-3, returned through precision casting."""
    cfg = FixedStepControlConfig(precision=np.float32, n=3)
    assert cfg.dt == pytest.approx(np.float32(1e-3))


def test_config_dt_custom():
    """Custom dt value is stored and returned through precision."""
    cfg = FixedStepControlConfig(
        precision=np.float64, n=2, dt=0.005,
    )
    assert cfg.dt == pytest.approx(np.float64(0.005))


def test_config_dt_invalid_raises():
    """Negative dt is rejected by the getype_validator."""
    with pytest.raises((ValueError, TypeError)):
        FixedStepControlConfig(precision=np.float32, n=1, dt=-0.01)


def test_config_dt_min_equals_dt():
    """dt_min returns the same value as dt."""
    cfg = FixedStepControlConfig(precision=np.float64, dt=0.002)
    assert cfg.dt_min == cfg.dt


def test_config_dt_max_equals_dt():
    """dt_max returns the same value as dt."""
    cfg = FixedStepControlConfig(precision=np.float64, dt=0.002)
    assert cfg.dt_max == cfg.dt


def test_config_is_adaptive_false():
    """is_adaptive always returns False for fixed controller."""
    cfg = FixedStepControlConfig(precision=np.float32)
    assert cfg.is_adaptive is False


def test_config_settings_dict_contains_dt_and_n():
    """settings_dict contains 'dt' from this class and 'n' from super."""
    cfg = FixedStepControlConfig(
        precision=np.float32, n=5, dt=0.003,
    )
    sd = cfg.settings_dict
    assert sd["dt"] == pytest.approx(np.float32(0.003))
    assert sd["n"] == 5


# ── FixedStepController (via fixture) ───────────────────────── #


def test_controller_init_creates_config(step_controller):
    """__init__ sets up compile_settings as FixedStepControlConfig."""
    # isinstance justified: combined with value checks below
    assert isinstance(
        step_controller.compile_settings, FixedStepControlConfig,
    )
    assert step_controller.compile_settings.is_adaptive is False


@pytest.mark.parametrize(
    "prop, child_attr",
    [
        ("dt", "dt"),
        ("dt_min", "dt_min"),
        ("dt_max", "dt_max"),
        ("is_adaptive", "is_adaptive"),
        ("n", "n"),
    ],
)
def test_forwarding_to_compile_settings(
    step_controller, prop, child_attr,
):
    """Forwarding properties delegate to compile_settings."""
    assert getattr(step_controller, prop) == getattr(
        step_controller.compile_settings, child_attr,
    )


def test_registers_no_timestep_buffer(step_controller):
    """Fixed controller keeps no history, so registers no local memory."""
    assert step_controller.persistent_local_buffer_size == 0


def test_build_returns_controller_cache(step_controller):
    """build() returns a ControllerCache with a callable device function.

    Device invocation tested in test_controllers.py.
    """
    df = step_controller.device_function  # triggers build
    cache = step_controller._cache
    # justified: type IS the functionality
    assert isinstance(cache, ControllerCache)
    assert cache.device_function is df
    # callable justified: cross-file device invocation in test_controllers.py
    assert callable(df)


def test_dt_matches_solver_settings(step_controller, solver_settings):
    """Controller dt matches the dt from solver_settings."""
    expected = solver_settings["precision"](solver_settings["dt"])
    assert step_controller.dt == pytest.approx(expected)


# ── resolve_step_params translation ────────────────────────── #


def test_resolve_fixed_dt_only():
    """dt alone passes through unchanged."""
    ctrl = FixedStepController(precision=np.float64, dt=0.01)
    assert ctrl.dt == pytest.approx(np.float64(0.01))


def test_resolve_fixed_dt_min_only():
    """dt_min alone is translated to dt."""
    ctrl = FixedStepController(precision=np.float64, dt_min=0.001)
    assert ctrl.dt == pytest.approx(np.float64(0.001))


def test_resolve_fixed_dt_max_only():
    """dt_max alone is translated to dt."""
    ctrl = FixedStepController(precision=np.float64, dt_max=0.5)
    assert ctrl.dt == pytest.approx(np.float64(0.5))


def test_resolve_fixed_both_bounds():
    """dt_min + dt_max translates to dt = dt_min (first available)."""
    ctrl = FixedStepController(
        precision=np.float64, dt_min=0.001, dt_max=0.5,
    )
    assert ctrl.dt == pytest.approx(np.float64(0.001))


def test_resolve_fixed_dt_plus_bounds():
    """dt + dt_min keeps dt (first available)."""
    ctrl = FixedStepController(
        precision=np.float64, dt=0.01, dt_min=0.001,
    )
    assert ctrl.dt == pytest.approx(np.float64(0.01))


def test_resolve_fixed_default():
    """No step params gives the default dt."""
    ctrl = FixedStepController(precision=np.float64)
    assert ctrl.dt == pytest.approx(np.float64(1e-3))


def test_update_dt_directly():
    """Fixed controller update accepts dt directly."""
    ctrl = FixedStepController(precision=np.float64, dt=0.01)
    ctrl.update({"dt": 0.005})
    assert ctrl.dt == pytest.approx(np.float64(0.005))


def test_update_dt_min_warns():
    """Fixed controller warns when dt_min passed to update."""
    ctrl = FixedStepController(precision=np.float64, dt=0.01)
    with pytest.warns(UserWarning, match="dt_min.*not recognized"):
        ctrl.update({"dt_min": 0.005})
    # dt unchanged
    assert ctrl.dt == pytest.approx(np.float64(0.01))
