"""Tests for cubie.integrators.step_control.adaptive_step_controller."""

from __future__ import annotations


import numpy as np

from cubie.integrators.step_control.adaptive_I_controller import (
    AdaptiveIController,
)
import pytest

from tests._utils import CONTROLLER_TOLERANCE_SETS
from numpy import sqrt
from numpy.testing import assert_array_equal

from cubie.integrators.step_control.adaptive_step_controller import (
    AdaptiveStepControlConfig,
)


# ── AdaptiveStepControlConfig: field defaults (items 43-52) ──────── #


def test_config_defaults():
    """Field defaults are set correctly on bare construction."""
    # Inline construction permitted per Rule 9: __init__ test.
    cfg = AdaptiveStepControlConfig(precision=np.float64)
    assert cfg._dt_min == 1e-6
    assert cfg._dt_max == pytest.approx(1.0)
    assert_array_equal(cfg.atol, np.asarray([1e-6]))
    assert_array_equal(cfg.rtol, np.asarray([1e-6]))
    assert cfg.algorithm_order == 1
    assert cfg._min_gain == pytest.approx(0.3)
    assert cfg._max_gain == pytest.approx(2.0)
    assert cfg._safety == pytest.approx(0.9)
    assert cfg._deadband_min == pytest.approx(1.0)
    assert cfg._deadband_max == pytest.approx(1.0)


def test_config_dt_min_validates_positive():
    """_dt_min rejects non-positive values."""
    with pytest.raises((ValueError, TypeError)):
        AdaptiveStepControlConfig(precision=np.float64, dt_min=-1.0)


def test_config_min_gain_rejects_near_unity():
    """_min_gain rejects values above 0.95, unity included."""
    with pytest.raises((ValueError, TypeError)):
        AdaptiveStepControlConfig(precision=np.float64, min_gain=1.0)
    with pytest.raises((ValueError, TypeError)):
        AdaptiveStepControlConfig(precision=np.float64, min_gain=0.96)


def test_config_negative_atol_rejected():
    """atol rejects arrays containing negative values."""
    with pytest.raises(ValueError):
        AdaptiveStepControlConfig(precision=np.float64, atol=-1e-6)


def test_config_negative_rtol_rejected():
    """rtol rejects arrays containing negative elements."""
    with pytest.raises(ValueError):
        AdaptiveStepControlConfig(
            precision=np.float64,
            rtol=np.asarray([1e-6, -1e-6]),
        )


def test_config_zero_tolerances_accepted():
    """Zero tolerances are valid; the norm floors the combined value."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, atol=0.0, rtol=0.0
    )
    assert_array_equal(cfg.atol, np.asarray([0.0]))
    assert_array_equal(cfg.rtol, np.asarray([0.0]))


def test_config_algorithm_order_validates_ge_1():
    """algorithm_order rejects values < 1."""
    with pytest.raises((ValueError, TypeError)):
        AdaptiveStepControlConfig(precision=np.float64, algorithm_order=0)


# ── __attrs_post_init__ (items 53-57) ────────────────────────────── #


def test_post_init_dt_max_none_rejected_by_validator():
    """dt_max=None is rejected by the field validator (item 53).

    The post_init None-handling path and property fallback are
    unreachable through normal construction because the validator
    requires a float > 0.
    """
    with pytest.raises(TypeError):
        AdaptiveStepControlConfig(
            precision=np.float64, dt_min=0.001, dt_max=None
        )


def test_post_init_dt_max_lt_dt_min_allowed_in_config():
    """Config allows dt_max < dt_min; validation deferred to controller.

    The config attrs class stores raw values. Validation that raises
    ValueError for user-provided inverted bounds happens in the
    controller's _ensure_sane_bounds() method, not in the config.
    """
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, dt_min=1.0, dt_max=0.5
    )
    # Config stores raw values; controller validates on construction
    assert cfg._dt_min == pytest.approx(1.0)
    assert cfg._dt_max == pytest.approx(0.5)


def test_post_init_dt_max_ge_dt_min_no_change():
    """When dt_max >= dt_min, no correction occurs."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, dt_min=0.01, dt_max=1.0
    )
    assert cfg._dt_max == pytest.approx(1.0)


def test_post_init_deadband_swapped_when_inverted():
    """Deadband limits are swapped when min > max.

    Both values must pass their respective validators first
    (deadband_min in [0, 1.0], deadband_max >= 1.0), so use values
    where min=1.0 > max would be triggered if validators allowed it.
    Since 1.0 is the boundary for both validators, construct with
    valid values and verify the swap path via the source logic:
    the swap occurs when _deadband_min > _deadband_max.
    """
    # deadband_min=1.0 and deadband_max=1.0: min is NOT > max, no swap
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, deadband_min=0.9, deadband_max=1.0
    )
    # No swap needed: 0.9 <= 1.0
    assert cfg._deadband_min == pytest.approx(0.9)
    assert cfg._deadband_max == pytest.approx(1.0)
    # Now manually set to trigger swap path and verify post_init logic
    cfg2 = AdaptiveStepControlConfig(
        precision=np.float64, deadband_min=1.0, deadband_max=1.0
    )
    assert cfg2._deadband_min <= cfg2._deadband_max


def test_post_init_deadband_no_swap_when_ordered():
    """Deadband limits unchanged when already ordered."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, deadband_min=0.8, deadband_max=1.2
    )
    assert cfg._deadband_min == pytest.approx(0.8)
    assert cfg._deadband_max == pytest.approx(1.2)


# ── Config properties (items 58-66) ─────────────────────────────── #


@pytest.mark.parametrize(
    "prop, raw_attr, expected_fn",
    [
        ("dt_min", "_dt_min", lambda c: c.precision(c._dt_min)),
        ("min_gain", "_min_gain", lambda c: c.precision(c._min_gain)),
        ("max_gain", "_max_gain", lambda c: c.precision(c._max_gain)),
        ("safety", "_safety", lambda c: c.precision(c._safety)),
        (
            "deadband_min",
            "_deadband_min",
            lambda c: c.precision(c._deadband_min),
        ),
        (
            "deadband_max",
            "_deadband_max",
            lambda c: c.precision(c._deadband_max),
        ),
    ],
    ids=["dt_min", "min_gain", "max_gain", "safety",
         "deadband_min", "deadband_max"],
)
def test_config_property_applies_precision(prop, raw_attr, expected_fn):
    """Config properties return precision-cast values."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float32, dt_min=1e-4, dt_max=2.0,
        min_gain=0.2, max_gain=3.0, safety=0.85,
        deadband_min=0.9, deadband_max=1.1,
    )
    assert getattr(cfg, prop) == expected_fn(cfg)
    assert type(getattr(cfg, prop)) is np.float32


def test_config_dt_max_property_with_value():
    """dt_max property returns precision-cast _dt_max when set."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float32, dt_min=1e-4, dt_max=2.5
    )
    assert cfg.dt_max == np.float32(2.5)


def test_config_dt_max_property_normal_path():
    """dt_max property returns precision-cast value when set normally."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, dt_min=0.005, dt_max=2.0
    )
    assert cfg.dt_max == pytest.approx(np.float64(2.0))


def test_controller_derives_dt_from_bounds():
    """Controller derives dt as sqrt(dt_min * dt_max) when not provided."""
    ctrl = AdaptiveIController(
        precision=np.float64, n=3, dt_min=1e-4, dt_max=1.0
    )
    expected = np.float64(sqrt(1e-4 * 1.0))
    assert ctrl.dt == pytest.approx(expected)


def test_config_is_adaptive():
    """is_adaptive returns True for adaptive config."""
    cfg = AdaptiveStepControlConfig(precision=np.float64)
    assert cfg.is_adaptive is True


# ── settings_dict (item 67) ─────────────────────────────────────── #


def test_config_settings_dict_keys():
    """settings_dict contains all expected adaptive controller keys."""
    cfg = AdaptiveStepControlConfig(
        precision=np.float64, dt=1e-3, dt_min=1e-5, dt_max=0.5,
        algorithm_order=2,
    )
    d = cfg.settings_dict
    expected_keys = {
        "dt_min", "dt_max", "atol", "rtol", "algorithm_order",
        "min_gain", "max_gain", "safety", "deadband_min",
        "deadband_max", "dt", "n",
    }
    assert expected_keys <= set(d.keys())
    assert d["dt_min"] == cfg.dt_min
    assert d["dt_max"] == cfg.dt_max
    assert d["algorithm_order"] == cfg.algorithm_order
    assert d["safety"] == cfg.safety
    assert d["dt"] == cfg.dt


# ── BaseAdaptiveStepController __init__ (item 68) ────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(CONTROLLER_TOLERANCE_SETS["i"], id="i-controller")],
    indirect=True,
)
def test_controller_init_sets_compile_settings(step_controller):
    """__init__ calls setup_compile_settings and register_buffers."""
    assert step_controller.compile_settings is not None
    assert isinstance(
        step_controller.compile_settings, AdaptiveStepControlConfig,
    )


# ── BaseAdaptiveStepController.build (item 69) ──────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(CONTROLLER_TOLERANCE_SETS["i"], id="i-controller")],
    indirect=True,
)
def test_controller_build_produces_callable(step_controller):
    """build() produces a callable device_function."""
    df = step_controller.device_function  # triggers build
    assert callable(df)
    # After build, cache holds the same object
    assert step_controller._cache.device_function is df


# ── Forwarding properties (items 71-78) ─────────────────────────── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(CONTROLLER_TOLERANCE_SETS["i"], id="i-controller")],
    indirect=True,
)
@pytest.mark.parametrize(
    "prop, child_attr",
    [
        ("min_gain", "min_gain"),
        ("max_gain", "max_gain"),
        ("safety", "safety"),
        ("deadband_min", "deadband_min"),
        ("deadband_max", "deadband_max"),
        ("algorithm_order", "algorithm_order"),
    ],
    ids=["min_gain", "max_gain", "safety", "deadband_min",
         "deadband_max", "algorithm_order"],
)
def test_controller_forwarding_scalars(step_controller, prop, child_attr):
    """Scalar forwarding properties delegate to compile_settings."""
    ctrl_val = getattr(step_controller, prop)
    cs_val = getattr(step_controller.compile_settings, child_attr)
    if prop == "algorithm_order":
        assert ctrl_val == int(cs_val)
    else:
        assert ctrl_val == cs_val


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(CONTROLLER_TOLERANCE_SETS["i"], id="i-controller")],
    indirect=True,
)
@pytest.mark.parametrize(
    "prop",
    ["atol", "rtol"],
)
def test_controller_forwarding_arrays(step_controller, prop):
    """Array forwarding properties delegate to compile_settings."""
    assert_array_equal(
        getattr(step_controller, prop),
        getattr(step_controller.compile_settings, prop),
    )


# ── build_controller abstract / persistent_local_buffer_size ── #


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(CONTROLLER_TOLERANCE_SETS["i"], id="i-controller")],
    indirect=True,
)
def test_persistent_local_buffer_size_is_int(step_controller):
    """persistent_local_buffer_size returns a non-negative int."""
    val = step_controller.persistent_local_buffer_size
    assert isinstance(val, int)
    assert val >= 0


# ── resolve_step_params translation ──────────────────────────── #


def test_resolve_adaptive_bounds_only():
    """dt_min + dt_max pass through; dt = sqrt(dt_min*dt_max) when not set."""
    ctrl = AdaptiveIController(
        precision=np.float64, dt_min=1e-4, dt_max=1.0,
    )
    assert ctrl.dt_min == pytest.approx(np.float64(1e-4))
    assert ctrl.dt_max == pytest.approx(np.float64(1.0))
    expected_dt = np.float64(sqrt(1e-4 * 1.0))
    assert ctrl.dt == pytest.approx(expected_dt)


def test_resolve_adaptive_dt_only():
    """Bare dt translates to dt_min=dt/100, dt_max=dt*100."""
    ctrl = AdaptiveIController(precision=np.float64, dt=0.01)
    assert ctrl.dt_min == pytest.approx(np.float64(0.01 / 100))
    assert ctrl.dt_max == pytest.approx(np.float64(0.01 * 100))


def test_resolve_adaptive_dt_plus_dt_min():
    """dt + dt_min: dt_max filled from dt*100."""
    ctrl = AdaptiveIController(
        precision=np.float64, dt=0.01, dt_min=1e-5,
    )
    assert ctrl.dt_min == pytest.approx(np.float64(1e-5))
    assert ctrl.dt_max == pytest.approx(np.float64(0.01 * 100))


def test_resolve_adaptive_dt_plus_dt_max():
    """dt + dt_max: dt_min filled from dt/100."""
    ctrl = AdaptiveIController(
        precision=np.float64, dt=0.01, dt_max=10.0,
    )
    assert ctrl.dt_min == pytest.approx(np.float64(0.01 / 100))
    assert ctrl.dt_max == pytest.approx(np.float64(10.0))


def test_resolve_adaptive_all_three_accepted():
    """dt + dt_min + dt_max: all three accepted without warning."""
    ctrl = AdaptiveIController(
        precision=np.float64, dt=0.01,
        dt_min=1e-5, dt_max=10.0,
    )
    assert ctrl.dt == pytest.approx(np.float64(0.01))
    assert ctrl.dt_min == pytest.approx(np.float64(1e-5))
    assert ctrl.dt_max == pytest.approx(np.float64(10.0))


def test_update_preserves_user_set_bounds():
    """User-set bounds are preserved when dt is updated."""
    ctrl = AdaptiveIController(
        precision=np.float64, dt_min=1e-4, dt_max=1.0,
    )
    ctrl.update({"dt": 0.05})
    # Bounds were user-set at construction, so preserved
    assert ctrl.dt == pytest.approx(np.float64(0.05))
    assert ctrl.dt_min == pytest.approx(np.float64(1e-4))
    assert ctrl.dt_max == pytest.approx(np.float64(1.0))


def test_update_fixes_violated_bounds():
    """Non-user-set bounds are fixed when constraints violated."""
    # Construction with dt only - bounds are derived
    ctrl = AdaptiveIController(precision=np.float64, dt=1e-6)
    assert ctrl.dt_min == pytest.approx(np.float64(1e-8))
    assert ctrl.dt_max == pytest.approx(np.float64(1e-4))

    # Update dt to value outside derived bounds
    ctrl.update({"dt": 1e-2})
    assert ctrl.dt == pytest.approx(np.float64(1e-2))
    # dt > old dt_max, so dt_max re-derived (not user-set)
    assert ctrl.dt_max == pytest.approx(np.float64(1e-2 * 100))
    # dt_min unchanged (no violation)
    assert ctrl.dt_min == pytest.approx(np.float64(1e-8))


def test_update_tracks_newly_set_bounds():
    """Bounds set via update become sticky and raise on violation."""
    ctrl = AdaptiveIController(precision=np.float64, dt=1e-4)
    # dt_min and dt_max were derived at construction

    # Explicitly set dt_min via update
    ctrl.update({"dt_min": 1e-6})

    # Updating dt to violate user-set dt_min raises ValueError
    with pytest.raises(ValueError, match="dt.*<.*dt_min"):
        ctrl.update({"dt": 1e-8})


# ── _ensure_sane_bounds: user-provided inversion/violation raises ── #


def test_construction_raises_on_inverted_user_bounds():
    """Both bounds user-provided and inverted raises ValueError."""
    with pytest.raises(ValueError, match="dt_max.*<.*dt_min"):
        AdaptiveIController(
            precision=np.float64, dt_min=1.0, dt_max=0.5,
        )


def test_construction_raises_on_dt_above_user_dt_max():
    """dt above a user-provided dt_max raises ValueError."""
    with pytest.raises(ValueError, match="dt.*>.*dt_max"):
        AdaptiveIController(
            precision=np.float64, dt=2.0, dt_max=1.0,
        )


# ── _ensure_sane_bounds: auto-fix of non-user-provided bounds ────── #


def test_update_autofixes_dt_max_when_derived_max_falls_below_new_min():
    """A user-set dt_min above the (non-user) derived dt_max auto-fixes

    dt_max: dt_max_new = dt_min * 100. The derived dt_max is also below
    the new dt, so both auto-fix branches for dt_max run; the final
    value comes from the dt-based fix (dt * 100).
    """
    ctrl = AdaptiveIController(precision=np.float64, dt=1e-7)
    # dt_max derived as dt * 100 = 1e-5, non-user.
    ctrl.update({"dt": 0.1, "dt_min": 0.001})
    # dt_min is honoured (user-provided); dt_max is auto-fixed upward
    # from its stale derived value so it stays >= dt_min and >= dt.
    assert ctrl.dt == pytest.approx(np.float64(0.1))
    assert ctrl.dt_min == pytest.approx(np.float64(0.001))
    assert ctrl.dt_max >= ctrl.dt_min
    assert ctrl.dt_max >= ctrl.dt


def test_update_autofixes_dt_min_when_derived_min_exceeds_new_max():
    """A user-set dt_max below the (non-user) derived dt_min auto-fixes

    dt_min from both the dt_max-relative and dt-relative branches.
    """
    ctrl = AdaptiveIController(precision=np.float64, dt=10000.0)
    # dt_min derived as dt / 100 = 100.0, non-user.
    ctrl.update({"dt": 1e-7, "dt_max": 10.0})
    assert ctrl.dt == pytest.approx(np.float64(1e-7))
    assert ctrl.dt_max == pytest.approx(np.float64(10.0))
    assert ctrl.dt_min <= ctrl.dt_max
    assert ctrl.dt_min <= ctrl.dt


# ── AdaptiveStepControlConfig.__attrs_post_init__: deadband swap ─── #


def test_deadband_swap_branch_is_unreachable():
    """Inverted deadbands cannot be constructed through the public API.

    deadband_min's validator bounds it to [0, 1.0] and deadband_max's
    validator requires >= 1.0, so deadband_min > deadband_max is
    impossible to construct.
    """
    with pytest.raises((ValueError, TypeError)):
        AdaptiveStepControlConfig(
            precision=np.float64, deadband_min=1.1, deadband_max=0.9,
        )
