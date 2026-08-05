"""Tests for float-or-callable controller gain specifications."""

import numpy as np
import pytest

from cubie.integrators.algorithms.generic_dirk import (
    DIRK_ADAPTIVE_DEFAULTS,
    dirk_default_ki,
    dirk_default_kp,
)
from cubie.integrators.step_control.adaptive_PI_controller import (
    AdaptivePIController,
    PIStepControlConfig,
)
from cubie.integrators.step_control.adaptive_PID_controller import (
    PIDStepControlConfig,
)
from cubie.integrators.step_control.adaptive_step_controller import (
    OrderDependentGain,
    gain_converter,
)


def test_gain_converter_constant_and_callable():
    """gain_converter passes floats through and wraps callables."""
    assert gain_converter(0.7) == 0.7
    wrapped = gain_converter(dirk_default_kp)
    assert isinstance(wrapped, OrderDependentGain)
    assert wrapped(3) == pytest.approx(0.7 * 4 / 3)
    assert gain_converter(wrapped) is wrapped


def test_pi_config_resolves_callable_gains_at_order():
    """Callable kp/ki resolve against the config's algorithm order."""
    cfg = PIStepControlConfig(
        precision=np.float64,
        algorithm_order=3,
        kp=dirk_default_kp,
        ki=dirk_default_ki,
    )
    assert cfg.kp == pytest.approx(0.7 * 4 / 3)
    assert cfg.ki == pytest.approx(-0.4 * 4 / 3)
    assert cfg.settings_dict["kp"].fn is dirk_default_kp


def test_pid_config_resolves_callable_kd():
    """PID's kd accepts a callable of the order."""
    cfg = PIDStepControlConfig(
        precision=np.float64,
        algorithm_order=4,
        kd=lambda order: 0.05 / order,
    )
    assert cfg.kd == pytest.approx(0.05 / 4)


def test_pi_config_rejects_non_numeric_non_callable():
    """A gain that is neither float nor callable raises."""
    with pytest.raises((TypeError, ValueError)):
        PIStepControlConfig(precision=np.float64, kp="not a gain")


def test_callable_gain_hashes_by_rule_and_order():
    """values_hash keys on the gain rule and the algorithm order."""
    base = PIStepControlConfig(
        precision=np.float64, algorithm_order=3, kp=dirk_default_kp
    )
    same_rule = PIStepControlConfig(
        precision=np.float64, algorithm_order=3, kp=dirk_default_kp
    )
    other_order = PIStepControlConfig(
        precision=np.float64, algorithm_order=5, kp=dirk_default_kp
    )
    other_rule = PIStepControlConfig(
        precision=np.float64, algorithm_order=3, kp=dirk_default_ki
    )
    assert base.values_hash == same_rule.values_hash
    assert base.values_hash != other_order.values_hash
    assert base.values_hash != other_rule.values_hash


def test_controller_reresolves_gains_on_order_update():
    """An algorithm_order update re-evaluates callable gains."""
    controller = AdaptivePIController(
        precision=np.float64,
        n=3,
        dt=0.01,
        algorithm_order=3,
        kp=dirk_default_kp,
        ki=dirk_default_ki,
    )
    assert controller.kp == pytest.approx(0.7 * 4 / 3)
    controller.update_compile_settings({"algorithm_order": 5})
    assert controller.kp == pytest.approx(0.7 * 6 / 5)
    assert controller.ki == pytest.approx(-0.4 * 6 / 5)


def test_controller_settings_dict_preserves_gain_specs():
    """settings_dict carries the supplied rule, not a resolved float."""
    controller = AdaptivePIController(
        precision=np.float64,
        n=3,
        dt=0.01,
        algorithm_order=3,
        kp=dirk_default_kp,
        ki=-0.4,
    )
    settings = controller.settings_dict
    assert settings["kp"].fn is dirk_default_kp
    assert settings["ki"] == pytest.approx(-0.4)


def test_settings_dict_round_trips_through_reconstruction():
    """A config rebuilt from settings_dict keeps order tracking."""
    controller = AdaptivePIController(
        precision=np.float64,
        n=3,
        dt=0.01,
        algorithm_order=3,
        kp=dirk_default_kp,
        ki=dirk_default_ki,
    )
    settings = controller.settings_dict
    rebuilt = PIStepControlConfig(
        precision=np.float64,
        algorithm_order=5,
        kp=settings["kp"],
        ki=settings["ki"],
    )
    assert rebuilt.kp == pytest.approx(0.7 * 6 / 5)
    assert rebuilt.ki == pytest.approx(-0.4 * 6 / 5)


def test_dirk_adaptive_defaults_use_order_callable_pi():
    """DIRK adaptive defaults select PI with the order-callable gains."""
    defaults = DIRK_ADAPTIVE_DEFAULTS.step_controller
    assert defaults["step_controller"] == "pi"
    assert defaults["kp"] is dirk_default_kp
    assert defaults["ki"] is dirk_default_ki
    assert defaults["min_gain"] == 0.2
    assert defaults["max_gain"] == 10.0
    assert defaults["safety"] == 0.9
    assert defaults["deadband_min"] == pytest.approx(1.0 / 1.2)
    assert defaults["deadband_max"] == 1.0
    assert dirk_default_kp(3) == pytest.approx(0.7 * 4 / 3)
    assert dirk_default_ki(5) == pytest.approx(-0.4 * 6 / 5)
