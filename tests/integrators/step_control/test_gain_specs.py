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
    resolve_gain_spec,
)


def test_resolve_gain_spec_constant_and_callable():
    """resolve_gain_spec passes floats through and calls callables."""
    assert resolve_gain_spec(0.7, 3) == 0.7
    assert resolve_gain_spec(lambda order: 0.1 * order, 5) == 0.5


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
    assert cfg.settings_dict["kp"] is dirk_default_kp


def test_pid_config_resolves_callable_kd():
    """PID's kd accepts a callable of the order."""
    cfg = PIDStepControlConfig(
        precision=np.float64,
        algorithm_order=4,
        kd=lambda order: 0.05 / order,
    )
    assert cfg.kd == pytest.approx(0.05 / 4)


def test_pi_config_rejects_non_numeric_non_callable():
    """A gain that is neither float nor callable raises TypeError."""
    with pytest.raises(TypeError):
        PIStepControlConfig(precision=np.float64, kp="0.7")


def test_callable_gain_hashes_as_resolved_value():
    """values_hash keys on the resolved gain, not the spec object."""
    from_callable = PIStepControlConfig(
        precision=np.float64, algorithm_order=3, kp=dirk_default_kp
    )
    from_constant = PIStepControlConfig(
        precision=np.float64, algorithm_order=3, kp=0.7 * 4 / 3
    )
    different = PIStepControlConfig(
        precision=np.float64, algorithm_order=3, kp=0.9
    )
    assert from_callable.values_hash == from_constant.values_hash
    assert from_callable.values_hash != different.values_hash


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
    """settings_dict carries the supplied spec, not the resolved value."""
    controller = AdaptivePIController(
        precision=np.float64,
        n=3,
        dt=0.01,
        algorithm_order=3,
        kp=dirk_default_kp,
        ki=-0.4,
    )
    settings = controller.settings_dict
    assert settings["kp"] is dirk_default_kp
    assert settings["ki"] == pytest.approx(-0.4)


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
