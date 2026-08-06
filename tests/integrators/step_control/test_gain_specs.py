"""Tests for float-or-callable controller gain specifications."""

import pytest

from cubie.integrators.algorithms.generic_dirk import (
    DIRK_ADAPTIVE_DEFAULTS,
    dirk_default_ki,
    dirk_default_kp,
)
from cubie.integrators.step_control.adaptive_step_controller import (
    OrderDependentGain,
    gain_converter,
)


def half_over_order(order):
    """Reference kd callable for the PID case."""
    return 0.05 / order


PI_CALLABLE_GAINS = {
    "step_controller": "pi",
    "kp": dirk_default_kp,
    "ki": dirk_default_ki,
}
PID_CALLABLE_GAINS = {
    "step_controller": "pid",
    "kp": dirk_default_kp,
    "ki": dirk_default_ki,
    "kd": half_over_order,
}


def test_gain_converter_constant_and_callable():
    """gain_converter passes floats through and wraps callables."""
    assert gain_converter(0.7) == 0.7
    wrapped = gain_converter(dirk_default_kp)
    assert isinstance(wrapped, OrderDependentGain)
    assert wrapped(3) == pytest.approx(0.7 * 4 / 3)
    assert gain_converter(wrapped) is wrapped


def test_gain_converter_rejects_non_numeric_non_callable():
    """A gain that is neither float nor callable raises."""
    with pytest.raises((TypeError, ValueError)):
        gain_converter("not a gain")


@pytest.mark.parametrize(
    "solver_settings_override",
    [PI_CALLABLE_GAINS],
    ids=["pi-callable-gains"],
    indirect=True,
)
class TestPICallableGains:
    def test_config_resolves_callable_gains_at_order(
        self, step_controller
    ):
        """Callable kp/ki resolve at the config's algorithm order."""
        cfg = step_controller.compile_settings
        order = cfg.algorithm_order
        assert cfg.kp == pytest.approx(0.7 * (order + 1) / order)
        assert cfg.ki == pytest.approx(-0.4 * (order + 1) / order)

    def test_settings_dict_preserves_gain_specs(self, step_controller):
        """settings_dict carries the supplied rule, not a float."""
        settings = step_controller.settings_dict
        assert settings["kp"].fn is dirk_default_kp
        assert settings["ki"].fn is dirk_default_ki

    def test_hash_keys_on_rule_and_order(self, step_controller):
        """values_hash keys on the gain rule and the order."""
        cfg = step_controller.compile_settings
        base_hash = cfg.values_hash
        same_rule, _, changed = cfg.update({"kp": dirk_default_kp})
        assert changed == set()
        assert same_rule.values_hash == base_hash
        other_order, _, changed = cfg.update(
            {"algorithm_order": cfg.algorithm_order + 2}
        )
        assert "algorithm_order" in changed
        assert other_order.values_hash != base_hash
        other_rule, _, _ = cfg.update({"kp": dirk_default_ki})
        assert other_rule.values_hash != base_hash

    def test_order_update_reresolves_gains(
        self, step_controller_mutable
    ):
        """An algorithm_order update re-evaluates callable gains."""
        controller = step_controller_mutable
        order = controller.algorithm_order + 2
        controller.update_compile_settings({"algorithm_order": order})
        assert controller.kp == pytest.approx(
            0.7 * (order + 1) / order
        )
        assert controller.ki == pytest.approx(
            -0.4 * (order + 1) / order
        )

    def test_settings_dict_round_trips(self, step_controller):
        """The exported spec re-enters the config unchanged."""
        cfg = step_controller.compile_settings
        settings = step_controller.settings_dict
        replacement, _, changed = cfg.update({"kp": settings["kp"]})
        assert changed == set()
        assert replacement.kp == cfg.kp


@pytest.mark.parametrize(
    "solver_settings_override",
    [PID_CALLABLE_GAINS],
    ids=["pid-callable-gains"],
    indirect=True,
)
def test_pid_config_resolves_callable_kd(step_controller):
    """PID's kd accepts a callable of the order."""
    cfg = step_controller.compile_settings
    assert cfg.kd == pytest.approx(0.05 / cfg.algorithm_order)


def test_dirk_adaptive_defaults_use_order_callable_pi():
    """DIRK adaptive defaults select PI with the order callables."""
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
