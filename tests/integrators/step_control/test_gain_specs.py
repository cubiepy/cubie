"""Tests for float-or-callable controller gain specifications."""

import numpy as np
import pytest

from cubie.integrators.algorithms.generic_dirk import (
    dirk_default_integral_gain,
    dirk_default_proportional_gain,
)
from cubie.integrators.step_control import get_controller
from cubie.integrators.step_control.adaptive_step_controller import (
    OrderDependentGain,
    gain_converter,
)
from cubie.integrators.step_control.base_step_controller import (
    FILTER_COEFFICIENT_PRESETS,
    filter_coefficients_to_gains,
)


def half_over_order(order):
    """Reference derivative_gain callable for the PID case."""
    return 0.05 / order


PI_CALLABLE_GAINS = {
    "step_controller": "pi",
    "integral_gain": dirk_default_integral_gain,
    "proportional_gain": dirk_default_proportional_gain,
}
PID_CALLABLE_GAINS = {
    "step_controller": "pid",
    "integral_gain": dirk_default_integral_gain,
    "proportional_gain": dirk_default_proportional_gain,
    "derivative_gain": half_over_order,
}


def test_gain_converter_constant_and_callable():
    """gain_converter passes floats through and wraps callables."""
    assert gain_converter(0.7) == 0.7
    wrapped = gain_converter(dirk_default_integral_gain)
    assert isinstance(wrapped, OrderDependentGain)
    assert wrapped(3) == pytest.approx(dirk_default_integral_gain(3))
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
        """Callable gains resolve at the config's algorithm order."""
        cfg = step_controller.compile_settings
        order = cfg.algorithm_order
        assert cfg.integral_gain == pytest.approx(
            dirk_default_integral_gain(order)
        )
        assert cfg.proportional_gain == pytest.approx(
            dirk_default_proportional_gain(order)
        )

    def test_hash_keys_on_rule_and_order(self, step_controller):
        """values_hash keys on the gain rule and the order."""
        cfg = step_controller.compile_settings
        base_hash = cfg.values_hash
        same_rule, _, changed = cfg.update(
            {"integral_gain": dirk_default_integral_gain}
        )
        assert changed == set()
        assert same_rule.values_hash == base_hash
        other_order, _, changed = cfg.update(
            {"algorithm_order": cfg.algorithm_order + 2}
        )
        assert "algorithm_order" in changed
        assert other_order.values_hash != base_hash
        other_rule, _, _ = cfg.update(
            {"integral_gain": dirk_default_proportional_gain}
        )
        assert other_rule.values_hash != base_hash

    def test_order_update_reresolves_gains(
        self, step_controller_mutable
    ):
        """An algorithm_order update re-evaluates callable gains."""
        controller = step_controller_mutable
        order = controller.algorithm_order + 2
        controller.update_compile_settings({"algorithm_order": order})
        assert controller.integral_gain == pytest.approx(
            dirk_default_integral_gain(order)
        )
        assert controller.proportional_gain == pytest.approx(
            dirk_default_proportional_gain(order)
        )

    def test_wrapped_gain_spec_round_trips(self, step_controller):
        """A wrapped gain rule re-enters the config unchanged."""
        cfg = step_controller.compile_settings
        replacement, _, changed = cfg.update(
            {"integral_gain": gain_converter(dirk_default_integral_gain)}
        )
        assert changed == set()
        assert replacement.integral_gain == cfg.integral_gain


@pytest.mark.parametrize(
    "solver_settings_override",
    [PID_CALLABLE_GAINS],
    ids=["pid-callable-gains"],
    indirect=True,
)
def test_pid_config_resolves_callable_derivative_gain(step_controller):
    """PID's derivative_gain accepts a callable of the order."""
    cfg = step_controller.compile_settings
    assert cfg.derivative_gain == pytest.approx(
        half_over_order(cfg.algorithm_order)
    )


# ── filter_coefficients translation ─────────────────────────────── #


def test_filter_coefficients_to_gains_inverts_petsc_map():
    """The tuple form inverts PETSc's gain-to-exponent formula."""
    gains = filter_coefficients_to_gains((0.7, -0.4, 0.0))
    assert gains["integral_gain"] == pytest.approx(0.3)
    assert gains["proportional_gain"] == pytest.approx(0.4)
    assert gains["derivative_gain"] == pytest.approx(0.0)
    beta = (1.0 / 18.0, 1.0 / 9.0, 1.0 / 18.0)
    gains = filter_coefficients_to_gains(beta)
    assert gains["integral_gain"] == pytest.approx(
        beta[0] + beta[1] + beta[2]
    )
    assert gains["proportional_gain"] == pytest.approx(
        -beta[1] - 2.0 * beta[2]
    )
    assert gains["derivative_gain"] == pytest.approx(beta[2])


@pytest.mark.parametrize("preset", sorted(FILTER_COEFFICIENT_PRESETS))
def test_filter_coefficients_presets_round_trip(preset):
    """Every preset's gains map back to the tabulated exponents."""
    beta = FILTER_COEFFICIENT_PRESETS[preset]
    gains = filter_coefficients_to_gains(preset)
    integral = gains["integral_gain"]
    proportional = gains["proportional_gain"]
    derivative = gains["derivative_gain"]
    assert integral + proportional + derivative == pytest.approx(beta[0])
    assert -(proportional + 2.0 * derivative) == pytest.approx(beta[1])
    assert derivative == pytest.approx(beta[2])


def test_filter_coefficients_preset_case_insensitive():
    """Preset lookup ignores case."""
    assert filter_coefficients_to_gains("PI34") == (
        filter_coefficients_to_gains("pi34")
    )


def test_filter_coefficients_unknown_preset_raises():
    """An unknown preset name raises with the known names listed."""
    with pytest.raises(ValueError, match="Unknown filter_coefficients"):
        filter_coefficients_to_gains("no-such-preset")


def test_filter_coefficients_wrong_length_raises():
    """A tuple without exactly three entries raises."""
    with pytest.raises(ValueError, match="three numbers"):
        filter_coefficients_to_gains((0.7, -0.4))


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("fixed", ()),
        ("gustafsson", ()),
        ("i", ("integral_gain",)),
        ("pi", ("integral_gain", "proportional_gain")),
        ("pid", ("integral_gain", "proportional_gain", "derivative_gain")),
    ],
)
def test_gain_names_follow_config_fields(kind, expected):
    """gain_names lists the gain fields the controller config declares."""
    settings = {"step_controller": kind}
    if kind == "fixed":
        settings["dt"] = 0.01
    controller = get_controller(np.float64, settings)
    assert controller.gain_names == expected


def test_pid_controller_accepts_filter_coefficients():
    """A beta triple lands as the equivalent gains on a PID config."""
    controller = get_controller(
        np.float64,
        {
            "step_controller": "pid",
            "filter_coefficients": (0.7, -0.4, 0.0),
        },
    )
    assert controller.integral_gain == pytest.approx(0.3)
    assert controller.proportional_gain == pytest.approx(0.4)
    assert controller.derivative_gain == pytest.approx(0.0)


def test_pi_controller_accepts_preset_string():
    """A named preset configures the PI gains."""
    controller = get_controller(
        np.float64,
        {"step_controller": "pi", "filter_coefficients": "PI42"},
    )
    assert controller.integral_gain == pytest.approx(0.4)
    assert controller.proportional_gain == pytest.approx(0.2)


def test_i_controller_accepts_pure_integral_filter():
    """The basic preset reaches the I controller as integral_gain."""
    controller = get_controller(
        np.float64,
        {"step_controller": "i", "filter_coefficients": "basic"},
    )
    assert controller.integral_gain == pytest.approx(1.0)


def test_i_controller_rejects_proportional_filter():
    """A filter needing a proportional gain raises on the I controller."""
    with pytest.raises(ValueError, match="proportional_gain"):
        get_controller(
            np.float64,
            {"step_controller": "i", "filter_coefficients": "PI34"},
        )


def test_pi_controller_rejects_derivative_filter():
    """A filter needing a derivative gain raises on the PI controller."""
    with pytest.raises(ValueError, match="derivative_gain"):
        get_controller(
            np.float64,
            {"step_controller": "pi", "filter_coefficients": "H312PID"},
        )


def test_filter_coefficients_conflicts_with_explicit_gains():
    """filter_coefficients alongside an explicit gain raises."""
    with pytest.raises(ValueError, match="cannot be combined"):
        get_controller(
            np.float64,
            {
                "step_controller": "pid",
                "filter_coefficients": "PI34",
                "integral_gain": 0.5,
            },
        )


def test_update_accepts_filter_coefficients():
    """update() translates filter_coefficients into new gains."""
    controller = get_controller(
        np.float64, {"step_controller": "pid"}
    )
    recognised = controller.update(
        {"filter_coefficients": (0.6, -0.2, 0.0)}
    )
    assert "filter_coefficients" in recognised
    assert controller.integral_gain == pytest.approx(0.4)
    assert controller.proportional_gain == pytest.approx(0.2)
    assert controller.derivative_gain == pytest.approx(0.0)


def test_gustafsson_warns_on_filter_coefficients_update():
    """A gainless controller warns and ignores filter_coefficients."""
    controller = get_controller(
        np.float64, {"step_controller": "gustafsson"}
    )
    with pytest.warns(UserWarning, match="filter_coefficients"):
        controller.update({"filter_coefficients": "PI34"})
