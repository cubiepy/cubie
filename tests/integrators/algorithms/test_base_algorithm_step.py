"""Tests for cubie.integrators.algorithms.base_algorithm_step."""

import pytest

from cubie.integrators.algorithms import algorithm_is_adaptive
from cubie.integrators.algorithms.base_algorithm_step import (
    ButcherTableau,
)
from cubie.integrators.algorithms.explicit_euler import ExplicitEulerStep
from cubie.integrators.algorithms.generic_erk_tableaus import ERKTableau


def test_tableau_b_hat_length_mismatch_raises():
    """b_hat length must equal the number of stages in b."""
    with pytest.raises(ValueError, match="b_hat must match"):
        ButcherTableau(
            a=((0.0, 0.0), (1.0, 0.0)),
            b=(0.5, 0.5),
            c=(0.0, 1.0),
            order=2,
            b_hat=(1.0, 0.0, 0.0),
        )


def test_erk_tableau_b_hat_sum_not_one_raises():
    """RK-family tableaus reject b_hat weights that do not sum to one."""
    with pytest.raises(ValueError, match="b_hat must sum to one"):
        ERKTableau(
            a=((0.0, 0.0), (1.0, 0.0)),
            b=(0.5, 0.5),
            b_hat=(0.5, 0.6),
            c=(0.0, 1.0),
            order=2,
        )


def test_erk_tableau_b_sum_not_one_raises():
    """RK-family tableaus reject b weights that do not sum to one."""
    with pytest.raises(ValueError, match="b must sum to one"):
        ERKTableau(
            a=((0.0, 0.0), (1.0, 0.0)),
            b=(0.5, 0.6),
            c=(0.0, 1.0),
            order=2,
        )


def test_typed_rows_pads_short_rows():
    """typed_rows zero-pads rows shorter than the tableau's stage count."""
    tableau = ButcherTableau(
        a=((0.0,), (0.5, 0.5)),
        b=(0.5, 0.5),
        c=(0.0, 1.0),
        order=2,
    )
    typed = tableau.typed_rows(tableau.a, float)
    assert typed == ((0.0, 0.0), (0.5, 0.5))


def test_config_first_same_as_last_false_without_tableau(step_object):
    """BaseStepConfig.first_same_as_last is False for non-tableau steps."""
    assert step_object.tableau is None
    assert step_object.compile_settings.first_same_as_last is False


def test_config_can_reuse_accepted_start_false_without_tableau(step_object):
    """can_reuse_accepted_start is False for non-tableau steps."""
    assert step_object.compile_settings.can_reuse_accepted_start is False


def test_update_with_no_changes_returns_empty_set(precision):
    """update() with no arguments returns an empty set without error."""
    step = ExplicitEulerStep(precision=precision, n=2)
    assert step.update() == set()
    assert step.update(updates_dict={}) == set()


def test_update_warns_on_valid_but_inapplicable_parameter(precision):
    """A recognised algorithm-step parameter unused by this algorithm

    warns instead of raising, and is reported as recognised.
    """
    step = ExplicitEulerStep(precision=precision, n=2)
    with pytest.warns(UserWarning, match="not recognized by"):
        recognised = step.update(beta=0.5)
    assert "beta" in recognised


def test_update_raises_on_truly_invalid_parameter(precision):
    """An entirely unknown update key raises KeyError."""
    step = ExplicitEulerStep(precision=precision, n=2)
    with pytest.raises(KeyError, match="Unrecognized parameters"):
        step.update(not_a_real_parameter=1)


def test_n_drivers_property(step_object, system):
    """n_drivers returns the configured driver count."""
    assert system.sizes.drivers > 0
    assert step_object.n_drivers == system.sizes.drivers


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("euler", False),
        ("backwards_euler", False),
        ("backwards_euler_pc", False),
        ("crank_nicolson", True),
        ("tsit5", True),
        ("implicit_midpoint", False),
        ("radau_iia_5", True),
        ("ros3p", True),
    ],
)
def test_algorithm_is_adaptive_by_alias(alias, expected):
    """algorithm_is_adaptive reports the embedded-estimate flag."""
    assert algorithm_is_adaptive(alias) is expected


def test_algorithm_is_adaptive_family_alias_raises():
    """A bare family alias has no tableau, so adaptivity is undefined."""
    with pytest.raises(ValueError, match="algorithm family"):
        algorithm_is_adaptive("dirk")


def test_algorithm_is_adaptive_unknown_alias_raises():
    """An unregistered alias raises KeyError."""
    with pytest.raises(KeyError):
        algorithm_is_adaptive("not_an_algorithm")
