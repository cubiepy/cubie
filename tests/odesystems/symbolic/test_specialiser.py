"""Tests for cubie.odesystems.symbolic.specialiser."""

import pytest

from cubie.odesystems.symbolic.parsing import parse_system
from cubie.odesystems.symbolic.specialiser import (
    SpecialiserConfig,
    SystemSpecialiser,
)
from cubie.odesystems.variables import ParameterSet


@pytest.fixture
def parsed():
    """Return a parsed two-parameter system."""
    return parse_system(
        dxdt=["dx = -k * x + c0"],
        states={"x": 1.0},
        parameters={"k": 0.5, "c0": 1.0},
    )


@pytest.fixture
def specialiser(parsed):
    """Return a specialiser sweeping ``k``."""
    return SystemSpecialiser(
        parsed, ParameterSet.build(parsed.declared_parameters, ["k"])
    )


def test_products_assemble_on_demand(parsed, specialiser):
    """Build the specialised system on the first read of a product."""
    assert specialiser.cache_valid is False
    assert specialiser.layout.swept_names == ("k",)
    assert specialiser.cache_valid is True
    assert specialiser.fixed_parameter_values == {
        "c0": parsed.declared_parameters.defaults["c0"]
    }
    assert specialiser.mass is None


def test_equal_compiled_parameters_keep_the_build(parsed, specialiser):
    """Keep the build when the new set equals the compiled one."""
    specialiser.layout
    same = ParameterSet.build(parsed.declared_parameters, ["k"], {"c0": 1.0})
    recognised = specialiser.update(compiled_parameters=same)
    assert recognised == {"compiled_parameters"}
    assert specialiser.cache_valid is True


def test_new_compiled_parameters_invalidate_the_build(parsed, specialiser):
    """Invalidate the assembled system for a new sweep or fixed value."""
    hash_before = specialiser.config_hash
    layout_before = specialiser.layout
    specialiser.update(
        compiled_parameters=ParameterSet.build(
            parsed.declared_parameters, ["k"], {"c0": 4.0}
        )
    )
    assert specialiser.cache_valid is False
    assert specialiser.config_hash != hash_before
    assert specialiser.layout is not layout_before
    assert specialiser.fixed_parameter_values == {"c0": 4.0}


def test_a_swept_default_is_outside_the_identity(parsed):
    """Keep the settings equal when a swept parameter's default changes."""
    declared = parsed.declared_parameters
    config = SpecialiserConfig(
        compiled_parameters=ParameterSet.build(declared, ["k"])
    )
    _, _, changed = config.update(
        compiled_parameters=ParameterSet.build(
            declared.with_defaults({"k": 3.0}), ["k"]
        )
    )
    assert changed == set()
    refixed, _, changed = config.update(
        compiled_parameters=ParameterSet.build(
            declared, (), {"k": declared.defaults["k"]}
        )
    )
    assert changed == {"compiled_parameters"}
    assert refixed.values_hash != config.values_hash
