"""Tests for cubie.odesystems.symbolic.specialiser."""

import numpy as np
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
        parsed,
        np.float32,
        ParameterSet.build(parsed.declared_parameters, ["k"]),
    )


def test_products_assemble_on_demand(specialiser):
    """Build the specialised system on the first read of a product."""
    assert specialiser.cache_valid is False
    assert specialiser.layout.swept_names == ("k",)
    assert specialiser.cache_valid is True
    assert specialiser.fixed_parameter_values == {"c0": 1.0}
    assert specialiser.mass is None
    assert specialiser.fn_hash == parsed_hash(specialiser)


def parsed_hash(specialiser):
    """Return the source hash of a fresh assembly of the same request."""
    return specialiser._parsed_system.assemble(
        specialiser.compiled_parameters
    ).fn_hash


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
        precision=np.float32,
        compiled_parameters=ParameterSet.build(declared, ["k"]),
    )
    _, _, changed = config.update(
        compiled_parameters=ParameterSet.build(
            declared.with_defaults({"k": 3.0}), ["k"]
        )
    )
    assert changed == set()
    refixed, _, changed = config.update(
        compiled_parameters=ParameterSet.build(declared, (), {"k": 0.5})
    )
    assert changed == {"compiled_parameters"}
    assert refixed.values_hash != config.values_hash


def test_key_for_matches_the_built_set(parsed):
    """Return the key of the set the same request builds."""
    declared = parsed.declared_parameters
    built = ParameterSet.build(declared, ["k"], {"c0": 2.0})
    assert (
        ParameterSet.key_for(declared, ["k"], {"c0": 2.0})
        == built._cubie_canonical_()
    )
    assert ParameterSet.key_for(declared, ["k"], {"c0": 1.0}) == (
        ParameterSet.build(declared, ["k"])._cubie_canonical_()
    )
    with pytest.raises(KeyError, match="not_a_key"):
        ParameterSet.key_for(declared, ["not_a_key"])
    with pytest.raises(ValueError, match="swept"):
        ParameterSet.key_for(declared, ["k"], {"k": 1.0})
