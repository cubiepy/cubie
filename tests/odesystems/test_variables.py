"""Tests for cubie.odesystems.variables."""

import numpy as np
import pytest

from cubie._serialize import canonical_digest
from cubie.odesystems.variables import (
    CompiledParameter,
    Parameter,
    ParameterSet,
    State,
    ValuedSet,
    Variable,
    VariableSet,
)


STATES = VariableSet.of(State, {"x": 1.0, "y": 2.0}, {"y": "mV"})

PARAMETERS = VariableSet.of(Parameter, {"a": 1.0, "b": 2.0, "c": 3.0})


# ── VariableSet ──────────────────────────────────────────────────── #

def test_of_builds_records_of_the_given_kind():
    """``of`` builds one record per name, in order, with units."""
    assert STATES.names == ("x", "y")
    assert STATES["y"] == State("y", "mV", 2.0)
    assert STATES.units["y"] == "mV"


def test_bare_names_build_valueless_variables():
    """Bare names build variables with no value."""
    observables = VariableSet.of(Variable, ["v", "w"])
    assert observables.names == ("v", "w")
    assert isinstance(observables["w"], Variable)


def test_valued_kinds_build_valued_sets():
    """States and parameters build sets with defaults; variables do not."""
    assert isinstance(STATES, ValuedSet)
    assert type(VariableSet.of(Variable, ["v"])) is VariableSet


def test_valued_set_rejects_valueless_variables():
    """Building a ValuedSet from a Variable raises TypeError."""
    with pytest.raises(TypeError, match="no default"):
        ValuedSet([Variable("v")])


def test_parameter_set_rejects_uncompiled_parameters():
    """Building a ParameterSet from a Parameter raises TypeError."""
    with pytest.raises(TypeError, match="not compiled parameters"):
        ParameterSet([Parameter("a")])


def test_subset_keeps_the_class():
    """A subset has the class of the set it came from."""
    assert STATES.subset(["y"]).defaults == {"y": 2.0}
    batch = ParameterSet.build(PARAMETERS, ["a"])
    assert isinstance(batch.subset(["a"]), ParameterSet)


def test_repeated_names_raise():
    """A set names each variable once."""
    with pytest.raises(ValueError, match="more than once"):
        VariableSet([Variable("x"), Variable("x")])


@pytest.mark.parametrize(
    "keys, expected",
    [
        ("y", [1]),
        (1, [1]),
        (["y", "x"], [1, 0]),
        ([1, 0], [1, 0]),
        (slice(None), [0, 1]),
        (np.array([1]), [1]),
    ],
    ids=["name", "index", "names", "indices", "slice", "array"],
)
def test_get_indices_resolves_names_and_indices(keys, expected):
    """Names, indices, slices and arrays resolve to row indices."""
    np.testing.assert_array_equal(STATES.get_indices(keys), expected)


def test_get_indices_skips_unknown_names_when_silent():
    """Unknown names raise, or are skipped when silent."""
    with pytest.raises(KeyError, match="z"):
        STATES.get_indices(["z"])
    np.testing.assert_array_equal(
        STATES.get_indices(["y", "z"], silent=True), [1]
    )


@pytest.mark.parametrize(
    "keys, error",
    [([5], IndexError), (["x", 0], TypeError), (1.5, TypeError)],
    ids=["out_of_range", "mixed", "float"],
)
def test_get_indices_rejects_bad_keys(keys, error):
    """Out-of-range indices and unsupported keys raise."""
    with pytest.raises(error):
        STATES.get_indices(keys)


def test_get_labels_returns_names_at_indices():
    assert STATES.get_labels(np.array([1, 0])) == ["y", "x"]


def test_defaults_array_takes_the_precision():
    array = STATES.defaults_array(np.float32)
    assert array.dtype == np.float32
    np.testing.assert_array_equal(array, [1.0, 2.0])


def test_with_defaults_returns_a_changed_copy():
    """``with_defaults`` changes the named defaults on a copy."""
    changed = STATES.with_defaults({"y": 5.0})
    assert changed.defaults == {"x": 1.0, "y": 5.0}
    assert STATES.defaults == {"x": 1.0, "y": 2.0}
    with pytest.raises(KeyError, match="z"):
        STATES.with_defaults({"z": 1.0})


# ── subset and without_values ────────────────────────────────────── #

def test_subset_keeps_the_set_order():
    """A subset lists its variables in the set's order."""
    assert STATES.subset(["y", "x"]).names == ("x", "y")
    assert STATES.subset(["y"])["y"] == STATES["y"]
    with pytest.raises(KeyError, match="z"):
        STATES.subset(["z"])


def test_without_values_keeps_names_and_units():
    """The valueless view keeps names and units, so defaults never
    change it."""
    view = STATES.without_values()
    assert view["y"] == Variable("y", "mV")
    assert STATES.with_defaults({"x": 9.0}).without_values() == view


# ── CompiledParameter ────────────────────────────────────────────── #

@pytest.mark.parametrize(
    "swept, value",
    [(True, 1.0), (False, None)],
    ids=["swept_with_value", "fixed_without_value"],
)
def test_compiled_parameter_is_swept_or_valued(swept, value):
    """A compiled parameter is swept with no value, or fixed at one."""
    with pytest.raises(ValueError, match="swept or have a value"):
        CompiledParameter("a", swept=swept, value=value)


# ── ParameterSet ─────────────────────────────────────────────────── #

def test_swept_names_follow_the_parameter_order():
    """The sweep takes the parameters' order, whatever order it is given."""
    batch = ParameterSet.build(PARAMETERS, ["c", "a"])
    assert batch.swept == ("a", "c")


def test_fixed_holds_every_parameter_that_is_not_swept():
    """Fixed values come from the batch, else from the defaults."""
    batch = ParameterSet.build(PARAMETERS, ["b"], {"c": 9.0})
    assert batch.fixed == {"a": 1.0, "c": 9.0}


def test_reading_fixed_cannot_change_the_set():
    """``fixed`` is built on each read, so writing to it changes nothing."""
    batch = ParameterSet.build(PARAMETERS, ["b"])
    batch.fixed["a"] = 5.0
    assert batch.fixed["a"] == 1.0
    with pytest.raises(AttributeError):
        batch["a"].value = 5.0


def test_batches_compiling_the_same_values_compare_equal():
    """A batch value equal to the default builds an equal set that
    hashes alike."""
    given = ParameterSet.build(PARAMETERS, (), {"a": 1.0})
    default = ParameterSet.build(PARAMETERS)
    assert given == default
    assert hash(given) == hash(default)


@pytest.mark.parametrize(
    "swept, values, error",
    [
        (["a", "a"], None, ValueError),
        (["z"], None, KeyError),
        ((), {"z": 1.0}, KeyError),
        (["a"], {"a": 1.0}, ValueError),
    ],
    ids=["repeat", "unknown_swept", "unknown_value", "swept_and_fixed"],
)
def test_build_rejects_invalid_batches(swept, values, error):
    with pytest.raises(error):
        ParameterSet.build(PARAMETERS, swept, values)


def test_identity_covers_the_sweep_and_compiled_in_values():
    """Changing the sweep or a compiled-in value changes the identity,
    and changing a swept parameter's default does not."""
    batch = ParameterSet.build(PARAMETERS, ["a"])
    assert canonical_digest(batch) != canonical_digest(
        ParameterSet.build(PARAMETERS)
    )
    assert canonical_digest(
        ParameterSet.build(PARAMETERS, ["a"], {"b": 7.0})
    ) != canonical_digest(batch)
    assert canonical_digest(
        ParameterSet.build(PARAMETERS.with_defaults({"a": 7.0}), ["a"])
    ) == canonical_digest(batch)
