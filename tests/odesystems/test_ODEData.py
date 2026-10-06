"""Tests for cubie.odesystems.ODEData."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from cubie.odesystems.ODEData import (
    ODEData,
    OPERATION_ORDERINGS,
    SystemSizes,
)


# ── SystemSizes ───────────────────────────────────────────────── #

def test_system_sizes_construction():
    """All fields stored correctly on frozen attrs class."""
    sizes = SystemSizes(
        states=3, observables=2, swept_parameters=4, drivers=1,
    )
    assert sizes.states == 3
    assert sizes.observables == 2
    assert sizes.swept_parameters == 4
    assert sizes.drivers == 1


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("states", 1.5),
        ("observables", "x"),
        ("swept_parameters", None),
        ("drivers", 2.0),
    ],
    ids=["states", "observables", "swept_parameters", "drivers"],
)
def test_system_sizes_validates_int(field, bad_value):
    """Each field rejects non-int values."""
    kwargs = dict(
        states=1,
        observables=1,
        swept_parameters=1,
        drivers=1,
    )
    kwargs[field] = bad_value
    with pytest.raises(TypeError):
        SystemSizes(**kwargs)


# ── ODEData construction ──────────────────────────────────────── #

def _make_odedata(
    precision=np.float32,
    num_drivers=1,
    operation_ordering="kahn",
):
    """Helper to create ODEData via from_BaseODE_initargs."""
    return ODEData.from_BaseODE_initargs(
        precision=precision,
        default_initial_values={"x": 0.0, "y": 1.0},
        default_parameters={"a": 0.5, "b": 0.3, "g": 9.81},
        default_observable_names={"v": 0.0, "w": 0.0},
        num_drivers=num_drivers,
        operation_ordering=operation_ordering,
    )


def test_odedata_construction():
    """ODEData stores SystemValues for each component."""
    data = _make_odedata()
    assert data.initial_states.n == 2
    assert data.parameters.n == 3
    assert data.observables.n == 2


def test_operation_ordering_public_values_are_exact():
    """The compile setting exposes only the approved policy names."""
    assert OPERATION_ORDERINGS == (
        "kahn",
        "greedy",
        "dfs",
        "liveness_auto",
    )


@pytest.mark.parametrize(
    "operation_ordering",
    ["greedy", "dfs", "liveness_auto"],
)
def test_odedata_operation_ordering_participates_in_identity(
    operation_ordering,
):
    """Every opt-in ordering is validated and compile-critical."""
    kahn = _make_odedata(operation_ordering="kahn")
    alternative = _make_odedata(
        operation_ordering=operation_ordering
    )
    assert alternative.operation_ordering == operation_ordering
    assert alternative.values_hash != kahn.values_hash


@pytest.mark.parametrize("operation_ordering", ["liveness", "bogus"])
def test_odedata_rejects_invalid_operation_ordering(operation_ordering):
    """Only the four supported ordering policies are accepted."""
    with pytest.raises(ValueError, match="operation_ordering"):
        _make_odedata(operation_ordering=operation_ordering)


# ── ODEData.update_precisions ─────────────────────────────────── #

def test_update_precision_propagates_to_all_containers():
    """A precision update re-materialises every SystemValues container."""
    data = _make_odedata(precision=np.float32)
    replacement, recognized, changed = data.update(
        {"precision": np.float64}
    )
    assert "precision" in changed
    assert replacement.parameters.precision == np.float64
    assert replacement.initial_states.precision == np.float64
    assert replacement.observables.precision == np.float64
    assert replacement.parameters.values_array.dtype == np.float64
    # Original snapshot untouched
    assert data.parameters.precision == np.float32


def test_update_precision_noop_without_key():
    """Precision stays unchanged when the key is absent."""
    data = _make_odedata(precision=np.float32)
    replacement, _, changed = data.update({"unrelated": 42})
    assert changed == set()
    assert replacement is data
    assert replacement.parameters.precision == np.float32


# ── ODEData properties ────────────────────────────────────────── #

@pytest.mark.parametrize(
    "prop, expected",
    [
        ("num_states", 2),
        ("num_observables", 2),
        ("num_parameters", 3),
        ("num_swept_parameters", 0),
    ],
)
def test_odedata_count_properties(prop, expected):
    """Count properties delegate to the correct SystemValues.n."""
    data = _make_odedata()
    assert getattr(data, prop) == expected


def test_odedata_sizes_returns_system_sizes():
    """sizes property returns SystemSizes with all counts."""
    data = _make_odedata(num_drivers=3)
    sizes = data.sizes
    assert sizes.states == 2
    assert sizes.observables == 2
    assert sizes.swept_parameters == 0
    assert sizes.drivers == 3


def test_odedata_mass_returns_stored_value():
    """mass property returns the _mass field."""
    data = _make_odedata()
    assert data.mass is None


def test_mass_change_alters_values_hash():
    """A mass change moves values_hash (forcing recompilation) while an
    equal mass leaves it unchanged, across None/ndarray/sympy.Matrix."""
    data = _make_odedata()  # _mass defaults to None
    baseline = data.values_hash

    data, _, _ = data.update({"mass": np.eye(2, dtype=np.float64)})
    hash_identity = data.values_hash
    assert hash_identity != baseline  # None -> ndarray recompiles

    replacement, _, changed = data.update(
        {"mass": np.eye(2, dtype=np.float64)}
    )
    assert changed == set()  # equal ndarray: no recompile
    assert replacement.values_hash == hash_identity

    data, _, _ = data.update({"mass": np.diag([1.0, 2.0])})
    hash_diag = data.values_hash
    assert hash_diag != hash_identity  # different ndarray recompiles

    data, _, _ = data.update({"mass": sp.Matrix([[1, 0], [0, 3]])})
    assert data.values_hash != hash_diag  # sympy.Matrix participates

    # Input-form independence: a sympy matrix and its numeric array
    # normalise to the same stored mass and the same hash.
    via_sympy, _, _ = data.update({"mass": sp.Matrix([[2, 0], [0, 5]])})
    via_array, _, _ = data.update(
        {"mass": np.diag([2.0, 5.0])}
    )
    assert via_sympy.values_hash == via_array.values_hash


# ── ODEData.from_BaseODE_initargs ─────────────────────────────── #

def test_from_base_ode_initargs_handles_none_optional():
    """Factory handles None for optional arguments gracefully."""
    data = ODEData.from_BaseODE_initargs(
        precision=np.float32,
        default_initial_values={"x": 1.0},
        default_parameters=None,
        default_observable_names=None,
        num_drivers=0,
    )
    assert data.num_states == 1
    assert data.num_drivers == 0
    assert data.parameters.n == 0
    assert data.observables.n == 0


def test_from_base_ode_initargs_overrides_defaults():
    """User values override defaults in from_BaseODE_initargs."""
    data = ODEData.from_BaseODE_initargs(
        precision=np.float64,
        initial_values={"x": 5.0},
        default_initial_values={"x": 0.0, "y": 1.0},
        default_parameters={"a": 0.5},
    )
    # User override for x should apply; y keeps default
    assert data.num_states == 2
    val = data.initial_states.values_dict["x"]
    assert float(val) == pytest.approx(5.0)


# ── Swept parameters ──────────────────────────────────────────── #

def test_default_fixes_every_parameter():
    """With nothing swept every parameter is compiled in."""
    data = _make_odedata()
    assert data.swept_parameters == ()
    assert data.fixed_parameter_values == pytest.approx(
        {"a": 0.5, "b": 0.3, "g": 9.81}
    )


def test_swept_names_keep_their_order():
    """Swept names keep their order and the rest are compiled in."""
    data, _, changed = _make_odedata().update(
        swept_parameters=["b", "a"], fixed_parameters={"g": 9.81}
    )
    assert changed == {"swept_parameters", "fixed_parameters"}
    assert data.swept_parameters == ("b", "a")
    assert data.fixed_parameter_values == pytest.approx({"g": 9.81})
    assert data.sizes.swept_parameters == 2


def test_swept_names_reject_repeats_and_unknowns():
    """Swept and fixed names cover each known parameter once."""
    data = _make_odedata()
    with pytest.raises(ValueError, match="repeat"):
        data.update(
            swept_parameters=["a", "a"],
            fixed_parameters={"b": 0.3, "g": 9.81},
        )
    with pytest.raises(ValueError, match="once"):
        data.update(swept_parameters=["nope"])


def test_compiled_in_values_participate_in_identity():
    """Fixed values and swept names enter values_hash, defaults don't."""
    data = _make_odedata()
    swept = data.update(
        swept_parameters=["a"], fixed_parameters={"b": 0.3, "g": 9.81}
    )[0]
    refixed = data.update(
        fixed_parameters={"a": 0.7, "b": 0.3, "g": 9.81}
    )[0]
    parameters = swept.parameters.copy()
    parameters.update_from_dict({"a": 0.7})
    new_default = swept.update(parameters=parameters)[0]
    assert swept.values_hash != data.values_hash
    assert refixed.values_hash != data.values_hash
    assert new_default.values_hash == swept.values_hash
