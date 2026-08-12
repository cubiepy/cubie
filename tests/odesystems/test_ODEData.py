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
from cubie.odesystems.operation_ordering import (
    CACHED_AUX_FAMILIES,
    OPERATION_ORDERING_FAMILIES,
    OperationOrderingMap,
    normalize_operation_ordering,
    resolve_operation_ordering,
)


# ── SystemSizes ───────────────────────────────────────────────── #

def test_system_sizes_construction():
    """All fields stored correctly on frozen attrs class."""
    sizes = SystemSizes(
        states=3, observables=2, parameters=4, constants=5, drivers=1,
    )
    assert sizes.states == 3
    assert sizes.observables == 2
    assert sizes.parameters == 4
    assert sizes.constants == 5
    assert sizes.drivers == 1


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("states", 1.5),
        ("observables", "x"),
        ("parameters", None),
        ("constants", [1]),
        ("drivers", 2.0),
    ],
    ids=["states", "observables", "parameters", "constants", "drivers"],
)
def test_system_sizes_validates_int(field, bad_value):
    """Each field rejects non-int values."""
    kwargs = dict(states=1, observables=1, parameters=1, constants=1, drivers=1)
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
        default_parameters={"a": 0.5, "b": 0.3},
        default_constants={"g": 9.81},
        default_observable_names={"v": 0.0, "w": 0.0},
        num_drivers=num_drivers,
        operation_ordering=operation_ordering,
    )


def test_odedata_construction():
    """ODEData stores SystemValues for each component."""
    data = _make_odedata()
    assert data.initial_states.n == 2
    assert data.parameters.n == 2
    assert data.constants.n == 1
    assert data.observables.n == 2
    assert data.operation_ordering == "kahn"


def test_operation_ordering_public_values_are_exact():
    """The compile setting exposes only the approved policy names."""
    assert OPERATION_ORDERINGS == (
        "kahn",
        "greedy",
        "dfs",
        "liveness_auto",
    )


def test_operation_ordering_families_are_exact():
    """Only concrete generated scheduling families are public keys."""
    assert OPERATION_ORDERING_FAMILIES == (
        "dxdt",
        "observables",
        "linear_operator",
        "linear_operator_cached",
        "linear_operator_at_state",
        "neumann_preconditioner",
        "neumann_preconditioner_cached",
        "neumann_preconditioner_at_state",
        "jacobi_preconditioner",
        "jacobi_preconditioner_cached",
        "jacobi_preconditioner_at_state",
        "evaluate_inv_mass_f",
        "stage_residual",
        "n_stage_residual",
        "n_stage_linear_operator",
        "n_stage_neumann_preconditioner",
        "n_stage_jacobi_preconditioner",
        "prepare_jac",
        "calculate_cached_jvp",
        "time_derivative_rhs",
    )


def test_operation_ordering_map_is_canonical_immutable_and_hashable():
    """Mapping input is copied to sparse immutable compile state."""
    supplied = {
        "stage_residual": "dfs",
        "dxdt": "kahn",
    }
    data = _make_odedata(operation_ordering=supplied)
    supplied["stage_residual"] = "greedy"

    assert isinstance(data.operation_ordering, OperationOrderingMap)
    assert data.operation_ordering.overrides == (
        ("stage_residual", "dfs"),
    )
    assert hash(data.operation_ordering)
    with pytest.raises(TypeError):
        data.operation_ordering.overrides[0] = (
            "stage_residual",
            "greedy",
        )


def test_operation_ordering_sparse_map_defaults_to_kahn():
    """Omitted family keys resolve to stable Kahn ordering."""
    configured = normalize_operation_ordering(
        {"n_stage_residual": "liveness_auto"}
    )
    assert resolve_operation_ordering(
        configured,
        "n_stage_residual",
    ) == "liveness_auto"
    assert resolve_operation_ordering(configured, "dxdt") == "kahn"


def test_operation_ordering_map_has_one_canonical_identity():
    """Input order and explicit Kahn entries do not alter identity."""
    first = _make_odedata(
        operation_ordering={
            "stage_residual": "dfs",
            "dxdt": "kahn",
        }
    )
    second = _make_odedata(
        operation_ordering={
            "dxdt": "kahn",
            "stage_residual": "dfs",
        }
    )
    assert first.operation_ordering == second.operation_ordering
    assert first.values_hash == second.values_hash


def test_operation_ordering_map_rejects_unknown_family():
    """Only generated function families are accepted as map keys."""
    with pytest.raises(
        ValueError,
        match="Unknown operation_ordering families: 'analytical_jvp'",
    ):
        _make_odedata(
            operation_ordering={"analytical_jvp": "dfs"}
        )


def test_operation_ordering_map_rejects_invalid_family_method():
    """Every family override must name an approved method."""
    with pytest.raises(
        ValueError,
        match=(
            "Invalid operation_ordering method for family "
            "'stage_residual': 'bogus'"
        ),
    ):
        _make_odedata(
            operation_ordering={"stage_residual": "bogus"}
        )


def test_operation_ordering_rejects_non_string_non_mapping():
    """The frozen compile boundary rejects other input shapes."""
    with pytest.raises(TypeError, match="must be a supported method"):
        _make_odedata(operation_ordering=("dxdt", "dfs"))


def test_cached_aux_families_are_exact():
    """The shared-buffer cluster names exactly the cached kinds."""
    assert CACHED_AUX_FAMILIES == (
        "linear_operator_cached",
        "neumann_preconditioner_cached",
        "jacobi_preconditioner_cached",
        "prepare_jac",
        "calculate_cached_jvp",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"prepare_jac": "dfs"},
        {"calculate_cached_jvp": "greedy"},
        {"prepare_jac": "dfs", "calculate_cached_jvp": "dfs"},
        {
            "prepare_jac": "dfs",
            "calculate_cached_jvp": "dfs",
            "linear_operator_cached": "dfs",
            "neumann_preconditioner_cached": "dfs",
            "jacobi_preconditioner_cached": "greedy",
        },
    ],
)
def test_operation_ordering_rejects_split_cached_aux_cluster(
    overrides,
):
    """Cached-aux families must not resolve to differing methods."""
    with pytest.raises(
        ValueError,
        match="sharing cached Jacobian auxiliaries",
    ):
        _make_odedata(operation_ordering=overrides)


def test_operation_ordering_accepts_uniform_cached_aux_cluster():
    """One method across the whole cached-aux cluster is accepted."""
    uniform = {family: "dfs" for family in CACHED_AUX_FAMILIES}
    data = _make_odedata(operation_ordering=uniform)
    for family in CACHED_AUX_FAMILIES:
        assert resolve_operation_ordering(
            data.operation_ordering, family
        ) == "dfs"
    assert resolve_operation_ordering(
        data.operation_ordering, "dxdt"
    ) == "kahn"


def test_operation_ordering_empty_and_all_kahn_maps_collapse():
    """Maps with no non-Kahn overrides normalize to the string."""
    assert normalize_operation_ordering({}) == "kahn"
    assert normalize_operation_ordering({"dxdt": "kahn"}) == "kahn"
    data = _make_odedata(operation_ordering={})
    assert data.operation_ordering == "kahn"
    assert data.values_hash == _make_odedata().values_hash


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
    assert replacement.constants.precision == np.float64
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
        ("num_parameters", 2),
        ("num_constants", 1),
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
    assert sizes.parameters == 2
    assert sizes.constants == 1
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
        default_constants=None,
        default_observable_names=None,
        num_drivers=0,
    )
    assert data.num_states == 1
    assert data.num_drivers == 0
    assert data.parameters.n == 0
    assert data.constants.n == 0
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
