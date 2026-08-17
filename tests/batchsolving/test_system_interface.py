"""Tests for cubie.batchsolving.SystemInterface."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubie.batchsolving.SystemInterface import SystemInterface


# ── __init__ and from_system ─────────────────────────────── #

def test_init_stores_system_values(system_interface, system):
    """__init__ stores parameters, states, observables as SystemValues."""
    assert system_interface.parameters is system.parameters
    assert system_interface.states is system.initial_values
    assert system_interface.observables is system.observables


def test_from_system_creates_interface(system):
    """from_system wraps system.parameters, initial_values, observables."""
    si = SystemInterface(system)
    assert si.parameters is system.parameters
    assert si.states is system.initial_values
    assert si.observables is system.observables



# ── update ───────────────────────────────────────────────── #

def test_update_returns_none_when_no_updates(system_interface):
    """Returns None when updates is None and no kwargs."""
    result = system_interface.update(None)
    assert result is None


def test_update_returns_none_when_empty(system_interface):
    """Returns None when updates is empty dict and no kwargs."""
    result = system_interface.update({})
    assert result is None


def test_update_merges_kwargs(system_interface_mutable, system):
    """Merges kwargs into updates dict, writing the live system."""
    name = system.initial_values.names[0]
    recognized = system_interface_mutable.update(None, **{name: 99.0})
    assert name in recognized
    assert system_interface_mutable.states.values_dict[name] == 99.0
    # The interface is a live view onto the system's containers.
    assert system.initial_values.values_dict[name] == 99.0


def test_update_merges_kwargs_into_updates_dict(
    system_interface_mutable, system
):
    """Merges kwargs into a provided updates dict and applies values."""
    name = system.parameters.names[0]
    original = system_interface_mutable.parameters.values_dict[name]
    new_val = original + 1.0
    recognized = system_interface_mutable.update({}, **{name: new_val})
    assert name in recognized
    assert (
        system_interface_mutable.parameters.values_dict[name] == new_val
    )


def test_update_applies_to_parameters_and_states(
    system_interface_mutable, system
):
    """Attempts update on both parameters and states."""
    state_name = system.initial_values.names[0]
    param_name = system.parameters.names[0]
    recognized = system_interface_mutable.update(
        {state_name: 42.0, param_name: 3.14}
    )
    assert state_name in recognized
    assert param_name in recognized
    assert np.isclose(
        system_interface_mutable.states.values_dict[state_name], 42.0
    )
    assert np.isclose(
        system_interface_mutable.parameters.values_dict[param_name], 3.14
    )


def test_update_raises_keyerror_on_unrecognized(system_interface):
    """Raises KeyError when unrecognized keys and silent=False."""
    with pytest.raises(KeyError, match="not recognized"):
        system_interface.update({"not_a_key": 1.0})


def test_update_returns_recognized_when_silent(system_interface_mutable):
    """Returns recognized keys set when silent=True."""
    recognized = system_interface_mutable.update(
        {"not_a_key": 1.0}, silent=True
    )
    assert recognized == set()


def test_update_returns_recognized_keys(system_interface_mutable, system):
    """Returns set of recognized keys on success."""
    state_name = system.initial_values.names[0]
    recognized = system_interface_mutable.update({state_name: 5.0})
    assert recognized == {state_name}


# ── state_indices / observable_indices / parameter_indices ─ #

def test_state_indices_none_returns_all(system_interface, system):
    """Returns all state indices when keys_or_indices is None."""
    result = system_interface.state_indices(None)
    expected = np.arange(system.sizes.states, dtype=np.int32)
    assert_array_equal(result, expected)


def test_state_indices_delegates(system_interface, system):
    """Delegates to states.get_indices for specific keys."""
    names = system.initial_values.names
    result = system_interface.state_indices(names)
    expected = np.arange(system.sizes.states, dtype=np.int32)
    assert_array_equal(result, expected)


def test_observable_indices_none_returns_all(system_interface, system):
    """Returns all observable indices when keys_or_indices is None."""
    result = system_interface.observable_indices(None)
    expected = np.arange(system.sizes.observables, dtype=np.int32)
    assert_array_equal(result, expected)


def test_observable_indices_delegates(system_interface, system):
    """Delegates to observables.get_indices for specific keys."""
    names = system.observables.names
    result = system_interface.observable_indices(names)
    expected = np.arange(system.sizes.observables, dtype=np.int32)
    assert_array_equal(result, expected)


def test_parameter_indices_delegates(system_interface, system):
    """Delegates to parameters.get_indices."""
    names = system.parameters.names
    result = system_interface.parameter_indices(names)
    expected = np.arange(system.sizes.parameters, dtype=np.int32)
    assert_array_equal(result, expected)


# ── get_labels / state_labels / observable_labels / parameter_labels ── #

def test_get_labels_delegates(system_interface, system):
    """Delegates to values_object.get_labels(indices)."""
    indices = np.array([0], dtype=np.int32)
    result = system_interface.get_labels(system_interface.states, indices)
    assert result == [system.initial_values.names[0]]


@pytest.mark.parametrize(
    "method, expected_attr",
    [
        ("state_labels", "initial_values"),
        ("observable_labels", "observables"),
        ("parameter_labels", "parameters"),
    ],
    ids=["state", "observable", "parameter"],
)
def test_labels_none_returns_all_names(
    system_interface, system, method, expected_attr
):
    """Returns all names when indices is None."""
    result = getattr(system_interface, method)(None)
    expected = getattr(system, expected_attr).names
    assert result == expected


@pytest.mark.parametrize(
    "method, expected_attr",
    [
        ("state_labels", "initial_values"),
        ("observable_labels", "observables"),
        ("parameter_labels", "parameters"),
    ],
    ids=["state", "observable", "parameter"],
)
def test_labels_with_indices(
    system_interface, system, method, expected_attr
):
    """Delegates to get_labels when indices provided."""
    indices = np.array([0], dtype=np.int32)
    result = getattr(system_interface, method)(indices)
    expected_name = getattr(system, expected_attr).names[0]
    assert result == [expected_name]


# ── resolve_variable_labels ──────────────────────────────── #

def test_resolve_variable_labels_none(system_interface):
    """Returns (None, None) when labels is None."""
    state_idx, obs_idx = system_interface.resolve_variable_labels(None)
    assert state_idx is None
    assert obs_idx is None


def test_resolve_variable_labels_empty(system_interface):
    """Returns empty int32 arrays when labels is empty list."""
    state_idx, obs_idx = system_interface.resolve_variable_labels([])
    assert len(state_idx) == 0
    assert len(obs_idx) == 0
    assert state_idx.dtype == np.int32
    assert obs_idx.dtype == np.int32


def test_resolve_variable_labels_states(system_interface, system):
    """Resolves state labels to state indices."""
    state_names = list(system.initial_values.names)
    state_idx, obs_idx = system_interface.resolve_variable_labels(
        state_names
    )
    expected = np.arange(len(state_names), dtype=np.int32)
    assert_array_equal(state_idx, expected)
    assert state_idx.dtype == np.int32
    assert obs_idx.dtype == np.int32


def test_resolve_variable_labels_raises_on_invalid(system_interface):
    """Raises ValueError for unresolved labels when silent=False."""
    with pytest.raises(ValueError, match="Variables not found"):
        system_interface.resolve_variable_labels(["not_real"])


def test_resolve_variable_labels_silent_invalid(system_interface):
    """Silent mode does not raise for unresolved labels."""
    state_idx, obs_idx = system_interface.resolve_variable_labels(
        ["not_real"], silent=True
    )
    assert len(state_idx) == 0
    assert len(obs_idx) == 0


# ── merge_variable_inputs ────────────────────────────────── #

def test_merge_variable_inputs_all_none(system_interface, system):
    """Returns full range when all three inputs None."""
    state_idx, obs_idx = system_interface.merge_variable_inputs(
        None, None, None
    )
    assert_array_equal(
        state_idx, np.arange(system.sizes.states, dtype=np.int32)
    )
    assert_array_equal(
        obs_idx, np.arange(system.sizes.observables, dtype=np.int32)
    )


def test_merge_variable_inputs_replaces_none_with_empty(
    system_interface, system
):
    """Replaces None inputs with empty arrays before union."""
    state_idx, obs_idx = system_interface.merge_variable_inputs(
        [], None, None
    )
    assert len(state_idx) == 0
    assert len(obs_idx) == 0


def test_merge_variable_inputs_indices_without_labels(
    system_interface, system
):
    """Indices without labels bypass the all-None full-range path."""
    state_idx, obs_idx = system_interface.merge_variable_inputs(
        None, np.array([0], dtype=np.int32), None,
    )
    assert_array_equal(state_idx, np.array([0], dtype=np.int32))
    assert len(obs_idx) == 0


def test_merge_variable_inputs_union(system_interface, system):
    """Computes union of label-resolved and directly-provided indices."""
    state_names = list(system.initial_values.names)
    state_idx, obs_idx = system_interface.merge_variable_inputs(
        [state_names[0]], [1], None
    )
    assert 0 in state_idx
    assert 1 in state_idx
    # Deduplication: union removes duplicates
    state_idx2, _ = system_interface.merge_variable_inputs(
        [state_names[0]], [0], None
    )
    assert len(state_idx2) == 1
    assert state_idx2[0] == 0


# ── merge_variable_labels_and_idxs ──────────────────────── #

def test_merge_labels_and_idxs_pops_keys(system_interface, system):
    """Pops save_variables and summarise_variables from dict."""
    state_names = list(system.initial_values.names)
    settings = {
        "save_variables": [state_names[0]],
        "summarise_variables": [state_names[0]],
    }
    system_interface.merge_variable_labels_and_idxs(settings)
    assert "save_variables" not in settings
    assert "summarise_variables" not in settings


def test_merge_labels_and_idxs_sets_all_four_keys(
    system_interface, system
):
    """Updates dict in-place with final index arrays."""
    settings = {}
    system_interface.merge_variable_labels_and_idxs(settings)
    assert_array_equal(
        settings["saved_state_indices"],
        np.arange(system.sizes.states, dtype=np.int32),
    )
    assert_array_equal(
        settings["saved_observable_indices"],
        np.arange(system.sizes.observables, dtype=np.int32),
    )
    assert_array_equal(
        settings["summarised_state_indices"],
        np.arange(system.sizes.states, dtype=np.int32),
    )
    assert_array_equal(
        settings["summarised_observable_indices"],
        np.arange(system.sizes.observables, dtype=np.int32),
    )


def test_merge_labels_and_idxs_summarise_defaults_to_saved(
    system_interface, system
):
    """Defaults summarise indices to saved indices when all summarise None."""
    state_names = list(system.initial_values.names)
    settings = {"save_variables": [state_names[0]]}
    system_interface.merge_variable_labels_and_idxs(settings)
    assert_array_equal(
        settings["saved_state_indices"],
        settings["summarised_state_indices"],
    )
    assert_array_equal(
        settings["saved_observable_indices"],
        settings["summarised_observable_indices"],
    )


def test_merge_labels_and_idxs_explicit_summarise(
    system_interface, system
):
    """Calls merge_variable_inputs for summarise when not all None."""
    state_names = list(system.initial_values.names)
    obs_names = list(system.observables.names)
    settings = {
        "save_variables": state_names,
        "summarise_variables": [obs_names[0]],
    }
    system_interface.merge_variable_labels_and_idxs(settings)
    # Summarised observables should include obs_names[0]
    assert 0 in settings["summarised_observable_indices"]
    # Saved and summarised should differ
    assert not np.array_equal(
        settings["saved_observable_indices"],
        settings["summarised_observable_indices"],
    )


# ── Properties ───────────────────────────────────────────── #

def test_all_input_labels(system_interface, system):
    """all_input_labels -> state_labels() + parameter_labels()."""
    expected = (
        list(system.initial_values.names)
        + list(system.parameters.names)
    )
    assert system_interface.all_input_labels == expected


def test_all_output_labels(system_interface, system):
    """all_output_labels -> state_labels() + observable_labels()."""
    expected = (
        list(system.initial_values.names)
        + list(system.observables.names)
    )
    assert system_interface.all_output_labels == expected
