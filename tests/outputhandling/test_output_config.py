"""Tests for cubie.outputhandling.output_config."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cubie.outputhandling.output_config import (
    OutputCompileFlags,
    OutputConfig,
    _indices_validator,
)
from cubie.outputhandling.summarymetrics import summary_metrics


# -- _indices_validator --------------------------------------------------- #


def test_indices_validator_none_passes():
    """No error when array is None."""
    result = _indices_validator(None, 10)
    assert result is None


def test_indices_validator_valid_array():
    """No error for valid unique in-range array."""
    valid = np.array([0, 2, 4], dtype=np.int32)
    result = _indices_validator(valid, 5)
    assert result is None


def test_indices_validator_not_ndarray():
    """TypeError when array is not an ndarray."""
    with pytest.raises(TypeError, match="numpy array of integers"):
        _indices_validator([0, 1], 5)


def test_indices_validator_wrong_dtype():
    """TypeError when array dtype is not np.int32."""
    bad = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(TypeError, match="numpy array of integers"):
        _indices_validator(bad, 5)


def test_indices_validator_negative():
    """ValueError when index is negative."""
    neg = np.array([-1, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="Indices must be in the range"):
        _indices_validator(neg, 5)


def test_indices_validator_out_of_bounds():
    """ValueError when index >= max_index."""
    oob = np.array([0, 5], dtype=np.int32)
    with pytest.raises(ValueError, match="Indices must be in the range"):
        _indices_validator(oob, 5)


def test_indices_validator_duplicates():
    """ValueError listing duplicated indices."""
    dup = np.array([0, 1, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="Duplicate indices found"):
        _indices_validator(dup, 5)


# -- OutputCompileFlags --------------------------------------------------- #


def test_compile_flags_all_defaults():
    """All defaults are False."""
    flags = OutputCompileFlags()
    assert flags.save_state is False
    assert flags.save_observables is False
    assert flags.summarise is False
    assert flags.summarise_observables is False
    assert flags.summarise_state is False
    assert flags.save_counters is False


def test_compile_flags_bool_validation():
    """Each boolean attribute validated as instance_of(bool)."""
    with pytest.raises(TypeError):
        OutputCompileFlags(save_state=1)


def test_compile_flags_hash_stable():
    """Equal flag snapshots share a values_hash."""
    f1 = OutputCompileFlags(save_state=True)
    f2 = OutputCompileFlags(save_state=True)
    assert f1.values_hash == f2.values_hash
    assert len(f1.values_hash) == 64


# -- OutputConfig construction & validation_passes ------------------------ #


def test_init_calls_validation_passes():
    """Construction calls validation_passes which sets flags."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["time"], precision=np.float32,
    )
    assert cfg._save_state is False
    assert cfg._save_time is True


def test_check_saved_indices_converts_to_numpy():
    """_check_saved_indices converts lists to numpy int arrays."""
    cfg = OutputConfig(
        max_states=5, max_observables=3,
        saved_state_indices=[0, 1],
        saved_observable_indices=[0, 2],
        output_types=["state", "observables"],
        precision=np.float32,
    )
    assert cfg._saved_state_indices.dtype == np.int32
    assert cfg._saved_observable_indices.dtype == np.int32


def test_check_summarised_indices_converts_to_numpy():
    """_check_summarised_indices converts lists to numpy int arrays."""
    cfg = OutputConfig(
        max_states=5, max_observables=3,
        summarised_state_indices=[0, 1],
        summarised_observable_indices=[0],
        output_types=["mean"], precision=np.float32,
    )
    assert cfg._summarised_state_indices.dtype == np.int32
    assert cfg._summarised_observable_indices.dtype == np.int32


def test_validate_index_arrays_state_bounds():
    """State indices validated against _max_states."""
    with pytest.raises(ValueError, match="Indices must be in the range"):
        OutputConfig(
            max_states=3, max_observables=2,
            saved_state_indices=[0, 3],
            output_types=["state"], precision=np.float32,
        )


def test_validate_index_arrays_observable_bounds():
    """Observable indices validated against _max_observables."""
    with pytest.raises(ValueError, match="Indices must be in the range"):
        OutputConfig(
            max_states=5, max_observables=2,
            saved_observable_indices=[0, 2],
            output_types=["observables"], precision=np.float32,
        )


def test_check_for_no_outputs_raises():
    """Raises ValueError when no output types enabled."""
    with pytest.raises(ValueError, match="At least one output type"):
        OutputConfig(
            max_states=3, max_observables=2,
            output_types=[], precision=np.float32,
        )


@pytest.mark.parametrize(
    "output_types",
    [
        pytest.param(["state"], id="state_only"),
        pytest.param(["observables"], id="observables_only"),
        pytest.param(["time"], id="time_only"),
        pytest.param(["iteration_counters"], id="counters_only"),
        pytest.param(["mean"], id="summary_only"),
    ],
)
def test_check_for_no_outputs_passes_each_branch(output_types):
    """At least one output path active passes validation."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        saved_state_indices=[0, 1],
        saved_observable_indices=[0],
        summarised_state_indices=[0],
        summarised_observable_indices=[0],
        output_types=output_types,
        precision=np.float32,
    )
    assert cfg.output_types == tuple(output_types)


# -- max_states / max_observables properties + setters -------------------- #


def test_max_states_getter():
    """Getter returns _max_states."""
    cfg = OutputConfig(
        max_states=7, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0, 1],
        precision=np.float32,
    )
    assert cfg.max_states == 7


def test_max_states_assignment_raises():
    """Snapshots are immutable: direct assignment raises."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        saved_state_indices=np.arange(3, dtype=np.int32),
        output_types=["state"], precision=np.float32,
    )
    with pytest.raises(AttributeError):
        cfg.max_states = 5


def test_index_arrays_are_sealed_owned_copies():
    """Snapshots are deeply sealed, not just top-level frozen.

    A dtype-matching input array is copied, not aliased; the index
    properties return read-only storage; and the memoized
    ``values_hash`` therefore cannot be desynchronised by mutating
    either the constructor input or a returned property.
    """
    caller_indices = np.array([0, 1], dtype=np.int32)
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        saved_state_indices=caller_indices,
        output_types=["state"], precision=np.float32,
    )
    digest = cfg.values_hash

    # Mutating the caller's own array never reaches the snapshot.
    caller_indices[0] = 2
    assert_array_equal(
        cfg.saved_state_indices, np.array([0, 1], dtype=np.int32)
    )

    # The returned property is read-only storage.
    with pytest.raises(ValueError):
        cfg.saved_state_indices[0] = 2

    # The memoized digest still matches an equal fresh snapshot.
    assert cfg.values_hash == digest
    fresh = OutputConfig(
        max_states=3, max_observables=2,
        saved_state_indices=np.array([0, 1], dtype=np.int32),
        output_types=["state"], precision=np.float32,
    )
    assert fresh.values_hash == digest
    different = OutputConfig(
        max_states=3, max_observables=2,
        saved_state_indices=np.array([2, 1], dtype=np.int32),
        output_types=["state"], precision=np.float32,
    )
    assert different.values_hash != digest


def test_max_observables_getter():
    """Getter returns _max_observables."""
    cfg = OutputConfig(
        max_states=3, max_observables=7,
        output_types=["time"], precision=np.float32,
    )
    assert cfg.max_observables == 7


def test_max_observables_update_derives_replacement():
    """max_observables changes through the pure update path."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        saved_observable_indices=[0, 2],
        output_types=["observables"], precision=np.float32,
    )
    replacement, _, changed = cfg.update({"max_observables": 10})
    assert "max_observables" in changed
    assert replacement.max_observables == 10
    assert cfg.max_observables == 5
    assert_array_equal(
        replacement.saved_observable_indices,
        np.array([0, 2], dtype=np.int32),
    )


# -- save_state calculated property --------------------------------------- #


@pytest.mark.parametrize(
    "output_types, indices, expected",
    [
        pytest.param(
            ["state"], [0, 1], True, id="true_with_indices"
        ),
        pytest.param(
            ["time"], [0, 1], False, id="false_save_state_off"
        ),
    ],
)
def test_save_state(output_types, indices, expected):
    """save_state depends on _save_state flag AND non-empty indices."""
    cfg = OutputConfig(
        max_states=5, max_observables=2,
        saved_state_indices=indices,
        output_types=output_types, precision=np.float32,
    )
    assert cfg.save_state is expected


def test_save_state_true_but_empty_indices():
    """save_state False when _save_state True but indices empty."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        saved_state_indices=[],
        output_types=["state", "time"], precision=np.float32,
    )
    assert cfg.save_state is False


# -- save_observables calculated property --------------------------------- #


@pytest.mark.parametrize(
    "output_types, indices, expected",
    [
        pytest.param(
            ["observables"], [0, 1], True, id="true_with_indices"
        ),
        pytest.param(
            ["time"], [0, 1], False, id="false_save_obs_off"
        ),
    ],
)
def test_save_observables(output_types, indices, expected):
    """save_observables depends on flag AND non-empty indices."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        saved_observable_indices=indices,
        output_types=output_types, precision=np.float32,
    )
    assert cfg.save_observables is expected


def test_save_observables_true_but_empty_indices():
    """save_observables False when flag True but indices empty."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        saved_observable_indices=[],
        output_types=["observables", "time"], precision=np.float32,
    )
    assert cfg.save_observables is False


# -- save_time / save_counters forwarding properties ---------------------- #


def test_save_time():
    """save_time True when 'time' in output_types."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["time"], precision=np.float32,
    )
    assert cfg.save_time is True


def test_save_counters():
    """save_counters True when 'iteration_counters' in output_types."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["iteration_counters"], precision=np.float32,
    )
    assert cfg.save_counters is True


# -- save_summaries calculated property ----------------------------------- #


def test_save_summaries_true():
    """save_summaries True when summary types non-empty."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["mean"], precision=np.float32,
    )
    assert cfg.save_summaries is True


def test_save_summaries_false():
    """save_summaries False when no summary types."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.save_summaries is False


# -- summarise_state / summarise_observables ------------------------------ #


def test_summarise_state_true():
    """True when summaries active AND summarised state indices > 0."""
    cfg = OutputConfig(
        max_states=5, max_observables=2,
        summarised_state_indices=[0, 1],
        output_types=["mean"], precision=np.float32,
    )
    assert cfg.summarise_state is True


def test_summarise_state_false_no_summaries():
    """False when summaries not active."""
    cfg = OutputConfig(
        max_states=5, max_observables=2,
        summarised_state_indices=[0, 1],
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.summarise_state is False


def test_summarise_state_false_no_indices():
    """False when n_summarised_states == 0."""
    cfg = OutputConfig(
        max_states=5, max_observables=2,
        summarised_state_indices=[],
        output_types=["mean"], precision=np.float32,
    )
    assert cfg.summarise_state is False


def test_summarise_observables_true():
    """True when summaries active AND obs indices > 0."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        summarised_observable_indices=[0, 1],
        output_types=["mean"], precision=np.float32,
    )
    assert cfg.summarise_observables is True


def test_summarise_observables_false_no_summaries():
    """False when summaries not active."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        summarised_observable_indices=[0, 1],
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.summarise_observables is False


def test_summarise_observables_false_no_indices():
    """False when n_summarised_observables == 0."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        summarised_observable_indices=[],
        output_types=["mean"], precision=np.float32,
    )
    assert cfg.summarise_observables is False


# -- compile_flags calculated property ------------------------------------ #


def test_compile_flags_matches_config():
    """compile_flags fields derived from current config properties."""
    cfg = OutputConfig(
        max_states=5, max_observables=3,
        saved_state_indices=[0, 1],
        saved_observable_indices=[0],
        summarised_state_indices=[0],
        summarised_observable_indices=[0],
        output_types=[
            "state", "observables", "mean", "iteration_counters",
        ],
        precision=np.float32,
    )
    flags = cfg.compile_flags
    assert flags.save_state == cfg.save_state
    assert flags.save_observables == cfg.save_observables
    assert flags.summarise == cfg.save_summaries
    assert flags.summarise_state == cfg.summarise_state
    assert flags.summarise_observables == cfg.summarise_observables
    assert flags.save_counters == cfg.save_counters


# -- saved/summarised indices property + setter --------------------------- #


def test_saved_state_indices_empty_when_disabled():
    """Getter returns empty array when _save_state is False."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        saved_state_indices=[0, 1],
        output_types=["time"], precision=np.float32,
    )
    assert len(cfg.saved_state_indices) == 0


def test_saved_state_indices_returns_array_when_enabled():
    """Getter returns _saved_state_indices when enabled."""
    cfg = OutputConfig(
        max_states=5, max_observables=2,
        saved_state_indices=[0, 2, 4],
        output_types=["state"], precision=np.float32,
    )
    assert_array_equal(
        cfg.saved_state_indices,
        np.array([0, 2, 4], dtype=np.int32),
    )


def test_saved_state_indices_setter():
    """Setter converts to numpy int array and validates."""
    cfg = OutputConfig(
        max_states=5, max_observables=2,
        saved_state_indices=[0, 1],
        output_types=["state"], precision=np.float32,
    )
    replacement, _, changed = cfg.update({"saved_state_indices": [0, 3]})
    assert "saved_state_indices" in changed
    assert_array_equal(
        replacement.saved_state_indices,
        np.array([0, 3], dtype=np.int32),
    )


def test_saved_observable_indices_empty_when_disabled():
    """Getter returns empty when _save_observables is False."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        saved_observable_indices=[0, 1],
        output_types=["time"], precision=np.float32,
    )
    assert len(cfg.saved_observable_indices) == 0


def test_saved_observable_indices_returns_when_enabled():
    """Getter returns indices when _save_observables is True."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        saved_observable_indices=[1, 3],
        output_types=["observables"], precision=np.float32,
    )
    assert_array_equal(
        cfg.saved_observable_indices,
        np.array([1, 3], dtype=np.int32),
    )


def test_saved_observable_indices_setter():
    """Setter converts, validates, checks no-outputs."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        saved_observable_indices=[0],
        output_types=["observables"], precision=np.float32,
    )
    replacement, _, changed = cfg.update(
        {"saved_observable_indices": [0, 4]}
    )
    assert "saved_observable_indices" in changed
    assert_array_equal(
        replacement.saved_observable_indices,
        np.array([0, 4], dtype=np.int32),
    )


def test_summarised_state_indices_empty_no_summaries():
    """Getter returns empty when save_summaries is False."""
    cfg = OutputConfig(
        max_states=5, max_observables=2,
        summarised_state_indices=[0, 1],
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert len(cfg.summarised_state_indices) == 0


def test_summarised_state_indices_returns_when_summaries():
    """Getter returns indices when save_summaries is True."""
    cfg = OutputConfig(
        max_states=5, max_observables=2,
        summarised_state_indices=[0, 2],
        output_types=["mean"], precision=np.float32,
    )
    assert_array_equal(
        cfg.summarised_state_indices,
        np.array([0, 2], dtype=np.int32),
    )


def test_summarised_state_indices_setter():
    """Setter converts, validates, checks no-outputs."""
    cfg = OutputConfig(
        max_states=5, max_observables=2,
        summarised_state_indices=[0],
        output_types=["mean"], precision=np.float32,
    )
    replacement, _, changed = cfg.update(
        {"summarised_state_indices": [0, 3]}
    )
    assert "summarised_state_indices" in changed
    assert_array_equal(
        replacement.summarised_state_indices,
        np.array([0, 3], dtype=np.int32),
    )


def test_summarised_obs_indices_empty_no_summaries():
    """Getter returns empty when save_summaries is False."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        summarised_observable_indices=[0, 1],
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert len(cfg.summarised_observable_indices) == 0


def test_summarised_obs_indices_returns_when_summaries():
    """Getter returns indices when save_summaries is True."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        summarised_observable_indices=[1, 4],
        output_types=["mean"], precision=np.float32,
    )
    assert_array_equal(
        cfg.summarised_observable_indices,
        np.array([1, 4], dtype=np.int32),
    )


def test_summarised_observable_indices_setter():
    """Setter converts, validates, checks no-outputs."""
    cfg = OutputConfig(
        max_states=3, max_observables=5,
        summarised_observable_indices=[0],
        output_types=["mean"], precision=np.float32,
    )
    replacement, _, changed = cfg.update(
        {"summarised_observable_indices": [0, 3]}
    )
    assert "summarised_observable_indices" in changed
    assert_array_equal(
        replacement.summarised_observable_indices,
        np.array([0, 3], dtype=np.int32),
    )


# -- n_saved / n_summarised calculated properties ------------------------- #


def test_n_saved_states_when_enabled():
    """Returns length of indices when _save_state True."""
    cfg = OutputConfig(
        max_states=10, max_observables=2,
        saved_state_indices=[0, 3, 7],
        output_types=["state"], precision=np.float32,
    )
    assert cfg.n_saved_states == 3


def test_n_saved_states_when_disabled():
    """Returns 0 when _save_state False."""
    cfg = OutputConfig(
        max_states=10, max_observables=2,
        saved_state_indices=[0, 3],
        output_types=["time"], precision=np.float32,
    )
    assert cfg.n_saved_states == 0


def test_n_saved_observables_when_enabled():
    """Returns length of indices when _save_observables True."""
    cfg = OutputConfig(
        max_states=3, max_observables=10,
        saved_observable_indices=[0, 5, 9],
        output_types=["observables"], precision=np.float32,
    )
    assert cfg.n_saved_observables == 3


def test_n_saved_observables_when_disabled():
    """Returns 0 when _save_observables False."""
    cfg = OutputConfig(
        max_states=3, max_observables=10,
        saved_observable_indices=[0, 5],
        output_types=["time"], precision=np.float32,
    )
    assert cfg.n_saved_observables == 0


def test_n_summarised_states_when_summaries():
    """Returns length of indices when summaries active."""
    cfg = OutputConfig(
        max_states=10, max_observables=2,
        summarised_state_indices=[0, 1, 2],
        output_types=["mean"], precision=np.float32,
    )
    assert cfg.n_summarised_states == 3


def test_n_summarised_states_no_summaries():
    """Returns 0 when summaries not active."""
    cfg = OutputConfig(
        max_states=10, max_observables=2,
        summarised_state_indices=[0, 1],
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.n_summarised_states == 0


def test_n_summarised_observables_when_summaries():
    """Returns length of indices when summaries active."""
    cfg = OutputConfig(
        max_states=3, max_observables=10,
        summarised_observable_indices=[0, 2, 4],
        output_types=["mean"], precision=np.float32,
    )
    assert cfg.n_summarised_observables == 3


def test_n_summarised_observables_no_summaries():
    """Returns 0 when summaries not active."""
    cfg = OutputConfig(
        max_states=3, max_observables=10,
        summarised_observable_indices=[0, 2],
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.n_summarised_observables == 0


# -- summary_types forwarding property ------------------------------------ #


def test_summary_types_returns_tuple():
    """Returns _summary_types tuple."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["mean", "max", "state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.summary_types == ("mean", "max")


# -- summary_legend_per_variable ------------------------------------------ #


def test_summary_legend_empty_when_no_types():
    """Returns empty dict when _summary_types is empty."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.summary_legend_per_variable == {}


def test_summary_legend_maps_indices():
    """Returns dict mapping indices to metric legend strings."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["mean", "max"], precision=np.float32,
    )
    legend = cfg.summary_legend_per_variable
    expected_tuple = summary_metrics.legend(["mean", "max"])
    expected = dict(
        zip(range(len(expected_tuple)), expected_tuple)
    )
    assert legend == expected


def test_summary_legend_maps_indices_fused_extrema():
    """Fused max/min legend reports the requested metric names."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["max", "min"], precision=np.float32,
    )
    legend = cfg.summary_legend_per_variable
    assert legend == {0: "max", 1: "min"}


def test_summary_legend_maps_indices_fused_mean_std_rms():
    """Fused mean/std/rms legend reports the requested metric names."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["mean", "std", "rms"], precision=np.float32,
    )
    legend = cfg.summary_legend_per_variable
    assert legend == {0: "mean", 1: "std", 2: "rms"}


# -- summary_unit_modifications ------------------------------------------- #


def test_summary_unit_modifications_empty():
    """Returns empty dict when _summary_types is empty."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.summary_unit_modifications == {}


def test_summary_unit_modifications_maps_indices():
    """Returns dict mapping indices to unit modification strings."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["mean", "max"], precision=np.float32,
    )
    unit_mods = cfg.summary_unit_modifications
    expected_tuple = summary_metrics.unit_modifications(
        ["mean", "max"]
    )
    expected = dict(
        zip(range(len(expected_tuple)), expected_tuple)
    )
    assert unit_mods == expected


# -- sample_summaries_every forwarding property --------------------------- #


def test_sample_summaries_every_forwarding():
    """Returns _sample_summaries_every."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        sample_summaries_every=0.05,
        precision=np.float32,
    )
    assert cfg.sample_summaries_every == 0.05


def test_sample_summaries_every_default_none():
    """Default is None."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.sample_summaries_every is None


# -- summaries_buffer_height_per_var -------------------------------------- #


def test_summaries_buf_height_per_var_zero_no_types():
    """Returns 0 when summary_types is empty."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.summaries_buffer_height_per_var == 0


def test_summaries_buf_height_per_var_delegates():
    """Delegates to summary_metrics.summaries_buffer_height."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["mean", "max"], precision=np.float32,
    )
    expected = summary_metrics.summaries_buffer_height(
        ["mean", "max"]
    )
    assert cfg.summaries_buffer_height_per_var == expected


# -- summaries_output_height_per_var -------------------------------------- #


def test_summaries_out_height_per_var_zero():
    """Returns 0 when _summary_types is empty."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.summaries_output_height_per_var == 0


def test_summaries_out_height_per_var_delegates():
    """Delegates to summary_metrics.summaries_output_height."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["mean", "max"], precision=np.float32,
    )
    expected = summary_metrics.summaries_output_height(
        ["mean", "max"]
    )
    assert cfg.summaries_output_height_per_var == expected


# -- composite summary heights -------------------------------------------- #


def test_state_summaries_buffer_height():
    """Returns buffer_per_var * n_summarised_states."""
    cfg = OutputConfig(
        max_states=10, max_observables=2,
        summarised_state_indices=[0, 1, 2],
        output_types=["mean"], precision=np.float32,
    )
    expected = (
        cfg.summaries_buffer_height_per_var
        * cfg.n_summarised_states
    )
    assert cfg.state_summaries_buffer_height == expected


def test_observable_summaries_buffer_height():
    """Returns buffer_per_var * n_summarised_observables."""
    cfg = OutputConfig(
        max_states=3, max_observables=10,
        summarised_observable_indices=[0, 2],
        output_types=["mean"], precision=np.float32,
    )
    expected = (
        cfg.summaries_buffer_height_per_var
        * cfg.n_summarised_observables
    )
    assert cfg.observable_summaries_buffer_height == expected


def test_total_summary_buffer_size():
    """Returns sum of state and observable buffer heights."""
    cfg = OutputConfig(
        max_states=10, max_observables=5,
        summarised_state_indices=[0, 1],
        summarised_observable_indices=[0],
        output_types=["mean"], precision=np.float32,
    )
    expected = (
        cfg.state_summaries_buffer_height
        + cfg.observable_summaries_buffer_height
    )
    assert cfg.total_summary_buffer_size == expected


def test_state_summaries_output_height():
    """Returns output_per_var * n_summarised_states."""
    cfg = OutputConfig(
        max_states=10, max_observables=2,
        summarised_state_indices=[0, 1, 2],
        output_types=["mean"], precision=np.float32,
    )
    expected = (
        cfg.summaries_output_height_per_var
        * cfg.n_summarised_states
    )
    assert cfg.state_summaries_output_height == expected


def test_observable_summaries_output_height():
    """Returns output_per_var * n_summarised_observables."""
    cfg = OutputConfig(
        max_states=3, max_observables=10,
        summarised_observable_indices=[0, 2, 4],
        output_types=["mean"], precision=np.float32,
    )
    expected = (
        cfg.summaries_output_height_per_var
        * cfg.n_summarised_observables
    )
    assert cfg.observable_summaries_output_height == expected


# -- buffer_sizes_dict ---------------------------------------------------- #


def test_buffer_sizes_dict():
    """Returns dict with expected keys matching property values."""
    cfg = OutputConfig(
        max_states=5, max_observables=3,
        saved_state_indices=[0, 1],
        saved_observable_indices=[0],
        summarised_state_indices=[0],
        summarised_observable_indices=[0],
        output_types=["state", "observables", "mean"],
        precision=np.float32,
    )
    bsd = cfg.buffer_sizes_dict
    assert bsd["n_saved_states"] == cfg.n_saved_states
    assert bsd["n_saved_observables"] == (
        cfg.n_saved_observables
    )
    assert bsd["n_summarised_states"] == (
        cfg.n_summarised_states
    )
    assert bsd["n_summarised_observables"] == (
        cfg.n_summarised_observables
    )
    assert bsd["state_summaries_buffer_height"] == (
        cfg.state_summaries_buffer_height
    )
    assert bsd["observable_summaries_buffer_height"] == (
        cfg.observable_summaries_buffer_height
    )
    assert bsd["total_summary_buffer_size"] == (
        cfg.total_summary_buffer_size
    )
    assert bsd["state_summaries_output_height"] == (
        cfg.state_summaries_output_height
    )
    assert bsd["observable_summaries_output_height"] == (
        cfg.observable_summaries_output_height
    )
    flags = bsd["compile_flags"]
    assert flags.save_state == cfg.save_state


# -- output_types property + setter --------------------------------------- #


def test_output_types_getter():
    """Getter returns the normalised output-types tuple."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state", "time"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert cfg.output_types == ("state", "time")


def test_output_types_update_tuple():
    """Update accepts a tuple and derives the flags."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    replacement, _, changed = cfg.update(
        {"output_types": ("state", "time")}
    )
    assert "output_types" in changed
    assert replacement.save_time is True
    assert cfg.save_time is False


def test_output_types_update_string():
    """A bare string normalises to a one-entry tuple."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    replacement, _, _ = cfg.update({"output_types": "time"})
    assert replacement.save_time is True
    assert replacement.output_types == ("time",)


def test_output_types_bad_type_raises():
    """The converter raises TypeError for unsupported input."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    with pytest.raises(TypeError, match="Output types must be"):
        cfg.update({"output_types": 42})


# -- update_from_outputs_list --------------------------------------------- #


def test_output_types_update_empty_raises():
    """An empty output-types update leaves no active output path."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state", "mean"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    with pytest.raises(ValueError, match="At least one output type"):
        cfg.update({"output_types": []})


def test_output_types_update_sets_state():
    """The replacement derives _save_state from the new list."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["time"], precision=np.float32,
    )
    replacement, _, _ = cfg.update({"output_types": ["state", "time"]})
    assert replacement._save_state is True
    assert cfg._save_state is False


def test_output_types_update_sets_observables():
    """The replacement derives _save_observables from the new list."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["time"], precision=np.float32,
    )
    replacement, _, _ = cfg.update(
        {"output_types": ["observables", "time"]}
    )
    assert replacement._save_observables is True


def test_output_types_update_sets_time():
    """The replacement derives _save_time from the new list."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["state"],
        saved_state_indices=[0],
        precision=np.float32,
    )
    replacement, _, _ = cfg.update({"output_types": ["time"]})
    assert replacement._save_time is True


def test_output_types_update_sets_counters():
    """The replacement derives _save_counters for iteration counters."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["time"], precision=np.float32,
    )
    replacement, _, _ = cfg.update(
        {"output_types": ["iteration_counters"]}
    )
    assert replacement._save_counters is True


def test_output_types_update_collects_summaries():
    """Summary types matching implemented metrics are collected."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["time"], precision=np.float32,
    )
    replacement, _, _ = cfg.update(
        {"output_types": ["mean", "rms", "time"]}
    )
    assert replacement._summary_types == ("mean", "rms")


def test_output_types_update_warns_unknown():
    """Warns for unrecognised output types."""
    cfg = OutputConfig(
        max_states=3, max_observables=2,
        output_types=["time"], precision=np.float32,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg.update({"output_types": ["bogus_metric", "time"]})
        assert len(caught) >= 1
        assert any(
            "is not implemented" in str(w.message) for w in caught
        )


# -- from_loop_settings --------------------------------------------------- #


def test_from_loop_settings_none_indices():
    """Converts None indices to empty numpy arrays."""
    cfg = OutputConfig.from_loop_settings(
        output_types=["time"],
        max_states=5, max_observables=3,
        precision=np.float32,
    )
    assert_array_equal(
        cfg._saved_state_indices,
        np.array([], dtype=np.int32),
    )
    assert_array_equal(
        cfg._saved_observable_indices,
        np.array([], dtype=np.int32),
    )
    assert_array_equal(
        cfg._summarised_state_indices,
        np.array([], dtype=np.int32),
    )
    assert_array_equal(
        cfg._summarised_observable_indices,
        np.array([], dtype=np.int32),
    )


def test_from_loop_settings_copies_output_types():
    """Copies output_types list before modifying."""
    original = ["state", "time"]
    OutputConfig.from_loop_settings(
        output_types=original,
        max_states=5, max_observables=3,
        saved_state_indices=[0],
        precision=np.float32,
    )
    assert original == ["state", "time"]


def test_from_loop_settings_passes_all_params():
    """Passes all parameters through to constructor."""
    cfg = OutputConfig.from_loop_settings(
        output_types=["state", "mean"],
        max_states=10, max_observables=5,
        saved_state_indices=[0, 1],
        saved_observable_indices=[0],
        summarised_state_indices=[0],
        summarised_observable_indices=[0],
        sample_summaries_every=0.1,
        precision=np.float64,
    )
    assert cfg.max_states == 10
    assert cfg.max_observables == 5
    assert_array_equal(
        cfg.saved_state_indices,
        np.array([0, 1], dtype=np.int32),
    )
    assert cfg.sample_summaries_every == 0.1
    assert cfg.summary_types == ("mean",)
