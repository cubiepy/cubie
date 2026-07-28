"""Tests for cubie.outputhandling.OutputFunctions."""

from __future__ import annotations

import numpy as np
import pytest

from tests._utils import SUMMARY_ONLY_NO_TIMING, STATE_ONLY_NO_SUMMARIES
from numpy.testing import assert_array_equal

from cubie.outputhandling.output_functions import (
    OutputFunctionCache,
    OutputFunctions,
)
from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.outputhandling.output_sizes import OutputArrayHeights


# ── __init__ ─────────────────────────────────────────────── #


def test_default_output_types_is_state(system, precision):
    """output_types defaults to ['state'] when None is passed."""
    of = OutputFunctions(
        system.sizes.states,
        system.sizes.observables,
        precision=precision,
    )
    assert "state" in of.output_types
    assert len(of.output_types) == 1


def test_init_creates_output_config(output_functions):
    """Construction installs an OutputConfig with matching output_types."""
    cs = output_functions.compile_settings
    assert cs.output_types == output_functions.output_types


# ── update ───────────────────────────────────────────────── #


def test_update_merges_dict_and_kwargs(output_functions_mutable):
    """update merges updates_dict and kwargs; both keys recognised."""
    recognised = output_functions_mutable.update(
        {"saved_state_indices": [0]},
        saved_observable_indices=[0],
    )
    assert recognised == {"saved_state_indices", "saved_observable_indices"}


def test_update_returns_empty_set_for_empty_dict(
    output_functions_mutable,
):
    """update({}) returns empty set without modifying state."""
    result = output_functions_mutable.update({})
    assert result == set()


def test_update_applies_to_compile_settings(output_functions_mutable):
    """update delegates to compile_settings and values propagate."""
    output_functions_mutable.update(
        {"saved_state_indices": [0]}, silent=True,
    )
    assert_array_equal(
        output_functions_mutable.saved_state_indices,
        np.array([0]),
    )


def test_update_raises_for_unrecognised_silent_false(
    output_functions_mutable,
):
    """update raises KeyError for unrecognised params when silent=False."""
    with pytest.raises(KeyError, match="Unrecognized"):
        output_functions_mutable.update(
            {"nonexistent_param": 42}, silent=False,
        )


def test_update_suppresses_unrecognised_silent_true(
    output_functions_mutable,
):
    """update returns empty recognised set for unknown params, silent."""
    recognised = output_functions_mutable.update(
        {"nonexistent_param": 42}, silent=True,
    )
    assert recognised == set()


# ── build / cache ────────────────────────────────────────── #


def test_build_populates_cache(output_functions):
    """build produces cache whose functions match the property accessors."""
    # Trigger build by accessing a cached property
    _ = output_functions.save_state_func
    cache = output_functions._cache
    assert isinstance(cache, OutputFunctionCache)
    assert cache.save_state_function is output_functions.save_state_func
    assert (
        cache.update_summaries_function
        is output_functions.update_summaries_func
    )
    assert (
        cache.save_summaries_function
        is output_functions.save_summary_metrics_func
    )


def test_build_cache_has_three_distinct_functions(output_functions):
    """The three cached functions are distinct callables."""
    fns = {
        output_functions.save_state_func,
        output_functions.update_summaries_func,
        output_functions.save_summary_metrics_func,
    }
    assert len(fns) == 3


# ── Forwarding properties (scalar) ──────────────────────── #


@pytest.mark.parametrize(
    "prop, child_attr",
    [
        ("output_types", "output_types"),
        ("save_time", "save_time"),
        ("n_saved_states", "n_saved_states"),
        ("n_saved_observables", "n_saved_observables"),
        ("state_summaries_output_height", "state_summaries_output_height"),
        (
            "observable_summaries_output_height",
            "observable_summaries_output_height",
        ),
        (
            "summaries_buffer_height_per_var",
            "summaries_buffer_height_per_var",
        ),
        (
            "state_summaries_buffer_height",
            "state_summaries_buffer_height",
        ),
        (
            "observable_summaries_buffer_height",
            "observable_summaries_buffer_height",
        ),
        (
            "summaries_output_height_per_var",
            "summaries_output_height_per_var",
        ),
        (
            "summary_legend_per_variable",
            "summary_legend_per_variable",
        ),
        (
            "summary_unit_modifications",
            "summary_unit_modifications",
        ),
        ("buffer_sizes_dict", "buffer_sizes_dict"),
    ],
)
def test_scalar_forwarding_to_compile_settings(
    output_functions, prop, child_attr,
):
    """Scalar forwarding properties delegate to compile_settings."""
    assert getattr(output_functions, prop) == getattr(
        output_functions.compile_settings, child_attr,
    )


def test_compile_flags_forwarded(output_functions):
    """compile_flags is the same object as compile_settings.compile_flags."""
    assert (
        output_functions.compile_flags
        == output_functions.compile_settings.compile_flags
    )
    assert isinstance(output_functions.compile_flags, OutputCompileFlags)


# ── Forwarding properties (array) ───────────────────────── #


@pytest.mark.parametrize(
    "prop, child_attr",
    [
        ("saved_state_indices", "saved_state_indices"),
        ("saved_observable_indices", "saved_observable_indices"),
        ("summarised_state_indices", "summarised_state_indices"),
        (
            "summarised_observable_indices",
            "summarised_observable_indices",
        ),
    ],
)
def test_array_forwarding_to_compile_settings(
    output_functions, prop, child_attr,
):
    """Array forwarding properties match compile_settings values."""
    assert_array_equal(
        getattr(output_functions, prop),
        getattr(output_functions.compile_settings, child_attr),
    )


# ── Cached function forwarding ──────────────────────────── #


@pytest.mark.parametrize(
    "prop, cache_attr",
    [
        ("save_state_func", "save_state_function"),
        ("update_summaries_func", "update_summaries_function"),
        ("save_summary_metrics_func", "save_summaries_function"),
    ],
)
def test_func_properties_forward_from_cache(
    output_functions, prop, cache_attr,
):
    """Function properties return the same object stored in the cache."""
    assert getattr(output_functions, prop) is getattr(
        output_functions._cache, cache_attr,
    )


# ── Computed properties: heights ─────────────────────────── #


def test_state_summaries_output_height_computed(output_functions):
    """state_summaries_output_height = per_var * n_summarised_states."""
    cs = output_functions.compile_settings
    expected = cs.summaries_output_height_per_var * len(
        output_functions.summarised_state_indices,
    )
    assert output_functions.state_summaries_output_height == expected


def test_observable_summaries_output_height_computed(output_functions):
    """observable_summaries_output_height = per_var * n_summarised_obs."""
    cs = output_functions.compile_settings
    expected = cs.summaries_output_height_per_var * len(
        output_functions.summarised_observable_indices,
    )
    assert output_functions.observable_summaries_output_height == expected


def test_state_summaries_buffer_height_computed(output_functions):
    """state_summaries_buffer_height = per_var * n_summarised_states."""
    cs = output_functions.compile_settings
    expected = cs.summaries_buffer_height_per_var * len(
        output_functions.summarised_state_indices,
    )
    assert output_functions.state_summaries_buffer_height == expected


def test_observable_summaries_buffer_height_computed(output_functions):
    """observable_summaries_buffer_height = per_var * n_summarised_obs."""
    cs = output_functions.compile_settings
    expected = cs.summaries_buffer_height_per_var * len(
        output_functions.summarised_observable_indices,
    )
    assert output_functions.observable_summaries_buffer_height == expected


def test_n_saved_states_matches_indices_length(
    output_functions, solver_settings,
):
    """n_saved_states equals length of saved_state_indices."""
    expected = len(solver_settings["saved_state_indices"])
    assert output_functions.n_saved_states == expected


def test_n_saved_observables_matches_indices_length(
    output_functions, solver_settings,
):
    """n_saved_observables equals length of saved_observable_indices."""
    expected = len(solver_settings["saved_observable_indices"])
    assert output_functions.n_saved_observables == expected


# ── has_time_domain_outputs ──────────────────────────────── #


def test_has_time_domain_outputs_default(output_functions):
    """has_time_domain_outputs True with default config (has state+time)."""
    assert output_functions.has_time_domain_outputs is True


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(SUMMARY_ONLY_NO_TIMING, id="summary-only")],
    indirect=True,
)
def test_has_time_domain_outputs_false(output_functions):
    """has_time_domain_outputs False when only summary types present."""
    assert output_functions.has_time_domain_outputs is False


# ── has_summary_outputs ──────────────────────────────────── #


def test_has_summary_outputs_default(output_functions):
    """has_summary_outputs True with default config (has mean)."""
    assert output_functions.has_summary_outputs is True


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(STATE_ONLY_NO_SUMMARIES, id="no-summaries")],
    indirect=True,
)
def test_has_summary_outputs_false(output_functions):
    """has_summary_outputs False when no summary types configured."""
    assert output_functions.has_summary_outputs is False


# ── output_array_heights ─────────────────────────────────── #


def test_output_array_heights_values(output_functions):
    """output_array_heights fields match individually computed values."""
    heights = output_functions.output_array_heights
    assert isinstance(heights, OutputArrayHeights)
    expected_state = (
        output_functions.n_saved_states
        + (1 if output_functions.save_time else 0)
    )
    assert heights.state == expected_state
    assert heights.observables == output_functions.n_saved_observables
    assert (
        heights.state_summaries
        == output_functions.state_summaries_output_height
    )
    assert (
        heights.observable_summaries
        == output_functions.observable_summaries_output_height
    )
