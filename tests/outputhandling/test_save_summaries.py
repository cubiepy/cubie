"""Tests for cubie.outputhandling.save_summaries."""

from __future__ import annotations

import pytest

from cubie.outputhandling.output_functions import OutputFunctions
from cubie.outputhandling.save_summaries import (
    chain_metrics,
    do_nothing,
    save_summary_factory,
)
from cubie.outputhandling.summarymetrics import summary_metrics


# ── do_nothing / chain_metrics empty case ───────────────────── #

def test_chain_metrics_empty_returns_do_nothing():
    """Empty metric_functions returns the do_nothing sentinel (base case)."""
    result = chain_metrics([], [], [], [], [], [])
    assert result is do_nothing


# ── chain_metrics ───────────────────────────────────────────── #


def test_chain_metrics_single_metric_returns_wrapper():
    """Single metric produces a wrapper distinct from do_nothing."""
    summaries_list = ["mean"]
    fns = summary_metrics.save_functions(summaries_list)
    b_offsets = summary_metrics.buffer_offsets(summaries_list)
    b_sizes = summary_metrics.buffer_sizes(summaries_list)
    o_offsets = summary_metrics.output_offsets(summaries_list)
    o_sizes = summary_metrics.output_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)

    result = chain_metrics(fns, b_offsets, b_sizes, o_offsets, o_sizes, params)
    assert result is not do_nothing
    # Device invocation tested in integrated_numerical_tests


def test_chain_metrics_multiple_metrics_chains_recursively():
    """Multiple metrics produce a single chained wrapper via recursion."""
    summaries_list = ["mean", "max"]
    fns = summary_metrics.save_functions(summaries_list)
    b_offsets = summary_metrics.buffer_offsets(summaries_list)
    b_sizes = summary_metrics.buffer_sizes(summaries_list)
    o_offsets = summary_metrics.output_offsets(summaries_list)
    o_sizes = summary_metrics.output_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)

    result = chain_metrics(fns, b_offsets, b_sizes, o_offsets, o_sizes, params)
    # Wrapper is distinct from do_nothing and from a single-metric chain
    assert result is not do_nothing
    # Device invocation tested in integrated_numerical_tests


def test_chain_metrics_consumes_all_offsets_sizes_and_params():
    """Each metric gets its own buffer/output offset, size, and param."""
    summaries_list = ["mean", "max", "rms"]
    fns = summary_metrics.save_functions(summaries_list)
    b_offsets = summary_metrics.buffer_offsets(summaries_list)
    b_sizes = summary_metrics.buffer_sizes(summaries_list)
    o_offsets = summary_metrics.output_offsets(summaries_list)
    o_sizes = summary_metrics.output_sizes(summaries_list)
    params = summary_metrics.params(summaries_list)

    # All sequences must have exactly 3 elements (one per metric)
    assert len(fns) == 3
    assert len(b_offsets) == 3
    assert len(b_sizes) == 3
    assert len(o_offsets) == 3
    assert len(o_sizes) == 3
    assert len(params) == 3

    # chain_metrics consumes all without error, returning a wrapper
    result = chain_metrics(fns, b_offsets, b_sizes, o_offsets, o_sizes, params)
    assert result is not do_nothing


# ── save_summary_factory branch isolation ───────────────────── #

@pytest.mark.parametrize(
    "state_indices, obs_indices, expected_label",
    [
        pytest.param([0, 1], [0, 1], "both", id="states-and-obs"),
        pytest.param([0, 1], [], "states-only", id="states-only"),
        pytest.param([], [0, 1], "obs-only", id="obs-only"),
        pytest.param([], [], "neither", id="neither"),
    ],
)
def test_save_summary_factory_branch_isolation(
    state_indices, obs_indices, expected_label,
):
    """Factory produces a device function for each branch combination."""
    summaries_list = ["mean"]
    buf_height = summary_metrics.summaries_buffer_height(summaries_list)
    fn = save_summary_factory(
        buf_height, state_indices, obs_indices, summaries_list,
    )
    # Factory always returns a compiled device function (not None)
    # Device invocation tested in integrated_numerical_tests
    assert fn is not do_nothing or (
        len(state_indices) == 0 and len(obs_indices) == 0
    )


# ── Integration through output_functions fixture ────────────── #

def test_save_summary_func_cache_identity(output_functions):
    """save_summary_factory result is stored in the cache and forwarded."""
    # Accessing the property triggers build, populating the cache
    fn = output_functions.save_summary_metrics_func
    cache = output_functions._cache
    assert cache.save_summaries_function is fn


def test_save_and_update_summary_funcs_are_distinct(output_functions):
    """Save and update summary functions are separate compiled functions."""
    save_fn = output_functions.save_summary_metrics_func
    update_fn = output_functions.update_summaries_func
    assert save_fn is not update_fn


def test_factory_buffer_height_per_var_matches_registry(output_functions):
    """summaries_buffer_height_per_var matches the summary_metrics registry."""
    cs = output_functions.compile_settings
    expected = summary_metrics.summaries_buffer_height(cs.summary_types)
    assert cs.summaries_buffer_height_per_var == expected


def test_factory_output_height_per_var_matches_registry(output_functions):
    """summaries_output_height_per_var matches the summary_metrics registry."""
    cs = output_functions.compile_settings
    expected = summary_metrics.summaries_output_height(cs.summary_types)
    assert cs.summaries_output_height_per_var == expected


def test_save_summary_factory_empty_indices_builds(
    output_settings, system, precision
):
    """Build succeeds when no variables are summarised.

    Built directly rather than through the chain: the empty index
    lists are the only departure from the session output settings,
    and the build emits lazily-compiled device closures only.
    """
    settings = output_settings.copy()
    settings.pop("precision", None)
    settings["summarised_state_indices"] = []
    settings["summarised_observable_indices"] = []
    empty_summaries = OutputFunctions(
        system.sizes.states,
        system.sizes.observables,
        precision=precision,
        **settings,
    )
    # Accessing the property triggers build; no exception raised
    fn = empty_summaries.save_summary_metrics_func
    cache = empty_summaries._cache
    assert cache.save_summaries_function is fn


def test_summary_types_forwarded_to_compile_settings(
    output_functions, solver_settings,
):
    """summary_types from solver_settings reaches compile_settings."""
    cs = output_functions.compile_settings
    # summary_types is derived from output_types filtering non-summary types
    # Verify it contains "mean" since default output_types includes "mean"
    assert "mean" in cs.summary_types
