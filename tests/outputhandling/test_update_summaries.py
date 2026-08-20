"""Tests for cubie.outputhandling.update_summaries."""

from __future__ import annotations


from cubie.outputhandling.update_summaries import (
    chain_metrics,
    do_nothing,
    update_summary_factory,
)
from cubie.outputhandling.summarymetrics import summary_metrics


# -- do_nothing / chain_metrics base case -------------------------------- #


def test_chain_metrics_empty_returns_do_nothing():
    """Empty metric_functions list returns the do_nothing sentinel."""
    result = chain_metrics([], [], [], [])
    assert result is do_nothing


# -- chain_metrics with metrics ------------------------------------------ #


def test_chain_metrics_single_metric_returns_distinct_wrapper():
    """Single metric produces a wrapper that is not do_nothing."""
    fns = summary_metrics.update_functions(["mean"])
    offsets = summary_metrics.buffer_offsets(["mean"])
    sizes = summary_metrics.buffer_sizes(["mean"])
    params = summary_metrics.params(["mean"])

    result = chain_metrics(fns, offsets, sizes, params)
    assert result is not do_nothing
    # Device invocation tested in integrated_numerical_tests


def test_chain_metrics_multiple_metrics_returns_distinct_wrapper():
    """Multiple metrics produce a single chained wrapper."""
    metrics_list = ["mean", "max"]
    fns = summary_metrics.update_functions(metrics_list)
    offsets = summary_metrics.buffer_offsets(metrics_list)
    sizes = summary_metrics.buffer_sizes(metrics_list)
    params = summary_metrics.params(metrics_list)

    result = chain_metrics(fns, offsets, sizes, params)
    assert result is not do_nothing


def test_chain_metrics_single_and_multiple_produce_different_wrappers():
    """Chains of different length produce distinct wrapper objects."""
    single_fns = summary_metrics.update_functions(["mean"])
    single_offsets = summary_metrics.buffer_offsets(["mean"])
    single_sizes = summary_metrics.buffer_sizes(["mean"])
    single_params = summary_metrics.params(["mean"])

    multi_list = ["mean", "max"]
    multi_fns = summary_metrics.update_functions(multi_list)
    multi_offsets = summary_metrics.buffer_offsets(multi_list)
    multi_sizes = summary_metrics.buffer_sizes(multi_list)
    multi_params = summary_metrics.params(multi_list)

    single_result = chain_metrics(
        single_fns, single_offsets, single_sizes, single_params,
    )
    multi_result = chain_metrics(
        multi_fns, multi_offsets, multi_sizes, multi_params,
    )
    assert single_result is not multi_result


# -- update_summary_factory: branch isolation ----------------------------- #


def test_factory_states_and_observables_with_metrics():
    """Factory callable when both state and observable indices are non-empty.
    """
    result = update_summary_factory(
        summaries_buffer_height_per_var=1,
        summarised_state_indices=[0, 1],
        summarised_observable_indices=[0],
        summaries_list=["mean"],
    )
    assert result.__name__ == "update_summary_metrics_func"


def test_factory_states_only_no_observables():
    """summarise_states=True, summarise_observables=False."""
    result = update_summary_factory(
        summaries_buffer_height_per_var=1,
        summarised_state_indices=[0],
        summarised_observable_indices=[],
        summaries_list=["mean"],
    )
    assert result.__name__ == "update_summary_metrics_func"


def test_factory_observables_only_no_states():
    """summarise_states=False, summarise_observables=True."""
    result = update_summary_factory(
        summaries_buffer_height_per_var=1,
        summarised_state_indices=[],
        summarised_observable_indices=[0, 1],
        summaries_list=["mean"],
    )
    assert result.__name__ == "update_summary_metrics_func"


def test_factory_no_indices_skips_both_branches():
    """Both summarise_states and summarise_observables are False."""
    result = update_summary_factory(
        summaries_buffer_height_per_var=1,
        summarised_state_indices=[],
        summarised_observable_indices=[],
        summaries_list=["mean"],
    )
    assert result.__name__ == "update_summary_metrics_func"


def test_factory_empty_summaries_skips_both_branches():
    """Empty summaries_list makes num_metrics=0, disabling both branches."""
    result = update_summary_factory(
        summaries_buffer_height_per_var=0,
        summarised_state_indices=[0],
        summarised_observable_indices=[0],
        summaries_list=[],
    )
    assert result.__name__ == "update_summary_metrics_func"


def test_factory_multiple_metrics_buffer_height():
    """Factory uses registry buffer heights for multi-metric chains."""
    metrics_list = ["mean", "max", "rms"]
    height = summary_metrics.summaries_buffer_height(metrics_list)
    result = update_summary_factory(
        summaries_buffer_height_per_var=height,
        summarised_state_indices=[0, 1, 2],
        summarised_observable_indices=[0, 1],
        summaries_list=metrics_list,
    )
    assert result.__name__ == "update_summary_metrics_func"


# -- Wiring through output_functions ------------------------------------- #


def test_update_summaries_wired_through_output_functions(output_functions):
    """update_summary_factory result is stored in output_functions cache."""
    # Access property to trigger lazy build
    fn = output_functions.update_summaries_func
    cache = output_functions._cache
    assert cache.update_summaries_function is fn
