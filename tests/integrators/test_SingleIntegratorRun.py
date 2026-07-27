"""Tests for cubie.integrators.SingleIntegratorRun."""

from __future__ import annotations

import numpy as np
import pytest
from tests._utils import (
    FLOAT64_PRECISION,
    STATE_OBS_NO_TIMING,
    STATE_ONLY_NO_SUMMARIES,
    SUMMARY_ONLY_TIMED,
)


# ── Forwarding property tests ──────────────────────────────────────────── #
#
# SingleIntegratorRun is a thin property-aggregation layer.  Tests are
# grouped by the source component each property delegates to.
#
# These use SHORT_RUN defaults (no solver_settings_override) — the
# cheapest fixture set.  Only tests needing finer resolution or
# specific algorithm/controller combos override settings.

# -- compile_settings forwarding ----------------------------------------- #

def test_compile_settings_forwarding(
    single_integrator_run, solver_settings
):
    """algorithm and step_controller forward from compile_settings."""
    run = single_integrator_run
    cs = run.compile_settings
    assert run.algorithm == cs.algorithm
    assert run.step_controller == cs.step_controller
    assert solver_settings["algorithm"].lower() in run.algorithm
    assert run.step_controller == solver_settings["step_controller"].lower()


# -- _loop forwarding --------------------------------------------------- #

def test_loop_forwarding(single_integrator_run):
    """Properties that delegate to _loop."""
    run = single_integrator_run
    loop = run._loop

    assert run.shared_memory_elements == loop.shared_buffer_size
    assert run.persistent_local_elements == loop.persistent_local_buffer_size

    assert run.save_every == loop.save_every
    assert run.summarise_every == loop.summarise_every
    assert run.sample_summaries_every == loop.sample_summaries_every
    assert run.save_last == loop.compile_settings.save_last

    assert run.compile_flags is loop.compile_flags
    assert run.save_state_fn is loop.save_state_fn
    assert run.update_summaries_fn is loop.update_summaries_fn
    assert run.save_summaries_fn is loop.save_summaries_fn


# -- _step_controller forwarding ---------------------------------------- #

def test_step_controller_forwarding(single_integrator_run, tolerance):
    """Properties that delegate to _step_controller."""
    run = single_integrator_run
    ctrl = run._step_controller

    assert run.dt == pytest.approx(
        ctrl.dt, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert run.dt_min == pytest.approx(
        ctrl.dt_min, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert run.dt_max == pytest.approx(
        ctrl.dt_max, rel=tolerance.rel_tight, abs=tolerance.abs_tight
    )
    assert run.is_adaptive == ctrl.is_adaptive

    # atol / rtol / dt use hasattr guard
    if hasattr(ctrl, "atol"):
        np.testing.assert_array_equal(np.asarray(run.atol),
                                      np.asarray(ctrl.atol))
    else:
        assert run.atol is None

    if hasattr(ctrl, "rtol"):
        np.testing.assert_array_equal(np.asarray(run.rtol),
                                      np.asarray(ctrl.rtol))
    else:
        assert run.rtol is None

    if hasattr(ctrl, "dt"):
        assert run.dt == pytest.approx(
            ctrl.dt, rel=tolerance.rel_tight, abs=tolerance.abs_tight
        )
    else:
        assert run.dt is None


# -- _algo_step forwarding ---------------------------------------------- #

def test_algo_step_forwarding(single_integrator_run):
    """Properties that delegate to _algo_step."""
    run = single_integrator_run
    assert run.threads_per_step == run._algo_step.threads_per_step
    assert run.evaluate_f is run._algo_step.evaluate_f


# -- _output_functions forwarding --------------------------------------- #

def test_output_functions_forwarding(single_integrator_run):
    """Properties that delegate to _output_functions."""
    run = single_integrator_run
    of = run._output_functions

    assert run.save_state_func is of.save_state_func
    assert run.update_summaries_func is of.update_summaries_func
    assert run.save_summary_metrics_func is of.save_summary_metrics_func
    assert run.output_types == of.output_types
    assert run.output_compile_flags == of.compile_flags
    assert run.save_time == of.save_time

    np.testing.assert_array_equal(run.saved_state_indices,
                                  of.saved_state_indices)
    np.testing.assert_array_equal(run.saved_observable_indices,
                                  of.saved_observable_indices)
    np.testing.assert_array_equal(run.summarised_state_indices,
                                  of.summarised_state_indices)
    np.testing.assert_array_equal(run.summarised_observable_indices,
                                  of.summarised_observable_indices)
    assert run.output_array_heights == of.output_array_heights
    assert run.summary_legend_per_variable == of.summary_legend_per_variable
    assert run.summary_unit_modifications == of.summary_unit_modifications


# -- _system forwarding ------------------------------------------------- #

def test_system_forwarding(single_integrator_run, system):
    """system and system_sizes forward from _system."""
    run = single_integrator_run
    assert run.system == system
    assert run.system_sizes == system.sizes


# -- chained forwarding ------------------------------------------------- #

def test_save_summaries_func_chain(single_integrator_run):
    """save_summaries_func chains through save_summary_metrics_func."""
    run = single_integrator_run
    assert run.save_summaries_func is run.save_summary_metrics_func


# ── shared_memory_bytes ─────────────────────────────────────────────────── #

def test_shared_memory_bytes(single_integrator_run, precision):
    """shared_memory_bytes = elements * dtype itemsize."""
    run = single_integrator_run
    expected = run.shared_memory_elements * np.dtype(precision).itemsize
    assert run.shared_memory_bytes == expected


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(FLOAT64_PRECISION, id="float64")],
    indirect=True,
)
def test_shared_memory_bytes_float64(single_integrator_run, precision):
    """shared_memory_bytes tracks the wider dtype's itemsize."""
    run = single_integrator_run
    expected = run.shared_memory_elements * np.dtype(precision).itemsize
    assert run.shared_memory_bytes == expected


# ── output_length ────────────────────────────────────────────────────────── #

def test_output_length_periodic(single_integrator_run, solver_settings):
    """output_length includes initial sample + floor(duration/save_every)."""
    duration = float(solver_settings["duration"])
    save_every = float(solver_settings["save_every"])
    expected = int(duration / save_every) + 1
    assert single_integrator_run.output_length(duration) == expected


@pytest.mark.parametrize(
    "solver_settings_override",
    [STATE_OBS_NO_TIMING],
    indirect=True,
)
def test_output_length_save_last_only(single_integrator_run):
    """output_length returns 2 when save_last=True and no save_every."""
    assert single_integrator_run.save_last is True
    assert single_integrator_run.output_length(0.2) == 2


@pytest.mark.parametrize(
    "solver_settings_override",
    [SUMMARY_ONLY_TIMED],
    indirect=True,
)
def test_output_length_no_time_domain(single_integrator_run):
    """output_length returns 1 when no time-domain outputs and no save_last."""
    assert single_integrator_run.save_last is False
    assert single_integrator_run.output_length(0.2) == 1


# ── summaries_length ────────────────────────────────────────────────────── #

def test_summaries_length_periodic(single_integrator_run, solver_settings):
    """summaries_length = int(duration / summarise_every)."""
    duration = float(solver_settings["duration"])
    summarise_every = float(solver_settings["summarise_every"])
    expected = int(duration / summarise_every)
    assert single_integrator_run.summaries_length(duration) == expected


@pytest.mark.parametrize(
    "solver_settings_override",
    [STATE_ONLY_NO_SUMMARIES],
    indirect=True,
)
def test_summaries_length_none(single_integrator_run):
    """summaries_length returns 0 when summarise_every is None."""
    assert single_integrator_run.summaries_length(0.3) == 0


# ── device_function ──────────────────────────────────────────────────────── #

def test_device_function_callable(single_integrator_run):
    """device_function returns a callable."""
    run = single_integrator_run
    fn = run.device_function
    assert callable(fn)


# ── Cache validity after build ─────────────────────────────────────────── #

def test_cache_valid_after_build(single_integrator_run):
    """All caches valid after device_function accessed."""
    run = single_integrator_run
    _ = run.device_function  # trigger build
    assert run.cache_valid is True
    assert run._loop.cache_valid is True
    assert run._algo_step.cache_valid is True
    assert run._step_controller.cache_valid is True
    assert run._output_functions.cache_valid is True
