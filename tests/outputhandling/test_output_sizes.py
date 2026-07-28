"""Tests for cubie.outputhandling.output_sizes."""

from __future__ import annotations

import pytest

from tests._utils import STATE_OBS_NO_TIMING

from cubie.outputhandling.output_sizes import (
    BatchInputSizes,
    BatchOutputSizes,
    OutputArrayHeights,
    SingleRunOutputSizes,
)


# ── ArraySizingClass.nonzero ─────────────────────────── #

def test_nonzero_coerces_zero_ints_to_one():
    """Zero int fields become 1; non-zero fields unchanged."""
    heights = OutputArrayHeights(
        state=0, observables=3, state_summaries=0,
        observable_summaries=7, per_variable=0,
    )
    nz = heights.nonzero
    assert nz.state == 1
    assert nz.observables == 3
    assert nz.state_summaries == 1
    assert nz.observable_summaries == 7
    assert nz.per_variable == 1


def test_nonzero_coerces_zero_tuples_to_ones():
    """Zero elements within tuples become 1; non-zero preserved."""
    sizes = SingleRunOutputSizes(
        state=(0, 5), observables=(3, 0),
        state_summaries=(0, 0), observable_summaries=(2, 4),
    )
    nz = sizes.nonzero
    # ensure_nonzero_size converts entire tuple to 1s if any zero
    assert nz.state == (1, 1)
    assert nz.observables == (1, 1)
    assert nz.state_summaries == (1, 1)
    assert nz.observable_summaries == (2, 4)


def test_nonzero_does_not_mutate_original():
    """The original object retains its zero values after .nonzero."""
    original = OutputArrayHeights(
        state=0, observables=3, state_summaries=0,
        observable_summaries=0, per_variable=0,
    )
    nz = original.nonzero
    assert original.state == 0
    assert original.per_variable == 0
    assert nz.state == 1
    assert nz.per_variable == 1


# ── OutputArrayHeights defaults ──────────────────────── #

def test_output_array_heights_defaults():
    """Default construction yields all-1 heights."""
    h = OutputArrayHeights()
    assert h.state == 1
    assert h.observables == 1
    assert h.state_summaries == 1
    assert h.observable_summaries == 1
    assert h.per_variable == 1


# ── OutputArrayHeights.from_output_fns ───────────────── #

def test_from_output_fns_state_height_with_time(output_functions):
    """state height = n_saved_states + 1 when save_time is True."""
    # Default solver_settings has output_types containing "time"
    # so save_time should be True
    heights = OutputArrayHeights.from_output_fns(output_functions)
    expected = output_functions.n_saved_states + int(
        output_functions.save_time
    )
    assert heights.state == expected


@pytest.mark.parametrize(
    "solver_settings_override",
    [pytest.param(STATE_OBS_NO_TIMING, id="no-time")],
    indirect=True,
)
def test_from_output_fns_state_height_without_time(output_functions):
    """state height = n_saved_states when save_time is False."""
    assert output_functions.save_time is False
    heights = OutputArrayHeights.from_output_fns(output_functions)
    assert heights.state == output_functions.n_saved_states


def test_from_output_fns_observables(output_functions):
    """observables height equals n_saved_observables."""
    heights = OutputArrayHeights.from_output_fns(output_functions)
    assert heights.observables == output_functions.n_saved_observables


def test_from_output_fns_state_summaries(output_functions):
    """state_summaries equals state_summaries_output_height."""
    heights = OutputArrayHeights.from_output_fns(output_functions)
    assert heights.state_summaries == (
        output_functions.state_summaries_output_height
    )


def test_from_output_fns_observable_summaries(output_functions):
    """observable_summaries equals observable_summaries_output_height."""
    heights = OutputArrayHeights.from_output_fns(output_functions)
    assert heights.observable_summaries == (
        output_functions.observable_summaries_output_height
    )


def test_from_output_fns_per_variable(output_functions):
    """per_variable equals summaries_output_height_per_var."""
    heights = OutputArrayHeights.from_output_fns(output_functions)
    assert heights.per_variable == (
        output_functions.summaries_output_height_per_var
    )


# ── SingleRunOutputSizes.from_solver ─────────────────── #

def test_single_run_state_shape(solverkernel):
    """state shape = (output_samples, heights.state)."""
    sizes = SingleRunOutputSizes.from_solver(solverkernel)
    heights = solverkernel.output_array_heights
    assert sizes.state == (solverkernel.output_length, heights.state)


def test_single_run_observables_shape(solverkernel):
    """observables shape = (output_samples, heights.observables)."""
    sizes = SingleRunOutputSizes.from_solver(solverkernel)
    heights = solverkernel.output_array_heights
    assert sizes.observables == (
        solverkernel.output_length, heights.observables,
    )


def test_single_run_state_summaries_shape(solverkernel):
    """state_summaries shape = (summarise_samples, heights.state_summaries)."""
    sizes = SingleRunOutputSizes.from_solver(solverkernel)
    heights = solverkernel.output_array_heights
    assert sizes.state_summaries == (
        solverkernel.summaries_length, heights.state_summaries,
    )


def test_single_run_observable_summaries_shape(solverkernel):
    """observable_summaries = (summarise_samples, heights.observable_summaries)."""
    sizes = SingleRunOutputSizes.from_solver(solverkernel)
    heights = solverkernel.output_array_heights
    assert sizes.observable_summaries == (
        solverkernel.summaries_length, heights.observable_summaries,
    )


# ── BatchInputSizes.from_solver ──────────────────────── #

def test_batch_input_initial_values(solverkernel):
    """initial_values = (states, num_runs)."""
    sizes = BatchInputSizes.from_solver(solverkernel)
    ss = solverkernel.system_sizes
    assert sizes.initial_values == (ss.states, solverkernel.num_runs)


def test_batch_input_parameters(solverkernel):
    """parameters = (parameters, num_runs)."""
    sizes = BatchInputSizes.from_solver(solverkernel)
    ss = solverkernel.system_sizes
    assert sizes.parameters == (ss.parameters, solverkernel.num_runs)


def test_batch_input_driver_coefficients(solverkernel):
    """driver_coefficients mirrors the kernel's pinned layout."""
    sizes = BatchInputSizes.from_solver(solverkernel)
    assert (
        sizes.driver_coefficients
        == solverkernel.driver_coefficients_shape
    )


def test_batch_input_driver_coefficients_follow_pin(solverkernel_mutable):
    """An updated coefficient layout flows into the sizes."""
    solverkernel_mutable.update(driver_coefficients_shape=(7, 2, 4))
    sizes = BatchInputSizes.from_solver(solverkernel_mutable)
    assert sizes.driver_coefficients == (7, 2, 4)


# ── BatchOutputSizes.from_solver ─────────────────────── #

def test_batch_output_adds_num_runs_dimension(solverkernel):
    """All single-run shapes gain num_runs as third dimension."""
    single = SingleRunOutputSizes.from_solver(solverkernel)
    batch = BatchOutputSizes.from_solver(solverkernel)
    nr = solverkernel.num_runs
    assert batch.state == (single.state[0], single.state[1], nr)
    assert batch.observables == (
        single.observables[0], single.observables[1], nr,
    )
    assert batch.state_summaries == (
        single.state_summaries[0], single.state_summaries[1], nr,
    )
    assert batch.observable_summaries == (
        single.observable_summaries[0],
        single.observable_summaries[1],
        nr,
    )


def test_batch_output_status_codes(solverkernel):
    """status_codes = (num_runs,)."""
    batch = BatchOutputSizes.from_solver(solverkernel)
    assert batch.status_codes == (solverkernel.num_runs,)


def test_batch_output_iteration_counters(solverkernel):
    """Counters are (n_saves, 4, num_runs) only when requested."""
    single = SingleRunOutputSizes.from_solver(solverkernel)
    batch = BatchOutputSizes.from_solver(solverkernel)
    nr = solverkernel.num_runs
    if solverkernel.save_counters:
        assert batch.iteration_counters == (single.state[0], 4, nr)
    else:
        assert batch.iteration_counters == (0, 0, 0)
