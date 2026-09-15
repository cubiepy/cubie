"""Eviction and staging through real solves; units in test_memmgmt."""


import numpy as np
import pytest

from cubie.memory import MemoryManager
from cubie.memory.array_requests import ArrayResponse
from tests._utils import (
    DEVICE_SOLVE_SETTINGS,
    MockMemoryManager,
    _build_solver_instance,
)

# Reported free bytes that chunk the default 9-run batch. Eviction
# and a collapsed budget rewrite a solver's run partition for good,
# so the tests that provoke them own a private manager at this limit
# rather than sharing the session low-memory manager.
_PRIVATE_CHUNKING_BYTES = 700


def test_idle_solver_evicted_under_pressure_and_self_heals(
    system,
    solver_settings,
    batch_input_arrays,
    driver_settings,
):
    """A competing solve evicts an idle solver that later recovers.

    Eviction rewrites both solvers' run partitions for good, so the
    pair and their manager are private to this test.
    """
    manager = MockMemoryManager(
        forced_free_mem=_PRIVATE_CHUNKING_BYTES
    )
    idle_solver = _build_solver_instance(
        system=system,
        solver_settings={
            **solver_settings, "stream_group": "evicted_idle",
        },
        driver_settings=driver_settings,
        memory_manager=manager,
    )
    competing_solver = _build_solver_instance(
        system=system,
        solver_settings={
            **solver_settings, "stream_group": "evicting_competitor",
        },
        driver_settings=driver_settings,
        memory_manager=manager,
    )
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)

    try:
        idle_solver.solve(y0, params, **solve_kwargs)
        idle_outputs_id = id(idle_solver.kernel.output_arrays)
        assert manager.registry[idle_outputs_id].allocated_bytes > 0

        competing_solver.solve(y0, params, **solve_kwargs)
        assert manager.registry[idle_outputs_id].allocated_bytes == 0

        result = idle_solver.solve(y0, params, **solve_kwargs)
        assert manager.registry[idle_outputs_id].allocated_bytes > 0
        assert np.isfinite(result.as_numpy["time_domain_array"]).all()
    finally:
        idle_solver.close()
        competing_solver.close()


def test_empty_peer_response_changes_nothing(solver_mutable):
    """An empty peer response leaves array state unchanged."""
    arrays = solver_mutable.kernel.output_arrays
    chunks = arrays._chunks
    memory_types = {
        name: slot.memory_type
        for name, slot in arrays.host.iter_managed_arrays()
    }
    arrays._on_allocation_complete(
        ArrayResponse(
            arr={}, chunks=99, chunk_length=1, chunked_shapes={}
        )
    )
    assert arrays._chunks == chunks
    assert {
        name: slot.memory_type
        for name, slot in arrays.host.iter_managed_arrays()
    } == memory_types


def test_repeat_solve_with_held_result_and_collapsed_vram(
    system,
    solver_settings,
    batch_input_arrays,
    driver_settings,
):
    """A held result plus vanished free VRAM does not break a re-solve.

    The first result keeps its buffers, forcing the second solve to
    reallocate. Free device memory then reads as zero (the first
    solve's buffers and pool retention account for it), so the
    reallocation must reuse the owner's existing run partition
    instead of recomputing one from a zero budget. Collapsing the
    budget rewrites the partition for good, so the solver and its
    manager are private to this test.
    """
    manager = MockMemoryManager(
        forced_free_mem=_PRIVATE_CHUNKING_BYTES
    )
    solver = _build_solver_instance(
        system=system,
        solver_settings={
            **solver_settings, "stream_group": "collapsed_vram",
        },
        driver_settings=driver_settings,
        memory_manager=manager,
    )
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)

    try:
        first = solver.solve(y0, params, **solve_kwargs)
        assert solver.chunks > 1

        manager._custom_limit = 0
        second = solver.solve(y0, params, **solve_kwargs)
        np.testing.assert_array_equal(
            first.time_domain_array, second.time_domain_array
        )
    finally:
        solver.close()


def test_outputs_above_pinned_ceiling_stay_pageable(
    system, solver_settings, driver_settings, batch_input_arrays,
):
    """With a tiny pinned ceiling every buffer is pageable, not pinned.

    The pinned ceiling is a property of the memory manager, not a
    solver setting, so the solver is built against its own manager.
    The solve runs entirely through the staged-transfer path and
    still produces correct results.
    """
    manager = MemoryManager(pinned_max_bytes=0)
    settings = solver_settings.copy()
    settings["stream_group"] = "pinned_ceiling"
    solver = _build_solver_instance(
        system=system,
        solver_settings=settings,
        driver_settings=driver_settings,
        memory_manager=manager,
    )
    try:
        result = solver.solve(
            batch_input_arrays[0],
            batch_input_arrays[1],
            drivers=driver_settings,
            duration=0.1,
        )
        slot_types = {
            slot.memory_type
            for _, slot in (
                solver.kernel.output_arrays.host.iter_managed_arrays()
            )
        }
        assert "pinned" not in slot_types
        assert np.isfinite(result.time_domain_array).all()
    finally:
        solver.close()


def test_iteration_counters_collapse_when_inactive(
    solver_mutable, batch_input_arrays, driver_settings
):
    """An unrequested counters buffer is a placeholder, not full size."""
    y0, params = batch_input_arrays
    result = solver_mutable.solve(
        y0, params, drivers=driver_settings, duration=0.1
    )
    assert result.iteration_counters is None
    assert result._iteration_counters.size == 1


@pytest.mark.parametrize(
    # Any chain that requests iteration_counters serves this test.
    "solver_settings_override",
    [DEVICE_SOLVE_SETTINGS],
    indirect=True,
)
def test_iteration_counters_full_size_when_requested(
    solver_mutable, batch_input_arrays, driver_settings
):
    """Requested counters come back per save point and per run."""
    y0, params = batch_input_arrays
    result = solver_mutable.solve(
        y0, params, drivers=driver_settings, duration=0.1
    )
    counters = result.iteration_counters
    assert counters is not None
    assert counters.shape[1] == 4
    assert counters.shape[2] == solver_mutable.num_runs
    assert counters.shape[0] > 1
