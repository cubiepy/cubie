"""Solver-level eviction, spill, and staging integration tests.

Memory-manager unit behaviour (thresholds, spill files, stream-local
waits) is covered in ``tests/memory/test_memmgmt.py``; these tests
exercise the same machinery through real solves.
"""

from pathlib import Path

import numpy as np
import pytest

from cubie.cuda_simsafe import cuda
from cubie.memory import MemoryManager
from cubie.memory.array_requests import ArrayResponse
from cubie.memory.mem_manager import HOST_STAGING_BYTES
from tests._utils import (
    DEVICE_SOLVE_SETTINGS,
    MockMemoryManager,
    _build_solver_instance,
)


# One chain for every spill test; states-only time-domain output
# makes the combined array a state-buffer view (zero-copy test).
_SPILL_THRESHOLD = {
    "host_spill_threshold": 512,
    "output_types": ["state", "time"],
}

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


@pytest.mark.parametrize(
    "solver_settings_override", [_SPILL_THRESHOLD], indirect=True
)
def test_host_arrays_spill_to_disk_and_results_match(
    solver_mutable,
    system,
    solver_settings,
    driver_settings,
    batch_input_arrays,
):
    """Spilled outputs match in-RAM outputs exactly.

    The reference differs from the chain by exactly one memory
    setting (``host_spill_threshold=None``), so it is built directly;
    it compiles the same kernel as the spilling side.
    """
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.2)

    spilled = solver_mutable.solve(y0, params, **solve_kwargs)
    assert isinstance(spilled.state, np.memmap)
    assert Path(spilled.state._cubie_spill_path).exists()

    reference_settings = solver_settings.copy()
    reference_settings["host_spill_threshold"] = None
    reference_settings["stream_group"] = "spill_reference"
    reference_solver = _build_solver_instance(
        system=system,
        solver_settings=reference_settings,
        driver_settings=driver_settings,
        memory_manager=MemoryManager(),
    )
    try:
        reference = reference_solver.solve(y0, params, **solve_kwargs)
        spilled_numpy = spilled.as_numpy["time_domain_array"]
        assert type(spilled_numpy) is np.ndarray
        np.testing.assert_array_equal(
            spilled_numpy, reference.as_numpy["time_domain_array"]
        )
    finally:
        reference_solver.close()


def test_solver_spill_policies_are_independent(
    system,
    solver_settings,
    driver_settings,
    batch_input_arrays,
    tmp_path,
):
    """Solvers sharing one manager keep separate spill policies.

    The pair differs only in registration settings, and each side
    needs its own threshold, so both are built directly.
    """
    manager = MemoryManager()
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)

    spill_settings = solver_settings.copy()
    spill_settings["host_spill_threshold"] = 1
    spill_settings["spill_directory"] = str(tmp_path)
    spill_settings["stream_group"] = "spill_policy_spill"
    ram_settings = solver_settings.copy()
    ram_settings["host_spill_threshold"] = 2**30
    ram_settings["stream_group"] = "spill_policy_ram"

    spill_solver = _build_solver_instance(
        system=system,
        solver_settings=spill_settings,
        driver_settings=driver_settings,
        memory_manager=manager,
    )
    ram_solver = _build_solver_instance(
        system=system,
        solver_settings=ram_settings,
        driver_settings=driver_settings,
        memory_manager=manager,
    )
    try:
        spill_result = spill_solver.solve(y0, params, **solve_kwargs)
        ram_result = ram_solver.solve(y0, params, **solve_kwargs)

        assert isinstance(spill_result.state, np.memmap)
        assert len(list(tmp_path.iterdir())) > 0
        assert not isinstance(ram_result.state, np.memmap)
    finally:
        spill_solver.close()
        ram_solver.close()


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


@pytest.mark.parametrize(
    "solver_settings_override", [_SPILL_THRESHOLD], indirect=True
)
@pytest.mark.parametrize(
    "batch_settings_override",
    [{"num_state_vals_0": 9, "num_param_vals_0": 9}],
    indirect=True,
)
def test_shape_change_updates_memmap_metadata(
    solver_mutable, batch_input_arrays, driver_settings
):
    """Host backing metadata follows each replacement array."""
    y0, params = batch_input_arrays
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)

    first = solver_mutable.solve(y0, params, **solve_kwargs)
    assert isinstance(first.state, np.memmap)
    old_path = Path(first.state._cubie_spill_path)
    assert old_path.exists()

    # Drop the result: the spilled buffer returns to its slot and is
    # released when the smaller solve replaces it.
    del first
    second = solver_mutable.solve(y0[:, :3], params[:, :3], **solve_kwargs)
    assert not isinstance(second.state, np.memmap)
    assert not old_path.exists()


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override", [_SPILL_THRESHOLD], indirect=True
)
def test_spill_solve_is_async(
    solver_mutable,
    batch_input_arrays,
    driver_settings,
    start_cuda_busy_work,
):
    """Spill staging transfers leave unrelated CUDA work running."""
    y0, params = batch_input_arrays
    # Warm solve: compile the kernel, set the driver coefficients, and
    # allocate the spill-backed host arrays.
    solver_mutable.solve(
        y0, params, drivers=driver_settings, duration=0.1
    )

    # Numba's deferred-deallocation queue flushes onto the legacy
    # default stream, which serializes every blocking stream —
    # including the canary against the solver's run stream. Hold the
    # queue from before the canary launches until the assertion, so
    # the canary sees only waits the solve itself performs. The
    # bracketed solve omits drivers: re-supplying them rebuilds the
    # driver function and recompiles, which is not the staging path
    # under test.
    with cuda.defer_cleanup():
        work, stream, done, release = start_cuda_busy_work()
        try:
            solver_mutable.solve(y0, params, duration=0.1)
            assert not done.query()
            pool = solver_mutable.kernel.output_arrays._buffer_pool
            for buffers in pool._buffers.values():
                assert all(
                    buffer.array.nbytes <= HOST_STAGING_BYTES
                    for buffer in buffers
                )
        finally:
            release()
            stream.synchronize()
            assert work.copy_to_host()[0] > 0


@pytest.mark.parametrize(
    "solver_settings_override", [_SPILL_THRESHOLD], indirect=True
)
def test_spilled_result_assembly_is_zero_copy(
    solver_mutable, batch_input_arrays, driver_settings
):
    """Assembling a spilled result never materialises it in RAM.

    With states as the only time-domain output, the combined array
    and the time samples are views of the owned disk-backed buffer.
    """
    y0, params = batch_input_arrays
    result = solver_mutable.solve(
        y0, params, drivers=driver_settings, duration=0.2
    )
    assert isinstance(result.state, np.memmap)
    assert np.shares_memory(result.time_domain_array, result.state)
    assert np.shares_memory(result.time, result.state)
    result.close()


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
            for _, slot in solver.kernel.output_arrays.host.iter_managed_arrays()
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
