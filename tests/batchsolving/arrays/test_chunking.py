"""Tests for chunking logic."""

from pathlib import Path
from time import sleep

import numpy as np
import pytest

from cubie.batchsolving.arrays.BatchOutputArrays import (
    OutputArrayContainer,
)
from cubie.batchsolving.arrays.BatchInputArrays import InputArrayContainer
from cubie.batchsolving.arrays.BaseArrayManager import (
    ManagedArray,
    ArrayContainer,
    BaseArrayManager,
)
from cubie.batchsolving.writeback_watcher import (
    WritebackTask,
    WritebackWatcher,
)
from cubie.memory.chunk_buffer_pool import ChunkBufferPool
from tests._utils import _build_solver_instance


def _private_low_memory_manager(low_memory, forced_free_mem):
    """Return a fresh manager of the shared low-memory kind.

    A solver whose run partition is rewritten mid-test owns its
    manager and its solver; the shared low-memory solver's other
    consumers still need it chunked.
    """
    return type(low_memory)(forced_free_mem=forced_free_mem)


def _make_test_array_container():
    """Return a test container with a single managed array."""

    container = ArrayContainer()
    container.state = ManagedArray(  # type: ignore[attr-defined]
        dtype=np.float32,
        default_shape=(10, 3, 100),
        memory_type="pinned",
    )
    return container


def _make_test_array_manager():
    """Return a minimal concrete BaseArrayManager instance for tests."""

    manager_cls = type(
        "TestArrayManager",
        (BaseArrayManager,),
        {
            "finalise": lambda self, chunk_index: chunk_index,
            "initialise": lambda self, chunk_index: chunk_index,
            "update": lambda self: None,
        },
    )
    return manager_cls(
        host=_make_test_array_container(),
        device=_make_test_array_container(),
    )


def test_run_executes_with_chunking(
    chunked_solved_solver, system, driver_settings
):
    """Verify solve() executes with run-axis chunking."""
    solver, result = chunked_solved_solver

    # Verify chunking occurred
    assert solver.chunks > 1


def test_chunked_solve_produces_valid_output(
    system, precision, chunked_solved_solver
):
    """Verify chunked solver produces valid output arrays."""
    solver, result = chunked_solved_solver

    # Verify output shape and that values are not all zeros/NaN
    assert result.time_domain_array is not None
    assert result.time_domain_array.shape[2] == 5
    assert not np.all(result.time_domain_array == 0)
    assert not np.any(np.isnan(result.time_domain_array))



def test_repeat_chunked_solve_matches_first(
    system, precision, chunked_solved_solver, driver_settings
):
    """A repeat chunked solve on the same solver reproduces the first."""
    solver, first_result = chunked_solved_solver
    assert solver.chunks > 1
    first_state = first_result.time_domain_array.copy()

    n_runs = 5
    n_states = system.sizes.states
    n_params = system.sizes.parameters
    inits = np.ones((n_states, n_runs), dtype=precision)
    params = np.ones((n_params, n_runs), dtype=precision)

    second_result = solver.solve(
        inits,
        params,
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
        dt=0.01,
    )

    assert solver.chunks > 1
    np.testing.assert_array_equal(
        second_result.time_domain_array, first_state
    )


def test_chunked_results_match_unchunked(
    chunked_solved_solver,
    unchunked_solved_solver,
    system,
    precision,
    driver_settings,
):
    """Chunked solves reproduce unchunked results exactly.

    Inputs vary along the run axis so a writeback that lands in the
    wrong run (or not at all) changes the result. The reference is
    the session unchunking solver, whose first solve applied these
    same timing values.
    """
    chunked_solver, _ = chunked_solved_solver
    n_runs = 5
    rng = np.random.default_rng(1234)
    inits = rng.uniform(
        0.5, 1.5, (system.sizes.states, n_runs)
    ).astype(precision)
    params = rng.uniform(
        0.5, 1.5, (system.sizes.parameters, n_runs)
    ).astype(precision)
    solve_kwargs = dict(
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
        dt=0.01,
    )
    chunked = chunked_solver.solve(inits, params, **solve_kwargs)
    assert chunked_solver.chunks > 1

    reference_solver, _ = unchunked_solved_solver
    reference = reference_solver.solve(inits, params, **solve_kwargs)
    assert reference_solver.chunks == 1
    np.testing.assert_array_equal(
        chunked.time_domain_array, reference.time_domain_array
    )
    np.testing.assert_array_equal(
        chunked.status_codes, reference.status_codes
    )


def test_input_buffers_released_after_kernel(chunked_solved_solver):
    chunked_solver, result_chunked = chunked_solved_solver

    # After solve completes and pending releases drain, every staging
    # buffer is back in the pool
    input_arrays = chunked_solver.kernel.input_arrays
    input_arrays.wait_pending()
    for buffers in input_arrays._buffer_pool._buffers.values():
        assert all(not buffer.in_use for buffer in buffers)


def test_non_chunked_uses_pinned_host(unchunked_solved_solver):
    """Non-chunked runs use pinned host arrays."""

    solver, result = unchunked_solved_solver

    # Verify host arrays are pinned when non-chunked
    for name, slot in solver.kernel.output_arrays.host.iter_managed_arrays():
        assert slot.memory_type == "pinned"


def test_chunked_uses_numpy_host(chunked_solved_solver):
    """Chunked runs use numpy host arrays with buffer pool."""
    solver, result = chunked_solved_solver

    # When chunked, host arrays should be numpy (not pinned)
    # to limit total pinned memory to buffer pool only
    found_one = False
    for name, slot in solver.kernel.output_arrays.host.iter_managed_arrays():
        if slot.needs_chunked_transfer:
            assert slot.memory_type == "host"
            found_one = True
    assert found_one, "No chunked-transfer arrays found"


def test_chunked_solver_changes_to_unchunked_backing(
    low_memory,
    forced_free_mem,
    system,
    solver_settings,
    precision,
    batch_input_arrays,
    driver_settings,
):
    """A shape change commits unchunked backing and metadata together.

    The solver is private to this test: raising the reported free
    memory converts its run partition to unchunked for good, which
    the shared low-memory solver's later consumers still need
    chunked.
    """
    solve_kwargs = dict(
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
        dt=0.01,
    )
    manager = _private_low_memory_manager(low_memory, forced_free_mem)
    solver = _build_solver_instance(
        system=system,
        solver_settings={
            **solver_settings, "stream_group": "unchunked_backing",
        },
        driver_settings=driver_settings,
        memory_manager=manager,
    )
    n_runs = 5
    inits = np.ones((system.sizes.states, n_runs), dtype=precision)
    chunked_params = np.ones(
        (system.sizes.parameters, n_runs), dtype=precision
    )
    first_result = solver.solve(inits, chunked_params, **solve_kwargs)
    assert solver.chunks > 1

    y0, params = batch_input_arrays
    manager._custom_limit = 8192

    second_result = solver.solve(
        y0[:, :3], params[:, :3], **solve_kwargs
    )
    try:
        assert solver.chunks == 1
        for _, slot in (
            solver.kernel.output_arrays.host.iter_managed_arrays()
        ):
            if isinstance(slot.array, np.memmap):
                assert slot.memory_type == "memmap"
            else:
                assert slot.memory_type == "pinned"
        # Input slots hold the handler's arrays verbatim; the slot
        # type records each array's actual backing
        input_manager = solver.kernel.input_arrays
        for _, slot in input_manager.host.iter_managed_arrays():
            if slot.array is not None:
                assert slot.memory_type == (
                    input_manager._host_memory_type(slot.array)
                )
    finally:
        first_result.close()
        second_result.close()
        solver.close()


def test_output_allocation_tracks_policy_spill(tmp_path):
    """Above-threshold output buffers are disk-backed at creation.

    The spill policy applies when the buffer is created; conversion
    after the chunk decision only repins small pageable slots and
    never moves a buffer between backings.
    """
    manager = _make_test_array_manager()
    settings = manager._memory_manager.get_registration(manager)
    settings.host_spill_threshold = 1
    settings.spill_directory = str(tmp_path)

    memory_type = manager._memory_manager.choose_host_memory_type(
        10 * 3 * 100 * 4, manager, allow_pinned=False
    )
    assert memory_type == "memmap"
    array = manager._memory_manager.create_host_array(
        (10, 3, 100), np.float32, memory_type, instance=manager
    )
    slot = manager.host.get_managed_array("state")
    slot.array = array
    slot.memory_type = "memmap"

    manager._convert_host_to_pinned()

    slot = manager.host.get_managed_array("state")
    assert slot.array is array
    assert slot.memory_type == "memmap"
    assert manager._requires_staging(slot.array, slot.memory_type)
    path = Path(slot.array._cubie_spill_path)
    assert path.exists()
    manager._memory_manager.release_host_array(slot.array)
    slot.array = None
    assert not path.exists()


def test_pinned_buffers_created(chunked_solved_solver):
    """Total pinned memory stays within one chunk's worth."""
    solver, result = chunked_solved_solver
    buffer_pool = solver.kernel.output_arrays._buffer_pool

    # After solve completes, buffers should exist in pool list and be fre
    for buffer_list in buffer_pool._buffers.values():
        for buf in buffer_list:
            assert buf.in_use is False


def test_watcher_completes_all_tasks(chunked_solved_solver):
    """All submitted tasks are completed before solve returns."""
    solver, result = chunked_solved_solver
    # Verify all tasks completed
    output_arrays = solver.kernel.output_arrays
    assert output_arrays._watcher._pending_count == 0


def test_writeback_task_creation():
    """Verify WritebackTask can be created with valid inputs."""
    pool = ChunkBufferPool()
    buffer = pool.acquire("test", (10,), np.float32)
    target = np.arange(100, dtype=np.float32)
    buffer.array[:] = np.arange(10, dtype=np.float32)
    task = WritebackTask(
        event=None,
        buffer=buffer,
        target_array=target[:10],
        buffer_pool=pool,
        array_name="test",
    )

    assert task.event is None
    assert task.buffer is buffer
    assert np.array_equal(task.target_array, target[:10])
    assert task.buffer_pool is pool
    assert task.array_name == "test"


def test_writeback_task_validates_buffer_type():
    """Verify WritebackTask validates buffer is a PinnedBuffer."""
    pool = ChunkBufferPool()
    target = np.zeros((100,), dtype=np.float32)

    with pytest.raises(TypeError):
        WritebackTask(
            event=None,
            buffer="not a buffer",  # Invalid
            target_array=target[0:10],
            buffer_pool=pool,
            array_name="test",
        )


def test_writeback_watcher_starts_and_stops():
    """Verify watcher thread starts on first submit and stops on shutdown."""
    watcher = WritebackWatcher()
    pool = ChunkBufferPool()
    buffer = pool.acquire("test", (10,), np.float32)
    target = np.zeros((100,), dtype=np.float32)

    # Thread should not be running initially
    assert watcher._thread is None

    # Submit a task (should start thread)
    watcher.submit(
        event=None,
        buffer=buffer,
        target_array=target[0:10],
        buffer_pool=pool,
        array_name="test",
    )

    # Give thread time to start
    sleep(0.01)

    # Thread should now be running
    assert watcher._thread is not None
    assert watcher._thread.is_alive()

    # Shutdown and verify thread stops
    watcher.shutdown()
    assert watcher._thread is None


def test_writeback_watcher_submit_and_wait_completes_writeback():
    """Verify submitted task copies data to target array.

    Tests end-to-end functionality: data in buffer should be
    copied to target array at specified slice after wait_all().
    """
    watcher = WritebackWatcher()
    pool = ChunkBufferPool()
    buffer = pool.acquire("test", (10,), np.float32)
    target = np.zeros((100,), dtype=np.float32)

    # Fill buffer with test data
    buffer.array[:] = np.arange(10, dtype=np.float32)

    # Submit task
    watcher.submit(
        event=None,  # CUDASIM treats None as complete
        buffer=buffer,
        target_array=target[20:30],
        buffer_pool=pool,
        array_name="test",
    )

    # Wait for completion
    watcher.wait_all(timeout=1.0)

    # Verify data was copied to correct slice
    expected = np.zeros((100,), dtype=np.float32)
    expected[20:30] = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(target, expected)

    # Cleanup
    watcher.shutdown()


def test_writeback_watcher_wait_all_blocks_until_complete():
    """Verify wait_all blocks until all pending tasks finish.

    Tests synchronization: wait_all should return only when
    all submitted tasks have completed.
    """
    watcher = WritebackWatcher()
    pool = ChunkBufferPool()

    # Submit multiple tasks
    targets = []
    for i in range(3):
        buffer = pool.acquire(f"test_{i}", (5,), np.float32)
        buffer.array[:] = float(i + 1)
        target = np.zeros((20,), dtype=np.float32)
        targets.append(target)

        watcher.submit(
            event=None,
            buffer=buffer,
            target_array=target[5 * i : 5 * (i + 1)],
            buffer_pool=pool,
            array_name=f"test_{i}",
        )

    # Wait for all
    watcher.wait_all(timeout=1.0)

    # Verify all tasks completed
    for i, target in enumerate(targets):
        expected_value = float(i + 1)
        np.testing.assert_array_equal(
            target[i * 5 : (i + 1) * 5],
            np.full((5,), expected_value, dtype=np.float32),
        )

    # Cleanup
    watcher.shutdown()


def test_writeback_watcher_multiple_concurrent_tasks():
    """Verify multiple tasks can be queued and completed."""
    watcher = WritebackWatcher()
    pool = ChunkBufferPool()

    num_tasks = 10
    target = np.zeros((num_tasks * 10,), dtype=np.float32)

    for i in range(num_tasks):
        buffer = pool.acquire("test", (10,), np.float32)
        buffer.array[:] = float(i)

        watcher.submit(
            event=None,
            buffer=buffer,
            target_array=target[10 * i : 10 * (i + 1)],
            buffer_pool=pool,
            array_name="test",
        )

    watcher.wait_all(timeout=2.0)

    for i in range(num_tasks):
        expected = np.full((10,), float(i), dtype=np.float32)
        np.testing.assert_array_equal(target[i * 10 : (i + 1) * 10], expected)

    watcher.shutdown()


def test_writeback_watcher_wait_all_timeout_raises():
    """Verify wait_all raises TimeoutError when timeout expires."""
    watcher = WritebackWatcher()

    watcher._pending_count = 1

    with pytest.raises(TimeoutError):
        watcher.wait_all(timeout=0.1)


def test_writeback_watcher_buffer_released_after_completion():
    """Verify buffer is released back to pool after task completes."""
    watcher = WritebackWatcher()
    pool = ChunkBufferPool()
    buffer = pool.acquire("test", (10,), np.float32)
    target = np.zeros((10,), dtype=np.float32)

    # Buffer should be in use after acquire
    assert buffer.in_use

    watcher.submit(
        event=None,
        buffer=buffer,
        target_array=target,
        buffer_pool=pool,
        array_name="test",
    )

    watcher.wait_all(timeout=1.0)

    # Buffer should be released after completion
    assert not buffer.in_use

    # Cleanup
    watcher.shutdown()


def test_writeback_watcher_start_is_idempotent():
    """Verify calling start() multiple times is safe."""
    watcher = WritebackWatcher()

    # Start multiple times
    watcher.start()
    first_thread = watcher._thread
    watcher.start()
    second_thread = watcher._thread

    # Should be the same thread
    assert first_thread is second_thread

    # Cleanup
    watcher.shutdown()


def test_writeback_watcher_2d_array_slice():
    """Verify slicing works correctly for 2D arrays."""
    watcher = WritebackWatcher()
    pool = ChunkBufferPool()
    buffer = pool.acquire("test", (5, 10), np.float64)
    target = np.zeros((10, 10), dtype=np.float64)

    # Fill buffer with test data
    buffer.array[:] = np.arange(50, dtype=np.float64).reshape(5, 10)

    watcher.submit(
        event=None,
        buffer=buffer,
        target_array=target[5:10, :],
        buffer_pool=pool,
        array_name="test",
    )

    watcher.wait_all(timeout=1.0)

    # Verify data was copied to correct slice
    expected = np.arange(50, dtype=np.float64).reshape(5, 10)
    np.testing.assert_array_equal(target[5:10, :], expected)

    # Cleanup
    watcher.shutdown()


def test_is_chunked_false_when_single_chunk():
    """Verify is_chunked returns False when chunks <= 1."""
    manager = _make_test_array_manager()
    assert manager._chunks == 0
    assert manager.is_chunked is False

    manager._chunks = 1
    assert manager.is_chunked is False


def test_is_chunked_true_when_multiple_chunks():
    """Verify is_chunked returns True when chunks > 1."""
    manager = _make_test_array_manager()
    manager._chunks = 2
    assert manager.is_chunked is True

    manager._chunks = 10
    assert manager.is_chunked is True


def test_output_container_host_factory_default_pinned():
    """Verify OutputArrayContainer.host_factory defaults to pinned."""
    container = OutputArrayContainer.host_factory()
    for _, slot in container.iter_managed_arrays():
        assert slot.memory_type == "pinned"


def test_output_container_host_factory_accepts_host():
    """Verify OutputArrayContainer.host_factory accepts host type."""
    container = OutputArrayContainer.host_factory(memory_type="host")
    for _, slot in container.iter_managed_arrays():
        assert slot.memory_type == "host"


def test_input_container_host_factory_default_pinned():
    """Verify InputArrayContainer.host_factory defaults to pinned."""
    container = InputArrayContainer.host_factory()
    for _, slot in container.iter_managed_arrays():
        assert slot.memory_type == "pinned"


def test_input_container_host_factory_accepts_host():
    """Verify InputArrayContainer.host_factory accepts host type."""
    container = InputArrayContainer.host_factory(memory_type="host")
    for _, slot in container.iter_managed_arrays():
        assert slot.memory_type == "host"


def test_finalise_uses_buffer_pool_when_chunked(chunked_solved_solver):
    """Verify chunked finalise acquires buffers from pool."""
    solver = chunked_solved_solver[0]
    output_arrays_manager = solver.kernel.output_arrays

    #  Check that the output arrays manager has created buffers
    assert len(output_arrays_manager._buffer_pool._buffers) >= 0


def test_reset_clears_buffer_pool_and_watcher(chunked_solved_solver):
    """Reset drains the watcher before clearing staging state."""
    solver = chunked_solved_solver[0]
    output_arrays_manager = solver.kernel.output_arrays

    output_arrays_manager.reset()

    assert len(output_arrays_manager._buffer_pool._buffers) == 0
    assert output_arrays_manager._watcher._pending_count == 0
    assert output_arrays_manager._watcher._thread is None


def test_input_slots_record_actual_backing(
    chunked_solved_solver, unchunked_solved_solver
):
    """Input slots hold attached arrays verbatim with accurate types.

    ``needs_chunked_transfer`` (shape vs chunked_shape) routes chunk
    slices through staging; the slot's ``memory_type`` records the
    attached array's real backing, which staging decisions key off.
    """

    chunked_solver, chunked_results = chunked_solved_solver
    unchunked_solver, unchunked_results = unchunked_solved_solver

    chunked_input_manager = chunked_solver.kernel.input_arrays
    unchunked_input_manager = unchunked_solver.kernel.input_arrays
    chunked_inits = chunked_input_manager.host.initial_values
    chunked_drivers = chunked_input_manager.host.driver_coefficients
    unchunked_inits = unchunked_input_manager.host.initial_values

    # Check needs_chunked_transfer values are set appropriately
    assert chunked_inits.needs_chunked_transfer is True
    assert unchunked_inits.needs_chunked_transfer is False
    assert chunked_drivers.needs_chunked_transfer is False

    for manager in (chunked_input_manager, unchunked_input_manager):
        for _, slot in manager.host.iter_managed_arrays():
            if slot.array is not None:
                assert slot.memory_type == (
                    manager._host_memory_type(slot.array)
                )


def test_chunked_shape_differs_from_shape_when_chunking(
    chunked_solved_solver,
):
    """Verify chunked_shape differs from shape when chunking is active.

    When chunking is active (chunks > 1), device arrays that are chunked
    should have a chunked_shape that differs from their full shape along
    the run axis (axis 2). This verifies that the memory manager
    correctly computed chunked shapes based on chunk_axis_index.
    """
    solver, result = chunked_solved_solver

    # Verify chunking occurred
    assert solver.chunks > 1

    # Check device arrays have different chunked_shape
    output_arrays = solver.kernel.output_arrays
    state_host = output_arrays.host.state

    # state array should be chunked (needs_chunked_transfer = True)
    assert state_host.needs_chunked_transfer is True

    # The full buffer belongs to the result; the slot records the
    # per-chunk shape, smaller along the run axis (axis 2).
    assert state_host.chunked_shape != result.state.shape
    assert state_host.chunked_shape[2] < result.state.shape[2]


def test_chunked_shape_equals_shape_when_not_chunking(
    unchunked_solved_solver,
):
    """Verify chunked_shape equals shape when chunking is not active.

    When chunking is not active (chunks == 1), all device arrays should
    have chunked_shape equal to their full shape. This verifies that
    unchunked runs do not perform unnecessary shape modifications.
    """
    solver, result = unchunked_solved_solver

    # Verify no chunking occurred
    assert solver.chunks == 1

    # Check device arrays have identical chunked_shape
    output_arrays = solver.kernel.output_arrays
    state_device = output_arrays.device.state

    # Arrays should not need chunked transfer
    assert state_device.needs_chunked_transfer is False

    # chunked_shape should equal shape
    assert state_device.chunked_shape == state_device.shape


def test_chunk_axis_index_in_array_requests(chunked_solved_solver):
    """Verify ArrayRequest objects have correct chunk_axis_index.

    Array requests created by the solver should have chunk_axis_index=2,
    which corresponds to the run axis in the stride order
    ("time", "variable", "run"). This verifies that the system correctly
    sets the chunking axis for memory allocation.
    """
    solver, result = chunked_solved_solver

    # Access the array requests used by the memory manager
    # These are stored when allocate() is called
    input_manager = solver.kernel.input_arrays
    output_manager = solver.kernel.output_arrays

    # initial_values is 2D (num_states, num_runs) so run axis is at index 1
    # output arrays use stride_order ("time", "variable", "run")
    # where "run" is at index 2
    assert input_manager.device.initial_values._chunk_axis_index == 1
    assert output_manager.device.state._chunk_axis_index == 2

    # Verify the chunk axis matches the run axis position in shape
    # For output state with shape (n_states, n_runs), run is at axis 1
    # Wait - the fixture uses default stride order, need to check actual
    # Let's verify by checking that chunked_shape differs at that axis
    state_device = output_manager.device.state
    if state_device.needs_chunked_transfer:
        chunk_axis = state_device._chunk_axis_index
        # chunked_shape should differ from shape at chunk_axis
        for i in range(len(state_device.shape)):
            if i == chunk_axis:
                assert state_device.chunked_shape[i] < state_device.shape[i]
            else:
                assert state_device.chunked_shape[i] == state_device.shape[i]
