"""Solver resource cleanup tests."""

import gc
from pathlib import Path

import numpy as np
import pytest

from cubie.batchsolving.solver import Solver, solve_ivp
from cubie.batchsolving.solveresult import SolveResult
from cubie.cuda_simsafe import cuda, CUDA_SIMULATION
from cubie.memory.mem_manager import MemoryManager
from tests._utils import _build_solver_instance


if not CUDA_SIMULATION:
    # The canary work spins until the host releases it, so the
    # not-yet-done assertions cannot flake under GPU contention:
    # completion is gated on the host, not on outlasting close() by
    # wall clock. The iteration cap (roughly half a minute of atomic
    # polling) bounds a regression in which close() synchronizes the
    # whole device — the test then fails its canary assertion instead
    # of deadlocking against a kernel the host can never release.
    @cuda.jit
    def _spin_until_released(flag, out):
        spins = 0.0
        while cuda.atomic.add(flag, 0, 0) == 0 and spins < 1.0e9:
            spins += 1.0
        out[0] = spins


    def _start_cuda_work():
        stream = cuda.stream()
        flag = cuda.to_device(np.zeros(1, dtype=np.int32))
        out = cuda.device_array(1, dtype=np.float32)
        done = cuda.event()
        _spin_until_released[1, 1, stream](flag, out)
        done.record(stream)
        # Both device arrays stay referenced until the spin exits:
        # dropping one mid-flight queues a free that blocks on the
        # resident kernel, and the kernel's exit write would land in
        # reallocated memory.
        return (flag, out), stream, done


    def _finish_cuda_work(work, stream, done):
        # Release by async copy: a concurrent release kernel never
        # reaches the device while the spin kernel is resident, but a
        # copy-engine write to the polled flag lands immediately.
        flag, out = work
        release_stream = cuda.stream()
        flag.copy_to_device(
            np.ones(1, dtype=np.int32), stream=release_stream
        )
        stream.synchronize()
        assert done.query()


def _instance_ids(solver):
    """Return the ids of the three memory-manager clients of a solver."""
    kernel = solver.kernel
    return (
        id(kernel),
        id(kernel.input_arrays),
        id(kernel.output_arrays),
    )


def _still_registered(manager, ids):
    """Return the subset of ``ids`` still present in the registry."""
    return [instance_id for instance_id in ids if instance_id in
            manager.registry]


def _registered_bytes(manager, ids):
    """Total device bytes the registry keeps alive for ``ids``."""
    total = 0
    for instance_id in ids:
        settings = manager.registry.get(instance_id)
        if settings is not None:
            total += settings.allocated_bytes
    return total


def test_solver_releases_registry_on_gc(
    system, batch_input_arrays, thread_mem_manager
):
    """Collection defers teardown to the manager's next entry point.

    GC finalizers only record the teardown — running it inside the
    collection could mutate the registry while the manager iterates
    it — so the entries survive gc.collect() and disappear once any
    manager entry point drains the recorded teardowns.
    """
    manager = thread_mem_manager
    solver = Solver(system, algorithm="euler", dt=0.01, memory_manager=manager)
    y0, params = batch_input_arrays
    solver.solve(y0, params, duration=0.1)

    ids = _instance_ids(solver)
    assert _still_registered(manager, ids) == list(ids)
    assert _registered_bytes(manager, ids) > 0

    del solver
    gc.collect()

    assert len(manager._pending_teardowns) > 0
    manager._purge_dead_instances()

    assert _still_registered(manager, ids) == []
    assert manager._pending_teardowns == []


def test_close_releases_registry_immediately(
    solver_mutable, batch_input_arrays, driver_settings, thread_mem_manager
):
    """``close`` deregisters without waiting for garbage collection."""
    manager = thread_mem_manager
    solver = solver_mutable
    y0, params = batch_input_arrays
    solver.solve(y0, params, drivers=driver_settings, duration=0.1)

    ids = _instance_ids(solver)
    assert _still_registered(manager, ids) == list(ids)

    solver.close()

    assert _still_registered(manager, ids) == []
    solver.close()
    assert _still_registered(manager, ids) == []


def test_closed_solver_raises_on_solve(
    solver_mutable, batch_input_arrays, driver_settings
):
    """A closed solver rejects another solve."""
    solver = solver_mutable
    y0, params = batch_input_arrays
    solver.solve(y0, params, drivers=driver_settings, duration=0.1)
    solver.close()

    with pytest.raises(RuntimeError, match="closed"):
        solver.solve(y0, params, duration=0.1)


def test_context_manager_releases_on_exit(
    solver_mutable, batch_input_arrays, driver_settings, thread_mem_manager
):
    """Context exit releases the solver."""
    manager = thread_mem_manager
    y0, params = batch_input_arrays
    with solver_mutable as solver:
        solver.solve(y0, params, drivers=driver_settings, duration=0.1)
        ids = _instance_ids(solver)
        assert _still_registered(manager, ids) == list(ids)

    assert _still_registered(manager, ids) == []


@pytest.mark.nocudasim
def test_close_timeout_is_retryable(
    solver_mutable, batch_input_arrays, driver_settings, thread_mem_manager
):
    """A timed-out close can be retried."""
    manager = thread_mem_manager
    solver = solver_mutable
    y0, params = batch_input_arrays
    solver.solve(y0, params, drivers=driver_settings, duration=0.1)
    input_arrays = solver.kernel.input_arrays
    instance_id = id(input_arrays)
    settings = manager.registry[instance_id]
    group = manager.get_stream_group(input_arrays)
    work_output, work_stream, work_done = _start_cuda_work()
    buffer = input_arrays._buffer_pool.acquire(
        "close_gate", (1,), np.dtype(np.float32)
    )
    input_arrays._transfer_watcher.submit_release(
        work_done,
        buffer,
        input_arrays._buffer_pool,
        "close_gate",
    )

    try:
        with pytest.raises(TimeoutError, match="wait_all timed out"):
            solver.close(shutdown_timeout=0.0)

        assert manager.registry[instance_id] is settings
        assert instance_id in manager._auto_pool
        assert instance_id in manager.stream_groups.get_instances_in_group(
            group
        )
    finally:
        _finish_cuda_work(work_output, work_stream, work_done)

    solver.close()
    assert _still_registered(manager, _instance_ids(solver)) == []


def test_solve_ivp_releases_temporary_solver(
    system, batch_input_arrays, thread_mem_manager
):
    """solve_ivp releases its temporary solver."""
    manager = thread_mem_manager
    # Reclaim earlier tests' dead registrants first: the baseline
    # must not contain entries whose deferred teardown would drain
    # during solve_ivp's own manager calls.
    gc.collect()
    manager._purge_dead_instances()
    baseline = set(manager.registry)
    y0, params = batch_input_arrays

    solve_ivp(
        system,
        y0,
        params,
        duration=0.1,
        grid_type="verbatim",
        dt=0.01,
        memory_manager=manager,
    )

    assert set(manager.registry) == baseline


def test_solve_ivp_spill_survives_solver_close(
    system, batch_input_arrays, tmp_path
):
    """Spilled results remain readable after temporary solver cleanup."""
    y0, params = batch_input_arrays
    result = solve_ivp(
        system,
        y0,
        params,
        duration=0.1,
        grid_type="verbatim",
        dt=0.01,
        host_spill_threshold=1,
        spill_directory=tmp_path,
    )
    assert isinstance(result, SolveResult)
    spill_paths = [
        Path(array._cubie_spill_path)
        for array in (result.state, result.status_codes)
        if isinstance(array, np.memmap)
    ]
    try:
        assert spill_paths
        assert all(path.exists() for path in spill_paths)
        assert np.isfinite(
            np.array(result.time_domain_array, copy=True)
        ).all()
    finally:
        result.close()
    assert all(not path.exists() for path in spill_paths)


@pytest.mark.nocudasim
def test_close_does_not_wait_for_unrelated_stream(
    solver_mutable,
    batch_input_arrays,
    driver_settings,
    solver_settings,
    system,
    thread_mem_manager,
):
    """Close waits only for the solver's own stream.

    Every run launches on the kernel's memory-manager stream; close
    must synchronize that stream alone, leaving work on unrelated
    streams running.
    """
    manager = thread_mem_manager
    target_solver = solver_mutable
    y0, params = batch_input_arrays
    ids = _instance_ids(target_solver)
    assert _registered_bytes(manager, ids) == 0

    target_solver.kernel.run(
        y0,
        params,
        target_solver.driver_interpolator.coefficients,
        duration=0.1,
    )
    closed_state_view = target_solver.kernel.state
    assert _registered_bytes(manager, ids) > 0
    work_output, unrelated_stream, unrelated_done = _start_cuda_work()
    try:
        # Numba's deferred-deallocation queue may hold driver frees
        # from earlier tests whose execution synchronizes the whole
        # device. Defer them so the canary assertion sees only the
        # waits close() itself performs.
        with cuda.defer_cleanup():
            target_solver.close()

            assert _still_registered(manager, ids) == []
            closed_state = closed_state_view.copy()
            assert not unrelated_done.query()
    finally:
        _finish_cuda_work(work_output, unrelated_stream, unrelated_done)

    reference_settings = solver_settings.copy()
    reference_settings["stream_group"] = "close_reference"
    reference_solver = _build_solver_instance(
        system=system,
        solver_settings=reference_settings,
        driver_settings=driver_settings,
        memory_manager=MemoryManager(),
    )
    try:
        reference_solver.kernel.run(
            y0,
            params,
            reference_solver.driver_interpolator.coefficients,
            duration=0.1,
        )
        reference_solver.kernel.synchronize()
        reference_solver.kernel.wait_for_writeback()
        expected_state = reference_solver.kernel.state.copy()
    finally:
        reference_solver.close()
    np.testing.assert_array_equal(closed_state, expected_state)


def test_repeated_solvers_do_not_grow_registry(
    system, batch_input_arrays, thread_mem_manager
):
    """Repeated solvers do not grow the registry."""
    manager = thread_mem_manager
    y0, params = batch_input_arrays

    gc.collect()
    baseline = len(manager.registry)

    # Three iterations: enough to reach and repeat steady-state
    # reclamation.
    for _ in range(3):
        solver = Solver(
            system, algorithm="euler", dt=0.01, memory_manager=manager
        )
        solver.solve(y0, params, duration=0.1)
        del solver
        gc.collect()

    # Each Solver construction is a manager entry point, so every
    # iteration reclaims its predecessor's deferred teardown; one
    # final drain covers the last solver.
    manager._purge_dead_instances()
    assert len(manager.registry) <= baseline
