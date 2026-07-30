"""Tests for cubie.batchsolving.writeback_watcher."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.batchsolving.writeback_watcher import (
    WritebackTask,
    WritebackWatcher,
)
from cubie.cuda_simsafe import cuda

from cubie.cuda_simsafe import CUDA_SIMULATION
from cubie.memory.chunk_buffer_pool import ChunkBufferPool, PinnedBuffer


# ── Helpers ─────────────────────────────────────────────────── #


if not CUDA_SIMULATION:
    @cuda.jit
    def _busy_kernel(out):
        """Single-thread kernel that runs long enough to stay pending."""
        x = 0.0
        for _ in range(20_000_000):
            x += 1.0
        out[0] = x


def _record_busy_event():
    """Launch the busy kernel on a fresh stream and record an event.

    The kernel keeps the stream busy for tens of milliseconds, so an
    event recorded behind it reads as still-pending on an immediate
    query.
    """
    stream = cuda.stream()
    event = cuda.event()
    out = cuda.device_array(1, dtype=np.float32)
    _busy_kernel[1, 1, stream](out)
    event.record(stream)
    return stream, event


def _make_pool():
    """Return a fresh ChunkBufferPool."""
    return ChunkBufferPool()


def _make_pinned_buffer(shape=(4, 3), dtype=np.float32, fill=1.0):
    """Return a PinnedBuffer with known data."""
    arr = np.full(shape, fill, dtype=dtype)
    return PinnedBuffer(buffer_id=0, array=arr)


# ── WritebackTask attrs dataclass (item 2) ──────────────────── #


def test_writeback_task_stores_all_fields():
    """WritebackTask stores event, buffer, target_array, buffer_pool, array_name, data_shape."""
    buf = _make_pinned_buffer()
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    event = object()
    task = WritebackTask(
        event=event,
        buffer=buf,
        target_array=target,
        buffer_pool=pool,
        array_name="state",
        data_shape=(2, 3),
    )
    assert task.event is event
    assert task.buffer is buf
    assert task.target_array is target
    assert task.buffer_pool is pool
    assert task.array_name == "state"
    assert task.data_shape == (2, 3)


def test_writeback_task_data_shape_defaults_none():
    """WritebackTask data_shape defaults to None."""
    buf = _make_pinned_buffer()
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None,
        buffer=buf,
        target_array=target,
        buffer_pool=pool,
        array_name="state",
    )
    assert task.data_shape is None


# ── WritebackWatcher.__init__ (item 4) ───────────────────────── #


def test_watcher_init_defaults():
    """Watcher initializes with expected defaults."""
    w = WritebackWatcher()
    assert w._poll_interval == 0.001
    assert w._pending_count == 0
    assert w._thread is None
    assert not w._stop_event.is_set()
    assert not w._tasks


def test_watcher_init_custom_poll_interval():
    """Watcher accepts custom poll_interval."""
    w = WritebackWatcher(poll_interval=0.05)
    assert w._poll_interval == 0.05


# ── start (items 5, 6) ───────────────────────────────────────── #


def test_start_creates_daemon_thread():
    """start() creates and starts a daemon polling thread."""
    w = WritebackWatcher()
    w.start()
    assert w._thread is not None
    assert w._thread.is_alive()
    assert w._thread.daemon is True
    w.shutdown()


def test_start_noop_when_already_running():
    """start() is no-op when thread already alive."""
    w = WritebackWatcher()
    w.start()
    first_thread = w._thread
    w.start()
    assert w._thread is first_thread
    w.shutdown()


# ── _submit_task (items 7, 8, 9) ─────────────────────────────── #


def test_submit_task_starts_thread_and_processes():
    """_submit_task enqueues task, starts thread, task gets processed."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=99.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    w._submit_task(task)
    # Thread was started (item 9)
    assert w._thread is not None
    assert w._thread.is_alive()
    w.wait_all(timeout=2.0)
    # Task was processed: pending_count back to 0, data copied (items 7, 8)
    assert w._pending_count == 0
    np.testing.assert_array_equal(target, 99.0)
    w.shutdown()


# ── submit (item 11) ─────────────────────────────────────────── #


def test_submit_with_individual_args():
    """submit() creates WritebackTask from individual args and submits."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=3.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    w.submit(
        event=None,
        buffer=buf,
        target_array=target,
        buffer_pool=pool,
        array_name="state",
    )
    w.wait_all(timeout=2.0)
    np.testing.assert_array_equal(target, buf.array)
    w.shutdown()


# ── wait_all (items 12, 13, 14) ──────────────────────────────── #


def test_wait_all_returns_immediately_when_no_pending():
    """wait_all returns immediately when pending_count == 0."""
    w = WritebackWatcher()
    # Should not block or raise
    w.wait_all(timeout=0.1)


def test_wait_all_polls_until_complete():
    """wait_all polls until pending_count == 0."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=5.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    w.submit(event=None, buffer=buf, target_array=target,
             buffer_pool=pool, array_name="s")
    w.wait_all(timeout=2.0)
    assert w._pending_count == 0
    np.testing.assert_array_equal(target, buf.array)
    w.shutdown()


def test_wait_all_raises_timeout_error():
    """wait_all raises TimeoutError when timeout expires."""
    w = WritebackWatcher()
    # Artificially set pending count without actual task
    w._pending_count = 1
    with pytest.raises(TimeoutError, match="wait_all timed out"):
        w.wait_all(timeout=0.05)


def test_wait_all_drains_inline_without_thread():
    """wait_all completes queued tasks in the calling thread."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=7.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    # Enqueue without submit() so no background thread starts.
    w._tasks.append(task)
    w._pending_count = 1
    w.wait_all(timeout=1.0)
    assert w._pending_count == 0
    assert w._thread is None
    np.testing.assert_array_equal(target, 7.0)


# ── shutdown (items 15, 16, 17) ───────────────────────────────── #


def test_shutdown_sets_stop_event_and_clears_thread():
    """shutdown() sets stop_event, joins thread, sets _thread = None."""
    w = WritebackWatcher()
    w.start()
    assert w._thread is not None
    w.shutdown()
    assert w._stop_event.is_set()
    assert w._thread is None


# ── _poll_loop / _process_task integration (items 18-26) ──────── #


def test_process_task_copies_full_buffer_when_no_data_shape():
    """_process_task copies full buffer when data_shape is None (item 24)."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(shape=(3, 2), fill=9.0)
    target = np.zeros((3, 2), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state", data_shape=None,
    )
    result = w._process_task(task)
    assert result is True
    np.testing.assert_array_equal(target, 9.0)


def test_process_task_copies_sliced_buffer_when_data_shape_provided():
    """_process_task copies buffer[0:s] when data_shape provided (item 23)."""
    w = WritebackWatcher()
    # Buffer is 4x3 but data_shape is (2, 3) — only first 2 rows
    buf = _make_pinned_buffer(shape=(4, 3), fill=0.0)
    buf.array[:] = np.arange(12, dtype=np.float32).reshape(4, 3)
    target = np.zeros((2, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state", data_shape=(2, 3),
    )
    result = w._process_task(task)
    assert result is True
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(target, expected)


def test_process_task_releases_buffer_to_pool():
    """_process_task releases buffer back to pool (item 25)."""
    w = WritebackWatcher()
    pool = _make_pool()
    buf = pool.acquire("state", (3,), np.float32)
    assert buf.in_use is True
    target = np.zeros((3,), dtype=np.float32)
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    w._process_task(task)
    assert buf.in_use is False


def test_process_task_cudasim_immediate_complete():
    """_process_task treats as immediately complete in CUDA_SIMULATION ."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=42.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event="not_a_real_event",  # Not None, not a cuda.Event
        buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    # Under CUDASIM this completes immediately regardless of event type
    if CUDA_SIMULATION:
        assert w._process_task(task) is True
        np.testing.assert_array_equal(target, 42.0)
    else:
        try:
            task_completion = w._process_task(task)
            assert task_completion is True
        except AttributeError as e:
            # Check that our event handling hasn't allowed a vapid True on invalid watch tasks.
            assert "object has no attribute" in str(e)


def test_process_task_none_event_immediate_complete():
    """_process_task treats as immediately complete when event is None (item 21)."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=11.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    assert w._process_task(task) is True
    np.testing.assert_array_equal(target, 11.0)


def test_shutdown_drains_and_completes_remaining_tasks():
    """On shutdown, poll_loop drains queue and completes remaining tasks (item 20)."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=77.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    w.submit(event=None, buffer=buf, target_array=target,
             buffer_pool=pool, array_name="state")
    w.shutdown()
    # After shutdown, data should be copied
    np.testing.assert_array_equal(target, 77.0)
    assert w._thread is None


@pytest.mark.nocudasim
def test_process_task_pending_event_returns_false_until_recorded():
    """A real, still-running CUDA event causes _process_task to report
    not-yet-complete (returns False) without copying data, then True
    once the stream is synchronized."""
    stream, event = _record_busy_event()

    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=55.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=event, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )

    result = w._process_task(task)
    assert result is False
    np.testing.assert_array_equal(target, 0.0)

    stream.synchronize()
    result2 = w._process_task(task)
    assert result2 is True
    np.testing.assert_array_equal(target, 55.0)


@pytest.mark.nocudasim
def test_poll_loop_requeues_still_pending_task_in_main_loop():
    """A task whose event has not completed is re-queued as
    still_pending within the live-polling loop, then completes on a
    later iteration once the stream finishes (main-loop still_pending
    branch)."""
    stream, event = _record_busy_event()

    w = WritebackWatcher(poll_interval=0.001)
    buf = _make_pinned_buffer(fill=33.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    w.submit(
        event=event, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    # Watcher thread starts polling immediately; the task is very
    # likely still pending on submission given the busy kernel above.
    w.wait_all(timeout=10.0)
    np.testing.assert_array_equal(target, 33.0)
    w.shutdown()


@pytest.mark.nocudasim
def test_poll_loop_drain_requeues_still_pending_task_on_shutdown():
    """A task whose event has not completed is re-queued as
    still_pending within the post-shutdown drain loop and completes
    once the event is ready (drain-loop still_pending branch)."""
    stream, event = _record_busy_event()

    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=44.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=event, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    w._tasks.append(task)
    w._pending_count = 1
    w._stop_event.set()
    # The drain loop retries until the busy kernel's event completes.
    w._poll_loop()
    np.testing.assert_array_equal(target, 44.0)


@pytest.mark.nocudasim
def test_shutdown_timeout_keeps_live_thread_handle():
    """A timed-out shutdown keeps the draining thread reachable."""
    _, event = _record_busy_event()
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=18.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    w.submit(event, buf, target, pool, "state")

    with pytest.raises(TimeoutError, match="did not stop"):
        w.shutdown(timeout=0.0)

    assert w._thread is not None
    assert w._thread.is_alive()
    w.shutdown()
    np.testing.assert_array_equal(target, 18.0)


def test_poll_loop_drains_queue_directly_on_shutdown():
    """When _poll_loop is invoked with stop_event already set, it skips
    the live-polling loop and drains any queued tasks synchronously
    (the post-shutdown drain branch), rather than requiring a race
    between submit() and shutdown()."""
    w = WritebackWatcher()
    buf = _make_pinned_buffer(fill=21.0)
    target = np.zeros((4, 3), dtype=np.float32)
    pool = _make_pool()
    task = WritebackTask(
        event=None, buffer=buf, target_array=target,
        buffer_pool=pool, array_name="state",
    )
    w._tasks.append(task)
    w._pending_count = 1
    w._stop_event.set()
    # No thread involved: call the loop body directly and synchronously.
    w._poll_loop()
    np.testing.assert_array_equal(target, 21.0)
    assert not w._tasks


def test_multiple_tasks_all_complete():
    """Multiple submitted tasks all complete correctly (items 18, 19)."""
    w = WritebackWatcher()
    pool = _make_pool()
    targets = []
    expected_values = [1.0, 2.0, 3.0]
    for val in expected_values:
        buf = _make_pinned_buffer(shape=(2,), fill=val)
        target = np.zeros((2,), dtype=np.float32)
        targets.append((target, val))
        w.submit(event=None, buffer=buf, target_array=target,
                 buffer_pool=pool, array_name="arr")
    w.wait_all(timeout=5.0)
    for target, val in targets:
        np.testing.assert_array_equal(target, val)
    w.shutdown()
