"""Tests for cubie.memory.chunk_buffer_pool.

Pools are built on the ``mgr`` fixture for a fixed pinned budget.
"""

from __future__ import annotations

import threading

import numpy as np

from cubie.memory.chunk_buffer_pool import ChunkBufferPool, PinnedBuffer
from cubie.memory.mem_manager import STAGING_POOL_DEPTH
from cubie.memory.pinned_arena import MIN_SLAB_BYTES, PINNED_ALIGNMENT_BYTES


# ── PinnedBuffer ──────────────────────────────────────────────── #

def test_pinned_buffer_construction():
    """PinnedBuffer stores its bytes and starts idle."""
    storage = np.zeros((800,), dtype=np.uint8)
    buf = PinnedBuffer(buffer_id=0, storage=storage)
    assert buf.buffer_id == 0
    assert buf.array is storage
    assert buf.capacity == storage.nbytes
    assert buf.in_use is False


def test_pinned_buffer_in_use_override():
    """in_use can be set to True at construction."""
    storage = np.zeros((40,), dtype=np.uint8)
    buf = PinnedBuffer(buffer_id=1, storage=storage, in_use=True)
    assert buf.in_use is True


def test_pinned_buffer_view_shares_storage():
    """view points array at the leading bytes in the given shape."""
    storage = np.zeros((800,), dtype=np.uint8)
    buf = PinnedBuffer(buffer_id=0, storage=storage)
    buf.view((10, 5), np.float32)
    assert buf.array.shape == (10, 5)
    assert buf.array.dtype == np.float32
    buf.array[:] = 1.0
    assert (storage[:200].view(np.float32) == 1.0).all()


# ── acquire ───────────────────────────────────────────────────── #

def test_acquire_returns_buffer_with_correct_shape_dtype(mgr):
    """acquire returns a PinnedBuffer matching requested shape and dtype."""
    pool = ChunkBufferPool(memory_manager=mgr)
    buf = pool.acquire("state", (100, 4), np.float32)
    assert buf.array.shape == (100, 4)
    assert buf.array.dtype == np.float32
    assert buf.in_use is True


def test_acquire_reuses_released_buffer(mgr):
    """acquire reuses a released buffer with matching shape/dtype."""
    pool = ChunkBufferPool(memory_manager=mgr)
    buf1 = pool.acquire("state", (50, 3), np.float32)
    bid = buf1.buffer_id
    pool.release(buf1)
    buf2 = pool.acquire("state", (50, 3), np.float32)
    assert buf2.buffer_id == bid
    assert buf2 is buf1


def test_acquire_allocates_new_when_all_in_use(mgr):
    """acquire grows the pool when in-use buffers block reuse."""
    pool = ChunkBufferPool(memory_manager=mgr)
    buf1 = pool.acquire("x", (10,), np.float32)
    buf2 = pool.acquire("x", (10,), np.float32)
    assert buf1.buffer_id != buf2.buffer_id


def test_acquire_reuses_a_buffer_for_a_smaller_shape(mgr):
    """An idle buffer serves any shape and dtype that fits it."""
    pool = ChunkBufferPool(memory_manager=mgr)
    buf1 = pool.acquire("x", (20,), np.float64)
    pool.release(buf1)
    buf2 = pool.acquire("x", (4, 5), np.float32)
    assert buf2 is buf1
    assert buf2.array.shape == (4, 5)
    assert buf2.array.dtype == np.float32


def test_acquire_replaces_an_idle_buffer_too_small(mgr):
    """A request larger than the idle buffer replaces it."""
    pool = ChunkBufferPool(memory_manager=mgr)
    buf1 = pool.acquire("x", (10,), np.float32)
    pool.release(buf1)
    buf2 = pool.acquire("x", (20,), np.float64)
    assert buf2.buffer_id != buf1.buffer_id
    assert pool._buffers["x"] == [buf2]


def test_acquire_creates_new_array_name_entry(mgr):
    """First acquire for a name creates a new entry in _buffers."""
    pool = ChunkBufferPool(memory_manager=mgr)
    pool.acquire("new_name", (5,), np.float32)
    assert "new_name" in pool._buffers
    assert len(pool._buffers["new_name"]) == 1


# ── release ───────────────────────────────────────────────────── #

def test_release_marks_not_in_use(mgr):
    """release sets buffer.in_use to False."""
    pool = ChunkBufferPool(memory_manager=mgr)
    buf = pool.acquire("x", (10,), np.float32)
    assert buf.in_use is True
    pool.release(buf)
    assert buf.in_use is False


# ── clear ─────────────────────────────────────────────────────── #

def test_clear_empties_pool_and_resets_id(mgr):
    """clear removes all buffers and resets _next_id to 0."""
    pool = ChunkBufferPool(memory_manager=mgr)
    pool.acquire("a", (10,), np.float32)
    pool.acquire("b", (20,), np.float64)
    pool.clear()
    assert len(pool._buffers) == 0
    assert pool._next_id == 0


# ── _allocate_buffer ──────────────────────────────────────────── #

def test_allocate_buffer_increments_id(mgr):
    """Each allocated buffer gets an incrementing buffer_id."""
    pool = ChunkBufferPool(memory_manager=mgr)
    ids = []
    for i in range(5):
        buf = pool.acquire(f"arr_{i}", (3,), np.float32)
        ids.append(buf.buffer_id)
    assert ids == [0, 1, 2, 3, 4]


# ── Thread safety ─────────────────────────────────────────────── #

def test_thread_safe_concurrent_acquire_release(mgr):
    """Concurrent acquire/release does not raise or corrupt state."""
    pool = ChunkBufferPool(memory_manager=mgr)
    errors = []

    def worker(tid):
        try:
            for _ in range(20):
                buf = pool.acquire(f"arr_{tid}", (10,), np.float32)
                pool.release(buf)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 0


# ── budget-bounded growth ─────────────────────────────────────── #

def _assert_second_acquire_waits_for_release(pool, shape, dtype):
    """Check a second matching acquire waits instead of growing."""
    first = pool.acquire("state", shape, dtype)
    acquired = []
    started = threading.Event()

    def worker():
        started.set()
        acquired.append(pool.acquire("state", shape, dtype))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started.wait(timeout=2.0)
    thread.join(timeout=0.2)
    assert thread.is_alive()
    assert acquired == []

    pool.release(first)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert acquired[0] is first


def test_acquire_grows_first_buffer_past_the_budget(mgr):
    """A label with nothing in flight always gets one buffer."""
    mgr.pinned_max_bytes = 0
    pool = ChunkBufferPool(memory_manager=mgr)
    buf = pool.acquire("state", (10,), np.float32)
    assert buf.in_use is True


def test_acquire_blocks_until_release_when_the_budget_refuses(mgr):
    """Budget with no room for a second buffer: acquire waits."""
    mgr.pinned_max_bytes = 0
    # Two of these cannot share one slab.
    shape = (MIN_SLAB_BYTES // 2 + PINNED_ALIGNMENT_BYTES,)
    _assert_second_acquire_waits_for_release(
        ChunkBufferPool(memory_manager=mgr), shape, np.uint8
    )


# ── Depth cap ─────────────────────────────────────────────────── #

def test_acquire_blocks_at_the_depth_cap(mgr):
    """At the depth cap, acquire waits for a release, not growth."""
    pool = ChunkBufferPool(memory_manager=mgr)
    held = [
        pool.acquire("state", (10,), np.float32)
        for _ in range(STAGING_POOL_DEPTH)
    ]
    assert len({buf.buffer_id for buf in held}) == STAGING_POOL_DEPTH

    acquired = []
    started = threading.Event()

    def worker():
        started.set()
        acquired.append(pool.acquire("state", (10,), np.float32))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started.wait(timeout=2.0)
    thread.join(timeout=0.2)
    assert thread.is_alive()
    assert acquired == []

    pool.release(held[3])
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert acquired[0] is held[3]
    assert len(pool._buffers["state"]) == STAGING_POOL_DEPTH
