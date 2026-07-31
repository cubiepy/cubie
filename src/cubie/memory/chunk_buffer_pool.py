"""Reusable pinned buffer pool for chunked array transfers.

This module manages allocation and lifecycle of pinned memory buffers
used for staging data during chunked host-device transfers. Buffers
are sized for one transfer block and reused across blocks and chunks
to avoid repeated allocation overhead.

Pool depth is bounded by RAM headroom rather than a fixed count: the
pool grows while available physical RAM stays above the reserve
fraction, so a spilled solve can pre-stage a whole chunk while the
kernel runs, and a RAM-resident solve (whose pageable result arrays
already occupy that headroom) stays shallow. When the pool cannot
grow, :meth:`ChunkBufferPool.acquire` blocks until an in-flight
buffer is released by the transfer watcher.

Published Classes
-----------------
:class:`PinnedBuffer`
    Wrapper for a reusable pinned memory buffer.

    >>> from numpy import zeros, float64
    >>> buf = PinnedBuffer(buffer_id=0, array=zeros((10,), float64))
    >>> buf.in_use
    False

:class:`ChunkBufferPool`
    Pool of reusable pinned buffers for chunked transfers.

    >>> pool = ChunkBufferPool()
    >>> buf = pool.acquire("state", (100, 4), float64)
    >>> pool.release(buf)

See Also
--------
:class:`~cubie.memory.mem_manager.MemoryManager`
    Coordinates chunked allocations that consume these buffers.
:class:`~cubie.batchsolving.arrays.BatchInputArrays.InputArrays`
    Primary consumer for host-to-device staging.
:class:`~cubie.batchsolving.arrays.BatchOutputArrays.OutputArrays`
    Primary consumer for device-to-host staging.
"""

from math import prod
from typing import Dict, List, Tuple
from threading import Condition

from attrs import define, field
from attrs.validators import instance_of as attrsval_instance_of
from numpy import ndarray, zeros as np_zeros
from numpy import dtype as np_dtype

from cubie.cuda_simsafe import CUDA_SIMULATION, cupyx
from cubie.memory.mem_manager import host_headroom_bytes


@define
class PinnedBuffer:
    """Wrapper for a reusable pinned memory buffer.

    Attributes
    ----------
    buffer_id : int
        Unique identifier for this buffer.
    array : ndarray
        The pinned numpy array.
    in_use : bool
        Whether the buffer is currently in use.
    """

    buffer_id: int = field(validator=attrsval_instance_of(int))
    array: ndarray = field(validator=attrsval_instance_of(ndarray))
    in_use: bool = field(default=False, validator=attrsval_instance_of(bool))


@define
class ChunkBufferPool:
    """Pool of reusable pinned buffers for chunked transfers.

    Manages allocation and lifecycle of pinned memory buffers used
    for staging data during chunked device transfers. Buffers are
    sized for one transfer block and reused across blocks and chunks.

    Attributes
    ----------
    _buffers : Dict[str, List[PinnedBuffer]]
        Pool of buffers organized by array name.
    _condition : Condition
        Guards the pool and wakes blocked acquirers on release.
    _next_id : int
        Counter for unique buffer IDs.
    """

    _buffers: Dict[str, List[PinnedBuffer]] = field(factory=dict)
    _condition: Condition = field(factory=Condition)
    _next_id: int = field(default=0)

    def acquire(
        self,
        array_name: str,
        shape: Tuple[int, ...],
        dtype: np_dtype,
    ) -> PinnedBuffer:
        """Acquire a pinned buffer for the given array.

        Reuses a free matching buffer when one exists, grows the pool
        while RAM headroom allows, and otherwise blocks until the
        transfer watcher releases an in-flight buffer. Blocking is the
        pipeline's natural pacing: the CPU runs ahead of the GPU by
        exactly the depth the machine's RAM can hold.

        Parameters
        ----------
        array_name
            Identifier for the array type (e.g., 'state', 'observables').
        shape
            Required shape for the buffer.
        dtype
            Data type for the buffer elements.

        Returns
        -------
        PinnedBuffer
            A buffer ready for use, either reused or newly allocated.
        """
        with self._condition:
            while True:
                matching_in_flight = False
                for buf in self._buffers.get(array_name, []):
                    if (buf.array.shape == shape
                            and buf.array.dtype == dtype):
                        if not buf.in_use:
                            buf.in_use = True
                            return buf
                        matching_in_flight = True

                # Grow the pool unless RAM headroom is exhausted; a
                # label with nothing in flight must always get one
                # buffer, or no release could ever unblock it.
                if not matching_in_flight or self._headroom_allows(
                    shape, dtype
                ):
                    new_buffer = self._allocate_buffer(shape, dtype)
                    new_buffer.in_use = True
                    self._buffers.setdefault(array_name, []).append(
                        new_buffer
                    )
                    return new_buffer

                self._condition.wait()

    @staticmethod
    def _headroom_allows(shape: Tuple[int, ...], dtype: np_dtype) -> bool:
        """Return whether RAM headroom permits one more buffer.

        The pool may grow while the new buffer fits the available RAM
        above the operating-system reserve. When RAM cannot be probed
        the pool grows freely.
        """
        if CUDA_SIMULATION:  # pragma: no cover - simulated
            return True
        headroom = host_headroom_bytes()
        if headroom is None:
            return True
        nbytes = int(prod(shape)) * np_dtype(dtype).itemsize
        return nbytes < headroom

    def release(self, buffer: PinnedBuffer) -> None:
        """Release a buffer back to the pool.

        Parameters
        ----------
        buffer
            The buffer to release.
        """
        with self._condition:
            buffer.in_use = False
            self._condition.notify_all()

    def clear(self) -> None:
        """Clear all buffers from the pool.

        Should be called on cleanup or error to free pinned memory.
        Wakes any blocked acquirer so it re-evaluates against the
        emptied pool.
        """
        with self._condition:
            self._buffers.clear()
            self._next_id = 0
            self._condition.notify_all()

    def _allocate_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: np_dtype,
    ) -> PinnedBuffer:
        """Allocate a new pinned buffer.

        Parameters
        ----------
        shape
            Shape for the buffer.
        dtype
            Data type for the buffer elements.

        Returns
        -------
        PinnedBuffer
            Newly allocated pinned buffer.
        """
        # Use np.zeros for CUDASIM mode; CuPy's pinned memory pool
        # otherwise, since CuPy is the single device allocation
        # provider on a real GPU.
        if CUDA_SIMULATION:  # pragma: no cover - simulated
            arr = np_zeros(shape, dtype=dtype)
        else:
            arr = cupyx.empty_pinned(shape, dtype=dtype)
            arr.fill(0)

        buffer_id = self._next_id
        self._next_id += 1

        return PinnedBuffer(buffer_id=buffer_id, array=arr)
