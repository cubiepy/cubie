"""Reusable pinned buffer pool for chunked array transfers.

This module manages allocation and lifecycle of pinned memory buffers
used for staging data during chunked host-device transfers. Buffers
are sized for one transfer block and reused across blocks and chunks
to avoid repeated allocation overhead.

An idle buffer serves any block that fits. :meth:`acquire` waits
for a release once ``STAGING_POOL_DEPTH`` buffers per label are in
use or pinned memory runs out; a label's first buffer always
allocates.

Published Classes
-----------------
:class:`PinnedBuffer`
    Wrapper for a reusable pinned memory buffer.

    >>> from numpy import zeros, uint8
    >>> buf = PinnedBuffer(buffer_id=0, storage=zeros((80,), uint8))
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
from typing import Dict, List, Optional, Tuple
from threading import Condition

from attrs import define, field
from attrs.validators import instance_of as attrsval_instance_of
from numpy import ndarray, uint8 as np_uint8
from numpy import dtype as np_dtype

from cubie.memory import default_memmgr
from cubie.memory.mem_manager import (
    MemoryManager,
    STAGING_POOL_DEPTH,
)


@define
class PinnedBuffer:
    """Wrapper for a reusable pinned memory buffer.

    Attributes
    ----------
    buffer_id : int
        Unique identifier for this buffer.
    storage : ndarray
        The buffer's pinned memory, as bytes.
    in_use : bool
        Whether the buffer is currently in use.
    array : ndarray or None
        The data being staged, set by :meth:`shape_as`.
    """

    buffer_id: int = field(validator=attrsval_instance_of(int))
    storage: ndarray = field(validator=attrsval_instance_of(ndarray))
    in_use: bool = field(default=False, validator=attrsval_instance_of(bool))
    array: Optional[ndarray] = field(default=None, init=False)

    @property
    def capacity(self) -> int:
        """Bytes the buffer can hold."""
        return self.storage.nbytes

    def shape_as(self, shape: Tuple[int, ...], dtype: np_dtype) -> None:
        """Set ``array`` to the start of the buffer in this shape."""
        nbytes = int(prod(shape)) * np_dtype(dtype).itemsize
        self.array = self.storage[:nbytes].view(dtype).reshape(shape)


@define
class ChunkBufferPool:
    """Pool of reusable pinned buffers for chunked transfers.

    Manages allocation and lifecycle of pinned memory buffers used
    for staging data during chunked device transfers. Buffers are
    reused for any block that fits.

    Attributes
    ----------
    _buffers : Dict[str, List[PinnedBuffer]]
        Pool of buffers organized by array name.
    _condition : Condition
        Guards the pool and wakes blocked acquirers on release.
    _next_id : int
        Counter for unique buffer IDs.
    _memory_manager : MemoryManager
        Manager whose cumulative pinned budget accounts the pool's
        buffers.
    """

    _buffers: Dict[str, List[PinnedBuffer]] = field(factory=dict)
    _condition: Condition = field(factory=Condition)
    _next_id: int = field(default=0)
    _memory_manager: MemoryManager = field(
        default=default_memmgr,
        validator=attrsval_instance_of(MemoryManager),
    )

    def acquire(
        self,
        array_name: str,
        shape: Tuple[int, ...],
        dtype: np_dtype,
    ) -> PinnedBuffer:
        """Acquire a pinned buffer for the given array.

        Reuses the smallest idle buffer that fits, else allocates
        within the depth and pinned limits, else waits.

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
            A buffer ready for use, its ``array`` in ``shape``.
        """
        nbytes = int(prod(shape)) * np_dtype(dtype).itemsize
        with self._condition:
            while True:
                buffers = self._buffers.setdefault(array_name, [])
                in_flight = [buf for buf in buffers if buf.in_use]
                idle = [buf for buf in buffers if not buf.in_use]
                fitting = [buf for buf in idle if buf.capacity >= nbytes]
                if fitting:
                    buf = min(fitting, key=lambda item: item.capacity)
                    buf.in_use = True
                    buf.shape_as(shape, dtype)
                    return buf
                if idle:
                    # Too small: replace it.
                    buffers.remove(idle[0])
                    continue

                if not in_flight:
                    # First buffer per label: forced, never None.
                    new_buffer = self._allocate_buffer(
                        nbytes, force=True
                    )
                else:
                    # None when a bound refuses.
                    new_buffer = None
                    if len(in_flight) < STAGING_POOL_DEPTH:
                        new_buffer = self._allocate_buffer(nbytes)
                if new_buffer is not None:
                    new_buffer.in_use = True
                    new_buffer.shape_as(shape, dtype)
                    buffers.append(new_buffer)
                    return new_buffer

                # Wait for a buffer release, then retry.
                self._condition.wait()

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
        nbytes: int,
        force: bool = False,
    ) -> Optional[PinnedBuffer]:
        """Allocate a new pinned buffer.

        Parameters
        ----------
        nbytes
            Capacity of the buffer in bytes.
        force
            Ignore the manager's pinned-memory limits.

        Returns
        -------
        PinnedBuffer or None
            Newly allocated pinned buffer, or ``None`` when no
            pinned memory is available.
        """
        storage = self._memory_manager.allocate_pinned_array(
            (nbytes,), np_uint8, force=force
        )
        if storage is None:
            return None
        storage.fill(0)

        buffer_id = self._next_id
        self._next_id += 1

        return PinnedBuffer(buffer_id=buffer_id, storage=storage)
