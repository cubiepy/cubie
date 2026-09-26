"""Page-locked slabs handed out as host arrays of any shape.

A collected array's bytes hold the next array of any shape. The
memory manager decides when slabs are added and freed.

Published Classes
-----------------
:class:`PinnedArena`
    Page-locked slabs handed out as host arrays.

    >>> arena = PinnedArena()
    >>> arena.add_slab(1024, limit=None)
    True
    >>> array = arena.allocate((4, 3), "float32")
"""

import ctypes
import sys
from bisect import insort
from collections import deque
from math import prod
from typing import Any, List, Optional, Tuple
from weakref import finalize

from attrs import define, field
from numpy import dtype as np_dtype, empty as np_empty, ndarray
from numpy import uint8 as np_uint8
from numpy.typing import DTypeLike

from cubie.cuda_simsafe import CUDA_SIMULATION, cupy


PAGE_BYTES = 4096
"""Host page size; every array starts on a page boundary."""

MIN_SLAB_BYTES = 64 * 1024**2
"""Smallest slab, so that many small arrays share one."""


if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32")
    _kernel32.VirtualAlloc.restype = ctypes.c_void_p
    _kernel32.VirtualAlloc.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong
    ]
    _kernel32.VirtualFree.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong
    ]

    def _check_commit(nbytes: int) -> None:
        """Raise ``MemoryError`` if Windows cannot commit ``nbytes``."""
        # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
        address = _kernel32.VirtualAlloc(None, nbytes, 0x3000, 0x04)
        if not address:
            raise MemoryError(f"Windows cannot commit {nbytes} bytes")
        # MEM_RELEASE
        _kernel32.VirtualFree(ctypes.c_void_p(address), 0, 0x8000)

else:

    def _check_commit(nbytes: int) -> None:
        """Do nothing; only Windows needs the check."""


def page_locked_slab(nbytes: int) -> Tuple[int, Any]:
    """Return the address and owner of ``nbytes`` of pinned memory.

    Dropping the owner frees it and waits for the whole device.

    Raises
    ------
    MemoryError
        If the memory cannot be page-locked.
    """
    if CUDA_SIMULATION:
        buffer = np_empty(nbytes, dtype=np_uint8)
        return buffer.ctypes.data, buffer
    _check_commit(nbytes)
    try:
        memory = cupy.cuda.pinned_memory.PinnedMemory(
            nbytes, cupy.cuda.runtime.hostAllocPortable
        )
    except cupy.cuda.runtime.CUDARuntimeError as error:
        raise MemoryError(f"cannot page-lock {nbytes} bytes") from error
    return int(memory.ptr), memory


def _page_round(nbytes: int) -> int:
    """Round ``nbytes`` up to whole pages, at least one."""
    pages = -(-max(nbytes, 1) // PAGE_BYTES)
    return pages * PAGE_BYTES


@define(eq=False)
class _Slab:
    """One page-locked allocation and its unused ranges."""

    address: int
    size: int
    owner: Any
    # Sorted (offset, length) ranges not backing an array.
    free: List[Tuple[int, int]] = field(factory=list)
    arrays: int = 0
    # Consecutive idle checks that found no array in the slab.
    idle_checks: int = 0

    def take(self, index: int, nbytes: int) -> int:
        """Take ``nbytes`` from the front of free range ``index``."""
        offset, length = self.free.pop(index)
        if length > nbytes:
            insort(self.free, (offset + nbytes, length - nbytes))
        self.arrays += 1
        return offset

    def give_back(self, offset: int, nbytes: int) -> None:
        """Return a range, merging it with free neighbours."""
        insort(self.free, (offset, nbytes))
        merged = []
        for start, length in self.free:
            if merged and sum(merged[-1]) == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((start, length))
        self.free = merged
        self.arrays -= 1


@define(eq=False)
class PinnedArena:
    """Page-locked slabs handed out as host arrays; not thread-safe."""

    _slabs: List[_Slab] = field(factory=list, init=False)
    # Ranges of collected arrays; collection can happen mid-call.
    _returned: deque = field(factory=deque, init=False)
    _live_bytes: int = field(default=0, init=False)

    @property
    def reserved_bytes(self) -> int:
        """Page-locked bytes held in slabs."""
        return sum(slab.size for slab in self._slabs)

    @property
    def live_bytes(self) -> int:
        """Bytes backing arrays that are still reachable."""
        self._take_back_returned()
        return self._live_bytes

    def allocate(
        self, shape: Tuple[int, ...], dtype: DTypeLike
    ) -> Optional[ndarray]:
        """Return an uninitialised array; ``None`` if nothing fits."""
        dtype = np_dtype(dtype)
        nbytes = _page_round(int(prod(shape)) * dtype.itemsize)
        self._take_back_returned()
        best = None
        for slab in self._slabs:
            for index, (offset, length) in enumerate(slab.free):
                if length >= nbytes and (best is None or length < best[2]):
                    best = (slab, index, length)
        if best is None:
            return None
        slab, index, _ = best
        offset = slab.take(index, nbytes)
        self._live_bytes += nbytes
        buffer = (ctypes.c_uint8 * nbytes).from_address(slab.address + offset)
        array = ndarray(shape, dtype=dtype, buffer=buffer)
        # Fires once the array and all its views are collected.
        finalize(array, self._returned.append, (slab, offset, nbytes))
        return array

    def add_slab(self, nbytes: int, limit: Optional[int]) -> bool:
        """Page-lock a slab of at least ``nbytes`` within ``limit``.

        Uses :data:`MIN_SLAB_BYTES` unless that exceeds ``limit``.

        Returns
        -------
        bool
            ``False`` when even the array's size exceeds ``limit``.
        """
        needed = _page_round(nbytes)
        size = max(needed, MIN_SLAB_BYTES)
        if limit is not None and size > limit:
            size = needed
            if size > limit:
                return False
        address, owner = page_locked_slab(size)
        slab = _Slab(address=address, size=size, owner=owner)
        slab.free.append((0, size))
        self._slabs.append(slab)
        return True

    def contains(self, array: ndarray) -> bool:
        """Return whether ``array``'s memory lies in a slab."""
        address = array.ctypes.data
        return any(
            slab.address <= address < slab.address + slab.size
            for slab in self._slabs
        )

    def check_idle_slabs(self) -> None:
        """Count one more idle check for each slab with no arrays."""
        self._take_back_returned()
        for slab in self._slabs:
            slab.idle_checks = 0 if slab.arrays else slab.idle_checks + 1

    def free_idle_slabs(self, min_idle_checks: int = 0) -> int:
        """Free slabs with no arrays; waits for the whole device.

        Parameters
        ----------
        min_idle_checks
            Free only slabs found idle by at least this many
            consecutive :meth:`check_idle_slabs` calls.

        Returns
        -------
        int
            Bytes freed.
        """
        self._take_back_returned()
        idle = [
            slab for slab in self._slabs
            if not slab.arrays and slab.idle_checks >= min_idle_checks
        ]
        self._slabs = [slab for slab in self._slabs if slab not in idle]
        # Dropping a slab's owner frees its memory.
        return sum(slab.size for slab in idle)

    def _take_back_returned(self) -> None:
        """Return collected arrays' ranges to their slabs."""
        while self._returned:
            slab, offset, nbytes = self._returned.popleft()
            slab.give_back(offset, nbytes)
            self._live_bytes -= nbytes
