"""Page-locked slabs sub-allocated into host arrays of any shape.

Arrays take the best-fitting free extent of any slab; a collected
array returns its extent for the next request of any size. Slabs are
freed only by :meth:`PinnedArena.release_free_slabs`.

Published Classes
-----------------
:class:`PinnedArena`
    Slab sub-allocator for page-locked host arrays.

    >>> arena = PinnedArena()
    >>> array = arena.allocate((4, 3), "float32", cap=None)
"""

import ctypes
from bisect import insort
from collections import deque
from math import prod
from threading import Lock
from typing import Any, List, Optional, Tuple
from weakref import finalize

from attrs import define, field
from numpy import dtype as np_dtype, ndarray
from numpy.typing import DTypeLike

from cubie.cuda_simsafe import alloc_pinned_slab


PINNED_ALIGNMENT_BYTES = 4096
"""Alignment of every extent: one host page."""

SLAB_GRANULE_BYTES = 2 * 1024**2
"""Slab sizes round up to this many bytes."""

MIN_SLAB_BYTES = 64 * 1024**2
"""Smallest slab, so small arrays share one allocation."""


def _round_up(nbytes: int, granule: int) -> int:
    """Round ``nbytes`` up to a multiple of ``granule``."""
    return -(-nbytes // granule) * granule


@define(eq=False)
class _Slab:
    """One page-locked allocation and its free extents."""

    address: int
    size: int
    owner: Any
    # Sorted (offset, length) pairs.
    free: List[Tuple[int, int]] = field(factory=list)
    live: int = 0

    def take(self, nbytes: int) -> Optional[int]:
        """Return the offset of the best-fitting free extent."""
        best = None
        for index, (offset, length) in enumerate(self.free):
            if length >= nbytes and (
                best is None or length < self.free[best][1]
            ):
                best = index
        if best is None:
            return None
        offset, length = self.free.pop(best)
        if length > nbytes:
            insort(self.free, (offset + nbytes, length - nbytes))
        self.live += 1
        return offset

    def give(self, offset: int, nbytes: int) -> None:
        """Return an extent, merging it with adjacent free extents."""
        insort(self.free, (offset, nbytes))
        merged = []
        for start, length in self.free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((start, length))
        self.free = merged
        self.live -= 1

    def best_fit(self, nbytes: int) -> Optional[int]:
        """Return the length of the smallest free extent that fits."""
        fits = [length for _, length in self.free if length >= nbytes]
        return min(fits) if fits else None


@define(eq=False)
class PinnedArena:
    """Slab sub-allocator for page-locked host arrays.

    Attributes
    ----------
    reserved_bytes
        Page-locked bytes held in slabs.
    live_bytes
        Bytes of extents backing reachable arrays.
    """

    _slabs: List[_Slab] = field(factory=list, init=False)
    _lock: Lock = field(factory=Lock, init=False)
    # Collected arrays queue extents here; GC may run under the lock.
    _releases: deque = field(factory=deque, init=False)
    _live_bytes: int = field(default=0, init=False)

    @property
    def reserved_bytes(self) -> int:
        """Page-locked bytes held in slabs."""
        with self._lock:
            return sum(slab.size for slab in self._slabs)

    @property
    def live_bytes(self) -> int:
        """Bytes of extents backing reachable arrays."""
        with self._lock:
            self._apply_releases()
            return self._live_bytes

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        cap: Optional[int],
    ) -> Optional[ndarray]:
        """Return an uninitialised page-locked array.

        Parameters
        ----------
        shape
            Shape of the array.
        dtype
            Element type of the array.
        cap
            Reserved bytes a new slab may not exceed; ``None`` for
            no limit.

        Returns
        -------
        numpy.ndarray or None
            The array, or ``None`` when nothing fits and a new slab
            would exceed ``cap``.

        Raises
        ------
        Exception
            The driver's error when a new slab cannot be page-locked.
        """
        dtype = np_dtype(dtype)
        nbytes = int(prod(shape)) * dtype.itemsize
        extent = _round_up(max(nbytes, 1), PINNED_ALIGNMENT_BYTES)
        with self._lock:
            self._apply_releases()
            slab, offset = self._take(extent)
            if slab is None:
                slab = self._grow(extent, cap)
                if slab is None:
                    return None
                offset = slab.take(extent)
            self._live_bytes += extent
        buffer = (ctypes.c_uint8 * extent).from_address(
            slab.address + offset
        )
        array = ndarray(shape, dtype=dtype, buffer=buffer)
        # Fires once the array and all its views are collected.
        finalize(array, self._releases.append, (slab, offset, extent))
        return array

    def release_free_slabs(self) -> int:
        """Free every slab with no live array.

        Freeing page-locked memory synchronizes the device.

        Returns
        -------
        int
            Bytes released.
        """
        with self._lock:
            self._apply_releases()
            idle = [slab for slab in self._slabs if slab.live == 0]
            self._slabs = [slab for slab in self._slabs if slab.live]
        # Dropping the owner frees the page-locked memory.
        return sum(slab.size for slab in idle)

    def _take(self, extent: int) -> Tuple[Optional[_Slab], int]:
        """Carve ``extent`` from the best-fitting slab; lock held."""
        best = None
        best_length = None
        for slab in self._slabs:
            length = slab.best_fit(extent)
            if length is not None and (
                best_length is None or length < best_length
            ):
                best, best_length = slab, length
        if best is None:
            return None, 0
        return best, best.take(extent)

    def _grow(self, extent: int, cap: Optional[int]) -> Optional[_Slab]:
        """Page-lock a slab that fits ``extent``; lock held."""
        size = _round_up(max(extent, MIN_SLAB_BYTES), SLAB_GRANULE_BYTES)
        reserved = sum(slab.size for slab in self._slabs)
        if cap is not None and reserved + size > cap:
            # A slab sized to the request alone may still fit.
            size = _round_up(extent, SLAB_GRANULE_BYTES)
            if reserved + size > cap:
                return None
        address, owner = alloc_pinned_slab(size)
        slab = _Slab(address=address, size=size, owner=owner)
        slab.free.append((0, size))
        self._slabs.append(slab)
        return slab

    def _apply_releases(self) -> None:
        """Return queued extents to their slabs; lock held."""
        while True:
            try:
                slab, offset, extent = self._releases.popleft()
            except IndexError:
                return
            slab.give(offset, extent)
            self._live_bytes -= extent
