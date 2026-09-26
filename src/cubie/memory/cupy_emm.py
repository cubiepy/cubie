"""Numba EMM plugin allocating from the device's stream-ordered pool.

``cuda.device_array`` returns a native ``DeviceNDArray`` backed by
``cudaMallocAsync`` (CuPy's ``malloc_async``). Freed blocks above the
pool's release threshold (set by the memory manager) return to the
device at the next sync of their stream.

See Also
--------
:class:`~cubie.memory.mem_manager.MemoryManager`
    Coordinates allocation through this plugin.
"""

import ctypes
import logging
from typing import Any, Callable, Optional

from cubie.cuda_simsafe import cuda, cupy, CUDA_SIMULATION

logger = logging.getLogger(__name__)

CUDA_ERROR_MEMORY_ALLOCATION = 2
"""``cudaErrorMemoryAllocation``, from the CUDA runtime API."""


if not CUDA_SIMULATION:

    class CuPyAsyncNumbaManager(
        cuda.GetIpcHandleMixin, cuda.HostOnlyCUDAMemoryManager
    ):
        """EMM plugin allocating native Numba arrays from the device pool.

        Adapted from the numba cupy-EMM tutorial (BSD 2-Clause; see
        THIRD_PARTY_LICENSES). Blocks are allocated and freed on the
        CuPy stream current at allocation.
        """

        def __init__(self, context) -> None:
            super().__init__(context=context)
            # Kept alive so CuPy frees the block on finalize.
            self._allocations: dict[int, Any] = {}
            self._pool = None
            self.is_cupy = True

        def initialize(self) -> None:
            super().initialize()
            # Runs on every context activation; configure the pool once.
            if self._pool is None:
                runtime = cupy.cuda.runtime
                pool = runtime.deviceGetMemPool(runtime.getDevice())
                runtime.memPoolSetAttribute(
                    pool, runtime.cudaMemPoolAttrReleaseThreshold, 0
                )
                self._pool = pool

        def memalloc(self, nbytes: int) -> "cuda.MemoryPointer":
            try:
                cp_mp = cupy.cuda.memory.malloc_async(nbytes)
            except cupy.cuda.runtime.CUDARuntimeError as error:
                if error.status != CUDA_ERROR_MEMORY_ALLOCATION:
                    raise
                # Let queued frees complete, then retry once.
                cupy.cuda.get_current_stream().synchronize()
                cp_mp = cupy.cuda.memory.malloc_async(nbytes)
            self._allocations[cp_mp.ptr] = cp_mp
            return cuda.MemoryPointer(
                cuda.current_context(),
                ctypes.c_void_p(int(cp_mp.ptr)),
                nbytes,
                finalizer=self._make_finalizer(cp_mp.ptr),
            )

        def _make_finalizer(self, ptr: int) -> Callable[[], None]:
            allocations = self._allocations

            def finalizer() -> None:
                # Dropping the last reference frees the block.
                allocations.pop(ptr, None)

            return finalizer

        def pool_reserved_bytes(self) -> int:
            """Bytes the pool currently holds from the device."""
            if self._pool is None:
                return 0
            runtime = cupy.cuda.runtime
            return runtime.memPoolGetAttribute(
                self._pool, runtime.cudaMemPoolAttrReservedMemCurrent
            )

        def set_release_threshold(self, nbytes: int) -> None:
            """Retain up to ``nbytes`` of freed blocks across syncs."""
            if self._pool is not None:
                runtime = cupy.cuda.runtime
                runtime.memPoolSetAttribute(
                    self._pool,
                    runtime.cudaMemPoolAttrReleaseThreshold,
                    int(nbytes),
                )

        def get_memory_info(self) -> "cuda.MemoryInfo":
            # Device free plus the pool's reserved but unused bytes.
            runtime = cupy.cuda.runtime
            free, total = runtime.memGetInfo()
            if self._pool is not None:
                reserved = runtime.memPoolGetAttribute(
                    self._pool, runtime.cudaMemPoolAttrReservedMemCurrent
                )
                used = runtime.memPoolGetAttribute(
                    self._pool, runtime.cudaMemPoolAttrUsedMemCurrent
                )
                free += reserved - used
            return cuda.MemoryInfo(free=free, total=total)

        def reset(self, stream: Optional[Any] = None) -> None:
            super().reset()
            if self._pool is not None:
                if stream is None:
                    stream = cupy.cuda.get_current_stream()
                stream.synchronize()
                cupy.cuda.runtime.memPoolTrimTo(self._pool, 0)

        @property
        def interface_version(self) -> int:
            return 1

    def install_async_emm() -> None:
        """Install the device pool as Numba's device memory manager.

        Must run before the CUDA context is created; the manager takes effect
        on first context creation.
        """
        cuda.set_memory_manager(CuPyAsyncNumbaManager)

else:  # pragma: no cover - simulated: no device, no EMM
    CuPyAsyncNumbaManager = None

    def install_async_emm() -> None:
        return None
