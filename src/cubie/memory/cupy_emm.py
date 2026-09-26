"""Numba EMM plugin allocating from the device's stream-ordered pool.

``cuda.device_array`` returns a native ``DeviceNDArray`` from
``cudaMallocAsync``. Freed memory beyond what the manager keeps
returns to the device at the next sync of the freeing stream.

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

# cudaErrorMemoryAllocation in the CUDA runtime API.
_OUT_OF_MEMORY = 2


if not CUDA_SIMULATION:

    class CuPyAsyncNumbaManager(
        cuda.GetIpcHandleMixin, cuda.HostOnlyCUDAMemoryManager
    ):
        """Numba EMM plugin allocating from the device's memory pool.

        Adapted from the numba cupy-EMM tutorial (BSD 2-Clause; see
        THIRD_PARTY_LICENSES). Uses the current CuPy stream.
        """

        def __init__(self, context) -> None:
            super().__init__(context=context)
            # Kept alive so CuPy frees the block on finalize.
            self._allocations: dict[int, Any] = {}
            self._pool = None
            self._release_threshold = 0
            self.is_cupy = True

        def initialize(self) -> None:
            super().initialize()
            # Runs on every context activation; configure the pool once.
            if self._pool is None:
                runtime = cupy.cuda.runtime
                self._pool = runtime.deviceGetMemPool(runtime.getDevice())
                self._set_release_threshold(0)

        def memalloc(self, nbytes: int) -> "cuda.MemoryPointer":
            try:
                cp_mp = cupy.cuda.memory.malloc_async(nbytes)
            except cupy.cuda.runtime.CUDARuntimeError as error:
                if error.status != _OUT_OF_MEMORY:
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

        def reserved_bytes(self) -> int:
            """Bytes the pool currently holds from the device."""
            if self._pool is None:
                return 0
            runtime = cupy.cuda.runtime
            return runtime.memPoolGetAttribute(
                self._pool, runtime.cudaMemPoolAttrReservedMemCurrent
            )

        def keep_reserved(self) -> None:
            """Keep the pool's current memory through stream syncs."""
            reserved = self.reserved_bytes()
            if reserved > self._release_threshold:
                self._set_release_threshold(reserved)

        def release_beyond(self, nbytes: int) -> None:
            """Return freed memory beyond ``nbytes`` at stream syncs."""
            if nbytes < self._release_threshold:
                self._set_release_threshold(nbytes)

        def _set_release_threshold(self, nbytes: int) -> None:
            """Set how many freed bytes the pool keeps through syncs."""
            if self._pool is None:
                return
            runtime = cupy.cuda.runtime
            runtime.memPoolSetAttribute(
                self._pool,
                runtime.cudaMemPoolAttrReleaseThreshold,
                int(nbytes),
            )
            self._release_threshold = nbytes

        def get_memory_info(self) -> "cuda.MemoryInfo":
            # Device free plus the pool's reserved but unused bytes.
            runtime = cupy.cuda.runtime
            free, total = runtime.memGetInfo()
            if self._pool is not None:
                used = runtime.memPoolGetAttribute(
                    self._pool, runtime.cudaMemPoolAttrUsedMemCurrent
                )
                free += self.reserved_bytes() - used
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
