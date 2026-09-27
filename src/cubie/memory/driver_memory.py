"""Device and page-locked memory through the CUDA driver.

Under the CUDA simulator these names are the stand-ins from
:mod:`cubie.cubie_cudasim_extensions`.

Published Functions
-------------------
:func:`page_locked_block`
    Page-locked host memory as a buffer.
:func:`stream_ordered_buffer`
    Device memory from the stream-ordered pool, freed on its stream.
:func:`pool_idle_bytes`
    Bytes the device pool holds but no array uses.
:func:`stream_idle`
    Whether a stream's queued work has finished.
:func:`flush_deferred_frees`
    Run Numba's queued frees.
:func:`is_pinned_array`
    Whether a host array is backed by page-locked memory.
"""

from ctypes import c_void_p
from typing import Any

from numpy import uint8 as np_uint8

from cubie.cubie_cudasim_extensions import CUDA_SIMULATION, cuda


if CUDA_SIMULATION:  # pragma: no cover - simulated
    from cubie.cubie_cudasim_extensions import (  # noqa: F401
        flush_deferred_frees,
        is_pinned_array,
        page_locked_block,
        pool_idle_bytes,
        stream_idle,
        stream_ordered_buffer,
    )

else:  # pragma: no cover - exercised in GPU environments
    from cuda.bindings import driver as cuda_driver

    def page_locked_block(nbytes: int) -> Any:
        """Return ``nbytes`` of page-locked host memory as a buffer.

        Numba frees it once the block is collected.
        """
        return cuda.current_context().memhostalloc(nbytes)

    def _driver_value(result: Any) -> Any:
        """Return a driver call's value, raising on its error code."""
        if result[0] != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver call failed: {result[0]}")
        return result[1]

    def stream_ordered_buffer(nbytes: int, stream: Any) -> Any:
        """Return ``nbytes`` of device memory ordered on ``stream``.

        The bytes come from the device's stream-ordered pool and are
        freed on ``stream`` once the array and every view of it are
        collected, so the free never waits for the device. Freed
        bytes return to the device at the stream's next sync.

        Raises
        ------
        MemoryError
            If the pool cannot supply ``nbytes``.
        """
        out_of_memory = cuda_driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY
        error, pointer = cuda_driver.cuMemAllocAsync(nbytes, stream.handle)
        if error == out_of_memory:
            # Frees queued on the stream complete at its sync.
            stream.synchronize()
            error, pointer = cuda_driver.cuMemAllocAsync(
                nbytes, stream.handle
            )
        if error == out_of_memory:
            raise MemoryError(
                f"the device could not allocate {nbytes} bytes"
            )
        _driver_value((error, pointer))
        address = int(pointer)

        def free() -> None:
            cuda_driver.cuMemFreeAsync(address, stream.handle)

        memory = cuda.MemoryPointer(
            cuda.current_context(), c_void_p(address), nbytes,
            finalizer=free,
        )
        return cuda.devicearray.DeviceNDArray(
            (nbytes,), (1,), np_uint8, stream=stream, gpu_data=memory
        )

    def pool_idle_bytes() -> int:
        """Return bytes the device pool holds but no array uses."""
        attribute = cuda_driver.CUmemPool_attribute
        device = cuda.get_current_device().id
        pool = _driver_value(
            cuda_driver.cuDeviceGetDefaultMemPool(cuda_driver.CUdevice(device))
        )
        reserved = _driver_value(cuda_driver.cuMemPoolGetAttribute(
            pool, attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
        ))
        used = _driver_value(cuda_driver.cuMemPoolGetAttribute(
            pool, attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT
        ))
        return int(reserved) - int(used)

    def stream_idle(stream: Any) -> bool:
        """Return whether all work queued on ``stream`` has finished."""
        error = cuda_driver.cuStreamQuery(stream.handle)[0]
        return error == cuda_driver.CUresult.CUDA_SUCCESS

    def flush_deferred_frees() -> None:
        """Run Numba's queued frees; each waits for the whole device."""
        cuda.current_context().memory_manager.deallocations.clear()

    def is_pinned_array(array: Any) -> bool:
        """Return whether a host array is backed by page-locked memory.

        Asks the driver about the array's first byte.
        """
        if array.size == 0:
            return False
        error, memory_type = cuda_driver.cuPointerGetAttribute(
            cuda_driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            array.ctypes.data,
        )
        return (
            error == cuda_driver.CUresult.CUDA_SUCCESS
            and memory_type == cuda_driver.CUmemorytype.CU_MEMORYTYPE_HOST
        )


__all__ = [
    "flush_deferred_frees",
    "is_pinned_array",
    "page_locked_block",
    "pool_idle_bytes",
    "stream_idle",
    "stream_ordered_buffer",
]
