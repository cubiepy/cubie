"""The ``cuda`` module and cubie's CUDA simulator extensions.

``cuda`` is ``numba_cuda_mlir.cuda``, or the vendored simulator with
the extensions below applied under ``NUMBA_ENABLE_CUDASIM=1``. Import
``cuda`` from here, never from ``cubie.vendored.cudasim``.

Published Objects
-----------------
:data:`CUDA_SIMULATION`
    ``True`` when ``NUMBA_ENABLE_CUDASIM=1``.
:data:`cuda`
    The CUDA module.
:class:`Stream`, :class:`DeviceNDArrayBase`, :class:`DeviceNDArray`,
:class:`MappedNDArray`, :class:`CudaSupportError`
    Classes of ``cuda``.
:func:`fmax`, :func:`fmin`
    Max and min that drop a NaN operand.
:func:`page_locked_block`, :func:`stream_ordered_buffer`,
:func:`pool_idle_bytes`, :func:`stream_idle`,
:func:`flush_deferred_frees`, :func:`is_pinned_array`
    Simulator stand-ins for :mod:`cubie.memory.driver_memory`.

Simulator Extensions
--------------------
- ``cuda.jit`` ignores GPU-only options.
- ``activemask`` reports all lanes; ``all_sync``/``any_sync`` return
  the thread's predicate; ``syncwarp`` does nothing; ``stwt`` stores.
- ``cuda.experimental.consteval`` returns its argument.
- Streams have a null ``handle``; ``cuda.cudadrv.driver.Stream`` is
  the stream class.
- ``DeviceNDArrayBase``/``DeviceNDArray``/``MappedNDArray`` are the
  simulator's device array.
- Kernels report ``targetoptions``; the context reports 1 GiB free of
  8 GiB.
- Local and shared arrays take a Numba type in ``view``.
"""

from ctypes import c_void_p
from inspect import signature
import os
import sys
import traceback
from types import ModuleType
from typing import Any

from numpy import (
    empty as np_empty,
    fmax as np_fmax,
    fmin as np_fmin,
    frombuffer as np_frombuffer,
    ndarray as np_ndarray,
    uint8 as np_uint8,
)

from numba_cuda_mlir.numba_cuda import types as numba_types
from numba_cuda_mlir.numba_cuda.np.numpy_support import as_dtype


CUDA_SIMULATION: bool = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

if CUDA_SIMULATION:  # pragma: no cover - simulated
    from cubie.vendored import cudasim as cuda
else:  # pragma: no cover - exercised in GPU environments
    from numba_cuda_mlir import cuda

# Device max/min drop a NaN operand; the simulator needs numpy's.
fmax = np_fmax if CUDA_SIMULATION else max
fmin = np_fmin if CUDA_SIMULATION else min


def page_locked_block(nbytes: int) -> Any:
    """Return a host buffer; the simulator has no pinning."""
    return bytearray(nbytes)


def stream_ordered_buffer(nbytes: int, stream: Any) -> Any:
    """Return a flat byte device array."""
    return cuda.device_array(nbytes, dtype=np_uint8)


def pool_idle_bytes() -> int:
    """Return zero; the simulator has no device pool."""
    return 0


def stream_idle(stream: Any) -> bool:
    """Return ``True``; simulated work completes on launch."""
    return True


def flush_deferred_frees() -> None:
    """Do nothing; the simulator defers no frees."""


def is_pinned_array(array: Any) -> bool:
    """Return ``False``; the simulator has no page-locked memory."""
    return False


class TypedViewArray(np_ndarray):
    """Array whose ``view`` also takes a Numba type, as device code may."""

    def view(self, dtype=None, *args, **kwargs):
        if isinstance(dtype, numba_types.Type):
            dtype = as_dtype(dtype)
        if dtype is None:
            return super().view(*args, **kwargs)
        return super().view(dtype, *args, **kwargs)


def _local_array(self, shape, dtype, alignment=None):
    """Return a local array that accepts a Numba type in ``view``."""
    if alignment is not None:
        raise RuntimeError("Array alignment is not supported in cudasim")
    if isinstance(dtype, numba_types.Type):
        dtype = as_dtype(dtype)
    return np_empty(shape, dtype).view(TypedViewArray)


def _shared_array(self, shape, dtype, alignment=None):
    """Return a shared array that accepts a Numba type in ``view``.

    Static allocations are keyed by the calling line, as upstream.
    """
    if alignment is not None:
        raise RuntimeError("Array alignment is not supported in cudasim")
    if isinstance(dtype, numba_types.Type):
        dtype = as_dtype(dtype)
    if shape == 0:
        count = self._dynshared_size // dtype.itemsize
        return np_frombuffer(
            self._dynshared.data, dtype=dtype, count=count
        ).view(TypedViewArray)
    stack = traceback.extract_stack(sys._getframe())
    caller = stack[-2][0:2]
    array = self._allocations.get(caller)
    if array is None:
        array = np_empty(shape, dtype).view(TypedViewArray)
        self._allocations[caller] = array
    return array


def _extend_simulator() -> None:  # pragma: no cover - simulated
    """Add cubie's CUDA API to the vendored simulator."""
    from cubie.vendored.cudasim import api, kernel, kernelapi
    from cubie.vendored.cudasim.cudadrv import devicearray, devices, driver

    simulator_jit = api.jit
    jit_options = frozenset(signature(simulator_jit).parameters)

    def jit(func_or_sig=None, **options):
        """Simulator ``jit`` that ignores GPU-only options."""
        return simulator_jit(
            func_or_sig,
            **{
                name: value for name, value in options.items()
                if name in jit_options
            },
        )

    def stwt(array, index, value):
        array[index] = value

    fake_module = kernelapi.FakeCUDAModule
    fake_module.activemask = lambda self: 0xFFFFFFFF
    fake_module.all_sync = lambda self, mask, predicate: predicate
    fake_module.any_sync = lambda self, mask, predicate: predicate
    fake_module.syncwarp = lambda self, mask=0xFFFFFFFF: None
    fake_module.stwt = lambda self, array, index, value: stwt(
        array, index, value
    )
    kernelapi.FakeCUDALocal.array = _local_array
    kernelapi.FakeCUDAShared.array = _shared_array

    kernel.FakeCUDAKernel.targetoptions = property(
        lambda self: {"device": self._device}
    )
    devices.FakeCUDAContext.get_memory_info = (
        lambda self: devices._MemoryInfo(1024**3, 8 * 1024**3)
    )

    api.stream.handle = c_void_p(0)
    driver.Stream = api.stream
    devicearray.DeviceNDArrayBase = devicearray.FakeCUDAArray
    devicearray.DeviceNDArray = devicearray.FakeCUDAArray
    devicearray.MappedNDArray = devicearray.FakeCUDAArray

    experimental = ModuleType(f"{cuda.__name__}.experimental")
    experimental.consteval = lambda value: value

    for module in (api, cuda):
        module.jit = jit
        module.stwt = stwt
    cuda.experimental = experimental


if CUDA_SIMULATION:  # pragma: no cover - simulated
    _extend_simulator()

Stream = cuda.cudadrv.driver.Stream
DeviceNDArrayBase = cuda.devicearray.DeviceNDArrayBase
DeviceNDArray = cuda.devicearray.DeviceNDArray
MappedNDArray = cuda.devicearray.MappedNDArray
CudaSupportError = cuda.cudadrv.error.CudaSupportError


__all__ = [
    "CUDA_SIMULATION",
    "CudaSupportError",
    "cuda",
    "DeviceNDArray",
    "DeviceNDArrayBase",
    "flush_deferred_frees",
    "fmax",
    "fmin",
    "is_pinned_array",
    "MappedNDArray",
    "page_locked_block",
    "pool_idle_bytes",
    "Stream",
    "stream_idle",
    "stream_ordered_buffer",
]
