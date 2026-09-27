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
:data:`cuda_driver`
    The CUDA driver bindings (``cuda.bindings.driver``), or the
    simulated driver.
:class:`Stream`, :class:`DeviceNDArrayBase`, :class:`DeviceNDArray`,
:class:`MappedNDArray`, :class:`CudaSupportError`
    Classes of ``cuda``.
:func:`fmax`, :func:`fmin`
    Max and min that drop a NaN operand.

Simulator Extensions
--------------------
- ``cuda.jit`` ignores GPU-only options.
- ``activemask`` reports all lanes; ``all_sync``/``any_sync`` return
  the thread's predicate; ``syncwarp`` does nothing; ``stwt`` stores.
- ``cuda.experimental.consteval`` returns its argument.
- Streams have a null ``handle``; ``cuda.cudadrv.driver.Stream`` is
  the stream class.
- Events record the host time; ``query`` reports complete and
  ``cuda.event_elapsed_time`` returns the milliseconds between two.
- ``DeviceNDArrayBase``/``DeviceNDArray``/``MappedNDArray`` are the
  simulator's device array, which also builds from a shape, strides,
  dtype and ``gpu_data``; ``cuda.MemoryPointer`` wraps host bytes.
- Driver ``host_to_device``/``device_to_host`` copy flat bytes.
- The device reports the attributes cubie's launch sizing reads; the
  context reports 1 GiB free of 8 GiB, one resident block per SM and
  an empty deallocation queue.
- ``memhostalloc`` returns a host buffer that is not an array, as
  the driver's is.
- Kernels report ``targetoptions``; ``compile_for`` gives a kernel
  one empty-signature overload, and kernels report zero registers and
  local memory.
- Local and shared arrays take a Numba type in ``view``.
- :data:`cuda_driver` allocates host bytes, reports every stream idle,
  an empty pool and no page-locked memory.
"""

from collections import defaultdict
from ctypes import c_void_p
from enum import IntEnum
from inspect import signature
import os
from struct import pack
import sys
from time import perf_counter
import traceback
from types import ModuleType, SimpleNamespace
from weakref import finalize

from numpy import (
    asarray as np_asarray,
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


class _SimulatedDriver:
    """The ``cuda.bindings.driver`` calls cubie makes, on host memory."""

    class CUresult(IntEnum):
        CUDA_SUCCESS = 0
        CUDA_ERROR_INVALID_VALUE = 1
        CUDA_ERROR_OUT_OF_MEMORY = 2

    class CUmemPool_attribute(IntEnum):
        CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 0
        CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 1

    class CUpointer_attribute(IntEnum):
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 0

    class CUmemorytype(IntEnum):
        CU_MEMORYTYPE_HOST = 1

    class CUfunction_attribute(IntEnum):
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 0

    CUdevice = int

    def __init__(self) -> None:
        # Allocated host bytes until a MemoryPointer takes them.
        self.allocations = {}

    def cuMemAllocAsync(self, nbytes, stream):
        buffer = np_empty(nbytes, dtype=np_uint8)
        address = buffer.ctypes.data
        self.allocations[address] = buffer
        return self.CUresult.CUDA_SUCCESS, address

    def cuMemFreeAsync(self, address, stream):
        return (self.CUresult.CUDA_SUCCESS,)

    def cuDeviceGetDefaultMemPool(self, device):
        return self.CUresult.CUDA_SUCCESS, device

    def cuMemPoolGetAttribute(self, pool, attribute):
        return self.CUresult.CUDA_SUCCESS, 0

    def cuStreamQuery(self, stream):
        return (self.CUresult.CUDA_SUCCESS,)

    def cuPointerGetAttribute(self, attribute, pointer):
        return self.CUresult.CUDA_ERROR_INVALID_VALUE, 0

    def cuFuncSetAttribute(self, function, attribute, value):
        return (self.CUresult.CUDA_SUCCESS,)


class _MemoryPointer:
    """Owner of the host bytes behind a simulated device allocation."""

    def __init__(self, context, pointer, size, owner=None, finalizer=None):
        self.device_pointer = pointer
        self.size = size
        self.host_bytes = cuda_driver.allocations.pop(pointer.value)
        if finalizer is not None:
            finalize(self, finalizer)


class _Event:
    """Event that records the host time; simulated work is complete."""

    def __init__(self, timing=True):
        self.time = None

    def record(self, stream=0):
        self.time = perf_counter()

    def wait(self, stream=0):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, event):
        return (event.time - self.time) * 1000.0


def _event_elapsed_time(start, end):
    """Return the milliseconds between two recorded events."""
    return start.elapsed_time(end)


def _flat_bytes(array):
    """Return a contiguous array's bytes as a flat writable view."""
    return np_asarray(getattr(array, "_ary", array)).reshape(-1).view(
        np_uint8
    )


def _host_to_device(dst, src, size, stream=0):
    _flat_bytes(dst)[:size] = _flat_bytes(src)[:size]


def _device_to_host(dst, src, size, stream=0):
    _flat_bytes(dst)[:size] = _flat_bytes(src)[:size]


# An ELF64 image with no sections: a kernel without machine code.
_EMPTY_CUBIN = b"\x7fELF\x02" + bytes(53) + pack("<HHH", 64, 0, 0)


class _CodeLibrary:
    """Compiled code of a simulated kernel: none."""

    _cubin = _EMPTY_CUBIN

    def get_cufunc(self):
        return SimpleNamespace(handle=c_void_p(0))


_COMPILED = SimpleNamespace(signature=SimpleNamespace(args=()))
"""A simulated kernel's compile result: one empty signature."""


def _extend_simulator() -> None:  # pragma: no cover - simulated
    """Add cubie's CUDA API to the vendored simulator."""
    from cubie.vendored.cudasim import api, kernel, kernelapi
    from cubie.vendored.cudasim.cudadrv import devicearray, devices, driver
    from cubie.vendored.cudasim.cudadrv.devicearray import FakeCUDAArray

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

    fake_kernel = kernel.FakeCUDAKernel
    fake_kernel.targetoptions = property(
        lambda self: {"device": self._device}
    )
    def compile_for(self, *args):
        """Record the kernel's one overload and return it."""
        self.compiled_overloads = {(): kernel.FakeOverload()}
        return _COMPILED

    fake_kernel.compile_for = compile_for
    fake_kernel.overloads = property(
        lambda self: kernel.FakeOverloadDict(
            getattr(self, "compiled_overloads", {})
        )
    )
    fake_kernel.get_regs_per_thread = lambda self: defaultdict(int)
    fake_kernel.get_local_mem_per_thread = lambda self: defaultdict(int)
    kernel.FakeOverload._codelibrary = _CodeLibrary()

    context = devices.FakeCUDAContext
    context.get_memory_info = (
        lambda self: devices._MemoryInfo(1024**3, 8 * 1024**3)
    )
    context.get_active_blocks_per_multiprocessor = (
        lambda self, function, blocksize, dynamic_shared: 1
    )
    context.memory_manager = SimpleNamespace(deallocations=[])
    context.memhostalloc = (
        lambda self, size, mapped=False, portable=False, wc=False:
        bytearray(size)
    )

    device = devices.FakeCUDADevice
    device.id = 0
    device.MULTIPROCESSOR_COUNT = 1
    device.L2_CACHE_SIZE = 0
    device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 49152
    device.RESERVED_SHARED_MEMORY_PER_BLOCK = 0
    device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 49152
    device.MAX_REGISTERS_PER_MULTIPROCESSOR = 65536
    device.MAX_THREADS_PER_MULTIPROCESSOR = 1024
    device.MAX_BLOCKS_PER_MULTIPROCESSOR = 16
    device.WARP_SIZE = 32

    array_init = FakeCUDAArray.__init__

    def device_array_init(self, *args, gpu_data=None, **kwargs):
        """Wrap an array, or build one over ``gpu_data``'s bytes."""
        if gpu_data is None:
            array_init(self, *args, **kwargs)
            return
        shape, strides, dtype = args
        array_init(
            self,
            np_ndarray(
                shape, dtype, buffer=gpu_data.host_bytes, strides=strides
            ),
            kwargs.get("stream", 0),
        )
        self.gpu_data = gpu_data

    FakeCUDAArray.__init__ = device_array_init

    api.stream.handle = c_void_p(0)
    driver.Stream = api.stream
    driver.host_to_device = _host_to_device
    driver.device_to_host = _device_to_host
    devicearray.DeviceNDArrayBase = FakeCUDAArray
    devicearray.DeviceNDArray = FakeCUDAArray
    devicearray.MappedNDArray = FakeCUDAArray

    experimental = ModuleType(f"{cuda.__name__}.experimental")
    experimental.consteval = lambda value: value

    for module in (api, cuda):
        module.jit = jit
        module.stwt = stwt
        module.event = _Event
    cuda.Event = _Event
    cuda.event_elapsed_time = _event_elapsed_time
    cuda.MemoryPointer = _MemoryPointer
    cuda.experimental = experimental


if CUDA_SIMULATION:  # pragma: no cover - simulated
    cuda_driver = _SimulatedDriver()
    _extend_simulator()
else:  # pragma: no cover - exercised in GPU environments
    from cuda.bindings import driver as cuda_driver

Stream = cuda.cudadrv.driver.Stream
DeviceNDArrayBase = cuda.devicearray.DeviceNDArrayBase
DeviceNDArray = cuda.devicearray.DeviceNDArray
MappedNDArray = cuda.devicearray.MappedNDArray
CudaSupportError = cuda.cudadrv.error.CudaSupportError


__all__ = [
    "CUDA_SIMULATION",
    "CudaSupportError",
    "cuda",
    "cuda_driver",
    "DeviceNDArray",
    "DeviceNDArrayBase",
    "fmax",
    "fmin",
    "MappedNDArray",
    "Stream",
]
