"""CUDA import hub: every CUDA symbol CuBIE uses comes from here.

``cuda`` is ``numba_cuda_mlir.cuda``, or :mod:`cubie.vendored.cudasim`
under ``NUMBA_ENABLE_CUDASIM=1``. Only the CuPy hooks and
``fmax``/``fmin`` differ under the simulator.

Published Functions
-------------------
:func:`from_dtype`
    Return the Numba type of a NumPy dtype.
:func:`is_devfunc`
    Test whether a callable is a CUDA device function.
:func:`is_device_array`
    Check whether a value is a GPU-resident array (not host numpy).
:func:`is_cudasim_enabled`
    Return whether the CUDA simulator is active.
:func:`get_jit_kwargs`
    Render a ``JITFlags`` to ``cuda.jit`` keyword arguments.

Published Device Functions
--------------------------
``selp``, ``activemask``, ``all_sync``, ``any_sync``, ``syncwarp``
    Wrappers around the CUDA intrinsics.
``stwt``
    The backend's store write-through hint.
``consteval``: compile-time loop marker; MLIR unrolls it.
``unroll_if``: ``unroll_if(range(n), flag[, count])``; ``flag`` sets
    whether MLIR adds a loop-unroll hint, ``count`` its unroll count.
:data:`UnrollFlag`: one ``(unroll, count)`` pair, the ``flag`` argument.

Published Constants
-------------------
:data:`CUDA_SIMULATION`
    ``True`` when ``NUMBA_ENABLE_CUDASIM=1``.
:data:`compile_kwargs`
    Default keyword arguments for ``@cuda.jit`` decorators.
:data:`JIT_FLAG_DEFAULTS`
    Default value of every managed jit flag except ``lineinfo``.
:data:`cuda`, :data:`int32`, :data:`float32`, :data:`float64`,
:data:`bool_`
    The ``cuda`` module object and scalar types.

See Also
--------
:mod:`cubie.vendored.cudasim`
    The CUDA simulator.
:mod:`cubie._utils`
    Imports ``compile_kwargs`` and ``is_devfunc`` from this module.
:mod:`cubie.memory.mem_manager`
    Uses ``Stream``, ``current_mem_info`` and the driver memory
    helpers exported here.
"""

from ctypes import c_void_p
import os
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from numpy import (
    fmax as np_fmax,
    fmin as np_fmin,
    ndarray as np_ndarray,
    uint8,
)

from numba_cuda_mlir.types import (
    boolean as bool_,
    float32,
    float64,
    int32,
)
from numba_cuda_mlir.numba_cuda.np.numpy_support import from_dtype
from numba_cuda_mlir.caching import (
    MLIRCache as CUDACache,
    MLIRCacheImpl as CacheImpl,
)
from numba_cuda_mlir.numba_cuda.core.caching import (  # noqa: F401
    _CacheLocator,
    IndexDataCacheFile,
)
from numba_cuda_mlir.numba_cuda import types as numba_types

from cubie._env import lineinfo_default


CUDA_SIMULATION: bool = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

if CUDA_SIMULATION:  # pragma: no cover - simulated
    from cubie.vendored import cudasim as cuda
    from cubie.vendored.cudasim.experimental import consteval

else:  # pragma: no cover - exercised in GPU environments
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.cuda.experimental import consteval


Stream = cuda.cudadrv.driver.Stream
DeviceNDArrayBase = cuda.cudadrv.devicearray.DeviceNDArrayBase
DeviceNDArray = cuda.cudadrv.devicearray.DeviceNDArray
MappedNDArray = cuda.cudadrv.devicearray.MappedNDArray
CudaSupportError = cuda.cudadrv.error.CudaSupportError


UnrollFlag = Tuple[bool, Optional[int]]
"""Loop-group flag: ``(unroll, count)``."""


# MLIR jit options carried by every compile.
_BACKEND_JIT_OPTIONS: Mapping[str, Any] = MappingProxyType(
    {"experimental_ast_transforms": True}
)

JIT_FLAG_DEFAULTS: Mapping[str, bool] = MappingProxyType(
    {
        "nsz": True,
        "contract": True,
        "arcp": True,
        "afn": True,
        "ftz": True,
        "lto": True,
    }
)
"""Default ``cuda.jit`` flags; ``lineinfo`` follows ``lineinfo_default``."""


def _render_jit_kwargs(lineinfo: bool) -> dict[str, Any]:
    """Return the default jit kwargs with ``lineinfo`` set."""
    return {
        "fastmath": {
            name for name, on in JIT_FLAG_DEFAULTS.items()
            if on and name != "lto"
        },
        "lineinfo": lineinfo,
        "lto": JIT_FLAG_DEFAULTS["lto"],
        **_BACKEND_JIT_OPTIONS,
    }


# Defaults for import-time device functions; factory builds use get_jit_kwargs.
compile_kwargs: Mapping[str, Any] = MappingProxyType(
    _render_jit_kwargs(lineinfo_default())
)


def get_jit_kwargs(
    jit_flags: Optional[Union[Any, bool]] = None,
) -> dict[str, Any]:
    """Return per-build ``cuda.jit`` keyword arguments.

    Parameters
    ----------
    jit_flags
        A ``JITFlags`` (any object with ``fastmath``, ``lineinfo`` and
        ``lto``), a bool as ``lineinfo`` over the defaults, or ``None``
        for the defaults.

    Returns
    -------
    dict
        ``fastmath``, ``lineinfo``, ``lto`` and
        ``experimental_ast_transforms`` keyword arguments.
    """
    if jit_flags is None:
        return _render_jit_kwargs(lineinfo_default())
    if isinstance(jit_flags, bool):
        return _render_jit_kwargs(jit_flags)
    return {
        "fastmath": jit_flags.fastmath,
        "lineinfo": jit_flags.lineinfo,
        "lto": jit_flags.lto,
        **_BACKEND_JIT_OPTIONS,
    }


def current_mem_info() -> Tuple[int, int]:
    """Return free and total memory from the active CUDA context."""
    return cuda.current_context().get_memory_info()


if CUDA_SIMULATION:  # pragma: no cover - simulated

    def page_locked_block(nbytes: int) -> Any:
        """Return a host buffer; the simulator has no pinning."""
        return bytearray(nbytes)

    def stream_ordered_buffer(nbytes: int, stream: Any) -> Any:
        """Return a flat byte device array."""
        return cuda.device_array(nbytes, dtype=uint8)

    def pool_idle_bytes() -> int:
        """Return zero; the simulator has no device pool."""
        return 0

    def stream_idle(stream: Any) -> bool:
        """Return ``True``; simulated work completes on launch."""
        return True

    def flush_deferred_frees() -> None:
        """Do nothing; the simulator defers no frees."""

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
        return DeviceNDArray(
            (nbytes,), (1,), uint8, stream=stream, gpu_data=memory
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


def is_device_array(value: Any) -> bool:
    """Check whether ``value`` is a GPU-resident array.

    Parameters
    ----------
    value
        Object to test.

    Returns
    -------
    bool
        ``True`` for device arrays and objects exposing
        ``__cuda_array_interface__``; ``False`` for host arrays.
    """
    if value is None or isinstance(value, np_ndarray):
        return False
    if isinstance(value, DeviceNDArrayBase):
        return True
    return hasattr(value, "__cuda_array_interface__")


def is_pinned_array(array: Any) -> bool:
    """Return whether a host array is backed by page-locked memory.

    Asks the driver about the array's first byte. Always ``False``
    under the CUDA simulator, which has no page-locked memory.
    """
    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return False
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


def is_devfunc(func: Callable[..., Any]) -> bool:
    """Test whether ``func`` represents a CUDA device function.

    Parameters
    ----------
    func
        Callable object to inspect for CUDA device metadata.

    Returns
    -------
    bool
        ``True`` when ``func`` is tagged as a CUDA device function.
    """

    target_options = getattr(func, "targetoptions", None)
    if isinstance(target_options, dict):
        return bool(target_options.get("device", False))
    return False


def devfunc_returns_nonfloat(func: Callable[..., Any]) -> bool:
    """Report whether every compiled overload returns int or bool.

    Parameters
    ----------
    func
        Callable object to inspect for compiled device overloads.

    Returns
    -------
    bool
        ``True`` when every overload returns an integer or boolean.
    """

    overloads = getattr(func, "overloads", None)
    if not overloads:
        return False
    return all(
        isinstance(
            overload.signature.return_type,
            (numba_types.Integer, numba_types.Boolean),
        )
        for overload in overloads.values()
    )


# Device max/min drop a NaN operand; the simulator needs numpy's fmax/fmin.
fmax = np_fmax if CUDA_SIMULATION else max
fmin = np_fmin if CUDA_SIMULATION else min


# no cover: start
@cuda.jit(
    device=True,
    inline=True,
    **compile_kwargs,
)
def selp(pred, true_value, false_value):
    return cuda.selp(pred, true_value, false_value)


@cuda.jit(
    device=True,
    inline=True,
    **compile_kwargs,
)
def activemask():
    return cuda.activemask()


@cuda.jit(
    device=True,
    inline=True,
    **compile_kwargs,
)
def all_sync(mask, predicate):
    return cuda.all_sync(mask, predicate)


@cuda.jit(
    device=True,
    inline=True,
    **compile_kwargs,
)
def any_sync(mask, predicate):
    return cuda.any_sync(mask, predicate)


@cuda.jit(
    device=True,
    inline=True,
    **compile_kwargs,
)
def syncwarp(mask):
    return cuda.syncwarp(mask)
# no cover: end


stwt = cuda.stwt


def unroll_if(iterable, flag, count=None):
    """Return ``iterable``; the UnrollIf pass consumes the call."""
    return iterable


def is_cudasim_enabled() -> bool:
    """Return ``True`` when running under the CUDA simulator."""

    return CUDA_SIMULATION


__all__ = [
    "activemask",
    "all_sync",
    "any_sync",
    "bool_",
    "CacheImpl",
    "compile_kwargs",
    "JIT_FLAG_DEFAULTS",
    "consteval",
    "cuda",
    "get_jit_kwargs",
    "IndexDataCacheFile",
    "CUDA_SIMULATION",
    "CUDACache",
    "current_mem_info",
    "DeviceNDArray",
    "DeviceNDArrayBase",
    "flush_deferred_frees",
    "float32",
    "float64",
    "from_dtype",
    "int32",
    "is_cudasim_enabled",
    "is_device_array",
    "is_pinned_array",
    "is_devfunc",
    "MappedNDArray",
    "selp",
    "fmax",
    "fmin",
    "Stream",
    "page_locked_block",
    "pool_idle_bytes",
    "stream_idle",
    "stream_ordered_buffer",
    "stwt",
    "syncwarp",
    "unroll_if",
    "UnrollFlag",
]
