"""Shared CUDA import hub and simulation-safe helpers.

This module is the single surface through which the rest of CuBIE
reaches its CUDA backend. The active backend (``numba-cuda`` or
``numba-cuda-mlir``) is resolved by :mod:`cubie.cuda_backend`; the
``cuda`` module object, scalar types, ``from_dtype``, driver
internals, and cache base classes are all re-exported here so no
other module imports a backend package directly.

Under ``numba-cuda`` the module also centralises the CUDA-simulator
(``NUMBA_ENABLE_CUDASIM=1``) stand-ins. numba-cuda-mlir has no
simulator — backend resolution prefers numba-cuda when the simulator
is requested, so the MLIR backend only reaches this module with
``NUMBA_ENABLE_CUDASIM=1`` when it was explicitly selected or is the
only backend installed, and that combination raises at import.

Published Functions
-------------------
:func:`from_dtype`
    Return a CUDA-ready or simulator-safe dtype.
:func:`is_devfunc`
    Test whether a callable is a CUDA device function.
:func:`is_cuda_array`
    Check whether a value should be treated as a CUDA array.
:func:`is_device_array`
    Check whether a value is a GPU-resident array (not host numpy).
:func:`is_cudasim_enabled`
    Return whether the CUDA simulator is active.
:func:`get_jit_kwargs`
    Render a :class:`JITFlags` to ``cuda.jit`` keyword arguments.

Published Device Functions
--------------------------
``selp``, ``activemask``, ``all_sync``, ``any_sync``, ``syncwarp``
    Wrappers around CUDA intrinsics with CUDASIM fallbacks.
``stwt``
    The backend's store write-through hint, re-exported directly on
    a real GPU with a CUDASIM fallback.
``narrow_f64``: narrow float64 to float32 without subnormal flushing.

Published Classes
-----------------
:class:`JITFlags`
    Managed ``cuda.jit`` compile options stored on every factory's
    compile settings and rendered to decorator kwargs by
    :func:`get_jit_kwargs`.

Published Constants
-------------------
:data:`CUDA_SIMULATION`
    ``True`` when ``NUMBA_ENABLE_CUDASIM=1`` (never true under the
    MLIR backend).
:data:`compile_kwargs`
    Default keyword arguments for ``@cuda.jit`` decorators.
:data:`INLINE_ALWAYS`
    Backend-correct value for the ``cuda.jit`` ``inline`` argument
    (``"always"`` on numba-cuda, ``True`` on numba-cuda-mlir).
:data:`cuda`, :data:`int32`, :data:`float32`, :data:`float64`,
:data:`bool_`
    The backend's ``cuda`` module object and scalar types.

See Also
--------
:mod:`cubie.cuda_backend`
    Backend resolution (installed packages + ``CUBIE_CUDA_BACKEND``).
:mod:`cubie._utils`
    Imports ``compile_kwargs`` and ``is_devfunc`` from this module.
:mod:`cubie.memory.mem_manager`
    Uses the ``Stream`` stand-in, ``current_mem_info``, and the
    ``cupy``/``cupyx`` imports exported here. This module owns the
    single conditional import of ``cupy``/``cupyx``: both are
    imported eagerly on a real GPU (CuPy is CuBIE's device
    allocation provider, so it is a hard requirement there) and are
    ``None`` under the CUDA simulator, which never touches device
    memory. Consumers import them from here rather than importing
    CuPy directly.
"""

from __future__ import annotations

from ctypes import c_void_p
import os
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from attrs import Factory, field, frozen
from attrs import evolve as attrs_evolve
from attrs import validators as attrs_validators
from numpy import dtype, empty as np_empty, ndarray as np_ndarray

from cubie.cuda_backend import IS_MLIR
from cubie._env import lineinfo_default


CUDA_SIMULATION: bool = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

if IS_MLIR and CUDA_SIMULATION:
    raise ImportError(
        "NUMBA_ENABLE_CUDASIM=1 is set, but numba-cuda-mlir has no "
        "CUDA simulator. Unset the variable and run on a real GPU, "
        "or install numba-cuda (and leave CUBIE_CUDA_BACKEND unset "
        "or set to 'numba-cuda') for simulator work."
    )

if IS_MLIR:
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.types import (
        boolean as bool_,
        float32,
        float64,
        int32,
    )
    from numba_cuda_mlir.numba_cuda.np.numpy_support import (
        from_dtype as numba_from_dtype,
    )
    from numba_cuda_mlir.caching import (
        MLIRCache as CUDACache,
        MLIRCacheImpl as CacheImpl,
    )
    from numba_cuda_mlir.numba_cuda.core.caching import (  # noqa: F401
        _CacheLocator,
        IndexDataCacheFile,
    )
    from numba_cuda_mlir.numba_cuda.cudadrv.error import (  # noqa: F401
        CudaSupportError,
    )

    # The MLIR backend accepts a boolean cuda.jit inline argument;
    # numba-cuda takes the string form and deprecates the boolean.
    INLINE_ALWAYS: Union[str, bool] = True
else:
    from numba import cuda
    from numba import bool_, float32, float64, int32
    from numba import from_dtype as numba_from_dtype
    from numba.cuda.core.caching import (  # noqa: F401
        _CacheLocator,
        CacheImpl,
        IndexDataCacheFile,
    )
    from numba.cuda.cudadrv.error import (  # noqa: F401
        CudaSupportError,
    )

    INLINE_ALWAYS = "always"


@frozen
class JITFlags:
    """Per-factory ``cuda.jit`` compile flags.

    Every managed jit option travels the same path: stored on the
    factory's compile settings (hashed into the config, so a change
    triggers a rebuild), then rendered to decorator keyword arguments
    by :func:`get_jit_kwargs`. New jit options are added here as new
    fields. Instances are immutable snapshots; :meth:`update` derives
    a replacement rather than mutating in place.

    Attributes
    ----------
    lineinfo
        Compile with source-line correlation data. Defaults to the
        ``CUBIE_LINEINFO`` environment variable.
    nsz
        Treat signed zero as insignificant in floating-point ops.
    contract
        Allow floating-point contraction (fused multiply-add).
    arcp
        Allow reciprocal approximation of division.
    afn
        Allow approximate transcendental functions (``LG2``/``EX2``
        hardware paths for ``log``/``exp``/``pow``).
    ftz
        Flush denormal float results and inputs to zero.
    lto
        Enable link-time optimisation across device functions.
    """

    lineinfo: bool = field(
        default=Factory(lineinfo_default),
        validator=attrs_validators.instance_of(bool),
    )
    nsz: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )
    contract: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )
    arcp: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )
    afn: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )
    ftz: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )
    lto: bool = field(
        default=True, validator=attrs_validators.instance_of(bool)
    )

    @property
    def fastmath(self) -> set:
        """Return the set of enabled LLVM fast-math flag names."""
        enabled = {
            "nsz": self.nsz,
            "contract": self.contract,
            "arcp": self.arcp,
            "afn": self.afn,
            "ftz": self.ftz,
        }
        return {name for name, on in enabled.items() if on}

    def update(self, updates_dict=None, **kwargs):
        """Derive a replacement snapshot with new flag values.

        Parameters
        ----------
        updates_dict
            Mapping of flag names to new boolean values. Unknown keys
            are ignored so composite configs can broadcast one updates
            dict to every nested attrs class.
        **kwargs
            Additional flag updates.

        Returns
        -------
        tuple[JITFlags, set[str], set[str]]
            Replacement snapshot (``self`` when unchanged), names of
            recognised settings, and names of changed settings.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = {**updates_dict, **kwargs}
        recognized = set()
        changed = set()
        replacements = {}
        flag_names = {
            "lineinfo",
            "nsz",
            "contract",
            "arcp",
            "afn",
            "ftz",
            "lto",
        }
        for key, value in updates_dict.items():
            if key not in flag_names:
                continue
            recognized.add(key)
            if getattr(self, key) != value:
                replacements[key] = bool(value)
                changed.add(key)
        if not changed:
            return self, recognized, changed
        return attrs_evolve(self, **replacements), recognized, changed


# Defaults for import-time device functions; factory builds use get_jit_kwargs.
compile_kwargs: Mapping[str, Any] = MappingProxyType(
    {}
    if CUDA_SIMULATION
    else {
        "fastmath": JITFlags().fastmath,
        "lineinfo": lineinfo_default(),
        "lto": JITFlags().lto,
    }
)


def get_jit_kwargs(
    jit_flags: Optional[Union["JITFlags", bool]] = None,
) -> dict[str, Any]:
    """Return per-build ``cuda.jit`` keyword arguments.

    Parameters
    ----------
    jit_flags
        Flags for the build. A :class:`JITFlags` instance renders all
        of its fields; a bare boolean is accepted as the ``lineinfo``
        value with default fast-math flags (the form generated system
        modules use); ``None`` uses the default flag set.

    Returns
    -------
    dict
        ``{"fastmath": set, "lineinfo": bool, "lto": bool}``
        rendered from the flags. Under the CUDA simulator every
        GPU-only option is omitted and an empty dict is returned,
        regardless of the flags passed.
    """
    if CUDA_SIMULATION:
        return {}
    if jit_flags is None:
        jit_flags = JITFlags()
    elif isinstance(jit_flags, bool):
        jit_flags = JITFlags(lineinfo=jit_flags)
    return {
        "fastmath": jit_flags.fastmath,
        "lineinfo": jit_flags.lineinfo,
        "lto": jit_flags.lto,
    }


class FakeStream:  # pragma: no cover - placeholder
    """Placeholder CUDA stream."""

    handle = c_void_p(0)


class FakeMemoryInfo:  # pragma: no cover - placeholder
    """Container for fake memory statistics."""

    free = 1024**3
    total = 8 * 1024**3


if CUDA_SIMULATION:  # pragma: no cover - simulated
    from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray
    from cubie.vendored.numba_cuda_cache import CUDACache  # noqa: F811

    # The simulator never touches real device memory, so CuPy is not
    # required; code paths guarded by CUDA_SIMULATION never use these.
    cupy = None
    cupyx = None

    Stream = FakeStream
    DeviceNDArrayBase = FakeCUDAArray
    DeviceNDArray = FakeCUDAArray
    MappedNDArray = FakeCUDAArray

    def current_mem_info() -> Tuple[int, int]:
        """Return fake free and total memory values."""

        fakemem = FakeMemoryInfo()
        return fakemem.free, fakemem.total

    def empty_pinned(shape, dtype) -> np_ndarray:
        """Return a plain host array; the simulator has no pinning."""
        return np_empty(shape, dtype=dtype)

    def free_all_pinned_blocks() -> None:
        """Do nothing; the simulator has no pinned-memory pool."""

else:  # pragma: no cover - exercised in GPU environments
    try:
        import cupy
        import cupyx
    except ImportError as e:
        raise ImportError(
            "CuPy is required for CuBIE's device memory allocations "
            "on a real GPU. Install it via the cuda12/cuda13 or "
            "mlir-cuda12/mlir-cuda13 extra, or pip install "
            "cupy-cuda12x directly (assuming CUDA toolkit 12.x)."
        ) from e

    if IS_MLIR:
        from numba_cuda_mlir.cuda import (
            is_cuda_array as _is_cuda_array,
        )
        from numba_cuda_mlir.numba_cuda.cudadrv.driver import (
            Stream,
        )
        from numba_cuda_mlir.numba_cuda.cudadrv.devicearray import (
            DeviceNDArrayBase,
            DeviceNDArray,
            MappedNDArray,
        )
    else:
        from numba.cuda import (  # type: ignore[attr-defined]
            is_cuda_array as _is_cuda_array,
        )
        from numba.cuda.cudadrv.driver import (  # type: ignore[attr-defined]
            Stream,
        )
        from numba.cuda.cudadrv.devicearray import (  # type: ignore
            DeviceNDArrayBase,
            DeviceNDArray,
            MappedNDArray,
        )
        # Linter can't find cuda.dispatcher.
        from numba.cuda.dispatcher import CUDACache  # noqa: F401,F811

    def current_mem_info() -> Tuple[int, int]:
        """Return free and total memory from the active CUDA context."""

        return cuda.current_context().get_memory_info()

    def empty_pinned(shape, dtype) -> np_ndarray:
        """Return a page-locked host array from CuPy's pinned pool."""
        return cupyx.empty_pinned(shape, dtype=dtype)

    def free_all_pinned_blocks() -> None:
        """Release the page-locked blocks CuPy's pinned pool holds."""
        cupy.get_default_pinned_memory_pool().free_all_blocks()


def is_cuda_array(value: Any) -> bool:
    """Check whether ``value`` should be treated as a CUDA array."""

    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return hasattr(value, "shape")
    return _is_cuda_array(value)


def is_device_array(value: Any) -> bool:
    """Check whether ``value`` is a GPU-resident array.

    Parameters
    ----------
    value
        Object to test.

    Returns
    -------
    bool
        ``True`` for device arrays (Numba device arrays, CuPy arrays,
        or any non-numpy object exposing
        ``__cuda_array_interface__``; the simulator's fake device
        arrays under CUDASIM), ``False`` for host numpy arrays and
        everything else.

    Notes
    -----
    Unlike :func:`is_cuda_array`, this is never truthy for host numpy
    arrays, so it distinguishes device inputs under the simulator.
    """
    if value is None or isinstance(value, np_ndarray):
        return False
    if isinstance(value, DeviceNDArrayBase):
        return True
    return hasattr(value, "__cuda_array_interface__")


def is_pinned_array(array: Any) -> bool:
    """Return whether a host array is backed by page-locked memory.

    Walks the view chain to the owning object and checks for the
    CuPy pinned-pool pointer, which backs every pinned allocation
    CuBIE makes. Always ``False`` under the CUDA simulator, which
    has no page-locked memory.
    """
    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return False
    base = array
    while isinstance(base, np_ndarray):
        base = base.base
    return isinstance(base, cupy.cuda.PinnedMemoryPointer)


def from_dtype(dt: dtype):
    """Return a CUDA-ready dtype or a simulator-safe placeholder.

    Parameters
    ----------
    dt
        NumPy dtype to adapt for use with CUDA or the simulator.

    Returns
    -------
    dtype
        A Numba CUDA-compatible dtype when running on a real GPU, or
        the original dtype unchanged when running in CUDA simulation
        mode.
    """

    if not CUDA_SIMULATION:
        return numba_from_dtype(dt)
    return dt  # pragma: no cover - simulated


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

    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return bool(getattr(func, "_device", False))
    target_options = getattr(func, "targetoptions", None)
    if isinstance(target_options, dict):
        return bool(target_options.get("device", False))
    return False


if CUDA_SIMULATION:  # pragma: no cover - simulated

    # no cover: start
    @cuda.jit(
        device=True,
        inline=True,
    )
    def selp(pred, true_value, false_value):
        """Select ``true_value`` or ``false_value`` based on predicate.

        Parameters
        ----------
        pred : bool
            Condition to evaluate.
        true_value : numba scalar
            Value returned when ``pred`` is true.
        false_value : numba scalar
            Value returned when ``pred`` is false.

        Returns
        -------
        numba scalar
            Selected value.
        """
        return true_value if pred else false_value

    @cuda.jit(
        device=True,
        inline=True,
    )
    def activemask():
        """Return the active thread mask for the current warp.

        Returns
        -------
        int32
            Bitmask of active threads (all-ones in CUDASIM).
        """
        return 0xFFFFFFFF

    @cuda.jit(
        device=True,
        inline=True,
    )
    def all_sync(mask, predicate):
        """Return whether all threads in ``mask`` satisfy ``predicate``.

        Parameters
        ----------
        mask : int32
            Active thread mask.
        predicate : bool
            Per-thread condition.

        Returns
        -------
        bool
            ``True`` if all masked threads satisfy ``predicate``.
        """
        return predicate

    @cuda.jit(
        device=True,
        inline=True,
    )
    def any_sync(mask, predicate):
        """Return whether any thread in ``mask`` satisfies ``predicate``.

        Parameters
        ----------
        mask : int32
            Active thread mask.
        predicate : bool
            Per-thread condition.

        Returns
        -------
        bool
            ``True`` if any masked threads satisfy ``predicate``.
        """
        return predicate

    @cuda.jit(
        device=True,
        inline=True,
    )
    def syncwarp(mask):
        """Synchronise threads within a warp.

        Parameters
        ----------
        mask : int32
            Active thread mask.
        """
        pass

    @cuda.jit(
        device=True,
        inline=True,
    )
    def stwt(array, index, value):
        """Store-through write: write ``value`` to ``array[index]``.

        Parameters
        ----------
        array : device array
            Target array.
        index : int32
            Element index.
        value : numba scalar
            Value to write.
        """
        array[index] = value

    @cuda.jit(
        device=True,
        inline=True,
    )
    def narrow_f64(value):
        """Narrow float64 to float32 without subnormal flushing."""
        return float32(value)

    # no cover: end

else:  # pragma: no cover - relies on GPU runtime

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

    if IS_MLIR:
        from cubie.backend._mlir_intrinsics import narrow_f64
    else:

        @cuda.jit(
            device=True,
            inline=True,
            **compile_kwargs,
        )
        def narrow_f64(value):
            """Narrow float64 to float32 without subnormal flushing."""
            return float32(value)


def is_cudasim_enabled() -> bool:
    """Return ``True`` when running under the CUDA simulator."""

    return CUDA_SIMULATION


def compute_capability_code() -> Optional[str]:
    """Return the current device's compute capability as ``"M.m"``.

    Returns
    -------
    str or None
        Architecture code such as ``"8.9"``, or ``None`` under
        CUDASIM, where no physical architecture exists.
    """
    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return None
    major, minor = cuda.get_current_device().compute_capability
    return f"{major}.{minor}"


def max_shared_memory_per_block() -> int:
    """Return the device's dynamic shared-memory limit per block.

    Returns
    -------
    int
        Per-block shared-memory limit in bytes. numba-cuda does not
        set ``CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES``, so
        the default (non-opt-in) device limit applies to every
        launch. Under CUDASIM the ubiquitous 48 kiB default is
        returned.
    """
    if CUDA_SIMULATION:  # pragma: no cover - simulated
        return 49152
    return int(
        cuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
    )


__all__ = [
    "activemask",
    "all_sync",
    "any_sync",
    "bool_",
    "CacheImpl",
    "compile_kwargs",
    "cuda",
    "compute_capability_code",
    "get_jit_kwargs",
    "IndexDataCacheFile",
    "INLINE_ALWAYS",
    "JITFlags",
    "CUDA_SIMULATION",
    "CUDACache",
    "cupy",
    "cupyx",
    "current_mem_info",
    "DeviceNDArray",
    "DeviceNDArrayBase",
    "empty_pinned",
    "FakeMemoryInfo",
    "FakeStream",
    "free_all_pinned_blocks",
    "float32",
    "float64",
    "from_dtype",
    "int32",
    "is_cuda_array",
    "is_cudasim_enabled",
    "is_device_array",
    "is_pinned_array",
    "max_shared_memory_per_block",
    "is_devfunc",
    "MappedNDArray",
    "selp",
    "Stream",
    "narrow_f64",
    "stwt",
    "syncwarp",
]
