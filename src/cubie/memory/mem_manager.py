"""GPU memory management utilities for coordinating CuBIE allocations.

This module provides the :class:`MemoryManager` singleton that
coordinates GPU memory allocation, stream usage, and automatic
chunking across registered instances.

Published Classes
-----------------
:class:`MemoryManager`
    Singleton interface coordinating GPU memory allocation and stream
    usage.

    >>> mgr = MemoryManager()

:class:`InstanceMemorySettings`
    Per-instance registry entry tracking allocations and hooks.

Published Constants
-------------------
:data:`ALL_MEMORY_MANAGER_PARAMETERS`
    Parameter set accepted by the memory manager configuration.

Module-Level Functions
----------------------
:func:`get_portioned_request_size`
    Calculate chunkable and unchunkable byte totals for a request
    dictionary.

:func:`is_request_chunkable`
    Determine whether a single :class:`ArrayRequest` can be chunked.

:func:`replace_with_chunked_size`
    Replace the run axis in a shape tuple with a chunked size.

Notes
-----
Chunking is performed along the run axis to handle batches that
exceed available GPU memory. The chunking process is automatic and
coordinated across all instances in a stream group.

See Also
--------
:class:`~cubie.memory.array_requests.ArrayRequest`
    Describes a single allocation request.
:class:`~cubie.memory.array_requests.ArrayResponse`
    Reports allocation outcomes including chunking metadata.
:class:`~cubie.memory.stream_groups.StreamGroups`
    Manages CUDA stream groups used by the memory manager.
"""

from tempfile import gettempdir, mkstemp
from threading import Lock
from types import TracebackType
from functools import partial
from typing import Any, Optional, Callable, Dict, Tuple, Union
from warnings import warn
from copy import deepcopy
from inspect import ismethod
from weakref import WeakMethod, finalize, ref as weakref_ref
import ctypes
import os
import sys

from cubie.cuda_simsafe import cuda
from cubie._utils import opt_getype_validator
from attrs import define, Factory as attrsFactory, field
from attrs.validators import (
    in_ as attrsval_in,
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)
from numpy import (
    ceil as np_ceil,
    dtype as np_dtype,
    memmap as np_memmap,
    ndarray,
    floor as np_floor,
    zeros as np_zeros,
)
from numpy.typing import DTypeLike
from math import prod

from cubie.cuda_simsafe import (
    CUDA_SIMULATION,
    Stream,
    cupy,
    current_mem_info,
    empty_pinned,
    free_all_pinned_blocks,
)
from cubie.memory.stream_groups import StreamGroups
from cubie.memory.array_requests import ArrayRequest, ArrayResponse


# Recognised configuration parameters for memory manager settings.
# These keys mirror the solver API so helpers can filter keyword
# arguments consistently.
ALL_MEMORY_MANAGER_PARAMETERS = {
    "memory_manager",
    "stream_group",
    "mem_proportion",
    "host_spill_threshold",
    "spill_directory",
}
"""Solver memory keyword names."""


MIN_AUTOPOOL_SIZE = 0.05

HOST_SPILL_FRACTION = 0.8
"""Fraction of total system RAM available to host arrays."""

HOST_STAGING_BYTES = 64 * 1024**2
"""Maximum pinned staging bytes used by one host transfer.

64 MiB is large enough to reach full PCIe transfer bandwidth while
bounding the pinned footprint of a chunked or spilled solve.
"""


if sys.platform == "win32":
    class _MemoryStatusEx(ctypes.Structure):
        """Layout of the Win32 ``MEMORYSTATUSEX`` structure."""

        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]


def _memory_status() -> "_MemoryStatusEx":
    """Return the filled Win32 ``MEMORYSTATUSEX`` structure."""
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
        ctypes.byref(status)
    ):
        raise ctypes.WinError()
    return status


def total_system_ram() -> int:
    """Return total physical RAM in bytes."""
    if sys.platform == "win32":
        return int(_memory_status().ullTotalPhys)
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def available_system_ram() -> int:
    """Return currently available physical RAM in bytes."""
    if sys.platform == "win32":
        return int(_memory_status().ullAvailPhys)
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def host_headroom_bytes() -> int:
    """Return available RAM above the OS reserve."""
    total = total_system_ram()
    available = available_system_ram()
    return available - int((1 - HOST_SPILL_FRACTION) * total)


def _remove_spill_file(mapping: Any, path: str) -> None:
    """Close and delete one spill file.

    Runs as a garbage-collection finalizer, where raising is not
    possible, so a filesystem failure is reported as a warning.
    """
    try:
        mapping.close()
        os.remove(path)
    except OSError as error:  # pragma: no cover - filesystem failure
        warn(f"Could not remove spill file '{path}': {error}", ResourceWarning)


def _spill_directory_converter(
    value: Optional[os.PathLike | str],
) -> Optional[str]:
    """Normalise a spill directory setting to a string path."""
    if value is None:
        return None
    return os.fspath(value)


def _existing_directory_validator(instance, attribute, value) -> None:
    """Reject spill directories that do not exist."""
    if not os.path.isdir(value):
        raise ValueError(
            f"{attribute.name} must be an existing directory, "
            f"got '{value}'"
        )


def placeholder_invalidate() -> None:
    """
    Default invalidate hook placeholder that performs no operations.

    """
    pass


def placeholder_dataready(response: ArrayResponse) -> None:
    """
    Default placeholder data ready hook that performs no operations.

    Parameters
    ----------
    response
        Array response object (unused).

    """
    pass


class _WeakCallable:
    """Callable wrapper that holds bound methods weakly.

    Registered instances supply bound methods as registry hooks; a
    strong reference to those methods would keep the instance alive
    for the lifetime of the registry. Bound methods are therefore
    stored through :class:`weakref.WeakMethod`, while plain functions
    are stored directly. Calling a wrapper whose referent has been
    garbage collected is a no-op.

    Parameters
    ----------
    func
        Callable to wrap. An existing wrapper is copied rather than
        double-wrapped.
    """

    def __init__(self, func: Callable) -> None:
        if isinstance(func, _WeakCallable):
            self._weak = func._weak
            self._strong = func._strong
        elif ismethod(func):
            self._weak = WeakMethod(func)
            self._strong = None
        else:
            self._weak = None
            self._strong = func

    def target(self) -> Optional[Callable]:
        """Return the wrapped callable, or None once collected."""
        if self._weak is not None:
            return self._weak()
        return self._strong

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target = self.target()
        if target is None:
            return None
        return target(*args, **kwargs)

    # Unhashable by design: equality tracks the referent, which can
    # be collected mid-lifetime, so no stable hash exists.
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _WeakCallable):
            return self.target() == other.target()
        return self.target() == other


def _ensure_cuda_context() -> None:
    """
    Ensure CUDA context is initialized before memory operations.

    This function validates that a CUDA context exists and is functional,
    triggering initialization if needed. If the context cannot be created
    or is in a bad state, it raises a clear exception rather than causing
    a segfault.

    Raises
    ------
    RuntimeError
        If CUDA context cannot be initialized or is not functional.
    """
    if not CUDA_SIMULATION:
        try:
            # Attempt to access current context - triggers creation if
            # needed.
            ctx = cuda.current_context()
            if ctx is None:
                raise RuntimeError(
                    "CUDA context is None - GPU may not be accessible"
                )
            # Skip memory info check for performance; context existence
            # is sufficient validation for most operations
        except Exception as e:
            # Provide helpful error message instead of segfault
            raise RuntimeError(
                f"Failed to initialize or verify CUDA context: {e}. "
                "This may indicate GPU driver issues, insufficient "
                "permissions, or the GPU may be in an unrecoverable "
                "state. Try restarting the process or checking GPU "
                "availability."
            ) from e


def _numba_stream_ptr(
    nb_stream: Optional[Stream],
) -> Optional[int]:
    """
    Extract a ``CUstream`` pointer from a Numba stream wrapper.

    Parameters
    ----------
    nb_stream
        Numba CUDA stream whose ``CUstream`` pointer should be extracted. When
        ``None``, pointer extraction is skipped.

    Returns
    -------
    int or None
        Pointer value compatible with CuPy external streams, or ``None`` when
        extraction fails.

    Notes
    -----
    The function checks common attribute layouts across supported Numba
    versions to maintain compatibility.
    """
    if nb_stream is None:
        return None
    h = getattr(nb_stream, "handle", None)
    if h is None:
        return None
    # ctypes.c_void_p or int-like
    if isinstance(h, ctypes.c_void_p):
        return int(h.value) if h.value is not None else None
    try:
        return int(getattr(h, "value", h))
    except Exception:
        return None


class current_cupy_stream:
    """Context manager that forwards a Numba stream into CuPy APIs.

    CuPy is CuBIE's single GPU allocation provider on a real device.
    Wrapping allocations and host/device copies in this context keeps
    them ordered on the same stream as the Numba-launched integration
    kernel.

    Parameters
    ----------
    nb_stream
        Numba CUDA stream to expose to CuPy.

    Attributes
    ----------
    nb_stream
        The Numba stream being forwarded.
    cupy_ext_stream
        CuPy external stream wrapper around the Numba stream.

    Notes
    -----
    Numba's default stream (handle ``0``) is left as CuPy's ambient
    current stream rather than wrapped, matching Numba's own default
    stream semantics.
    """

    def __init__(self, nb_stream: Stream) -> None:
        self.nb_stream = nb_stream
        self.cupy_ext_stream = None

    def __enter__(self) -> "current_cupy_stream":
        """
        Enter the context and set up a CuPy external stream.

        Returns
        -------
        current_cupy_stream
            The active context manager instance.
        """
        ptr = _numba_stream_ptr(self.nb_stream)
        if ptr:
            # Numba streams implement the __cuda_stream__ protocol, so
            # from_external wraps the stream object directly.
            self.cupy_ext_stream = cupy.cuda.Stream.from_external(
                self.nb_stream
            )
            self.cupy_ext_stream.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context and clean up the CuPy external stream.

        Parameters
        ----------
        exc_type
            Exception type if an exception occurred.
        exc
            Exception instance if an exception occurred.
        tb
            Traceback object if an exception occurred.

        Returns
        -------
        Optional[bool]
            The inner stream's suppression decision, or ``None``
            when no external stream is active.
        """
        if self.cupy_ext_stream is not None:
            result = self.cupy_ext_stream.__exit__(exc_type, exc, tb)
            self.cupy_ext_stream = None
            return result
        return None


# These will be keys to a dict, so must be hashable: eq=False
@define(eq=False)
class InstanceMemorySettings:
    """
    Memory registry information for a registered instance.

    Parameters
    ----------
    proportion
        Proportion of total VRAM assigned to this instance.
    allocations
        Dictionary of current allocations keyed by label.
    invalidate_hook
        Function to call when CUDA memory system changes occur.
    allocation_ready_hook
        Function to call when allocations are ready.
    cap
        Maximum allocatable bytes for this instance.
    instance_ref
        Weak reference to the registered instance, used to detect
        and purge entries whose instance has been collected.

    Attributes
    ----------
    proportion : float
        Proportion of total VRAM assigned to this instance.
    allocations : dict
        Dictionary of current allocations keyed by array label.
    invalidate_hook : callable
        Function to call when CUDA memory system changes.
    allocation_ready_hook : callable
        Function to call when allocations are ready.
    cap : int or None
        Maximum allocatable bytes for this instance.
    instance_ref : weakref.ref or None
        Weak reference to the registered instance.

    Properties
    ----------
    allocated_bytes : int
        Total number of bytes across all allocated arrays for the instance.

    Notes
    -----
    The allocations dictionary serves both as a "keepalive" reference and a way
    to calculate total allocated memory. The invalidate_hook is called when the
    allocator/memory manager changes, requiring arrays and kernels to be
    re-allocated or redefined.

    Bound-method hooks are stored weakly so the registry never keeps
    a registered instance alive; once the instance is collected the
    hooks become no-ops and the entry is purged by the manager.
    """

    proportion: float = field(
        default=1.0, validator=attrsval_instance_of(float)
    )
    allocations: dict = field(
        default=attrsFactory(dict), validator=attrsval_instance_of(dict)
    )
    invalidate_hook: Callable[[], None] = field(
        default=placeholder_invalidate,
        converter=_WeakCallable,
        validator=attrsval_instance_of(Callable),
    )
    allocation_ready_hook: Callable[[ArrayResponse], None] = field(
        default=placeholder_dataready,
        converter=_WeakCallable,
    )
    cap: Optional[int] = field(
        default=None, validator=attrsval_optional(attrsval_instance_of(int))
    )
    instance_ref: Optional[weakref_ref] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(weakref_ref)),
    )
    last_stream: Optional[Any] = field(default=None)
    submitting: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )
    completion_event: Optional[Any] = field(default=None)
    # register() always assigns an owner id; registrations sharing one
    # owner id are evicted together or not at all.
    owner_id: Optional[int] = field(default=None)
    last_used: int = field(default=0, validator=attrsval_instance_of(int))
    # Bytes above which this instance's host arrays spill to disk;
    # None defers to the manager-wide policy.
    host_spill_threshold: Optional[int] = field(
        default=None,
        validator=opt_getype_validator(int, 0),
    )
    # Directory for this instance's spill files; None defers to the
    # manager-wide setting.
    spill_directory: Optional[str] = field(
        default=None,
        converter=_spill_directory_converter,
        validator=attrsval_optional(_existing_directory_validator),
    )

    def add_allocation(self, key: str, arr: Any) -> None:
        """Add an allocation to the instance's allocations list.

        Parameters
        ----------
        key
            Label for the allocation.
        arr
            Allocated array object.

        Notes
        -----
        If a previous allocation exists with the same key, it is
        freed before adding the new allocation.
        """

        if key in self.allocations:
            # Free the old allocation before adding the new one
            self.free(key)
        self.allocations[key] = arr

    def free(self, key: str) -> None:
        """Free an allocation by key.

        Parameters
        ----------
        key
            Label of the allocation to free.

        Notes
        -----
        Emits a warning if the key is not found in allocations.
        """
        if key in self.allocations:
            del self.allocations[key]
        else:
            warn(
                f"Attempted to free allocation for {key}, but "
                f"it was not found in the allocations list."
            )

    def free_all(self) -> None:
        """Release allocations on their last stream."""
        if CUDA_SIMULATION or self.last_stream is None:
            self.allocations.clear()
            return
        with current_cupy_stream(self.last_stream):
            self.allocations.clear()

    @property
    def allocated_bytes(self) -> int:
        """Total bytes allocated across tracked arrays."""
        total = 0
        for arr in self.allocations.values():
            total += arr.nbytes
        return total

    @property
    def work_complete(self) -> bool:
        """Return whether the owner's submitted CUDA work is complete."""
        if self.submitting:
            return False
        if self.completion_event is None or CUDA_SIMULATION:
            return True
        return bool(self.completion_event.query())


@define
class MemoryManager:
    """Singleton interface coordinating GPU memory allocation and
    stream usage.

    Parameters
    ----------
    totalmem
        Total GPU memory in bytes. Determined automatically when
        omitted.
    registry
        Registry mapping instance identifiers to their memory
        settings.
    stream_groups
        Manager for organising instances into stream groups.

    Notes
    -----
    The manager accepts :class:`ArrayRequest` objects and returns
    :class:`ArrayResponse` instances that reference allocated arrays
    and chunking information. Active mode enforces per-instance VRAM
    proportions while passive mode mirrors standard allocation
    behaviour using chunking only when necessary.

    See Also
    --------
    :class:`~cubie.memory.array_requests.ArrayRequest`
        Describes a single allocation request.
    :class:`~cubie.memory.array_requests.ArrayResponse`
        Reports allocation outcomes.
    :class:`~cubie.memory.stream_groups.StreamGroups`
        Manages CUDA stream groups.
    """

    totalmem: int = field(
        default=None, validator=attrsval_optional(attrsval_instance_of(int))
    )
    registry: dict[int, InstanceMemorySettings] = field(
        default=attrsFactory(dict),
        validator=attrsval_optional(attrsval_instance_of(dict)),
    )
    stream_groups: StreamGroups = field(default=attrsFactory(StreamGroups))
    _mode: str = field(
        default="passive", validator=attrsval_in(["passive", "active"])
    )
    _auto_pool: list[int] = field(
        default=attrsFactory(list), validator=attrsval_instance_of(list)
    )
    _manual_pool: list[int] = field(
        default=attrsFactory(list), validator=attrsval_instance_of(list)
    )
    _queued_allocations: Dict[str, Dict] = field(
        default=attrsFactory(dict), validator=attrsval_instance_of(dict)
    )
    # Chunk parameters per (stream group, owner id): the partition a
    # solver's live device arrays are laid out for.
    _group_chunk_parameters: Dict[Tuple[str, int], Tuple[int, int]] = field(
        default=attrsFactory(dict), validator=attrsval_instance_of(dict)
    )
    _usage_clock: int = field(default=0, init=False)
    # Teardowns recorded by garbage-collection finalizers, run at the
    # manager's next entry point. GC can fire inside any allocation —
    # including mid-iteration over the registry, or on the transfer
    # watcher's own thread — so finalizers must not deregister (or
    # join threads) directly; they append here instead.
    _pending_teardowns: list = field(factory=list, init=False)
    # Bytes above which a host array is backed by a disk spill file;
    # None derives the threshold from total RAM at creation time.
    host_spill_threshold: Optional[int] = field(
        default=None,
        validator=opt_getype_validator(int, 0),
    )
    # Directory for spill files; None uses the system temp directory.
    spill_directory: Optional[str] = field(
        default=None,
        converter=_spill_directory_converter,
        validator=attrsval_optional(_existing_directory_validator),
    )
    # Cumulative pinned ceiling; None resolves to total VRAM at
    # construction, after which the field holds an integer.
    pinned_max_bytes: Optional[int] = field(
        default=None,
        validator=opt_getype_validator(int, 0),
    )
    # Pinned ledger: live backs reachable arrays; retained is
    # page-locked memory held only by CuPy's pinned pool.
    _pinned_lock: Lock = field(factory=Lock, init=False)
    _pinned_live_bytes: int = field(default=0, init=False)
    _pinned_retained_bytes: int = field(default=0, init=False)

    def __attrs_post_init__(self) -> None:
        """Initialise the manager with current GPU memory information."""
        try:
            free, total = self.get_memory_info()
        except ValueError as e:
            if e.args[0].startswith("not enough values to unpack"):
                warn(
                    "memory manager was initialised in a cuda-less "
                    "environment - memory manager will allow import but not"
                    "provide any memory (1 byte)"
                )
                total = 1
        except Exception as e:
            warn(
                f"Unexpected exception {e} encountered while instantiating "
                "memory manager"
            )
            total = 1
        self.totalmem = total
        if self.pinned_max_bytes is None:
            self.pinned_max_bytes = total
        self.registry = {}

    def register(
        self,
        instance: object,
        proportion: Optional[float] = None,
        invalidate_cache_hook: Callable = placeholder_invalidate,
        allocation_ready_hook: Callable = placeholder_dataready,
        stream_group: str = "default",
        owner: Optional[object] = None,
        host_spill_threshold: Optional[int] = None,
        spill_directory: Optional[os.PathLike | str] = None,
    ) -> None:
        """
        Register an instance and configure its memory allocation settings.

        Parameters
        ----------
        instance
            Instance to register for memory management.
        proportion
            Proportion of VRAM to allocate (0.0 to 1.0). When omitted, the
            instance joins the automatic allocation pool.
        invalidate_cache_hook
            Function to call when CUDA memory system changes occur.
        allocation_ready_hook
            Function to call when allocations are ready.
        stream_group
            Name of the stream group to assign the instance to.
        owner
            Eviction and settings unit for this registration. A solver
            kernel passes itself when registering its input and output
            array managers, so pressure evicts the whole solver or
            nothing and the managers resolve spill settings from the
            kernel's registration. Defaults to the instance itself.
        host_spill_threshold
            Bytes above which this instance's host arrays spill to
            disk. ``None`` defers to the owner's registration, then
            the manager-wide policy.
        spill_directory
            Directory for this instance's spill files. ``None`` defers
            to the owner's registration, then the manager-wide
            setting.

        Raises
        ------
        ValueError
            If instance is already registered or proportion is not between 0
            and 1.

        """
        self._purge_dead_instances()
        instance_id = id(instance)
        if instance_id in self.registry:
            raise ValueError("Instance already registered")

        try:
            instance_ref = weakref_ref(instance)
        except TypeError:
            instance_ref = None
        owner_id = instance_id if owner is None else id(owner)
        # Constructing the settings validates the spill options, so it
        # runs before the instance joins a stream group.
        settings = InstanceMemorySettings(
            invalidate_hook=invalidate_cache_hook,
            allocation_ready_hook=allocation_ready_hook,
            instance_ref=instance_ref,
            owner_id=owner_id,
            host_spill_threshold=host_spill_threshold,
            spill_directory=spill_directory,
        )
        self.stream_groups.add_instance(instance, stream_group)
        self.registry[instance_id] = settings

        if proportion:
            if not 0 <= proportion <= 1:
                raise ValueError("Proportion must be between 0 and 1")
            self._add_manual_proportion(instance, proportion)
        else:
            self._add_auto_proportion(instance)

    def get_registration(self, instance: object) -> InstanceMemorySettings:
        """Return the registry entry for a registered instance.

        Parameters
        ----------
        instance
            Registered instance to look up.

        Raises
        ------
        KeyError
            If the instance is not registered.
        """
        return self.registry[id(instance)]

    def set_limit_mode(self, mode: str) -> None:
        """
        Set the memory allocation limiting mode.

        Parameters
        ----------
        mode
            Either ``"passive"`` or ``"active"`` memory management mode.

        Raises
        ------
        ValueError
            If mode is not "passive" or "active".

        """
        if mode not in ["passive", "active"]:
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode

    def get_stream(self, instance: object) -> object:
        """
        Get the CUDA stream associated with an instance.

        Parameters
        ----------
        instance
            Instance to retrieve the stream for.

        Returns
        -------
        object
            CUDA stream associated with the instance.
        """
        return self.stream_groups.get_stream(instance)

    def get_group_stream(self, group: str = "default") -> Union[Stream, int]:
        """
        Get the dedicated stream for a named group, creating it if needed.

        Parameters
        ----------
        group
            Name of the stream group.

        Returns
        -------
        Stream
            The group's dedicated CUDA stream. Because each process
            owns its own manager, this stream is private to the
            process; synchronizing it never orders against the
            device-wide default stream.
        """
        return self.stream_groups.get_group_stream(group)

    def change_stream_group(self, instance: object, new_group: str) -> None:
        """
        Move instance to another stream group.

        Parameters
        ----------
        instance
            Instance to move.
        new_group
            Name of the new stream group.

        """
        self.stream_groups.change_group(instance, new_group)

    def reinit_streams(self) -> None:
        """
        Reinitialise all streams after a CUDA context reset.

        """
        self.stream_groups.reinit_streams()

    def invalidate_all(self) -> None:
        """
        Call each invalidate hook and release all allocations.

        """
        self.free_all()
        for registered_instance in self.registry.values():
            registered_instance.invalidate_hook()

    def set_manual_proportion(
        self, instance: object, proportion: float
    ) -> None:
        """
        Set manual allocation proportion for an instance.

        If instance is currently in the auto-allocation pool, shift it to
        manual.

        Parameters
        ----------
        instance
            Instance to update proportion for.
        proportion
            New proportion between 0 and 1.

        Raises
        ------
        ValueError
            If proportion is not between 0 and 1.

        """
        self._purge_dead_instances()
        instance_id = id(instance)
        if proportion < 0 or proportion > 1:
            raise ValueError("Proportion must be between 0 and 1")
        if instance_id in self._auto_pool:
            self._auto_pool.remove(instance_id)
        else:
            self._manual_pool.remove(instance_id)
        self._add_manual_proportion(instance, proportion)

    def set_manual_limit_mode(
        self, instance: object, proportion: float
    ) -> None:
        """
        Convert an auto-limited instance to manual allocation mode.

        Parameters
        ----------
        instance
            Instance to convert to manual mode.
        proportion
            Memory proportion to assign (0.0 to 1.0).

        Notes
        -----
        If the instance is already in the manual pool, this is a no-op.
        """
        self._purge_dead_instances()
        instance_id = id(instance)
        if instance_id in self._manual_pool:
            return
        self._auto_pool.remove(instance_id)
        self._add_manual_proportion(instance, proportion)

    def set_auto_limit_mode(self, instance: object) -> None:
        """
        Convert a manual-limited instance to auto allocation mode.

        Parameters
        ----------
        instance
            Instance to convert to auto mode.

        Notes
        -----
        If the instance is already in the auto pool, this is a no-op.
        """
        self._purge_dead_instances()
        instance_id = id(instance)
        settings = self.registry[instance_id]
        if instance_id in self._auto_pool:
            return
        self._manual_pool.remove(instance_id)
        settings.proportion = self._add_auto_proportion(instance)

    def proportion(self, instance: object) -> float:
        """
        Get the maximum proportion of VRAM allocated to an instance.

        Parameters
        ----------
        instance
            Instance to query.

        Returns
        -------
        float
            Proportion of VRAM allocated to this instance.
        """
        instance_id = id(instance)
        return self.registry[instance_id].proportion

    def cap(self, instance: object) -> Optional[int]:
        """
        Get the maximum allocatable bytes for an instance.

        Parameters
        ----------
        instance
            Instance to query.

        Returns
        -------
        int or None
            Maximum allocatable bytes for this instance.
        """
        instance_id = id(instance)
        settings = self.registry.get(instance_id)
        return settings.cap

    @property
    def manual_pool_proportion(self):
        """Total proportion of VRAM currently assigned manually."""
        self._purge_dead_instances()
        manual_settings = [
            self.registry[instance_id] for instance_id in self._manual_pool
        ]
        pool_proportion = sum(
            [settings.proportion for settings in manual_settings]
        )
        return pool_proportion

    @property
    def auto_pool_proportion(self):
        """Total proportion of VRAM currently distributed automatically."""
        self._purge_dead_instances()
        auto_settings = [
            self.registry[instance_id] for instance_id in self._auto_pool
        ]
        pool_proportion = sum(
            [settings.proportion for settings in auto_settings]
        )
        return pool_proportion

    def _add_manual_proportion(
        self, instance: object, proportion: float
    ) -> None:
        """
        Add an instance to the manual allocation pool with the specified proportion.

        Parameters
        ----------
        instance
            Instance to add to manual allocation pool.
        proportion
            Memory proportion to assign (0.0 to 1.0).

        Raises
        ------
        ValueError
            If manual proportion would exceed total available memory or leave
            insufficient memory for auto-allocated processes.

        Warnings
        --------
        UserWarning
            If manual proportion leaves less than 5% of memory for auto allocation.

        Notes
        -----
        Updates the instance's proportion and cap, then rebalances the auto pool.
        Enforces minimum auto pool size constraints.

        """
        instance_id = id(instance)
        new_manual_pool_size = self.manual_pool_proportion + proportion
        if new_manual_pool_size > 1.0:
            raise ValueError(
                "Manual proportion would exceed total available memory"
            )
        elif new_manual_pool_size > 1.0 - MIN_AUTOPOOL_SIZE:
            if len(self._auto_pool) > 0:
                raise ValueError(
                    "Manual proportion would leave less than 5% "
                    "of memory for auto-allocated processes. If "
                    "this is desired, adjust MIN_AUTOPOOL_SIZE in "
                    "mem_manager.py."
                )
            else:
                warn(
                    "Manual proportion leaves less than 5% of memory for "
                    "auto allocation if management mode == 'active'."
                )
        self._manual_pool.append(instance_id)
        self.registry[instance_id].proportion = proportion
        self.registry[instance_id].cap = int(proportion * self.totalmem)

        self._rebalance_auto_pool()

    def _add_auto_proportion(self, instance: object) -> float:
        """
        Add an instance to the auto allocation pool with equal share.

        Parameters
        ----------
        instance
            Instance to add to auto allocation pool.

        Returns
        -------
        float
            Proportion assigned to this instance.

        Raises
        ------
        ValueError
            If available auto-allocation pool is less than minimum required size.

        Notes
        -----
        Splits the non-manually-allocated portion of VRAM equally among all
        auto-allocated instances. Triggers rebalancing of the auto pool.
        """
        instance_id = id(instance)
        autopool_available = 1.0 - self.manual_pool_proportion
        if autopool_available <= MIN_AUTOPOOL_SIZE:
            raise ValueError(
                "Available auto-allocation pool is less than "
                "5% of total due to manual allocations. If "
                "this is desired, adjust MIN_AUTOPOOL_SIZE in "
                "mem_manager.py."
            )
        self._auto_pool.append(instance_id)
        self._rebalance_auto_pool()
        return self.registry[instance_id].proportion

    def defer_teardown(self, teardown: Callable[[], None]) -> None:
        """Record a teardown to run at the manager's next entry point.

        Garbage-collection finalizers call this instead of
        deregistering directly: GC runs inside whatever allocation
        triggered it — possibly while this manager iterates its
        registry, possibly on the transfer watcher's own thread — so
        the callback only appends. ``_purge_dead_instances`` runs the
        recorded teardowns in normal call context.
        """
        self._pending_teardowns.append(teardown)

    def _purge_dead_instances(self) -> None:
        """
        Drop registry entries whose instance has been collected.

        Registered instances are held weakly. Once an instance is
        garbage collected, its manual or auto reservation is
        released, its stream-group membership is removed, and any
        allocations it still had queued are discarded. Teardowns
        recorded by GC finalizers run first, here, in normal call
        context.

        """
        while self._pending_teardowns:
            self._pending_teardowns.pop()()
        dead_ids = [
            instance_id
            for instance_id, settings in self.registry.items()
            if settings.instance_ref is not None
            and settings.instance_ref() is None
        ]
        for instance_id in dead_ids:
            self._drop_instance(instance_id)
        if dead_ids:
            # Freed reservations flow back to the surviving auto pool.
            # The nested purge this triggers finds nothing dead, so
            # the recursion terminates after one level.
            self._rebalance_auto_pool()

    def _drop_instance(self, instance_id: int) -> None:
        """Free and deregister one instance."""
        settings = self.registry.get(instance_id)
        if settings is not None:
            settings.free_all()
            self.registry.pop(instance_id)
        if instance_id in self._manual_pool:
            self._manual_pool.remove(instance_id)
        if instance_id in self._auto_pool:
            self._auto_pool.remove(instance_id)
        self.stream_groups.remove_instance(instance_id)
        for queued in self._queued_allocations.values():
            queued.pop(instance_id, None)

    def release_instance(
        self, instance_id: int, settings: "InstanceMemorySettings"
    ) -> None:
        """Release a matching registered instance."""
        if self.registry.get(instance_id) is not settings:
            return
        self._drop_instance(instance_id)
        self._rebalance_auto_pool()

    def _owner_settings(self, owner_id: int) -> list[InstanceMemorySettings]:
        """Return registry entries owned by one client."""
        return [
            settings
            for settings in self.registry.values()
            if settings.owner_id == owner_id
        ]

    def begin_work(self, owner: object) -> None:
        """Mark an owner as submitting CUDA work."""
        self._usage_clock += 1
        for settings in self._owner_settings(id(owner)):
            settings.submitting = True
            settings.last_used = self._usage_clock

    def end_work(self, owner: object, stream: Stream) -> None:
        """Record completion of an owner's submitted CUDA work."""
        owned = self._owner_settings(id(owner))
        event = None
        if not CUDA_SIMULATION:
            event = cuda.event()
            event.record(stream)
        for settings in owned:
            settings.last_stream = stream
            settings.completion_event = event
            settings.submitting = False

    def _evict_idle_owners(
        self, exclude_ids: set[int], required_bytes: int
    ) -> int:
        """Free the least-recently-used idle owners.

        Whole owners are evicted, oldest first, until ``required_bytes``
        are released. Returns the number of bytes actually released.
        """
        excluded_owners = {
            self.registry[instance_id].owner_id
            for instance_id in exclude_ids
            if instance_id in self.registry
        }
        # Group registrations into eviction units by owner id.
        owners: Dict[int, list] = {}
        for settings in self.registry.values():
            owners.setdefault(settings.owner_id, []).append(settings)

        candidates = []
        for owner_id, owned in owners.items():
            # Never evict the requester's own registrations.
            if owner_id in excluded_owners:
                continue
            # Skip owners with submitted CUDA work still running.
            if not all(settings.work_complete for settings in owned):
                continue
            allocated = sum(settings.allocated_bytes for settings in owned)
            if allocated == 0:
                continue
            oldest_use = min(settings.last_used for settings in owned)
            candidates.append((oldest_use, owned))

        released = 0
        for _, owned in sorted(candidates, key=lambda item: item[0]):
            if released >= required_bytes:
                break
            for settings in owned:
                released += settings.allocated_bytes
                settings.free_all()
                settings.invalidate_hook()
        return released

    def _rebalance_auto_pool(self) -> None:
        """
        Redistribute available memory equally among auto-allocated instances.

        Notes
        -----
        Calculates the available proportion after manual allocations and
        divides it equally among all instances in the auto pool. Updates
        both proportion and cap for each auto-allocated instance.

        """
        available_proportion = 1.0 - self.manual_pool_proportion
        if len(self._auto_pool) == 0:
            return
        each_proportion = available_proportion / len(self._auto_pool)
        cap = int(each_proportion * self.totalmem)
        for instance_id in self._auto_pool:
            self.registry[instance_id].proportion = each_proportion
            self.registry[instance_id].cap = cap

    def free(self, array_label: str) -> None:
        """
        Free an allocation by label across all instances.

        Parameters
        ----------
        array_label
            Label of the allocation to free.

        """
        for settings in self.registry.values():
            if array_label in settings.allocations:
                settings.free(array_label)

    def free_all(self) -> None:
        """
        Free all allocations across all registered instances.

        """
        for settings in self.registry.values():
            settings.free_all()

    def _check_requests(self, requests: dict[str, ArrayRequest]) -> None:
        """
        Validate that all requests are properly formatted.

        Parameters
        ----------
        requests
            Dictionary of requests to validate.

        Raises
        ------
        TypeError
            If requests is not a dict or contains invalid ArrayRequest objects.

        """
        if not isinstance(requests, dict):
            raise TypeError(
                f"Expected dict for requests, got {type(requests)}"
            )
        for key, request in requests.items():
            if not isinstance(request, ArrayRequest):
                raise TypeError(
                    f"Expected ArrayRequest for {key}, got {type(request)}"
                )

    @property
    def pinned_budget_bytes(self) -> int:
        """``pinned_max_bytes`` capped at the spill fraction of RAM."""
        ram_cap = int(HOST_SPILL_FRACTION * total_system_ram())
        return min(self.pinned_max_bytes, ram_cap)

    @property
    def pinned_live_bytes(self) -> int:
        """Pinned bytes currently backing reachable arrays."""
        return self._pinned_live_bytes

    @property
    def pinned_retained_bytes(self) -> int:
        """Pinned bytes held only by CuPy's pinned pool."""
        return self._pinned_retained_bytes

    def _reserve_pinned_bytes(self, nbytes: int, force: bool = False) -> bool:
        """Atomically reserve ``nbytes`` against the pinned budget.

        Live and retained bytes count together; pressure empties the
        CuPy pinned pool before a refusal. ``force`` reserves past
        the budget.
        """
        with self._pinned_lock:
            budget = self.pinned_budget_bytes
            held = self._pinned_live_bytes + self._pinned_retained_bytes
            if held + nbytes > budget:
                free_all_pinned_blocks()
                self._pinned_retained_bytes = 0
            if not force and self._pinned_live_bytes + nbytes > budget:
                return False
            self._pinned_live_bytes += nbytes
            return True

    def _on_pinned_released(self, nbytes: int) -> None:
        """Move a collected pinned array's bytes from live to retained."""
        with self._pinned_lock:
            self._pinned_live_bytes -= nbytes
            self._pinned_retained_bytes += nbytes

    def allocate_pinned_array(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        force: bool = False,
    ) -> Optional[ndarray]:
        """Allocate one budget-accounted pinned host array.

        Reserves bytes, attempts the driver allocation, and attaches
        a finalizer that releases the bytes once the array and every
        view of it are collected.

        Parameters
        ----------
        shape
            Shape of the array to allocate.
        dtype
            Data type for the array elements.
        force
            Reserve even past the budget. A driver failure then
            propagates instead of returning ``None``.

        Returns
        -------
        numpy.ndarray or None
            The pinned array, or ``None`` when the budget or the
            driver refuses.
        """
        nbytes = int(prod(shape)) * np_dtype(dtype).itemsize
        if not self._reserve_pinned_bytes(nbytes, force=force):
            return None
        try:
            arr = empty_pinned(shape, dtype)
        except Exception:
            # Driver refused: return the reservation to the budget.
            with self._pinned_lock:
                self._pinned_live_bytes -= nbytes
            if force:
                # Forced callers were promised pinned: re-raise.
                raise
            return None
        finalize(arr, self._on_pinned_released, nbytes)
        return arr

    def create_host_array(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        memory_type: str = "pinned",
        like: Optional[ndarray] = None,
        instance: Optional[object] = None,
    ) -> ndarray:
        """
        Create a C-contiguous host array.

        Parameters
        ----------
        shape
            Shape of the array to create.
        dtype
            Data type for the array elements.
        memory_type
            ``"pinned"``, ``"host"``, or ``"memmap"``.
        like
            Optional source data.
        instance
            Registered instance whose spill directory applies for
            ``"memmap"`` arrays; required for that type, unused
            otherwise.

        Returns
        -------
        numpy.ndarray
            C-contiguous host array. A :class:`numpy.memmap` when the
            array spilled to disk. A ``"pinned"`` request whose
            reservation or driver allocation fails lands pageable.

        Raises
        ------
        ValueError
            If ``memory_type`` is not ``"pinned"``, ``"host"``, or
            ``"memmap"``.
        """
        _ensure_cuda_context()
        if memory_type not in ("pinned", "host", "memmap"):
            raise ValueError(
                f"memory_type must be 'pinned', 'host', or 'memmap', "
                f"got '{memory_type}'"
            )
        if memory_type == "memmap":
            arr = self._create_spill_array(shape, dtype, instance)
        elif memory_type == "pinned":
            arr = self.allocate_pinned_array(shape, dtype)
            if arr is None:
                arr = np_zeros(shape, dtype=dtype)
            elif like is None:
                arr.fill(0.0)
        else:
            # zeros() maps untouched pages lazily, so a large pageable
            # array costs nothing until the transfer writes it.
            arr = np_zeros(shape, dtype=dtype)
        if like is not None:
            arr[:] = like
        return arr

    def choose_host_memory_type(
        self,
        nbytes: int,
        instance: object,
        allow_pinned: bool = True,
    ) -> str:
        """Pick the backing for a host array of ``nbytes`` bytes.

        Arrays above the spill threshold are disk-backed. Below it,
        sizes within ``pinned_max_bytes`` choose pinned and the rest
        pageable. :meth:`allocate_pinned_array` budgets the actual
        allocation and may land a pinned choice pageable.

        Parameters
        ----------
        nbytes
            Size of the array in bytes.
        instance
            Registered instance whose spill policy applies, resolved
            through its owner's registration when unset there.
        allow_pinned
            Permit the ``"pinned"`` choice. Callers that cannot use a
            direct transfer (for example chunked staging) pass
            ``False``.

        Returns
        -------
        str
            ``"pinned"``, ``"host"``, or ``"memmap"``.
        """
        threshold = self._resolved_spill_threshold(instance)
        if nbytes > threshold:
            return "memmap"
        if allow_pinned and nbytes <= self.pinned_max_bytes:
            return "pinned"
        return "host"

    def _resolved_spill_threshold(self, instance: object) -> int:
        """Return the applicable spill threshold in bytes.

        Precedence: the instance's registered threshold, then its
        owner's, then the manager-wide threshold, then
        ``HOST_SPILL_FRACTION`` of total system RAM. An unregistered
        ``instance`` raises ``KeyError``.
        """
        settings = self.registry[id(instance)]
        if settings.host_spill_threshold is not None:
            return settings.host_spill_threshold
        owner = self.registry.get(settings.owner_id)
        if owner is not None and owner.host_spill_threshold is not None:
            return owner.host_spill_threshold
        if self.host_spill_threshold is not None:
            return self.host_spill_threshold
        return int(total_system_ram() * HOST_SPILL_FRACTION)

    def _create_spill_array(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        instance: object,
    ) -> np_memmap:
        """Create a temporary disk-backed array.

        Directory precedence mirrors ``_resolved_spill_threshold``:
        instance setting, then owner setting, then manager setting,
        then the system temp directory. An unregistered ``instance``
        raises ``KeyError``.
        """
        settings = self.registry[id(instance)]
        directory = settings.spill_directory
        if directory is None:
            owner = self.registry.get(settings.owner_id)
            if owner is not None:
                directory = owner.spill_directory
        if directory is None:
            directory = self.spill_directory
        directory = directory or gettempdir()
        handle, path = mkstemp(
            prefix="cubie-spill-", suffix=".dat", dir=directory
        )
        os.close(handle)
        arr = np_memmap(path, dtype=dtype, mode="w+", shape=shape)
        cleanup = finalize(arr, _remove_spill_file, arr._mmap, path)
        arr._cubie_spill_path = path
        arr._cubie_spill_cleanup = cleanup
        return arr

    def release_host_array(self, array: ndarray) -> None:
        """Release a spill-backed host array."""
        cleanup = getattr(array, "_cubie_spill_cleanup", None)
        if cleanup is None or not cleanup.alive:
            return
        path = array._cubie_spill_path
        try:
            array._mmap.close()
            os.remove(path)
        except OSError as error:
            raise OSError(f"Could not remove spill file '{path}'") from error
        cleanup.detach()

    def get_available_memory(self, group: str) -> int:
        """
        Get available memory for an entire stream group.

        Parameters
        ----------
        group
            Name of the stream group.

        Returns
        -------
        int
            Available memory in bytes for the group.

        Warnings
        --------
        UserWarning
            If group has used more than 95% of allocated memory.
        """
        free, total = self.get_memory_info()
        instances = self.stream_groups.get_instances_in_group(group)
        if self._mode == "passive":
            return free
        else:
            allocated = 0
            cap = 0
            for instance_id in instances:
                allocated += self.registry[instance_id].allocated_bytes
                cap += self.registry[instance_id].cap
            headroom = cap - allocated
            if headroom / cap < 0.05:
                warn(
                    f"Stream group {group} has used more than 95% of it's "
                    "allotted memory already, and future requests will run "
                    "slowly/in many chunks"
                )
            return min(headroom, free)

    def get_memory_info(self) -> tuple[int, int]:
        """
        Get free and total GPU memory information.

        Returns
        -------
        tuple of int
            (free_memory, total_memory) in bytes.
        """
        return current_mem_info()

    def get_stream_group(self, instance: object) -> str:
        """
        Get the name of the stream group for an instance.

        Parameters
        ----------
        instance
            Instance to query.

        Returns
        -------
        str
            Name of the stream group.
        """
        return self.stream_groups.get_group(instance)

    def is_grouped(self, instance: object) -> bool:
        """
        Check if instance is grouped with others in a named stream.

        Parameters
        ----------
        instance
            Instance to check.

        Returns
        -------
        bool
            True if instance shares a stream group with other instances.
        """
        group = self.get_stream_group(instance)
        if group == "default":
            return False
        peers = self.stream_groups.get_instances_in_group(group)
        if len(peers) == 1:
            return False
        return True

    def allocate_all(
        self,
        requests: dict[str, ArrayRequest],
        instance_id: int,
        stream: "cuda.cudadrv.driver.Stream",
        settings: Optional[InstanceMemorySettings] = None,
    ) -> dict[str, object]:
        """
        Allocate multiple arrays based on a dictionary of requests.

        Parameters
        ----------
        requests
            Dictionary mapping labels to array requests.
        instance_id
            ID of the requesting instance.
        stream
            CUDA stream for the allocations.
        settings
            Registration to record the allocations in. Passing it
            avoids a registry lookup that can fail when the client is
            garbage collected between queueing and allocation.

        Returns
        -------
        dict of str to object
            Dictionary mapping labels to allocated arrays.
        """
        responses = {}
        instance_settings = (
            settings if settings is not None else self.registry[instance_id]
        )
        instance_settings.last_stream = stream
        for key, request in requests.items():
            arr = self.allocate(
                shape=request.shape,
                dtype=request.dtype,
                memory_type=request.memory,
                stream=stream,
            )
            if CUDA_SIMULATION or request.memory != "device":
                instance_settings.add_allocation(key, arr)
            else:
                with current_cupy_stream(stream):
                    instance_settings.add_allocation(key, arr)
            responses[key] = arr
        return responses

    def allocate(
        self,
        shape: tuple[int, ...],
        dtype: Callable,
        memory_type: str,
        stream: "cuda.cudadrv.driver.Stream" = 0,
    ) -> object:
        """
        Allocate a single C-contiguous array with specified parameters.

        Parameters
        ----------
        shape
            Shape of the array to allocate.
        dtype
            Constructor returning the precision object for the array elements.
        memory_type
            Type of memory: "device" or "pinned".
        stream
            CUDA stream for the allocation. Defaults to 0.

        Returns
        -------
        object
            Allocated GPU array.

        Raises
        ------
        ValueError
            If memory_type is not "device" or "pinned".
        """
        _ensure_cuda_context()
        if memory_type == "device":
            # Native Numba array from the CuPy async pool (via the EMM).
            # current_cupy_stream makes the pool allocation stream-ordered.
            if CUDA_SIMULATION:  # pragma: no cover - simulated
                return cuda.device_array(shape, dtype)
            with current_cupy_stream(stream):
                return cuda.device_array(shape, dtype)
        elif memory_type == "pinned":
            # Pinned was requested by name: force, driver errors raise.
            return self.allocate_pinned_array(shape, dtype, force=True)
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")

    def queue_request(
        self, instance: object, requests: dict[str, ArrayRequest]
    ) -> None:
        """
        Queue allocation requests for batched stream group processing.

        Parameters
        ----------
        instance
            The instance making the request.
        requests
            Dictionary mapping labels to array requests.

        Raises
        ------
        ValueError
            If the instance registered without a live invalidate hook.
            Allocation holders are eviction candidates, so they must
            be able to drop handles when their buffers are freed.

        Notes
        -----
        Requests are queued per stream group, allowing multiple components
        to contribute to a single coordinated allocation that can be
        optimally chunked together.
        """
        self._check_requests(requests)
        instance_id = id(instance)
        settings = self.registry[instance_id]
        if settings.invalidate_hook.target() in (
            None,
            placeholder_invalidate,
        ):
            raise ValueError(
                "Instances that request allocations must register a "
                "live invalidate_cache_hook: allocated buffers are "
                "evicted under VRAM pressure, and the hook is how the "
                "instance drops its handles."
            )
        stream_group = self.get_stream_group(instance)
        if self._queued_allocations.get(stream_group) is None:
            self._queued_allocations[stream_group] = {}
        self._queued_allocations[stream_group].update({instance_id: requests})

    def to_device(
        self,
        instance: object,
        from_arrays: list[object],
        to_arrays: list[object],
        stream: Optional[Stream] = None,
    ) -> None:
        """
        Copy data to device arrays using the instance's stream.

        Parameters
        ----------
        instance
            Instance whose stream to use for copying.
        from_arrays
            Source arrays to copy from.
        to_arrays
            Destination device arrays to copy to.

        """
        _ensure_cuda_context()
        if stream is None:
            stream = self.get_stream(instance)
        self.registry[id(instance)].last_stream = stream
        # Pinned host buffer -> device, streamed async H2D. The low-level
        # driver copy skips copy_to_device's per-call np.array re-wrap and
        # compatibility checks (~50us/call), which are unnecessary here: the
        # source is an already-pinned, C-contiguous, size-matched buffer.
        for i, from_array in enumerate(from_arrays):
            if CUDA_SIMULATION:  # pragma: no cover - simulated
                cuda.to_device(from_array, stream=stream, to=to_arrays[i])
                continue
            if from_array.size == 0:
                continue
            # Sized by the pinned host buffer so the copy can never run
            # past it, whatever the device allocation rounded up to.
            cuda.cudadrv.driver.host_to_device(
                to_arrays[i], from_array, from_array.nbytes,
                stream=stream,
            )

    def from_device(
        self,
        instance: object,
        from_arrays: list[object],
        to_arrays: list[object],
        stream: Optional[Stream] = None,
    ) -> None:
        """
        Copy data from device arrays using the instance's stream.

        Parameters
        ----------
        instance
            Instance whose stream to use for copying.
        from_arrays
            Source device arrays to copy from.
        to_arrays
            Destination arrays to copy to.

        """
        _ensure_cuda_context()
        if stream is None:
            stream = self.get_stream(instance)
        self.registry[id(instance)].last_stream = stream
        # Device -> pinned host buffer, streamed async D2H via the low-level
        # driver copy (to_arrays are pinned, C-contiguous, size-matched).
        for i, from_array in enumerate(from_arrays):
            if CUDA_SIMULATION:  # pragma: no cover - simulated
                from_array.copy_to_host(to_arrays[i], stream=stream)
                continue
            if from_array.size == 0:
                continue
            # Sized by the pinned host buffer so the copy can never run
            # past it, whatever the device allocation rounded up to.
            cuda.cudadrv.driver.device_to_host(
                to_arrays[i], from_array, to_arrays[i].nbytes,
                stream=stream,
            )

    def sync_stream(
        self, instance: object, stream: Optional[Stream] = None
    ) -> None:
        """
        Synchronize the CUDA stream for an instance.

        Parameters
        ----------
        instance
            Instance whose stream to synchronize.

        """
        _ensure_cuda_context()
        if stream is None:
            settings = self.registry.get(id(instance))
            stream = None if settings is None else settings.last_stream
        if stream is None:
            stream = self.get_stream(instance)
        if stream == 0:
            stream = self.get_stream(instance)
        stream.synchronize()

    def allocate_queue(
        self,
        triggering_instance: object,
        stream: Optional[Stream] = None,
    ) -> None:
        """
        Process all queued requests for a stream group with coordinated chunking.

        Chunking is always performed along the run axis when memory
        constraints require splitting the batch.

        Parameters
        ----------
        triggering_instance
            The instance that triggered queue processing.

        Notes
        -----
        Processes all pending requests in the same stream group, applying
        coordinated chunking based on available memory. Calls
        allocation_ready_hook for each instance with their results.
        Instances in the group with no queued requests receive an empty
        response carrying the group's chunk parameters. When nothing is
        queued (all allocations already in place from an earlier call),
        every instance receives the group's stored chunk parameters, so
        per-instance chunk state is restored on repeat runs.

        """
        stream_group = self.get_stream_group(triggering_instance)
        queued_requests = self._queued_allocations.pop(stream_group, {})
        peers = self.stream_groups.get_instances_in_group(stream_group)

        # The run partition belongs to the launch's owner (the solver
        # whose registrations participate), not to the whole group:
        # solvers sharing a group must not inherit each other's
        # chunking.
        owner_id = self.registry[id(triggering_instance)].owner_id
        cache_key = (stream_group, owner_id)

        if not queued_requests:
            cached_parameters = self._group_chunk_parameters.get(cache_key)
            if cached_parameters is None:
                return None
            chunk_length, num_chunks = cached_parameters
            for peer in peers:
                peer_settings = self.registry.get(peer)
                if (
                    peer_settings is None
                    or peer_settings.owner_id != owner_id
                ):
                    continue
                peer_settings.allocation_ready_hook(
                    ArrayResponse(
                        arr={},
                        chunks=num_chunks,
                        chunk_length=chunk_length,
                        chunked_shapes={},
                    )
                )
            return None

        if stream is None:
            stream = self.get_stream(triggering_instance)
        for peer in peers:
            peer_settings = self.registry.get(peer)
            if peer_settings is not None:
                peer_settings.last_stream = stream

        # Get total_runs from first request
        num_runs = 1
        for requests_dict in queued_requests.values():
            for request in requests_dict.values():
                num_runs = request.total_runs
                break
            if num_runs > 1:
                break

        notaries = set(peers) - set(queued_requests.keys())
        # An owner's peer that is not reallocating keeps device arrays
        # laid out for the owner's current run partition, so the
        # partition must not move under it: reuse the stored chunk
        # parameters. Only a full reallocation of the owner's
        # registrations may pick a new partition.
        bound_to_cached = False
        for peer in notaries:
            peer_settings = self.registry.get(peer)
            if (
                peer_settings is not None
                and peer_settings.owner_id == owner_id
                and peer_settings.allocations
            ):
                bound_to_cached = True
                break
        cached_parameters = self._group_chunk_parameters.get(cache_key)
        if (
            bound_to_cached
            and cached_parameters is not None
            and cached_parameters[0] <= num_runs
        ):
            chunk_length, num_chunks = cached_parameters
        else:
            chunk_length, num_chunks = self.get_chunk_parameters(
                queued_requests, num_runs, stream_group
            )
        self._group_chunk_parameters[cache_key] = (
            chunk_length,
            num_chunks,
        )
        for instance_id, requests_dict in queued_requests.items():
            # Resolve the registration once: a client can be garbage
            # collected at any allocation, and its finalizer drops the
            # registry entry mid-loop. The held settings object stays
            # valid; a dead client's hook is a weak no-op and its
            # arrays are released with the settings object.
            settings = self.registry.get(instance_id)
            if settings is None:
                continue
            chunked_shapes = self.compute_chunked_shapes(
                requests_dict,
                chunk_length,
            )

            chunked_requests = deepcopy(requests_dict)
            for key, request in chunked_requests.items():
                request.shape = chunked_shapes[key]

            arrays = self.allocate_all(
                chunked_requests,
                instance_id,
                stream=stream,
                settings=settings,
            )
            response = ArrayResponse(
                arr=arrays,
                chunks=num_chunks,
                chunk_length=chunk_length,
                chunked_shapes=chunked_shapes,
            )

            if CUDA_SIMULATION:
                settings.allocation_ready_hook(response)
            else:
                with current_cupy_stream(stream):
                    settings.allocation_ready_hook(response)

        for peer in notaries:
            peer_settings = self.registry.get(peer)
            if peer_settings is None:
                continue
            peer_settings.allocation_ready_hook(
                ArrayResponse(
                    arr={},
                    chunks=num_chunks,
                    chunk_length=chunk_length,
                    chunked_shapes={},
                )
            )

        return None

    def get_chunk_parameters(
        self,
        requests: Dict[str, Dict],
        axis_length: int,
        stream_group: str,
    ) -> Tuple[int, int]:
        """
        Calculate number of chunks and chunk size for a dict of array requests.

        Chunking is performed along the run axis only.

        Parameters
        ----------
        requests
            Dictionary mapping instance IDs to their array requests.
        axis_length
            Unchunked length of the chunking axis.
        stream_group
            Name of the stream group making the request.

        Returns
        -------
        int, int
            Length of chunked axis and number of chunks needed to fit the
            request.

        Warnings
        --------
        UserWarning
            If request exceeds available VRAM by more than 20x.
        """
        free, _ = self.get_memory_info()
        available_memory = self.get_available_memory(stream_group)
        cap_headroom = None
        if self._mode == "active":
            members = self.stream_groups.get_instances_in_group(stream_group)
            cap_headroom = sum(
                self.registry[member].cap
                - self.registry[member].allocated_bytes
                for member in members
            )
        chunkable_size, unchunkable_size = get_portioned_request_size(
            requests,
        )

        request_size = chunkable_size + unchunkable_size

        # A requester's current device buffers are replaced by this
        # allocation, so their bytes are usable for it: without this
        # credit a same-size reallocation reads as a shortage of its
        # own footprint.
        own_reclaimable = sum(
            self.registry[instance_id].allocated_bytes
            for instance_id in requests
            if instance_id in self.registry
        )
        free_effective = free + own_reclaimable

        # Evict only for a genuine physical VRAM shortage that eviction
        # can fix. A configured cap (active mode) is a policy limit:
        # when the request exceeds cap headroom the run chunks anyway,
        # and evicting peers cannot raise the cap.
        physical_shortage = request_size >= free_effective
        cap_allows_request = (
            cap_headroom is None or request_size < cap_headroom
        )
        if physical_shortage and cap_allows_request:
            released = self._evict_idle_owners(
                set(requests.keys()), request_size - free_effective + 1
            )
            free_effective += released
        physical_headroom = (
            free_effective
            if cap_headroom is None
            else min(free_effective, cap_headroom)
        )
        # Group accounting and the physical probe are both estimates;
        # trust whichever allows more, since pool-held blocks satisfy
        # allocations without showing up as free device memory.
        available_memory = max(available_memory, physical_headroom)
        if request_size < available_memory:
            return axis_length, 1  # No chunking needed

        if free > 0 and request_size / free > 20:
            warn(
                "This request exceeds available VRAM by more than 20x. "
                f"Available VRAM = {free}, request size = {request_size}.",
                UserWarning,
            )

        # Check for all arrays unchunkable
        if chunkable_size == 0:
            raise ValueError(
                f"All requested arrays are unchunkable, but request size "
                f"({request_size}) exceeds available memory "
                f"({available_memory}). Cannot proceed."
            )

        # Guard: unchunkable arrays alone exceed available memory
        if unchunkable_size >= available_memory:
            raise ValueError(
                f"Unchunkable arrays require {unchunkable_size} bytes but only "
                f"{available_memory} bytes available. Cannot proceed."
            )

        # Calculate chunk size and number of chunks once we know it's eligible
        else:
            available_to_chunk = available_memory - unchunkable_size
            chunk_ratio = chunkable_size / available_to_chunk

            # Maximum chunk size that fits in available memory
            max_chunk_size = int(np_floor(axis_length / chunk_ratio))
            if max_chunk_size == 0:
                raise ValueError(
                    "Can't fit a single run in GPU VRAM. "
                    f"Available memory: {available_memory}. "
                    f"Request size: {request_size}. "
                    f"Chunkable request size: {chunkable_size}."
                )
            # With floor rounding, we might end up with an extra chunk or two
            num_chunks = int(np_ceil(axis_length / max_chunk_size))

        return max_chunk_size, num_chunks

    def compute_chunked_shapes(
        self,
        requests: dict[str, ArrayRequest],
        chunk_size: int,
    ) -> dict[str, Tuple[int, ...]]:
        """
        Compute per-array chunked shapes based on available memory.

        Parameters
        ----------
        requests
            Dictionary mapping labels to array requests.
        chunk_size
            Length of chunked arrays along run axis

        Returns
        -------
        dict[str, tuple[int, ...]]
            Mapping from array labels to their per-chunk shapes.

        Notes
        -----
        Unchunkable arrays retain their original shape.
        """
        chunked_shapes = {}
        for key, request in requests.items():
            if is_request_chunkable(request):
                axis_index = request.chunk_axis_index
                newshape = replace_with_chunked_size(
                    shape=request.shape,
                    axis_index=axis_index,
                    chunked_size=chunk_size,
                )
                chunked_shapes[key] = newshape
            else:
                chunked_shapes[key] = request.shape

        return chunked_shapes


def defer_instance_teardown(
    memory_manager: MemoryManager,
    instance_id: int,
    settings: InstanceMemorySettings,
    cleanups: Tuple[Callable[[], None], ...],
) -> None:
    """Record a collected client's teardown; the GC finalizer target.

    See :meth:`MemoryManager.defer_teardown` for why the callback
    must not run the teardown itself.
    """
    memory_manager.defer_teardown(
        partial(
            run_instance_teardown,
            memory_manager,
            instance_id,
            settings,
            cleanups,
        )
    )


def run_instance_teardown(
    memory_manager: MemoryManager,
    instance_id: int,
    settings: InstanceMemorySettings,
    cleanups: Tuple[Callable[[], None], ...],
) -> None:
    """Best-effort cleanup for a collected client."""
    try:
        if CUDA_SIMULATION or settings.last_stream is None:
            for cleanup in cleanups:
                cleanup()
            memory_manager.release_instance(instance_id, settings)
        else:
            with current_cupy_stream(settings.last_stream):
                for cleanup in cleanups:
                    cleanup()
                memory_manager.release_instance(instance_id, settings)
    except Exception:  # pragma: no cover - defensive at shutdown
        # Keep the entry alive if cleanup could not safely finish.
        settings.instance_ref = None


def get_portioned_request_size(
    requests: dict[str, dict[str, ArrayRequest]],
) -> tuple[int, int]:
    """
    Calculate total memory requested for the chunkable and unchunkable
    portions of the request.

    Chunking is performed along the run axis only.

    Parameters
    ----------
    requests
        Dictionary of array requests to analyze.

    Returns
    -------
    int, int
        chunkable, unchunkable - Total bytes for arrays that can be chunked
        or not, respectively, along the "run" axis.

    Notes
    -----
    Arrays are chunkable if:
    - request.unchunkable is False
    - The array has a "run" axis
    """
    chunkable = 0
    unchunkable = 0
    for reqs in requests.values():
        chunkable += sum(
            prod(req.shape) * req.dtype().itemsize
            for req in reqs.values()
            if is_request_chunkable(req)
        )
        unchunkable += sum(
            prod(req.shape) * req.dtype().itemsize
            for req in reqs.values()
            if not is_request_chunkable(req)
        )
    return chunkable, unchunkable


def is_request_chunkable(request) -> bool:
    """
    Determine if a single ArrayRequest is chunkable.

    Chunking is always performed along the run axis.

    Parameters
    ----------
    request
        The ArrayRequest to evaluate.

    Returns
    -------
    bool
        True if the request is chunkable, False otherwise.

    Notes
    -----
    A request is considered chunkable if:
    - request.unchunkable is False
    - chunk_axis_index is not None and within bounds
    - run axis has length > 1 (not a degenerate run axis)
    """
    if request.unchunkable:
        return False
    if len(request.shape) == 0:
        return False
    if request.chunk_axis_index is None:
        return False
    if request.chunk_axis_index >= len(request.shape):
        return False
    if request.shape[request.chunk_axis_index] == 1:
        return False
    return True


def replace_with_chunked_size(
    shape: Tuple[int, ...],
    axis_index: int,
    chunked_size: int,
) -> Tuple[int, ...]:
    """
    Replace the "run" axis in shape with chunked size.

    Parameters
    ----------
    shape
        Original shape of the array.
    axis_index
        integer index of the run axis in shape
    chunked_size
        Length of array after chunking along run axis

    Returns
    -------
    tuple[int, ...]
        New shape with chunked size along the "run" axis.
    """
    newshape = tuple(
        dim if i != axis_index else chunked_size for i, dim in enumerate(shape)
    )
    return newshape
