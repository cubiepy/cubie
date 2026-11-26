"""Base utilities for managing batch arrays on host and device.

Published Classes
-----------------
:class:`ManagedArray`
    Metadata wrapper for a single managed array with shape, dtype,
    stride order, and chunking information.

:class:`ArrayContainer`
    Abstract attrs container storing per-array metadata and references.

:class:`BaseArrayManager`
    Abstract coordinator for host and device array allocation,
    transfer, and chunking.

Published Functions
-------------------
:func:`staging_blocks`
    Cut a device array and its host counterpart into matching blocks
    that each fit a staging buffer.

Notes
-----
A batch that does not fit in GPU memory is split into chunks along the
run axis. The managers then copy each chunk's inputs to the device
before its launch and its outputs back to the host after it, without
making the host wait for the GPU.

See Also
--------
:class:`~cubie.batchsolving.arrays.BatchInputArrays.InputArrays`
    Concrete input array manager.
:class:`~cubie.batchsolving.arrays.BatchOutputArrays.OutputArrays`
    Concrete output array manager.
:mod:`cubie.memory`
    Memory management infrastructure used for allocation.
"""

from abc import ABC, abstractmethod
from math import prod
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from warnings import warn
from weakref import finalize, ref as weakref_ref

from attrs import define, field
from attrs.validators import (
    deep_iterable as attrsval_deep_iterable,
    in_ as attrsval_in,
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)
from numpy import (
    array_equal as np_array_equal,
    ascontiguousarray as np_ascontiguousarray,
    float32 as np_float32,
    memmap as np_memmap,
    zeros as np_zeros,
    ndarray,
)
from numpy.typing import NDArray

from cubie._utils import (
    ensure_nonzero_size,
    getype_validator,
    opt_gttype_validator,
)
from cubie._cudasim_extensions import DeviceNDArrayBase
from cubie.memory.driver_memory import is_pinned_array
from cubie.memory import default_memmgr
from cubie.memory.mem_manager import (
    ArrayRequest,
    ArrayResponse,
    MemoryManager,
    defer_instance_teardown,
)
from cubie.outputhandling.output_sizes import ArraySizingClass


def staging_blocks(
    device_array: DeviceNDArrayBase,
    host_array: NDArray,
    budget_bytes: int,
) -> Iterator[tuple[DeviceNDArrayBase, NDArray]]:
    """Yield matching device and host blocks within ``budget_bytes``.

    Cut both arrays along their leading axis into groups of whole
    rows. When a single row is larger than the budget, cut each row
    along its own leading axis in the same way. Each device block is
    a contiguous region of device memory. Cut only the leading-axis
    extent that the two arrays share, so a host array with fewer rows
    than the device array gets blocks up to its own length.

    Parameters
    ----------
    device_array
        Device array to cut.
    host_array
        Host array, or host view, that matches ``device_array`` up to
        its leading extent.
    budget_bytes
        Largest size of one device block, in bytes.

    Yields
    ------
    tuple[DeviceNDArrayBase, numpy.ndarray]
        A device block and the host block that holds the same
        elements.
    """
    itemsize = device_array.dtype.itemsize
    row_bytes = int(prod(device_array.shape[1:])) * itemsize
    length = min(device_array.shape[0], host_array.shape[0])
    if row_bytes == 0:
        return
    if row_bytes <= budget_bytes:
        rows = max(1, budget_bytes // row_bytes)
        for start in range(0, length, rows):
            stop = min(start + rows, length)
            yield device_array[start:stop], host_array[start:stop]
        return
    for index in range(length):
        yield from staging_blocks(
            device_array[index], host_array[index], budget_bytes
        )


@define(slots=False)
class ManagedArray:
    """Metadata wrapper for a single managed array.

    Parameters
    ----------
    dtype
        NumPy scalar type of the array's elements.
    stride_order
        Name of each axis, such as ``("time", "variable", "run")``.
        The axis named ``"run"`` is the one cubie chunks.
    default_shape
        Shape to report while no array is attached.
    memory_type
        Backing of the attached array: ``"device"``, ``"pinned"``,
        ``"host"`` (pageable) or ``"memmap"``.
    is_chunked
        Whether the array may be split into chunks along its run axis.
    array
        Array whose shape to start from. The class keeps only the
        shape and attaches a zero array of that shape instead, so
        attach your data afterwards through :attr:`array`.
    chunked_shape
        Shape of one chunk, set once the memory manager has allocated
        the array.
    chunk_length
        Number of runs in each chunk except the last.
    num_chunks
        Number of chunks the batch is split into.
    num_runs
        Total number of runs in the batch.
    """

    dtype: type = field(
        default=np_float32, validator=attrsval_instance_of(type)
    )
    stride_order: tuple[str, ...] = field(
        factory=tuple,
        validator=attrsval_deep_iterable(
            member_validator=attrsval_instance_of(str),
            iterable_validator=attrsval_instance_of(tuple),
        ),
    )
    default_shape: tuple[Optional[int], ...] = field(
        factory=tuple,
        validator=attrsval_deep_iterable(
            member_validator=opt_gttype_validator(int, 0),
            iterable_validator=attrsval_instance_of(tuple),
        ),
    )
    memory_type: str = field(
        default="device",
        validator=attrsval_in(["device", "pinned", "host", "memmap"]),
    )
    is_chunked: bool = field(
        default=True, validator=attrsval_instance_of(bool)
    )
    _array: Optional[Union[NDArray, DeviceNDArrayBase]] = field(
        default=None,
        repr=False,
    )
    chunked_shape: Optional[tuple[int, ...]] = field(
        default=None,
        validator=attrsval_optional(
            attrsval_deep_iterable(
                member_validator=attrsval_instance_of(int),
                iterable_validator=attrsval_instance_of(tuple),
            )
        ),
    )
    chunk_length: Optional[int] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(int)),
    )
    num_chunks: int = field(
        default=1,
        validator=getype_validator(int, 1),
    )
    num_runs: int = field(
        default=1,
        validator=getype_validator(int, 1),
    )
    _chunk_axis_index: Optional[int] = field(
        default=None,
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self):
        shape = self.shape
        stride_order = self.stride_order
        defaultshape = shape if shape else (1,) * len(stride_order)
        self._array = np_zeros(defaultshape, dtype=self.dtype)

        if "run" in stride_order:
            self._chunk_axis_index = stride_order.index("run")

    @property
    def shape(self) -> tuple[Optional[int], ...]:
        """Return the current shape of the array."""
        if self._array is not None:
            return self._array.shape
        else:
            return self.default_shape

    @property
    def needs_chunked_transfer(self) -> bool:
        """Return True if this array transfers one chunk at a time.

        Return True only when the memory manager has set a chunk
        shape and the array's full shape differs from it.
        """
        if self.chunked_shape is None:
            return False
        return self.shape != self.chunked_shape

    def chunk_slice(
        self, chunk_index: int
    ) -> Union[ndarray, DeviceNDArrayBase]:
        """Return a slice of the array for the specified chunk index.

        Parameters
        ----------
        chunk_index
            Zero-based index of the chunk to slice.

        Returns
        -------
        Union[ndarray, DeviceNDArrayBase]
            View or slice of the array for the specified chunk.

        Raises
        ------
        TypeError
            If chunk_index is not an integer.

        Notes
        -----
        Return the full array when the array has no run axis, when
        ``is_chunked`` is False, or before a chunk length is set. The
        last chunk runs to the end of the run axis, so it holds the
        remaining runs.
        """
        # Validate chunk_index type
        if not isinstance(chunk_index, int):
            raise TypeError(
                f"chunk_index must be int, got {type(chunk_index).__name__}"
            )

        # Fast path: no chunking
        if (
            self._chunk_axis_index is None
            or self.is_chunked is False
            or self.chunk_length is None
        ):
            return self.array

        start = chunk_index * self.chunk_length

        if chunk_index == self.num_chunks - 1:
            end = None
        else:
            end = start + self.chunk_length

        # Build slice tuple - slice on chunk axis, full slice on others
        chunk_slice_list = [slice(None)] * len(self.shape)
        chunk_slice_list[self._chunk_axis_index] = slice(start, end)

        return self.array[tuple(chunk_slice_list)]

    @property
    def array(self) -> Optional[Union[NDArray, DeviceNDArrayBase]]:
        """Return the attached array reference."""
        return self._array

    @array.setter
    def array(
        self, value: Optional[Union[NDArray, DeviceNDArrayBase]]
    ) -> None:
        """Attach an array, or detach it with ``None``.

        Parameters
        ----------
        value
            Host or device array to attach, or ``None``.
        """

        self._array = value


@define(slots=False)
class ArrayContainer(ABC):
    """Store per-array metadata and references for CUDA managers.

    Subclasses declare one :class:`ManagedArray` field per array. The
    container finds those fields by scanning its attributes, and each
    field name is the label of its array.
    """

    _managed_map_cache: Optional[Dict[str, ManagedArray]] = field(
        default=None, init=False, eq=False, repr=False
    )

    def _managed_map(self) -> Dict[str, ManagedArray]:
        """Return a dict from array labels to their ManagedArray fields.

        Returns
        -------
        dict[str, ManagedArray]
            The container's managed arrays by label, built on the
            first call and reused afterwards.
        """
        if self._managed_map_cache is None:
            self._managed_map_cache = {
                name: value
                for name, value in self.__dict__.items()
                if isinstance(value, ManagedArray)
            }
        return self._managed_map_cache

    def _iter_field_items(self) -> Iterator[tuple[str, ManagedArray]]:
        """Return an iterator over ``(label, managed)`` pairs.

        Returns
        -------
        Iterator[tuple[str, ManagedArray]]
            Each array's label and metadata wrapper.
        """
        return iter(self._managed_map().items())

    def iter_managed_arrays(self) -> Iterator[tuple[str, ManagedArray]]:
        """Return an iterator over ``(label, managed)`` pairs.

        Returns
        -------
        Iterator[tuple[str, ManagedArray]]
            Each array's label and metadata wrapper.
        """

        return self._iter_field_items()

    def array_names(self) -> List[str]:
        """Return the labels of the arrays in this container.

        Returns
        -------
        list[str]
            One label per managed array.
        """

        return list(self._managed_map())

    def get_managed_array(self, label: str) -> ManagedArray:
        """Return the metadata wrapper for ``label``.

        Parameters
        ----------
        label
            Name of the array.

        Returns
        -------
        ManagedArray
            The array's metadata wrapper.

        Raises
        ------
        AttributeError
            If the container has no array called ``label``.
        """

        managed = self._managed_map().get(label)
        if managed is None:
            raise AttributeError(
                f"Managed array with label '{label}' does not exist."
            )
        return managed

    def get_array(
        self, label: str
    ) -> Optional[Union[NDArray, DeviceNDArrayBase]]:
        """Return the array attached to ``label``.

        Parameters
        ----------
        label
            Name of the array.

        Returns
        -------
        numpy.ndarray, DeviceNDArrayBase or None
            The attached array, or ``None`` when nothing is attached.

        Raises
        ------
        AttributeError
            If the container has no array called ``label``.
        """

        return self.get_managed_array(label).array

    def set_array(
        self, label: str, array: Optional[Union[NDArray, DeviceNDArrayBase]]
    ) -> None:
        """Attach ``array`` to ``label``.

        Parameters
        ----------
        label
            Name of the array.
        array
            Host or device array to attach, or ``None`` to detach.

        Raises
        ------
        AttributeError
            If the container has no array called ``label``.
        """
        self.get_managed_array(label).array = array

    def set_memory_type(self, memory_type: str) -> None:
        """Set the memory type of every managed array.

        Parameters
        ----------
        memory_type
            ``"device"``, ``"pinned"``, ``"host"`` or ``"memmap"``.
        """

        for _, managed in self.iter_managed_arrays():
            managed.memory_type = memory_type

    @property
    def memory_type(self) -> str:
        """Return the memory type of the first managed array.

        Return ``"No arrays managed"`` for a container with no arrays.
        """

        for _, managed in self.iter_managed_arrays():
            return managed.memory_type
        return "No arrays managed"

    def delete_all(self) -> None:
        """Detach every array, leaving each slot empty."""

        for _, managed in self.iter_managed_arrays():
            managed.array = None

    def attach(self, label: str, array: NDArray) -> None:
        """Attach ``array`` to ``label``, or warn for an unknown label.

        Parameters
        ----------
        label
            Name of the array.
        array
            Host or device array to attach.

        Warns
        -----
        UserWarning
            If the container has no array called ``label``. The
            container then ignores ``array``.
        """

        try:
            self.set_array(label, array)
        except AttributeError:
            warn(
                f"Device array with label '{label}' does not exist. ignoring",
                UserWarning,
            )


@define
class BaseArrayManager(ABC):
    """Coordinate allocation and transfer for batch host and device arrays.

    Parameters
    ----------
    precision
        NumPy floating type of the solve. The manager gives every
        floating-point array this dtype.
    sizes
        Expected shape of each managed array.
    device
        Container for device-side arrays.
    host
        Container for host-side arrays.
    chunks
        Number of chunks the batch is split into along the run axis.
    stream_group
        Name of the stream group to register in.
    memory_proportion
        Fraction of GPU memory this manager may use. Pass ``None`` to
        share the memory that no fraction claims equally with the
        other managers that give no fraction.
    memory_manager
        Memory manager that allocates the device arrays.
    memory_owner
        Object to register the manager's allocations under. The
        memory manager evicts all of an owner's allocations together.
        ``None`` registers the manager as its own owner.
    num_runs
        Total number of runs in the batch.

    Notes
    -----
    Subclasses implement :meth:`update` to attach a solve's arrays,
    :meth:`initialise` to copy a chunk's data to the device before
    its launch, and :meth:`finalise` to copy a chunk's data back
    after it.
    """

    _precision: type = field(
        default=np_float32, validator=attrsval_instance_of(type)
    )
    _sizes: Optional[ArraySizingClass] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(ArraySizingClass)),
    )
    device: ArrayContainer = field(
        factory=ArrayContainer, validator=attrsval_instance_of(ArrayContainer)
    )
    host: ArrayContainer = field(
        factory=ArrayContainer, validator=attrsval_instance_of(ArrayContainer)
    )
    _chunks: int = field(default=0, validator=attrsval_instance_of(int))
    _stream_group: str = field(
        default="default", validator=attrsval_instance_of(str)
    )
    _memory_proportion: Optional[float] = field(
        default=None, validator=attrsval_optional(attrsval_instance_of(float))
    )
    _needs_reallocation: list[str] = field(factory=list, init=False)
    # Labels sent in the most recent allocation request; a response
    # missing one of these is a genuine allocation mismatch.
    _requested_labels: set[str] = field(factory=set, init=False)
    _needs_overwrite: list[str] = field(factory=list, init=False)
    _memory_manager: MemoryManager = field(default=default_memmgr)
    # Owner whose registration carries spill settings and whose
    # registrations are evicted as one unit; the kernel for solver
    # array managers, this instance itself when unset.
    _memory_owner: Optional[object] = field(default=None, eq=False)
    num_runs: int = field(default=1, validator=getype_validator(int, 1))
    # Signature of the solver state that determines array sizes; when
    # unchanged, update_from_solver skips rebuilding the size objects.
    _size_sig: object = field(default=None, init=False)
    # Host buffers loaned to a result object: (weakref to the owner,
    # label -> array, label -> memory type). If the owner is collected
    # before the next solve the buffers return to their slots;
    # otherwise the next transfer builds fresh backing.
    _loan: Optional[tuple] = field(
        default=None, init=False, eq=False, repr=False
    )
    # weakref.finalize handle that deregisters this manager and frees its
    # buffers when the manager is collected (or when close() runs it early).
    _finalizer: object = field(default=None, init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        """Register the manager and queue every device array.

        Every device array starts queued for allocation, and the first
        :meth:`allocate` call requests it.
        """
        self.register_with_memory_manager()
        self._invalidate_hook()

    @property
    def is_chunked(self) -> bool:
        """Return True if arrays are being processed in multiple chunks."""
        return self._chunks > 1

    def set_array_runs(self, num_runs: int) -> None:
        """Set the batch's run count on the manager and every array.

        The chunks are sized from this count when the arrays are
        allocated.

        Parameters
        ----------
        num_runs
            Total number of runs in the batch, at least 1.
        """
        # Update the num_runs attribute
        self.num_runs = num_runs
        for _, array in self._iter_managed_arrays:
            array.num_runs = num_runs

    @property
    def _iter_managed_arrays(self) -> Iterator[tuple[str, ManagedArray]]:
        """Yield ``(label, managed)`` pairs for device and host arrays.

        Returns
        -------
        Iterator[tuple[str, ManagedArray]]
            The device container's arrays, then the host container's
            arrays.
        """
        for label, managed in self.device.iter_managed_arrays():
            yield label, managed
        for label, managed in self.host.iter_managed_arrays():
            yield label, managed

    @abstractmethod
    def update(self, *args: object, **kwargs: object) -> None:
        """Attach a solve's arrays and queue their allocation.

        Parameters
        ----------
        *args
            Positional arguments defined by the subclass.
        **kwargs
            Keyword arguments defined by the subclass.
        """

    def _on_allocation_complete(self, response: ArrayResponse) -> None:
        """Attach the allocated device arrays and record the chunking.

        Parameters
        ----------
        response
            Response object containing allocated arrays and metadata.

        Warns
        -----
        UserWarning
            If an array requested in the last :meth:`allocate` call is
            missing from the response.

        Notes
        -----
        Do nothing for a response with no arrays. Otherwise attach each
        array the manager was waiting for, store the chunk shape,
        chunk length and chunk count on its host and device wrappers,
        then call :meth:`_after_allocation`. Leave any array missing
        from the response queued for the next allocation.
        """
        chunked_shapes = response.chunked_shapes
        arrays = response.arr
        if not arrays:
            return

        # Extract chunk parameters from response
        chunks = response.chunks
        chunk_length = response.chunk_length

        for array_label in self._needs_reallocation:
            if array_label not in arrays:
                # Only a label that was actually requested and not
                # answered signals a mismatch; labels waiting on host
                # data are simply not requested yet.
                if array_label in self._requested_labels:
                    warn(
                        f"Device array {array_label} not found in "
                        f"allocation response. See "
                        f"BaseArrayManager._on_allocation_complete "
                        f"docstring for more info.",
                        UserWarning,
                    )
                continue
            self.device.attach(array_label, arrays[array_label])
            # Store chunked_shape and chunk parameters in ManagedArray
            if array_label in response.chunked_shapes:
                for container in (self.device, self.host):
                    array = container.get_managed_array(array_label)
                    array.chunked_shape = chunked_shapes[array_label]
                    array.chunk_length = chunk_length
                    array.num_chunks = chunks

        self._chunks = response.chunks
        self._needs_reallocation = [
            label
            for label in self._needs_reallocation
            if label not in arrays
        ]
        self._requested_labels -= set(arrays)
        self._after_allocation()

    def _after_allocation(self) -> None:
        """Run subclass work once an allocation's chunking is known.

        The base class does nothing here.
        """

    def _request_shape(self, label: str) -> Optional[tuple]:
        """Return the device shape to request for ``label``.

        Parameters
        ----------
        label
            Name of the array.

        Returns
        -------
        tuple or None
            The shape of the attached host array, or ``None`` while no
            host array is attached. A host array with a zero dimension
            gets a shape of all ones.
        """
        host_array = self.host.get_managed_array(label).array
        if host_array is None:
            return None
        return ensure_nonzero_size(tuple(host_array.shape))

    def register_with_memory_manager(self) -> None:
        """Register this manager with its memory manager.

        Register :meth:`_invalidate_hook` and
        :meth:`_on_allocation_complete` as the manager's callbacks.
        Attach a finalizer that, once the manager is garbage collected,
        queues the release of its registration and its buffers.
        """
        self._memory_manager.register(
            self,
            proportion=self._memory_proportion,
            invalidate_cache_hook=self._invalidate_hook,
            allocation_ready_hook=self._on_allocation_complete,
            stream_group=self._stream_group,
            owner=self._memory_owner,
        )
        settings = self._memory_manager.get_registration(self)
        self._finalizer = finalize(
            self,
            defer_instance_teardown,
            self._memory_manager,
            id(self),
            settings,
            tuple(self._teardown_cleanups()),
        )

    def _teardown_cleanups(self) -> List[Callable[[], None]]:
        """Return the calls that release this manager's resources.

        The calls hold no reference to the manager, so the finalizer
        can run them after the manager is garbage collected.

        Returns
        -------
        list of callable
            Calls that take no arguments.
        """
        return [self.device.delete_all]

    def close(self) -> None:
        """Release this manager's buffers and its memory registration.

        Do nothing if the manager is not registered. If a cleanup call
        raises, the manager stays registered and you can call
        ``close`` again.
        """
        settings = self._memory_manager.registry.get(id(self))
        if settings is None:
            return
        for cleanup in self._teardown_cleanups():
            cleanup()
        self._memory_manager.release_instance(id(self), settings)
        BaseArrayManager.reset(self)
        if self._finalizer is not None:
            self._finalizer.detach()

    def request_allocation(
        self,
        request: dict[str, ArrayRequest],
    ) -> None:
        """Queue a request for device arrays with the memory manager.

        When ``allocate_queue`` next runs for this manager's owner, the
        queued requests of every manager with that owner are allocated
        together, and :meth:`_on_allocation_complete` is called with
        the result.

        Parameters
        ----------
        request
            Dictionary mapping array names to allocation requests.
        """
        self._memory_manager.queue_request(self, request)

    def _invalidate_hook(self) -> None:
        """Drop every device array and queue each one for reallocation.

        This is called when the memory manager frees the manager's
        device memory, for example when the GPU runs short of memory
        for another allocation.
        The next :meth:`update_from_solver` call also rebuilds the
        array sizes.
        """
        self._needs_reallocation.clear()
        self._needs_overwrite.clear()
        self.device.delete_all()
        self._needs_reallocation.extend(self.device.array_names())
        # Force the next update_from_solver to rebuild sizes, in case an
        # invalidating config change altered a size the signature misses.
        self._size_sig = None

    def _arrays_equal(
        self,
        arr1: Optional[NDArray],
        arr2: Optional[NDArray],
        check_type: bool = True,
        shape_only: bool = False,
    ) -> bool:
        """
        Return whether two arrays match in shape and optionally content.

        Parameters
        ----------
        arr1
            First array or ``None``.
        arr2
            Second array or ``None``.
        check_type
            Check dtype equality. Defaults to ``True``.
        shape_only
            Skip element comparison; only check shape and optionally dtype.
            Faster for output arrays that will be overwritten. Defaults to
            ``False``.

        Returns
        -------
        bool
            ``True`` if arrays are equal, ``False`` otherwise.
        """
        if arr1 is None or arr2 is None:
            return arr1 is arr2
        if arr1.shape != arr2.shape:
            return False
        if check_type:
            if arr1.dtype is not arr2.dtype:
                return False
        if shape_only:
            return True
        return np_array_equal(arr1, arr2)

    def update_sizes(self, sizes: ArraySizingClass) -> None:
        """
        Replace the expected sizes of the arrays in this manager.

        Parameters
        ----------
        sizes
            Array sizing configuration with new dimensions.

        Raises
        ------
        TypeError
            If ``sizes`` is not an instance of the current sizes
            object's class.
        """
        if not isinstance(sizes, type(self._sizes)):
            raise TypeError(
                "Expected the new sizes object to be the "
                f"same size as the previous one "
                f"({type(self._sizes)}), got {type(sizes)}"
            )
        self._sizes = sizes

    def check_type(self, arrays: Dict[str, NDArray]) -> Dict[str, bool]:
        """
        Return whether each array's dtype matches its host slot's dtype.

        Parameters
        ----------
        arrays
            Dictionary mapping array names to arrays.

        Returns
        -------
        Dict[str, bool]
            Whether each array matches its slot's dtype. A ``None``
            array counts as a match.
        """
        matches = {}
        for array_name, array in arrays.items():
            host_dtype = self.host.get_managed_array(array_name).dtype
            if array is not None and array.dtype != host_dtype:
                matches[array_name] = False
            else:
                matches[array_name] = True
        return matches

    def check_sizes(
        self, new_arrays: Dict[str, NDArray], location: str = "host"
    ) -> Dict[str, bool]:
        """
        Return whether each array has the shape the manager expects.

        Parameters
        ----------
        new_arrays
            Dictionary mapping array names to arrays.
        location
            ``"host"`` or ``"device"`` indicating which container to inspect.

        Returns
        -------
        Dict[str, bool]
            Whether each array matches its expected shape. An array
            matches when it has the expected number of dimensions and
            every dimension that the sizes fix. An array with the
            expected number of dimensions also matches when it holds
            no elements. A name the container does not hold never
            matches.

        Raises
        ------
        AttributeError
            If the location is neither ``"host"`` nor ``"device"``.
        """
        try:
            container = getattr(self, location)
        except AttributeError:
            raise AttributeError(
                f"Invalid location: {location} - must be 'host' or 'device'"
            )
        expected_sizes = self._sizes
        matches = {}

        for array_name, array in new_arrays.items():
            if array_name not in container.array_names():
                matches[array_name] = False
                continue

            array_shape = array.shape
            expected_size_tuple = getattr(expected_sizes, array_name)
            expected_shape = list(expected_size_tuple)

            if len(array_shape) != len(expected_shape):
                matches[array_name] = False
            elif array.size == 0:
                # Zero-size data fits the unit slot sizing floors it to.
                matches[array_name] = True
            else:
                shape_matches = True
                for actual_dim, expected_dim in zip(
                    array_shape, expected_shape
                ):
                    if expected_dim is not None and actual_dim != expected_dim:
                        shape_matches = False
                        break
                matches[array_name] = shape_matches
        return matches

    @abstractmethod
    def finalise(self, chunk_index: int) -> None:
        """
        Queue the work that follows a chunk's kernel launch.

        Parameters
        ----------
        chunk_index
            Index of the chunk that has just been launched.
        """

    @abstractmethod
    def initialise(self, chunk_index: int) -> None:
        """
        Queue the work that precedes a chunk's kernel launch.

        Parameters
        ----------
        chunk_index
            Index of the chunk about to be launched.
        """

    def check_incoming_arrays(
        self, arrays: Dict[str, NDArray], location: str = "host"
    ) -> Dict[str, bool]:
        """
        Return whether each array has the expected shape and dtype.

        Parameters
        ----------
        arrays
            Dictionary mapping array names to arrays.
        location
            ``"host"`` or ``"device"`` indicating the target container.

        Returns
        -------
        Dict[str, bool]
            Whether each array passes both :meth:`check_sizes` and
            :meth:`check_type`.
        """
        dims_ok = self.check_sizes(arrays, location=location)
        types_ok = self.check_type(arrays)
        all_ok = {}
        for array_name in arrays:
            all_ok[array_name] = dims_ok[array_name] and types_ok[array_name]
        return all_ok

    def _update_host_array(
        self,
        new_array: NDArray,
        current_array: Optional[NDArray],
        label: str,
    ) -> None:
        """
        Attach one incoming host array and record allocation needs.

        Parameters
        ----------
        new_array
            Replacement array for the slot.
        current_array
            Previously stored host array or ``None``.
        label
            Array name used to index tracking lists.

        Raises
        ------
        ValueError
            If ``new_array`` is ``None``.

        Notes
        -----
        Incoming arrays are attached verbatim: the slot records the
        array's actual backing (pinned, pageable, or memmap) and
        transfers read from it directly, so no copy is made. The one
        exception is a dtype mismatch, which is cast once into a fresh
        buffer. The solve reads input arrays when it runs — mutating a
        submitted array between solves changes what the next solve
        integrates.
        """
        if new_array is None:
            raise ValueError("New array is None")
        managed = self.host.get_managed_array(label)

        if new_array.dtype != managed.dtype:
            # The only copy in the update path: device transfers need
            # the slot dtype, so a mismatched array is cast once.
            new_array = np_ascontiguousarray(
                new_array.astype(managed.dtype)
            )

        if current_array is new_array:
            if label not in self._needs_overwrite:
                self._needs_overwrite.append(label)
            return None

        if current_array is None:
            self._needs_reallocation.append(label)
            self._needs_overwrite.append(label)
        else:
            same_size = (
                current_array.shape == new_array.shape
                and new_array.dtype == managed.dtype
            )
            if not same_size and label not in self._needs_reallocation:
                self._needs_reallocation.append(label)
            if label not in self._needs_overwrite:
                self._needs_overwrite.append(label)

        self._memory_manager.release_host_array(current_array)
        self.host.attach(label, new_array)
        managed.memory_type = self._host_memory_type(new_array)
        return None

    def loan_host_arrays(self, owner: object) -> None:
        """Hand every host buffer to ``owner``, emptying the slots.

        ``owner`` is a result object that already references the
        arrays, and it keeps the data for as long as it lives. If it
        has been garbage collected by the next solve, the manager puts
        the buffers back in their slots and reuses them. Otherwise the
        next solve allocates new buffers.

        Parameters
        ----------
        owner
            Object to hand the buffers to. The manager holds only a
            weak reference to it.
        """
        self.reclaim_or_release_loan()
        arrays = {}
        types = {}
        for label, managed in self.host.iter_managed_arrays():
            arrays[label] = managed.array
            types[label] = managed.memory_type
            managed.array = None
        self._loan = (weakref_ref(owner), arrays, types)

    def reclaim_or_release_loan(self) -> None:
        """Recover loaned host buffers if their owner was collected.

        If the owner is still alive, forget the loan, so the buffers
        belong to the owner alone and the next transfer allocates new
        ones. If the owner has been collected, return the buffers to
        their slots for reuse. The manager tracks only the owner, so
        cubie overwrites, in the next solve, any of these arrays that
        other code kept after the owner was collected.
        """
        if self._loan is None:
            return
        owner_ref, arrays, types = self._loan
        self._loan = None
        if owner_ref() is not None:
            return
        for label, array in arrays.items():
            managed = self.host.get_managed_array(label)
            managed.array = array
            managed.memory_type = types[label]

    @staticmethod
    def _host_memory_type(array: NDArray) -> str:
        """Return the memory type of a host array's backing.

        Only C-contiguous page-locked memory counts as ``"pinned"``,
        and the managers copy it to and from the device directly. The
        managers copy every other array through pinned staging
        buffers.

        Parameters
        ----------
        array
            Host array to classify.

        Returns
        -------
        str
            ``"memmap"`` for a :class:`numpy.memmap`, ``"pinned"`` for
            C-contiguous page-locked memory, and ``"host"`` for every
            other array.
        """
        if isinstance(array, np_memmap):
            return "memmap"
        if is_pinned_array(array) and array.flags["C_CONTIGUOUS"]:
            return "pinned"
        return "host"

    @staticmethod
    def _requires_staging(array: NDArray, memory_type: str) -> bool:
        """Return whether a host array needs pinned staging.

        Parameters
        ----------
        array
            Host array to transfer.
        memory_type
            The array's memory type.

        Returns
        -------
        bool
            True for every memory type except ``"pinned"``.
        """
        return memory_type != "pinned"

    def update_host_arrays(self, new_arrays: Dict[str, NDArray]) -> None:
        """
        Attach new host arrays and queue their uploads and allocations.

        Parameters
        ----------
        new_arrays
            Dictionary mapping array names to new host arrays.

        Raises
        ------
        ValueError
            If an array does not have the shape the manager expects.

        Warns
        -----
        UserWarning
            If ``new_arrays`` names an array the manager does not hold.
            The manager ignores that entry.
        """
        host_names = set(self.host.array_names())
        badnames = [
            array_name
            for array_name in new_arrays
            if array_name not in host_names
        ]
        new_arrays = {k: v for k, v in new_arrays.items() if k in host_names}

        if any(badnames):
            warn(
                f"Host arrays '{badnames}' does not exist, ignoring update",
                UserWarning,
            )
        mismatched = [
            f"{name}: got {tuple(new_arrays[name].shape)}, expected "
            f"{tuple(getattr(self._sizes, name))}"
            for name, matches in self.check_sizes(new_arrays).items()
            if not matches
        ]
        if mismatched:
            raise ValueError(
                "Provided arrays do not match the expected system sizes: "
                + "; ".join(mismatched)
            )
        for array_name in new_arrays:
            current_array = self.host.get_array(array_name)
            self._update_host_array(
                new_arrays[array_name], current_array, array_name
            )

    def allocate(self) -> None:
        """
        Queue allocation requests for arrays that need reallocation.

        Notes
        -----
        Build one :class:`ArrayRequest` for each queued array whose
        shape is known, and drop the device array it replaces. Mark
        each request as chunkable along the array's run axis unless
        the array's ``is_chunked`` is False. Leave an input array with
        no host data queued until data arrives.
        """
        requests = {}
        for array_label in list(set(self._needs_reallocation)):
            shape = self._request_shape(array_label)
            if shape is None:
                # Pending data; requested once it arrives.
                continue
            host_array_object = self.host.get_managed_array(array_label)
            device_array_object = self.device.get_managed_array(array_label)
            total_runs = self.num_runs
            request = ArrayRequest(
                shape=shape,
                dtype=device_array_object.dtype,
                memory=device_array_object.memory_type,
                chunk_axis_index=host_array_object._chunk_axis_index,
                unchunkable=not host_array_object.is_chunked,
                total_runs=total_runs,
            )
            requests[array_label] = request
            # Drop the buffer this request replaces.
            device_array_object.array = None
        self._requested_labels = set(requests)
        if requests:
            self.request_allocation(requests)

    def reset(self) -> None:
        """Release the host arrays and queue every device array again.

        Release loaned buffers too when their owner has been
        collected. A live owner keeps its buffers.
        """
        if self._loan is not None:
            owner_ref, arrays, _ = self._loan
            self._loan = None
            if owner_ref() is None:
                # No owner survives to use or release these buffers.
                for array in arrays.values():
                    self._memory_manager.release_host_array(array)
        for _, managed in self.host.iter_managed_arrays():
            if managed.array is not None:
                self._memory_manager.release_host_array(managed.array)
        self.host.delete_all()
        self._invalidate_hook()

    def to_device(
        self,
        from_arrays: List[object],
        to_arrays: List[object],
        stream: Optional[Any] = None,
    ) -> None:
        """
        Queue copies of host arrays to the device.

        Parameters
        ----------
        from_arrays
            Pinned, C-contiguous host arrays to copy.
        to_arrays
            Destination device arrays.
        stream
            Stream to queue the copies on, or ``None`` for the
            manager's stream.
        """
        self._memory_manager.to_device(
            self, from_arrays, to_arrays, stream=stream
        )

    def from_device(
        self,
        from_arrays: List[object],
        to_arrays: List[object],
        stream: Optional[Any] = None,
    ) -> None:
        """
        Queue copies of device arrays back to the host.

        Parameters
        ----------
        from_arrays
            Device arrays to copy.
        to_arrays
            Pinned, C-contiguous destination host arrays.
        stream
            Stream to queue the copies on, or ``None`` for the
            manager's stream.
        """
        self._memory_manager.from_device(
            self, from_arrays, to_arrays, stream=stream
        )

