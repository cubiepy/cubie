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

Notes
-----
Array chunking for memory management is performed along the run axis when
batches exceed available GPU memory. The chunking process coordinates transfers
and synchronization across chunks automatically.

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

from cubie._utils import opt_gttype_validator, getype_validator
from cubie.cuda_simsafe import (
    CUDA_SIMULATION,
    DeviceNDArrayBase,
    is_pinned_array,
)
from cubie.memory import default_memmgr
from cubie.memory.mem_manager import (
    ArrayRequest,
    ArrayResponse,
    MemoryManager,
    current_cupy_stream,
    defer_instance_teardown,
)
from cubie.outputhandling.output_sizes import ArraySizingClass


def staging_blocks(
    device_array: DeviceNDArrayBase,
    host_array: NDArray,
    budget_bytes: int,
) -> Iterator[tuple[DeviceNDArrayBase, NDArray]]:
    """Yield ``(device_block, host_block)`` pairs of at most
    ``budget_bytes`` each, covering the extent both arrays share.

    Blocks are cut along the leading axis; a single leading row that
    exceeds the budget is cut along its own leading axis, so every
    device block is a contiguous region and the block size is bounded
    however wide the run axis is. The host array may be a strided
    view whose run extent is shorter than the device array's; blocks
    stop at the shorter extent on every axis.
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
    """Metadata wrapper for a single managed array."""

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
        """Return True if this array requires chunked transfers.

        Chunked transfers are needed when the array's full shape differs
        from its per-chunk shape. This comparison replaces complex
        is_chunked flag logic.
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
        When chunking is inactive (is_chunked=False or _chunk_axis_index=None),
        returns the full array. Otherwise computes slice based on stored
        chunk parameters and _chunk_axis_index.
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
        """Attach an array and update stored shape metadata."""

        self._array = value


@define(slots=False)
class ArrayContainer(ABC):
    """Store per-array metadata and references for CUDA managers."""

    _managed_map_cache: Optional[Dict[str, ManagedArray]] = field(
        default=None, init=False, eq=False, repr=False
    )

    def _managed_map(self) -> Dict[str, ManagedArray]:
        """Map array labels to ManagedArray fields, built once."""
        if self._managed_map_cache is None:
            self._managed_map_cache = {
                name: value
                for name, value in self.__dict__.items()
                if isinstance(value, ManagedArray)
            }
        return self._managed_map_cache

    def _iter_field_items(self) -> Iterator[tuple[str, ManagedArray]]:
        return iter(self._managed_map().items())

    def iter_managed_arrays(self) -> Iterator[tuple[str, ManagedArray]]:
        """Yield ``(label, managed)`` pairs for each array."""

        return self._iter_field_items()

    def array_names(self) -> List[str]:
        """Return array labels managed by this container."""

        return list(self._managed_map())

    def get_managed_array(self, label: str) -> ManagedArray:
        """Retrieve the metadata wrapper for ``label``."""

        managed = self._managed_map().get(label)
        if managed is None:
            raise AttributeError(
                f"Managed array with label '{label}' does not exist."
            )
        return managed

    def get_array(
        self, label: str
    ) -> Optional[Union[NDArray, DeviceNDArrayBase]]:
        """Return the stored array for ``label``."""

        return self.get_managed_array(label).array

    def set_array(
        self, label: str, array: Optional[Union[NDArray, DeviceNDArrayBase]]
    ) -> None:
        """Attach an array reference to ``label``."""
        self.get_managed_array(label).array = array

    def set_memory_type(self, memory_type: str) -> None:
        """Apply ``memory_type`` to all managed arrays."""

        for _, managed in self.iter_managed_arrays():
            managed.memory_type = memory_type

    @property
    def memory_type(self) -> str:
        """Return the memory type of the first managed array."""

        for _, managed in self.iter_managed_arrays():
            return managed.memory_type
        return "No arrays managed"

    def delete_all(self) -> None:
        """Delete all array references."""

        for _, managed in self.iter_managed_arrays():
            managed.array = None

    def attach(self, label: str, array: NDArray) -> None:
        """Attach an array to this container."""

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
    _precision
        Precision factory used to create new arrays.
    _sizes
        Size specifications for arrays managed by this instance.
    device
        Container for device-side arrays.
    host
        Container for host-side arrays.
    _chunks
        Number of chunks for memory management. Chunking is always
        performed along the run axis.
    _stream_group
        Stream group identifier for CUDA operations.
    _memory_proportion
        Proportion of available memory to use.
    _needs_reallocation
        Array names that require device reallocation.
    _needs_overwrite
        Array names that require host overwrite.
    _memory_manager
        Memory manager instance for handling GPU memory.

    Notes
    -----
    Subclasses must implement :meth:`update`, :meth:`finalise`, and
    :meth:`initialise` to wire batching behaviour into host and device
    execution paths.
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
    # label -> array, label -> memory type, saved size signature).
    # If the owner is collected before the next solve the buffers
    # return to their slots; otherwise the next solve allocates
    # fresh backing and the owner keeps the data.
    _loan: Optional[tuple] = field(
        default=None, init=False, eq=False, repr=False
    )
    # weakref.finalize handle that deregisters this manager and frees its
    # buffers when the manager is collected (or when close() runs it early).
    _finalizer: object = field(default=None, init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        """
        Initialize the array manager after attrs initialization.

        Notes
        -----
        This method registers with the memory manager and sets up
        invalidation hooks.

        """
        self.register_with_memory_manager()
        self._invalidate_hook()

    @property
    def is_chunked(self) -> bool:
        """Return True if arrays are being processed in multiple chunks."""
        return self._chunks > 1

    def set_array_runs(self, num_runs: int) -> None:
        """Update num_runs in all ManagedArray instances.

        This method sets the num_runs attribute to specify the total number
        of runs in the batch. This value is used during allocation to
        determine chunking behavior.

        Parameters
        ----------
        num_runs : int
            Total number of runs in the batch. Must be >= 1.

        """
        # Update the num_runs attribute
        self.num_runs = num_runs
        for _, array in self._iter_managed_arrays:
            array.num_runs = num_runs

    @property
    def _iter_managed_arrays(self) -> Iterator[tuple[str, ManagedArray]]:
        """
        Yield ``(label, managed)`` pairs for each managed array.

        Returns
        -------
        Iterator[tuple[str, ManagedArray]]
            Iterator over array labels and their metadata wrappers.
        """
        for label, managed in self.device.iter_managed_arrays():
            yield label, managed
        for label, managed in self.host.iter_managed_arrays():
            yield label, managed

    @abstractmethod
    def update(self, *args: object, **kwargs: object) -> None:
        """
        Update arrays from external data.

        This method should handle updating the manager's arrays based on
        provided input data and trigger reallocation/allocation as needed.

        Parameters
        ----------
        *args
            Positional arguments passed by subclasses.
        **kwargs
            Keyword arguments passed by subclasses.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses
        with the desired behavior for updating arrays from external data.

        """

    def _on_allocation_complete(self, response: ArrayResponse) -> None:
        """
        Callback for when the allocation response is received.

        Parameters
        ----------
        response
            Response object containing allocated arrays and metadata.

        Warns
        -----
        UserWarning
            If a device array is not found in the allocation response.


        Notes
        -----
        Warnings are only issued if the response contains some arrays but
        not the expected one, indicating a potential allocation mismatch.

        Stores chunk parameters from response in ManagedArray objects for
        both host and device containers.
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
        if self.is_chunked:
            self._convert_host_to_numpy()
        else:
            self._convert_host_to_pinned()
        self._needs_reallocation = [
            label
            for label in self._needs_reallocation
            if label not in arrays
        ]
        self._requested_labels -= set(arrays)

    def register_with_memory_manager(self) -> None:
        """
        Register this instance with the MemoryManager.

        Notes
        -----
        This method sets up the necessary hooks and callbacks for memory
        management integration.

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
        """Return cleanup calls that do not capture this manager."""
        return [self.device.delete_all]

    @property
    def host_spill_threshold(self) -> Optional[int]:
        """Owner's spill threshold; None without an owner."""
        if self._memory_owner is None:
            return None
        return self._memory_owner.host_spill_threshold

    @property
    def spill_directory(self) -> Optional[str]:
        """Owner's spill directory; None without an owner."""
        if self._memory_owner is None:
            return None
        return self._memory_owner.spill_directory

    def close(self) -> None:
        """Release this manager's resources."""
        settings = self._memory_manager.registry.get(id(self))
        if settings is None:
            return
        if CUDA_SIMULATION or settings.last_stream is None:
            for cleanup in self._teardown_cleanups():
                cleanup()
            self._memory_manager.release_instance(id(self), settings)
            BaseArrayManager.reset(self)
        else:
            with current_cupy_stream(settings.last_stream):
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
        """
        Send a request for allocation of device arrays.

        Parameters
        ----------
        request
            Dictionary mapping array names to allocation requests.

        Notes
        -----
        If the object is the only instance in its stream group, or is on
        the default group, then the request will be sent as a "single"
        request and be allocated immediately. If the object shares a stream
        group, then the response will be queued, and the allocation will be
        grouped with other requests in the same group, until one of the
        instances calls "process_queue" to process the queue. This behaviour
        can be overridden by setting force_type to "single" or "group".

        """
        self._memory_manager.queue_request(self, request)

    def _invalidate_hook(self) -> None:
        """
        Drop all references and assign all arrays for reallocation.

        Notes
        -----
        This method is called when the memory cache needs to be invalidated.
        It clears all device array references and marks them for reallocation.

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
        Check if two arrays are equal in shape and optionally content.

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
        Update the expected sizes for arrays in this manager.

        Parameters
        ----------
        sizes
            Array sizing configuration with new dimensions.

        Raises
        ------
        TypeError
            If the new sizes object is not the same size as the existing one.

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
        Check if the dtype of arrays matches their stored dtype.

        Parameters
        ----------
        arrays
            Dictionary mapping array names to arrays.

        Returns
        -------
        Dict[str, bool]
            Dictionary indicating whether each array matches the expected
            precision.
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
        Check whether arrays match configured sizes and stride order.

        Parameters
        ----------
        new_arrays
            Dictionary mapping array names to arrays.
        location
            ``"host"`` or ``"device"`` indicating which container to inspect.

        Returns
        -------
        Dict[str, bool]
            Dictionary indicating whether each array matches its expected
            shape.

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
        Execute post-chunk behaviour for device outputs.

        Parameters
        ----------
        chunk_index
            Chunk index about to run on the device

        """

    @abstractmethod
    def initialise(self, chunk_index: int) -> None:
        """
        Execute pre-chunk behaviour for device inputs.

        Parameters
        ----------
        chunk_index
            Chunk index about to run on the device.

        """

    def check_incoming_arrays(
        self, arrays: Dict[str, NDArray], location: str = "host"
    ) -> Dict[str, bool]:
        """
        Validate shape and precision for incoming arrays.

        Parameters
        ----------
        arrays
            Dictionary mapping array names to arrays.
        location
            ``"host"`` or ``"device"`` indicating the target container.

        Returns
        -------
        Dict[str, bool]
            Dictionary indicating whether each array is ready for attachment.
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
        shape_only: bool = False,
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
        shape_only
            The stored buffer only needs to match ``new_array``'s shape;
            values are ignored. Used for output arrays, which the kernel
            overwrites.

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

        if not shape_only and new_array.dtype != managed.dtype:
            # The only copy in the update path: device transfers need
            # the slot dtype, so a mismatched array is cast once.
            new_array = np_ascontiguousarray(
                new_array.astype(managed.dtype)
            )

        if current_array is new_array:
            if not shape_only and label not in self._needs_overwrite:
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
            if not shape_only and label not in self._needs_overwrite:
                self._needs_overwrite.append(label)
            if 0 in new_array.shape:
                # Zero-size updates keep a unit placeholder buffer.
                newshape = (1,) * len(current_array.shape)
                if shape_only:
                    new_array = self._memory_manager.create_host_array(
                        newshape,
                        managed.dtype,
                        self._base_memory_type(managed.memory_type),
                        spill_directory=self.spill_directory,
                    )
                else:
                    new_array = np_zeros(newshape, dtype=managed.dtype)

        if current_array is not new_array:
            self._memory_manager.release_host_array(current_array)
        self.host.attach(label, new_array)
        managed.memory_type = self._host_memory_type(new_array)
        return None

    def loan_host_arrays(self, owner: object) -> None:
        """Hand every host buffer to ``owner``, emptying the slots.

        ``owner`` (a result object that already references the
        arrays) keeps the data for as long as it lives. If it has
        been garbage collected by the next solve, the buffers return
        to their slots and are reused; otherwise the next solve
        allocates fresh backing.
        """
        self.reclaim_or_release_loan()
        arrays = {}
        types = {}
        for label, managed in self.host.iter_managed_arrays():
            arrays[label] = managed.array
            types[label] = managed.memory_type
            managed.array = None
        # The emptied slots must not satisfy the same-size fast path
        # before the loan is resolved; the signature is restored when
        # the buffers come back.
        size_sig = self._size_sig
        self._size_sig = None
        self._loan = (weakref_ref(owner), arrays, types, size_sig)

    def reclaim_or_release_loan(self) -> None:
        """Recover loaned host buffers if their owner was collected.

        A live owner keeps its buffers: the loan record is dropped so
        the arrays belong solely to the owner, and the next
        allocation builds fresh backing. A collected owner cannot be
        holding views, so the buffers return to their slots for
        reuse.
        """
        if self._loan is None:
            return
        owner_ref, arrays, types, size_sig = self._loan
        self._loan = None
        if owner_ref() is not None:
            return
        for label, array in arrays.items():
            managed = self.host.get_managed_array(label)
            managed.array = array
            managed.memory_type = types[label]
        self._size_sig = size_sig

    @staticmethod
    def _base_memory_type(memory_type: str) -> str:
        """Return the type to request when replacing a slot's array.

        A spilled slot re-requests pinned backing; the replacement
        spills again only if its size still exceeds the policy.
        """
        return "pinned" if memory_type == "memmap" else memory_type

    @staticmethod
    def _host_memory_type(array: NDArray) -> str:
        """Classify a host array's actual backing.

        ``"pinned"`` requires C-contiguous page-locked memory, which
        transfers directly and asynchronously. Strided or pageable
        arrays stage through bounded pinned blocks, and memmaps are
        disk-backed.
        """
        if isinstance(array, np_memmap):
            return "memmap"
        if is_pinned_array(array) and array.flags["C_CONTIGUOUS"]:
            return "pinned"
        return "host"

    @staticmethod
    def _requires_staging(array: NDArray, memory_type: str) -> bool:
        """Return whether a host array needs pinned staging."""
        return memory_type != "pinned"

    def update_host_arrays(
        self,
        new_arrays: Dict[str, NDArray],
        shape_only: bool = False,
    ) -> None:
        """
        Update host arrays and record allocation requirements.

        Parameters
        ----------
        new_arrays
            Dictionary mapping array names to new host arrays.
        shape_only
            Stored buffers only need to match the new arrays' shapes;
            values are ignored. Used for output arrays, which the kernel
            overwrites. Defaults to ``False``.

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
        if not any([check for check in self.check_sizes(new_arrays).values()]):
            warn(
                "Provided arrays do not match the expected system "
                "sizes, ignoring update",
                UserWarning,
            )
        for array_name in new_arrays:
            current_array = self.host.get_array(array_name)
            self._update_host_array(
                new_arrays[array_name],
                current_array,
                array_name,
                shape_only=shape_only,
            )

    def allocate(self) -> None:
        """
        Queue allocation requests for arrays that need reallocation.

        Notes
        -----
        Builds :class:`ArrayRequest` objects for arrays marked for
        reallocation and sets the ``unchunkable`` hint based on host metadata.

        Chunking is always performed along the run axis by convention.
        The specific axis index is determined by each array's chunk_axis_index.

        """
        requests = {}
        for array_label in list(set(self._needs_reallocation)):
            host_array_object = self.host.get_managed_array(array_label)
            host_array = host_array_object.array
            if host_array is None:
                # No host data yet (e.g. driver coefficients that have
                # not been supplied); the label stays pending and is
                # requested once data arrives.
                continue
            device_array_object = self.device.get_managed_array(array_label)
            total_runs = self.num_runs
            request = ArrayRequest(
                shape=host_array.shape,
                dtype=device_array_object.dtype,
                memory=device_array_object.memory_type,
                chunk_axis_index=host_array_object._chunk_axis_index,
                unchunkable=not host_array_object.is_chunked,
                total_runs=total_runs,
            )
            requests[array_label] = request
            # The slot's device buffer is replaced by this request;
            # dropping it here lets the manager release it before the
            # replacement is allocated.
            device_array_object.array = None
        self._requested_labels = set(requests)
        if requests:
            self.request_allocation(requests)

    def reset(self) -> None:
        """Clear cached arrays and allocation tracking."""
        if self._loan is not None:
            owner_ref, arrays, _, _ = self._loan
            self._loan = None
            if owner_ref() is None:
                # No owner survives to use or release these buffers.
                for array in arrays.values():
                    self._memory_manager.release_host_array(array)
        for _, managed in self.host.iter_managed_arrays():
            if managed.array is not None:
                self._memory_manager.release_host_array(managed.array)
        self.host.delete_all()
        self.device.delete_all()
        self._needs_reallocation.clear()
        self._needs_overwrite.clear()

    def to_device(
        self,
        from_arrays: List[object],
        to_arrays: List[object],
        stream: Optional[Any] = None,
    ) -> None:
        """
        Copy host arrays to the device using the memory manager.

        Parameters
        ----------
        from_arrays
            Host arrays to copy.
        to_arrays
            Destination device arrays.

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
        Copy device arrays back to the host using the memory manager.

        Parameters
        ----------
        from_arrays
            Device arrays to copy.
        to_arrays
            Destination host arrays.

        """
        self._memory_manager.from_device(
            self, from_arrays, to_arrays, stream=stream
        )

    def _convert_host_to_pinned(self) -> None:
        """Repin unchunked kernel-written slots for direct transfers.

        Runs after the chunk decision, so pinning never happens for a
        chunked solve. Slot content is not preserved: this applies
        only to buffers the kernel's device transfers overwrite, and
        input managers override it as a no-op because their slots hold
        caller-supplied arrays verbatim. A slot whose replacement the
        pinned budget refuses keeps its pageable buffer.
        """
        for _, slot in self.host.iter_managed_arrays():
            old_array = slot.array
            if old_array is None or slot.memory_type in (
                "pinned",
                "memmap",
            ):
                continue
            target_type = self._memory_manager.choose_host_memory_type(
                old_array.nbytes, self.host_spill_threshold
            )
            if target_type != "pinned":
                continue
            new_array = self._memory_manager.allocate_pinned_array(
                old_array.shape, old_array.dtype
            )
            if new_array is None:
                # The pinned budget refused the reservation.
                continue
            new_array.fill(0)
            self._memory_manager.release_host_array(old_array)
            slot.array = new_array
            slot.memory_type = "pinned"

    def _convert_host_to_numpy(self) -> None:
        """Move chunk-staged kernel-written slots to pageable backing.

        When chunking is active, full-size host buffers stay pageable
        and per-chunk transfers stage through the bounded pinned
        pool. Slot content is not preserved: this applies only to
        buffers each chunk writes back into, and input managers
        override it as a no-op because their slots hold
        caller-supplied arrays verbatim.
        """
        for _, slot in self.host.iter_managed_arrays():
            if slot.memory_type == "pinned" and slot.needs_chunked_transfer:
                old_array = slot.array
                if old_array is not None:
                    new_array = self._memory_manager.create_host_array(
                        old_array.shape,
                        old_array.dtype,
                        "host",
                    )
                    self._memory_manager.release_host_array(old_array)
                    slot.array = new_array
                    slot.memory_type = "host"
