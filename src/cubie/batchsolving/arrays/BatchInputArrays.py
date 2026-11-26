"""Manage host and device input arrays for batch integrations.

Published Classes
-----------------
:class:`InputArrayContainer`
    Attrs container holding :class:`ManagedArray` fields for initial
    values, parameters, and driver coefficients.

:class:`InputArrays`
    Concrete :class:`BaseArrayManager` subclass coordinating host-to-device
    transfers for batch input data.

See Also
--------
:class:`~cubie.batchsolving.arrays.BaseArrayManager.BaseArrayManager`
    Abstract base providing allocation and transfer infrastructure.
:class:`~cubie.batchsolving.arrays.BatchOutputArrays.OutputArrays`
    Counterpart managing output arrays.
:class:`~cubie.batchsolving.BatchSolverKernel.BatchSolverKernel`
    Primary consumer that owns input array instances.
"""

from attrs import Factory as attrsFactory, define, field
from attrs.validators import (
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
)
from numpy import (
    dtype as np_dtype,
    float32 as np_float32,
    floating as np_floating,
    issubdtype as np_issubdtype,
)

from numpy.typing import NDArray
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

from cubie._utils import ensure_nonzero_size, is_device_array
from cubie._cudasim_extensions import cuda
from cubie.memory.chunk_buffer_pool import ChunkBufferPool
from cubie.memory.mem_manager import HOST_STAGING_BYTES
from cubie.batchsolving.writeback_watcher import WritebackWatcher
from cubie.outputhandling.output_sizes import BatchInputSizes
from cubie.batchsolving.arrays.BaseArrayManager import (
    ArrayContainer,
    BaseArrayManager,
    ManagedArray,
    staging_blocks,
)
from cubie.batchsolving import ArrayTypes


@define(slots=False)
class InputArrayContainer(ArrayContainer):
    """Container for batch input arrays used by solver kernels."""

    initial_values: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("variable", "run"),
            default_shape=(1, 1),
        )
    )
    parameters: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("variable", "run"),
            default_shape=(1, 1),
        )
    )
    driver_coefficients: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            default_shape=(1, 1, 1),
            is_chunked=False,
        )
    )

    @classmethod
    def host_factory(
        cls, memory_type: str = "pinned"
    ) -> "InputArrayContainer":
        """Create a container for host-side input arrays.

        Parameters
        ----------
        memory_type
            Memory type to give every slot until an array is attached:
            ``"pinned"`` or ``"host"``.

        Returns
        -------
        InputArrayContainer
            Host-side container instance.

        Notes
        -----
        Each slot records the backing of the array attached to it,
        so this memory type lasts only until the first attach.
        """
        container = cls()
        container.set_memory_type(memory_type)
        return container

    @classmethod
    def device_factory(cls) -> "InputArrayContainer":
        """Create a container for device-side input arrays.

        Returns
        -------
        InputArrayContainer
            Device-side container instance.
        """
        container = cls()
        container.set_memory_type("device")
        return container


@define
class InputArrays(BaseArrayManager):
    """Manage allocation and transfer of batch input arrays.

    The manager holds the initial values, parameters and driver
    coefficients of a solve. It attaches the host arrays it is given
    without copying them, unless their dtype differs from the solve's
    precision, and copies them to the device before the kernel reads
    them. It hands device arrays to the kernel as they are.

    Parameters
    ----------
    sizes
        Size specifications for the input arrays.
    host
        Container for host-side arrays.

    Notes
    -----
    Build an instance with :meth:`from_solver`. Its other parameters
    are those of :class:`.BaseArrayManager`.
    """

    _sizes: Optional[BatchInputSizes] = field(
        factory=BatchInputSizes,
        validator=attrsval_optional(attrsval_instance_of(BatchInputSizes)),
    )
    host: InputArrayContainer = field(
        factory=InputArrayContainer.host_factory,
        validator=attrsval_instance_of(InputArrayContainer),
        init=True,
    )
    device: InputArrayContainer = field(
        factory=InputArrayContainer.device_factory,
        validator=attrsval_instance_of(InputArrayContainer),
        init=False,
    )
    # The pool charges its pinned buffers to this manager's budget.
    _buffer_pool: ChunkBufferPool = field(
        default=attrsFactory(
            lambda self: ChunkBufferPool(
                memory_manager=self._memory_manager
            ),
            takes_self=True,
        ),
        init=False,
    )
    _transfer_watcher: WritebackWatcher = field(
        factory=WritebackWatcher, init=False
    )
    _device_inputs: Dict[str, object] = field(factory=dict, init=False)

    def __attrs_post_init__(self) -> None:
        """Register the manager and set the containers' memory types.

        The host slots start as ``"pinned"`` and the device slots as
        ``"device"``. Each host slot then records the backing of the
        array attached to it.
        """
        super().__attrs_post_init__()
        self.host.set_memory_type("pinned")
        self.device.set_memory_type("device")

    def update(
        self,
        solver_instance: "BatchSolverKernel",
        initial_values: NDArray,
        parameters: NDArray,
        driver_coefficients: Optional[NDArray],
    ) -> None:
        """Set host arrays and request device allocations.

        Parameters
        ----------
        solver_instance
            The solver instance providing configuration and sizing information.
        initial_values
            Initial state values for each integration run.
        parameters
            Parameter values for each integration run.
        driver_coefficients
            Driver coefficient table, or ``None`` to keep the attached
            table.

        Raises
        ------
        ValueError
            If a host array, or a device array, does not have the
            expected shape.
        TypeError
            If a device array's dtype differs from the solve's
            precision.

        Notes
        -----
        The manager attaches device arrays as the kernel's inputs
        directly. It allocates no device buffer for them and copies
        nothing to the device. If you pass back the same host arrays
        as the last call, the manager only queues their values for
        upload. If you pass back one of the manager's own device
        buffers, it uploads nothing for that array.
        """
        self.update_from_solver(solver_instance)
        if self._fast_path_update(
            initial_values, parameters, driver_coefficients
        ):
            return
        updates_dict = {
            "initial_values": initial_values,
            "parameters": parameters,
        }
        if driver_coefficients is not None:
            updates_dict["driver_coefficients"] = driver_coefficients
        device_updates = {}
        host_updates = {}
        for name, arr in updates_dict.items():
            if not is_device_array(arr):
                host_updates[name] = arr
            elif (
                name not in self._device_inputs
                and arr is self.device.get_managed_array(name).array
            ):
                # The slot's own buffer: nothing to upload.
                continue
            else:
                device_updates[name] = arr
        for name in list(self._device_inputs):
            if name in host_updates:
                # Back to host input: reallocate the device buffer.
                del self._device_inputs[name]
                if name not in self._needs_reallocation:
                    self._needs_reallocation.append(name)
        if host_updates:
            self.update_host_arrays(host_updates)
        if device_updates:
            self._attach_device_inputs(device_updates)
        self.allocate()  # Will queue request if in a stream group

    def _attach_device_inputs(
        self, device_arrays: Dict[str, object]
    ) -> None:
        """Attach caller-supplied device arrays as kernel inputs.

        Parameters
        ----------
        device_arrays
            Mapping of array names to device arrays.

        Raises
        ------
        ValueError
            If a device array's shape differs from the expected size.
        TypeError
            If a device array's dtype differs from the slot dtype.
        """
        for name, arr in device_arrays.items():
            slot = self.device.get_managed_array(name)
            expected = tuple(getattr(self._sizes, name))
            shape = tuple(arr.shape)
            if ensure_nonzero_size(shape) != expected:
                raise ValueError(
                    f"Device input '{name}' has shape {shape}; "
                    f"expected {expected}."
                )
            if np_dtype(arr.dtype) != np_dtype(slot.dtype):
                raise TypeError(
                    f"Device input '{name}' has dtype {arr.dtype}; "
                    f"expected {np_dtype(slot.dtype).name}. Cast it "
                    f"on the device before solving."
                )
            self.device.attach(name, arr)
            self._device_inputs[name] = arr
            if name in self._needs_reallocation:
                self._needs_reallocation.remove(name)
            if name in self._needs_overwrite:
                self._needs_overwrite.remove(name)

    def _fast_path_update(
        self,
        initial_values: NDArray,
        parameters: NDArray,
        driver_coefficients: Optional[NDArray],
    ) -> bool:
        """Queue only the uploads when the attached inputs come back.

        Parameters
        ----------
        initial_values
            Initial values passed to :meth:`update`.
        parameters
            Parameters passed to :meth:`update`.
        driver_coefficients
            Driver coefficients passed to :meth:`update`, or ``None``.

        Returns
        -------
        bool
            True if every given array is the host array already
            attached, with the slot's dtype, and no device input is
            attached. :meth:`update` then has nothing more to do.
        """

        # Device inputs always take the full update path.
        if self._device_inputs:
            return False
        # Bail unless the caller re-supplied the attached arrays.
        if not self._matches_slot("initial_values", initial_values):
            return False
        if not self._matches_slot("parameters", parameters):
            return False
        overwrite = ["initial_values", "parameters"]
        if driver_coefficients is not None:
            if not self._matches_slot(
                "driver_coefficients", driver_coefficients
            ):
                return False
            overwrite.append("driver_coefficients")
        # Same buffers as last solve: only queue the value uploads.
        for name in overwrite:
            if name not in self._needs_overwrite:
                self._needs_overwrite.append(name)
        # Rebuild device buffers dropped by an invalidation.
        if self._needs_reallocation:
            self.allocate()
        return True

    def _matches_slot(self, name: str, array: NDArray) -> bool:
        """Return whether ``array`` is the host array of ``name``.

        Parameters
        ----------
        name
            Name of the input array.
        array
            Array to compare with the attached one.

        Returns
        -------
        bool
            True if ``array`` is the attached array and has the slot's
            dtype.
        """
        slot = self.host.get_managed_array(name)
        return array is slot.array and array.dtype == slot.dtype

    @property
    def has_device_inputs(self) -> bool:
        """Return whether any input was supplied as a device array."""
        return bool(self._device_inputs)

    @property
    def initial_values(self) -> ArrayTypes:
        """Initial values used in the last run.

        Returns the caller's device array when initial values were
        supplied on device; otherwise the host array.
        """
        return self._device_inputs.get(
            "initial_values", self.host.initial_values.array
        )

    @property
    def parameters(self) -> ArrayTypes:
        """Parameters used in the last run.

        Returns the caller's device array when parameters were
        supplied on device; otherwise the host array.
        """
        return self._device_inputs.get(
            "parameters", self.host.parameters.array
        )

    @property
    def driver_coefficients(self) -> ArrayTypes:
        """Host driver coefficients array."""

        return self.host.driver_coefficients.array

    @property
    def device_initial_values(self) -> ArrayTypes:
        """Device initial values array."""
        return self.device.initial_values.array

    @property
    def device_parameters(self) -> ArrayTypes:
        """Device parameters array."""
        return self.device.parameters.array

    @property
    def device_driver_coefficients(self) -> ArrayTypes:
        """Device driver coefficients array."""

        return self.device.driver_coefficients.array

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "InputArrays":
        """
        Create an InputArrays instance from a solver.

        Creates an empty instance from a solver instance, importing the heights
        of the parameters, initial values, and driver arrays from the ODE
        system for checking inputs against. Does not allocate host or device
        arrays.

        Parameters
        ----------
        solver_instance
            The solver instance to extract configuration from.

        Returns
        -------
        InputArrays
            A new InputArrays instance configured for the solver.
        """
        sizes = BatchInputSizes.from_solver(solver_instance)
        return cls(
            sizes=sizes,
            precision=solver_instance.precision,
            memory_manager=solver_instance.memory_manager,
            stream_group=solver_instance.stream_group,
            memory_owner=solver_instance,
        )

    def update_from_solver(self, solver_instance: "BatchSolverKernel") -> None:
        """Refresh the sizes, precision and run count from the solver.

        Give every floating-point array the solver's precision. Do
        nothing when the run count, precision, system sizes and driver
        coefficient shape are those of the last call.

        Parameters
        ----------
        solver_instance
            The solver instance to update from.
        """
        # Input sizes are system sizes scaled by num_runs; skip the rebuild
        # (attrs construction) when those determinants are unchanged.
        sysz = solver_instance.system_sizes
        sig = (
            solver_instance.num_runs,
            solver_instance.precision,
            sysz.states,
            sysz.swept_parameters,
            sysz.drivers,
            solver_instance.coefficients_shape,
        )
        if sig == self._size_sig:
            return
        self._sizes = BatchInputSizes.from_solver(solver_instance).nonzero
        self._precision = solver_instance.precision
        self.set_array_runs(solver_instance.num_runs)

        for name, arr_obj in self._iter_managed_arrays:
            if np_issubdtype(np_dtype(arr_obj.dtype), np_floating):
                arr_obj.dtype = self._precision
        self._size_sig = sig

    def finalise(self, chunk_index: int, stream=None) -> None:
        """Do nothing, because inputs need no work after a chunk.

        Parameters
        ----------
        chunk_index
            Index of the chunk that has just been launched.
        stream
            Unused.
        """

    def initialise(self, chunk_index: int, stream=None) -> None:
        """Queue the copies of a chunk's host inputs to the device.

        In an unchunked run, copy only the arrays queued since the
        last copy. In a chunked run, copy every input's slice for
        ``chunk_index``. Skip host arrays with no elements.

        Parameters
        ----------
        chunk_index
            Index of the chunk about to be launched.
        stream
            Stream to queue the copies on, or ``None`` for the
            manager's stream.

        Notes
        -----
        Copy pinned, C-contiguous host arrays straight to the device.
        Stage every other array, and every chunk slice, through pooled
        pinned buffers, each released by the transfer watcher once its
        copy completes. This method does not wait for the GPU, so the
        host can stage the next chunk while the previous kernel runs.
        """
        if stream is None:
            stream = self._memory_manager.get_stream(self)
        from_ = []
        to_ = []

        if self._chunks <= 1:
            arrays_to_copy = [array for array in self._needs_overwrite]
            self._needs_overwrite = []
        else:
            # Device-input passthrough cannot reach here: an attached
            # slot queues no allocation, so this manager's chunk count
            # stays 1. BatchSolverKernel guards chunked runs against
            # device inputs before the chunk loop starts.
            arrays_to_copy = list(self.device.array_names())

        for array_name in arrays_to_copy:
            device_obj = self.device.get_managed_array(array_name)
            host_obj = self.host.get_managed_array(array_name)
            if host_obj.array.size == 0:
                continue
            host_slice = (
                host_obj.chunk_slice(chunk_index)
                if host_obj.needs_chunked_transfer
                else host_obj.array
            )
            needs_staging = host_obj.needs_chunked_transfer or (
                self._requires_staging(host_slice, host_obj.memory_type)
            )
            if needs_staging:
                self._stage_array(
                    array_name,
                    host_slice,
                    device_obj.array,
                    stream,
                )
            else:
                from_.append(host_slice)
                to_.append(device_obj.array)

        if from_:
            self.to_device(from_, to_, stream=stream)

    def _stage_array(
        self, array_name, host_array, device_array, stream
    ) -> None:
        """Stage one non-pinned input through pooled pinned buffers.

        Cut the arrays into blocks with :func:`staging_blocks`, copy
        each host block into a pinned buffer and queue the buffer's
        copy to the device. The host source can be a strided view,
        such as a chunk slice or a memmap, and on the final chunk it
        can hold fewer runs than the device array. Copy each block by
        its shape, so the runs line up. Hand each buffer to the
        transfer watcher with its own event, so the buffer is returned
        to the pool as soon as its copy reaches the device.

        Parameters
        ----------
        array_name
            Name of the input array.
        host_array
            Host array or chunk slice to copy.
        device_array
            Destination device array.
        stream
            Stream to queue the copies on.
        """
        dtype = host_array.dtype
        for device_block, host_block in staging_blocks(
            device_array, host_array, HOST_STAGING_BYTES
        ):
            buffer = self._buffer_pool.acquire(
                array_name, device_block.shape, dtype
            )
            # Ours to release until the watcher takes it.
            try:
                trim = tuple(
                    slice(0, extent) for extent in host_block.shape
                )
                buffer.array[trim] = host_block
                self.to_device(
                    [buffer.array], [device_block], stream=stream
                )
                event = cuda.event()
                event.record(stream)
                self._transfer_watcher.submit_release(
                    event,
                    buffer,
                    self._buffer_pool,
                    array_name,
                )
            except BaseException:
                # Drain any queued copy before the buffer is reusable.
                try:
                    if stream is not None:
                        stream.synchronize()
                finally:
                    self._buffer_pool.release(buffer)
                raise

    def wait_pending(self, timeout: Optional[float] = None) -> None:
        """Wait until the transfer watcher releases every buffer.

        Parameters
        ----------
        timeout
            Maximum seconds to wait, or ``None`` to wait until done.
        """
        self._transfer_watcher.wait_all(timeout=timeout)

    def _invalidate_hook(self) -> None:
        """Drop the device arrays and queue the host data for upload.

        Forget any attached device inputs. The next copy uploads the
        attached host arrays into the new device buffers.
        """
        super()._invalidate_hook()
        self._device_inputs.clear()
        self._needs_overwrite.extend(self.host.array_names())

    def reset(self) -> None:
        """Release the arrays and buffers and stop the watcher."""
        super().reset()
        self._transfer_watcher.shutdown()
        self._buffer_pool.clear()
        self._device_inputs.clear()

    def _teardown_cleanups(self):
        """Return the calls that stop the watcher and free the buffers.

        Returns
        -------
        list of callable
            Calls that shut down the transfer watcher, free the
            staging buffers and drop the device arrays.
        """
        return [
            self._transfer_watcher.shutdown,
            self._buffer_pool.clear,
            *super()._teardown_cleanups(),
        ]
