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

from cubie.cuda_simsafe import (
    cuda,
    CUDA_SIMULATION,
    is_device_array,
)
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
        """Create a container configured for host memory transfers.

        Parameters
        ----------
        memory_type
            Memory type for host arrays: "pinned" or "host".
            Default is "pinned" for non-chunked operation.

        Returns
        -------
        InputArrayContainer
            Host-side container instance.

        Notes
        -----
        Uses pinned (page-locked) memory to enable asynchronous
        host-to-device transfers with CUDA streams. Using ``"host"``
        memory type instead would result in pageable memory that blocks
        async transfers due to required intermediate buffering.
        """
        container = cls()
        container.set_memory_type(memory_type)
        return container

    @classmethod
    def device_factory(cls) -> "InputArrayContainer":
        """Create a container configured for device memory transfers.

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

    Parameters
    ----------
    _sizes
        Size specifications for the input arrays.
    host
        Container for host-side arrays.
    device
        Container for device-side arrays.

    Notes
    -----
    Instances are configured from
    :class:`~cubie.batchsolving.BatchSolverKernel` metadata. Updates request
    memory through the shared manager, ensure array heights match solver
    expectations, and attach received buffers prior to device transfers.
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
        """Ensure host and device containers use explicit memory types.

        Notes
        -----
        Host containers use pinned memory to enable asynchronous
        host-to-device transfers with CUDA streams.
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
            Horner-ordered driver interpolation coefficients.

        Notes
        -----
        Inputs supplied as device arrays are attached directly as the
        kernel's device inputs: no host staging or host-to-device
        transfer occurs, and no managed device buffer is allocated for
        them. They must already match the expected shape and dtype.
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
        device_updates = {
            name: arr
            for name, arr in updates_dict.items()
            if is_device_array(arr)
        }
        host_updates = {
            name: arr
            for name, arr in updates_dict.items()
            if name not in device_updates
        }
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
            if shape != expected:
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
        """Queue overwrites when the attached inputs are re-supplied."""

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
        """Return whether ``array`` is the attached, dtype-current slot."""
        slot = self.host.get_managed_array(name)
        return array is slot.array and array.dtype == slot.dtype

    @property
    def has_device_inputs(self) -> bool:
        """Whether any input was supplied as a device array."""
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
        """Refresh size, precision, and chunk axis from the solver.

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
            sysz.parameters,
            sysz.drivers,
            solver_instance.driver_coefficients_shape,
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

    def _convert_host_to_pinned(self) -> None:
        """Input slots hold caller-supplied arrays verbatim.

        Their backing is classified at attach time and never
        converted; non-pinned sources stage through the bounded
        pinned pool instead.
        """

    def _convert_host_to_numpy(self) -> None:
        """Input slots hold caller-supplied arrays verbatim.

        Chunked transfers stage slices straight from the attached
        array, so no conversion is needed.
        """

    def finalise(self, chunk_index: int, stream=None) -> None:
        """Finish input handling for a chunk."""

    def initialise(self, chunk_index: int, stream=None) -> None:
        """Copy a batch chunk of host data to device buffers.

        Parameters
        ----------
        chunk_index
            Indices for the chunk being initialized.

        Notes
        -----
        Pinned host arrays transfer directly and asynchronously.
        Everything else stages through pooled pinned buffers whose
        releases are event-driven via the transfer watcher, so this
        method never blocks on the stream: with the pool deep enough,
        the CPU stages the next chunk while the previous kernel runs.
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

        The host source may be a strided view (a chunk slice or a
        memmap) whose run extent is smaller than the device array's on
        the final chunk, so each block is copied into its buffer with
        shape-aware indexing; a flat copy would misalign the runs.
        :func:`staging_blocks` bounds every pinned buffer by
        ``HOST_STAGING_BYTES``. Each block's buffer is handed to the
        transfer watcher with its own event, so it returns to the pool
        as soon as its copy lands on the device.
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
                if CUDA_SIMULATION:
                    self._buffer_pool.release(buffer)
                else:
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
        """Wait for pending staging-buffer releases."""
        self._transfer_watcher.wait_all(timeout=timeout)

    def _invalidate_hook(self) -> None:
        """Drop device-input references alongside managed arrays."""
        super()._invalidate_hook()
        self._device_inputs.clear()

    def reset(self) -> None:
        """Clear all cached arrays and reset allocation tracking."""
        super().reset()
        self._transfer_watcher.shutdown()
        self._buffer_pool.clear()
        self._device_inputs.clear()

    def _teardown_cleanups(self):
        """Return transfer cleanup calls."""
        return [
            self._transfer_watcher.shutdown,
            self._buffer_pool.clear,
            *super()._teardown_cleanups(),
        ]
