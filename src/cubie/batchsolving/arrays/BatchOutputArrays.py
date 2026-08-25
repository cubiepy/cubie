"""Manage output array lifecycles for batch solver executions.

Published Classes
-----------------
:class:`OutputArrayContainer`
    Attrs container holding :class:`ManagedArray` fields for state,
    observables, summaries, status codes, and iteration counters.

:class:`OutputArrays`
    Concrete :class:`BaseArrayManager` subclass coordinating device-to-host
    transfers and async writeback for batch output data.

See Also
--------
:class:`~cubie.batchsolving.arrays.BaseArrayManager.BaseArrayManager`
    Abstract base providing allocation and transfer infrastructure.
:class:`~cubie.batchsolving.arrays.BatchInputArrays.InputArrays`
    Counterpart managing input arrays.
:class:`~cubie.batchsolving.writeback_watcher.WritebackWatcher`
    Background thread handling async writeback for chunked transfers.
:class:`~cubie.batchsolving.BatchSolverKernel.BatchSolverKernel`
    Primary consumer that owns output array instances.
"""

from typing import TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

from math import prod

from attrs import Factory as attrsFactory, define, field
from attrs.validators import instance_of as attrsval_instance_of
from cubie.cuda_simsafe import cuda
from numpy import (
    dtype as np_dtype,
    float32 as np_float32,
    floating as np_floating,
    int32 as np_int32,
    integer as np_integer,
    issubdtype as np_issubdtype,
)
from numpy.typing import NDArray

from cubie.outputhandling.output_sizes import BatchOutputSizes
from cubie.batchsolving.arrays.BaseArrayManager import (
    ArrayContainer,
    BaseArrayManager,
    ManagedArray,
    staging_blocks,
)
from cubie.batchsolving import ArrayTypes
from cubie.memory.chunk_buffer_pool import ChunkBufferPool
from cubie.memory.mem_manager import HOST_STAGING_BYTES
from cubie.batchsolving.writeback_watcher import WritebackWatcher
from cubie.cuda_simsafe import CUDA_SIMULATION

ChunkIndices = Union[slice, NDArray[np_integer]]


@define(slots=False)
class OutputArrayContainer(ArrayContainer):
    """Container for batch output arrays."""

    state: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 1, 1),
        )
    )
    observables: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 1, 1),
        )
    )
    state_summaries: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 1, 1),
        )
    )
    observable_summaries: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_float32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 1, 1),
        )
    )
    status_codes: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_int32,
            stride_order=("run",),
            default_shape=(1,),
        )
    )
    iteration_counters: ManagedArray = field(
        factory=lambda: ManagedArray(
            dtype=np_int32,
            stride_order=("time", "variable", "run"),
            default_shape=(1, 4, 1),
        )
    )

    @classmethod
    def host_factory(
        cls, memory_type: str = "pinned"
    ) -> "OutputArrayContainer":
        """
        Create a new host memory container.

        Parameters
        ----------
        memory_type
            Memory type for host arrays: "pinned" or "host".
            Default is "pinned" for non-chunked operation.

        Returns
        -------
        OutputArrayContainer
            A new container configured for the specified memory type.

        Notes
        -----
        Uses pinned (page-locked) memory to enable asynchronous
        device-to-host transfers with CUDA streams. Using ``"host"``
        memory type instead would result in pageable memory that blocks
        async transfers due to required intermediate buffering.
        """
        container = cls()
        container.set_memory_type(memory_type)
        return container

    @classmethod
    def device_factory(cls) -> "OutputArrayContainer":
        """
        Create a new device memory container.

        Returns
        -------
        OutputArrayContainer
            A new container configured for device memory.
        """
        container = cls()
        container.set_memory_type("device")
        return container


@define
class OutputArrays(BaseArrayManager):
    """
    Manage batch integration output arrays between host and device.

    This class manages the allocation, transfer, and synchronization of output
    arrays generated during batch integration operations. It handles state
    trajectories, observables, summary statistics, and per-run status codes.

    Parameters
    ----------
    _sizes
        Size specifications for the output arrays.
    host
        Container for host-side arrays.
    device
        Container for device-side arrays.

    Notes
    -----
    This class is initialized with a BatchOutputSizes instance (which is drawn
    from a solver instance using the from_solver factory method), which sets
    the allowable 3D array sizes from the ODE system's data and run settings.
    Once initialized, the object can be updated with a solver instance to
    update the expected sizes, check the cache, and allocate if required.
    """

    _sizes: BatchOutputSizes = field(
        factory=BatchOutputSizes,
        validator=attrsval_instance_of(BatchOutputSizes),
    )
    host: OutputArrayContainer = field(
        factory=OutputArrayContainer.host_factory,
        validator=attrsval_instance_of(OutputArrayContainer),
        init=True,
    )
    device: OutputArrayContainer = field(
        factory=OutputArrayContainer.device_factory,
        validator=attrsval_instance_of(OutputArrayContainer),
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
    _watcher: WritebackWatcher = field(factory=WritebackWatcher, init=False)
    # Outputs the kernel writes this run; None means transfer all.
    _active_names: Optional[frozenset] = field(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        """
        Configure default memory types after initialization.

        Notes
        -----
        Host containers use pinned memory to enable asynchronous
        device-to-host transfers with CUDA streams.
        """
        super().__attrs_post_init__()
        self.host.set_memory_type("pinned")
        self.device.set_memory_type("device")

    def update(self, solver_instance: "BatchSolverKernel") -> None:
        """
        Update output arrays from solver instance.

        Parameters
        ----------
        solver_instance
            The solver instance providing configuration and sizing information.

        """
        cf = solver_instance.compile_flags
        active = {"status_codes"}  # always written by the kernel
        if cf.save_state:
            active.add("state")
        if cf.save_observables:
            active.add("observables")
        if cf.summarise_state:
            active.add("state_summaries")
        if cf.summarise_observables:
            active.add("observable_summaries")
        if cf.save_counters:
            active.add("iteration_counters")
        self._active_names = frozenset(active)
        new_arrays = self.update_from_solver(solver_instance)
        if new_arrays is None:
            # Sizes unchanged; reallocate only after an invalidation.
            if self._needs_reallocation:
                self.allocate()
            return
        self.update_host_arrays(new_arrays, shape_only=True)
        self.allocate()

    @property
    def state(self) -> ArrayTypes:
        """Host state output array."""
        return self.host.state.array

    @property
    def observables(self) -> ArrayTypes:
        """Host observables output array."""
        return self.host.observables.array

    @property
    def state_summaries(self) -> ArrayTypes:
        """Host state summary output array."""
        return self.host.state_summaries.array

    @property
    def observable_summaries(self) -> ArrayTypes:
        """Host observable summary output array."""
        return self.host.observable_summaries.array

    @property
    def device_state(self) -> ArrayTypes:
        """Device state output array."""
        return self.device.state.array

    @property
    def device_observables(self) -> ArrayTypes:
        """Device observables output array."""
        return self.device.observables.array

    @property
    def device_state_summaries(self) -> ArrayTypes:
        """Device state summary output array."""
        return self.device.state_summaries.array

    @property
    def device_observable_summaries(self) -> ArrayTypes:
        """Device observable summary output array."""
        return self.device.observable_summaries.array

    @property
    def status_codes(self) -> ArrayTypes:
        """Host status code output array."""
        return self.host.status_codes.array

    @property
    def device_status_codes(self) -> ArrayTypes:
        """Device status code output array."""
        return self.device.status_codes.array

    @property
    def iteration_counters(self) -> ArrayTypes:
        """Host iteration counters output array."""
        return self.host.iteration_counters.array

    @property
    def device_iteration_counters(self) -> ArrayTypes:
        """Device iteration counters output array."""
        return self.device.iteration_counters.array

    @classmethod
    def from_solver(
        cls, solver_instance: "BatchSolverKernel"
    ) -> "OutputArrays":
        """
        Create an OutputArrays instance from a solver.

        Does not allocate arrays, just sets up size specifications.

        Parameters
        ----------
        solver_instance
            The solver instance to extract configuration from.

        Returns
        -------
        OutputArrays
            A new OutputArrays instance configured for the solver.
        """
        sizes = BatchOutputSizes.from_solver(solver_instance).nonzero
        return cls(
            sizes=sizes,
            precision=solver_instance.precision,
            memory_manager=solver_instance.memory_manager,
            stream_group=solver_instance.stream_group,
            memory_owner=solver_instance,
        )

    def update_from_solver(
        self, solver_instance: "BatchSolverKernel"
    ) -> Dict[str, NDArray[np_floating]]:
        """
        Update sizes and precision from solver, returning new host arrays.

        Only creates new pinned arrays when existing arrays do not match
        the expected shape and dtype. This avoids expensive pinned memory
        allocation on repeated solver runs with identical configurations.

        Parameters
        ----------
        solver_instance
            The solver instance to update from.

        Returns
        -------
        dict[str, numpy.ndarray] or None
            Host arrays with updated shapes for ``update_host_arrays``,
            or ``None`` when sizes are unchanged and the current host
            arrays already match.
        """
        # Buffers loaned to a collected result come back for reuse
        # before sizes are compared; a live result keeps its buffers
        # and fresh ones are allocated below.
        self.reclaim_or_release_loan()
        # Output sizes depend on num_runs, precision, the time dimensions
        # (output_length / summaries_length, which fold in duration and the
        # save/summarise intervals), the per-variable heights (which fold
        # in the output selection), and whether iteration counters are
        # saved. Skip the rebuild when all are unchanged; the current
        # host arrays already match.
        h = solver_instance.output_array_heights
        sig = (
            solver_instance.num_runs,
            solver_instance.precision,
            solver_instance.output_length,
            solver_instance.summaries_length,
            h.state,
            h.observables,
            h.state_summaries,
            h.observable_summaries,
            h.per_variable,
            solver_instance.save_counters,
        )
        if sig == self._size_sig:
            return None
        self._sizes = BatchOutputSizes.from_solver(solver_instance).nonzero
        self._precision = solver_instance.precision
        self.set_array_runs(solver_instance.num_runs)
        new_arrays = {}
        for name, slot in self.host.iter_managed_arrays():
            newshape = getattr(self._sizes, name)
            dtype = slot.dtype
            if np_issubdtype(dtype, np_floating):
                slot.dtype = self._precision
                dtype = slot.dtype
            # Fast path: skip allocation if existing array matches
            current = slot.array
            if (
                current is not None
                and current.shape == newshape
                and current.dtype == dtype
            ):
                new_arrays[name] = current
            else:
                # This runs before the chunk decision, so pinning is
                # deferred: a chunked solve must not hold full-size
                # pinned buffers. _convert_host_to_pinned repins small
                # unchunked slots once the decision lands; chunked and
                # oversized slots stay pageable (or disk-backed above
                # the spill policy) and D2H stays asynchronous by
                # staging through the pooled pinned buffers.
                nbytes = int(prod(newshape)) * np_dtype(dtype).itemsize
                base_type = self._memory_manager.choose_host_memory_type(
                    nbytes, self.host_spill_threshold, allow_pinned=False
                )
                new_array = self._memory_manager.create_host_array(
                    newshape, dtype, base_type,
                    spill_directory=self.spill_directory,
                )
                slot.memory_type = base_type
                new_arrays[name] = new_array
        for name, slot in self.device.iter_managed_arrays():
            dtype = slot.dtype
            if np_issubdtype(dtype, np_floating):
                slot.dtype = self._precision
        self._size_sig = sig
        return new_arrays

    def finalise(self, chunk_index: int, stream=None) -> None:
        """Queue device-to-host transfers for a chunk.

        Parameters
        ----------
        chunk_index
            Indices for the chunk being finalized.

        Notes
        -----
        Host slices are made contiguous before transfer to ensure
        compatible strides with device arrays. For chunked mode, data
        is transferred to pooled pinned buffers and submitted to the
        watcher thread for async writeback. For non-chunked mode,
        the writeback call is made immediately (but will happen
        asynchronously).
        """
        from_ = []
        to_ = []
        if stream is None:
            stream = self._memory_manager.get_stream(self)
        active = self._active_names

        for array_name, slot in self.host.iter_managed_arrays():
            # Skip inactive outputs: their device array is a (1, 1, 1)
            # placeholder, so transferring it just wastes a d2h dispatch.
            if active is not None and array_name not in active:
                continue
            device_array = self.device.get_array(array_name)
            host_array = slot.array

            host_slice = (
                slot.chunk_slice(chunk_index)
                if slot.needs_chunked_transfer
                else host_array
            )
            needs_staging = slot.needs_chunked_transfer or (
                self._requires_staging(host_slice, slot.memory_type)
            )
            if needs_staging:
                self._stage_array(
                    array_name, device_array, host_slice, stream
                )
            else:
                to_.append(host_slice)
                from_.append(device_array)

        if from_:
            self.from_device(from_, to_, stream=stream)

    def _stage_array(
        self, array_name, device_array, host_array, stream
    ) -> None:
        """Stage one device output through pooled pinned buffers.

        The host target may be a strided view (a chunk slice or a
        memmap), so completed blocks are written through strided
        assignment; flattening such a view would silently copy it and
        discard the writeback. :func:`staging_blocks` bounds every
        pinned buffer by ``HOST_STAGING_BYTES``, and the buffer is
        trimmed to the host block's shape because the device array
        can carry extra run-axis padding on the final chunk. Each
        block is handed to the writeback watcher with its own event:
        the watcher copies it to the host target and releases the
        buffer as soon as its transfer lands, so this method never
        blocks on the stream and the drain of one chunk overlaps the
        next chunk's kernel.
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
                self.from_device(
                    [device_block], [buffer.array], stream=stream
                )
                if CUDA_SIMULATION:
                    trim = tuple(
                        slice(0, extent) for extent in host_block.shape
                    )
                    host_block[...] = buffer.array[trim]
                    self._buffer_pool.release(buffer)
                else:
                    event = cuda.event()
                    event.record(stream)
                    self._watcher.submit(
                        event=event,
                        buffer=buffer,
                        target_array=host_block,
                        buffer_pool=self._buffer_pool,
                        array_name=array_name,
                        data_shape=host_block.shape,
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
        """Wait for all pending async writebacks to complete.

        Parameters
        ----------
        timeout
            Maximum seconds to wait. None waits indefinitely.

        Notes
        -----
        Only applies to chunked mode with watcher-based writebacks.
        """
        self._watcher.wait_all(timeout=timeout)

    def initialise(self, chunk_index: int, stream=None) -> None:
        """
        Initialize device arrays before kernel execution.

        Parameters
        ----------
        chunk_index
            Indices for the chunk being initialized.

        Notes
        -----
        No initialization to zeros is needed unless chunk calculations in time
        leave a dangling sample at the end, which is possible but not expected.
        """
        pass

    def reset(self) -> None:
        """Drain transfers and clear all arrays and staging buffers."""
        super().reset()
        self._watcher.shutdown()
        self._buffer_pool.clear()

    def _teardown_cleanups(self):
        """Return writeback cleanup calls."""
        return [
            self._watcher.shutdown,
            self._buffer_pool.clear,
            *super()._teardown_cleanups(),
        ]
