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

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

from math import prod

from attrs import Factory as attrsFactory, define, field
from attrs.validators import instance_of as attrsval_instance_of
from cubie._cudasim_extensions import cuda
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
        Create a container for host-side output arrays.

        Parameters
        ----------
        memory_type
            Memory type to give every slot until a buffer is attached:
            ``"pinned"`` or ``"host"``.

        Returns
        -------
        OutputArrayContainer
            Host-side container instance.

        Notes
        -----
        Each slot records the backing of the buffer attached to it,
        so this memory type lasts only until the first attach.
        """
        container = cls()
        container.set_memory_type(memory_type)
        return container

    @classmethod
    def device_factory(cls) -> "OutputArrayContainer":
        """
        Create a container for device-side output arrays.

        Returns
        -------
        OutputArrayContainer
            Device-side container instance.
        """
        container = cls()
        container.set_memory_type("device")
        return container


@define
class OutputArrays(BaseArrayManager):
    """
    Manage batch integration output arrays between host and device.

    The manager allocates the device arrays the kernel writes: state and
    observable trajectories, their summaries, per-run status codes and
    iteration counters. For a solve that returns host results, it
    copies the outputs the kernel writes back to host buffers after
    each chunk, then lends those buffers to the
    :class:`~cubie.batchsolving.solveresult.SolveResult`.

    Parameters
    ----------
    sizes
        Size specifications for the output arrays.
    host
        Container for host-side arrays.

    Notes
    -----
    Build an instance with :meth:`from_solver`, then call
    :meth:`update` before each solve. Its other parameters are those of
    :class:`.BaseArrayManager`.
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
    # Whether this run copies its outputs to the host.
    _transfer: bool = field(default=True, init=False)

    def __attrs_post_init__(self) -> None:
        """Register the manager and set the containers' memory types.

        The host slots start as ``"pinned"`` and the device slots as
        ``"device"``. Each host slot then records the backing of the
        buffer attached to it.
        """
        super().__attrs_post_init__()
        self.host.set_memory_type("pinned")
        self.device.set_memory_type("device")

    def update(
        self,
        solver_instance: "BatchSolverKernel",
        transfer_outputs: bool = True,
    ) -> None:
        """
        Size the device outputs for a run and queue their allocation.

        Record which outputs the kernel writes in this run, so that
        :meth:`finalise` copies only those outputs.

        Parameters
        ----------
        solver_instance
            The solver instance providing configuration and sizing information.
        transfer_outputs
            Whether the run copies its outputs to host buffers.

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
        self._transfer = bool(transfer_outputs)
        self.update_from_solver(solver_instance)
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
    ) -> Optional[bool]:
        """
        Refresh sizes and precision from the solver.

        Parameters
        ----------
        solver_instance
            The solver instance to update from.

        Returns
        -------
        bool or None
            ``True`` when the sizes changed and every device output is
            queued for reallocation, ``None`` when they are unchanged.
        """
        # Only a transferring run takes a dead result's buffers back.
        if self._transfer:
            self.reclaim_or_release_loan()
        # Output sizes depend on num_runs, precision, the time dimensions
        # (output_length / summaries_length, which fold in duration and the
        # save/summarise intervals), the per-variable heights (which fold
        # in the output selection), and whether iteration counters are
        # saved.
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
        for name, slot in self.host.iter_managed_arrays():
            if np_issubdtype(slot.dtype, np_floating):
                slot.dtype = self._precision
            current = slot.array
            if current is not None and not self._host_fits(name, slot):
                # A host buffer of another shape is dropped here.
                self._memory_manager.release_host_array(current)
                slot.array = None
        for name, slot in self.device.iter_managed_arrays():
            if np_issubdtype(slot.dtype, np_floating):
                slot.dtype = self._precision
            if name not in self._needs_reallocation:
                self._needs_reallocation.append(name)
        self._size_sig = sig
        return True

    def _request_shape(self, label: str) -> tuple:
        """Return the device shape of ``label`` from the output sizes.

        Parameters
        ----------
        label
            Name of the output array.

        Returns
        -------
        tuple
            The shape in the manager's sizes, whatever host buffer is
            attached.
        """
        return tuple(getattr(self._sizes, label))

    def _host_fits(self, label: str, slot: ManagedArray) -> bool:
        """Return whether the host array has the sized shape and dtype.

        Parameters
        ----------
        label
            Name of the output array.
        slot
            The output's host wrapper.

        Returns
        -------
        bool
            False when no host array is attached.
        """
        array = slot.array
        return (
            array is not None
            and array.shape == tuple(getattr(self._sizes, label))
            and array.dtype == slot.dtype
        )

    def _after_allocation(self) -> None:
        """Give the outputs host buffers if the run copies them back."""
        if self._transfer:
            self._ensure_host_arrays()

    def _ensure_host_arrays(self) -> None:
        """Give every output a host buffer the transfer can land in.

        A buffer is kept when it has the sized shape and dtype and a
        backing the run partition accepts: pinned only unchunked,
        pageable only when the policy would not pin it, memmap always.
        A new pinned buffer is zeroed only for an output no transfer
        writes.
        """
        allow_pinned = not self.is_chunked
        manager = self._memory_manager
        active = self._active_names
        for name, slot in self.host.iter_managed_arrays():
            shape = tuple(getattr(self._sizes, name))
            dtype = slot.dtype
            nbytes = int(prod(shape)) * np_dtype(dtype).itemsize
            wanted = manager.choose_host_memory_type(
                nbytes, allow_pinned=allow_pinned
            )
            if self._host_fits(name, slot):
                backing = slot.memory_type
                if backing == "memmap" or backing == wanted:
                    continue
                if backing == "host" and wanted != "pinned":
                    continue
            backing = wanted
            array = None
            if wanted == "pinned":
                array = manager.allocate_pinned_array(shape, dtype)
                if array is None:
                    # The pinned budget refused; stage through the pool.
                    backing = "host"
                elif active is not None and name not in active:
                    array.fill(0)
            if array is None:
                array = manager.create_host_array(shape, dtype, backing)
            manager.release_host_array(slot.array)
            slot.array = array
            slot.memory_type = backing

    def finalise(self, chunk_index: int, stream=None) -> None:
        """Queue the copies of a chunk's outputs back to the host.

        Copy only the outputs the kernel writes in this run. Before
        the first chunk's copies, give every output a host buffer.

        Parameters
        ----------
        chunk_index
            Index of the chunk that has just been launched.
        stream
            Stream to queue the copies on, or ``None`` for the
            manager's stream.

        Notes
        -----
        Copy an unchunked output with a pinned host buffer straight
        into that buffer. Stage every other output, and every chunk,
        through pooled pinned buffers. The writeback watcher copies
        each staged block into the host buffer once its transfer
        completes. This method does not wait for the GPU, so the copy
        of one chunk overlaps the next chunk's kernel.
        """
        from_ = []
        to_ = []
        if stream is None:
            stream = self._memory_manager.get_stream(self)
        if chunk_index == 0:
            self._ensure_host_arrays()
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

        Cut the arrays into blocks with :func:`staging_blocks`, and
        queue each device block's copy into a pinned buffer. Hand each
        buffer to the writeback watcher with its own event and the
        host block as its target. Once the block's copy completes, the
        watcher writes the buffer, trimmed to the host block's shape,
        into the host target and releases the buffer. The host target
        can be a strided view, such as a chunk slice or a memmap, and
        on the final chunk the device array can hold more runs than
        the host block. This method does not wait for the GPU.

        Parameters
        ----------
        array_name
            Name of the output array.
        device_array
            Device array to copy from.
        host_array
            Host array or chunk slice to write into.
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
                self.from_device(
                    [device_block], [buffer.array], stream=stream
                )
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
        """Wait until the writeback watcher writes every staged block.

        This waits only for staged outputs, because the watcher does
        not handle the outputs copied straight into pinned buffers.

        Parameters
        ----------
        timeout
            Maximum seconds to wait, or ``None`` to wait until done.
        """
        self._watcher.wait_all(timeout=timeout)

    def initialise(self, chunk_index: int, stream=None) -> None:
        """
        Do nothing, because outputs need no work before a chunk.

        Parameters
        ----------
        chunk_index
            Index of the chunk about to be launched.
        stream
            Unused.
        """
        pass

    def reset(self) -> None:
        """Release the arrays and buffers and stop the watcher."""
        super().reset()
        self._watcher.shutdown()
        self._buffer_pool.clear()

    def _teardown_cleanups(self):
        """Return the calls that stop the watcher and free the buffers.

        Returns
        -------
        list of callable
            Calls that shut down the writeback watcher, free the
            staging buffers and drop the device arrays.
        """
        return [
            self._watcher.shutdown,
            self._buffer_pool.clear,
            *super()._teardown_cleanups(),
        ]
