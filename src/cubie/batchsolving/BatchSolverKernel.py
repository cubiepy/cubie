# -*- coding: utf-8 -*-
"""CUDA batch solver kernel utilities.

Published Classes
-----------------
:class:`RunParams`
    Frozen attrs dataclass holding run duration, warmup, t0, and chunking
    metadata.

:class:`BatchSolverKernel`
    :class:`CUDAFactory` subclass that compiles and launches the integration
    kernel for batched GPU solves.

Notes
-----
Chunking is performed along the run axis when memory constraints require
splitting the batch. This chunking is automatic and transparent to users.

See Also
--------
:class:`~cubie.batchsolving.solver.Solver`
    User-facing API that delegates to this kernel.
:class:`~cubie.integrators.SingleIntegratorRun.SingleIntegratorRun`
    Generates the compiled loop function consumed by the kernel.
:class:`~cubie.batchsolving.arrays.BatchInputArrays.InputArrays`
    Input array manager owned by the kernel.
:class:`~cubie.batchsolving.arrays.BatchOutputArrays.OutputArrays`
    Output array manager owned by the kernel.
"""

import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
from warnings import warn
from pathlib import Path
from weakref import finalize

from numpy import (
    ceil as np_ceil,
    float64 as np_float64,
    floating,
    int32 as np_int32,
)
from cubie.cuda_simsafe import cuda, float64
from cubie.cuda_simsafe import int32

from attrs import define, field, evolve

from cubie.odesystems import SymbolicODE
from cubie.cuda_backend import IS_MLIR
from cubie.backend.utils import (
    LAUNCH_BLOCKSIZES,
    SHARED_SKEW_BYTES,
    active_blocks_per_multiprocessor,
    compile_kernel_specialization,
    device_hardware,
    kernel_resources,
    max_shared_memory_per_block,
)
from cubie.batchsolving.optimize import (
    default_launch,
    resident_blocks_within_l2,
)
from cubie.cuda_simsafe import is_cudasim_enabled
from cubie.cubie_cache import CUBIECache

from cubie.time_logger import CUDAEvent, default_timelogger
from numpy.typing import NDArray

from cubie.array_interpolator import ArrayInterpolator
from cubie.memory import default_memmgr
from cubie.memory.mem_manager import (
    ALL_MEMORY_MANAGER_PARAMETERS,
    defer_instance_teardown,
)
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie.batchsolving.arrays.BatchInputArrays import InputArrays
from cubie.batchsolving.arrays.BatchOutputArrays import (
    OutputArrays,
)
from cubie.batchsolving.BatchSolverConfig import (
    ALL_KERNEL_PARAMETERS,
    DEFAULT_BLOCKSIZE,
    ActiveOutputs,
    BatchSolverConfig,
)
from cubie.batchsolving._utils import (
    format_time_domain_label,
    name_and_compile_kernel,
)
from cubie.odesystems.baseODE import BaseODE
from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.outputhandling.output_sizes import OutputArrayHeights
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie._utils import (
    build_config,
    getype_validator,
    merge_kwargs_into_settings,
    precision_converter,
    precision_validator,
    unpack_dict_values,
)

if TYPE_CHECKING:
    from cubie.memory import MemoryManager
    from cubie.memory.array_requests import ArrayResponse


DEFAULT_MEMORY_SETTINGS = {
    "memory_manager": default_memmgr,
    "stream_group": "solver",
    "mem_proportion": None,
}


@define(frozen=True)
class RunParams:
    """Run parameters with optional chunking metadata.

    Chunking always occurs along the run axis.

    Parameters
    ----------
    duration : float
        Full duration of the simulation window.
    warmup : float
        Full warmup time before the main simulation.
    t0 : float
        Initial integration time.
    runs : int
        Total number of runs in the batch.
    num_chunks : int, default=1
        Number of chunks the batch is divided into.
    chunk_length : int, default=0
        Number of runs per chunk (except possibly the last).
    precision : type, default=numpy.float64
        Run precision; :attr:`time_scalars` casts to this dtype.

    Notes
    -----
    When num_chunks=1, no chunking has occurred.
    When num_chunks>1, chunk_length represents the standard chunk size.
    """

    duration: float = field(validator=getype_validator(float, 0.0))
    warmup: float = field(validator=getype_validator(float, 0.0))
    t0: float = field(validator=getype_validator(float, 0.0))
    runs: int = field(validator=getype_validator(int, 1))
    num_chunks: int = field(
        default=1, repr=False, validator=getype_validator(int, 1)
    )
    chunk_length: int = field(
        default=0, repr=False, validator=getype_validator(int, 0)
    )
    precision: type = field(
        default=np_float64,
        repr=False,
        validator=precision_validator,
        converter=precision_converter,
    )

    @property
    def time_scalars(self) -> Tuple[floating, floating, floating]:
        """Return duration, warmup, and t0 cast to the run precision."""
        return (
            self.precision(self.duration),
            self.precision(self.warmup),
            self.precision(self.t0),
        )

    def __getitem__(self, index: int) -> "RunParams":
        """Return RunParams for a specific chunk.

        Parameters
        ----------
        index : int
            Chunk index (0-based).

        Returns
        -------
        RunParams
            New RunParams instance with runs set to chunk size.

        Raises
        ------
        IndexError
            If index is out of range [0, num_chunks).

        Notes
        -----
        For the last chunk (index == num_chunks - 1), the number of runs
        is calculated as runs - (num_chunks - 1) * chunk_length to handle
        the "dangling" chunk case.
        """
        # Validation
        if index < 0 or index >= self.num_chunks:
            raise IndexError(
                f"Chunk index {index} out of range "
                f"(valid range: 0 to {self.num_chunks - 1})"
            )

        if index == self.num_chunks - 1:
            # Last chunk: calculate remaining runs
            chunk_runs = self.runs - (self.num_chunks - 1) * self.chunk_length
        else:
            chunk_runs = self.chunk_length

        return evolve(self, runs=chunk_runs)

    def update_from_allocation(self, response: "ArrayResponse") -> "RunParams":
        """Update with chunking metadata from allocation response.

        Parameters
        ----------
        response : ArrayResponse
            Allocation response containing chunking information.

        Returns
        -------
        RunParams
            New RunParams instance with updated chunking metadata.

        Raises
        ------
        ValueError
            If the partition holds fewer runs than the batch.

        Notes
        -----
        Extracts num_chunks and chunk_length from the response.
        """
        capacity = response.chunks * response.chunk_length
        if capacity < self.runs:
            raise ValueError(
                f"Allocation partition of {response.chunks} chunk(s) x "
                f"{response.chunk_length} run(s) holds {capacity} runs, "
                f"fewer than the batch's {self.runs}."
            )
        return evolve(
            self,
            num_chunks=response.chunks,
            chunk_length=response.chunk_length,
        )


@define(frozen=True)
class DurationCounts:
    """Event counts of one build for one duration."""

    output_length: int
    summaries_length: int
    save_events: int
    summary_samples: int


@define()
class BatchSolverCache(CUDADispatcherCache):
    """Compiled kernel plus the per-build memos of its consumers."""

    solver_kernel: Union[int, Callable] = field(default=-1)
    signature: Optional[Tuple] = field(default=None)
    specializations: Dict[Tuple, Tuple] = field(factory=dict)
    output_array_heights: Optional[OutputArrayHeights] = field(default=None)
    duration_counts: Dict[float, DurationCounts] = field(factory=dict)
    launch_geometries: Dict[Tuple, Tuple[int, int]] = field(factory=dict)
    default_launches: Dict[Tuple, Tuple[int, Optional[int]]] = field(
        factory=dict
    )
    time_domain_legend: Dict[int, str] = field(factory=dict)
    summaries_legend: Dict[int, str] = field(factory=dict)


DYNAMIC_SHARED_PAD_STEP = 256
"""Bytes the residency pad steps by."""


class BatchSolverKernel(CUDAFactory):
    """Factory for CUDA kernel which coordinates a batch integration.

    Parameters
    ----------
    system
        ODE system describing the problem to integrate.
    **settings
        Loop, step, controller, algorithm, output, memory, cache and
        kernel settings as one flat dict.

    Attributes
    ----------
    resident_blocks
        Blocks per SM on the GPU, set by ``auto_performance`` and
        ``Solver.optimize``.

    Notes
    -----
    The kernel delegates integration logic to :class:`SingleIntegratorRun`
    instances and expects upstream APIs to perform batch construction. It
    executes the compiled loop function against kernel-managed memory slices
    and distributes work across GPU threads for each input batch.
    """

    settings_keys = frozenset(ALL_KERNEL_PARAMETERS)

    def __init__(self, system: "SymbolicODE", **settings: Any) -> None:
        super().__init__()
        self._disk_cache = None
        self._disk_cache_settings = None
        settings, _ = unpack_dict_values(settings)
        memory_settings, _ = merge_kwargs_into_settings(
            settings, ALL_MEMORY_MANAGER_PARAMETERS
        )

        precision = system.precision

        # Initialize run parameters with defaults
        self.run_params = RunParams(
            duration=precision(0.0),
            warmup=precision(0.0),
            t0=precision(0.0),
            runs=1,
            precision=precision,
        )

        # CUDA event tracking for timing
        self._cuda_events: List = []
        self._gpu_workload_event: Optional[CUDAEvent] = None
        self._cuda_events_key = None

        self._closed = False
        self._last_stream = None
        self._work_complete = True
        self._specialization_cache = None
        self._memory_manager = self._setup_memory_manager(memory_settings)
        self.resident_blocks = None

        self.driver_interpolator = ArrayInterpolator(
            precision=precision,
            memory_manager=self._memory_manager,
        )
        self.driver_interpolator.update(settings, silent=True)

        system_name = system.name
        system_hash = system.fn_hash
        if system_name == system_hash:
            system_name = f"unnamed_{system_hash[:8]}"
        self._system_name = system_name

        self.single_integrator = SingleIntegratorRun(
            system,
            drivers_fn=self.driver_interpolator.drivers_fn,
            driver_derivative_fn=self.driver_interpolator.driver_derivative_fn,
            **settings,
        )
        run = self.single_integrator
        self.setup_compile_settings(
            build_config(
                BatchSolverConfig,
                required={
                    "precision": precision,
                    "loop_fn": run.device_function,
                    "compile_flags": run.output_compile_flags,
                    "coefficients_shape": (
                        self.driver_interpolator.coefficients_shape
                    ),
                },
                **settings,
            )
        )

        self.input_arrays = InputArrays.from_solver(self)
        self.output_arrays = OutputArrays.from_solver(self)

        self.output_arrays.update(self)

        self._known_system_config = system.compile_settings

    def _setup_memory_manager(
        self, settings: Dict[str, Any]
    ) -> "MemoryManager":
        """Register the kernel with a memory manager instance.

        Parameters
        ----------
        settings
            Mapping of memory configuration options recognised by the memory
            manager.

        Returns
        -------
        MemoryManager
            Memory manager configured for solver allocations.
        """

        merged_settings = DEFAULT_MEMORY_SETTINGS.copy()
        merged_settings.update(settings)
        memory_manager = merged_settings["memory_manager"]
        memory_manager.register(
            self,
            stream_group=merged_settings["stream_group"],
            proportion=merged_settings["mem_proportion"],
            allocation_ready_hook=self._on_allocation,
            owner=self,
        )
        settings = memory_manager.get_registration(self)
        self._finalizer = finalize(
            self,
            defer_instance_teardown,
            memory_manager,
            id(self),
            settings,
            (),
        )
        return memory_manager

    def _setup_cuda_events(self, chunks: int) -> None:
        """Provide the timing events for this run.

        Parameters
        ----------
        chunks : int
            Number of chunks to process

        Notes
        -----
        One workload event plus three per chunk. While timing is on
        they are kept between runs and rebuilt when the chunk count or
        logger verbosity changes; with timing off each run gets fresh
        no-op events.
        """
        verbosity = default_timelogger.verbosity
        key = (chunks, verbosity)
        if verbosity is not None and key == self._cuda_events_key:
            self._gpu_workload_event.register()
            for event in self._cuda_events:
                event.register()
            return
        self._gpu_workload_event = CUDAEvent("gpu_workload")
        self._cuda_events = []
        for i in range(chunks):
            h2d_event = CUDAEvent(f"h2d_transfer_chunk_{i}")
            kernel_event = CUDAEvent(f"kernel_chunk_{i}")
            d2h_event = CUDAEvent(f"d2h_transfer_chunk_{i}")
            self._cuda_events.extend([h2d_event, kernel_event, d2h_event])
        self._cuda_events_key = key

    def _get_chunk_events(self, chunk_idx: int) -> Tuple:
        """Get the three CUDA events for a specific chunk.

        Parameters
        ----------
        chunk_idx : int
            Chunk index (0-based)

        Returns
        -------
        tuple
            (h2d_event, kernel_event, d2h_event) for the chunk
        """
        base_idx = chunk_idx * 3
        return (
            self._cuda_events[base_idx],
            self._cuda_events[base_idx + 1],
            self._cuda_events[base_idx + 2],
        )

    def run(
        self,
        inits: NDArray[floating],
        params: NDArray[floating],
        duration: float,
        blocksize: Optional[int] = None,
        warmup: float = 0.0,
        t0: float = 0.0,
        transfer_outputs: bool = True,
    ) -> None:
        """Execute the solver kernel for batch integration.

        Chunking is performed along the run axis when memory constraints
        require splitting the batch.

        Parameters
        ----------
        inits
            Initial conditions with shape ``(n_states, n_runs)``. Host
            or device arrays are accepted; device arrays are used in
            place with no host-to-device transfer.
        params
            Parameter table with shape ``(n_params, n_runs)``. Host or
            device arrays are accepted, as for ``inits``.
        duration
            Duration of the simulation window.
        blocksize
            CUDA block size for this launch; ``None`` uses the
            ``blocksize`` setting, or the automatic launch when that
            is unset.
        warmup
            Warmup time before the main simulation.
        t0
            Initial integration time.
        transfer_outputs
            When ``True`` (default), output arrays are copied
            device-to-host after each chunk. ``False`` leaves results
            in the device output buffers and touches no host output
            buffer; the run must fit in a single chunk.

        Notes
        -----
        The kernel prepares array views, queues allocations, and executes the
        device loop on each chunked workload. Shared-memory demand may reduce
        the block size automatically, emitting a warning when the limit drops
        below a warp. Every launch and transfer runs on this kernel's
        memory-manager stream (:attr:`stream`); there is no per-run
        stream selection.

        Raises
        ------
        RuntimeError
            If the kernel has been closed.
        ValueError
            Drivers declared but no evaluator wired; chunked batch with
            ``transfer_outputs=False`` or device inputs.
        """
        if self._closed:
            raise RuntimeError(
                "This solver has been closed and its GPU resources "
                "released; build a new Solver to run again."
            )
        if self.system.sizes.drivers and (
            self.single_integrator._loop.drivers_fn is None
        ):
            raise ValueError(
                f"System declares {self.system.sizes.drivers} driver(s) "
                "but no driver samples are given; pass drivers= to "
                "solve."
            )
        stream = self.stream
        self._memory_manager.begin_work(self)
        try:
            self._execute_run(
                inits,
                params,
                duration,
                blocksize,
                stream,
                warmup,
                t0,
                transfer_outputs,
            )
        finally:
            self._memory_manager.end_work(self, stream)

    def compile(
        self,
        duration: float,
        warmup: float = 0.0,
        t0: float = 0.0,
    ) -> None:
        """Record the time parameters and compile the kernel.

        Parameters
        ----------
        duration
            Duration of the simulation window.
        warmup
            Warmup time before the main simulation.
        t0
            Initial integration time.
        """
        if self._closed:
            raise RuntimeError(
                "This solver has been closed and its GPU resources "
                "released; build a new Solver to run again."
            )
        self.run_params = evolve(
            self.run_params,
            duration=np_float64(duration),
            warmup=np_float64(warmup),
            t0=np_float64(t0),
            precision=self.single_integrator.precision,
        )
        self._compile_specialization()

    def _compile_specialization(
        self, args: Optional[Tuple] = None
    ) -> Callable:
        """Compile the selected specialization and return the kernel."""
        dispatcher = self.kernel
        if args is None:
            if self._cache.signature is not None:
                return dispatcher
            # Use a dummy allocation to avoid full host-device transfer.
            args = self._specialization_args()
        if IS_MLIR or is_cudasim_enabled():
            self._cache.signature = compile_kernel_specialization(
                dispatcher, args
            )
            return dispatcher
        # Array types are cached by Numba; scalar types are fixed per build.
        array_types = tuple(
            getattr(array, "_numba_type_", None)
            or dispatcher.typeof_pyval(array)
            for array in args[:9]
        )
        signature = self._cache.specializations.get(array_types)
        if signature is None:
            signature = compile_kernel_specialization(dispatcher, args)
            self._cache.specializations[array_types] = signature
        self._cache.signature = signature
        return dispatcher

    @property
    def signature(self) -> Tuple:
        """Signature used for launch sizing and resource queries."""
        if not self._cache_valid or self._cache.signature is None:
            self._compile_specialization()
        return self._cache.signature

    def _specialization_args(self) -> Tuple:
        """Return arguments for kernel compilation."""
        precision = self.precision
        cached = self._specialization_cache
        if cached is not None and cached[0] is precision:
            return cached[1]
        stream = self.stream
        allocate = self.memory_manager.allocate

        def unit(ndim, dtype):
            return allocate((1,) * ndim, dtype, "device", stream=stream)

        args = (
            unit(2, precision),  # initial values
            unit(2, precision),  # parameters
            unit(3, precision),  # driver coefficients
            unit(3, precision),  # state
            unit(3, precision),  # observables
            unit(3, precision),  # state summaries
            unit(3, precision),  # observable summaries
            unit(3, np_int32),  # iteration counters
            unit(1, np_int32),  # status codes
            precision(0.0),  # duration
            precision(0.0),  # warmup
            precision(0.0),  # t0
            np_int32(0),  # save count
            np_int32(0),  # summary count
            1,  # runs
        )
        self._specialization_cache = (precision, args)
        return args

    def _duration_counts(self, duration: float) -> DurationCounts:
        """Return the event counts for ``duration``, memoised per build."""
        counts = self.get_cached_output("duration_counts")
        key = float(duration)
        entry = counts.get(key)
        if entry is None:
            integrator = self.single_integrator
            entry = DurationCounts(
                output_length=integrator.output_length(duration),
                summaries_length=integrator.summaries_length(duration),
                save_events=integrator.save_event_count(duration),
                summary_samples=integrator.summary_sample_count(duration),
            )
            counts[key] = entry
        return entry

    def kernel_is_cached(self) -> bool:
        """Whether the disk cache holds this configuration's kernel;
        ``False`` when caching is off."""
        if self._closed:
            raise RuntimeError(
                "This solver has been closed and its GPU resources "
                "released; build a new Solver to run again."
            )
        # Building the dispatcher attaches the disk cache.
        self.kernel
        disk_cache = self._disk_cache
        return disk_cache is not None and disk_cache.holds_kernel()

    def _kernel_launch_args(self, chunk_run_params: RunParams) -> Tuple:
        """Return the kernel's positional arguments for one chunk."""
        duration, warmup, t0 = chunk_run_params.time_scalars
        counts = self._duration_counts(duration)
        save_count = np_int32(counts.save_events)
        summary_count = np_int32(counts.summary_samples)
        return (
            self.input_arrays.device_initial_values,
            self.input_arrays.device_parameters,
            self.input_arrays.device_driver_coefficients,
            self.output_arrays.device_state,
            self.output_arrays.device_observables,
            self.output_arrays.device_state_summaries,
            self.output_arrays.device_observable_summaries,
            self.output_arrays.device_iteration_counters,
            self.output_arrays.device_status_codes,
            duration,
            warmup,
            t0,
            save_count,
            summary_count,
            chunk_run_params.runs,
        )

    def _prepare_batch(
        self,
        inits: NDArray[floating],
        params: NDArray[floating],
        duration: float,
        warmup: float,
        t0: float,
        stream: Optional[Any],
        transfer_outputs: bool = True,
    ) -> None:
        """Set run parameters, refresh settings, and queue allocations."""
        # Time parameters always use float64 for accumulation accuracy
        duration = np_float64(duration)

        # The partition follows the live arrays; an allocation replaces it.
        self.run_params = evolve(
            self.run_params,
            duration=duration,
            warmup=np_float64(warmup),
            t0=np_float64(t0),
            runs=inits.shape[1],
            precision=self.single_integrator.precision,
        )

        # An attached table is a cached build output: nothing to upload.
        driver_coefficients = self.driver_interpolator.coefficients
        attached = self.input_arrays.host.driver_coefficients.array
        if driver_coefficients is attached:
            driver_coefficients = None
        self.input_arrays.update(self, inits, params, driver_coefficients)
        self.output_arrays.update(self, transfer_outputs)

        # Process allocations into chunks
        self.memory_manager.allocate_queue(self, stream=stream)

    def _execute_run(
        self,
        inits: NDArray[floating],
        params: NDArray[floating],
        duration: float,
        blocksize: Optional[int],
        stream: Optional[Any],
        warmup: float,
        t0: float,
        transfer_outputs: bool,
    ) -> None:
        """Allocate, chunk, and launch the batch kernel."""
        self._last_stream = stream
        self._work_complete = False

        self._prepare_batch(
            inits, params, duration, warmup, t0, stream, transfer_outputs
        )

        # ------------ from here on dimensions are "chunked" -----------------
        # self.run_params is updated in the on_allocation callback.
        chunks = self.run_params.num_chunks

        if chunks > 1:
            # Host arrays are the stitch target for chunked runs, so
            # device-resident results and inputs cannot span chunks.
            # This is the only place that can guard device inputs: an
            # attached slot queues no allocation, so InputArrays never
            # learns the run's chunk count.
            if not transfer_outputs:
                raise ValueError(
                    "Device-resident results require the batch to fit "
                    "in a single chunk, but this run is split into "
                    f"{chunks} chunks. Reduce the batch size or use a "
                    "host solve."
                )
            if self.input_arrays.has_device_inputs:
                raise ValueError(
                    "Device-array inputs require the batch to fit in "
                    "a single chunk, but this run is split into "
                    f"{chunks} chunks. Pass host arrays or reduce the "
                    "batch size."
                )

        first_chunk_args = self._kernel_launch_args(self.run_params[0])
        if not IS_MLIR:
            self._compile_specialization(first_chunk_args)
        blocksize, dynamic_sharedmem = self.launch_geometry(
            blocksize, runs=self.run_params[0].runs
        )
        threads_per_loop = self.single_integrator.threads_per_step
        runsperblock = int(blocksize / self.single_integrator.threads_per_step)

        # Setup CUDA events for timing (no-op when verbosity is None)
        self._setup_cuda_events(chunks)

        # Record start of overall GPU workload
        self._gpu_workload_event.record_start(stream)
        for i in range(chunks):
            # Get parameters for this specific chunk
            chunk_run_params = self.run_params[i]

            # Use the chunk-local run count
            runs = chunk_run_params.runs

            # Recompute blocks needed for this chunk's actual run count
            chunk_blocks = int(max(1, np_ceil(runs / blocksize)))

            # Get events for this chunk
            h2d_event, kernel_event, d2h_event = self._get_chunk_events(i)

            # h2d transfer timing
            h2d_event.record_start(stream)
            self.input_arrays.initialise(i, stream=stream)
            self.output_arrays.initialise(i, stream=stream)
            h2d_event.record_end(stream)

            # Kernel execution timing
            kernel_event.record_start(stream)
            args = (
                first_chunk_args
                if i == 0
                else self._kernel_launch_args(chunk_run_params)
            )
            self.kernel[
                chunk_blocks,
                (threads_per_loop, runsperblock),
                stream,
                dynamic_sharedmem,
            ](*args)
            kernel_event.record_end(stream)

            # d2h transfer timing
            d2h_event.record_start(stream)
            self.input_arrays.finalise(i, stream=stream)
            if transfer_outputs:
                self.output_arrays.finalise(i, stream=stream)
            d2h_event.record_end(stream)

        # Finalize GPU workload timing
        self._gpu_workload_event.record_end(stream)

    def limit_blocksize(
        self,
        blocksize: int,
        dynamic_sharedmem: int,
        bytes_per_run: int,
        numruns: int,
    ) -> tuple[int, int]:
        """Halve the block size until dynamic shared memory is launchable.

        Parameters
        ----------
        blocksize
            Requested CUDA block size.
        dynamic_sharedmem
            Shared-memory footprint per block at the current block size.
        bytes_per_run
            Shared-memory requirement per run.
        numruns
            Runs the launch places in one block.

        Returns
        -------
        tuple[int, int]
            Adjusted block size and shared-memory footprint per block,
            within the device's opt-in per-block limit.

        Raises
        ------
        ValueError
            If a single run's shared-memory demand exceeds the
            device's per-block limit, so no block size can launch.
        """
        hardware_limit = max_shared_memory_per_block()
        if dynamic_sharedmem > hardware_limit:
            if bytes_per_run > hardware_limit:
                raise ValueError(
                    f"A single run requires {bytes_per_run} B of "
                    f"shared memory, exceeding the device limit of "
                    f"{hardware_limit} B per block. Move buffers to "
                    "local memory to reduce per-run shared usage."
                )
            while dynamic_sharedmem > hardware_limit and blocksize > 1:
                blocksize = int(blocksize // 2)
                dynamic_sharedmem = int(
                    bytes_per_run * min(numruns, blocksize)
                )
            if blocksize < 32:
                warn(
                    "Per-run shared memory exceeds the device's "
                    "per-block limit at one warp per block; block size "
                    "is reduced below warp width. Performance will "
                    "degrade. Consider moving buffers to local memory."
                )
        return blocksize, dynamic_sharedmem

    def launch_geometry(
        self, blocksize: Optional[int] = None, runs: Optional[int] = None
    ) -> tuple[int, int]:
        """Return the block size and dynamic shared bytes of a launch.

        Parameters
        ----------
        blocksize
            Requested CUDA block size; ``None`` uses the ``blocksize``
            setting, or the automatic launch when that is unset.
        runs
            Runs in the launch; ``None`` sizes a full block.

        Returns
        -------
        tuple[int, int]
            Block size and dynamic shared bytes, padded to hold the
            resident block count.
        """
        resident = self.resident_blocks
        if blocksize is None:
            blocksize, chosen = self._default_launch(runs)
            if resident is None:
                resident = chosen
        key = (
            self.signature,
            blocksize,
            runs,
            resident,
            self.compile_settings.auto_performance,
        )
        geometries = self.get_cached_output("launch_geometries")
        geometry = geometries.get(key)
        if geometry is None:
            geometry = self._compute_launch_geometry(
                blocksize, runs, resident
            )
            geometries[key] = geometry
        return geometry

    def _launch_shape(
        self, blocksize: int, runs: Optional[int]
    ) -> tuple[int, int]:
        """Return a launch's block size, halved until its shared
        footprint fits, and dynamic shared bytes."""
        pad = SHARED_SKEW_BYTES if self.shared_memory_needs_padding else 0
        padded_bytes = self.shared_memory_bytes + pad
        runs_in_block = blocksize if runs is None else min(runs, blocksize)
        blocksize, dynamic_sharedmem = self.limit_blocksize(
            blocksize,
            int(padded_bytes * runs_in_block),
            padded_bytes,
            runs_in_block,
        )
        # The compiler needs a nonzero dynamic shared declaration.
        return blocksize, max(4, dynamic_sharedmem)

    def _natural_blocks(self, blocksize: int, dynamic_sharedmem: int) -> int:
        """Return the blocks per SM the driver fits at this launch shape."""
        dispatcher = self._compile_specialization()
        return active_blocks_per_multiprocessor(
            dispatcher, blocksize, dynamic_sharedmem, self.signature
        )

    def launchable_shapes(
        self,
        blocksizes: Sequence[int] = LAUNCH_BLOCKSIZES,
        runs: Optional[int] = None,
    ) -> Dict[int, Tuple[int, int]]:
        """Dynamic shared bytes and blocks per SM per launchable block size.

        Parameters
        ----------
        blocksizes
            Block sizes to consider.
        runs
            Runs in the launch; ``None`` sizes a full block.
        """
        shapes = {}
        for blocksize in blocksizes:
            actual, dynamic = self._launch_shape(blocksize, runs)
            if actual != blocksize:
                continue
            blocks = self._natural_blocks(blocksize, dynamic)
            if blocks > 0:
                shapes[blocksize] = (dynamic, blocks)
        return shapes

    def _default_launch(
        self, runs: Optional[int]
    ) -> tuple[int, Optional[int]]:
        """Return the block size and resident blocks per SM of a launch
        with no block size requested.

        Returns
        -------
        tuple[int, int or None]
            The automatic choice, or the ``blocksize`` setting (unset:
            ``DEFAULT_BLOCKSIZE``) with ``None`` when none is made.
        """
        configured = self.compile_settings.blocksize
        # A set block size, or the default without auto_performance.
        if configured is not None:
            return configured, None
        if not self.compile_settings.auto_performance:
            return DEFAULT_BLOCKSIZE, None
        launches_by_runs = self.get_cached_output("default_launches")
        key = (self.signature, runs)
        chosen = launches_by_runs.get(key)
        if chosen is not None:
            return chosen
        shapes = {
            blocksize: blocks
            for blocksize, (_, blocks) in self.launchable_shapes(
                runs=runs
            ).items()
        }
        if not shapes:
            return DEFAULT_BLOCKSIZE, None
        resources = kernel_resources(self.kernel, self.signature)
        chosen = default_launch(
            shapes,
            resources.local_bytes_per_thread,
            resources.sass_bytes,
            device_hardware(),
        )
        launches_by_runs[key] = chosen
        return chosen

    def _compute_launch_geometry(
        self, blocksize: int, runs: Optional[int], resident: Optional[int]
    ) -> tuple[int, int]:
        """Return the geometry holding ``resident`` blocks per SM."""
        blocksize, dynamic_sharedmem = self._launch_shape(blocksize, runs)
        blocks = resident
        if blocks is None and not self.compile_settings.auto_performance:
            return blocksize, dynamic_sharedmem
        natural = self._natural_blocks(blocksize, dynamic_sharedmem)
        dispatcher = self.kernel
        if blocks is None:
            blocks = resident_blocks_within_l2(
                kernel_resources(
                    dispatcher, self.signature
                ).local_bytes_per_thread,
                blocksize,
                natural,
                device_hardware(),
            )
        if blocks >= natural:
            return blocksize, dynamic_sharedmem
        return blocksize, self._dynamic_shared_for_blocks(
            dispatcher, blocksize, dynamic_sharedmem, blocks, self.signature
        )

    @staticmethod
    def _dynamic_shared_for_blocks(
        dispatcher: Any, blocksize: int, dynamic_sharedmem: int, blocks: int,
        signature: Optional[Tuple] = None,
    ) -> int:
        """Return the smallest dynamic shared pad holding ``blocks`` per SM."""
        hardware = device_hardware()
        limit = hardware.max_dynamic_shared_memory_per_block
        # Start at the most any block may take when ``blocks`` share an SM.
        padded = (
            hardware.shared_memory_per_multiprocessor // blocks
            - hardware.reserved_shared_memory_per_block
        )
        padded = max(min(padded, limit), dynamic_sharedmem)
        # Step down until the driver reports the target block count.
        while padded > dynamic_sharedmem:
            resident = active_blocks_per_multiprocessor(
                dispatcher, blocksize, padded, signature
            )
            if resident >= blocks:
                return padded
            padded -= DYNAMIC_SHARED_PAD_STEP
        return dynamic_sharedmem

    def build_kernel(self) -> None:
        """Build and compile the CUDA integration kernel."""
        config = self.compile_settings
        simsafe_precision = config.simsafe_precision
        precision = config.numba_precision

        loopfunction = self.single_integrator.device_function

        output_flags = self.active_outputs
        save_state = output_flags.state
        save_observables = output_flags.observables
        save_state_summaries = output_flags.state_summaries
        save_observable_summaries = output_flags.observable_summaries
        save_iteration_counters = output_flags.iteration_counters
        needs_padding = self.shared_memory_needs_padding

        shared_elems_per_run = self.shared_memory_elements
        f32_per_element = 2 if (precision is float64) else 1
        f32_pad_perrun = 1 if needs_padding else 0
        run_stride_f32 = int(
            (f32_per_element * shared_elems_per_run + f32_pad_perrun)
        )

        # Get memory allocators from buffer registry
        alloc_shared, alloc_persistent = (
            buffer_registry.get_toplevel_allocators(self)
        )

        jit_kwargs = self.jit_kwargs
        if config.max_registers is not None and not is_cudasim_enabled():
            jit_kwargs["max_registers"] = config.max_registers

        # no cover: start
        def integration_kernel(
            inits,
            params,
            d_coefficients,
            state_output,
            observables_output,
            state_summaries_output,
            observables_summaries_output,
            iteration_counters_output,
            status_codes_output,
            duration,
            warmup,
            t0,
            save_count,
            summary_count,
            n_runs,
        ):
            """Execute the compiled single-run loop for each batch chunk.

            Parameters
            ----------
            inits
                Device array containing initial values for each run.
            params
                Device array containing parameter values for each run.
            d_coefficients
                Device array of driver interpolation coefficients.
            state_output
                Device array where state trajectories are written.
            observables_output
                Device array where observable trajectories are written.
            state_summaries_output
                Device array containing state summary reductions.
            observables_summaries_output
                Device array containing observable summary reductions.
            iteration_counters_output
                Device array storing iteration counter values at each save
                point.
            status_codes_output
                Device array storing per-run solver status codes.
            duration
                Duration assigned to the current chunk integration.
            warmup
                Warmup duration applied before the chunk starts.
            t0
                Start time of the chunk integration window.
            save_count
                Number of scheduled save rows, initial included.
            summary_count
                Number of scheduled summary samples.
            n_runs
                Number of runs scheduled for the kernel launch.

            Returns
            -------
            None
                The device kernel performs integration for its side effects.
            """
            tx = int32(cuda.threadIdx.x)
            ty = int32(cuda.threadIdx.y)
            block_index = int32(cuda.blockIdx.x)
            runs_per_block = int32(cuda.blockDim.y)
            run_index = int32(runs_per_block * block_index + ty)
            if run_index >= n_runs:
                return None
            shared_memory = alloc_shared()
            persistent_local = alloc_persistent()
            c_coefficients = cuda.const.array_like(d_coefficients)
            run_idx_low = int32(ty * run_stride_f32)
            run_idx_high = int32(
                run_idx_low + f32_per_element * shared_elems_per_run
            )
            rx_shared_memory = shared_memory[run_idx_low:run_idx_high].view(
                simsafe_precision
            )
            rx_inits = inits[:, run_index]
            rx_params = params[:, run_index]
            rx_state = state_output[:, :, run_index * save_state]
            rx_observables = observables_output[
                :, :, run_index * save_observables
            ]
            rx_state_summaries = state_summaries_output[
                :, :, run_index * save_state_summaries
            ]
            rx_observables_summaries = observables_summaries_output[
                :, :, run_index * save_observable_summaries
            ]
            rx_iteration_counters = iteration_counters_output[
                :, :, run_index * save_iteration_counters
            ]
            status = loopfunction(
                rx_inits,
                rx_params,
                c_coefficients,
                rx_shared_memory,
                persistent_local,
                rx_state,
                rx_observables,
                rx_state_summaries,
                rx_observables_summaries,
                rx_iteration_counters,
                duration,
                warmup,
                t0,
                save_count,
                summary_count,
            )
            if tx == 0:
                status_codes_output[run_index] = status
            return None

        # no cover: end

        integration_kernel = name_and_compile_kernel(
            integration_kernel, self.kernel_name, jit_kwargs
        )

        # Attach this configuration's disk cache, if caching is on.
        cache_settings = self.compile_settings.cache
        self._disk_cache = None
        self._disk_cache_settings = cache_settings
        if cache_settings.cache_enabled:
            self._disk_cache = CUBIECache(
                system_name=self._system_name,
                system_hash=self.system.fn_hash,
                config_hash=self.config_hash,
                max_entries=cache_settings.max_cache_entries,
                mode=cache_settings.cache_mode,
                custom_cache_dir=cache_settings.cache_dir,
            )
            integration_kernel._cache = self._disk_cache
        return integration_kernel

    def _update(self, updates: Dict[str, Any], silent: bool) -> Set[str]:
        """Update the memory manager, interpolator, run and kernel.

        Parameters
        ----------
        updates
            Setting names to new values; gains the derived driver
            settings after an interpolator change.
        silent
            Whether :meth:`update` ignores unrecognised names.

        Returns
        -------
        set[str]
            Names the memory manager, interpolator, run and kernel
            settings recognised.

        Notes
        -----
        The kernel settings take the run's ``loop_fn`` and output
        compile flags last.
        """
        recognised = self.memory_manager.update(self, updates, silent=True)
        interpolator = self.driver_interpolator
        known_hash = interpolator.config_hash
        recognised |= interpolator.update(updates, silent=True)
        # New sample values alone keep the compiled evaluators.
        if interpolator.config_hash != known_hash:
            updates.update(self._driver_settings())
        recognised |= self.single_integrator.update(updates, silent=True)
        run = self.single_integrator
        kernel_updates = {
            **updates,
            "loop_fn": run.device_function,
            "compile_flags": run.output_compile_flags,
        }
        blocksize = self.compile_settings.blocksize
        recognised |= self.update_compile_settings(kernel_updates, silent=True)
        if self.compile_settings.blocksize != blocksize:
            # A pinned residency belongs to the block size it was timed with.
            self.resident_blocks = None
        self._known_system_config = self.system.compile_settings
        return recognised

    def _driver_settings(self) -> Dict[str, Any]:
        """Return the interpolator's evaluators and coefficient layout."""
        interpolator = self.driver_interpolator
        return {
            "drivers_fn": interpolator.drivers_fn,
            "driver_derivative_fn": interpolator.driver_derivative_fn,
            "coefficients_shape": interpolator.coefficients_shape,
        }

    def wait_for_writeback(
        self, timeout: Optional[float] = None
    ) -> None:
        """Wait for pending staging-buffer work."""
        self.input_arrays.wait_pending(timeout=timeout)
        self.output_arrays.wait_pending(timeout=timeout)

    def synchronize(self) -> None:
        """Wait for this kernel's last run stream."""
        if self._work_complete or self._last_stream is None:
            return
        self.memory_manager.sync_stream(self, stream=self._last_stream)
        self._work_complete = True

    def close(self, shutdown_timeout: Optional[float] = None) -> None:
        """Release resources after pending transfers finish.

        Parameters
        ----------
        shutdown_timeout
            Maximum seconds to wait. None waits until transfers finish.
        """
        if self._closed:
            return
        self.synchronize()
        self.wait_for_writeback(timeout=shutdown_timeout)
        self.input_arrays.close()
        self.output_arrays.close()
        self._specialization_cache = None
        finalizer = getattr(self, "_finalizer", None)
        settings = self.memory_manager.registry.get(id(self))
        if settings is not None:
            self.memory_manager.release_instance(id(self), settings)
        if finalizer is not None:
            finalizer.detach()
        self._closed = True

    @property
    def persistent_local_elements(self) -> int:
        """Number of elements in the per-thread persistent local array."""
        return self.single_integrator.persistent_local_elements

    @property
    def shared_memory_elements(self) -> int:
        """Number of precision elements required in shared memory per run."""
        return self.single_integrator.shared_memory_elements

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Boolean compile-time controls for which output features are enabled.
        """

        return self.compile_settings.compile_flags

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Active output array flags derived from compile_flags."""

        return self.compile_settings.active_outputs

    def set_cache_dir(self, path: Union[str, Path]) -> None:
        """Set a custom cache directory for compiled kernels.

        Parameters
        ----------
        path
            New cache directory path. Can be absolute or relative.
        """
        self.update(cache_dir=Path(path))

    @property
    def shared_memory_needs_padding(self) -> bool:
        """Indicate whether shared-memory padding is required.

        Returns
        -------
        bool
            ``True`` when a four-byte skew reduces bank conflicts for single
            precision.

        Notes
        -----
        Shared memory load instructions for ``float64`` require eight-byte
        alignment. Padding in that scenario would misalign alternate runs and
        trigger misaligned-access faults, so padding only applies to single
        precision workloads where the skew preserves alignment.
        """
        if self.precision == np_float64:
            return False
        elif self.shared_memory_elements == 0:
            return False
        elif self.shared_memory_elements % 2 == 0:
            return True
        else:
            return False

    def _on_allocation(self, response: "ArrayResponse") -> None:
        """Update run parameters with chunking metadata from allocation."""
        self.run_params = self.run_params.update_from_allocation(response)

    def _invalidate_cache(self) -> None:
        """Drop the build; flush the disk cache in flush_on_change mode."""
        super()._invalidate_cache()
        disk_cache = self._disk_cache
        if disk_cache is None:
            return
        cache_settings = self.compile_settings.cache
        if cache_settings != self._disk_cache_settings:
            # A cache built under other cache settings is never flushed.
            self._disk_cache = None
            return
        if cache_settings.cache_mode == "flush_on_change":
            disk_cache.flush_cache()

    @property
    def kernel(self) -> Callable:
        """Compiled integration kernel callable."""
        return self.device_function

    @property
    def device_function(self):
        return self.get_cached_output("solver_kernel")

    def build(self) -> BatchSolverCache:
        """Compile the integration kernel and return it with its memos."""
        return BatchSolverCache(
            solver_kernel=self.build_kernel(),
            output_array_heights=self.single_integrator.output_array_heights,
            time_domain_legend=self._time_domain_legend(),
            summaries_legend=self._summaries_legend(),
        )

    def _variable_units(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return the system's state and observable units by label."""
        system = self.system
        return (
            getattr(system, "state_units", {}),
            getattr(system, "observable_units", {}),
        )

    def _time_domain_legend(self) -> Dict[int, str]:
        """Map time-domain output rows to labels with units."""
        system = self.system
        state_units, obs_units = self._variable_units()
        state_labels = system.states.get_labels(self.saved_state_indices)
        obs_labels = system.observables.get_labels(
            self.saved_observable_indices
        )
        legend = {}
        for i, label in enumerate(state_labels):
            unit = state_units.get(label, "dimensionless")
            legend[i] = format_time_domain_label(label, unit)
        offset = len(state_labels)
        for i, label in enumerate(obs_labels):
            unit = obs_units.get(label, "dimensionless")
            legend[offset + i] = format_time_domain_label(label, unit)
        return legend

    def _summaries_legend(self) -> Dict[int, str]:
        """Map summary output rows to labels with units and metric."""
        system = self.system
        state_units, obs_units = self._variable_units()
        singlevar_legend = self.summary_legend_per_variable
        unit_modifications = self.summary_unit_modifications
        per_variable = len(singlevar_legend)
        state_labels = system.states.get_labels(
            self.summarised_state_indices
        )
        obs_labels = system.observables.get_labels(
            self.summarised_observable_indices
        )
        legend = {}
        blocks = (
            (state_labels, state_units, 0),
            (obs_labels, obs_units, len(state_labels) * per_variable),
        )
        for labels, units, offset in blocks:
            for i, label in enumerate(labels):
                unit = units.get(label, "dimensionless")
                for j, summary_type in enumerate(singlevar_legend.values()):
                    index = offset + i * per_variable + j
                    if unit == "dimensionless":
                        legend[index] = f"{label} {summary_type}"
                        continue
                    # The modification keeps its brackets around the unit.
                    unit_mod = unit_modifications.get(j, "[unit]")
                    modified_unit = unit_mod.replace("unit", unit)
                    legend[index] = f"{label} {modified_unit} {summary_type}"
        return legend

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return the settings of this kernel and its run."""
        settings = super().settings_dict
        settings.update(self.single_integrator.settings_dict)
        settings.update(
            stream_group=self.stream_group,
            mem_proportion=self.memory_manager.manual_proportion(self),
        )
        return settings

    @property
    def memory_manager(self) -> "MemoryManager":
        """Registered memory manager for this kernel."""

        return self._memory_manager

    @property
    def stream_group(self) -> str:
        """Stream group label assigned by the memory manager."""

        return self.memory_manager.get_stream_group(self)

    @property
    def stream(self) -> Any:
        """CUDA stream used for kernel launches."""

        return self.memory_manager.get_stream(self)

    @property
    def mem_proportion(self) -> Optional[float]:
        """Fraction of managed memory reserved for this kernel."""

        return self.memory_manager.proportion(self)

    @property
    def shared_memory_bytes(self) -> int:
        """Shared-memory footprint per run for the compiled kernel."""
        return self.single_integrator.shared_memory_bytes

    @property
    def threads_per_loop(self) -> int:
        """CUDA threads consumed by each run in the loop."""

        return self.single_integrator.threads_per_step

    @property
    def duration(self) -> float:
        """Requested integration duration."""
        return np_float64(self.run_params.duration)

    @duration.setter
    def duration(self, value: float) -> None:
        oldparams = self.run_params
        self.run_params = evolve(oldparams, duration=np_float64(value))

    @property
    def dt(self) -> Optional[float]:
        """Current integrator step size when available."""
        return self.single_integrator.dt or None

    @property
    def warmup(self) -> float:
        """Configured warmup duration."""
        return np_float64(self.run_params.warmup)

    @warmup.setter
    def warmup(self, value: float) -> None:
        oldparams = self.run_params
        self.run_params = evolve(oldparams, warmup=np_float64(value))

    @property
    def t0(self) -> float:
        """Configured initial integration time."""
        return np_float64(self.run_params.t0)

    @t0.setter
    def t0(self, value: float) -> None:
        oldparams = self.run_params
        self.run_params = evolve(oldparams, t0=np_float64(value))

    @property
    def num_runs(self) -> int:
        """Number of runs scheduled for the batch integration."""
        return self.run_params.runs

    @num_runs.setter
    def num_runs(self, value: int) -> None:
        oldparams = self.run_params
        self.run_params = evolve(oldparams, runs=value)

    @property
    def chunks(self):
        """Number of chunks in the most recent run."""
        return self.run_params.num_chunks

    @property
    def output_length(self) -> int:
        """Number of saved trajectory samples in the main run."""
        return self._duration_counts(self.duration).output_length

    @property
    def summaries_length(self) -> int:
        """Number of complete summary intervals in the integration window."""
        return self._duration_counts(self.duration).summaries_length

    @property
    def system(self) -> "BaseODE":
        """Underlying ODE system handled by the kernel."""

        return self.single_integrator.system

    @property
    def system_config_stale(self) -> bool:
        """``True`` when the system changed outside the update chain."""

        return self.system.compile_settings is not self._known_system_config

    @property
    def algorithm(self) -> str:
        """Identifier of the selected integration algorithm."""

        return self.single_integrator.algorithm

    @property
    def kernel_name(self) -> str:
        """Name the compiled kernel is given on the device.

        Returns
        -------
        str
            The configured name, or ``{algorithm}_{system name}`` when
            unset, with the LTO state appended and illegal identifier
            characters replaced.
        """
        config = self.compile_settings
        name = config.kernel_name
        if name is None:
            name = f"{self.algorithm}_{self._system_name}"
        lto_state = "ltoon" if config.jit_flags.lto else "ltooff"
        return re.sub(r"\W", "_", f"{name}_{lto_state}")

    @property
    def dt_min(self) -> float:
        """Minimum allowable step size from the controller."""

        return self.single_integrator.dt_min

    @property
    def dt_max(self) -> float:
        """Maximum allowable step size from the controller."""

        return self.single_integrator.dt_max

    @property
    def atol(self) -> float:
        """Absolute error tolerance applied during adaptive stepping."""

        return self.single_integrator.atol

    @property
    def rtol(self) -> float:
        """Relative error tolerance applied during adaptive stepping."""

        return self.single_integrator.rtol

    @property
    def save_every(self) -> Optional[float]:
        """Interval between saved samples from the loop, or None if save_last
        only.
        """
        return self.single_integrator.save_every

    @property
    def summarise_every(self) -> Optional[float]:
        """Interval between summary reductions from the loop"""

        return self.single_integrator.summarise_every

    @property
    def sample_summaries_every(self) -> float:
        """Interval between summary metric samples from the loop."""

        return self.single_integrator.sample_summaries_every

    @property
    def system_sizes(self) -> Any:
        """Structured size metadata for the system."""

        return self.single_integrator.system_sizes

    @property
    def n_drivers(self) -> int:
        """Number of interpolated driver inputs for the system."""

        return self.system_sizes.drivers

    @property
    def output_array_heights(self) -> OutputArrayHeights:
        """Height metadata for the batched output arrays."""

        return self.get_cached_output("output_array_heights")

    @property
    def time_domain_legend(self) -> Dict[int, str]:
        """Labels of the time-domain output rows, from the build."""

        return self.get_cached_output("time_domain_legend")

    @property
    def summaries_legend(self) -> Dict[int, str]:
        """Labels of the summary output rows, from the build."""

        return self.get_cached_output("summaries_legend")

    @property
    def summary_legend_per_variable(self) -> Any:
        """Legend entries describing each summarised variable."""

        return self.single_integrator.summary_legend_per_variable

    @property
    def summary_unit_modifications(self) -> Any:
        """Unit modifications for each summarised variable."""

        return self.single_integrator.summary_unit_modifications

    @property
    def saved_state_indices(self) -> Any:
        """Indices of saved state variables."""

        return self.single_integrator.saved_state_indices

    @property
    def saved_observable_indices(self) -> Any:
        """Indices of saved observable variables."""

        return self.single_integrator.saved_observable_indices

    @property
    def summarised_state_indices(self) -> Any:
        """Indices of summarised state variables."""

        return self.single_integrator.summarised_state_indices

    @property
    def summarised_observable_indices(self) -> Any:
        """Indices of summarised observable variables."""

        return self.single_integrator.summarised_observable_indices

    @property
    def state(self) -> Any:
        """Host view of saved state trajectories."""

        return self.output_arrays.state

    @property
    def observables(self) -> Any:
        """Host view of saved observable trajectories."""

        return self.output_arrays.observables

    @property
    def state_summaries(self) -> Any:
        """Host view of state summary reductions."""

        return self.output_arrays.state_summaries

    @property
    def status_codes(self) -> Any:
        """Host view of integration status codes."""

        return self.output_arrays.status_codes

    @property
    def observable_summaries(self) -> Any:
        """Host view of observable summary reductions."""

        return self.output_arrays.observable_summaries

    @property
    def iteration_counters(self) -> Any:
        """Host view of iteration counters at each save point."""

        return self.output_arrays.iteration_counters

    @property
    def device_state(self) -> Any:
        """Device buffer of saved state trajectories."""

        return self.output_arrays.device_state

    @property
    def device_observables(self) -> Any:
        """Device buffer of saved observable trajectories."""

        return self.output_arrays.device_observables

    @property
    def device_state_summaries(self) -> Any:
        """Device buffer of state summary reductions."""

        return self.output_arrays.device_state_summaries

    @property
    def device_observable_summaries(self) -> Any:
        """Device buffer of observable summary reductions."""

        return self.output_arrays.device_observable_summaries

    @property
    def device_status_codes(self) -> Any:
        """Device buffer of integration status codes."""

        return self.output_arrays.device_status_codes

    @property
    def device_iteration_counters(self) -> Any:
        """Device buffer of iteration counters at each save point."""

        return self.output_arrays.device_iteration_counters

    def _resident_input(self, name: str) -> Any:
        """Device input ``name``; raises ValueError after a chunked run."""
        array = getattr(self.input_arrays, "device_" + name)
        if array is not None and self.run_params.num_chunks > 1:
            raise ValueError(
                f"The device {name} buffer holds one chunk of the last "
                f"run ({self.run_params.num_chunks} chunks); "
                "device-resident inputs need a single-chunk run."
            )
        return array

    @property
    def device_initial_values(self) -> Any:
        """Device initial values of the last single-chunk run."""

        return self._resident_input("initial_values")

    @property
    def device_parameters(self) -> Any:
        """Device parameters of the last single-chunk run."""

        return self._resident_input("parameters")

    @property
    def initial_values(self) -> Any:
        """Initial state values used in the last run.

        A host view, or the caller's device array when initial values
        were supplied on device.
        """

        return self.input_arrays.initial_values

    @property
    def parameters(self) -> Any:
        """Parameter tables used in the last run.

        A host view, or the caller's device array when parameters
        were supplied on device.
        """

        return self.input_arrays.parameters

    @property
    def driver_coefficients(self) -> Optional[NDArray[floating]]:
        """Horner-ordered driver coefficients on the host."""

        return self.input_arrays.driver_coefficients

    @property
    def coefficients_shape(self) -> tuple[int, int, int]:
        """Expected driver-coefficient layout for input validation.

        A :class:`BatchSolverConfig` compile setting the owning
        :class:`Solver` keeps aligned with
        ``ArrayInterpolator.coefficients_shape`` — the exact
        ``(num_segments, num_drivers, order + 1)`` layout baked into
        the compiled driver evaluators — so supplied coefficient
        arrays are checked against the shape the kernel was compiled
        for. Update via ``update(coefficients_shape=...)``.
        """
        return self.compile_settings.coefficients_shape

    @property
    def device_driver_coefficients(self) -> Optional[NDArray[floating]]:
        """Device-resident driver coefficients."""

        return self.input_arrays.device_driver_coefficients

    @property
    def save_time(self) -> bool:
        """Whether time samples are saved alongside states."""

        return self.single_integrator.save_time

    @property
    def save_counters(self) -> bool:
        """Whether iteration counters are saved at each save point."""

        return self.single_integrator.save_counters

    @property
    def output_types(self) -> Any:
        """Active output type identifiers configured for the run."""

        return self.single_integrator.output_types
