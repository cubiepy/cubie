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
    zeros as np_zeros,
)
from cubie.cuda_simsafe import cuda, float64
from cubie.cuda_simsafe import int32

from attrs import define, field, evolve

from cubie.odesystems import SymbolicODE
from cubie.cuda_simsafe import (
    is_cudasim_enabled,
    max_shared_memory_per_block,
)
from cubie.cubie_cache import (
    ALL_CACHE_PARAMETERS,
    CachePolicy,
    CubieCacheHandler,
)

from cubie.time_logger import CUDAEvent
from numpy.typing import NDArray

from cubie.array_interpolator import ArrayInterpolator
from cubie.memory import default_memmgr
from cubie.memory.mem_manager import defer_instance_teardown
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie.batchsolving.arrays.BatchInputArrays import InputArrays
from cubie.batchsolving.arrays.BatchOutputArrays import (
    OutputArrays,
)
from cubie.batchsolving.BatchSolverConfig import ActiveOutputs
from cubie.batchsolving.BatchSolverConfig import BatchSolverConfig
from cubie.batchsolving._utils import name_and_compile_kernel
from cubie.odesystems.baseODE import BaseODE
from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
from cubie._utils import unpack_dict_values, getype_validator

if TYPE_CHECKING:
    from cubie.memory import MemoryManager
    from cubie.memory.array_requests import ArrayResponse


DEFAULT_MEMORY_SETTINGS = {
    "memory_manager": default_memmgr,
    "stream_group": "solver",
    "mem_proportion": None,
    "host_spill_threshold": None,
    "spill_directory": None,
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

        Notes
        -----
        Extracts num_chunks and chunk_length from the response. When
        num_chunks=1, chunk_length is set equal to runs (no chunking).
        """

        return evolve(
            self,
            num_chunks=response.chunks,
            chunk_length=response.chunk_length,
        )


@define()
class BatchSolverCache(CUDADispatcherCache):
    solver_kernel: Union[int, Callable] = field(default=-1)


class BatchSolverKernel(CUDAFactory):
    """Factory for CUDA kernel which coordinates a batch integration.

    Parameters
    ----------
    system
        ODE system describing the problem to integrate.
    loop_settings
        Mapping of loop configuration forwarded to
        :class:`cubie.integrators.SingleIntegratorRun`. Recognised keys include
        ``"save_every"`` and ``"summarise_every"``.
    evaluate_driver_at_t
        Optional evaluation function for an interpolated forcing term.
    lineinfo
        Compile the kernel and all device functions with source-line
        correlation data for profilers. ``None`` defers to the
        ``CUBIE_LINEINFO`` environment variable (default off).
    step_control_settings
        Mapping of overrides forwarded to
        :class:`cubie.integrators.SingleIntegratorRun` for controller
        configuration.
    algorithm_settings
        Mapping of overrides forwarded to
        :class:`cubie.integrators.SingleIntegratorRun` for algorithm
        configuration.
    output_settings
        Mapping of output configuration forwarded to the integrator. See
        :class:`cubie.outputhandling.OutputFunctions` for recognised keys.
    memory_settings
        Mapping of memory configuration forwarded to the memory manager,
        typically via :mod:`cubie.memory`.
    cache_settings
        Mapping of cache configuration forwarded to
        :class:`cubie.cubie_cache.CachePolicy`.
    cache
        Cache mode control. ``True`` enables default caching, ``False``
        disables caching, or a string/``Path`` sets a custom cache
        directory.
    kernel_settings
        Kernel-level compile settings forwarded to
        :class:`BatchSolverConfig` (currently ``max_registers``).

    Notes
    -----
    The kernel delegates integration logic to :class:`SingleIntegratorRun`
    instances and expects upstream APIs to perform batch construction. It
    executes the compiled loop function against kernel-managed memory slices
    and distributes work across GPU threads for each input batch.
    """

    def __init__(
        self,
        system: "SymbolicODE",
        loop_settings: Optional[Dict[str, Any]] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        driver_del_t: Optional[Callable] = None,
        lineinfo: Optional[bool] = None,
        step_control_settings: Optional[Dict[str, Any]] = None,
        algorithm_settings: Optional[Dict[str, Any]] = None,
        output_settings: Optional[Dict[str, Any]] = None,
        memory_settings: Optional[Dict[str, Any]] = None,
        cache_settings: Optional[Dict[str, Any]] = None,
        cache: Union[bool, str, Path] = True,
        kernel_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if memory_settings is None:
            memory_settings = {}
        if output_settings is None:
            output_settings = {}
        if loop_settings is None:
            loop_settings = {}

        precision = system.precision

        # Initialize run parameters with defaults
        self.run_params = RunParams(
            duration=precision(0.0),
            warmup=precision(0.0),
            t0=precision(0.0),
            runs=1,
        )

        # CUDA event tracking for timing
        self._cuda_events: List = []
        self._gpu_workload_event: Optional[CUDAEvent] = None

        self._closed = False
        self._last_stream = None
        self._work_complete = True
        self._memory_manager = self._setup_memory_manager(memory_settings)

        # Child factory: driver settings join config_hash; the
        # placeholder input covers zero-driver operation.
        self.driver_interpolator = ArrayInterpolator(
            precision=precision,
            input_dict={
                "placeholder": np_zeros(6, dtype=precision),
                "driver_sample_period": 0.1,
            },
        )

        system_name = system.name
        system_hash = system.fn_hash
        if system_name == system_hash:
            system_name = f"unnamed_{system_hash[:8]}"
        self._system_name = system_name
        if cache_settings is None:
            cache_settings = {}
        cache_params = CachePolicy.params_from_user_kwarg(cache)
        cache_params.update(cache_settings)
        cache_policy = CachePolicy(**cache_params)
        self.cache_handler = CubieCacheHandler(
            cache_policy, system_name=system_name
        )
        self._solver_helper_fn = system.solver_helper_getter(cache_policy)

        # Build the single integrator to derive compile-critical metadata
        self.single_integrator = SingleIntegratorRun(
            system,
            loop_settings=loop_settings,
            evaluate_driver_at_t=evaluate_driver_at_t,
            driver_del_t=driver_del_t,
            step_control_settings=step_control_settings,
            algorithm_settings=algorithm_settings,
            output_settings=output_settings,
            solver_helper_fn=self._solver_helper_fn,
        )
        # An explicit lineinfo argument must reach every child factory;
        # None leaves the CUBIE_LINEINFO-derived config defaults in place.
        if lineinfo is not None:
            self.single_integrator.update(
                {"lineinfo": lineinfo}, silent=True
            )

        if kernel_settings is None:
            kernel_settings = {}
        kernel_settings = kernel_settings.copy()
        # The compiled kernel bakes the driver-coefficient layout in
        # as closure constants; seed it from the owned interpolator
        # unless the caller supplied a layout explicitly.
        kernel_settings.setdefault(
            "driver_coefficients_shape",
            self.driver_interpolator.coefficients_shape,
        )
        initial_config = BatchSolverConfig(
            precision=precision,
            loop_fn=None,
            compile_flags=self.single_integrator.output_compile_flags,
            **kernel_settings,
        )
        self.setup_compile_settings(initial_config)
        if lineinfo is not None:
            self.update_compile_settings(
                {"lineinfo": lineinfo}, silent=True
            )

        self.input_arrays = InputArrays.from_solver(self)
        self.output_arrays = OutputArrays.from_solver(self)

        self.output_arrays.update(self)

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
        stream_group = merged_settings["stream_group"]
        mem_proportion = merged_settings["mem_proportion"]
        memory_manager.register(
            self,
            stream_group=stream_group,
            proportion=mem_proportion,
            allocation_ready_hook=self._on_allocation,
            owner=self,
            host_spill_threshold=merged_settings["host_spill_threshold"],
            spill_directory=merged_settings["spill_directory"],
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
        """Create CUDA events for timing instrumentation.

        Parameters
        ----------
        chunks : int
            Number of chunks to process

        Notes
        -----
        Creates one GPU workload event and 3 events per chunk
        (h2d_transfer, kernel, d2h_transfer).
        Events are created regardless of verbosity - they become no-ops
        internally when verbosity is None.
        """
        # Create overall GPU workload event
        self._gpu_workload_event = CUDAEvent("gpu_workload")

        # Create per-chunk events (3 events per chunk: h2d, kernel, d2h)
        self._cuda_events = []
        for i in range(chunks):
            h2d_event = CUDAEvent(f"h2d_transfer_chunk_{i}")
            kernel_event = CUDAEvent(f"kernel_chunk_{i}")
            d2h_event = CUDAEvent(f"d2h_transfer_chunk_{i}")
            self._cuda_events.extend([h2d_event, kernel_event, d2h_event])

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

    def _validate_timing_parameters(self, duration: float) -> None:
        """Validate timing parameters to prevent invalid array accesses.

        Parameters
        ----------
        duration
            Integration duration in time units.

        Raises
        ------
        ValueError
            When timing parameters would result in no outputs or invalid
            sampling.

        Notes
        -----
        Uses dt_min as an absolute tolerance when comparing floating
        point timing parameters by adding dt_min to the requested
        duration. Small in-loop timing oversteps smaller than dt_min
        are treated as valid and do not trigger validation errors.
        """
        integrator = self.single_integrator
        end_time = self.precision(duration) + self.dt_min

        # Validate time-domain output timing parameters
        if integrator.has_time_domain_outputs:
            save_every = integrator.save_every
            save_last = integrator.save_last
            if (
                save_every is not None
                and save_every > end_time
                and not save_last
            ):
                raise ValueError(
                    f"save_every ({save_every}) > duration ({duration}) "
                    f"so this loop will produce no outputs"
                )

        # Validate summary timing parameters
        if integrator.has_summary_outputs:
            sample_summaries_every = integrator.sample_summaries_every
            summarise_every = integrator.summarise_every

            if sample_summaries_every is None:
                raise ValueError(
                    "Summary outputs are enabled but sample_summaries_every "
                    "is None"
                )
            if summarise_every is None:
                raise ValueError(
                    "Summary outputs are enabled but summarise_every is None"
                )

            if sample_summaries_every >= summarise_every:
                raise ValueError(
                    f"sample_summaries_every ({sample_summaries_every}) "
                    f">= summarise_every ({summarise_every}); "
                    f"The saved summary will be based on 0 samples, so will "
                    f"result in 0/inf/NaN values."
                )

            if summarise_every > end_time:
                raise ValueError(
                    f"summarise_every ({summarise_every}) > duration "
                    f"({duration}), so this loop will produce no summary "
                    f"outputs"
                )

    def run(
        self,
        inits: NDArray[floating],
        params: NDArray[floating],
        driver_coefficients: Optional[NDArray[floating]],
        duration: float,
        blocksize: int = 256,
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
        driver_coefficients
            Optional Horner-ordered driver interpolation coefficients with
            shape ``(num_segments, num_drivers, order + 1)``.
        duration
            Duration of the simulation window.
        blocksize
            CUDA block size for kernel execution.
        warmup
            Warmup time before the main simulation.
        t0
            Initial integration time.
        transfer_outputs
            When ``True`` (default), output arrays are copied
            device-to-host after each chunk. ``False`` skips the copy
            so results stay in the device output buffers; the run must
            fit in a single chunk.

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
            If the batch is chunked while ``transfer_outputs`` is
            ``False`` or while inputs were supplied as device arrays.
        """
        if self._closed:
            raise RuntimeError(
                "This solver has been closed and its GPU resources "
                "released; build a new Solver to run again."
            )
        stream = self.stream
        self._memory_manager.begin_work(self)
        try:
            self._execute_run(
                inits,
                params,
                driver_coefficients,
                duration,
                blocksize,
                stream,
                warmup,
                t0,
                transfer_outputs,
            )
        finally:
            self._memory_manager.end_work(self, stream)

    def _execute_run(
        self,
        inits: NDArray[floating],
        params: NDArray[floating],
        driver_coefficients: Optional[NDArray[floating]],
        duration: float,
        blocksize: int,
        stream: Optional[Any],
        warmup: float,
        t0: float,
        transfer_outputs: bool,
    ) -> None:
        """Allocate, chunk, and launch the batch kernel."""
        self._last_stream = stream
        self._work_complete = False

        # Time parameters always use float64 for accumulation accuracy
        duration = np_float64(duration)

        # Update run params with actual values before allocation
        self.run_params = RunParams(
            duration=duration,
            warmup=np_float64(warmup),
            t0=np_float64(t0),
            runs=inits.shape[1],
        )

        # Update the single integrator with requested duration if required
        self.single_integrator.set_summary_timing_from_duration(duration)

        # Validate timing parameters to prevent array index errors
        self._validate_timing_parameters(duration)

        # Refresh compile-critical settings before array updates
        self.update_compile_settings(
            {
                "loop_fn": self.single_integrator.device_function,
                "precision": self.single_integrator.precision,
            }
        )

        # Queue allocations
        self.input_arrays.update(self, inits, params, driver_coefficients)
        self.output_arrays.update(self)

        # Process allocations into chunks
        self.memory_manager.allocate_queue(self, stream=stream)

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

        # Get first chunk runs for initial block size calculation
        first_chunk_params = self.run_params[0]
        runs = first_chunk_params.runs

        pad = 4 if self.shared_memory_needs_padding else 0
        padded_bytes = self.shared_memory_bytes + pad
        dynamic_sharedmem = int(padded_bytes * min(runs, blocksize))

        blocksize, dynamic_sharedmem = self.limit_blocksize(
            blocksize,
            dynamic_sharedmem,
            padded_bytes,
            runs,
        )

        # We need a nonzero number to tell the compiler we're using dynamic
        # memory. If zero, then the cuda.shared.array(0) call fails as we
        # can't declare a size-0 static shared memory array.
        dynamic_sharedmem = max(4, dynamic_sharedmem)
        threads_per_loop = self.single_integrator.threads_per_step
        runsperblock = int(blocksize / self.single_integrator.threads_per_step)

        # Setup CUDA events for timing (no-op when verbosity is None)
        self._setup_cuda_events(chunks)

        # Record start of overall GPU workload
        self._gpu_workload_event.record_start(stream)
        precision = self.precision
        for i in range(chunks):
            # Get parameters for this specific chunk
            chunk_run_params = self.run_params[i]
            duration = precision(chunk_run_params.duration)
            warmup = precision(chunk_run_params.warmup)
            t0 = precision(chunk_run_params.t0)
            save_stop = precision(
                self.single_integrator.save_stop_time(
                    duration, warmup, t0
                )
            )
            summary_stop = precision(
                self.single_integrator.summary_stop_time(
                    duration, warmup, t0
                )
            )

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
            self.kernel[
                chunk_blocks,
                (threads_per_loop, runsperblock),
                stream,
                dynamic_sharedmem,
            ](
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
                save_stop,
                summary_stop,
                runs,
            )
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
        """Reduce block size until dynamic shared memory fits within limits.

        Parameters
        ----------
        blocksize
            Requested CUDA block size.
        dynamic_sharedmem
            Shared-memory footprint per block at the current block size.
        bytes_per_run
            Shared-memory requirement per run.
        numruns
            Total number of runs queued for the launch.

        Returns
        -------
        tuple[int, int]
            Adjusted block size and shared-memory footprint per block.

        Raises
        ------
        ValueError
            If a single run's shared-memory demand exceeds the
            device's per-block limit, so no block size can launch.

        Notes
        -----
        Reduction is two-staged. The performance stage targets a
        32 kiB footprint (three blocks per SM on CC7* hardware;
        larger requests reduce per-thread L1 availability) but is
        floored at one warp: profiling shows sub-warp blocks starve
        the SMs of resident threads and run slower than exceeding
        the target. The hardware stage then reduces below one warp
        only when the device's per-block shared-memory limit leaves
        no alternative — there a sub-warp block is a launchability
        requirement, not a tuning choice.
        """
        while dynamic_sharedmem >= 32768 and blocksize > 32:
            blocksize = max(32, int(blocksize // 2))
            dynamic_sharedmem = int(bytes_per_run * min(numruns, blocksize))

        hardware_limit = max_shared_memory_per_block()
        if dynamic_sharedmem > hardware_limit:
            if bytes_per_run > hardware_limit:
                raise ValueError(
                    f"A single run requires {bytes_per_run} B of "
                    f"shared memory, exceeding the device limit of "
                    f"{hardware_limit} B per block. Move buffers to "
                    "local memory to reduce per-run shared usage."
                )
            warn(
                "Per-run shared memory exceeds the device's "
                "per-block limit at one warp per block; block size "
                "is reduced below warp width. Performance will "
                "degrade. Consider moving buffers to local memory."
            )
            while dynamic_sharedmem > hardware_limit and blocksize > 1:
                blocksize = int(blocksize // 2)
                dynamic_sharedmem = int(
                    bytes_per_run * min(numruns, blocksize)
                )
        elif dynamic_sharedmem >= 32768:
            warn(
                "Dynamic shared memory exceeds the 32 kiB per-block "
                "performance target at the minimum block size of 32 "
                "threads. Occupancy will be reduced. Consider moving "
                "buffers to local memory."
            )
        return blocksize, dynamic_sharedmem

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
            save_stop,
            summary_stop,
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
                Device array storing iteration counter values at each save point.
            status_codes_output
                Device array storing per-run solver status codes.
            duration
                Duration assigned to the current chunk integration.
            warmup
                Warmup duration applied before the chunk starts.
            t0
                Start time of the chunk integration window.
            save_stop
                Completion time of the regular save schedule, half
                an interval past its final scheduled event.
            summary_stop
                Completion time of the summary-update schedule,
                half a sample interval past its final scheduled
                event.
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
                save_stop,
                summary_stop,
            )
            if tx == 0:
                status_codes_output[run_index] = status
            return None

        # no cover: end

        integration_kernel = name_and_compile_kernel(
            integration_kernel, self.kernel_name, jit_kwargs
        )

        # Attach this configuration's disk cache, if caching is on.
        cfg_hash = self.config_hash
        configured_cache = self.cache_handler.configured_cache(
            self.system.fn_hash, cfg_hash
        )
        if configured_cache is not None:
            integration_kernel._cache = configured_cache
        return integration_kernel

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> set[str]:
        """Update solver configuration parameters.

        Parameters
        ----------
        updates_dict
            Mapping of parameter updates forwarded to the single integrator and
            compile settings.
        silent
            Flag suppressing errors when unrecognised parameters remain.
        **kwargs
            Additional parameter overrides merged into ``updates_dict``.

        Returns
        -------
        set[str]
            Names of parameters successfully applied.

        Raises
        ------
        KeyError
            Raised when unknown parameters persist and ``silent`` is ``False``.

        Notes
        -----
        The method applies updates to the single integrator before refreshing
        compile-critical settings so the kernel rebuild picks up new metadata.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        # Flatten nested dict values so that grouped settings can be passed
        # naturally. For example, step_controller_settings={'dt_min': 0.01}
        # becomes dt_min=0.01, allowing sub-components to recognize and
        # apply parameters correctly.
        updates_dict, unpacked_keys = unpack_dict_values(updates_dict)

        all_unrecognized = set(updates_dict.keys())

        policy_changed = self.cache_handler.update_policy_params(
            updates_dict
        )
        if policy_changed:
            self._solver_helper_fn = self.system.solver_helper_getter(
                self.cache_handler.policy
            )
            updates_dict["solver_helper_fn"] = self._solver_helper_fn
            self._invalidate_cache()
        all_unrecognized -= set(updates_dict.keys()) & ALL_CACHE_PARAMETERS

        driver_recognised = self.driver_interpolator.update(
            updates_dict, silent=True
        )
        if driver_recognised and self.n_drivers > 0:
            updates_dict["evaluate_driver_at_t"] = (
                self.driver_interpolator.evaluation_function
            )
            updates_dict["driver_del_t"] = (
                self.driver_interpolator.driver_del_t
            )
            updates_dict["driver_coefficients_shape"] = (
                self.driver_interpolator.coefficients_shape
            )
        all_unrecognized -= driver_recognised

        all_unrecognized -= self.single_integrator.update(
            updates_dict, silent=True
        )

        all_unrecognized -= buffer_registry.update(
            self.single_integrator._loop, updates_dict, silent=True
        )

        updates_dict.update(
            {
                "loop_fn": self.single_integrator.device_function,
                "compile_flags": self.single_integrator.output_compile_flags,
            }
        )

        all_unrecognized -= self.update_compile_settings(
            updates_dict, silent=True
        )

        recognised = set(updates_dict.keys()) - all_unrecognized

        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")

        # Include unpacked dict keys in recognized set
        return recognised | unpacked_keys

    def configure_drivers(self, drivers: Dict[str, Any]) -> None:
        """Update the owned driver interpolator and dependent settings.

        Parameters
        ----------
        drivers
            Driver samples plus interpolation settings, as accepted by
            :meth:`ArrayInterpolator.update_from_dict`.
        """
        drivers = ArrayInterpolator.check_against_system_drivers(
            drivers, self.system
        )
        fn_changed = self.driver_interpolator.update_from_dict(drivers)
        if fn_changed:
            self.update(
                {
                    "evaluate_driver_at_t": (
                        self.driver_interpolator.evaluation_function
                    ),
                    "driver_del_t": (
                        self.driver_interpolator.driver_del_t
                    ),
                    "driver_coefficients_shape": (
                        self.driver_interpolator.coefficients_shape
                    ),
                }
            )

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
        """Boolean compile-time controls for which output features are enabled."""

        return self.compile_settings.compile_flags

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Active output array flags derived from compile_flags."""

        return self.compile_settings.active_outputs

    @property
    def cache_policy(self) -> "CachePolicy":
        """Cache policy the kernel's disk cache follows."""
        return self.cache_handler.policy

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
        """Mark cached outputs as invalid, flushing cache if cache_handler
        in "flush on change" mode."""
        super()._invalidate_cache()
        self.cache_handler.invalidate()

    @property
    def output_heights(self) -> Any:
        """Height metadata for each host output array."""

        return self.single_integrator.output_array_heights

    @property
    def kernel(self) -> Callable:
        """Compiled integration kernel callable."""
        return self.device_function

    @property
    def device_function(self):
        return self.get_cached_output("solver_kernel")

    def build(self) -> BatchSolverCache:
        """Compile the integration kernel and return it."""
        return BatchSolverCache(solver_kernel=self.build_kernel())

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
        """Number of saved trajectory samples in the main run.

        Delegates to SingleIntegratorRun.output_length() with the current
        duration.
        """
        return self.single_integrator.output_length(self.duration)

    @property
    def summaries_length(self) -> int:
        """Number of complete summary intervals across the integration window.

        Delegates to SingleIntegratorRun.summaries_length() with the current
        duration.
        """
        return self.single_integrator.summaries_length(self.duration)

    @property
    def system(self) -> "BaseODE":
        """Underlying ODE system handled by the kernel."""

        return self.single_integrator.system

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
        """Interval between saved samples from the loop, or None if save_last only."""
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
    def output_array_heights(self) -> Any:
        """Height metadata for the batched output arrays."""

        return self.single_integrator.output_array_heights

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
    def driver_coefficients_shape(self) -> tuple[int, int, int]:
        """Expected driver-coefficient layout for input validation.

        A :class:`BatchSolverConfig` compile setting the owning
        :class:`Solver` keeps aligned with
        ``ArrayInterpolator.coefficients_shape`` — the exact
        ``(num_segments, num_drivers, order + 1)`` layout baked into
        the compiled driver evaluators — so supplied coefficient
        arrays are checked against the shape the kernel was compiled
        for. Update via ``update(driver_coefficients_shape=...)``.
        """
        return self.compile_settings.driver_coefficients_shape

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
