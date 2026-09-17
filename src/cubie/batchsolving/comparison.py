"""Time candidate settings of one solver on one device-resident batch.

Published Objects
-----------------
:class:`Candidate`
    Settings and launch of one configuration to time.
:class:`CandidateTiming`
    One candidate's kernel times and failed-run count.
:class:`ComparisonRunner`
    Switches one solver between candidates and times each one.
:func:`compile_kernels`
    Compile one kernel per settings set, in parallel when cheaper.
:func:`rank_timings`
    Order timings by success tier, then time.
:func:`tail_safe_runs`
    The batch near a wanted size whose last wave is fullest.
:func:`validate_sizing`
    Reject a ``waves`` or ``target_ms`` sizing argument out of range.
"""

import logging
import multiprocessing
import pickle
from math import ceil, isfinite
from typing import Any, Dict, List, Optional, Sequence, Tuple
from warnings import warn

from attrs import define, field, fields
from numpy import arange as np_arange
from numpy import asarray as np_asarray
from numpy import count_nonzero as np_count_nonzero
from numpy import int32 as np_int32
from numpy import ndarray
from numpy import take as np_take

from cubie.backend.utils import (
    active_blocks_per_multiprocessor,
    device_hardware,
)
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cuda_simsafe import cuda, float32
from cubie.CUDAFactory import UnrollFlags
from cubie.time_logger import default_timelogger

logger = logging.getLogger(__name__)

ROUNDS = 2
"""Timing rounds; the second visits the candidates in reverse order."""

SOLVES_PER_ROUND = 3
"""Timed solves per candidate per round."""

WARM_MS = 500.0
"""Busy-kernel milliseconds run before the first timed solve.

Empirical: RTX 4070 SUPER launches speed up 11% after 210 to 230 ms
of load and hold that rate through host gaps of up to 1 s.
"""

BUSY_CHAIN = 1 << 16
"""Dependent FMAs per thread in one busy launch.

Empirical: 2.6 to 2.9 ms per launch on the RTX 4070 SUPER.
"""

SUCCESS_TIER_FRACTION = 0.95
"""Share of the best success rate a candidate keeps to rank on time."""

TIMED_WAVES_FLOOR = 2
"""Fewest occupancy waves a timed batch fills at any launch."""

WORKER_STARTUP_SECONDS = 12.0
"""Wall seconds a spawned worker spends importing cubie.

Empirical: a fresh ``import cubie`` in a spawned process on the
development machine (mlir compat 6 s, odesystems 4.8 s, cupy 1.5 s).
"""


@cuda.jit
def _busy_kernel(sink, iterations):  # pragma: no cover - device code
    """Run a dependent FMA chain for ``iterations`` steps."""
    value = float32(cuda.grid(1))
    for _ in range(iterations):
        value = value * float32(0.999) + float32(0.001)
    if value < float32(0.0):
        sink[0] = value


def busy_launch(stream: Any) -> None:
    """Queue one busy launch on ``stream``: every SM at its block and
    thread limits, :data:`BUSY_CHAIN` FMAs per thread."""
    hardware = device_hardware()
    blocks_per_sm = hardware.max_blocks_per_multiprocessor
    threads = hardware.max_threads_per_multiprocessor // blocks_per_sm
    blocks = blocks_per_sm * hardware.multiprocessor_count
    sink = cuda.device_array(1, dtype="float32")
    _busy_kernel[blocks, threads, stream](sink, BUSY_CHAIN)


def warm_clocks(stream: Any) -> float:
    """Busy-launch on ``stream`` until :data:`WARM_MS` of kernel time
    has passed; return that time."""
    total = 0.0
    while total < WARM_MS:
        start = cuda.event()
        end = cuda.event()
        start.record(stream)
        busy_launch(stream)
        end.record(stream)
        end.synchronize()
        total += float(cuda.event_elapsed_time(start, end))
    return total


def validate_sizing(waves: int, target_ms: float) -> None:
    """Reject ``waves`` under 1 or fractional, ``target_ms`` under 10."""
    if int(waves) < 1 or waves != int(waves):
        raise ValueError(f"waves must be a positive integer, got {waves!r}")
    if not (isfinite(target_ms) and target_ms >= 10.0):
        raise ValueError(
            f"target_ms must be a finite number of at least 10, "
            f"got {target_ms!r}"
        )


def warn_low_waves(fewest: float, advice: str = "") -> None:
    """Warn that a given batch fills ``fewest`` waves, under the floor."""
    warn(
        f"The batch passed only fills {fewest:.2f} occupancy waves; the "
        "results might not represent the best timing for your system. "
        f"Try again with {TIMED_WAVES_FLOOR / fewest:.1f}x more runs in "
        f"the batch{advice}."
    )


def unused_wave_share(runs: int, concurrent: Sequence[int]) -> float:
    """Largest empty share of a last wave, ``(ceil(w) - w) / ceil(w)``
    with ``w = runs / count``, over the launches' concurrent counts."""
    worst = 0.0
    for count in concurrent:
        waves = runs / count
        worst = max(worst, (ceil(waves) - waves) / ceil(waves))
    return worst


def tail_safe_runs(
    concurrent: Sequence[int],
    wanted: int,
    floor: int,
    cap: Optional[int] = None,
) -> int:
    """Return the batch near ``wanted`` runs with the fullest last wave.

    Parameters
    ----------
    concurrent
        Runs each launch executes at once.
    wanted
        Runs the batch should reach.
    floor
        Fewest runs allowed.
    cap
        Most runs allowed; ``None`` for no cap.

    Returns
    -------
    int
        The wave boundary or interval end with the least
        :func:`unused_wave_share` in the one-wave interval above
        ``wanted``, clipped to ``[floor, cap]``; the smallest on a tie.
    """
    most = max(concurrent)
    low = max(int(wanted), int(floor))
    high = low + most
    if cap is not None:
        high = min(high, int(cap))
        low = max(int(floor), min(low, high - most))
        if low > high:
            return high
    candidates = {low, high}
    for count in concurrent:
        first = -(-low // count)
        candidates.update(
            multiple * count for multiple in range(first, high // count + 1)
        )
    return min(
        candidates,
        key=lambda runs: (unused_wave_share(runs, concurrent), runs),
    )


def settings_label(settings: Dict[str, Any]) -> str:
    """Return a short name for a settings dict."""
    parts = []
    for key, value in settings.items():
        name = getattr(value, "name", None)
        if name is not None and not isinstance(value, str):
            value = name.lower()
        key = key.removeprefix("unroll_").removesuffix("_location")
        parts.append(f"{key}={value}")
    return " ".join(parts) or "current"


def device_to_host(array: Any) -> ndarray:
    """Copy a device array to the host, whatever its container."""
    if hasattr(array, "copy_to_host"):
        return array.copy_to_host()
    if hasattr(array, "get"):
        return array.get()
    return np_asarray(array)


def _settings_key(settings: Dict[str, Any]) -> Tuple:
    """Return the identity of a settings dict."""
    return tuple(sorted(settings.items()))


@define
class Candidate:
    """One configuration and launch to time.

    Parameters
    ----------
    label
        Name shown in reports.
    settings
        Solver settings applied through ``Solver.update``.
    blocksize
        Threads per block of the launch; ``None`` uses the solver's.
    resident_blocks
        Blocks per SM held resident; ``None`` uses the solver's.
    """

    label: str
    settings: Dict[str, Any] = field(factory=dict)
    blocksize: Optional[int] = None
    resident_blocks: Optional[int] = None


@define
class CandidateTiming:
    """Measured outcome of one candidate.

    Parameters
    ----------
    candidate
        The configuration timed.
    times_ms
        Kernel milliseconds of every timed solve.
    failures
        Runs with a nonzero status code in the last solve read.
    runs
        Trajectories each solve integrated.
    blocks_per_sm
        Resident blocks per SM the driver reported for the launch.
    waves
        Occupancy waves the batch filled at the launch.
    error
        Why the candidate could not be timed, empty when it was.
    """

    candidate: Candidate
    times_ms: Tuple[float, ...] = ()
    failures: int = 0
    runs: int = 0
    blocks_per_sm: int = 0
    waves: float = 0.0
    error: str = ""

    @property
    def timed(self) -> bool:
        """Whether the candidate has a time."""
        return bool(self.times_ms)

    @property
    def best_ms(self) -> float:
        """Lowest kernel time, ``inf`` when never timed."""
        return min(self.times_ms) if self.times_ms else float("inf")

    @property
    def success_rate(self) -> float:
        """Share of the batch that integrated without a status flag."""
        if self.runs == 0:
            return 0.0
        return (self.runs - self.failures) / self.runs


def rank_timings(timings: Sequence[CandidateTiming]) -> List[CandidateTiming]:
    """Order timed candidates: the top success tier by time, then the rest.

    The top tier holds every candidate whose success rate is at least
    :data:`SUCCESS_TIER_FRACTION` of the best success rate.
    """
    timed = [timing for timing in timings if timing.timed]
    if not timed:
        return []
    best = max(timing.success_rate for timing in timed)
    floor = SUCCESS_TIER_FRACTION * best
    top = [timing for timing in timed if timing.success_rate >= floor]
    rest = [timing for timing in timed if timing.success_rate < floor]
    key = (lambda timing: timing.best_ms)
    return sorted(top, key=key) + sorted(rest, key=key)


def settings_in_effect(solver: Any) -> Dict[str, Any]:
    """Return every setting in effect on ``solver``, unroll flags included."""
    unroll = solver.system.compile_settings.unroll
    values = dict(solver.kernel.settings_dict)
    values.update(
        {fld.name: getattr(unroll, fld.name) for fld in fields(UnrollFlags)}
    )
    values.update(
        {
            key: value
            for key, value in solver.effective.as_kwargs().items()
            if value is not None
        }
    )
    return values


def _error(exc: Exception) -> str:
    """Return the one-line record of ``exc``."""
    return f"{type(exc).__name__}: {exc}"


def _compile_solver(solver: Any) -> None:
    """Compile the solver's current configuration."""
    kernel = solver.kernel
    kernel.compile(kernel.duration, kernel.warmup, kernel.t0)


def _pool_pays(
    compile_seconds: Optional[float], misses: int, max_parallel: int
) -> bool:
    """Whether spawning workers beats compiling ``misses`` in turn."""
    if compile_seconds is None or misses < 2 or max_parallel < 2:
        return False
    serial = compile_seconds * misses
    workers = min(max_parallel, misses)
    pooled = WORKER_STARTUP_SECONDS + compile_seconds * ceil(misses / workers)
    return serial > pooled


def _compile_candidate(payload: Tuple) -> Tuple[int, str]:
    """Compile one settings set in a worker; return (index, error)."""
    index, system_bytes, settings, drivers, cache_root = payload
    if cache_root is not None:
        set_cache_root(cache_root)
    from cubie.batchsolving.solver import Solver

    system = pickle.loads(system_bytes)
    solver = None
    try:
        solver = Solver(system, **settings)
        if drivers is not None:
            solver._configure_drivers(drivers)
        _compile_solver(solver)
        return index, ""
    except Exception as exc:
        return index, _error(exc)
    finally:
        if solver is not None:
            solver.close()


def _compile_in_pool(
    solver: Any, settings_sets: Sequence[Dict[str, Any]], max_parallel: int
) -> Tuple[str, ...]:
    """Compile each set in a spawned worker; return one error per set."""
    # Pickled into spawned workers; the manager holds CUDA state.
    system_bytes = pickle.dumps(solver.system)
    drivers = solver.kernel.driver_inputs()
    record = {
        key: value
        for key, value in solver.settings_dict.items()
        if key != "memory_manager"
    }
    payloads = [
        (
            index,
            system_bytes,
            {**record, **settings},
            drivers,
            get_cache_root_override(),
        )
        for index, settings in enumerate(settings_sets)
    ]
    errors = [""] * len(payloads)
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(max_parallel, len(payloads))) as pool:
        for index, error in pool.imap_unordered(_compile_candidate, payloads):
            errors[index] = error
    return tuple(errors)


def compile_kernels(
    solver: Any, settings_sets: Sequence[Dict[str, Any]], max_parallel: int
) -> Tuple[str, ...]:
    """Compile a kernel per set of settings in ``max_parallel`` threads
    if cheaper than serial.

    Parameters
    ----------
    solver
        The solver each set applies over; restored on return.
    settings_sets
        One ``Solver.update`` dict per kernel.
    max_parallel
        Maximum compilations to run in parallel.

    Returns
    -------
    tuple of str
        The error that rejected each set, empty when compiled.
    """
    given = dict(solver.given.as_kwargs())
    baseline = settings_in_effect(solver)
    keys = set()
    for settings in settings_sets:
        keys.update(settings)
    opening = {key: baseline.get(key) for key in keys}

    def select(settings):
        solver.update({**opening, **settings}, silent=True)

    errors = [""] * len(settings_sets)
    missing = []
    for index, settings in enumerate(settings_sets):
        try:
            select(settings)
            cached = (
                solver.cache_enabled and solver.kernel.kernel_is_cached()
            )
        except Exception as exc:
            errors[index] = _error(exc)
            continue
        if not cached:
            missing.append(index)
    # The first miss's compile time decides whether the rest pool.
    compile_seconds = None
    while missing:
        index = missing.pop(0)
        try:
            select(settings_sets[index])
            _compile_solver(solver)
        except Exception as exc:
            errors[index] = _error(exc)
            continue
        compile_seconds = default_timelogger.get_event_duration(
            "compile_cuda_kernel"
        )
        break
    if missing and solver.cache_enabled and _pool_pays(
        compile_seconds, len(missing), max_parallel
    ):
        pooled = _compile_in_pool(
            solver, [settings_sets[index] for index in missing], max_parallel
        )
        for index, error in zip(missing, pooled):
            errors[index] = error
    else:
        for index in missing:
            try:
                select(settings_sets[index])
                _compile_solver(solver)
            except Exception as exc:
                errors[index] = _error(exc)
    if keys:
        # Every value in effect back, then the given record.
        solver.update(baseline, silent=True)
        solver.update(
            {key: given.get(key) for key in keys | set(baseline)},
            silent=True,
        )
    return tuple(errors)


class ComparisonRunner:
    """Switch one solver between candidates and time each on one batch.

    The batch stays on the device; candidates apply over the
    configuration at :meth:`open`, which :meth:`close` restores.

    Parameters
    ----------
    solver
        The solver whose settings the candidates vary.
    inits, params
        The ``(variable, run)`` grid to time on.
    duration, settling_time, t0
        Integration window of the timed solves.
    verbose
        Print progress lines.
    max_parallel
        Maximum compilations to run in parallel.
    """

    def __init__(
        self,
        solver: Any,
        inits: Any,
        params: Any,
        duration: float,
        settling_time: float,
        t0: float,
        verbose: bool = False,
        max_parallel: int = 4,
    ) -> None:
        self._solver = solver
        self._max_parallel = int(max_parallel)
        self._grid = (device_to_host(inits), device_to_host(params))
        self.duration = float(duration)
        self.settling = float(settling_time)
        self._t0 = float(t0)
        self._verbose = bool(verbose)
        self._given = {}
        self._baseline = {}
        self._touched = set()
        self._rejected = {}
        self._resident_blocks = solver.kernel.resident_blocks
        self._verbosity = default_timelogger.verbosity
        self._inits = None
        self._params = None
        self._codes = None
        self.runs = 0
        self._concurrent = []
        self._floor = 0
        self._open = False

    def __enter__(self) -> "ComparisonRunner":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def emit(self, message: str) -> None:
        """Log ``message``; print it when verbose."""
        logger.debug(message)
        if self._verbose:
            print(message, flush=True)

    def open(self) -> None:
        """Arm event timing and record the solver's configuration."""
        if self._open:
            return
        self._open = True
        # "silent" records the kernel events the timings read.
        self._verbosity = default_timelogger.verbosity
        default_timelogger.set_verbosity("silent")
        solver = self._solver
        self._given = dict(solver.given.as_kwargs())
        self._baseline = settings_in_effect(solver)
        self._touched = set()

    def close(self) -> None:
        """Restore the solver's configuration and release the batch."""
        if not self._open:
            return
        self._open = False
        try:
            solver = self._solver
            touched = set(self._touched)
            if touched:
                # Opening values back first, then the given record.
                solver.update(
                    {key: self._baseline.get(key) for key in touched},
                    silent=True,
                )
            # Every timed solve records its duration.
            touched.add("duration")
            solver.update(
                {key: self._given.get(key) for key in touched},
                silent=True,
            )
            solver.kernel.resident_blocks = self._resident_blocks
        finally:
            self._touched = set()
            self._inits = None
            self._params = None
            self._codes = None
            default_timelogger.set_verbosity(self._verbosity)

    def select(self, candidate: Candidate) -> None:
        """Apply ``candidate`` over the opening configuration."""
        settings = {
            key: self._baseline.get(key)
            for key in self._touched
            if key not in candidate.settings
        }
        settings.update(candidate.settings)
        self._touched.update(candidate.settings)
        self._solver.update(settings, silent=True)
        self._solver.kernel.resident_blocks = candidate.resident_blocks

    def rejection(self, candidate: Candidate) -> str:
        """Return why ``candidate`` was rejected; empty when it was not."""
        return self._rejected.get(_settings_key(candidate.settings), "")

    def set_batch(self, runs: Optional[int] = None) -> None:
        """Stage ``runs`` grid columns on the device, cycling if short."""
        inits, params = self._grid
        runs = inits.shape[1] if runs is None else int(runs)
        if runs != inits.shape[1]:
            columns = np_arange(runs) % inits.shape[1]
            inits = np_take(inits, columns, axis=1)
            params = np_take(params, columns, axis=1)
        self._inits = self._stage(inits)
        self._params = self._stage(params)
        self._codes = None
        self.runs = runs

    @property
    def staged_bytes(self) -> int:
        """Device bytes of the staged grid."""
        return sum(
            grid.nbytes
            for grid in (self._inits, self._params)
            if grid is not None
        )

    @staticmethod
    def _stage(grid: ndarray) -> Any:
        """Upload ``grid``; a grid with no variables is passed as None."""
        if grid.shape[0] == 0:
            return None
        return cuda.to_device(grid)

    def compile(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """Compile and return the accepted candidates; a rejected one
        keeps its error for :meth:`time`."""
        fresh = []
        seen = set()
        for candidate in candidates:
            key = _settings_key(candidate.settings)
            if key in seen or key in self._rejected:
                continue
            seen.add(key)
            fresh.append(candidate)
        # Sets apply over the opening configuration.
        self.select(Candidate("opening"))
        errors = compile_kernels(
            self._solver,
            tuple(candidate.settings for candidate in fresh),
            self._max_parallel,
        )
        for candidate, error in zip(fresh, errors):
            if error:
                self._rejected[_settings_key(candidate.settings)] = error
                self.emit(f"  {candidate.label}: rejected ({error})")
            else:
                self.emit(f"  {candidate.label}: compiled")
        return [
            candidate
            for candidate in candidates
            if _settings_key(candidate.settings) not in self._rejected
        ]

    def solve_ms(self, blocksize: Optional[int] = None) -> float:
        """Solve the staged batch once; return its kernel milliseconds."""
        solver = self._solver
        solver.solve(
            self._inits,
            self._params,
            duration=self.duration,
            settling_time=self.settling,
            t0=self._t0,
            blocksize=blocksize,
            on_device=True,
        )
        kernel = solver.kernel
        kernel.synchronize()
        return float(
            sum(
                event.elapsed_time_ms()
                for event in kernel._cuda_events
                if event.name.startswith("kernel_chunk")
            )
        )

    def warm(self) -> float:
        """Lift the clocks with the busy kernel; return its milliseconds."""
        measured = warm_clocks(self._solver.kernel.stream)
        self.emit(f"warm: {measured:.1f} ms busy")
        return measured

    def failures(self) -> int:
        """Return the failed-run count of the last solve."""
        kernel = self._solver.kernel
        codes = kernel.device_status_codes
        if self._codes is None or self._codes.shape != codes.shape:
            self._codes = kernel.memory_manager.create_host_array(
                tuple(codes.shape), np_int32, "pinned"
            )
        stream = kernel.stream
        codes.copy_to_host(self._codes, stream=stream)
        stream.synchronize()
        return int(np_count_nonzero(self._codes))

    def _launch_blocks(
        self, blocksize: Optional[int], runs: Optional[int]
    ) -> Tuple[int, int]:
        """Return blocks per SM and runs per block of the current launch."""
        kernel = self._solver.kernel
        actual, dynamic = kernel.launch_geometry(blocksize, runs=runs)
        blocks = active_blocks_per_multiprocessor(
            kernel.kernel, actual, dynamic, kernel.signature
        )
        return int(blocks), actual // kernel.threads_per_loop

    def concurrent_runs(self, blocksize: Optional[int] = None) -> int:
        """Runs the current candidate's launch executes at once."""
        blocks, runs_per_block = self._launch_blocks(blocksize, None)
        multiprocessors = device_hardware().multiprocessor_count
        return blocks * multiprocessors * runs_per_block

    def geometry(self, blocksize: Optional[int]) -> Tuple[int, float]:
        """Return blocks per SM and waves of the current launch."""
        blocks, runs_per_block = self._launch_blocks(blocksize, self.runs)
        total_blocks = ceil(self.runs / runs_per_block)
        resident = blocks * device_hardware().multiprocessor_count
        return blocks, total_blocks / resident

    def size_batch(self, candidates: Sequence[Candidate], waves: int) -> None:
        """Stage the tail-safe batch at ``waves`` of the most concurrent
        candidate; keep the counts for :meth:`fit_batch`."""
        concurrent = []
        for candidate in candidates:
            self.select(candidate)
            concurrent.append(self.concurrent_runs(candidate.blocksize))
        self._concurrent = concurrent
        if not concurrent:
            self.set_batch()
            return
        most = max(concurrent)
        self._floor = TIMED_WAVES_FLOOR * most
        self.set_batch(
            tail_safe_runs(concurrent, int(waves) * most, self._floor)
        )
        self.emit(f"batch: {self.runs} runs")

    def batch_cap(self) -> int:
        """Runs that fit in memory at the staged batch's bytes per run."""
        kernel = self._solver.kernel
        manager = kernel.memory_manager
        allocated = sum(
            manager.get_registration(arrays).allocated_bytes
            for arrays in (kernel.input_arrays, kernel.output_arrays)
        )
        available = manager.get_available_memory(
            manager.get_stream_group(kernel)
        )
        bytes_per_run = (allocated + self.staged_bytes) / self.runs
        return int((available + allocated) // bytes_per_run)

    def fit_batch(
        self, measured: float, target_ms: float, grow: bool, shrink: bool
    ) -> float:
        """Move the batch once toward ``target_ms``, up within
        :meth:`batch_cap` if ``grow``, down to the floor if ``shrink``;
        return the kernel time at the staged batch."""
        concurrent = self._concurrent
        if not concurrent:
            return measured
        runs = self.runs
        wanted = int(runs * target_ms / measured)
        if grow and measured < target_ms:
            runs = tail_safe_runs(
                concurrent, wanted, self._floor, self.batch_cap()
            )
        elif shrink and measured > target_ms:
            runs = tail_safe_runs(concurrent, wanted, self._floor, self.runs)
        if runs == self.runs:
            return measured
        self.set_batch(runs)
        try:
            measured = self.solve_ms(None)
        except ValueError:
            partition = self._solver.kernel.run_params
            if partition.num_chunks <= 1:
                raise
            # The live partition is the fresh single-chunk fit.
            runs = tail_safe_runs(
                concurrent,
                partition.chunk_length,
                self._floor,
                partition.chunk_length,
            )
            self.set_batch(runs)
            measured = self.solve_ms(None)
        self.emit(f"batch: {self.runs} runs -> {measured:.3f} ms")
        return measured

    def time(self, candidates: Sequence[Candidate]) -> List[CandidateTiming]:
        """Time every candidate on the staged batch, forward then back.

        Every candidate solves :data:`SOLVES_PER_ROUND` times in each of
        :data:`ROUNDS` rounds; the second round reverses the order. A
        rejected or failing candidate carries the error and no times.
        """
        timings = [
            CandidateTiming(
                candidate=candidate,
                runs=self.runs,
                error=self.rejection(candidate),
            )
            for candidate in candidates
        ]
        for round_index in range(ROUNDS):
            ordered = timings if round_index == 0 else timings[::-1]
            for timing in ordered:
                if timing.error:
                    continue
                candidate = timing.candidate
                try:
                    self.select(candidate)
                    for _ in range(SOLVES_PER_ROUND):
                        timing.times_ms += (
                            self.solve_ms(candidate.blocksize),
                        )
                    if round_index == 0:
                        timing.failures = self.failures()
                        timing.blocks_per_sm, timing.waves = self.geometry(
                            candidate.blocksize
                        )
                except Exception as exc:
                    timing.times_ms = ()
                    timing.error = f"{type(exc).__name__}: {exc}"
                    self.emit(f"  {candidate.label}: failed ({timing.error})")
                    continue
                if round_index == ROUNDS - 1:
                    self.emit(
                        f"  {candidate.label}: {timing.best_ms:.3f} ms"
                        f" ({timing.failures} failed)"
                    )
        return timings
