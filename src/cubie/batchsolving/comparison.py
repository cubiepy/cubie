"""Time candidate settings of one solver on one device-resident batch.

Published Objects
-----------------
:class:`Candidate`
    Settings and launch of one configuration to time.
:class:`CandidateTiming`
    One candidate's kernel times and failed-run count.
:class:`ComparisonRunner`
    Switches one solver between candidates and times each one.
:func:`rank_timings`
    Order timings by success tier, then time.
"""

import logging
import multiprocessing
import pickle
from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Tuple

from attrs import define, field
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
from cubie.cuda_simsafe import cuda
from cubie.time_logger import default_timelogger

logger = logging.getLogger(__name__)

ROUNDS = 2
"""Timing rounds; the second visits the candidates in reverse order."""

SOLVES_PER_ROUND = 3
"""Timed solves per candidate per round."""

SUCCESS_TIER_FRACTION = 0.95
"""Share of the best success rate a candidate keeps to rank on time."""

WORKERS = 4
"""Compile processes the pool spawns."""

WORKER_STARTUP_SECONDS = 12.0
"""Wall seconds a spawned worker spends importing cubie.

Empirical: a fresh ``import cubie`` in a spawned process on the
development machine (mlir compat 6 s, odesystems 4.8 s, cupy 1.5 s).
"""


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


def _compile_candidate(payload: Tuple) -> Tuple[str, str]:
    """Compile one candidate in a worker process; return its hash."""
    (
        label,
        system_bytes,
        settings,
        drivers,
        duration,
        settling_time,
        t0,
        cache_root,
    ) = payload
    if cache_root is not None:
        set_cache_root(cache_root)
    from cubie.batchsolving.solver import Solver

    system = pickle.loads(system_bytes)
    solver = Solver(system, **settings)
    try:
        solver.compile(
            drivers=drivers,
            duration=duration,
            settling_time=settling_time,
            t0=t0,
        )
        return label, solver.kernel.config_hash
    finally:
        solver.close()


class ComparisonRunner:
    """Switch one solver between candidates and time each on one batch.

    The batch lives on the device once; every candidate solves it in
    place with no host output buffers. The solver's given settings and
    residency are restored on :meth:`close`.

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
    ) -> None:
        self._solver = solver
        self._grid = (device_to_host(inits), device_to_host(params))
        self.duration = float(duration)
        self.settling = float(settling_time)
        self._t0 = float(t0)
        self._verbose = bool(verbose)
        self._original = {}
        self._pinned = {}
        self._resident_blocks = solver.kernel.resident_blocks
        self._verbosity = default_timelogger.verbosity
        self._inits = None
        self._params = None
        self._codes = None
        self.runs = 0
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
        """Arm event timing and pin an unset summary window."""
        if self._open:
            return
        self._open = True
        # "silent" records the kernel events the timings read.
        self._verbosity = default_timelogger.verbosity
        default_timelogger.set_verbosity("silent")
        solver = self._solver
        # Every timed solve records its duration; close puts it back.
        self._original["duration"] = solver.given.as_kwargs().get(
            "duration"
        )
        if (
            solver.kernel.single_integrator.summary_outputs_requested
            and not solver.given.is_given("summarise_every")
        ):
            # An unset window follows the duration; pin it so probe
            # durations share one kernel.
            self._pinned = {"summarise_every": self.duration}
            self.apply(self._pinned)

    def close(self) -> None:
        """Restore the solver's settings and release the batch."""
        if not self._open:
            return
        self._open = False
        try:
            solver = self._solver
            if self._original:
                solver.update(dict(self._original), silent=True)
                self._original = {}
            solver.kernel.resident_blocks = self._resident_blocks
        finally:
            self._inits = None
            self._params = None
            self._codes = None
            default_timelogger.set_verbosity(self._verbosity)

    def apply(self, settings: Dict[str, Any]) -> None:
        """Apply ``settings`` to the solver, remembering the given values."""
        given = self._solver.given.as_kwargs()
        for key in settings:
            if key not in self._original:
                self._original[key] = given.get(key)
        self._solver.update(dict(settings), silent=True)

    def select(self, candidate: Candidate) -> None:
        """Make ``candidate`` the solver's configuration and residency."""
        self.apply(candidate.settings)
        self._solver.kernel.resident_blocks = candidate.resident_blocks

    def set_batch(self, runs: Optional[int] = None) -> None:
        """Stage ``runs`` grid columns on the device, cycling if short."""
        inits, params = self._grid
        runs = inits.shape[1] if runs is None else int(runs)
        if runs != inits.shape[1]:
            columns = np_arange(runs) % inits.shape[1]
            inits = np_take(inits, columns, axis=1)
            params = np_take(params, columns, axis=1)
        self._inits = cuda.to_device(inits)
        self._params = cuda.to_device(params)
        self._codes = None
        self.runs = runs

    def compile(self, candidates: Sequence[Candidate]) -> None:
        """Compile every candidate, in workers when that pays.

        The first uncached candidate compiles in this process and its
        measured time decides whether the remaining misses go to a
        spawn pool.
        """
        solver = self._solver
        missing = []
        seen = set()
        for candidate in candidates:
            key = tuple(sorted(candidate.settings.items()))
            if key in seen:
                continue
            seen.add(key)
            self.select(candidate)
            if solver.cache_enabled and solver.kernel.kernel_is_cached():
                self.emit(f"  {candidate.label}: cached")
            else:
                missing.append(candidate)
        if not missing:
            return
        first = missing.pop(0)
        self.select(first)
        self._compile_current()
        compile_seconds = default_timelogger.get_event_duration(
            "compile_cuda_kernel"
        )
        self.emit(f"  {first.label}: compiled")
        if not missing:
            return
        if solver.cache_enabled and self._pool_pays(
            compile_seconds, len(missing)
        ):
            self._compile_in_pool(missing)
            return
        for candidate in missing:
            self.select(candidate)
            self._compile_current()
            self.emit(f"  {candidate.label}: compiled")

    def _compile_current(self) -> None:
        """Compile the solver's current configuration."""
        self._solver.compile(
            duration=self.duration,
            settling_time=self.settling,
            t0=self._t0,
        )

    @staticmethod
    def _pool_pays(compile_seconds: Optional[float], misses: int) -> bool:
        """Whether spawning workers beats compiling ``misses`` in turn."""
        if compile_seconds is None or misses < 2:
            return False
        serial = compile_seconds * misses
        workers = min(WORKERS, misses)
        pooled = WORKER_STARTUP_SECONDS + compile_seconds * ceil(
            misses / workers
        )
        return serial > pooled

    def _compile_in_pool(self, candidates: Sequence[Candidate]) -> None:
        """Compile ``candidates`` into the kernel cache in spawned workers."""
        solver = self._solver
        # Pickled into spawned workers; the manager holds CUDA state.
        # Keys a candidate touched go back to their given values.
        settings = {
            key: value
            for key, value in solver.settings_dict.items()
            if key != "memory_manager" and key not in self._original
        }
        settings.update(
            {
                key: value
                for key, value in self._original.items()
                if value is not None
            }
        )
        settings.update(self._pinned)
        system_bytes = pickle.dumps(solver.system)
        drivers = solver.kernel.driver_inputs()
        payloads = [
            (
                candidate.label,
                system_bytes,
                {**settings, **candidate.settings},
                drivers,
                self.duration,
                self.settling,
                self._t0,
                get_cache_root_override(),
            )
            for candidate in candidates
        ]
        context = multiprocessing.get_context("spawn")
        with context.Pool(min(WORKERS, len(payloads))) as pool:
            for label, config_hash in pool.imap_unordered(
                _compile_candidate, payloads
            ):
                self.emit(f"  {label}: worker compiled {config_hash[:12]}")

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

    def geometry(self, blocksize: Optional[int]) -> Tuple[int, float]:
        """Return blocks per SM and waves of the current launch."""
        kernel = self._solver.kernel
        actual, dynamic = kernel.launch_geometry(blocksize, runs=self.runs)
        blocks = active_blocks_per_multiprocessor(
            kernel.kernel, actual, dynamic, kernel.signature
        )
        runs_per_block = actual // kernel.threads_per_loop
        total_blocks = ceil(self.runs / runs_per_block)
        resident = blocks * device_hardware().multiprocessor_count
        return int(blocks), total_blocks / resident

    def time(self, candidates: Sequence[Candidate]) -> List[CandidateTiming]:
        """Time every candidate on the staged batch, forward then back.

        Every candidate solves :data:`SOLVES_PER_ROUND` times in each of
        :data:`ROUNDS` rounds; the second round reverses the order. A
        candidate that fails to switch or solve carries the error and
        no times.
        """
        timings = [
            CandidateTiming(candidate=candidate, runs=self.runs)
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
