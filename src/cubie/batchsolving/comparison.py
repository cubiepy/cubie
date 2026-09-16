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
from cubie.cuda_simsafe import cuda
from cubie.CUDAFactory import UnrollFlags
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


def _compile_candidate(payload: Tuple) -> Tuple[int, str, str]:
    """Compile one candidate in a worker; return (index, hash, error)."""
    (
        index,
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
    solver = None
    try:
        solver = Solver(system, **settings)
        solver.compile(
            drivers=drivers,
            duration=duration,
            settling_time=settling_time,
            t0=t0,
        )
        return index, solver.kernel.config_hash, ""
    except Exception as exc:
        return index, "", f"{type(exc).__name__}: {exc}"
    finally:
        if solver is not None:
            solver.close()


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
        self._given = {}
        self._baseline = {}
        self._settings = {}
        self._touched = set()
        self._rejected = {}
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
        """Arm event timing and record the solver's configuration."""
        if self._open:
            return
        self._open = True
        # "silent" records the kernel events the timings read.
        self._verbosity = default_timelogger.verbosity
        default_timelogger.set_verbosity("silent")
        solver = self._solver
        self._given = dict(solver.given.as_kwargs())
        self._baseline = self.settings_in_effect()
        # The record a worker rebuilds the solver from.
        self._settings = {
            key: value
            for key, value in solver.settings_dict.items()
            if key != "memory_manager"
        }
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

    def settings_in_effect(self) -> Dict[str, Any]:
        """Return every setting in effect, unroll flags included."""
        solver = self._solver
        unroll = solver.system.compile_settings.unroll
        values = dict(solver.kernel.settings_dict)
        values.update(
            {
                fld.name: getattr(unroll, fld.name)
                for fld in fields(UnrollFlags)
            }
        )
        values.update(
            {
                key: value
                for key, value in solver.effective.as_kwargs().items()
                if value is not None
            }
        )
        return values

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

    def _reject(self, candidate: Candidate, exc: Exception) -> None:
        """Record ``candidate`` as rejected with ``exc``."""
        error = f"{type(exc).__name__}: {exc}"
        self._rejected[_settings_key(candidate.settings)] = error
        self.emit(f"  {candidate.label}: rejected ({error})")

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

    @staticmethod
    def _stage(grid: ndarray) -> Any:
        """Upload ``grid``; a grid with no variables is passed as None."""
        if grid.shape[0] == 0:
            return None
        return cuda.to_device(grid)

    def compile(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """Compile and return the accepted candidates; a rejected one
        keeps its error for :meth:`time`, and the first miss's compile
        time decides whether the rest go to a spawn pool."""
        solver = self._solver
        missing = []
        seen = set()
        for candidate in candidates:
            key = _settings_key(candidate.settings)
            if key in seen or key in self._rejected:
                continue
            seen.add(key)
            try:
                self.select(candidate)
                cached = (
                    solver.cache_enabled and solver.kernel.kernel_is_cached()
                )
            except Exception as exc:
                self._reject(candidate, exc)
                continue
            if cached:
                self.emit(f"  {candidate.label}: cached")
            else:
                missing.append(candidate)
        compile_seconds = None
        while missing:
            first = missing.pop(0)
            try:
                self.select(first)
                self._compile_current()
            except Exception as exc:
                self._reject(first, exc)
                continue
            compile_seconds = default_timelogger.get_event_duration(
                "compile_cuda_kernel"
            )
            self.emit(f"  {first.label}: compiled")
            break
        if missing and solver.cache_enabled and self._pool_pays(
            compile_seconds, len(missing)
        ):
            self._compile_in_pool(missing)
        else:
            for candidate in missing:
                try:
                    self.select(candidate)
                    self._compile_current()
                except Exception as exc:
                    self._reject(candidate, exc)
                    continue
                self.emit(f"  {candidate.label}: compiled")
        return [
            candidate
            for candidate in candidates
            if _settings_key(candidate.settings) not in self._rejected
        ]

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
        system_bytes = pickle.dumps(solver.system)
        drivers = solver.kernel.driver_inputs()
        payloads = [
            (
                index,
                system_bytes,
                {**self._settings, **candidate.settings},
                drivers,
                self.duration,
                self.settling,
                self._t0,
                get_cache_root_override(),
            )
            for index, candidate in enumerate(candidates)
        ]
        context = multiprocessing.get_context("spawn")
        with context.Pool(min(WORKERS, len(payloads))) as pool:
            for index, config_hash, error in pool.imap_unordered(
                _compile_candidate, payloads
            ):
                candidate = candidates[index]
                if error:
                    self._rejected[_settings_key(candidate.settings)] = error
                    self.emit(f"  {candidate.label}: rejected ({error})")
                else:
                    self.emit(
                        f"  {candidate.label}: worker compiled "
                        f"{config_hash[:12]}"
                    )

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
