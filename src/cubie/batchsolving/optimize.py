"""Backend of :meth:`cubie.Solver.optimize`: time unroll, placement and
launch candidates on solver copies and apply the fastest.

Published Objects
-----------------
:class:`LaunchResult`
    One candidate's timings at one launch.
:class:`OptimizeResult`
    Every launch, the best one, and the applied settings.
:func:`launch_candidates`
    Launches a kernel can time.
:func:`sized_batch_runs`
    Runs filling a number of occupancy waves at a kernel's launch.
:func:`apply_launch`
    Apply one launch to a solver.
:func:`run_optimization`
    Time a solver's candidates.
"""

import logging
import multiprocessing
import pickle
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from warnings import warn

from attrs import define
from numpy import arange as np_arange
from numpy import take as np_take
from numpy import zeros as np_zeros

from cubie.backend.utils import (
    active_blocks_per_multiprocessor,
    compile_kernel_specialization,
    device_hardware,
    kernel_resources,
)
from cubie.batchsolving.calibration import _achieved_waves
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cuda_simsafe import cuda
from cubie.time_logger import default_timelogger

logger = logging.getLogger(__name__)

WORKERS = 4
"""Parallel compile processes pre-warming the kernel cache."""

TIMED_SOLVES = 3
"""Timed solves per launch per round."""

ROUNDS = 2
"""Timing rounds; the second visits the launches in reverse order."""

EXCLUSION_RATIO = 2.0
"""Launches slower than this multiple of the fastest are dropped."""

CONFIRMATION_RATIO = 3.0
"""Slow first solves under this multiple of the fastest are repeated."""

PROBE_FRACTIONS = (0.01, 0.1, 1.0)
"""Fractions of the given duration the ``"auto"`` probe ramps through."""

LOCAL_LAUNCH_BLOCKSIZES = (64, 256)
"""Block sizes timed for local-only kernels."""

SHARED_LAUNCH_BLOCKSIZES = (32, 64, 128, 256)
"""Block sizes timed for shared-memory kernels."""


def _label(settings: Dict[str, Any]) -> str:
    """Return a short name for a candidate's settings."""
    parts = []
    for key, value in settings.items():
        if isinstance(value, Enum):
            value = value.name.lower()
        name = key.removeprefix("unroll_").removesuffix("_location")
        parts.append(f"{name}={value}")
    return " ".join(parts) or "current"


@define
class LaunchResult:
    """Timings of one candidate at one launch.

    Parameters
    ----------
    settings
        The candidate's unroll and placement settings.
    blocksize
        Threads per block of the launch.
    resident_blocks
        Blocks per SM held resident; ``None`` = the default residency.
    blocks_per_sm
        Resident blocks per SM the driver reported.
    times_ms
        Timed solve times in milliseconds.
    excluded
        Whether the launch was dropped as slow.
    """

    settings: Dict[str, Any]
    blocksize: int
    resident_blocks: Optional[int]
    blocks_per_sm: int = 0
    times_ms: Tuple[float, ...] = ()
    excluded: bool = False

    @property
    def timed(self) -> bool:
        """Whether the launch has a ranking time."""
        return bool(self.times_ms) and not self.excluded

    @property
    def best_ms(self) -> float:
        """Lowest solve time, ``inf`` when never solved."""
        return min(self.times_ms) if self.times_ms else float("inf")

    @property
    def label(self) -> str:
        """Candidate settings and launch as one line."""
        resident = "" if self.resident_blocks is None else (
            f" x{self.resident_blocks}"
        )
        return f"{_label(self.settings)} @bs{self.blocksize}{resident}"


@define
class OptimizeResult:
    """Complete optimisation report.

    Parameters
    ----------
    launches
        Every launch measured, in run order.
    best
        Fastest launch, or ``None`` when nothing was timed.
    applied_settings
        Settings applied to the calling solver, empty when none.
    runs
        Trajectories each timed solve integrated.
    duration
        Integration time each timed solve ran.
    """

    launches: List[LaunchResult]
    best: Optional[LaunchResult]
    applied_settings: Dict[str, Any]
    runs: int = 0
    duration: float = 0.0

    @property
    def ranking(self) -> List[LaunchResult]:
        """Every timed launch, fastest first."""
        timed = [launch for launch in self.launches if launch.timed]
        return sorted(timed, key=lambda launch: launch.best_ms)

    def summary(self) -> str:
        """Return a formatted table of every launch measurement."""
        header = f"{'launch':<48}{'blk/SM':>8}{'best ms':>10}  note"
        lines = [
            f"{self.runs} runs x {self.duration:g} time units per solve",
            header,
            "-" * len(header),
        ]
        ranks = {
            id(launch): position + 1
            for position, launch in enumerate(self.ranking)
        }
        for launch in self.launches:
            best = f"{launch.best_ms:.3f}"
            if launch.excluded:
                note = "excluded"
            else:
                rank = ranks.get(id(launch))
                note = "best" if rank == 1 else f"rank {rank}"
            lines.append(
                f"{launch.label:<48}{launch.blocks_per_sm:>8}"
                f"{best:>10}  {note}"
            )
        return "\n".join(lines)


def launch_candidates(
    kernel: Any, blocksizes: Optional[Sequence[int]] = None
) -> Tuple[Tuple[int, Optional[int]], ...]:
    """Return the ``(blocksize, resident_blocks)`` launches worth timing.

    Parameters
    ----------
    kernel
        A compiled :class:`~cubie.batchsolving.BatchSolverKernel`.
    blocksizes
        Block sizes to consider; ``None`` picks the measured set for
        shared-memory or local-only kernels.

    Returns
    -------
    tuple
        Launchable block sizes at the default residency, plus one and
        two blocks under natural occupancy for local frames.
    """
    if blocksizes is None:
        blocksizes = (
            SHARED_LAUNCH_BLOCKSIZES
            if kernel.shared_memory_bytes > 0
            else LOCAL_LAUNCH_BLOCKSIZES
        )
    runs = kernel.run_params[0].runs
    pad = 4 if kernel.shared_memory_needs_padding else 0
    padded_bytes = kernel.shared_memory_bytes + pad
    dispatcher = kernel.kernel
    compile_kernel_specialization(
        dispatcher, kernel._kernel_launch_args(kernel.run_params[0])
    )
    frame = kernel_resources(dispatcher).local_bytes_per_thread
    cells = []
    for blocksize in blocksizes:
        actual, dynamic_sharedmem = kernel.limit_blocksize(
            blocksize,
            int(padded_bytes * min(runs, blocksize)),
            padded_bytes,
            runs,
        )
        # A block size the shared footprint cannot launch is skipped.
        if actual != blocksize:
            continue
        cells.append((blocksize, None))
        if frame == 0:
            continue
        natural = active_blocks_per_multiprocessor(
            dispatcher, blocksize, max(4, dynamic_sharedmem)
        )
        for cut in (1, 2):
            if natural - cut >= 1:
                cells.append((blocksize, natural - cut))
    return tuple(cells)


def sized_batch_runs(kernel: Any, waves: int = 5) -> int:
    """Return the runs filling ``waves`` occupancy waves of ``kernel``.

    Parameters
    ----------
    kernel
        A compiled :class:`~cubie.batchsolving.BatchSolverKernel`.
    waves
        Occupancy waves the batch fills at the kernel's default launch.

    Returns
    -------
    int
        ``waves`` times the SMs times the resident blocks per SM times
        the runs per block at the default launch geometry.
    """
    blocksize, dynamic_sharedmem = kernel.launch_geometry()
    blocks_per_sm = active_blocks_per_multiprocessor(
        kernel.kernel, blocksize, dynamic_sharedmem
    )
    runs_per_block = blocksize // kernel.threads_per_loop
    multiprocessors = device_hardware().multiprocessor_count
    return int(waves) * multiprocessors * blocks_per_sm * runs_per_block


def apply_launch(parent: Any, launch: LaunchResult) -> Dict[str, Any]:
    """Apply a launch's settings, block size and residency to ``parent``.

    Returns
    -------
    dict
        The settings passed to ``parent.update``.
    """
    settings = {**launch.settings, "blocksize": launch.blocksize}
    parent.update(settings)
    parent.kernel.resident_blocks = launch.resident_blocks
    return settings


def _compile_candidate(payload: Tuple) -> Tuple[str, str]:
    """Compile one candidate in a worker process; return its hash."""
    (
        label,
        system_bytes,
        settings,
        candidate,
        n_runs,
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
    solver = Solver(system, **{**settings, **candidate})
    try:
        sizes = system.sizes
        solver.compile(
            np_zeros((sizes.states, n_runs), dtype=system.precision),
            np_zeros((sizes.parameters, n_runs), dtype=system.precision),
            drivers=drivers,
            duration=duration,
            settling_time=settling_time,
            t0=t0,
        )
        return label, solver.kernel.config_hash
    finally:
        solver.close()


class _OptimizeRunner:
    """Compile and time the candidate launches on solver copies.

    One compiled copy per candidate; one copy rebuilt per switch when
    the batch does not fit them all.
    """

    def __init__(
        self,
        parent: Any,
        inits: Any,
        params: Any,
        drivers: Optional[Dict[str, Any]],
        duration: float,
        settling_time: float,
        t0: float,
        verbose: bool,
    ) -> None:
        self._parent = parent
        self._grid = (inits, params)
        self._drivers = drivers
        self._given_duration = float(duration)
        self._given_settling = float(settling_time)
        self.duration = float(duration)
        self.settling = float(settling_time)
        self._t0 = float(t0)
        self._verbose = bool(verbose)
        self._sized = False
        self._candidates = ()
        self._twins = []
        self._live = True
        self._current = None
        self._inits = None
        self._params = None
        self.runs = 0
        self._fastest_ms = float("inf")
        self.achieved_waves = None

    def _emit(self, message: str) -> None:
        logger.debug(message)
        if self._verbose:
            print(message, flush=True)

    def close(self) -> None:
        """Release every solver copy."""
        for twin in self._twins:
            twin.close()
        self._twins = []

    def _make_twin(self, candidate: Dict[str, Any]) -> Any:
        """Return a parent copy carrying ``candidate``."""
        twin = self._parent.copy()
        if self._drivers is not None:
            twin._configure_drivers(self._drivers)
        twin.update(candidate, silent=True)
        integrator = twin.kernel.single_integrator
        if integrator.is_duration_dependent:
            # Pin the summary cadence at the given duration.
            twin.update(
                summarise_every=self._given_duration,
                sample_summaries_every=self._given_duration / 100.0,
            )
        return twin

    def build_twins(self, candidates: Sequence[Dict[str, Any]]) -> None:
        """Create one solver copy per candidate."""
        self._candidates = tuple(dict(c) for c in candidates)
        self._twins = [self._make_twin(c) for c in self._candidates]

    def set_batch(self, runs: int) -> None:
        """Stage ``runs`` grid columns on the device, cycling if short."""
        inits, params = self._grid
        columns = np_arange(int(runs)) % inits.shape[1]
        self._inits = cuda.to_device(np_take(inits, columns, axis=1))
        self._params = cuda.to_device(np_take(params, columns, axis=1))
        self.runs = int(runs)

    def _compile(self, twin: Any) -> None:
        """Compile ``twin`` for the staged batch."""
        twin.kernel.compile(
            self._inits, self._params, self.duration, self.settling, self._t0
        )

    def _is_cached(self, twin: Any) -> bool:
        """Whether the disk cache holds ``twin``'s kernel."""
        return twin.kernel.kernel_is_cached(
            self._inits, self._params, self.duration, self.settling, self._t0
        )

    def prewarm(self) -> None:
        """Compile the uncached candidates in workers; one miss waits."""
        parent = self._parent
        if not parent.cache_enabled:
            return
        missing = []
        for index, twin in enumerate(self._twins):
            label = _label(self._candidates[index])
            if self._is_cached(twin):
                self._emit(f"  {label}: cached")
            else:
                missing.append(index)
        if len(missing) < 2:
            return
        settings = parent.settings_dict
        system_bytes = pickle.dumps(parent.system)
        # Workers compile on one run; the batch size is not in the key.
        payloads = [
            (
                _label(self._candidates[index]),
                system_bytes,
                settings,
                self._candidates[index],
                1,
                self._drivers,
                self._given_duration,
                self._given_settling,
                self._t0,
                get_cache_root_override(),
            )
            for index in missing
        ]
        context = multiprocessing.get_context("spawn")
        with context.Pool(min(WORKERS, len(missing))) as pool:
            for label, config_hash in pool.imap_unordered(
                _compile_candidate, payloads
            ):
                self._emit(f"  {label}: worker compiled {config_hash[:12]}")

    def size_batch(self, waves: int) -> None:
        """Stage the batch filling ``waves`` at the first candidate."""
        twin = self._twins[0]
        self._compile(twin)
        self.set_batch(sized_batch_runs(twin.kernel, waves))
        self._sized = True
        self._emit(f"batch: {self.runs} runs fill {waves} waves")

    def _duration_floor(self) -> float:
        """Shortest duration the configured output cadence allows."""
        integrator = self._twins[0].kernel.single_integrator
        floor = 0.0
        if integrator.has_time_domain_outputs:
            save_every = integrator.save_every
            if save_every is not None and not integrator.save_last:
                floor = max(floor, float(save_every))
        if integrator.has_summary_outputs:
            floor = max(floor, float(integrator.summarise_every))
        return min(floor, self._given_duration)

    def _trial_durations(self) -> List[float]:
        """Ascending probe durations within the cadence and the given."""
        given = self._given_duration
        floor = self._duration_floor()
        trials = []
        for fraction in PROBE_FRACTIONS:
            trial = min(given, max(given * fraction, floor))
            if trial not in trials:
                trials.append(trial)
        return trials

    def use_shortest_duration(self) -> None:
        """Prepare the copies at the shortest probe duration."""
        self._set_duration(self._trial_durations()[0])

    def probe_duration(self, target_ms: float) -> None:
        """Ramp :data:`PROBE_FRACTIONS` of the duration to ``target_ms``."""
        given = self._given_duration
        floor = self._duration_floor()
        trials = self._trial_durations()
        twin = self._twins[0]
        measured = 0.0
        trial = given
        for trial in trials:
            self._set_duration(trial)
            measured = self._solve_ms(twin, None)
            self._emit(f"  probe: duration {trial:g} -> {measured:.3f} ms")
            if measured >= target_ms:
                break
        if measured >= target_ms:
            chosen = min(given, max(trial * target_ms / measured, floor))
        else:
            chosen = given
        self._set_duration(chosen)
        self._emit(f"duration: {chosen:g} per timed solve")

    def _set_duration(self, duration: float) -> None:
        """Set the timed duration, scaling the settling time with it."""
        given = self._given_duration
        scale = duration / given if given else 1.0
        self.duration = float(duration)
        self.settling = self._given_settling * scale

    def compile_twins(self) -> None:
        """Compile every copy; keep one if the batch chunks on any."""
        for twin in self._twins:
            self._compile(twin)
        chunked = any(
            twin.kernel.run_params.num_chunks > 1 for twin in self._twins
        )
        if chunked and len(self._twins) > 1:
            self._emit("memory: candidates take turns on one solver copy")
            for twin in self._twins[1:]:
                twin.close()
            del self._twins[1:]
            self._live = False
            self._current = 0

    def _twin(self, index: int) -> Any:
        """Return the compiled copy carrying candidate ``index``."""
        if self._live:
            return self._twins[index]
        if self._current != index:
            self._twins[0].close()
            self._twins[0] = self._make_twin(self._candidates[index])
            self._compile(self._twins[0])
            self._current = index
        return self._twins[0]

    def _solve_ms(
        self, twin: Any, blocksize: Optional[int]
    ) -> float:
        """Run one solve on ``twin`` and return its kernel milliseconds."""
        twin.solve(
            self._inits,
            self._params,
            duration=self.duration,
            settling_time=self.settling,
            t0=self._t0,
            blocksize=blocksize,
            on_device=True,
        )
        twin.kernel.synchronize()
        return float(
            sum(
                event.elapsed_time_ms()
                for event in twin.kernel._cuda_events
                if event.name.startswith("kernel_chunk")
            )
        )

    def time_candidates(
        self, blocksizes: Optional[Sequence[int]]
    ) -> List[LaunchResult]:
        """Time every launch of every candidate in ABBA order."""
        launches = []
        owners = {}
        for index, candidate in enumerate(self._candidates):
            kernel = self._twin(index).kernel
            for blocksize, resident in launch_candidates(kernel, blocksizes):
                launch = LaunchResult(
                    settings=dict(candidate),
                    blocksize=blocksize,
                    resident_blocks=resident,
                )
                launches.append(launch)
                owners[id(launch)] = index
        for round_index in range(ROUNDS):
            ordered = launches if round_index == 0 else launches[::-1]
            for launch in ordered:
                if launch.excluded:
                    continue
                twin = self._twin(owners[id(launch)])
                kernel = twin.kernel
                kernel.resident_blocks = launch.resident_blocks
                if round_index == 0:
                    blocksize, dynamic = kernel.launch_geometry(
                        launch.blocksize
                    )
                    launch.blocks_per_sm = active_blocks_per_multiprocessor(
                        kernel.kernel, blocksize, dynamic
                    )
                first = self._solve_ms(twin, launch.blocksize)
                launch.times_ms += (first,)
                solved = 1
                if self.achieved_waves is None:
                    self._probe_waves(twin, launch.blocksize)
                # A moderately slow first solve is repeated once.
                limit = EXCLUSION_RATIO * self._fastest_ms
                if round_index == 0 and first > limit:
                    if first <= CONFIRMATION_RATIO * self._fastest_ms:
                        launch.times_ms += (
                            self._solve_ms(twin, launch.blocksize),
                        )
                        solved += 1
                    if launch.best_ms > limit:
                        launch.excluded = True
                        self._emit(
                            f"  {launch.label}: excluded, "
                            f"{launch.best_ms:.3f} ms"
                        )
                        continue
                for _ in range(TIMED_SOLVES - solved):
                    launch.times_ms += (
                        self._solve_ms(twin, launch.blocksize),
                    )
                self._fastest_ms = min(self._fastest_ms, launch.best_ms)
                self._emit(f"  {launch.label}: {launch.best_ms:.3f} ms")
        return launches

    def _probe_waves(self, twin: Any, blocksize: int) -> None:
        """Record achieved occupancy waves; warn once when under two."""
        self.achieved_waves = _achieved_waves(twin, blocksize)
        if not self._sized and self.achieved_waves < 2.0:
            more = 2.0 / self.achieved_waves
            warn(
                f"The batch passed to optimize only fills "
                f"{self.achieved_waves:.2f} occupancy waves; the results "
                "might not represent the best timing for your system. "
                f"Try again with {more:.1f}x more runs in the batch and "
                "force=True to get the fastest full-GPU batch settings."
            )


def run_optimization(
    parent: Any,
    initial_values: Any,
    parameters: Any,
    drivers: Optional[Dict[str, Any]] = None,
    duration: float = 1.0,
    settling_time: float = 0.0,
    t0: float = 0.0,
    grid_type: str = "verbatim",
    apply: bool = True,
    verbose: bool = True,
    force: bool = False,
    mode: str = "auto",
    waves: int = 5,
    target_ms: float = 20.0,
) -> OptimizeResult:
    """Time the solver's candidate kernels on copies; apply the fastest.

    Parameters
    ----------
    parent
        The configured :class:`~cubie.batchsolving.solver.Solver`.
    initial_values
        Initial state values for each run, as accepted by
        :meth:`Solver.solve`.
    parameters
        Parameter values for each run, as accepted by
        :meth:`Solver.solve`.
    drivers
        Time-domain sampled driver values.
    duration
        Integration time of the caller's solves.
    settling_time
        Warm-up period before recording outputs.
    t0
        Initial integration time.
    grid_type
        Grid strategy when dict inputs trigger grid construction.
    apply
        Apply the best launch's settings to ``parent`` when ``True``.
    verbose
        Print per-launch progress lines.
    force
        Vary the settings given explicitly or applied earlier too.
    mode
        ``"auto"``: a ``waves``-wave batch from the grid, duration cut
        to ``target_ms`` per solve. ``"given"``: the whole grid at
        ``duration``.
    waves
        Occupancy waves the ``"auto"`` batch fills.
    target_ms
        Kernel milliseconds one ``"auto"`` solve aims for.

    Returns
    -------
    OptimizeResult
        Best launch, every measurement, and the applied settings.

    Raises
    ------
    ValueError
        Unknown ``mode``, ``waves`` under 1, or ``target_ms`` under 10.
    """
    if mode not in ("auto", "given"):
        raise ValueError(f"mode must be 'auto' or 'given', got {mode!r}")
    if int(waves) < 1 or waves != int(waves):
        raise ValueError(f"waves must be a positive integer, got {waves!r}")
    if not target_ms >= 10.0:
        raise ValueError(f"target_ms must be at least 10, got {target_ms!r}")
    inits, params = parent.build_grid(
        initial_values, parameters, grid_type=grid_type
    )
    candidates = parent.kernel.single_integrator.optimisation_candidates(
        force=force
    )
    blocksizes = (
        (parent.kernel.compile_settings.blocksize,)
        if parent.kernel.blocksize_given and not force
        else None
    )
    runner = _OptimizeRunner(
        parent,
        inits,
        params,
        drivers,
        duration,
        settling_time,
        t0,
        verbose,
    )
    # "silent" records the kernel events the timings read.
    verbosity = default_timelogger.verbosity
    default_timelogger.set_verbosity("silent")
    try:
        runner._emit(f"optimize: {len(candidates)} candidate kernels")
        runner.build_twins(candidates)
        runner.set_batch(inits.shape[1])
        if mode == "auto":
            runner.use_shortest_duration()
        runner.prewarm()
        if mode == "auto":
            runner.size_batch(waves)
            runner.probe_duration(target_ms)
        runner.compile_twins()
        launches = runner.time_candidates(blocksizes)
    finally:
        runner.close()
        default_timelogger.set_verbosity(verbosity)
    ranking = sorted(
        (launch for launch in launches if launch.timed),
        key=lambda launch: launch.best_ms,
    )
    best = ranking[0] if ranking else None
    applied_settings = {}
    if best is not None and apply:
        applied_settings = apply_launch(parent, best)
        runner._emit(f"applied: {best.label} -> parent solver")
    return OptimizeResult(
        launches=launches,
        best=best,
        applied_settings=applied_settings,
        runs=runner.runs,
        duration=runner.duration,
    )
