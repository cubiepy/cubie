"""Backend of :meth:`cubie.Solver.optimize`: time unroll, placement and
launch candidates on a solver copy and apply the fastest.

Published Objects
-----------------
:class:`LaunchResult`
    One candidate's timings at one launch.
:class:`OptimizeResult`
    Every launch, the best one, and the applied settings.
:func:`resident_blocks_within_l2`
    Blocks per SM whose local memory fits in L2.
:func:`default_launch`
    Block size and blocks per SM for a launch with no block size given.
:func:`launch_candidates`
    Launches a kernel can time.
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
from numpy import zeros as np_zeros

from cubie.backend.utils import (
    DeviceHardware,
    SASS_INSTRUCTION_BYTES,
    active_blocks_per_multiprocessor,
    device_hardware,
    kernel_resources,
)
from cubie.batchsolving.calibration import _achieved_waves
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.CUDAFactory import UnrollChoice
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
"""Launches whose first solve exceeds this multiple of the fastest drop."""

LOCAL_LAUNCH_BLOCKSIZES = (32, 64, 128, 256)
"""Block sizes timed for local-only kernels."""

SHARED_LAUNCH_BLOCKSIZES = (32, 64, 128, 256)
"""Block sizes timed for shared-memory kernels."""

RESIDENT_FOOTPRINT_L2_FRACTION = 2.0 / 3.0
"""Fraction of L2 three or more resident blocks' local memory may fill.

Empirical: fitted to the placement landscape, where one block beat two
only when two blocks overflowed the whole L2.
"""

RESIDENCY_CUT_MIN_FRAME_BYTES = 2048
"""Local memory per thread below which residency is never cut.

Empirical: on the RTX 4070 SUPER and RTX 2060 SUPER landscapes a cut
paid only from 2 KiB up; smaller frames ran 10 to 50% slower when cut.
"""

BUDGET_BLOCKSIZE = 64
"""Block size the L2 rule counts resident threads at; the shape follows.

Protocol choice: the block size the residency landscapes were recorded
at, so the cut points are validated at this granularity.
"""

LAUNCH_OCCUPANCY_TIE = 0.9
"""Share of the most resident threads a larger block must keep to win.

Empirical: over the instruction cache the largest block won 52/13 of
259 equal-residency comparisons on the RTX 2060 SUPER and 30/13 of 179
on the RTX 4070 SUPER; the band width is fitted to those records.
"""


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
        Whether the first solve exceeded the exclusion ratio.
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
    """

    launches: List[LaunchResult]
    best: Optional[LaunchResult]
    applied_settings: Dict[str, Any]

    @property
    def ranking(self) -> List[LaunchResult]:
        """Every timed launch, fastest first."""
        timed = [launch for launch in self.launches if launch.timed]
        return sorted(timed, key=lambda launch: launch.best_ms)

    def summary(self) -> str:
        """Return a formatted table of every launch measurement."""
        header = f"{'launch':<48}{'blk/SM':>8}{'best ms':>10}  note"
        lines = [header, "-" * len(header)]
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


def resident_blocks_within_l2(
    frame: int, blocksize: int, natural: int, hardware: DeviceHardware
) -> int:
    """Return the most blocks per SM whose local memory fits in L2.

    Parameters
    ----------
    frame
        Local memory per thread in bytes.
    blocksize
        Threads per block.
    natural
        Blocks per SM the driver fits at this launch shape.
    hardware
        The device's L2 size and SM count.

    Returns
    -------
    int
        The cut count; ``natural`` for a small frame or when none fits.
    """
    if frame < RESIDENCY_CUT_MIN_FRAME_BYTES:
        return natural
    l2_bytes = hardware.l2_cache_bytes
    footprint = frame * blocksize * hardware.multiprocessor_count

    def budget(count):
        # Two blocks may fill the whole L2; more share two-thirds.
        if count == 2:
            return l2_bytes
        return RESIDENT_FOOTPRINT_L2_FRACTION * l2_bytes

    blocks = natural
    while blocks > 1 and footprint * blocks > budget(blocks):
        blocks -= 1
    if footprint * blocks > budget(blocks):
        return natural
    return blocks


def default_launch(
    shapes: Dict[int, int],
    frame: int,
    sass_bytes: int,
    hardware: DeviceHardware,
) -> Tuple[int, int]:
    """Return the ``(blocksize, resident_blocks)`` of an untimed launch.

    Parameters
    ----------
    shapes
        Blocks per SM the driver fits at each launchable block size.
    frame
        Local memory per thread in bytes.
    sass_bytes
        Machine-code size of the kernel.
    hardware
        The device's L2 size, SM count and instruction cache.

    Returns
    -------
    tuple[int, int]
        The block size and blocks per SM to launch.
    """
    # Resident-thread budget from the L2 rule; None means no cut.
    budget = None
    if BUDGET_BLOCKSIZE in shapes:
        blocks = resident_blocks_within_l2(
            frame, BUDGET_BLOCKSIZE, shapes[BUDGET_BLOCKSIZE], hardware
        )
        if blocks < shapes[BUDGET_BLOCKSIZE]:
            budget = BUDGET_BLOCKSIZE * blocks
    # Block sizes within the budget or cut to it in whole blocks.
    launches = []
    for blocksize, blocks in shapes.items():
        if budget is None or budget >= blocksize * blocks:
            launches.append((blocksize, blocks))
        elif budget % blocksize == 0:
            launches.append((blocksize, budget // blocksize))
    most = max(size * blocks for size, blocks in launches)
    # Over the instruction cache: largest block within the tie band.
    if sass_bytes > hardware.instruction_cache_bytes:
        fitting = [
            launch for launch in launches
            if launch[0] * launch[1] >= LAUNCH_OCCUPANCY_TIE * most
        ]
        return max(fitting, key=lambda launch: launch[0])
    # Otherwise the most threads, the smaller block on a tie.
    return min(
        launches, key=lambda launch: (-launch[0] * launch[1], launch[0])
    )


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
    shapes = kernel.launchable_shapes(blocksizes)
    frame = kernel_resources(kernel.kernel).local_bytes_per_thread
    cells = []
    for blocksize, (_, natural) in shapes.items():
        cells.append((blocksize, None))
        if frame == 0:
            continue
        for cut in (1, 2):
            if natural - cut >= 1:
                cells.append((blocksize, natural - cut))
    return tuple(cells)


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
    """Compile and time the candidate launches on a solver copy."""

    def __init__(
        self,
        twin: Any,
        inits: Any,
        params: Any,
        drivers: Optional[Dict[str, Any]],
        duration: float,
        settling_time: float,
        t0: float,
        verbose: bool,
    ) -> None:
        self._twin = twin
        self._inits = cuda.to_device(inits)
        self._params = cuda.to_device(params)
        self._n_runs = int(inits.shape[1])
        self._drivers = drivers
        self._duration = float(duration)
        self._settling = float(settling_time)
        self._t0 = float(t0)
        self._verbose = bool(verbose)
        self._fastest_ms = float("inf")
        self.achieved_waves = None

    def _emit(self, message: str) -> None:
        logger.debug(message)
        if self._verbose:
            print(message, flush=True)

    def compile_candidates(
        self, parent: Any, candidates: Sequence[Dict[str, Any]]
    ) -> None:
        """Pre-warm the kernel cache with every candidate in workers."""
        if not parent.cache_enabled:
            return
        # Pickled into spawned workers; the manager holds CUDA state.
        settings = {
            key: value
            for key, value in parent.settings_dict.items()
            if key != "memory_manager"
        }
        system_bytes = pickle.dumps(parent.system)
        payloads = [
            (
                _label(candidate),
                system_bytes,
                settings,
                candidate,
                self._n_runs,
                self._drivers,
                self._duration,
                self._settling,
                self._t0,
                get_cache_root_override(),
            )
            for candidate in candidates
        ]
        context = multiprocessing.get_context("spawn")
        with context.Pool(min(WORKERS, len(candidates))) as pool:
            for label, config_hash in pool.imap_unordered(
                _compile_candidate, payloads
            ):
                self._emit(f"  {label}: worker compiled {config_hash[:12]}")

    def _select(self, candidate: Dict[str, Any]) -> None:
        """Apply a candidate to the twin and compile it."""
        self._twin.update(candidate, silent=True)
        self._twin.compile(
            self._inits,
            self._params,
            duration=self._duration,
            settling_time=self._settling,
            t0=self._t0,
        )

    def _solve_ms(self, launch: LaunchResult) -> float:
        """Run one solve at ``launch`` and return its kernel milliseconds."""
        twin = self._twin
        twin.solve(
            self._inits,
            self._params,
            duration=self._duration,
            settling_time=self._settling,
            t0=self._t0,
            blocksize=launch.blocksize,
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
        self,
        candidates: Sequence[Dict[str, Any]],
        blocksizes: Optional[Sequence[int]],
    ) -> List[LaunchResult]:
        """Time every launch of every candidate in ABBA order."""
        verbosity = default_timelogger.verbosity
        default_timelogger.set_verbosity("silent")
        try:
            return self._time_candidates(candidates, blocksizes)
        finally:
            default_timelogger.set_verbosity(verbosity)

    def _time_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        blocksizes: Optional[Sequence[int]],
    ) -> List[LaunchResult]:
        twin = self._twin
        launches = []
        for candidate in candidates:
            self._select(candidate)
            for blocksize, resident in launch_candidates(
                twin.kernel, blocksizes
            ):
                launches.append(
                    LaunchResult(
                        settings=dict(candidate),
                        blocksize=blocksize,
                        resident_blocks=resident,
                    )
                )
        current = None
        for round_index in range(ROUNDS):
            ordered = launches if round_index == 0 else launches[::-1]
            for launch in ordered:
                if launch.excluded:
                    continue
                if launch.settings != current:
                    self._select(launch.settings)
                    current = launch.settings
                twin.kernel.resident_blocks = launch.resident_blocks
                if round_index == 0:
                    blocksize, dynamic = twin.kernel.launch_geometry(
                        launch.blocksize
                    )
                    launch.blocks_per_sm = active_blocks_per_multiprocessor(
                        twin.kernel.kernel, blocksize, dynamic
                    )
                first = self._solve_ms(launch)
                launch.times_ms += (first,)
                if self.achieved_waves is None:
                    self._probe_waves(launch)
                # A first solve far behind the fastest ends the launch.
                if len(launch.times_ms) == 1 and (
                    first > EXCLUSION_RATIO * self._fastest_ms
                ):
                    launch.excluded = True
                    self._emit(f"  {launch.label}: excluded, {first:.3f} ms")
                    continue
                for _ in range(TIMED_SOLVES - 1):
                    launch.times_ms += (self._solve_ms(launch),)
                self._fastest_ms = min(self._fastest_ms, launch.best_ms)
                self._emit(f"  {launch.label}: {launch.best_ms:.3f} ms")
        return launches

    def _probe_waves(self, launch: LaunchResult) -> None:
        """Record achieved occupancy waves; warn once when under two."""
        self.achieved_waves = _achieved_waves(self._twin, launch.blocksize)
        if self.achieved_waves < 2.0:
            more = 2.0 / self.achieved_waves
            warn(
                f"The batch passed to optimize only fills "
                f"{self.achieved_waves:.2f} occupancy waves; the results "
                "might not represent the best timing for your system. "
                f"Try again with {more:.1f}x more runs in the batch and "
                "force=True to get the fastest full-GPU batch settings."
            )


def performance_defaults(given: Any, step: Any, system: Any) -> Dict[str, Any]:
    """Return the unroll and placement settings for a built solver.

    Parameters
    ----------
    given
        The given settings; a given setting is never overridden.
    step
        The built algorithm step.
    system
        The system being solved.

    Returns
    -------
    dict
        The settings ``auto_performance`` applies; empty when it is off.
    """
    if given.auto_performance is False:
        return {}
    hardware = device_hardware()
    defaults = dict(step.performance_defaults(hardware))
    # A Newton loop that overflows the instruction cache stays rolled.
    if step.is_implicit and step.newton_solves_per_step > 0:
        unrolled = system.operation_count + step.step_operation_count
        capacity = hardware.instruction_cache_bytes // SASS_INSTRUCTION_BYTES
        defaults["unroll_newton_exits"] = (
            UnrollChoice.ROLLED if unrolled > capacity else UnrollChoice.FULL
        )
    return {
        key: value
        for key, value in defaults.items()
        if not given.is_given(key)
    }


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
) -> OptimizeResult:
    """Time the solver's candidate kernels on a copy; apply the fastest.

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
        Integration time each timed solve runs.
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

    Returns
    -------
    OptimizeResult
        Best launch, every measurement, and the applied settings.

    Raises
    ------
    ValueError
        If the system declares drivers but none are supplied.
    """
    if parent.system.sizes.drivers > 0 and drivers is None:
        raise ValueError(
            "The system declares drivers; optimize requires the driver "
            "samples that solves will use."
        )
    inits, params = parent.build_grid(
        initial_values, parameters, grid_type=grid_type
    )
    candidates = parent.optimisation_candidates(force=force)
    blocksizes = (
        (parent.kernel.compile_settings.blocksize,)
        if parent.given.is_given("blocksize") and not force
        else None
    )
    twin = parent.copy()
    try:
        if drivers is not None:
            twin._configure_drivers(drivers)
        runner = _OptimizeRunner(
            twin, inits, params, drivers, duration, settling_time, t0, verbose
        )
        runner._emit(f"optimize: {len(candidates)} candidate kernels")
        runner.compile_candidates(parent, candidates)
        launches = runner.time_candidates(candidates, blocksizes)
    finally:
        twin.close()
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
        launches=launches, best=best, applied_settings=applied_settings
    )
