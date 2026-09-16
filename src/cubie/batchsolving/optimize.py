"""Backend of :meth:`cubie.Solver.optimize`: time launches, apply the best.

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
:func:`most_resident_runs`
    Most runs resident at once over a kernel's launches.
:func:`apply_launch`
    Apply one launch to a solver.
:func:`run_optimization`
    Time a solver's candidates.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from attrs import define

from cubie.backend.utils import (
    DeviceHardware,
    SASS_INSTRUCTION_BYTES,
    device_hardware,
    kernel_resources,
)
from cubie.batchsolving.comparison import (
    TIMED_WAVES_FLOOR,
    Candidate,
    CandidateTiming,
    ComparisonRunner,
    rank_timings,
    settings_label,
    validate_sizing,
    warn_low_waves,
)
from cubie.CUDAFactory import UnrollChoice

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


@define
class LaunchResult:
    """Timings of one candidate at one launch.

    Parameters
    ----------
    settings
        The candidate's unroll and placement settings.
    blocksize
        Threads per block; ``None`` for a rejected candidate.
    resident_blocks
        Blocks per SM held resident; ``None`` = the default residency.
    blocks_per_sm
        Resident blocks per SM the driver reported.
    times_ms
        Timed solve times in milliseconds.
    failures
        Runs with a nonzero status code.
    runs
        Trajectories each solve integrated.
    waves
        Occupancy waves the batch filled at the launch.
    error
        Why the launch could not be timed, empty when it was.
    """

    settings: Dict[str, Any]
    blocksize: Optional[int]
    resident_blocks: Optional[int]
    blocks_per_sm: int = 0
    times_ms: Tuple[float, ...] = ()
    failures: int = 0
    runs: int = 0
    waves: float = 0.0
    error: str = ""

    @classmethod
    def from_timing(cls, timing: CandidateTiming) -> "LaunchResult":
        """Build the launch record of one candidate timing."""
        candidate = timing.candidate
        return cls(
            settings=dict(candidate.settings),
            blocksize=candidate.blocksize,
            resident_blocks=candidate.resident_blocks,
            blocks_per_sm=timing.blocks_per_sm,
            times_ms=timing.times_ms,
            failures=timing.failures,
            runs=timing.runs,
            waves=timing.waves,
            error=timing.error,
        )

    @property
    def timed(self) -> bool:
        """Whether the launch has a time."""
        return bool(self.times_ms)

    @property
    def best_ms(self) -> float:
        """Lowest solve time, ``inf`` when never solved."""
        return min(self.times_ms) if self.times_ms else float("inf")

    @property
    def success_rate(self) -> float:
        """Share of the batch that integrated without a status flag."""
        if self.runs == 0:
            return 0.0
        return (self.runs - self.failures) / self.runs

    @property
    def label(self) -> str:
        """Candidate settings and launch as one line."""
        return _launch_label(
            self.settings, self.blocksize, self.resident_blocks
        )


def _launch_label(
    settings: Dict[str, Any],
    blocksize: Optional[int],
    resident_blocks: Optional[int],
) -> str:
    """Return the one-line name of a launch."""
    label = settings_label(settings)
    if blocksize is None:
        return label
    resident = "" if resident_blocks is None else f" x{resident_blocks}"
    return f"{label} @bs{blocksize}{resident}"


@define
class OptimizeResult:
    """Complete optimisation report.

    Parameters
    ----------
    launches
        Every launch measured, in run order.
    best
        Fastest launch of the top success tier, or ``None`` when
        nothing was timed.
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
        """Every timed launch: the top success tier by time, then the rest."""
        return rank_timings(self.launches)

    def summary(self) -> str:
        """Return a formatted table of every launch measurement."""
        header = (
            f"{'launch':<48}{'blk/SM':>8}{'best ms':>10}{'failed':>8}  note"
        )
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
            if launch.error:
                best = ""
                note = f"failed: {launch.error}"
            else:
                best = f"{launch.best_ms:.3f}"
                rank = ranks.get(id(launch))
                note = "best" if rank == 1 else f"rank {rank}"
            lines.append(
                f"{launch.label:<48}{launch.blocks_per_sm:>8}"
                f"{best:>10}{launch.failures:>8}  {note}"
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


def launch_blocksizes(
    kernel: Any, blocksizes: Optional[Sequence[int]] = None
) -> Sequence[int]:
    """Return ``blocksizes``, or the measured set for ``kernel``'s frame."""
    if blocksizes is not None:
        return blocksizes
    if kernel.shared_memory_bytes > 0:
        return SHARED_LAUNCH_BLOCKSIZES
    return LOCAL_LAUNCH_BLOCKSIZES


def launch_candidates(
    kernel: Any,
    blocksizes: Optional[Sequence[int]] = None,
    runs: Optional[int] = None,
) -> Tuple[Tuple[int, Optional[int]], ...]:
    """Return the ``(blocksize, resident_blocks)`` launches worth timing.

    Parameters
    ----------
    kernel
        A compiled :class:`~cubie.batchsolving.BatchSolverKernel`.
    blocksizes
        Block sizes to consider; ``None`` picks the measured set for
        shared-memory or local-only kernels.
    runs
        Runs in the timed launches; ``None`` sizes a full block.

    Returns
    -------
    tuple
        Launchable block sizes at the default residency, plus one and
        two blocks under natural occupancy for local frames.
    """
    blocksizes = launch_blocksizes(kernel, blocksizes)
    shapes = kernel.launchable_shapes(blocksizes, runs=runs)
    frame = kernel_resources(
        kernel.kernel, kernel.signature
    ).local_bytes_per_thread
    cells = []
    for blocksize, (_, natural) in shapes.items():
        cells.append((blocksize, None))
        if frame == 0:
            continue
        for cut in (1, 2):
            if natural - cut >= 1:
                cells.append((blocksize, natural - cut))
    return tuple(cells)


def most_resident_runs(
    kernel: Any, blocksizes: Optional[Sequence[int]] = None
) -> int:
    """Return the most runs resident at once over ``kernel``'s launches."""
    blocksizes = launch_blocksizes(kernel, blocksizes)
    shapes = kernel.launchable_shapes(blocksizes)
    multiprocessors = device_hardware().multiprocessor_count
    threads_per_loop = kernel.threads_per_loop
    return max(
        (
            blocks * multiprocessors * (blocksize // threads_per_loop)
            for blocksize, (_, blocks) in shapes.items()
        ),
        default=0,
    )


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


def _fits_sample_interval(solver: Any, duration: float) -> bool:
    """Whether ``solver`` accepts ``duration`` under its sample interval."""
    try:
        solver.update(duration=duration, silent=True)
    except ValueError as error:
        if "sample_summaries_every" in str(error):
            return False
        raise
    return True


def _duration_floor(solver: Any, given: float) -> float:
    """Shortest duration the configured output cadence allows."""
    effective = solver.effective
    floor = 0.0
    if effective.save_regularly:
        floor = max(floor, float(effective.save_every))
    if effective.summarise_regularly:
        floor = max(floor, float(effective.summarise_every))
    if effective.summarise_last:
        floor = max(floor, float(effective.sample_summaries_every))
    return min(floor, given)


def _set_duration(
    runner: ComparisonRunner, duration: float, given: float, settling: float
) -> None:
    """Set the timed duration, scaling the settling time with it."""
    scale = duration / given if given else 1.0
    runner.duration = float(duration)
    runner.settling = settling * scale


def _ramp_start(solver: Any, given: float) -> float:
    """Return the shortest duration to time: at least one save or
    summary sample and at least one step, or ``given`` when the
    sample interval rejects anything shorter."""
    trial = max(_duration_floor(solver, given), float(solver.effective.dt))
    trial = min(trial, given)
    if trial < given and not _fits_sample_interval(solver, trial):
        return given
    return trial


def _ramp_duration(
    runner: ComparisonRunner,
    solver: Any,
    start: float,
    given: float,
    settling: float,
    target_ms: float,
) -> float:
    """Double the timed duration from ``start`` while the kernel time
    is under ``target_ms`` and the duration under ``given``; return
    the last kernel time."""
    trial = start
    _set_duration(runner, trial, given, settling)
    measured = runner.solve_ms(None)
    runner.emit(f"  probe: duration {trial:g} -> {measured:.3f} ms")
    while measured < target_ms and trial < given:
        longer = min(trial * 2.0, given)
        if longer < given and not _fits_sample_interval(solver, longer):
            longer = given
        _set_duration(runner, longer, given, settling)
        measured = runner.solve_ms(None)
        runner.emit(f"  probe: duration {longer:g} -> {measured:.3f} ms")
        trial = longer
    runner.emit(f"duration: {trial:g} per timed solve")
    return measured


def _size_batch(
    runner: ComparisonRunner,
    solver: Any,
    launches: Sequence[Candidate],
    waves: int,
    given: float,
    settling: float,
    target_ms: float,
) -> None:
    """Stage ``auto_size``'s batch at ``waves``, ramp the duration toward
    ``target_ms`` never past ``given``, then correct the batch once."""
    runner.size_batch(launches, waves)
    if not launches:
        return
    runner.select(launches[0])
    start = _ramp_start(solver, given)
    measured = _ramp_duration(
        runner, solver, start, given, settling, target_ms
    )
    runner.fit_batch(
        measured,
        target_ms,
        grow=runner.duration >= given,
        shrink=runner.duration <= start,
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
    auto_size: bool = True,
    waves: int = 5,
    target_ms: float = 20.0,
) -> OptimizeResult:
    """Time the solver's candidate kernels on itself; apply the fastest.

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
    auto_size
        ``True`` optimizes at an automatically selected batch size and
        duration to reduce runtime; ``False`` optimizes at your given
        batch size and duration.
    waves
        Waves of the launch with the most concurrent runs the
        ``auto_size`` batch starts at; the batch then moves to hit
        ``target_ms``.
    target_ms
        Kernel milliseconds per timed solve ``auto_size`` aims for by
        raising the duration, never past yours, then the batch.

    Returns
    -------
    OptimizeResult
        Best launch, every measurement, and the applied settings.

    Raises
    ------
    ValueError
        ``waves`` under 1, or ``target_ms`` under 10 or not finite.
    """
    validate_sizing(waves, target_ms)
    inits, params = parent.build_grid(
        initial_values, parameters, grid_type=grid_type
    )
    if drivers is not None:
        parent._configure_drivers(drivers)
    candidates = [
        Candidate(settings_label(settings), dict(settings))
        for settings in parent.optimisation_candidates(force=force)
    ]
    blocksizes = (
        (parent.kernel.compile_settings.blocksize,)
        if parent.given.is_given("blocksize") and not force
        else None
    )
    runner = ComparisonRunner(
        parent, inits, params, duration, settling_time, t0, verbose
    )
    kernel = parent.kernel
    with runner:
        runner.emit(f"optimize: {len(candidates)} candidate kernels")
        accepted = runner.compile(candidates)
        launches = []
        for candidate in candidates:
            # A rejected candidate reports once, with no launch.
            if runner.rejection(candidate):
                launches.append(
                    Candidate(candidate.label, dict(candidate.settings))
                )
                continue
            runner.select(candidate)
            for blocksize, resident in launch_candidates(
                kernel, blocksizes
            ):
                launches.append(
                    Candidate(
                        _launch_label(
                            candidate.settings, blocksize, resident
                        ),
                        dict(candidate.settings),
                        blocksize,
                        resident,
                    )
                )
        runner.warm()
        if auto_size and accepted:
            _size_batch(
                runner,
                parent,
                [launch for launch in launches if launch.blocksize],
                int(waves),
                float(duration),
                float(settling_time),
                target_ms,
            )
        else:
            runner.set_batch()
        timings = runner.time(launches)
        runs = runner.runs
        timed_duration = runner.duration
    results = [LaunchResult.from_timing(timing) for timing in timings]
    achieved = [launch.waves for launch in results if launch.timed]
    if not auto_size and achieved and min(achieved) < TIMED_WAVES_FLOOR:
        warn_low_waves(
            min(achieved),
            " and force=True to get the fastest full-GPU batch settings",
        )
    ranking = rank_timings(results)
    best = ranking[0] if ranking else None
    applied_settings = {}
    if best is not None and apply:
        applied_settings = apply_launch(parent, best)
        runner.emit(f"applied: {best.label} -> parent solver")
    return OptimizeResult(
        launches=results,
        best=best,
        applied_settings=applied_settings,
        runs=runs,
        duration=timed_duration,
    )
