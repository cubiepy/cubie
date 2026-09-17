"""Tests for the per-solver unroll, placement and launch optimisation."""

import pytest

from cubie.backend.utils import DeviceHardware
from cubie.batchsolving.comparison import (
    ROUNDS,
    SOLVES_PER_ROUND,
    TIMED_WAVES_FLOOR,
    WARM_MS,
    Candidate,
    ComparisonRunner,
    settings_label,
    tail_safe_runs,
)
from cubie.batchsolving.optimize import (
    BUDGET_BLOCKSIZE,
    RESIDENCY_CUT_MIN_FRAME_BYTES,
    LaunchResult,
    _duration_floor,
    _ramp_start,
    apply_launch,
    default_launch,
    most_resident_runs,
    resident_blocks_within_l2,
)
from cubie.cuda_simsafe import cupy
from cubie.CUDAFactory import UnrollChoice
from cubie.time_logger import default_timelogger
from tests._utils import LARGE_FIRK

FULL = UnrollChoice.FULL
ROLLED = UnrollChoice.ROLLED

MIB = 1 << 20


def _candidates(solver, force=False):
    return solver.optimisation_candidates(force=force)


def _hardware(l2_cache_bytes, instruction_cache_bytes=128 * 1024):
    """An RTX 4070 SUPER-shaped device with the given L2 and icache."""
    return DeviceHardware(
        compute_capability=(8, 9),
        multiprocessor_count=56,
        l2_cache_bytes=l2_cache_bytes,
        shared_memory_per_multiprocessor=102400,
        reserved_shared_memory_per_block=1024,
        max_dynamic_shared_memory_per_block=101376,
        instruction_cache_bytes=instruction_cache_bytes,
        registers_per_multiprocessor=65536,
        max_threads_per_multiprocessor=1536,
        max_blocks_per_multiprocessor=24,
        warp_size=32,
    )


# 4 KiB frames: 14 MB of local memory per resident 64-thread block.
FRAME = 4096
# Blocks per SM per block size: 384, 512, 512 and 256 threads.
SHAPES = {32: 12, 64: 8, 128: 4, 256: 1}


def test_small_frames_are_never_cut():
    """Local memory under the floor keeps the driver's block count."""
    hardware = _hardware(4 * MIB)
    blocks = resident_blocks_within_l2(
        RESIDENCY_CUT_MIN_FRAME_BYTES - 1, 64, 8, hardware
    )
    assert blocks == 8


def test_residency_is_cut_to_the_count_that_fits_l2():
    """Two blocks may fill the whole L2; three would need two thirds."""
    assert resident_blocks_within_l2(FRAME, 64, 8, _hardware(48 * MIB)) == 2


def test_residency_cuts_to_one_block_within_two_thirds_of_l2():
    """One block is kept when its memory fits two thirds of L2."""
    assert resident_blocks_within_l2(FRAME, 64, 8, _hardware(24 * MIB)) == 1


def test_residency_stays_natural_when_no_count_fits():
    """A frame no block count fits keeps the driver's count."""
    assert resident_blocks_within_l2(FRAME, 64, 8, _hardware(4 * MIB)) == 8


def test_default_launch_takes_the_most_threads_smaller_block_on_a_tie():
    """Without a cut, the most resident threads win; 64 beats 128."""
    hardware = _hardware(48 * MIB)
    assert default_launch(SHAPES, 0, 1024, hardware) == (64, 8)


def test_default_launch_over_the_instruction_cache_takes_the_larger_block():
    """Over the instruction cache the largest block in the tie band wins."""
    hardware = _hardware(48 * MIB, instruction_cache_bytes=1024)
    assert default_launch(SHAPES, 0, 2048, hardware) == (128, 4)


def test_default_launch_cuts_every_block_size_to_the_budget():
    """The 64-thread-block budget caps every block size in whole blocks."""
    hardware = _hardware(48 * MIB)
    budget = BUDGET_BLOCKSIZE * resident_blocks_within_l2(
        FRAME, BUDGET_BLOCKSIZE, SHAPES[BUDGET_BLOCKSIZE], hardware
    )
    assert budget == 128
    assert default_launch(SHAPES, FRAME, 1024, hardware) == (32, 4)
    over_icache = _hardware(48 * MIB, instruction_cache_bytes=1024)
    assert default_launch(SHAPES, FRAME, 2048, over_icache) == (128, 1)


def test_default_launch_budget_excludes_block_sizes_that_do_not_divide_it():
    """A 192-thread budget admits 32- and 64-thread blocks only."""
    hardware = _hardware(72 * MIB)
    budget = BUDGET_BLOCKSIZE * resident_blocks_within_l2(
        FRAME, BUDGET_BLOCKSIZE, SHAPES[BUDGET_BLOCKSIZE], hardware
    )
    assert budget == 192
    assert default_launch(SHAPES, FRAME, 1024, hardware) == (32, 6)
    over_icache = _hardware(72 * MIB, instruction_cache_bytes=1024)
    assert default_launch(SHAPES, FRAME, 2048, over_icache) == (64, 3)


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "vern7", "unroll_other_small": None}],
    indirect=True,
)
def test_erk_candidates_cross_other_small_and_state(solver):
    """ERK varies ``other_small`` unrolling and ``state`` placement."""
    assert _candidates(solver) == (
        {"unroll_other_small": FULL, "state_location": "local"},
        {"unroll_other_small": FULL, "state_location": "shared"},
        {"unroll_other_small": ROLLED, "state_location": "local"},
        {"unroll_other_small": ROLLED, "state_location": "shared"},
        {
            "unroll_other_small": FULL,
            "state_location": "local",
            "stage_rhs_location": "shared",
        },
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{
        "algorithm": "radau_iia_3",
        "unroll_newton_exits": None,
        "unroll_other_small": None,
    }],
    indirect=True,
)
def test_firk_candidates_cross_newton_and_stage_increment(solver):
    """FIRK varies Newton unrolling and ``stage_increment`` placement."""
    assert _candidates(solver) == (
        {"unroll_newton_exits": FULL, "stage_increment_location": "local"},
        {"unroll_newton_exits": FULL, "stage_increment_location": "shared"},
        {"unroll_newton_exits": ROLLED, "stage_increment_location": "local"},
        {"unroll_newton_exits": ROLLED, "stage_increment_location": "shared"},
        {
            "unroll_newton_exits": ROLLED,
            "unroll_other_small": ROLLED,
            "stage_increment_location": "local",
        },
        {
            "unroll_newton_exits": ROLLED,
            "unroll_other_small": ROLLED,
            "stage_increment_location": "shared",
        },
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "algorithm": "kvaerno3",
            "linear_correction_type": "lu",
            "unroll_newton_exits": None,
            "unroll_other_small": None,
        }
    ],
    indirect=True,
)
def test_dirk_direct_candidates(solver):
    """A direct-solve DIRK crosses Newton unrolling and ``accumulator``."""
    assert _candidates(solver) == (
        {"unroll_newton_exits": FULL, "accumulator_location": "local"},
        {"unroll_newton_exits": FULL, "accumulator_location": "shared"},
        {"unroll_newton_exits": ROLLED, "accumulator_location": "local"},
        {"unroll_newton_exits": ROLLED, "accumulator_location": "shared"},
        {
            "unroll_newton_exits": ROLLED,
            "unroll_other_small": ROLLED,
            "accumulator_location": "local",
        },
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "algorithm": "kvaerno3",
            "linear_correction_type": "bicgstab",
            "unroll_newton_exits": None,
            "unroll_other_small": None,
        }
    ],
    indirect=True,
)
def test_dirk_iterative_candidates(solver):
    """An iterative-solve DIRK crosses Newton unrolling and ``accumulator``."""
    assert _candidates(solver) == (
        {"unroll_newton_exits": FULL, "accumulator_location": "local"},
        {"unroll_newton_exits": FULL, "accumulator_location": "shared"},
        {"unroll_newton_exits": ROLLED, "accumulator_location": "local"},
        {"unroll_newton_exits": ROLLED, "accumulator_location": "shared"},
        {
            "unroll_newton_exits": ROLLED,
            "unroll_other_small": ROLLED,
            "accumulator_location": "local",
        },
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "rosenbrock23", "unroll_other_small": None}],
    indirect=True,
)
def test_rosenbrock_candidates_vary_other_small(solver):
    """A Rosenbrock-W step varies ``other_small`` unrolling."""
    assert _candidates(solver) == (
        {"unroll_other_small": FULL},
        {"unroll_other_small": ROLLED},
    )


@pytest.mark.parametrize(
    "solver_settings_override", [{"algorithm": "euler"}], indirect=True
)
def test_other_steps_keep_current_settings(solver):
    """An Euler step has a single current candidate."""
    assert _candidates(solver) == ({},)


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "radau_iia_3", "unroll_newton_exits": (True, 1)}],
    indirect=True,
)
def test_user_fixed_axis_is_not_varied(solver):
    """An explicit setting removes its axis from the candidates."""
    assert _candidates(solver) == (
        {"stage_increment_location": "local"},
        {"stage_increment_location": "shared"},
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "radau_iia_3", "unroll_newton_exits": (True, 1)}],
    indirect=True,
)
def test_force_varies_user_fixed_axes(solver):
    """``force`` keeps every axis, the explicit ones included."""
    assert _candidates(solver, force=True) == (
        {"unroll_newton_exits": FULL, "stage_increment_location": "local"},
        {"unroll_newton_exits": FULL, "stage_increment_location": "shared"},
        {"unroll_newton_exits": ROLLED, "stage_increment_location": "local"},
        {"unroll_newton_exits": ROLLED, "stage_increment_location": "shared"},
        {
            "unroll_newton_exits": ROLLED,
            "unroll_other_small": ROLLED,
            "stage_increment_location": "local",
        },
        {
            "unroll_newton_exits": ROLLED,
            "unroll_other_small": ROLLED,
            "stage_increment_location": "shared",
        },
    )


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_FIRK], indirect=True
)
def test_derived_defaults_stay_free_axes_on_a_copy(solver):
    """Defaults the kernel derived are varied by the parent and its copy."""
    step = solver.kernel.single_integrator._algo_step.compile_settings
    assert step.stage_increment_location == "local"
    assert step.unroll.unroll_newton_exits == ROLLED.value
    # The shared fixture fixes unroll_other_small, so its arms fold in.
    expected = (
        {"unroll_newton_exits": FULL, "stage_increment_location": "local"},
        {"unroll_newton_exits": FULL, "stage_increment_location": "shared"},
        {"unroll_newton_exits": ROLLED, "stage_increment_location": "local"},
        {"unroll_newton_exits": ROLLED, "stage_increment_location": "shared"},
    )
    assert _candidates(solver) == expected
    twin = solver.copy()
    try:
        assert _candidates(twin) == expected
        assert twin.kernel.config_hash == solver.kernel.config_hash
        assert twin.kernel.driver_interpolator.config_hash == (
            solver.kernel.driver_interpolator.config_hash
        )
        assert twin.given.time_logging_level == default_timelogger.verbosity
    finally:
        twin.close()


def test_apply_launch_sets_settings_blocksize_and_residency(solver_mutable):
    """A launch's settings, block size and residency reach the solver."""
    launch = LaunchResult(
        settings={"unroll_other_small": ROLLED, "state_location": "shared"},
        blocksize=128,
        resident_blocks=2,
    )
    applied = apply_launch(solver_mutable, launch)
    assert applied == {
        "unroll_other_small": ROLLED,
        "state_location": "shared",
        "blocksize": 128,
    }
    kernel = solver_mutable.kernel
    assert kernel.compile_settings.blocksize == 128
    assert solver_mutable.given.is_given("blocksize")
    assert kernel.resident_blocks == 2
    loop = kernel.single_integrator._loop
    assert loop.compile_settings.state_location == "shared"
    assert loop.compile_settings.unroll.unroll_other_small == ROLLED.value


def test_block_size_change_clears_the_pinned_residency(solver_mutable):
    """A new block size drops the residency timed with the old one."""
    launch = LaunchResult(settings={}, blocksize=128, resident_blocks=2)
    apply_launch(solver_mutable, launch)
    kernel = solver_mutable.kernel
    assert kernel.resident_blocks == 2
    solver_mutable.update(dt=solver_mutable.dt * 0.5)
    assert kernel.resident_blocks == 2
    solver_mutable.update(blocksize=64)
    assert kernel.resident_blocks is None
    assert kernel.compile_settings.blocksize == 64


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "bogacki-shampine-32", "stage_rhs_location": "local"}],
    indirect=True,
)
def test_optimize_times_candidates_and_applies_the_fastest(
    solver_mutable, simple_initial_values, simple_parameters, driver_settings
):
    """Both placements are timed; the fastest launch is applied."""
    assert solver_mutable.optimisation_candidates() == (
        {"state_location": "local"},
        {"state_location": "shared"},
    )
    verbosity = default_timelogger.verbosity
    waves = 2
    result = solver_mutable.optimize(
        simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        grid_type="combinatorial",
        verbose=False,
        waves=waves,
        max_parallel=1,
    )
    assert default_timelogger.verbosity == verbosity
    timed = [launch for launch in result.launches if launch.timed]
    assert timed
    assert {launch.settings["state_location"] for launch in timed} == {
        "local",
        "shared",
    }
    assert min(launch.waves for launch in timed) >= waves
    outputs = solver_mutable.kernel.output_arrays
    for _, slot in outputs.host.iter_managed_arrays():
        assert slot.array is None
    for launch in result.launches:
        assert all(time_ms > 0.0 for time_ms in launch.times_ms)
        assert launch.blocks_per_sm >= 1
        assert launch.waves > 0.0
    assert result.best is min(timed, key=lambda launch: launch.best_ms)
    assert result.ranking[0] is result.best
    assert result.applied_settings == {
        **result.best.settings,
        "blocksize": result.best.blocksize,
    }
    kernel = solver_mutable.kernel
    assert kernel.compile_settings.blocksize == result.best.blocksize
    assert kernel.resident_blocks == result.best.resident_blocks
    loop = kernel.single_integrator._loop
    assert loop.compile_settings.state_location == (
        result.best.settings["state_location"]
    )
    for launch in result.launches:
        assert launch.error == ""
        assert len(launch.times_ms) == ROUNDS * SOLVES_PER_ROUND
        assert launch.failures == 0
        assert launch.runs == result.runs
    assert result.runs > 0
    # The duration ramps to the given; a batch still short of the
    # target there grows toward it.
    assert result.duration == pytest.approx(0.1)
    most = most_resident_runs(solver_mutable.kernel)
    assert result.runs >= waves * most
    assert result.runs > waves * most or result.best.best_ms >= 20.0
    assert "best" in result.summary()
    # Size the compiled candidates at two waves of the most concurrent.
    candidates = [
        Candidate(settings_label(settings), dict(settings))
        for settings in solver_mutable.optimisation_candidates()
    ]
    runner = _runner(solver_mutable, simple_initial_values, simple_parameters)
    with runner:
        runner.compile(candidates)
        assert runner.warm() >= WARM_MS
        runner.size_batch(candidates, TIMED_WAVES_FLOOR)
        concurrent = []
        for candidate in candidates:
            runner.select(candidate)
            concurrent.append(runner.concurrent_runs(None))
            _, filled = runner.geometry(None)
            assert filled >= TIMED_WAVES_FLOOR
        assert max(concurrent) > 0
        assert runner.runs == tail_safe_runs(
            concurrent,
            TIMED_WAVES_FLOOR * max(concurrent),
            TIMED_WAVES_FLOOR * max(concurrent),
        )
        assert runner.batch_cap() > runner.runs
        assert runner.solve_ms(None) > 0.0
        # Every solve on the sized batch reuses the same buffers.
        device_state = kernel.device_state
        assert runner.solve_ms(None) > 0.0
        assert kernel.device_state is device_state
        assert runner.staged_bytes == (
            kernel.input_arrays.device_initial_values.nbytes
            + kernel.input_arrays.device_parameters.nbytes
        )
        for _, slot in kernel.output_arrays.host.iter_managed_arrays():
            assert slot.array is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"waves": 0}, "waves"),
        ({"target_ms": 5.0}, "target_ms"),
        ({"target_ms": float("inf")}, "target_ms"),
    ],
)
def test_invalid_arguments_are_rejected(solver, kwargs, message):
    """Bad waves or target_ms raise before anything is built."""
    with pytest.raises(ValueError, match=message):
        solver.optimize({}, {}, max_parallel=1, **kwargs)


@pytest.mark.nocudasim
def test_kernel_is_cached_follows_the_cache_directory(
    solver_mutable, driver_settings, tmp_path
):
    """kernel_is_cached reads whichever directory the kernel points at."""
    kernel = solver_mutable.kernel
    kernel.kernel
    cache_root = kernel._disk_cache.cache_path.parent
    kernel.set_cache_dir(tmp_path / "fresh")
    assert not kernel.kernel_is_cached()
    kernel.set_cache_dir(cache_root)
    solver_mutable.compile(drivers=driver_settings, duration=0.1)
    assert kernel.kernel_is_cached()
    assert kernel._disk_cache.cache_path.parent == cache_root


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "bogacki-shampine-32", "stage_rhs_location": "local"}],
    indirect=True,
)
def test_compile_caches_the_optimize_candidates_without_a_solve(
    solver_mutable, driver_settings
):
    """Every candidate kernel is on disk; nothing launched."""
    solver = solver_mutable
    candidates = solver.optimisation_candidates()
    assert candidates == (
        {"state_location": "local"},
        {"state_location": "shared"},
    )
    solver.compile(
        drivers=driver_settings,
        duration=0.1,
        optimize_candidates=True,
        max_parallel=1,
    )
    for settings in candidates:
        solver.update(settings)
        assert solver.kernel.kernel_is_cached()
    assert solver.kernel._cuda_events == []


def _runner(solver, inits, params):
    """Return a runner on ``solver`` over a combinatorial grid."""
    grid_inits, grid_params = solver.build_grid(
        inits, params, grid_type="combinatorial"
    )
    return ComparisonRunner(
        solver, grid_inits, grid_params, 0.1, 0.0, 0.0, max_parallel=1
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"summarise_every": None, "sample_summaries_every": 0.03}],
    indirect=True,
)
def test_duration_floor_holds_the_final_summary_sample(
    solver_mutable, simple_initial_values, simple_parameters
):
    """The ramp starts on the cadence floor, keeping one sample inside,
    whatever the given duration."""
    runner = _runner(solver_mutable, simple_initial_values, simple_parameters)
    with runner:
        floor = _duration_floor(solver_mutable, 0.1)
        assert floor == pytest.approx(0.03)
        assert _ramp_start(solver_mutable, 0.1) == pytest.approx(floor)
        assert _ramp_start(solver_mutable, 1000.0) == pytest.approx(floor)


@pytest.mark.nocudasim
@pytest.mark.cupy
@pytest.mark.parametrize("auto_size", [True, False])
def test_optimize_takes_device_grids(
    solver_mutable, simple_initial_values, simple_parameters, auto_size
):
    """CuPy grids optimise with and without automatic sizing."""
    inits, params = solver_mutable.build_grid(
        simple_initial_values, simple_parameters, grid_type="combinatorial"
    )
    result = solver_mutable.optimize(
        cupy.asarray(inits),
        parameters=cupy.asarray(params),
        duration=0.1,
        verbose=False,
        apply=False,
        auto_size=auto_size,
        max_parallel=1,
    )
    assert result.best is not None
    assert result.runs > 0


def test_copy_registers_memory_like_its_parent(solver_mutable):
    """A copy joins the auto pool, or reserves what its parent reserved."""
    manager = solver_mutable.memory_manager
    assert manager.manual_proportion(solver_mutable.kernel) is None
    twin = solver_mutable.copy()
    try:
        manager = twin.kernel.memory_manager
        assert manager.manual_proportion(twin.kernel) is None
    finally:
        twin.close()
    solver_mutable.update(mem_proportion=0.2)
    try:
        assert solver_mutable.settings_dict["mem_proportion"] == 0.2
        twin = solver_mutable.copy()
        try:
            manager = twin.kernel.memory_manager
            assert manager.manual_proportion(twin.kernel) == 0.2
        finally:
            twin.close()
    finally:
        solver_mutable.update(mem_proportion=None)
