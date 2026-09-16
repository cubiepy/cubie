"""Tests for the shared candidate comparison runner."""

import numpy as np
import pytest

from cubie.batchsolving.comparison import (
    ROUNDS,
    SOLVES_PER_ROUND,
    SUCCESS_TIER_FRACTION,
    WORKER_STARTUP_SECONDS,
    Candidate,
    CandidateTiming,
    ComparisonRunner,
    rank_timings,
    settings_label,
)
from cubie.CUDAFactory import UnrollChoice
from cubie.time_logger import default_timelogger


def _timing(label, times, failures, runs=100):
    return CandidateTiming(
        candidate=Candidate(label),
        times_ms=tuple(times),
        failures=failures,
        runs=runs,
    )


def test_rank_timings_keeps_the_top_success_tier_ahead():
    """Candidates within the success tier rank on time, the rest after."""
    clean_slow = _timing("clean slow", (3.0, 3.1), 0)
    clean_fast = _timing("clean fast", (2.0, 2.2), 0)
    edge = _timing("edge", (1.0,), failures=5)
    failing = _timing("failing", (0.5,), failures=6)
    untimed = _timing("untimed", (), 0)
    ranking = rank_timings([clean_slow, failing, untimed, edge, clean_fast])
    assert ranking == [edge, clean_fast, clean_slow, failing]
    assert edge.success_rate >= SUCCESS_TIER_FRACTION * 1.0
    assert failing.success_rate < SUCCESS_TIER_FRACTION * 1.0


def test_rank_timings_tier_follows_the_best_success_rate():
    """A worse best rate widens the tier in proportion."""
    best = _timing("best", (2.0,), failures=20)
    within = _timing("within", (1.0,), failures=23)
    outside = _timing("outside", (0.5,), failures=25)
    assert rank_timings([outside, within, best]) == [within, best, outside]


def test_rank_timings_empty_without_timed_candidates():
    assert rank_timings([_timing("x", (), 0)]) == []


def test_pool_pays_only_when_serial_compiles_cost_more():
    """Spawning workers is chosen from the measured compile time."""
    pays = ComparisonRunner._pool_pays
    assert pays(None, 5) is False
    assert pays(80.0, 1) is False
    assert pays(1.0, 5) is False
    assert pays(80.0, 5) is True
    assert pays(WORKER_STARTUP_SECONDS, 2) is False
    assert pays(WORKER_STARTUP_SECONDS + 1.0, 2) is True


def test_settings_label_names_enums_and_placements():
    label = settings_label(
        {
            "unroll_other_small": UnrollChoice.ROLLED,
            "state_location": "shared",
            "algorithm": "tsit5",
        }
    )
    assert label == "other_small=rolled state=shared algorithm=tsit5"
    assert settings_label({}) == "current"


def _device_grid(solver, simple_initial_values, simple_parameters):
    return solver.build_grid(
        simple_initial_values, simple_parameters, grid_type="combinatorial"
    )


def test_runner_restores_given_settings_and_residency(
    solver_mutable, simple_initial_values, simple_parameters
):
    """Closing the runner puts back every setting a candidate touched."""
    solver = solver_mutable
    inits, params = _device_grid(
        solver, simple_initial_values, simple_parameters
    )
    given = dict(solver.given.as_kwargs())
    solver.kernel.resident_blocks = 3
    verbosity = default_timelogger.verbosity
    runner = ComparisonRunner(solver, inits, params, 0.1, 0.0, 0.0)
    with runner:
        assert default_timelogger.verbosity == "silent"
        runner.select(
            Candidate(
                "shared",
                {"state_location": "shared", "dt": given["dt"] / 2},
                resident_blocks=1,
            )
        )
        loop = solver.kernel.single_integrator._loop
        assert loop.compile_settings.state_location == "shared"
        assert solver.given.is_given("state_location")
        assert solver.kernel.resident_blocks == 1
    assert default_timelogger.verbosity == verbosity
    assert dict(solver.given.as_kwargs()) == given
    assert not solver.given.is_given("state_location")
    assert solver.kernel.resident_blocks == 3


def test_device_only_solve_creates_no_host_output_buffers(
    solver_mutable, batch_input_arrays, driver_settings
):
    """A device-only solve touches no host output buffer."""
    solver = solver_mutable
    inits, params = batch_input_arrays
    solver.solve(
        inits, params, drivers=driver_settings, duration=0.1, on_device=True
    )
    solver.kernel.synchronize()
    outputs = solver.kernel.output_arrays
    for _, slot in outputs.host.iter_managed_arrays():
        assert slot.array is None
    assert outputs.device_state is not None


def test_host_solve_after_device_solve_returns_the_same_results(
    solver_mutable, batch_input_arrays, driver_settings
):
    """Host buffers are built by the transfer that first needs them."""
    solver = solver_mutable
    inits, params = batch_input_arrays
    kwargs = dict(drivers=driver_settings, duration=0.1)
    reference = solver.solve(inits, params, **kwargs)
    expected = np.array(reference.time_domain_array)
    codes = np.array(reference.status_codes)
    device_state = solver.kernel.device_state
    solver.solve(inits, params, on_device=True, **kwargs)
    solver.kernel.synchronize()
    assert solver.kernel.device_state is device_state
    result = solver.solve(inits, params, **kwargs)
    np.testing.assert_array_equal(result.time_domain_array, expected)
    np.testing.assert_array_equal(result.status_codes, codes)


def test_live_result_keeps_its_buffers_without_device_reallocation(
    solver_mutable, batch_input_arrays, driver_settings
):
    """A held result keeps its host buffers; the device set is reused."""
    solver = solver_mutable
    inits, params = batch_input_arrays
    kwargs = dict(drivers=driver_settings, duration=0.1)
    first = solver.solve(inits, params, **kwargs)
    first_state = first.state
    device_state = solver.kernel.device_state
    second = solver.solve(inits, params, **kwargs)
    assert solver.kernel.device_state is device_state
    assert second.state is not first_state
    np.testing.assert_array_equal(first.state, first_state)
    np.testing.assert_array_equal(second.state, first.state)


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "vern7", "unroll_other_small": None}],
    indirect=True,
)
def test_runner_times_candidates_on_one_buffer_set(
    solver_mutable, simple_initial_values, simple_parameters
):
    """Every candidate solves the same staged batch in the same buffers."""
    solver = solver_mutable
    inits, params = _device_grid(
        solver, simple_initial_values, simple_parameters
    )
    candidates = [
        Candidate(settings_label(settings), dict(settings))
        for settings in solver.optimisation_candidates()
    ]
    runner = ComparisonRunner(solver, inits, params, 0.1, 0.0, 0.0)
    with runner:
        runner.compile(candidates)
        runner.set_batch(4 * inits.shape[1])
        assert runner.runs == 4 * inits.shape[1]
        first = runner.solve_ms(None)
        device_state = solver.kernel.device_state
        timings = runner.time(candidates)
        assert solver.kernel.device_state is device_state
        for _, slot in solver.kernel.output_arrays.host.iter_managed_arrays():
            assert slot.array is None
    assert first > 0.0
    assert len(timings) == len(candidates)
    for timing in timings:
        assert timing.error == ""
        assert len(timing.times_ms) == ROUNDS * SOLVES_PER_ROUND
        assert all(time_ms > 0.0 for time_ms in timing.times_ms)
        assert timing.runs == runner.runs
        assert timing.failures == 0
        assert timing.blocks_per_sm >= 1
        assert timing.waves > 0.0
