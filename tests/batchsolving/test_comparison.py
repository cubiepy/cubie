"""Tests for the shared candidate comparison runner."""

import gc

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


def test_runner_isolates_candidates_and_restores_the_solver(
    solver_mutable, simple_initial_values, simple_parameters
):
    """Candidates isolate, a rejection is recorded, close restores."""
    solver = solver_mutable
    inits, params = _device_grid(
        solver, simple_initial_values, simple_parameters
    )
    given = dict(solver.given.as_kwargs())
    loop = solver.kernel.single_integrator._loop
    state_location = loop.compile_settings.state_location
    observables_location = loop.compile_settings.observables_location
    assert not solver.given.is_given("observables_location")
    solver.kernel.resident_blocks = 3
    verbosity = default_timelogger.verbosity
    runner = ComparisonRunner(solver, inits, params, 0.1, 0.0, 0.0)
    with runner:
        assert default_timelogger.verbosity == "silent"
        runner.select(
            Candidate(
                "shared",
                {
                    "state_location": "shared",
                    "observables_location": "shared",
                    "dt": given["dt"] / 2,
                },
                resident_blocks=1,
            )
        )
        loop = solver.kernel.single_integrator._loop
        assert loop.compile_settings.state_location == "shared"
        assert loop.compile_settings.observables_location == "shared"
        assert solver.given.is_given("state_location")
        assert solver.kernel.resident_blocks == 1
        # An unset key goes back to its opening value.
        runner.select(Candidate("local", {"state_location": "local"}))
        loop = solver.kernel.single_integrator._loop
        assert loop.compile_settings.state_location == "local"
        assert (
            loop.compile_settings.observables_location
            == observables_location
        )
        assert solver.dt == given["dt"]
        assert solver.kernel.resident_blocks is None
        bogus = Candidate("bogus", {"state_location": "nowhere"})
        assert runner.compile([bogus]) == []
        assert "nowhere" in runner.rejection(bogus)
        timings = runner.time([bogus])
        assert timings[0].error == runner.rejection(bogus)
        assert timings[0].times_ms == ()
        runner.select(Candidate("shared", {"state_location": "shared"}))
        loop = solver.kernel.single_integrator._loop
        assert loop.compile_settings.state_location == "shared"
    assert default_timelogger.verbosity == verbosity
    assert dict(solver.given.as_kwargs()) == given
    assert not solver.given.is_given("state_location")
    assert not solver.given.is_given("observables_location")
    loop = solver.kernel.single_integrator._loop
    assert loop.compile_settings.state_location == state_location
    assert loop.compile_settings.observables_location == observables_location
    assert solver.dt == given["dt"]
    assert solver.kernel.resident_blocks == 3
    # A grid with no variables stages as None, the host-path default.
    runner = ComparisonRunner(solver, inits, params[:0], 0.1, 0.0, 0.0)
    runner.set_batch(2 * inits.shape[1])
    assert runner._params is None
    assert runner._inits.shape == (inits.shape[0], 2 * inits.shape[1])


def test_device_only_solves_share_the_host_solves_device_buffers(
    unchunked_solved_solver, system, precision, driver_settings
):
    """Device-only solves make no host buffer; host solves still match."""
    solver, first = unchunked_solved_solver
    inits = np.ones((system.sizes.states, 5), dtype=precision)
    params = np.ones((system.sizes.parameters, 5), dtype=precision)
    kwargs = dict(
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
        dt=0.01,
    )
    expected = np.array(first.time_domain_array)
    codes = np.array(first.status_codes)
    first_state = first.state
    first_state_copy = np.array(first_state)
    device_state = solver.kernel.device_state
    solver.solve(inits, params, on_device=True, **kwargs)
    solver.kernel.synchronize()
    outputs = solver.kernel.output_arrays
    for _, slot in outputs.host.iter_managed_arrays():
        assert slot.array is None
    assert solver.kernel.device_state is device_state
    second = solver.solve(inits, params, **kwargs)
    assert solver.kernel.device_state is device_state
    assert second.state is not first_state
    assert first.state is first_state
    np.testing.assert_array_equal(first.state, first_state_copy)
    np.testing.assert_array_equal(second.time_domain_array, expected)
    np.testing.assert_array_equal(second.status_codes, codes)


def test_device_only_solve_keeps_a_dead_result_loan_for_the_next_host_solve(
    unchunked_solved_solver, system, precision, driver_settings
):
    """A dead result's buffers return on the next host solve only."""
    solver, _ = unchunked_solved_solver
    rng = np.random.default_rng(99)
    inits = rng.uniform(0.5, 1.5, (system.sizes.states, 5)).astype(precision)
    params = rng.uniform(
        0.5, 1.5, (system.sizes.parameters, 5)
    ).astype(precision)
    kwargs = dict(
        drivers=driver_settings,
        duration=0.05,
        summarise_every=None,
        save_every=0.01,
        dt=0.01,
    )
    outputs = solver.kernel.output_arrays
    dropped = solver.solve(inits, params, **kwargs)
    loaned_state = dropped.state
    expected = np.array(dropped.time_domain_array)
    del dropped
    gc.collect()
    solver.solve(inits, params, on_device=True, **kwargs)
    solver.kernel.synchronize()
    for _, slot in outputs.host.iter_managed_arrays():
        assert slot.array is None
    second = solver.solve(inits, params, **kwargs)
    assert second.state is loaned_state
    np.testing.assert_array_equal(second.time_domain_array, expected)


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
