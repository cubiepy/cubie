"""Tests for the size and hardware defaults ``auto_performance`` applies."""

import pytest

from cubie.cuda_simsafe import (
    SASS_INSTRUCTION_BYTES,
    UnrollChoice,
    device_hardware,
)
from cubie.integrators.algorithms.generic_firk import (
    SHARED_STAGE_INCREMENT_MIN_STATES,
)
from tests._utils import LARGE_STATE_ONLY

# The shared fixture pins the exit flags and the Newton cap; unset them.
FREE_EXITS = {
    "unroll_newton_exits": None,
    "unroll_krylov_exits": None,
    "newton_max_iters": None,
}
LARGE_KVAERNO3 = {**LARGE_STATE_ONLY, "algorithm": "kvaerno3", **FREE_EXITS}
LARGE_RADAU3 = {**LARGE_STATE_ONLY, "algorithm": "radau_iia_3", **FREE_EXITS}


def _built_step(solver):
    """Return the algorithm step after the integrator has built."""
    run = solver.kernel.single_integrator
    run.device_function
    return run._algo_step


def _unrolled_instructions(solver, step):
    """Return the step's instruction count with the Newton loop unrolled."""
    return (
        solver.system.operation_count
        + step.per_step_operation_count
        + step.newton_max_iters
        * step.newton_solves_per_step
        * step.newton_body_operation_count
    )


def _capacity():
    return device_hardware().instruction_cache_bytes // SASS_INSTRUCTION_BYTES


@pytest.mark.parametrize(
    "solver_settings_override, solves",
    [
        ({"algorithm": "kvaerno3", **FREE_EXITS}, 3),
        ({"algorithm": "radau_iia_3", **FREE_EXITS}, 1),
        ({"algorithm": "crank_nicolson", **FREE_EXITS}, 2),
        ({"algorithm": "backwards_euler", **FREE_EXITS}, 1),
        ({"algorithm": "rosenbrock23", **FREE_EXITS}, 0),
    ],
    indirect=["solver_settings_override"],
)
def test_newton_solves_per_step(solver, solves):
    """Each implicit family reports its Newton solves per step."""
    step = _built_step(solver)
    assert step.newton_solves_per_step == solves


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "kvaerno3", **FREE_EXITS}],
    indirect=True,
)
def test_helper_counts_are_recorded(solver):
    """The step and system carry operator counts after a build."""
    step = _built_step(solver)
    counts = step.compile_settings.helper_operation_counts
    assert counts.residual > 0
    assert counts.lu_solve > 0
    assert step.newton_body_operation_count == counts.residual + counts.lu_solve
    assert solver.system.operation_count > 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "kvaerno3", **FREE_EXITS}],
    indirect=True,
)
def test_small_step_keeps_newton_loop_unrolled(solver):
    """A step under the instruction-cache capacity keeps a full loop."""
    step = _built_step(solver)
    assert _unrolled_instructions(solver, step) <= _capacity()
    assert step.compile_settings.unroll.unroll_newton_exits == (True, None)


@pytest.mark.parametrize(
    "solver_settings_override",
    [LARGE_KVAERNO3, LARGE_RADAU3],
    indirect=True,
)
def test_large_step_rolls_newton_loop(solver):
    """A step over the instruction-cache capacity rolls its Newton loop."""
    step = _built_step(solver)
    assert _unrolled_instructions(solver, step) > _capacity()
    rolled = UnrollChoice.ROLLED.value
    assert step.compile_settings.unroll.unroll_newton_exits == rolled
    assert step.solver.compile_settings.unroll.unroll_newton_exits == rolled


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**LARGE_KVAERNO3, "unroll_newton_exits": (True, None)}],
    indirect=True,
)
def test_user_newton_flag_wins(solver):
    """An explicit ``unroll_newton_exits`` is never overridden."""
    step = _built_step(solver)
    assert step.compile_settings.unroll.unroll_newton_exits == (True, None)


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_RADAU3], indirect=True
)
def test_large_firk_stage_increment_moves_to_shared(solver):
    """FIRK ``stage_increment`` is shared above the size cut."""
    step = _built_step(solver)
    assert step.n > SHARED_STAGE_INCREMENT_MIN_STATES
    assert step.compile_settings.stage_increment_location == "shared"
    assert solver.kernel.shared_memory_bytes > 0


@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "radau_iia_3", **FREE_EXITS}],
    indirect=True,
)
def test_small_firk_stage_increment_stays_local(solver):
    """FIRK ``stage_increment`` stays local at or below the size cut."""
    step = _built_step(solver)
    assert step.n <= SHARED_STAGE_INCREMENT_MIN_STATES
    assert step.compile_settings.stage_increment_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**LARGE_RADAU3, "stage_increment_location": "local"}],
    indirect=True,
)
def test_user_location_wins_over_placement(solver):
    """An explicit ``stage_increment_location`` is never overridden."""
    step = _built_step(solver)
    assert step.compile_settings.stage_increment_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**LARGE_RADAU3, "auto_performance": False}],
    indirect=True,
)
def test_auto_performance_off_leaves_settings_alone(solver):
    """``auto_performance=False`` changes no step setting."""
    step = _built_step(solver)
    assert step.compile_settings.stage_increment_location == "local"
    assert step.compile_settings.unroll.unroll_newton_exits == (True, None)


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_KVAERNO3], indirect=True
)
def test_defaults_rerun_after_algorithm_update(solver_mutable):
    """Switching algorithm re-applies the defaults on the next build."""
    step = _built_step(solver_mutable)
    assert step.compile_settings.stage_increment_location == "local"
    solver_mutable.update(algorithm="radau_iia_3")
    step = _built_step(solver_mutable)
    assert step.compile_settings.stage_increment_location == "shared"
    rolled = UnrollChoice.ROLLED.value
    assert step.compile_settings.unroll.unroll_newton_exits == rolled
