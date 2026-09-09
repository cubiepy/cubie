"""Tests for the size and hardware defaults ``auto_performance`` applies."""

import pytest

from numpy import dtype as np_dtype

from cubie.backend.utils import (
    MAX_REGISTERS_PER_THREAD,
    SASS_INSTRUCTION_BYTES,
    device_hardware,
    register_limited_threads,
    shared_limited_threads,
)
from cubie.cuda_simsafe import UnrollChoice
from tests._utils import (
    ALGORITHM_CHAIN_SETS,
    LARGE_DIRK,
    LARGE_FIRK,
    LARGE_STATE_ONLY,
    LARGE_TSIT5,
)

FULL = UnrollChoice.FULL.value
ROLLED = UnrollChoice.ROLLED.value

KRYLOV_FIRK = {
    "algorithm": "radau_iia_3",
    "linear_correction_type": "bicgstab",
    "preconditioner_type": "jacobi",
    "step_controller": "fixed",
}
LARGE_KRYLOV_FIRK = {**LARGE_STATE_ONLY, **KRYLOV_FIRK}


def shared_keeps_occupancy(step_object, elements_per_run):
    """Whether a shared footprint admits the register-limited threads."""
    hardware = device_hardware()
    itemsize = np_dtype(step_object.precision).itemsize
    return shared_limited_threads(
        hardware, (elements_per_run + 1) * itemsize
    ) >= register_limited_threads(hardware, MAX_REGISTERS_PER_THREAD)


@pytest.mark.parametrize(
    "solver_settings_override, solves",
    [
        (ALGORITHM_CHAIN_SETS["dirk"], None),
        (ALGORITHM_CHAIN_SETS["firk"], 1),
        (ALGORITHM_CHAIN_SETS["crank_nicolson"], 2),
        (ALGORITHM_CHAIN_SETS["backwards_euler"], 1),
        (ALGORITHM_CHAIN_SETS["rosenbrock"], 0),
    ],
    indirect=["solver_settings_override"],
)
def test_newton_solves_per_step(step_object, solves):
    """Each implicit family reports its Newton solves per step."""
    if solves is None:
        solves = len(step_object.tableau.implicit_stages)
    assert step_object.newton_solves_per_step == solves


@pytest.mark.parametrize(
    "solver_settings_override", [ALGORITHM_CHAIN_SETS["dirk"]], indirect=True
)
def test_helper_counts_are_recorded(single_integrator_run, step_object):
    """The step and system carry operator counts after a build."""
    single_integrator_run.device_function
    counts = step_object.compile_settings.helper_operation_counts
    assert counts.residual > 0
    assert counts.total(step_object.NEWTON_HELPERS) > counts.residual
    assert step_object.newton_body_operation_count == counts.total(
        step_object.NEWTON_HELPERS
    )
    assert single_integrator_run._system.operation_count > 0


@pytest.mark.parametrize(
    "solver_settings_override", [ALGORITHM_CHAIN_SETS["dirk"]], indirect=True
)
def test_small_step_keeps_newton_loop_unrolled(
    single_integrator_run, step_object, system
):
    """A step under the instruction-cache capacity keeps a full loop."""
    single_integrator_run.device_function
    unrolled = (
        system.operation_count
        + step_object.per_step_operation_count
        + step_object.newton_max_iters
        * step_object.newton_solves_per_step
        * step_object.newton_body_operation_count
    )
    capacity = device_hardware().instruction_cache_bytes
    assert unrolled <= capacity // SASS_INSTRUCTION_BYTES
    assert step_object.compile_settings.unroll.unroll_newton_exits == FULL


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_DIRK, LARGE_FIRK], indirect=True
)
def test_large_step_rolls_newton_loop(
    single_integrator_run, step_object, system
):
    """A step over the instruction-cache capacity rolls its Newton loop."""
    single_integrator_run.device_function
    unrolled = (
        system.operation_count
        + step_object.per_step_operation_count
        + step_object.newton_max_iters
        * step_object.newton_solves_per_step
        * step_object.newton_body_operation_count
    )
    capacity = device_hardware().instruction_cache_bytes
    assert unrolled > capacity // SASS_INSTRUCTION_BYTES
    assert step_object.compile_settings.unroll.unroll_newton_exits == ROLLED
    assert (
        step_object.solver.compile_settings.unroll.unroll_newton_exits
        == ROLLED
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**LARGE_DIRK, "unroll_newton_exits": (True, None)}],
    indirect=True,
)
def test_user_newton_flag_wins(solver):
    """An explicit ``unroll_newton_exits`` is never overridden."""
    run = solver.kernel.single_integrator
    run.device_function
    assert run._algo_step.compile_settings.unroll.unroll_newton_exits == FULL


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_FIRK, ALGORITHM_CHAIN_SETS["firk"]],
    indirect=True,
)
def test_direct_firk_stage_increment_stays_local(
    single_integrator_run, step_object
):
    """A direct-solve FIRK keeps ``stage_increment`` local at any size."""
    single_integrator_run.device_function
    assert step_object.uses_direct_solver
    assert step_object.compile_settings.stage_increment_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override", [KRYLOV_FIRK], indirect=True
)
def test_small_krylov_firk_stage_increment_moves_to_shared(
    single_integrator_run, step_object
):
    """A Krylov FIRK shares ``stage_increment`` while occupancy holds."""
    single_integrator_run.device_function
    elements = step_object.stage_count * step_object.n
    assert shared_keeps_occupancy(step_object, elements)
    assert step_object.compile_settings.stage_increment_location == "shared"
    assert single_integrator_run.shared_memory_elements > 0


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_KRYLOV_FIRK], indirect=True
)
def test_large_krylov_firk_stage_increment_stays_local(
    single_integrator_run, step_object
):
    """A Krylov FIRK whose shared footprint costs occupancy stays local."""
    single_integrator_run.device_function
    elements = step_object.stage_count * step_object.n
    assert not shared_keeps_occupancy(step_object, elements)
    assert step_object.compile_settings.stage_increment_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_TSIT5], indirect=True
)
def test_large_erk_state_placement_follows_occupancy(
    single_integrator_run, step_object
):
    """An ERK over the register file shares ``state`` while occupancy holds."""
    single_integrator_run.device_function
    assert step_object.n * step_object.stage_count > MAX_REGISTERS_PER_THREAD
    expected = (
        "shared" if shared_keeps_occupancy(step_object, step_object.n)
        else "local"
    )
    loop = single_integrator_run._loop
    assert loop.compile_settings.state_location == expected


@pytest.mark.parametrize(
    "solver_settings_override", [ALGORITHM_CHAIN_SETS["erk"]], indirect=True
)
def test_small_erk_state_stays_local(single_integrator_run, step_object):
    """An ERK whose stage vectors fit the register file keeps state local."""
    single_integrator_run.device_function
    assert step_object.n * step_object.stage_count <= MAX_REGISTERS_PER_THREAD
    loop = single_integrator_run._loop
    assert loop.compile_settings.state_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override, expected",
    [
        ({**LARGE_TSIT5, "state_location": "local"}, "local"),
        ({**LARGE_TSIT5, "state_location": "shared"}, "shared"),
    ],
    indirect=["solver_settings_override"],
)
def test_user_state_location_wins(single_integrator_run, expected):
    """An explicit ``state_location`` is never overridden."""
    single_integrator_run.device_function
    loop = single_integrator_run._loop
    assert loop.compile_settings.state_location == expected


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**KRYLOV_FIRK, "stage_increment_location": "local"}],
    indirect=True,
)
def test_user_location_wins_over_placement(
    single_integrator_run, step_object
):
    """An explicit ``stage_increment_location`` is never overridden."""
    single_integrator_run.device_function
    assert step_object.compile_settings.stage_increment_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**KRYLOV_FIRK, "auto_performance": False}],
    indirect=True,
)
def test_auto_performance_off_leaves_settings_alone(solver):
    """``auto_performance=False`` changes no step setting."""
    run = solver.kernel.single_integrator
    run.device_function
    step = run._algo_step
    assert step.compile_settings.stage_increment_location == "local"
    assert step.compile_settings.unroll.unroll_newton_exits == FULL


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_DIRK], indirect=True
)
def test_defaults_rerun_after_algorithm_update(solver_mutable):
    """Switching algorithm re-applies the defaults on the next build."""
    run = solver_mutable.kernel.single_integrator
    run.device_function
    assert (
        run._algo_step.compile_settings.stage_increment_location == "local"
    )
    assert run._algo_step.compile_settings.unroll.unroll_newton_exits == ROLLED
    solver_mutable.update(
        algorithm=KRYLOV_FIRK["algorithm"],
        linear_correction_type=KRYLOV_FIRK["linear_correction_type"],
        preconditioner_type=KRYLOV_FIRK["preconditioner_type"],
    )
    run.device_function
    step = run._algo_step
    assert step.compile_settings.stage_increment_location == "local"
    assert step.compile_settings.unroll.unroll_newton_exits == ROLLED
