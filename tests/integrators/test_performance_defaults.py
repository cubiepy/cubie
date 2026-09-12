"""Tests for the size and hardware defaults ``auto_performance`` applies."""

import pytest

from numpy import dtype as np_dtype

from cubie.backend.utils import (
    MAX_REGISTERS_PER_THREAD,
    SASS_INSTRUCTION_BYTES,
    device_hardware,
    shared_keeps_occupancy,
)
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import UnrollChoice
from tests._utils import (
    ALGORITHM_CHAIN_SETS,
    KRYLOV_DIRK,
    KRYLOV_FIRK,
    LARGE_DIRK,
    LARGE_FIRK,
    LARGE_KRYLOV_DIRK,
    LARGE_KRYLOV_FIRK,
    LARGE_TSIT5,
    LARGE_VERN7,
    MEDIUM_KRYLOV_DIRK,
)

FULL = UnrollChoice.FULL.value
ROLLED = UnrollChoice.ROLLED.value


def unrolled_operations(step, system):
    """Binary operations of a step with every Newton iteration unrolled."""
    return (
        system.operation_count
        + step.per_step_operation_count
        + step.newton_max_iters
        * step.newton_solves_per_step
        * step.newton_body_operation_count
    )


def instruction_cache_capacity():
    """Instructions the device's instruction cache holds."""
    return device_hardware().instruction_cache_bytes // SASS_INSTRUCTION_BYTES


def shared_buffer_keeps_occupancy(step, elements_per_run, fraction=1):
    """Whether this many shared elements per run keep the occupancy."""
    itemsize = np_dtype(step.precision).itemsize
    return shared_keeps_occupancy(
        device_hardware(), (elements_per_run + 1) * itemsize, fraction
    )


def accumulator_elements(step):
    """Elements of the DIRK explicit-stage accumulator."""
    return max(step.tableau.stage_count - 1, 0) * step.n_states


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
def test_small_direct_dirk_defaults(solver, system):
    """A small direct DIRK counts operators, unrolls Newton, stays local."""
    step = solver.kernel.single_integrator._algo_step
    counts = step.compile_settings.helper_operation_counts
    assert counts.residual > 0
    assert counts.total(step.NEWTON_HELPERS) > counts.residual
    assert step.newton_body_operation_count == counts.total(
        step.NEWTON_HELPERS
    )
    assert system.operation_count > 0
    assert unrolled_operations(step, system) <= instruction_cache_capacity()
    assert step.compile_settings.unroll.unroll_newton_exits == FULL
    assert step.uses_direct_solver
    assert step.compile_settings.accumulator_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override", [ALGORITHM_CHAIN_SETS["firk"]], indirect=True
)
def test_small_direct_firk_keeps_stage_increment_local(solver):
    """A small direct-solve FIRK keeps ``stage_increment`` local."""
    step = solver.kernel.single_integrator._algo_step
    assert step.uses_direct_solver
    assert step.compile_settings.stage_increment_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_DIRK, LARGE_FIRK], indirect=True
)
def test_large_direct_step_rolls_newton_and_stays_local(solver, system):
    """A direct step over the instruction cache rolls Newton, stays local."""
    step = solver.kernel.single_integrator._algo_step
    assert unrolled_operations(step, system) > instruction_cache_capacity()
    assert step.compile_settings.unroll.unroll_newton_exits == ROLLED
    assert (
        step.solver.compile_settings.unroll.unroll_newton_exits == ROLLED
    )
    assert step.uses_direct_solver
    placement = {
        "dirk": "accumulator_location",
        "firk": "stage_increment_location",
    }[step.algorithm_family]
    assert getattr(step.compile_settings, placement) == "local"


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**LARGE_DIRK, "unroll_newton_exits": (True, None)}],
    indirect=True,
)
def test_user_newton_flag_wins(solver):
    """An explicit ``unroll_newton_exits`` is never overridden."""
    step = solver.kernel.single_integrator._algo_step
    assert step.compile_settings.unroll.unroll_newton_exits == FULL


@pytest.mark.parametrize(
    "solver_settings_override", [KRYLOV_FIRK], indirect=True
)
def test_small_krylov_firk_shares_stage_increment(solver):
    """A Krylov FIRK shares ``stage_increment`` while occupancy holds."""
    run = solver.kernel.single_integrator
    step = run._algo_step
    assert not step.uses_direct_solver
    assert shared_buffer_keeps_occupancy(
        step, step.stage_count * step.n_states
    )
    assert step.compile_settings.stage_increment_location == "shared"
    assert run.shared_memory_elements > 0


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_KRYLOV_FIRK], indirect=True
)
def test_large_krylov_firk_keeps_stage_increment_local(solver):
    """A Krylov FIRK whose shared buffer would cost threads stays local."""
    step = solver.kernel.single_integrator._algo_step
    assert not step.uses_direct_solver
    assert not shared_buffer_keeps_occupancy(
        step, step.stage_count * step.n_states
    )
    assert step.compile_settings.stage_increment_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override", [KRYLOV_DIRK], indirect=True
)
def test_register_resident_krylov_dirk_keeps_accumulator_local(solver):
    """A Krylov DIRK within the registers keeps its accumulator local."""
    step = solver.kernel.single_integrator._algo_step
    assert not step.uses_direct_solver
    declared = buffer_registry.declared_local_elements(step)
    assert declared <= MAX_REGISTERS_PER_THREAD
    assert step.compile_settings.accumulator_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override", [MEDIUM_KRYLOV_DIRK], indirect=True
)
def test_spilling_krylov_dirk_shares_accumulator(solver_mutable):
    """A spilling Krylov DIRK shares its accumulator, also after a rebuild."""
    run = solver_mutable.kernel.single_integrator
    step = run._algo_step
    assert not step.uses_direct_solver
    declared = buffer_registry.declared_local_elements(step)
    assert declared > MAX_REGISTERS_PER_THREAD
    assert shared_buffer_keeps_occupancy(
        step, accumulator_elements(step), fraction=2
    )
    assert step.compile_settings.accumulator_location == "shared"
    assert run.shared_memory_elements > 0
    solver_mutable.update(krylov_max_iters=step.solver.krylov_max_iters + 1)
    run.device_function
    assert run._algo_step.compile_settings.accumulator_location == "shared"


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_KRYLOV_DIRK], indirect=True
)
def test_large_krylov_dirk_keeps_accumulator_local(solver):
    """A Krylov DIRK whose shared accumulator costs occupancy stays local."""
    step = solver.kernel.single_integrator._algo_step
    declared = buffer_registry.declared_local_elements(step)
    assert declared > MAX_REGISTERS_PER_THREAD
    assert not shared_buffer_keeps_occupancy(
        step, accumulator_elements(step), fraction=2
    )
    assert step.compile_settings.accumulator_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**MEDIUM_KRYLOV_DIRK, "accumulator_location": "local"}],
    indirect=True,
)
def test_user_accumulator_location_wins(solver):
    """An explicit ``accumulator_location`` is never overridden."""
    step = solver.kernel.single_integrator._algo_step
    assert step.compile_settings.accumulator_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_VERN7], indirect=True
)
def test_large_accumulating_erk_state_follows_occupancy(solver):
    """A spilling accumulating ERK shares ``state`` while occupancy holds."""
    run = solver.kernel.single_integrator
    step = run._algo_step
    assert step.tableau.accumulates_output
    assert step.n_states * step.stage_count > MAX_REGISTERS_PER_THREAD
    expected = (
        "shared"
        if shared_buffer_keeps_occupancy(step, step.n_states)
        else "local"
    )
    assert run._loop.compile_settings.state_location == expected


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_TSIT5], indirect=True
)
def test_large_copied_output_erk_keeps_state_local(solver):
    """An ERK that copies its output from a stage keeps ``state`` local."""
    run = solver.kernel.single_integrator
    step = run._algo_step
    assert not step.tableau.accumulates_output
    assert step.n_states * step.stage_count > MAX_REGISTERS_PER_THREAD
    assert run._loop.compile_settings.state_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override", [ALGORITHM_CHAIN_SETS["erk"]], indirect=True
)
def test_small_erk_keeps_state_local(solver):
    """An ERK whose stage vectors fit the registers keeps ``state`` local."""
    run = solver.kernel.single_integrator
    step = run._algo_step
    assert step.n_states * step.stage_count <= MAX_REGISTERS_PER_THREAD
    assert run._loop.compile_settings.state_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override, expected",
    [
        ({**LARGE_TSIT5, "state_location": "local"}, "local"),
        ({**LARGE_TSIT5, "state_location": "shared"}, "shared"),
    ],
    indirect=["solver_settings_override"],
)
def test_user_state_location_wins(solver, expected):
    """An explicit ``state_location`` is never overridden."""
    run = solver.kernel.single_integrator
    assert run._loop.compile_settings.state_location == expected


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**KRYLOV_FIRK, "stage_increment_location": "local"}],
    indirect=True,
)
def test_user_location_wins_over_placement(solver):
    """An explicit ``stage_increment_location`` is never overridden."""
    step = solver.kernel.single_integrator._algo_step
    assert step.compile_settings.stage_increment_location == "local"


@pytest.mark.parametrize(
    "solver_settings_override",
    [{**KRYLOV_FIRK, "auto_performance": False}],
    indirect=True,
)
def test_auto_performance_off_leaves_settings_alone(solver):
    """``auto_performance=False`` changes no step setting."""
    step = solver.kernel.single_integrator._algo_step
    assert step.compile_settings.stage_increment_location == "local"
    assert step.compile_settings.unroll.unroll_newton_exits == FULL


@pytest.mark.parametrize(
    "solver_settings_override", [LARGE_DIRK], indirect=True
)
def test_defaults_rerun_after_algorithm_update(solver_mutable):
    """Switching algorithm re-applies the defaults on the next build."""
    run = solver_mutable.kernel.single_integrator
    step = run._algo_step
    assert step.compile_settings.accumulator_location == "local"
    assert step.compile_settings.unroll.unroll_newton_exits == ROLLED
    solver_mutable.update(
        algorithm=KRYLOV_FIRK["algorithm"],
        linear_correction_type=KRYLOV_FIRK["linear_correction_type"],
        preconditioner_type=KRYLOV_FIRK["preconditioner_type"],
    )
    run.device_function
    step = run._algo_step
    assert not step.uses_direct_solver
    assert step.compile_settings.stage_increment_location == "local"
    assert step.compile_settings.unroll.unroll_newton_exits == ROLLED
