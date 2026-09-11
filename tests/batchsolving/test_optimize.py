"""Tests for the per-solver unroll, placement and launch optimisation."""

import pytest

from cubie.batchsolving.optimize import LaunchResult, apply_launch
from cubie.cuda_simsafe import UnrollChoice
from tests._utils import LARGE_FIRK

FULL = UnrollChoice.FULL
ROLLED = UnrollChoice.ROLLED


def _candidates(solver, force=False):
    return solver.kernel.optimisation_candidates(force=force)


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
    "solver_settings_override",
    [{**LARGE_FIRK, "unroll_newton_exits": None}],
    indirect=True,
)
def test_derived_defaults_stay_free_axes_on_a_copy(solver, driver_settings):
    """Defaults the kernel derived are varied by the parent and its copy."""
    run = solver.kernel.single_integrator
    run.device_function
    step = run._algo_step.compile_settings
    assert step.stage_increment_location == "shared"
    assert step.unroll.unroll_newton_exits == ROLLED.value
    expected = (
        {"unroll_newton_exits": FULL, "stage_increment_location": "local"},
        {"unroll_newton_exits": FULL, "stage_increment_location": "shared"},
        {"unroll_newton_exits": ROLLED, "stage_increment_location": "local"},
        {"unroll_newton_exits": ROLLED, "stage_increment_location": "shared"},
    )
    assert _candidates(solver) == expected
    twin = solver.copy()
    try:
        if driver_settings is not None:
            twin._configure_drivers(driver_settings)
        assert _candidates(twin) == expected
        assert twin.kernel.config_hash == solver.kernel.config_hash
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
    assert kernel.blocksize_given
    assert kernel.resident_blocks == 2
    loop = kernel.single_integrator._loop
    assert loop.compile_settings.state_location == "shared"
    assert loop.compile_settings.unroll.unroll_other_small == ROLLED.value


@pytest.mark.nocudasim
@pytest.mark.parametrize(
    "solver_settings_override",
    [{"algorithm": "vern7", "unroll_other_small": None}],
    indirect=True,
)
def test_optimize_applies_the_fastest_launch(
    solver_mutable, simple_initial_values, simple_parameters, driver_settings
):
    """The fastest timed launch is applied to the solver."""
    result = solver_mutable.optimize(
        simple_initial_values,
        parameters=simple_parameters,
        drivers=driver_settings,
        duration=0.1,
        grid_type="combinatorial",
        verbose=False,
    )
    timed = [launch for launch in result.launches if launch.timed]
    assert timed
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
    assert loop.compile_settings.unroll.unroll_other_small == (
        result.best.settings["unroll_other_small"].value
    )
    for launch in result.launches:
        assert len(launch.times_ms) >= 1
        if launch.excluded:
            assert len(launch.times_ms) == 1
    assert "best" in result.summary()
