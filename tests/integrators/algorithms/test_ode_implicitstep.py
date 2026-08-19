"""Tests for ODEImplicitStep tolerance parameter routing."""

import numpy as np
import pytest

from cubie.integrators.algorithms.backwards_euler import BackwardsEulerStep
from cubie.integrators.algorithms.generic_rosenbrock_w import (
    GenericRosenbrockWStep,
)
from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
    BiCGSTABSolver,
)
from cubie.integrators.matrix_free_solvers.linear_solver import (
    MRLinearSolver,
)
from tests._utils import (
    RESIDUAL_ARRANGEMENTS,
    RESIDUAL_SETTINGS,
)


def test_implicit_step_accepts_tolerance_arrays(precision):
    """Verify implicit step forwards tolerance arrays to nested solvers."""
    n = 3
    krylov_atol = np.array([1e-6, 1e-7, 1e-8], dtype=precision)
    krylov_rtol = np.array([1e-4, 1e-5, 1e-6], dtype=precision)
    newton_atol = np.array([1e-3, 1e-4, 1e-5], dtype=precision)
    newton_rtol = np.array([1e-2, 1e-3, 1e-4], dtype=precision)

    step = BackwardsEulerStep(
        precision=precision,
        n=n,
        krylov_atol=krylov_atol,
        krylov_rtol=krylov_rtol,
        newton_atol=newton_atol,
        newton_rtol=newton_rtol,
    )

    assert np.allclose(step.krylov_atol, krylov_atol)
    assert np.allclose(step.krylov_rtol, krylov_rtol)
    assert np.allclose(step.newton_atol, newton_atol)
    assert np.allclose(step.newton_rtol, newton_rtol)


def test_implicit_step_exposes_tolerance_properties(precision):
    """Verify tolerance array properties return correct values."""
    n = 5
    krylov_atol_scalar = 1e-6
    krylov_rtol_scalar = 1e-4
    newton_atol_scalar = 1e-3
    newton_rtol_scalar = 1e-2

    step = BackwardsEulerStep(
        precision=precision,
        n=n,
        krylov_atol=krylov_atol_scalar,
        krylov_rtol=krylov_rtol_scalar,
        newton_atol=newton_atol_scalar,
        newton_rtol=newton_rtol_scalar,
    )

    # Verify arrays have correct shape
    assert step.krylov_atol.shape == (n,)
    assert step.krylov_rtol.shape == (n,)
    assert step.newton_atol.shape == (n,)
    assert step.newton_rtol.shape == (n,)

    # Verify arrays have correct values (scalar broadcast to array)
    assert np.all(step.krylov_atol == precision(krylov_atol_scalar))
    assert np.all(step.krylov_rtol == precision(krylov_rtol_scalar))
    assert np.all(step.newton_atol == precision(newton_atol_scalar))
    assert np.all(step.newton_rtol == precision(newton_rtol_scalar))


def test_direct_construction_matches_hot_swap_products(precision, system):
    """Direct construction and an equivalent update sequence converge.

    Both routes must end with equal snapshots, equal config_hash, and
    the same bound helper members — equal generated products, not just
    equal hashes.
    """
    kwargs = {
        "precision": precision,
        "n": system.sizes.states,
        "evaluate_f": system.evaluate_f,
        "evaluate_observables": system.evaluate_observables,
        "get_solver_helper_fn": system.get_solver_helper,
    }
    direct = BackwardsEulerStep(preconditioner_order=2, **kwargs)
    swapped = BackwardsEulerStep(preconditioner_order=1, **kwargs)
    swapped.update(preconditioner_order=2)

    assert direct.compile_settings == swapped.compile_settings
    assert direct.config_hash == swapped.config_hash

    direct.build_implicit_helpers()
    swapped.build_implicit_helpers()

    assert direct.config_hash == swapped.config_hash
    assert direct.solver.config_hash == swapped.solver.config_hash

    f_direct = system.get_solver_helper(
        role=direct.compile_settings.preconditioner_type,
        **direct._helper_request_kwargs(),
    ).device_function
    f_swapped = system.get_solver_helper(
        role=swapped.compile_settings.preconditioner_type,
        **swapped._helper_request_kwargs(),
    ).device_function
    assert f_direct is f_swapped


def test_newton_wrapped_solver_assumes_zero_guess(precision):
    """Newton-wrapped linear solvers get zero_initial_guess."""
    step = BackwardsEulerStep(precision=precision, n=3)
    config = step.solver.linear_solver.compile_settings
    assert config.zero_initial_guess is True


def test_linearly_implicit_solver_keeps_initial_guess(precision):
    """Warm-started linearly-implicit solves keep the initial A @ x."""
    step = GenericRosenbrockWStep(precision=precision, n=3)
    assert step.solver.compile_settings.zero_initial_guess is False


def test_implicit_step_linear_solver_newton_atol_returns_none(precision):
    """Verify newton_atol/rtol return None for a linearly-implicit step."""
    step = GenericRosenbrockWStep(precision=precision, n=3)

    # MRLinearSolver doesn't have newton_atol/rtol, so properties return None
    assert step.newton_atol is None
    assert step.newton_rtol is None

    # But krylov_atol/rtol are still available
    assert step.krylov_atol is not None
    assert step.krylov_rtol is not None


def test_is_linear_marks_direct_linear_solver_ownership(precision):
    """is_linear is True only for linearly-implicit step classes."""
    assert GenericRosenbrockWStep.is_linear
    assert not BackwardsEulerStep.is_linear
    step = BackwardsEulerStep(precision=precision, n=3)
    assert not step.is_linear


def test_implicit_config_settings_dict_includes_implicit_fields(precision):
    """ImplicitStepConfig.settings_dict merges base and implicit fields."""
    step = BackwardsEulerStep(precision=precision, n=3)
    settings = step.compile_settings.settings_dict
    assert settings['beta'] == step.compile_settings.beta
    assert settings['gamma'] == step.compile_settings.gamma
    assert 'M' not in settings
    assert (
        settings['preconditioner_order']
        == step.compile_settings.preconditioner_order
    )
    assert (
        settings['preconditioner_type']
        == step.compile_settings.preconditioner_type
    )
    assert (
        settings['get_solver_helper_fn']
        == step.compile_settings.get_solver_helper_fn
    )


def test_implicit_step_beta_gamma_properties(precision):
    """beta and gamma forward to compile_settings."""
    step = BackwardsEulerStep(precision=precision, n=3)
    assert step.beta == step.compile_settings.beta
    assert step.gamma == step.compile_settings.gamma


def test_implicit_step_preconditioner_type_property(precision):
    """preconditioner_type forwards to compile_settings."""
    step = BackwardsEulerStep(
        precision=precision, n=3, preconditioner_type='jacobi',
    )
    assert step.preconditioner_type == 'jacobi'


@pytest.mark.parametrize(
    "preconditioner_type,expected",
    [
        ("neumann", 2),
        ("jacobi", 0),
    ],
    ids=["neumann", "jacobi"],
)
def test_unset_preconditioner_order_follows_the_type(
    precision, preconditioner_type, expected
):
    """An unset order takes the selected type's default."""
    step = BackwardsEulerStep(
        precision=precision,
        n=3,
        preconditioner_type=preconditioner_type,
    )
    assert step.preconditioner_order == expected


def test_set_preconditioner_order_survives_a_type_change(precision):
    """An explicit order survives a type change; an unset one re-resolves."""
    explicit = BackwardsEulerStep(
        precision=precision,
        n=3,
        preconditioner_type='jacobi',
        preconditioner_order=2,
    )
    explicit.update(preconditioner_type='neumann')
    assert explicit.preconditioner_order == 2

    unset = BackwardsEulerStep(
        precision=precision, n=3, preconditioner_type='jacobi',
    )
    assert unset.preconditioner_order == 0
    unset.update(preconditioner_type='neumann')
    assert unset.preconditioner_order == 2


def test_preconditioner_order_rejects_values_above_two(precision):
    """Implicit-step config rejects unsupported series orders."""
    with pytest.raises(ValueError):
        BackwardsEulerStep(precision=precision, n=3, preconditioner_order=3)


def test_implicit_step_update_invokes_register_buffers_override(precision):
    """update() dispatches to ODEImplicitStep's no-op register_buffers."""
    step = BackwardsEulerStep(precision=precision, n=3)
    recognised = step.update(newton_atol=1e-5)
    assert 'newton_atol' in recognised


def test_implicit_step_settings_dict_merges_solver_settings(precision):
    """ODEImplicitStep.settings_dict merges algorithm and solver keys."""
    step = BackwardsEulerStep(precision=precision, n=3)
    settings = step.settings_dict
    solver_settings = step.solver.settings_dict
    for key, value in solver_settings.items():
        assert key in settings


_RESIDUAL_IDS = ["newton-mr", "newton-bicgstab", "direct-linear"]


@pytest.mark.parametrize(
    "solver_settings_override",
    RESIDUAL_ARRANGEMENTS,
    ids=_RESIDUAL_IDS,
    indirect=True,
)
def test_implicit_step_routes_residual_settings(step_object, precision):
    """Every solver arrangement routes the linear stopping settings."""
    step = step_object
    assert step.krylov_residual_reduction == precision(0.2)
    assert step.krylov_residual_floor == precision(0.03)
    assert step.settings_dict["krylov_residual_reduction"] == precision(0.2)
    assert step.settings_dict["krylov_residual_floor"] == precision(0.03)


@pytest.mark.parametrize(
    "solver_settings_override",
    RESIDUAL_ARRANGEMENTS,
    ids=_RESIDUAL_IDS,
    indirect=True,
)
def test_implicit_step_updates_residual_settings(
    step_object_mutable, precision
):
    """update() reroutes the linear stopping settings to the solver."""
    step = step_object_mutable
    recognized = step.update(
        krylov_residual_reduction=0.25,
        krylov_residual_floor=0.04,
    )
    assert {
        "krylov_residual_reduction",
        "krylov_residual_floor",
    } <= recognized
    assert step.krylov_residual_reduction == precision(0.25)
    assert step.krylov_residual_floor == precision(0.04)


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {**RESIDUAL_SETTINGS, "algorithm": "backwards_euler"},
        {**RESIDUAL_SETTINGS, "algorithm": "ros3p"},
    ],
    ids=["newton", "linear"],
    indirect=True,
)
def test_update_swaps_linear_solver_to_bicgstab(step_object_mutable):
    """update() rebuilds the linear solver as BiCGSTAB, keeping state."""
    step = step_object_mutable
    assert isinstance(step.linear_solver, MRLinearSolver)
    atol_before = np.array(step.krylov_atol, copy=True)
    rtol_before = np.array(step.krylov_rtol, copy=True)
    reduction_before = step.krylov_residual_reduction
    floor_before = step.krylov_residual_floor

    recognized = step.update(linear_correction_type="bicgstab")

    assert "linear_correction_type" in recognized
    assert isinstance(step.linear_solver, BiCGSTABSolver)
    assert step.linear_correction_type == "bicgstab"
    assert step.krylov_residual_reduction == reduction_before
    assert step.krylov_residual_floor == floor_before
    assert np.allclose(step.krylov_atol, atol_before)
    assert np.allclose(step.krylov_rtol, rtol_before)
    assert step.step_function is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            **RESIDUAL_SETTINGS,
            "algorithm": "backwards_euler",
            "linear_correction_type": "bicgstab",
        },
    ],
    ids=["newton-bicgstab"],
    indirect=True,
)
def test_update_swaps_linear_solver_back_to_mr(step_object_mutable):
    """update() rebuilds a BiCGSTAB solver as MR, keeping state."""
    step = step_object_mutable
    assert isinstance(step.linear_solver, BiCGSTABSolver)
    reduction_before = step.krylov_residual_reduction
    floor_before = step.krylov_residual_floor

    recognized = step.update(
        linear_correction_type="minimal_residual"
    )

    assert "linear_correction_type" in recognized
    assert isinstance(step.linear_solver, MRLinearSolver)
    assert step.linear_correction_type == "minimal_residual"
    assert step.krylov_residual_reduction == reduction_before
    assert step.krylov_residual_floor == floor_before
    assert step.step_function is not None


def test_update_within_mr_class_switches_correction(precision):
    """MR/SD switches stay inside MRLinearSolver's own update."""
    step = BackwardsEulerStep(precision=precision, n=3)
    solver_before = step.linear_solver

    recognized = step.update(
        linear_correction_type="steepest_descent"
    )

    assert "linear_correction_type" in recognized
    assert step.linear_solver is solver_before
    assert step.linear_correction_type == "steepest_descent"


def test_rosenbrock_zero_guess_update_unrecognized(precision):
    """Rosenbrock updates never recognise zero_initial_guess."""
    step = GenericRosenbrockWStep(precision=precision, n=3)
    recognized = step.solver.update(
        zero_initial_guess=True, silent=True
    )
    assert "zero_initial_guess" not in recognized
    assert step.solver.compile_settings.zero_initial_guess is False
    recognized = step.update(zero_initial_guess=True, silent=True)
    assert "zero_initial_guess" not in recognized
    assert step.solver.compile_settings.zero_initial_guess is False


def test_newton_zero_guess_update_unrecognized(precision):
    """Newton-path updates never recognise zero_initial_guess."""
    step = BackwardsEulerStep(precision=precision, n=3)
    recognized = step.update(zero_initial_guess=False, silent=True)
    assert "zero_initial_guess" not in recognized
    config = step.solver.linear_solver.compile_settings
    assert config.zero_initial_guess is True


def test_hot_swap_preserves_zero_guess_newton(precision):
    """MR <-> BiCGSTAB swaps keep the Newton-derived True flag."""
    step = BackwardsEulerStep(precision=precision, n=3)
    step.update(linear_correction_type="bicgstab")
    config = step.solver.linear_solver.compile_settings
    assert config.zero_initial_guess is True
    step.update(linear_correction_type="minimal_residual")
    config = step.solver.linear_solver.compile_settings
    assert config.zero_initial_guess is True


def test_hot_swap_preserves_zero_guess_rosenbrock(precision):
    """MR <-> BiCGSTAB swaps keep the warm-start-derived False."""
    step = GenericRosenbrockWStep(precision=precision, n=3)
    step.update(linear_correction_type="bicgstab")
    assert step.solver.compile_settings.zero_initial_guess is False
    step.update(linear_correction_type="minimal_residual")
    assert step.solver.compile_settings.zero_initial_guess is False


def test_combined_update_ignores_zero_guess_rosenbrock(precision):
    """Valid keys apply while zero_initial_guess is ignored."""
    step = GenericRosenbrockWStep(precision=precision, n=3)
    recognized = step.update(
        n=4,
        linear_correction_type="bicgstab",
        zero_initial_guess=True,
        silent=True,
    )
    assert "zero_initial_guess" not in recognized
    assert step.compile_settings.n == 4
    assert step.solver.linear_correction_type == "bicgstab"
    assert step.solver.compile_settings.zero_initial_guess is False


def test_combined_update_ignores_zero_guess_newton(precision):
    """The Newton child keeps True through a combined update."""
    step = BackwardsEulerStep(precision=precision, n=3)
    recognized = step.update(
        n=4,
        linear_correction_type="bicgstab",
        zero_initial_guess=False,
        silent=True,
    )
    assert "zero_initial_guess" not in recognized
    assert step.compile_settings.n == 4
    child = step.solver.linear_solver
    assert child.linear_correction_type == "bicgstab"
    assert child.compile_settings.zero_initial_guess is True
