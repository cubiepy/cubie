import numpy as np
import pytest
from cubie.cuda_simsafe import cuda
from cubie.memory import default_memmgr
from numpy.testing import assert_allclose

from cubie.integrators.matrix_free_solvers.linear_solver import (
    MRLinearSolver,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)
from cubie.integrators.matrix_free_solvers import CUBIE_RESULT_CODES

STATUS_MASK = 0xFFFF


def test_newton_krylov_update_with_no_changes_returns_empty_set(precision):
    """update() with no arguments returns an empty set without error."""
    linear_solver_instance = MRLinearSolver(precision=precision, solver_width=1)
    newton_instance = NewtonKrylov(
        precision=precision, solver_width=1, linear_solver=linear_solver_instance,
    )
    assert newton_instance.update() == set()
    assert newton_instance.update(updates_dict={}) == set()


@pytest.mark.parametrize(
    "newton_edge_case",
    [
        "linear-failure-blocks-accept",
        "linear-failure-gates-commit",
        "max-iters-exceeded",
        "small-first-step",
        "stagnation-divergence",
        "theta-growth-divergence",
        "warm-start",
    ],
    indirect=True,
)
def test_newton_krylov_convergence_edges(
    newton_edge_case, newton_edge_outcome
):
    """Acceptance, warm start, and divergence match the NLNewton rules."""
    case = newton_edge_case
    finals, statuses, counts = newton_edge_outcome

    expected_statuses = [
        int(status) for status in case["expected_statuses"]
    ]
    assert list(statuses & STATUS_MASK) == expected_statuses
    assert list(counts) == list(case["expected_counts"])
    assert_allclose(
        finals,
        np.array(
            case["expected_finals"], dtype=finals.dtype
        ).reshape(finals.shape),
        atol=max(case["final_tolerance"], 0.0) + 1e-12,
    )


_NEWTON_SOLVER_SETTINGS = {
    "minimal_residual": {
        "linear_correction_type": "minimal_residual",
        "krylov_atol": 1e-6,
        "krylov_rtol": 1e-6,
        "krylov_max_iters": 1000,
        "newton_atol": 1e-6,
        "newton_rtol": 1e-6,
        "newton_max_iters": 1000,
    },
    "bicgstab": {
        "linear_correction_type": "bicgstab",
        "krylov_atol": 1e-6,
        "krylov_rtol": 1e-6,
        "krylov_max_iters": 200,
        "newton_atol": 1e-6,
        "newton_rtol": 1e-6,
        "newton_max_iters": 1000,
    },
}


# One challenging system per correction type: the fully coupled
# nonlinear system stresses the Newton iteration itself, at
# preconditioner order 1 so the generated preconditioner is in the
# loop. The preconditioner-order sweep lives with the linear solver
# in test_preconditioner_order_reduces_iterations.
@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
@pytest.mark.parametrize(
    "matrixfree_settings_override",
    [
        dict(settings, preconditioner_order=1)
        for settings in _NEWTON_SOLVER_SETTINGS.values()
    ],
    ids=list(_NEWTON_SOLVER_SETTINGS),
    indirect=True,
)
def test_newton_krylov_symbolic(
    system_setup, newton_solver_instance, newton_kernel, tolerance
):
    """Newton with each configured inner solver reaches the reference."""
    base_state = system_setup["base_state"]
    expected = system_setup["nk_expected"]
    h = system_setup["h"]

    kernel = newton_kernel(newton_solver_instance)

    base_vals = system_setup["base_state"].copy_to_host()
    expected_increment = expected - base_vals
    # The solver writes its iterate in place, so the session-scoped
    # seed is uploaded fresh for each cell.
    x = cuda.to_device(system_setup["state_init"].copy_to_host())
    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](x, base_state, out_flag, h)
    stream.synchronize()
    status_code = int(out_flag.copy_to_host()[0]) & STATUS_MASK
    assert status_code == CUBIE_RESULT_CODES.SUCCESS
    # Scaled norm may converge at different iterations than L2 norm,
    # producing slightly different final values (~5% difference).
    assert_allclose(
        x.copy_to_host(),
        expected_increment,
        rtol=tolerance.rel_loose * 1000,
        atol=tolerance.abs_loose * 1000,
    )


def test_newton_krylov_config_scalar_tolerance_broadcast(precision):
    """Verify scalar newton_atol/rtol broadcasts to array of length n."""
    n = 5
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
        newton_atol=1e-6,
        newton_rtol=1e-4,
    )
    assert newton.newton_atol.shape == (n,)
    assert newton.newton_rtol.shape == (n,)
    assert np.all(newton.newton_atol == precision(1e-6))
    assert np.all(newton.newton_rtol == precision(1e-4))


def test_newton_krylov_config_array_tolerance_accepted(precision):
    """Verify array tolerances of correct length are accepted."""
    n = 3
    atol = np.array([1e-6, 1e-8, 1e-4], dtype=precision)
    rtol = np.array([1e-3, 1e-5, 1e-2], dtype=precision)
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
        newton_atol=atol,
        newton_rtol=rtol,
    )
    assert np.allclose(newton.newton_atol, atol)
    assert np.allclose(newton.newton_rtol, rtol)


def test_newton_krylov_config_wrong_length_raises(precision):
    """Verify wrong-length tolerance array raises ValueError."""
    n = 3
    wrong_atol = np.array([1e-6, 1e-8], dtype=precision)  # length 2
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    with pytest.raises(ValueError, match="tol must have shape"):
        NewtonKrylov(
            precision=precision,
            solver_width=n,
            linear_solver=linear_solver,
            newton_atol=wrong_atol,
        )


def test_newton_krylov_uses_scaled_norm(precision):
    """Verify NewtonKrylov uses ScaledNorm for convergence checking."""
    from cubie.integrators.norms import ScaledNorm

    n = 3
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
        newton_atol=1e-6,
        newton_rtol=1e-4,
    )
    # Verify norm factory exists and is a ScaledNorm
    assert hasattr(newton, "norm")
    assert isinstance(newton.norm, ScaledNorm)
    # Verify norm has correct configuration
    assert newton.norm.solver_width == n
    assert newton.norm.precision == precision
    assert np.all(newton.norm.atol == precision(1e-6))
    assert np.all(newton.norm.rtol == precision(1e-4))


def test_newton_krylov_tolerance_update_propagates(precision):
    """Verify newton_atol/newton_rtol updates reach norm factory."""
    n = 3
    initial_atol = 1e-6
    initial_rtol = 1e-4
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
        newton_atol=initial_atol,
        newton_rtol=initial_rtol,
    )
    # Verify initial values
    assert np.all(newton.newton_atol == precision(initial_atol))
    assert np.all(newton.newton_rtol == precision(initial_rtol))
    assert np.all(newton.norm.atol == precision(initial_atol))
    assert np.all(newton.norm.rtol == precision(initial_rtol))

    # Update tolerances
    new_atol = 1e-8
    new_rtol = 1e-6
    newton.update(newton_atol=new_atol, newton_rtol=new_rtol)

    # Verify properties delegate to norm factory
    assert np.all(newton.newton_atol == precision(new_atol))
    assert np.all(newton.newton_rtol == precision(new_rtol))
    # Verify norm factory was updated
    assert np.all(newton.norm.atol == precision(new_atol))
    assert np.all(newton.norm.rtol == precision(new_rtol))


def test_newton_krylov_config_no_tolerance_fields(precision):
    """Verify NewtonKrylovConfig no longer has tolerance scalar fields."""
    from cubie.integrators.matrix_free_solvers.newton_krylov import (
        NewtonKrylovConfig,
    )
    import attrs

    # Get all field names from NewtonKrylovConfig
    field_names = {f.name for f in attrs.fields(NewtonKrylovConfig)}

    # Verify legacy tolerance scalar fields are NOT present
    assert "_newton_tolerance" not in field_names

    # Verify tolerance array fields are NOT in config (managed by norm)
    assert "newton_atol" not in field_names
    assert "newton_rtol" not in field_names

    # Verify we can still instantiate the config
    config = NewtonKrylovConfig(precision=precision, solver_width=3)
    assert config.solver_width == 3
    assert config.precision == precision


def test_newton_krylov_config_settings_dict_excludes_tolerance_arrays(
    precision,
):
    """Verify settings_dict does not include tolerance arrays."""
    from cubie.integrators.matrix_free_solvers.newton_krylov import (
        NewtonKrylovConfig,
    )

    config = NewtonKrylovConfig(precision=precision, solver_width=3)
    settings = config.settings_dict

    # Verify tolerance arrays are NOT in settings_dict
    assert "newton_atol" not in settings
    assert "newton_rtol" not in settings

    # Verify legacy tolerance scalar is NOT in settings_dict
    assert "newton_tolerance" not in settings

    # Verify other expected keys ARE present
    assert "newton_max_iters" in settings
    assert "delta_location" in settings
    assert "residual_location" in settings
    assert "krylov_iters_local_location" in settings
    assert "prev_theta_location" in settings


def test_newton_krylov_inherits_from_matrix_free_solver(precision):
    """Verify NewtonKrylov is instance of MatrixFreeSolver."""
    from cubie.integrators.matrix_free_solvers.base_solver import (
        MatrixFreeSolver,
    )

    n = 3
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
    )
    assert isinstance(newton, MatrixFreeSolver)
    # Verify solver_type is set correctly
    assert newton.solver_type == "newton"


def test_newton_krylov_update_preserves_original_dict(precision):
    """Verify update() does not modify the input updates_dict."""
    from cubie.cuda_simsafe import cuda

    @cuda.jit(device=True)
    def residual(state, parameters, drivers, t, h, a_ij, base_state, out):
        out[0] = state[0]

    n = 1
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
    )

    # Create a dict to pass to update
    original_dict = {
        "newton_atol": 1e-8,
        "newton_rtol": 1e-6,
        "residual_function": residual,
    }
    # Make a copy to compare against
    expected_dict = dict(original_dict)

    # Call update with the dict
    newton.update(original_dict)

    # Original dict should not be modified
    assert original_dict == expected_dict


def test_newton_krylov_no_manual_cache_invalidation(precision):
    """Verify cache invalidation happens through config update."""
    n = 3
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
        newton_atol=1e-6,
        newton_rtol=1e-4,
    )

    newton.update(solver_width=n)
    _ = newton.device_function
    config = newton.compile_settings
    old_norm_function = config.norm_device_function
    assert old_norm_function is not None

    newton.update(newton_atol=1e-8)

    config = newton.compile_settings
    assert config.norm_device_function is newton.norm.device_function
    assert config.norm_device_function is not old_norm_function
    assert newton._cache_valid is False


def test_newton_krylov_settings_dict_includes_tolerance_arrays(precision):
    """Verify settings_dict includes newton_atol and newton_rtol from norm."""
    n = 3
    newton_atol = np.array([1e-6, 1e-8, 1e-4], dtype=precision)
    newton_rtol = np.array([1e-3, 1e-5, 1e-2], dtype=precision)
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
        newton_atol=newton_atol,
        newton_rtol=newton_rtol,
    )
    settings = newton.settings_dict

    # Tolerance arrays should be in settings_dict
    assert "newton_atol" in settings
    assert "newton_rtol" in settings
    assert np.allclose(settings["newton_atol"], newton_atol)
    assert np.allclose(settings["newton_rtol"], newton_rtol)

    # Other expected settings from config should also be present
    assert "newton_max_iters" in settings
    assert "delta_location" in settings
    assert "residual_location" in settings

    # Linear solver settings should be merged in as well
    assert "krylov_max_iters" in settings
    assert "krylov_atol" in settings
    assert "krylov_rtol" in settings


def test_newton_krylov_init_with_newton_prefixed_kwargs(precision):
    """Verify NewtonKrylov accepts newton_* kwargs at init and they reach
    config/norm.
    """
    n = 3
    newton_atol = np.array([1e-10, 1e-9, 1e-8], dtype=precision)
    newton_rtol = np.array([1e-5, 1e-4, 1e-3], dtype=precision)

    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
        newton_atol=newton_atol,
        newton_rtol=newton_rtol,
        newton_max_iters=50,
    )

    # Verify tolerances reached NewtonKrylov's norm
    assert np.allclose(newton.newton_atol, newton_atol)
    assert np.allclose(newton.newton_rtol, newton_rtol)
    assert np.allclose(newton.norm.atol, newton_atol)
    assert np.allclose(newton.norm.rtol, newton_rtol)

    # Verify max_iters reached config
    assert newton.newton_max_iters == 50


def test_newton_krylov_forwards_krylov_kwargs_to_linear_solver(precision):
    """Verify krylov_* kwargs passed to NewtonKrylov reach the nested
    MRLinearSolver via update chain.
    """
    n = 3
    krylov_atol = np.array([1e-12, 1e-11, 1e-10], dtype=precision)
    krylov_rtol = np.array([1e-6, 1e-5, 1e-4], dtype=precision)

    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
    )

    # Update via newton with krylov-prefixed keys
    newton.update(krylov_atol=krylov_atol, krylov_rtol=krylov_rtol)

    # Verify update reached nested MRLinearSolver and its norm
    assert np.allclose(newton.krylov_atol, krylov_atol)
    assert np.allclose(newton.linear_solver.krylov_atol, krylov_atol)
    assert np.allclose(newton.linear_solver.norm.atol, krylov_atol)
    assert np.allclose(newton.krylov_rtol, krylov_rtol)
    assert np.allclose(newton.linear_solver.krylov_rtol, krylov_rtol)
    assert np.allclose(newton.linear_solver.norm.rtol, krylov_rtol)


def test_nested_prefix_propagation_init(precision):
    """Verify prefixed params reach nested objects via init chain.

    Tests that krylov_atol passed to MRLinearSolver constructor
    reaches the nested ScaledNorm at init time.
    """
    n = 3
    krylov_atol = np.array([1e-10, 1e-9, 1e-8], dtype=precision)
    krylov_rtol = np.array([1e-5, 1e-4, 1e-3], dtype=precision)

    linear_solver = MRLinearSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=krylov_atol,
        krylov_rtol=krylov_rtol,
    )

    # Verify tolerances reached MRLinearSolver's norm
    assert np.allclose(linear_solver.krylov_atol, krylov_atol)
    assert np.allclose(linear_solver.krylov_rtol, krylov_rtol)
    assert np.allclose(linear_solver.norm.atol, krylov_atol)
    assert np.allclose(linear_solver.norm.rtol, krylov_rtol)


def test_nested_prefix_propagation_update(precision):
    """Verify prefixed params reach nested objects via update chain.

    Tests that krylov_atol passed to NewtonKrylov.update()
    reaches the nested MRLinearSolver's ScaledNorm.
    """
    n = 3
    linear_solver = MRLinearSolver(precision=precision, solver_width=n)
    newton = NewtonKrylov(
        precision=precision,
        solver_width=n,
        linear_solver=linear_solver,
    )

    new_krylov_atol = np.array([1e-12, 1e-11, 1e-10], dtype=precision)

    # Update via newton with krylov-prefixed key
    newton.update(krylov_atol=new_krylov_atol)

    # Verify update reached nested MRLinearSolver and its norm
    assert np.allclose(newton.krylov_atol, new_krylov_atol)
    assert np.allclose(newton.linear_solver.krylov_atol, new_krylov_atol)
    assert np.allclose(newton.linear_solver.norm.atol, new_krylov_atol)
