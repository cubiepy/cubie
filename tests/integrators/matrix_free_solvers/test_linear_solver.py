import numpy as np
import pytest
from cubie.cuda_simsafe import cuda
from cubie.memory import default_memmgr
from numpy.testing import assert_allclose

from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
    BiCGSTABSolver,
)
from cubie.integrators.matrix_free_solvers.linear_solver import (
    MRLinearSolver,
)
from cubie.integrators.matrix_free_solvers import CUBIE_RESULT_CODES
from tests._utils import FLOAT64_PRECISION


@pytest.fixture(scope="session")
def placeholder_operator(precision):
    """Device operator applying a simple SPD matrix."""

    @cuda.jit(device=True)
    def operator(
        state, parameters, drivers, cached_aux, base_state, t, h,
        a_ij, vec, out,
    ):
        out[0] = precision(4.0) * vec[0] + precision(1.0) * vec[1]
        out[1] = precision(1.0) * vec[0] + precision(3.0) * vec[1]
        out[2] = precision(2.0) * vec[2]

    return operator


@pytest.mark.parametrize(
    "solver_settings_override",
    [FLOAT64_PRECISION],
    ids=[""],
    indirect=True,
)
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("system_setup", ["linear"], indirect=True)
def test_neumann_preconditioner(
    order,
    system_setup,
    neumann_kernel,
    precision,
    tolerance,
):
    """Validate Neumann preconditioner equals truncated series on the linear
    system.

    Uses the real generated preconditioner from system_setup and applies it to
    a vector of ones. For the 'linear' system, J is diagonal with 0.5 entries,
    beta=1, stage coefficient a_ij=1, and h=1, so the truncated series is
    sum_{k=0..order} (h*J)^k v.
    """

    n = system_setup["n"]
    h = system_setup["h"]
    precond = system_setup["preconditioner"](order)
    kernel = neumann_kernel(precond, n, h)

    residual = cuda.to_device(np.ones(n, dtype=precision))
    out = cuda.device_array(n, precision)
    state = system_setup["state_init"]
    empty_base = cuda.to_device(np.empty(0, dtype=precision))

    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](state, residual, empty_base, out)
    stream.synchronize()

    expected_scalar = sum((h * precision(0.5)) ** k for k in range(order + 1))
    expected = np.full(n, expected_scalar, dtype=precision)
    assert_allclose(
        out.copy_to_host(),
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


def test_linear_solver_update_with_no_changes_returns_empty_set(precision):
    """update() with no arguments returns an empty set without error."""
    solver = MRLinearSolver(precision=precision, solver_width=3)
    assert solver.update() == set()
    assert solver.update(updates_dict={}) == set()


@pytest.fixture(scope="session")
def placeholder_solver(
    request, placeholder_operator, solver_settings, precision
):
    """Build a solver over the placeholder operator.

    ``solver_settings`` is requested so ``buffer_registry.reset()``
    runs before the solver registers its buffers.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Supplies ``linear_correction_type`` through ``param``.
    placeholder_operator : callable
        Device operator applying the SPD matrix.
    solver_settings : dict
        Session-wide solver configuration.
    precision : np.dtype
        Floating point precision for the solver.

    Returns
    -------
    MRLinearSolver
        The configured solver instance.
    """

    solver = MRLinearSolver(
        precision=precision,
        solver_width=3,
        linear_correction_type=request.param,
        krylov_atol=1e-12,
        krylov_rtol=1e-12,
        krylov_max_iters=32,
    )
    solver.update(operator_apply=placeholder_operator)
    return solver


@pytest.mark.parametrize(
    "placeholder_solver",
    ["steepest_descent", "minimal_residual"],
    indirect=True,
)
def test_linear_solver_placeholder(
    placeholder_solver,
    solver_kernel,
    precision,
    tolerance,
):
    """Solve a simple linear system with the placeholder operator."""

    rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
    matrix = np.array(
        [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=precision,
    )
    expected = np.linalg.solve(matrix, rhs)
    h = precision(0.01)
    kernel = solver_kernel(placeholder_solver, 3, h, precision)
    base_state = np.array([1.0, -1.0, 0.5], dtype=precision)
    state = cuda.to_device(
        base_state + h * np.array([0.1, -0.2, 0.3], dtype=precision)
    )
    rhs_dev = cuda.to_device(rhs)
    x_dev = cuda.to_device(np.zeros(3, dtype=precision))
    flag = cuda.to_device(np.zeros(2, dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](state, rhs_dev, empty_base, x_dev, flag)
    stream.synchronize()
    code = flag.copy_to_host()[0] & 0xFF
    assert code == CUBIE_RESULT_CODES.SUCCESS
    assert np.allclose(
        x_dev.copy_to_host(),
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


def _run_symbolic_linear_solve(
    system_setup, linear_solver_instance, solver_kernel, precision, tolerance
):
    """Solve the fixture system and compare against the direct solution."""
    n = system_setup["n"]
    rhs_vec = system_setup["mr_rhs"]
    expected = system_setup["mr_expected"]
    h = system_setup["h"]

    kernel = solver_kernel(linear_solver_instance, n, h, precision)
    state = system_setup["state_init"]
    rhs_dev = cuda.to_device(rhs_vec)
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.zeros(2, dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](state, rhs_dev, empty_base, x_dev, flag)
    stream.synchronize()
    code = flag.copy_to_host()[0] & 0xFF
    assert code == CUBIE_RESULT_CODES.SUCCESS
    assert np.allclose(
        x_dev.copy_to_host(),
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


_LINEAR_SOLVER_SETTINGS = {
    "steepest_descent": {
        "linear_correction_type": "steepest_descent",
        "krylov_atol": 1e-8,
        "krylov_rtol": 1e-8,
        "krylov_max_iters": 1000,
    },
    "minimal_residual": {
        "linear_correction_type": "minimal_residual",
        "krylov_atol": 1e-8,
        "krylov_rtol": 1e-8,
        "krylov_max_iters": 1000,
    },
    "bicgstab": {
        "linear_correction_type": "bicgstab",
        "krylov_atol": 1e-8,
        "krylov_rtol": 1e-8,
        "krylov_max_iters": 200,
    },
}


# One challenging system per correction type: the coupled
# non-symmetric system for the two minimal-residual corrections, the
# moderately ill-conditioned stiff system for BiCGSTAB. Order 1
# exercises the real generated preconditioner; the order sweep lives
# in test_preconditioner_order_reduces_iterations.
@pytest.mark.parametrize(
    "system_setup, matrixfree_settings_override",
    [
        pytest.param(
            "coupled_linear",
            dict(
                _LINEAR_SOLVER_SETTINGS["steepest_descent"],
                preconditioner_order=1,
            ),
            id="steepest_descent-coupled_linear",
        ),
        pytest.param(
            "coupled_linear",
            dict(
                _LINEAR_SOLVER_SETTINGS["minimal_residual"],
                preconditioner_order=1,
            ),
            id="minimal_residual-coupled_linear",
        ),
        pytest.param(
            "stiff",
            dict(
                _LINEAR_SOLVER_SETTINGS["bicgstab"],
                preconditioner_order=1,
            ),
            id="bicgstab-stiff",
        ),
    ],
    indirect=True,
)
def test_linear_solver_symbolic(
    system_setup,
    linear_solver_instance,
    solver_kernel,
    precision,
    tolerance,
):
    """Each configured solver drives its challenging symbolic system."""
    _run_symbolic_linear_solve(
        system_setup,
        linear_solver_instance,
        solver_kernel,
        precision,
        tolerance,
    )


@pytest.mark.parametrize("system_setup", ["graded"], indirect=True)
def test_preconditioner_order_reduces_iterations(
    system_setup,
    matrixfree_settings,
    solver_kernel,
    precision,
):
    """Each Neumann preconditioner order cuts the iteration count.

    On the graded system ``h*J`` has eigenvalues 0.1, 0.5 and 0.9, so
    the truncated Neumann series converges and every extra order
    tightens the preconditioned spectrum. The minimal-residual solver
    reports its iteration count; the count must fall monotonically
    with the order and strictly from order 0 to order 2. This is the
    direct test of the preconditioner order's effect; convergence per
    correction type is covered above. ``matrixfree_settings`` is
    requested for its buffer-registry reset.
    """
    n = system_setup["n"]
    h = system_setup["h"]
    rhs_vec = system_setup["mr_rhs"]
    state = system_setup["state_init"]
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    stream = default_memmgr.get_group_stream()

    iterations = {}
    for order in (0, 1, 2):
        solver = MRLinearSolver(
            precision=precision,
            solver_width=n,
            linear_correction_type="minimal_residual",
            krylov_atol=1e-6,
            krylov_rtol=0.0,
            krylov_max_iters=1000,
        )
        solver.update(
            operator_apply=system_setup["operator"],
            preconditioner=(
                None if order == 0
                else system_setup["preconditioner"](order)
            ),
        )
        kernel = solver_kernel(solver, n, h, precision)
        rhs_dev = cuda.to_device(rhs_vec.copy())
        x_dev = cuda.to_device(np.zeros(n, dtype=precision))
        flag = cuda.to_device(np.zeros(2, dtype=np.int32))
        kernel[1, 1, stream](state, rhs_dev, empty_base, x_dev, flag)
        stream.synchronize()
        status, iters = flag.copy_to_host()
        assert status & 0xFF == CUBIE_RESULT_CODES.SUCCESS
        iterations[order] = int(iters)

    assert iterations[1] <= iterations[0]
    assert iterations[2] <= iterations[1]
    assert iterations[2] < iterations[0]


# Each solver class reports its own failure mode on the zero
# operator. Minimal residual makes no progress and exhausts its
# iteration budget; BiCGSTAB's pivot quotient rho/<r0_hat, v>
# overflows on the first iteration, which is a breakdown rather than
# an exhausted budget.
_DEGENERATE_EXPECTED_STATUS = {
    "minimal_residual": CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED,
    "bicgstab": CUBIE_RESULT_CODES.BICGSTAB_BREAKDOWN,
}


@pytest.mark.parametrize(
    "degenerate_linear_solver",
    ["minimal_residual", "bicgstab"],
    indirect=True,
)
def test_linear_solver_degenerate_operator(
    degenerate_linear_solver, solver_kernel, precision
):
    """A zero operator produces the per-class failure status."""
    n = 3
    h = precision(0.01)
    kernel = solver_kernel(degenerate_linear_solver, n, h, precision)
    state = cuda.to_device(np.ones(n, dtype=precision))
    rhs_dev = cuda.to_device(np.ones(n, dtype=precision))
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.zeros(2, dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](state, rhs_dev, empty_base, x_dev, flag)
    stream.synchronize()
    code = flag.copy_to_host()[0] & 0xFF
    expected = _DEGENERATE_EXPECTED_STATUS[
        degenerate_linear_solver.linear_correction_type
    ]
    assert code == expected


def test_linear_solver_config_scalar_tolerance_broadcast(precision):
    """Verify scalar krylov_atol/rtol broadcasts to array of length n."""
    n = 5
    solver = MRLinearSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
    )
    assert solver.krylov_atol.shape == (n,)
    assert solver.krylov_rtol.shape == (n,)
    assert np.all(solver.krylov_atol == precision(1e-6))
    assert np.all(solver.krylov_rtol == precision(1e-4))


def test_linear_solver_config_array_tolerance_accepted(precision):
    """Verify array tolerances of correct length are accepted."""
    n = 3
    atol = np.array([1e-6, 1e-8, 1e-4], dtype=precision)
    rtol = np.array([1e-3, 1e-5, 1e-2], dtype=precision)
    solver = MRLinearSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=atol,
        krylov_rtol=rtol,
    )
    assert np.allclose(solver.krylov_atol, atol)
    assert np.allclose(solver.krylov_rtol, rtol)


def test_linear_solver_config_wrong_length_raises(precision):
    """Verify wrong-length tolerance array raises ValueError."""
    n = 3
    wrong_atol = np.array([1e-6, 1e-8], dtype=precision)  # length 2
    with pytest.raises(ValueError, match="tol must have shape"):
        MRLinearSolver(
            precision=precision,
            solver_width=n,
            krylov_atol=wrong_atol,
        )


def test_linear_solver_uses_scaled_norm(precision):
    """Verify MRLinearSolver creates and uses ScaledNorm for convergence."""
    from cubie.integrators.norms import ScaledNorm

    n = 3
    solver = MRLinearSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
    )
    # Verify the norm factory is created
    assert hasattr(solver, "norm")
    assert isinstance(solver.norm, ScaledNorm)
    # Verify the norm factory has correct settings
    assert solver.norm.solver_width == n
    assert solver.norm.precision == precision
    assert np.all(solver.norm.atol == precision(1e-6))
    assert np.all(solver.norm.rtol == precision(1e-4))


def test_linear_solver_tolerance_update_propagates(precision):
    """Verify krylov_atol/krylov_rtol updates propagate to norm factory."""
    n = 3
    solver = MRLinearSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
    )
    # Initial values
    assert np.all(solver.krylov_atol == precision(1e-6))
    assert np.all(solver.krylov_rtol == precision(1e-4))

    # Update tolerances
    new_atol = np.array([1e-8, 1e-7, 1e-9], dtype=precision)
    new_rtol = np.array([1e-5, 1e-6, 1e-4], dtype=precision)
    solver.update(krylov_atol=new_atol, krylov_rtol=new_rtol)

    # Verify properties delegate to norm factory
    assert np.allclose(solver.krylov_atol, new_atol)
    assert np.allclose(solver.krylov_rtol, new_rtol)
    # Verify norm factory was updated
    assert np.allclose(solver.norm.atol, new_atol)
    assert np.allclose(solver.norm.rtol, new_rtol)


def test_linear_solver_config_no_tolerance_fields(precision):
    """Verify MRLinearSolverConfig no longer has krylov_atol/krylov_rtol
    fields.
    """
    from cubie.integrators.matrix_free_solvers.linear_solver import (
        MRLinearSolverConfig,
    )

    config = MRLinearSolverConfig(precision=precision, solver_width=3)

    # These fields should no longer exist on the config
    assert not hasattr(config, "krylov_atol")
    assert not hasattr(config, "krylov_rtol")

    # The legacy scalar tolerance should no longer exist
    assert not hasattr(config, "krylov_tolerance")


def test_linear_solver_config_settings_dict_excludes_tolerance_arrays(
    precision,
):
    """Verify settings_dict does not include tolerance arrays."""
    from cubie.integrators.matrix_free_solvers.linear_solver import (
        MRLinearSolverConfig,
    )

    config = MRLinearSolverConfig(precision=precision, solver_width=3)
    settings = config.settings_dict

    # Tolerance arrays should not be in settings_dict
    assert "krylov_atol" not in settings
    assert "krylov_rtol" not in settings

    # Legacy tolerance should not be in settings_dict
    assert "krylov_tolerance" not in settings

    # Other expected settings should be present
    assert "krylov_max_iters" in settings
    assert "linear_correction_type" in settings
    assert "preconditioned_vec_location" in settings
    assert "temp_location" in settings


def test_unset_max_iters_covers_krylov_space(precision):
    """Unset cap resolves to ceil(1.5 * width) and tracks the width."""
    solver = MRLinearSolver(precision=precision, solver_width=3)
    assert solver.max_iters == 5
    solver.update(solver_width=8, n=8)
    assert solver.max_iters == 12
    solver.update(krylov_max_iters=7)
    assert solver.max_iters == 7
    solver.update(solver_width=20, n=20)
    assert solver.max_iters == 7


def test_linear_solver_inherits_from_matrix_free_solver(precision):
    """Verify MRLinearSolver is instance of MatrixFreeSolver."""
    from cubie.integrators.matrix_free_solvers.base_solver import (
        MatrixFreeSolver,
    )

    solver = MRLinearSolver(
        precision=precision,
        solver_width=3,
    )
    assert isinstance(solver, MatrixFreeSolver)
    assert hasattr(solver, "solver_type")
    assert solver.solver_type == "krylov"


def test_linear_solver_update_preserves_original_dict(precision):
    """Verify update() does not modify the input updates_dict."""
    solver = MRLinearSolver(
        precision=precision,
        solver_width=3,
    )

    # Create an update dict with tolerance values
    original_updates = {
        "krylov_atol": 1e-8,
        "krylov_rtol": 1e-5,
        "krylov_max_iters": 50,
    }
    # Make a copy to compare later
    updates_copy = dict(original_updates)

    # Call update with the dict
    solver.update(updates_dict=original_updates)

    # Verify the original dict was not modified
    assert original_updates == updates_copy


def test_linear_solver_no_manual_cache_invalidation(precision):
    """Verify cache invalidation happens through config update, not manual."""
    solver = MRLinearSolver(
        precision=precision,
        solver_width=3,
        krylov_atol=1e-6,
        krylov_rtol=1e-4,
    )

    # Access device_function to populate cache
    _ = solver.device_function

    # Update tolerance - should update config's norm_device_function
    new_atol = np.array([1e-8, 1e-7, 1e-9], dtype=precision)
    solver.update(krylov_atol=new_atol)

    # Verify config was updated with new norm device function
    config2 = solver.compile_settings
    norm_fn2 = config2.norm_device_function

    # The norm device function should be set (not None)
    assert norm_fn2 is not None

    # Verify the solver's norm factory was updated
    assert np.allclose(solver.norm.atol, new_atol)


def test_linear_solver_settings_dict_includes_tolerance_arrays(precision):
    """Verify settings_dict includes krylov_atol and krylov_rtol from norm."""
    n = 3
    atol = np.array([1e-6, 1e-8, 1e-4], dtype=precision)
    rtol = np.array([1e-3, 1e-5, 1e-2], dtype=precision)
    solver = MRLinearSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=atol,
        krylov_rtol=rtol,
    )
    settings = solver.settings_dict

    # Tolerance arrays should be in settings_dict
    assert "krylov_atol" in settings
    assert "krylov_rtol" in settings
    assert np.allclose(settings["krylov_atol"], atol)
    assert np.allclose(settings["krylov_rtol"], rtol)

    # Other expected settings from config should also be present
    assert "krylov_max_iters" in settings
    assert "linear_correction_type" in settings
    assert "preconditioned_vec_location" in settings
    assert "temp_location" in settings


def test_linear_solver_init_with_krylov_prefixed_kwargs(precision):
    """Verify MRLinearSolver accepts krylov_* kwargs at init via build_config.

    The enhanced build_config with instance_label="krylov" should transform
    krylov_atol/krylov_rtol to atol/rtol for the underlying ScaledNormConfig.
    """
    n = 3
    krylov_atol = np.array([1e-10, 1e-9, 1e-8], dtype=precision)
    krylov_rtol = np.array([1e-5, 1e-4, 1e-3], dtype=precision)

    solver = MRLinearSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=krylov_atol,
        krylov_rtol=krylov_rtol,
    )

    # Verify tolerances reached MRLinearSolver's properties
    assert np.allclose(solver.krylov_atol, krylov_atol)
    assert np.allclose(solver.krylov_rtol, krylov_rtol)

    # Verify tolerances also reached the nested norm factory
    assert np.allclose(solver.norm.atol, krylov_atol)
    assert np.allclose(solver.norm.rtol, krylov_rtol)


def test_linear_solver_forwards_kwargs_to_norm(precision):
    """Verify kwargs passed to MRLinearSolver reach the nested ScaledNorm.

    MRLinearSolver's __init__ now forwards all kwargs to the parent
    MatrixFreeSolver, which creates ScaledNorm with those kwargs.
    """
    n = 3
    atol = np.array([1e-8, 1e-7, 1e-6], dtype=precision)
    rtol = np.array([1e-4, 1e-3, 1e-2], dtype=precision)

    solver = MRLinearSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=atol,
        krylov_rtol=rtol,
    )

    # Verify the norm factory exists and received the tolerances
    assert hasattr(solver, "norm")
    assert solver.norm is not None

    # Verify tolerances propagated through kwargs forwarding
    assert np.allclose(solver.norm.atol, atol)
    assert np.allclose(solver.norm.rtol, rtol)

    # Verify norm has correct instance_label from solver_type
    assert solver.norm.instance_label == "krylov"


@pytest.fixture(scope="session")
def identity_operator(precision):
    """Device operator applying the identity matrix."""

    @cuda.jit(device=True)
    def operator(
        state, parameters, drivers, cached_aux, base_state, t, h,
        a_ij, vec, out,
    ):
        for index in range(out.shape[0]):
            out[index] = vec[index]

    return operator


@pytest.fixture(scope="session")
def residual_reduction_solver(
    request, identity_operator, solver_settings, precision
):
    """Build a solver with a ten percent residual reduction target.

    ``krylov_atol`` of one and a reduction of a tenth are what let a
    five percent warm-start residual accept without moving the
    iterate. ``solver_settings`` is requested so
    ``buffer_registry.reset()`` runs before the solver registers its
    buffers.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Supplies ``linear_correction_type`` through ``param``.
    identity_operator : callable
        Device operator applying the identity matrix.
    solver_settings : dict
        Session-wide solver configuration.
    precision : np.dtype
        Floating point precision for the solver.

    Returns
    -------
    MatrixFreeSolver
        The configured solver instance.
    """
    common = {
        "precision": precision,
        "solver_width": 3,
        "krylov_atol": 1.0,
        "krylov_rtol": 0.0,
        "krylov_max_iters": 8,
        "krylov_residual_reduction": 0.1,
        "krylov_residual_floor": 0.0,
    }
    if request.param == "bicgstab":
        solver = BiCGSTABSolver(**common)
    else:
        solver = MRLinearSolver(
            linear_correction_type=request.param, **common
        )
    solver.update(operator_apply=identity_operator)
    return solver


@pytest.fixture(scope="session")
def residual_reduction_kernel(
    residual_reduction_solver, solver_kernel, precision
):
    """Compile one kernel per residual-reduction solver."""
    return solver_kernel(
        residual_reduction_solver, 3, precision(0.01), precision
    )


@pytest.mark.parametrize(
    "residual_reduction_solver",
    ["minimal_residual", "steepest_descent", "bicgstab"],
    indirect=True,
)
@pytest.mark.parametrize("warm_start", [True, False], ids=["warm", "cold"])
def test_residual_reduction_measures_entry_rhs(
    warm_start,
    residual_reduction_kernel,
    precision,
    tolerance,
):
    """The relative stopping target is fixed from the untouched RHS.

    With the identity operator a warm start at ``0.95 * rhs`` leaves a
    residual of five percent of the right-hand side, inside a ten
    percent reduction target, so the solve accepts without moving the
    iterate. A cold start must iterate to the solution.
    """
    rhs = np.array([10.0, -20.0, 30.0], dtype=precision)
    if warm_start:
        guess = (precision(0.95) * rhs).astype(precision)
    else:
        guess = np.zeros(3, dtype=precision)

    state = cuda.to_device(np.array([2.0, -4.0, 6.0], dtype=precision))
    rhs_dev = cuda.to_device(rhs.copy())
    x_dev = cuda.to_device(guess.copy())
    flag = cuda.to_device(np.zeros(2, dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))

    stream = default_memmgr.get_group_stream()
    residual_reduction_kernel[1, 1, stream](
        state, rhs_dev, empty_base, x_dev, flag
    )
    stream.synchronize()

    assert flag.copy_to_host()[0] & 0xFF == CUBIE_RESULT_CODES.SUCCESS
    if warm_start:
        assert np.array_equal(x_dev.copy_to_host(), guess)
    else:
        assert np.allclose(
            x_dev.copy_to_host(),
            rhs,
            rtol=tolerance.rel_loose,
            atol=tolerance.abs_loose,
        )


def test_residual_settings_derive_and_override(precision):
    """Unset stopping settings derive; explicit values stick and update."""
    solver = MRLinearSolver(precision=precision, solver_width=3)
    derived_floor = precision(float(np.finfo(precision).eps) ** 0.5)
    assert solver.krylov_residual_reduction == precision(
        np.finfo(precision).eps
    )
    assert solver.krylov_residual_floor == derived_floor
    assert solver.settings_dict["krylov_residual_reduction"] == precision(
        np.finfo(precision).eps
    )
    assert solver.settings_dict["krylov_residual_floor"] == derived_floor

    solver = MRLinearSolver(
        precision=precision,
        solver_width=3,
        krylov_residual_reduction=1e-3,
        krylov_residual_floor=0.25,
    )
    assert solver.krylov_residual_reduction == precision(1e-3)
    assert solver.krylov_residual_floor == precision(0.25)

    recognized = solver.update(
        krylov_residual_reduction=5e-3, krylov_residual_floor=0.5
    )
    assert {
        "krylov_residual_reduction",
        "krylov_residual_floor",
    } <= recognized
    assert solver.krylov_residual_reduction == precision(5e-3)
    assert solver.krylov_residual_floor == precision(0.5)


@pytest.mark.parametrize(
    "settings",
    [
        {"krylov_residual_reduction": -0.1},
        {"krylov_residual_reduction": 1.5},
        {"krylov_residual_floor": -0.5},
    ],
    ids=["reduction-negative", "reduction-above-one", "floor-negative"],
)
def test_residual_settings_reject_out_of_range(precision, settings):
    """The reduction stays inside [0, 1] and the floor non-negative."""
    with pytest.raises((ValueError, TypeError)):
        MRLinearSolver(precision=precision, solver_width=3, **settings)


# --- zero_initial_guess: operator-call counts and equivalence -------
@pytest.fixture(scope="session")
def counting_operator(precision):
    """SPD operator counting its applications into ``parameters[0]``."""

    @cuda.jit(device=True)
    def operator(
        state, parameters, drivers, cached_aux, base_state, t, h,
        a_ij, vec, out,
    ):
        parameters[0] += precision(1.0)
        out[0] = precision(4.0) * vec[0] + precision(1.0) * vec[1]
        out[1] = precision(1.0) * vec[0] + precision(3.0) * vec[1]
        out[2] = precision(2.0) * vec[2]

    return operator


def _run_counting_solve(
    solver, counting_solver_kernel, precision, rhs
):
    """Solve from a zero iterate and return (calls, x, status, iters)."""
    h = precision(0.01)
    kernel = counting_solver_kernel(solver, 3, h, precision)
    state = cuda.to_device(
        np.array([1.0, -1.0, 0.5], dtype=precision)
    )
    parameters = cuda.to_device(np.zeros(1, dtype=precision))
    rhs_dev = cuda.to_device(np.asarray(rhs, dtype=precision))
    x_dev = cuda.to_device(np.zeros(3, dtype=precision))
    flag = cuda.to_device(np.zeros(2, dtype=np.int32))
    empty_base = cuda.to_device(np.empty(0, dtype=precision))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](
        state, parameters, rhs_dev, empty_base, x_dev, flag
    )
    stream.synchronize()
    flags = flag.copy_to_host()
    return (
        int(parameters.copy_to_host()[0]),
        x_dev.copy_to_host(),
        flags[0] & 0xFF,
        flags[1],
    )


def _build_counting_solver(
    correction_type, counting_operator, precision, zero_initial_guess
):
    kwargs = dict(
        precision=precision,
        solver_width=3,
        krylov_atol=1e-6,
        krylov_rtol=0.0,
        krylov_max_iters=100,
        zero_initial_guess=zero_initial_guess,
    )
    if correction_type == "bicgstab":
        solver = BiCGSTABSolver(**kwargs)
    else:
        solver = MRLinearSolver(
            linear_correction_type=correction_type, **kwargs
        )
    solver.update(operator_apply=counting_operator)
    return solver


@pytest.mark.parametrize(
    "correction_type", ["minimal_residual", "bicgstab"]
)
def test_zero_guess_skips_one_operator_call(
    correction_type,
    counting_operator,
    counting_solver_kernel,
    solver_settings,
    precision,
):
    """The gate removes one operator call; results match bitwise."""
    rhs = [1.0, 2.0, 3.0]
    solver_plain = _build_counting_solver(
        correction_type, counting_operator, precision, False
    )
    calls_plain, x_plain, status_plain, iters_plain = (
        _run_counting_solve(
            solver_plain, counting_solver_kernel, precision, rhs
        )
    )
    solver_zero = _build_counting_solver(
        correction_type, counting_operator, precision, True
    )
    calls_zero, x_zero, status_zero, iters_zero = _run_counting_solve(
        solver_zero, counting_solver_kernel, precision, rhs
    )
    assert status_plain == CUBIE_RESULT_CODES.SUCCESS
    assert status_zero == CUBIE_RESULT_CODES.SUCCESS
    assert calls_plain == calls_zero + 1
    assert iters_plain == iters_zero
    assert np.array_equal(x_plain, x_zero)


@pytest.mark.parametrize(
    "correction_type", ["minimal_residual", "bicgstab"]
)
def test_zero_guess_initial_convergence_applies_no_operator(
    correction_type,
    counting_operator,
    counting_solver_kernel,
    solver_settings,
    precision,
):
    """A zero right-hand side converges with zero operator calls."""
    solver = _build_counting_solver(
        correction_type, counting_operator, precision, True
    )
    calls, x, status, iters = _run_counting_solve(
        solver, counting_solver_kernel, precision, [0.0, 0.0, 0.0]
    )
    assert status == CUBIE_RESULT_CODES.SUCCESS
    assert calls == 0
    assert iters == 0
    assert np.array_equal(x, np.zeros(3, dtype=precision))


@pytest.fixture(scope="session")
def nonfinite_operator(precision):
    """Diagonal operator with an infinite coefficient."""
    infinite = precision(np.inf)

    @cuda.jit(device=True)
    def operator(
        state, parameters, drivers, cached_aux, base_state, t, h,
        a_ij, vec, out,
    ):
        for index in range(out.shape[0]):
            out[index] = infinite * vec[index]

    return operator


def test_zero_guess_nonfinite_operator_policy_matches_cpu(
    nonfinite_operator,
    counting_solver_kernel,
    solver_settings,
    precision,
):
    """Device and CPU agree on the nonfinite-operator policy."""
    from tests.integrators.cpu_reference.cpu_utils import krylov_solve

    solver = MRLinearSolver(
        precision=precision,
        solver_width=3,
        krylov_atol=1e-6,
        krylov_rtol=0.0,
        krylov_max_iters=8,
        zero_initial_guess=True,
    )
    solver.update(operator_apply=nonfinite_operator)
    calls, x, status, iters = _run_counting_solve(
        solver, counting_solver_kernel, precision, [0.0, 0.0, 0.0]
    )
    assert status == CUBIE_RESULT_CODES.SUCCESS
    assert np.array_equal(x, np.zeros(3, dtype=precision))

    matrix = np.diag(np.full(3, np.inf)).astype(precision)
    rhs = np.zeros(3, dtype=precision)
    _, converged_skip, iters_skip = krylov_solve(
        matrix,
        rhs,
        tolerance=precision(1e-6),
        max_iterations=8,
        precision=precision,
    )
    assert converged_skip
    assert iters_skip == 0

    _, converged_eval, _ = krylov_solve(
        matrix,
        rhs,
        tolerance=precision(1e-6),
        max_iterations=8,
        precision=precision,
        initial_guess=np.zeros(3, dtype=precision),
    )
    assert not converged_eval


def test_zero_guess_update_is_construction_only(precision):
    """update never recognises the constructor-only key."""
    solver = MRLinearSolver(precision=precision, solver_width=3)
    assert solver.compile_settings.zero_initial_guess is False
    assert "zero_initial_guess" not in solver.update(
        zero_initial_guess=True, silent=True
    )
    assert solver.compile_settings.zero_initial_guess is False


@pytest.fixture(scope="session")
def overflow_norm_solver(request, identity_operator, solver_settings,
                         precision):
    """Solver whose floored weights let a large RHS overflow the norm."""
    kwargs = dict(
        precision=precision,
        solver_width=3,
        krylov_atol=0.0,
        krylov_rtol=0.0,
        krylov_max_iters=8,
        zero_initial_guess=True,
    )
    if request.param == "bicgstab":
        solver = BiCGSTABSolver(**kwargs)
    else:
        solver = MRLinearSolver(
            linear_correction_type=request.param, **kwargs
        )
    solver.update(operator_apply=identity_operator)
    return solver


@pytest.fixture(scope="session")
def overflow_norm_kernel(overflow_norm_solver, solver_kernel, precision):
    return solver_kernel(overflow_norm_solver, 3, precision(0.01), precision)


def _run_overflow_kernel(kernel, rhs, precision):
    state = cuda.to_device(np.zeros(3, dtype=precision))
    rhs_dev = cuda.to_device(rhs.copy())
    x_dev = cuda.to_device(np.zeros(3, dtype=precision))
    flag = cuda.to_device(np.zeros(2, dtype=np.int32))
    base = cuda.to_device(np.zeros(3, dtype=precision))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](state, rhs_dev, base, x_dev, flag)
    stream.synchronize()
    return flag.copy_to_host(), x_dev.copy_to_host()


@pytest.mark.parametrize(
    "overflow_norm_solver",
    ["minimal_residual", "bicgstab"],
    indirect=True,
)
def test_overflowed_entry_norm_still_solves(
    overflow_norm_solver, overflow_norm_kernel, precision
):
    """A RHS whose weighted entry norm overflows is solved in one iteration."""
    from tests.integrators.cpu_reference.cpu_utils import krylov_solve

    magnitude = 10.0 * float(np.sqrt(np.finfo(precision).max)) * 1e-16
    rhs = np.full(3, magnitude, dtype=precision)
    flag, x = _run_overflow_kernel(overflow_norm_kernel, rhs, precision)
    assert flag[0] & 0xFF == CUBIE_RESULT_CODES.SUCCESS
    assert flag[1] == 1
    assert np.array_equal(x, rhs)

    solution, converged, iterations = krylov_solve(
        np.eye(3, dtype=precision),
        rhs,
        tolerance=precision(0.0),
        max_iterations=8,
        precision=precision,
        correction_type=overflow_norm_solver.linear_correction_type,
    )
    assert converged is True
    assert iterations == 1
    assert np.array_equal(solution, rhs)


@pytest.mark.parametrize(
    "overflow_norm_solver",
    ["minimal_residual", "bicgstab"],
    indirect=True,
)
def test_unreducible_overflowed_norm_fails_the_solve(
    overflow_norm_solver, overflow_norm_kernel, precision
):
    """A RHS the iteration cannot reduce ends in a failure status."""
    from tests.integrators.cpu_reference.cpu_utils import krylov_solve

    magnitude = float(np.finfo(precision).max) * 1e-8
    rhs = np.full(3, magnitude, dtype=precision)
    flag, x = _run_overflow_kernel(overflow_norm_kernel, rhs, precision)
    assert (
        flag[0] & 0xFF
        == CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED
    )
    assert flag[1] == 8

    _, converged, iterations = krylov_solve(
        np.eye(3, dtype=precision),
        rhs,
        tolerance=precision(0.0),
        max_iterations=8,
        precision=precision,
        correction_type=overflow_norm_solver.linear_correction_type,
    )
    assert converged is False
    assert iterations == 8
