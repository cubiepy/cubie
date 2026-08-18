import numpy as np
import pytest
from cubie.cuda_simsafe import cuda
from cubie.memory import default_memmgr
from numpy.testing import assert_allclose

from cubie.integrators.algorithms.generic_dirk_tableaus import (
    KVAERNO3_TABLEAU,
)
from cubie.integrators.algorithms.generic_firk_tableaus import (
    RADAU_IIA_5_TABLEAU,
)
from cubie.integrators.matrix_free_solvers import CUBIE_RESULT_CODES
from cubie.integrators.matrix_free_solvers.lu_solver import (
    LUSolver,
    LUSolverConfig,
)

_LU_SETTINGS = {
    "linear_correction_type": "lu",
    "krylov_atol": 1e-6,
    "krylov_rtol": 1e-6,
    "krylov_max_iters": 1,
}


# Direct solves reach the dense reference in one reported iteration.
@pytest.mark.parametrize(
    "system_setup",
    ["linear", "coupled_linear", "coupled_nonlinear", "stiff"],
    indirect=True,
)
@pytest.mark.parametrize(
    "matrixfree_settings_override",
    [_LU_SETTINGS],
    ids=["lu"],
    indirect=True,
)
def test_lu_solver_symbolic(
    system_setup,
    linear_solver_instance,
    solver_kernel,
    precision,
    tolerance,
):
    """The generated direct solve matches the dense reference."""
    n = system_setup["n"]
    rhs_vec = system_setup["mr_rhs"]
    expected = system_setup["mr_expected"]
    h = system_setup["h"]

    kernel = solver_kernel(linear_solver_instance, n, h, precision)
    state = system_setup["state_init"]
    rhs_dev = cuda.to_device(rhs_vec.copy())
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.zeros(2, dtype=np.int32))
    base_state = system_setup["base_state"]
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](state, rhs_dev, base_state, x_dev, flag)
    stream.synchronize()
    status, iters = flag.copy_to_host()
    assert status == CUBIE_RESULT_CODES.SUCCESS
    assert iters == 1
    # rhs is read-only for the direct solve.
    assert np.array_equal(rhs_dev.copy_to_host(), rhs_vec)
    assert_allclose(
        x_dev.copy_to_host(),
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


@pytest.mark.parametrize("system_setup", ["linear"], indirect=True)
@pytest.mark.parametrize(
    "matrixfree_settings_override",
    [_LU_SETTINGS],
    ids=["lu"],
    indirect=True,
)
def test_lu_solver_singular_pivot(
    system_setup,
    linear_solver_instance,
    solver_kernel,
    precision,
):
    """A singular shifted matrix reports SINGULAR_PIVOT.

    The linear system's Jacobian is ``0.5 * I``, so at ``a_ij = 1``
    and ``h = 2`` the shifted matrix ``I - a_ij*h*J`` is zero.
    """
    n = system_setup["n"]
    kernel = solver_kernel(
        linear_solver_instance, n, precision(2.0), precision
    )
    state = cuda.to_device(np.zeros(n, dtype=precision))
    rhs_dev = cuda.to_device(np.ones(n, dtype=precision))
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.zeros(2, dtype=np.int32))
    base_state = cuda.to_device(np.zeros(n, dtype=precision))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](state, rhs_dev, base_state, x_dev, flag)
    stream.synchronize()
    status, iters = flag.copy_to_host()
    assert status == CUBIE_RESULT_CODES.SINGULAR_PIVOT
    assert iters == 1


def test_lu_solver_forces_zero_initial_guess(precision):
    """The config declares a zero guess whatever the caller passes."""
    solver = LUSolver(
        precision=precision,
        solver_width=3,
        zero_initial_guess=False,
    )
    assert solver.compile_settings.zero_initial_guess is True


def test_lu_solver_settings_dict_round_trips(precision):
    """settings_dict carries the keys a hot-swap rebuild consumes."""
    solver = LUSolver(
        precision=precision,
        solver_width=3,
        krylov_max_iters=25,
    )
    settings = solver.settings_dict
    assert settings["linear_correction_type"] == "lu"
    assert settings["krylov_max_iters"] == 25
    assert settings["zero_initial_guess"] is True
    assert "krylov_atol" in settings
    assert "krylov_rtol" in settings
    assert "krylov_residual_reduction" in settings
    assert "krylov_residual_floor" in settings


def test_lu_solver_config_lu_nnz_sizes_factor_buffer(precision):
    """The lu_nnz field sizes the registered factor buffer."""
    from cubie.buffer_registry import buffer_registry

    solver = LUSolver(precision=precision, solver_width=3)
    assert solver.compile_settings.lu_nnz == 0
    recognized = solver.update(lu_nnz=7)
    assert "lu_nnz" in recognized
    assert solver.compile_settings.lu_nnz == 7
    assert buffer_registry.local_buffer_size(solver) >= 7


def test_lu_solver_config_defaults(precision):
    """Config exposes the direct-solve fields with inert defaults."""
    config = LUSolverConfig(precision=precision, solver_width=3)
    assert config.lu_solve_function is None
    assert config.lu_nnz == 0
    settings = config.settings_dict
    assert settings["linear_correction_type"] == "lu"



def _dense_tolerances(precision):
    """FD-limited tolerances for dense-reference comparisons."""
    if np.dtype(precision) == np.float64:
        return {"rtol": 1e-7, "atol": 1e-9}
    return {"rtol": 2e-3, "atol": 2e-3}


def _finite_difference_jacobian(system_setup, precision, y):
    """Dense Jacobian of the compiled dxdt at ``y`` by central FD."""
    n = system_setup["n"]
    dxdt_func = system_setup["sym_system"].evaluate_f

    @cuda.jit
    def dxdt_kernel(state, params, drivers, observables, deriv, t):
        dxdt_func(state, params, drivers, observables, deriv, t)

    params = np.zeros(1, dtype=precision)
    drivers = np.zeros(1, dtype=precision)
    observables = np.zeros(n, dtype=precision)
    eps = float(np.cbrt(np.finfo(precision).eps)) * max(
        1.0, float(np.max(np.abs(y)))
    )
    jacobian = np.zeros((n, n), dtype=np.float64)
    zero_time = precision(0.0)
    for column in range(n):
        plus = np.asarray(y, dtype=precision).copy()
        minus = plus.copy()
        plus[column] += precision(eps)
        minus[column] -= precision(eps)
        deriv_plus = np.zeros(n, dtype=precision)
        deriv_minus = np.zeros(n, dtype=precision)
        dxdt_kernel[1, 1](
            plus, params, drivers, observables, deriv_plus, zero_time
        )
        dxdt_kernel[1, 1](
            minus, params, drivers, observables, deriv_minus, zero_time
        )
        jacobian[:, column] = (
            deriv_plus.astype(np.float64) - deriv_minus
        ) / (2.0 * eps)
    return jacobian


def _evaluation_state(system_setup):
    """Return the fixture's converged evaluation state on the host."""
    return (
        system_setup["base_state"].copy_to_host()
        + system_setup["state_init"].copy_to_host()
    )


@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
def test_lu_stacked_solve_matches_dense(system_setup, precision):
    """The coupled all-stages solve matches a dense FD reference."""
    n = system_setup["n"]
    h = system_setup["h"]
    tableau = RADAU_IIA_5_TABLEAU
    a_matrix = np.asarray(tableau.stage_coefficients)
    s = tableau.stage_count
    member = system_setup["sym_system"].get_solver_helper(
        "lu_solve",
        variant="stacked_stages",
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    lu_solve = member.device_function

    rng = np.random.default_rng(11)
    base_host = system_setup["base_state"].copy_to_host()
    increment = (
        rng.normal(size=s * n).astype(precision) * precision(0.05)
    )
    rhs = rng.normal(size=s * n).astype(precision)
    x_out = np.zeros(s * n, dtype=precision)
    factor = np.zeros(max(member.lu_nnz, 1), dtype=precision)
    flag = np.zeros(1, dtype=np.int32)
    params = np.zeros(1, dtype=precision)
    drivers = np.zeros(s, dtype=precision)

    @cuda.jit
    def kernel(state, params, drivers, base_state, rhs, x, factor, flag):
        flag[0] = lu_solve(
            state,
            params,
            drivers,
            base_state,
            precision(0.0),
            h,
            precision(0.0),
            rhs,
            x,
            factor,
        )

    kernel[1, 1](
        increment, params, drivers, base_host, rhs, x_out, factor, flag
    )
    assert flag[0] == 0

    coupled = np.zeros((s * n, s * n), dtype=np.float64)
    for row_stage in range(s):
        stage_state = base_host.astype(np.float64).copy()
        for k in range(s):
            stage_state += a_matrix[row_stage, k] * increment[
                k * n : (k + 1) * n
            ].astype(np.float64)
        stage_jac = _finite_difference_jacobian(
            system_setup, precision, stage_state
        )
        for col_stage in range(s):
            block = -float(h) * a_matrix[row_stage, col_stage] * stage_jac
            coupled[
                row_stage * n : (row_stage + 1) * n,
                col_stage * n : (col_stage + 1) * n,
            ] = block
        coupled[
            row_stage * n : (row_stage + 1) * n,
            row_stage * n : (row_stage + 1) * n,
        ] += np.eye(n)
    expected = np.linalg.solve(coupled, rhs.astype(np.float64))
    assert_allclose(x_out, expected, **_dense_tolerances(precision))


@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
def test_lu_prefactored_solve_matches_dense(system_setup, precision):
    """Prefactored substitution matches dense solves per diagonal."""
    n = system_setup["n"]
    h = system_setup["h"]
    tableau = KVAERNO3_TABLEAU
    member = system_setup["sym_system"].get_solver_helper(
        "lu_solve",
        variant="prefactored",
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    lu_solve = member.device_function
    prepare = member.prepare_jac
    assert member.lu_nnz == 0
    assert member.cached_auxiliary_count > 0

    state = _evaluation_state(system_setup).astype(precision)
    jacobian = _finite_difference_jacobian(
        system_setup, precision, state
    )
    diagonals = sorted(
        {
            float(row[idx])
            for idx, row in enumerate(tableau.stage_coefficients)
            if row[idx] != 0.0
        }
    )
    rng = np.random.default_rng(12)
    cached = np.zeros(member.cached_auxiliary_count, dtype=precision)
    params = np.zeros(1, dtype=precision)
    drivers = np.zeros(1, dtype=precision)
    factor = np.zeros(1, dtype=precision)

    for diagonal in diagonals:
        rhs = rng.normal(size=n).astype(precision)
        x_out = np.zeros(n, dtype=precision)
        flag = np.zeros(2, dtype=np.int32)
        a_ij = precision(diagonal)

        @cuda.jit
        def kernel(
            state, params, drivers, cached, rhs, x, factor, flag
        ):
            flag[0] = prepare(
                state, params, drivers, precision(0.0), h, cached
            )
            flag[1] = lu_solve(
                state,
                params,
                drivers,
                cached,
                state,
                precision(0.0),
                h,
                a_ij,
                rhs,
                x,
                factor,
            )

        kernel[1, 1](
            state, params, drivers, cached, rhs, x_out, factor, flag
        )
        assert flag[0] == 0
        assert flag[1] == 0
        dense = np.eye(n) - float(h) * diagonal * jacobian
        expected = np.linalg.solve(dense, rhs.astype(np.float64))
        assert_allclose(
            x_out, expected, **_dense_tolerances(precision)
        )


@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
def test_lu_transformed_solve_matches_dense(system_setup, precision):
    """The block-transform solve matches the dense frozen-J system."""
    n = system_setup["n"]
    h = system_setup["h"]
    tableau = RADAU_IIA_5_TABLEAU
    a_matrix = np.asarray(tableau.stage_coefficients)
    s = tableau.stage_count
    member = system_setup["sym_system"].get_solver_helper(
        "lu_solve",
        variant="cached_stacked",
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    lu_solve = member.device_function
    prepare = member.prepare_jac
    assert member.lu_nnz == 0
    assert member.cached_auxiliary_count > 0

    state = _evaluation_state(system_setup).astype(precision)
    jacobian = _finite_difference_jacobian(
        system_setup, precision, state
    )
    rng = np.random.default_rng(13)
    rhs = rng.normal(size=s * n).astype(precision)
    x_out = np.zeros(s * n, dtype=precision)
    cached = np.zeros(member.cached_auxiliary_count, dtype=precision)
    params = np.zeros(1, dtype=precision)
    drivers = np.zeros(1, dtype=precision)
    factor = np.zeros(1, dtype=precision)
    flag = np.zeros(2, dtype=np.int32)

    @cuda.jit
    def kernel(state, params, drivers, cached, rhs, x, factor, flag):
        flag[0] = prepare(
            state, params, drivers, precision(0.0), h, cached
        )
        flag[1] = lu_solve(
            state,
            params,
            drivers,
            cached,
            state,
            precision(0.0),
            h,
            precision(0.0),
            rhs,
            x,
            factor,
        )

    kernel[1, 1](
        state, params, drivers, cached, rhs, x_out, factor, flag
    )
    assert flag[0] == 0
    assert flag[1] == 0

    coupled = np.kron(np.eye(s), np.eye(n)) - float(h) * np.kron(
        a_matrix, jacobian
    )
    expected = np.linalg.solve(coupled, rhs.astype(np.float64))
    assert_allclose(x_out, expected, **_dense_tolerances(precision))


@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
def test_lu_smoothing_solve_matches_dense(system_setup, precision):
    """The smoothing solve matches the dense (I - g*h*J) system."""
    n = system_setup["n"]
    h = system_setup["h"]
    tableau = RADAU_IIA_5_TABLEAU
    member = system_setup["sym_system"].get_solver_helper(
        "lu_smoothing_solve",
        variant="cached_stacked",
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    smoothing_solve = member.device_function
    prepare = member.prepare_jac
    gamma_smooth = tableau.smoothing_gamma
    assert gamma_smooth > 0.0

    state = _evaluation_state(system_setup).astype(precision)
    jacobian = _finite_difference_jacobian(
        system_setup, precision, state
    )
    rng = np.random.default_rng(14)
    rhs = rng.normal(size=n).astype(precision)
    x_out = np.zeros(n, dtype=precision)
    cached = np.zeros(member.cached_auxiliary_count, dtype=precision)
    params = np.zeros(1, dtype=precision)
    drivers = np.zeros(1, dtype=precision)
    factor = np.zeros(1, dtype=precision)
    flag = np.zeros(1, dtype=np.int32)
    g_typed = precision(gamma_smooth)

    @cuda.jit
    def kernel(state, params, drivers, cached, rhs, x, factor, flag):
        prepare(state, params, drivers, precision(0.0), h, cached)
        flag[0] = smoothing_solve(
            state,
            params,
            drivers,
            cached,
            state,
            precision(0.0),
            h,
            g_typed,
            rhs,
            x,
            factor,
        )

    kernel[1, 1](
        state, params, drivers, cached, rhs, x_out, factor, flag
    )
    assert flag[0] == 0

    dense = np.eye(n) - gamma_smooth * float(h) * jacobian
    expected = np.linalg.solve(dense, rhs.astype(np.float64))
    assert_allclose(x_out, expected, **_dense_tolerances(precision))
