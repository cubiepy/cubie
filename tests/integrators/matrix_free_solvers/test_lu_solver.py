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
    stream = default_memmgr.get_group_stream()
    rhs_dev = cuda.to_device(rhs_vec.copy(), stream=stream)
    x_dev = cuda.to_device(
        np.zeros(n, dtype=precision), stream=stream
    )
    flag = cuda.to_device(
        np.zeros(2, dtype=np.int32), stream=stream
    )
    base_state = system_setup["base_state"]
    kernel[1, 1, stream](state, rhs_dev, base_state, x_dev, flag)
    flag_result = flag.copy_to_host(stream=stream)
    rhs_result = rhs_dev.copy_to_host(stream=stream)
    x_result = x_dev.copy_to_host(stream=stream)
    stream.synchronize()
    status, iters = flag_result
    assert status == CUBIE_RESULT_CODES.SUCCESS
    assert iters == 1
    # rhs is read-only for the direct solve.
    assert np.array_equal(rhs_result, rhs_vec)
    assert_allclose(
        x_result,
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
    stream = default_memmgr.get_group_stream()
    state = cuda.to_device(
        np.zeros(n, dtype=precision), stream=stream
    )
    rhs_dev = cuda.to_device(
        np.ones(n, dtype=precision), stream=stream
    )
    x_dev = cuda.to_device(
        np.zeros(n, dtype=precision), stream=stream
    )
    flag = cuda.to_device(
        np.zeros(2, dtype=np.int32), stream=stream
    )
    base_state = cuda.to_device(
        np.zeros(n, dtype=precision), stream=stream
    )
    kernel[1, 1, stream](state, rhs_dev, base_state, x_dev, flag)
    flag_result = flag.copy_to_host(stream=stream)
    stream.synchronize()
    status, iters = flag_result
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
        lu_factor_location="shared",
    )
    settings = solver.settings_dict
    assert settings["linear_correction_type"] == "lu"
    assert settings["zero_initial_guess"] is True
    assert settings["lu_factor_location"] == "shared"


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



def _dense_jacobian(system_setup, y):
    """Return the fixture system's analytic Jacobian at ``y``."""
    y = np.asarray(y, dtype=np.float64)
    system = system_setup["id"]
    if system == "linear":
        return np.diag([0.5, 0.5, 0.5])
    if system == "graded":
        return np.diag([10.0, 50.0, 90.0])
    if system == "stiff":
        return np.diag([1e-6, 0.5, 1e6])
    if system == "coupled_linear":
        return np.array(
            [
                [0.5, 0.1, 0.0],
                [0.2, 0.3, 0.0],
                [0.1, 0.2, 0.4],
            ]
        )
    if system == "coupled_nonlinear":
        return np.array(
            [
                [0.5, -2.0 * y[1], 0.0],
                [y[1], y[0] - 3.0 * y[1] ** 2, 0.0],
                [1.0, 2.0 * y[1], -2.0 * y[2]],
            ]
        )
    raise ValueError(f"Unknown system: {system}")


def _evaluation_state(system_setup):
    """Return the fixture's converged evaluation state on the host."""
    stream = default_memmgr.get_group_stream()
    base = system_setup["base_state"].copy_to_host(stream=stream)
    increment = system_setup["state_init"].copy_to_host(stream=stream)
    stream.synchronize()
    return base + increment


@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
def test_lu_stacked_solve_matches_dense(
    system_setup, precision, tolerance
):
    """The coupled all-stages solve matches the dense reference."""
    n = system_setup["n"]
    h = system_setup["h"]
    tableau = RADAU_IIA_5_TABLEAU
    a_matrix = np.asarray(tableau.stage_coefficients)
    s = tableau.stage_count
    member = system_setup["sym_system"].get_solver_helper(
        "lu_solve",
        jacobian_at="stage",
        stacked=True,
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    lu_solve = member.device_function

    stream = default_memmgr.get_group_stream()
    rng = np.random.default_rng(11)
    base_host = system_setup["base_state"].copy_to_host(stream=stream)
    stream.synchronize()
    increment = (
        rng.normal(size=s * n).astype(precision) * precision(0.05)
    )
    rhs = rng.normal(size=s * n).astype(precision)
    state_dev = cuda.to_device(increment, stream=stream)
    params = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    drivers = cuda.to_device(
        np.zeros(s, dtype=precision), stream=stream
    )
    base_dev = cuda.to_device(base_host, stream=stream)
    rhs_dev = cuda.to_device(rhs, stream=stream)
    x_dev = cuda.to_device(
        np.zeros(s * n, dtype=precision), stream=stream
    )
    factor = cuda.to_device(
        np.zeros(max(member.lu_nnz, 1), dtype=precision),
        stream=stream,
    )
    flag = cuda.to_device(
        np.zeros(1, dtype=np.int32), stream=stream
    )

    @cuda.jit
    def kernel(state, params, drivers, base_state, rhs, x, factor, flag):
        cached_aux = cuda.local.array(1, precision)
        flag[0] = lu_solve(
            state,
            params,
            drivers,
            cached_aux,
            base_state,
            precision(0.0),
            h,
            precision(0.0),
            rhs,
            x,
            factor,
        )

    kernel[1, 1, stream](
        state_dev, params, drivers, base_dev, rhs_dev, x_dev, factor,
        flag,
    )
    flag_result = flag.copy_to_host(stream=stream)
    x_out = x_dev.copy_to_host(stream=stream)
    stream.synchronize()
    assert flag_result[0] == 0

    coupled = np.zeros((s * n, s * n), dtype=np.float64)
    for row_stage in range(s):
        stage_state = base_host.astype(np.float64).copy()
        for k in range(s):
            stage_state += a_matrix[row_stage, k] * increment[
                k * n : (k + 1) * n
            ].astype(np.float64)
        stage_jac = _dense_jacobian(system_setup, stage_state)
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
    assert_allclose(
        x_out,
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
def test_lu_prefactored_solve_matches_dense(
    system_setup, precision, tolerance
):
    """Prefactored substitution matches dense solves per diagonal."""
    n = system_setup["n"]
    h = system_setup["h"]
    tableau = KVAERNO3_TABLEAU
    member = system_setup["sym_system"].get_solver_helper(
        "lu_solve",
        jacobian_at="step",
        prefactored=True,
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    lu_solve = member.device_function
    prepare = member.prepare_jac
    assert member.lu_nnz == 0
    assert member.cached_auxiliary_count > 0

    state = _evaluation_state(system_setup).astype(precision)
    jacobian = _dense_jacobian(system_setup, state)
    diagonals = sorted(
        {
            float(row[idx])
            for idx, row in enumerate(tableau.stage_coefficients)
            if row[idx] != 0.0
        }
    )
    stream = default_memmgr.get_group_stream()
    rng = np.random.default_rng(12)
    state_dev = cuda.to_device(state, stream=stream)
    cached = cuda.to_device(
        np.zeros(member.cached_auxiliary_count, dtype=precision),
        stream=stream,
    )
    params = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    drivers = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    factor = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )

    @cuda.jit
    def kernel(
        state, params, drivers, cached, a_ij, rhs, x, factor, flag
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

    for diagonal in diagonals:
        rhs = rng.normal(size=n).astype(precision)
        rhs_dev = cuda.to_device(rhs, stream=stream)
        x_dev = cuda.to_device(
            np.zeros(n, dtype=precision), stream=stream
        )
        flag = cuda.to_device(
            np.zeros(2, dtype=np.int32), stream=stream
        )
        kernel[1, 1, stream](
            state_dev, params, drivers, cached, precision(diagonal),
            rhs_dev, x_dev, factor, flag,
        )
        flag_result = flag.copy_to_host(stream=stream)
        x_out = x_dev.copy_to_host(stream=stream)
        stream.synchronize()
        assert flag_result[0] == 0
        assert flag_result[1] == 0
        dense = np.eye(n) - float(h) * diagonal * jacobian
        expected = np.linalg.solve(dense, rhs.astype(np.float64))
        assert_allclose(
            x_out,
            expected,
            rtol=tolerance.rel_loose,
            atol=tolerance.abs_loose,
        )


@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
def test_lu_transformed_solve_matches_dense(
    system_setup, precision, tolerance
):
    """The block-transform solve matches the dense frozen-J system."""
    n = system_setup["n"]
    h = system_setup["h"]
    tableau = RADAU_IIA_5_TABLEAU
    a_matrix = np.asarray(tableau.stage_coefficients)
    s = tableau.stage_count
    member = system_setup["sym_system"].get_solver_helper(
        "lu_solve",
        jacobian_at="step",
        prefactored=True,
        stacked=True,
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    lu_solve = member.device_function
    prepare = member.prepare_jac
    assert member.lu_nnz == 0
    assert member.cached_auxiliary_count > 0

    state = _evaluation_state(system_setup).astype(precision)
    jacobian = _dense_jacobian(system_setup, state)
    stream = default_memmgr.get_group_stream()
    rng = np.random.default_rng(13)
    rhs = rng.normal(size=s * n).astype(precision)
    state_dev = cuda.to_device(state, stream=stream)
    cached = cuda.to_device(
        np.zeros(member.cached_auxiliary_count, dtype=precision),
        stream=stream,
    )
    params = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    drivers = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    factor = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    rhs_dev = cuda.to_device(rhs, stream=stream)
    x_dev = cuda.to_device(
        np.zeros(s * n, dtype=precision), stream=stream
    )
    flag = cuda.to_device(
        np.zeros(2, dtype=np.int32), stream=stream
    )

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

    kernel[1, 1, stream](
        state_dev, params, drivers, cached, rhs_dev, x_dev, factor,
        flag,
    )
    flag_result = flag.copy_to_host(stream=stream)
    x_out = x_dev.copy_to_host(stream=stream)
    stream.synchronize()
    assert flag_result[0] == 0
    assert flag_result[1] == 0

    coupled = np.kron(np.eye(s), np.eye(n)) - float(h) * np.kron(
        a_matrix, jacobian
    )
    expected = np.linalg.solve(coupled, rhs.astype(np.float64))
    assert_allclose(
        x_out,
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
def test_lu_smoothing_solve_matches_dense(
    system_setup, precision, tolerance
):
    """The smoothing solve matches the dense (I - g*h*J) system."""
    n = system_setup["n"]
    h = system_setup["h"]
    tableau = RADAU_IIA_5_TABLEAU
    member = system_setup["sym_system"].get_solver_helper(
        "lu_smoothing_solve",
        jacobian_at="step",
        prefactored=True,
        stacked=True,
        stage_coefficients=tableau.stage_coefficients,
        stage_nodes=tableau.stage_nodes,
    )
    smoothing_solve = member.device_function
    prepare = member.prepare_jac
    gamma_smooth = tableau.smoothing_gamma
    assert gamma_smooth > 0.0

    state = _evaluation_state(system_setup).astype(precision)
    jacobian = _dense_jacobian(system_setup, state)
    stream = default_memmgr.get_group_stream()
    rng = np.random.default_rng(14)
    rhs = rng.normal(size=n).astype(precision)
    state_dev = cuda.to_device(state, stream=stream)
    cached = cuda.to_device(
        np.zeros(member.cached_auxiliary_count, dtype=precision),
        stream=stream,
    )
    params = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    drivers = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    factor = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    rhs_dev = cuda.to_device(rhs, stream=stream)
    x_dev = cuda.to_device(
        np.zeros(n, dtype=precision), stream=stream
    )
    flag = cuda.to_device(
        np.zeros(1, dtype=np.int32), stream=stream
    )
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

    kernel[1, 1, stream](
        state_dev, params, drivers, cached, rhs_dev, x_dev, factor,
        flag,
    )
    flag_result = flag.copy_to_host(stream=stream)
    x_out = x_dev.copy_to_host(stream=stream)
    stream.synchronize()
    assert flag_result[0] == 0

    dense = np.eye(n) - gamma_smooth * float(h) * jacobian
    expected = np.linalg.solve(dense, rhs.astype(np.float64))
    assert_allclose(
        x_out,
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )


@pytest.mark.parametrize(
    "system_setup", ["coupled_nonlinear"], indirect=True
)
def test_lu_baked_diagonal_governs_solve(
    system_setup, precision, tolerance
):
    """A baked a_ij literal governs the solve over the runtime value."""
    n = system_setup["n"]
    h = system_setup["h"]
    baked = 0.25
    member = system_setup["sym_system"].get_solver_helper(
        "lu_solve",
        jacobian_at="state",
        a_ij=baked,
    )
    lu_solve = member.device_function

    state = _evaluation_state(system_setup).astype(precision)
    jacobian = _dense_jacobian(system_setup, state)
    stream = default_memmgr.get_group_stream()
    rng = np.random.default_rng(15)
    rhs = rng.normal(size=n).astype(precision)
    state_dev = cuda.to_device(state, stream=stream)
    params = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    drivers = cuda.to_device(
        np.zeros(1, dtype=precision), stream=stream
    )
    rhs_dev = cuda.to_device(rhs, stream=stream)
    x_dev = cuda.to_device(
        np.zeros(n, dtype=precision), stream=stream
    )
    factor = cuda.to_device(
        np.zeros(max(member.lu_nnz, 1), dtype=precision),
        stream=stream,
    )
    flag = cuda.to_device(
        np.zeros(1, dtype=np.int32), stream=stream
    )

    @cuda.jit
    def kernel(state, params, drivers, rhs, x, factor, flag):
        cached_aux = cuda.local.array(1, precision)
        flag[0] = lu_solve(
            state,
            params,
            drivers,
            cached_aux,
            state,
            precision(0.0),
            h,
            precision(0.0),
            rhs,
            x,
            factor,
        )

    kernel[1, 1, stream](
        state_dev, params, drivers, rhs_dev, x_dev, factor, flag
    )
    flag_result = flag.copy_to_host(stream=stream)
    x_out = x_dev.copy_to_host(stream=stream)
    stream.synchronize()
    assert flag_result[0] == 0

    dense = np.eye(n) - baked * float(h) * jacobian
    expected = np.linalg.solve(dense, rhs.astype(np.float64))
    assert_allclose(
        x_out,
        expected,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
