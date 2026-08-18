import numpy as np
import pytest
from cubie.cuda_simsafe import cuda
from cubie.memory import default_memmgr
from numpy.testing import assert_allclose

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
