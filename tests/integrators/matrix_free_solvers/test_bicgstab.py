"""BiCGSTAB buffer placement settings and the cached-aux signature."""

import numpy as np
import pytest
from cubie.cuda_simsafe import cuda
from cubie.memory import default_memmgr
from numpy.testing import assert_allclose

from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
    BiCGSTABSolver,
    BiCGSTABSolverConfig,
)
from cubie.result_codes import CUBIE_RESULT_CODES


def test_bicgstab_linear_correction_type_is_bicgstab():
    """linear_correction_type always reports 'bicgstab'."""
    solver = BiCGSTABSolver(precision=np.float32, solver_width=3)
    assert solver.linear_correction_type == "bicgstab"


def test_bicgstab_settings_dict_reports_config_and_locations():
    """settings_dict exposes iteration limit and buffer placements."""
    solver = BiCGSTABSolver(
        precision=np.float32, solver_width=3, krylov_max_iters=42,
    )
    settings = solver.compile_settings.settings_dict
    assert settings["krylov_max_iters"] == 42
    assert settings["linear_correction_type"] == "bicgstab"
    assert settings["r0_hat_location"] == "local"
    assert settings["p_location"] == solver.compile_settings.p_location
    assert settings["v_location"] == solver.compile_settings.v_location
    assert settings["tmp_location"] == solver.compile_settings.tmp_location
    assert (
        settings["s_hat_location"]
        == solver.compile_settings.s_hat_location
    )


def test_bicgstab_unset_max_iters_covers_krylov_space():
    """Unset cap resolves to ceil(1.5 * width); settings keep it unset."""
    solver = BiCGSTABSolver(precision=np.float32, solver_width=6)
    assert solver.max_iters == 9
    settings = solver.compile_settings.settings_dict
    assert settings["krylov_max_iters"] is None


@pytest.mark.parametrize("build_precision", [np.float64, np.float16])
def test_bicgstab_build_selects_precision_specific_thresholds(
    build_precision,
):
    """build() compiles for float64 and non-float32/64 precisions,

    exercising the elif/else branches of the breakdown-threshold
    selection (float32 is covered by the ``precision``-fixture tests
    above).
    """
    solver = BiCGSTABSolver(precision=build_precision, solver_width=3)
    device_fn = solver.device_function
    assert callable(device_fn)


@pytest.mark.parametrize("solver_width", [8, 200])
def test_bicgstab_r0_hat_defaults_local(solver_width):
    """r0_hat is a local buffer at every width unless placed."""
    config = BiCGSTABSolverConfig(
        precision=np.float32, solver_width=solver_width
    )
    assert config.r0_hat_location == "local"


def test_bicgstab_r0_hat_placement_reaches_registry():
    """An explicit r0_hat_location registers the buffer there."""
    from cubie.buffer_registry import buffer_registry

    solver = BiCGSTABSolver(
        precision=np.float32, solver_width=8, r0_hat_location="shared"
    )
    entry = buffer_registry._groups[solver].entries["bicg_r0_hat"]
    assert entry.location == "shared"
    assert entry.size == 8


# --- Cached-auxiliaries path (Rosenbrock-W selects this) -------------
# The operator and preconditioner use the cached signature, taking
# cached_aux immediately after drivers. Here cached_aux carries the
# diagonal of A, so a correct solve proves cached_aux is threaded
# through the solver rather than ignored.
@cuda.jit(device=True, inline=True)
def _cached_diag_operator(
    state, parameters, drivers, cached_aux, base_state,
    t, h, a_ij, vin, vout,
):
    for i in range(vin.shape[0]):
        vout[i] = cached_aux[i] * vin[i]


@cuda.jit(device=True, inline=True)
def _cached_jacobi_precond(
    state, parameters, drivers, cached_aux, base_state,
    t, h, a_ij, rhs, out, temp,
):
    for i in range(rhs.shape[0]):
        out[i] = rhs[i] / cached_aux[i]


def _cached_solver_kernel(n, precision):
    """Kernel that invokes a solver with the cached-aux signature."""
    scratch_size = 2 * n

    def factory(solver, h):
        @cuda.jit
        def kernel(state_init, rhs, base_state, cached_aux, x, flag):
            time_scalar = precision(0.0)
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            shared = cuda.shared.array(scratch_size, dtype=precision)
            persistent_local = cuda.local.array(
                scratch_size, dtype=precision
            )
            counters = cuda.local.array(1, np.int32)
            flag[0] = solver(
                state, parameters, drivers, base_state, cached_aux,
                time_scalar, h, precision(1.0), rhs, x, shared,
                persistent_local, counters,
            )

        return kernel

    return factory


@pytest.mark.parametrize("with_precond", [False, True])
def test_bicgstab_cached_auxiliaries(precision, tolerance, with_precond):
    """Cached-aux BiCGSTAB compiles and solves (Rosenbrock-W path).

    Regression for the signature mismatch that made
    ``linear_correction_type='bicgstab'`` unusable with Rosenbrock-W:
    the solver received a cached-signature operator but emitted a
    non-cached call site.
    """
    n = 3
    diag = np.array([4.0, 5.0, 6.0], dtype=precision)
    rhs = np.array([1.0, 1.0, 1.0], dtype=precision)

    solver = BiCGSTABSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=1e-8,
        krylov_rtol=1e-8,
        krylov_max_iters=200,
    )
    solver.update(
        operator_apply=_cached_diag_operator,
        preconditioner=_cached_jacobi_precond if with_precond else None,
    )
    solver_fn = solver.device_function

    kernel = _cached_solver_kernel(n, precision)(solver_fn, precision(0.01))
    state = cuda.to_device(np.zeros(n, dtype=precision))
    base = cuda.to_device(np.zeros(n, dtype=precision))
    aux = cuda.to_device(diag)
    rhs_dev = cuda.to_device(rhs.copy())
    x_dev = cuda.to_device(np.zeros(n, dtype=precision))
    flag = cuda.to_device(np.array([0], dtype=np.int32))

    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](state, rhs_dev, base, aux, x_dev, flag)
    stream.synchronize()

    assert (flag.copy_to_host()[0] & 0xFF) == CUBIE_RESULT_CODES.SUCCESS
    assert_allclose(
        x_dev.copy_to_host(),
        rhs / diag,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
