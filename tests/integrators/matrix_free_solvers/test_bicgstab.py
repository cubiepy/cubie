"""BiCGSTAB-specific behaviour: breakdown reporting, witness-vector
placement, and the cached-auxiliaries operator signature.

Convergence coverage lives in ``test_linear_solver.py`` and
``test_newton_krylov.py``, where the central ``solver_settings``
fixture is parameterized with ``linear_correction_type="bicgstab"``.
"""

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


def test_bicgstab_breakdown_detection(precision):
    """BiCGSTAB returns BICGSTAB_BREAKDOWN on degenerate operator."""

    @cuda.jit(device=True)
    def zero_operator(
        state, parameters, drivers, base_state, t, h, a_ij, vec, out
    ):
        for i in range(out.shape[0]):
            out[i] = precision(0.0)

    n = 3
    solver = BiCGSTABSolver(
        precision=precision,
        solver_width=n,
        krylov_atol=1e-20,
        krylov_rtol=1e-20,
        krylov_max_iters=16,
    )
    solver.update(operator_apply=zero_operator)
    solver_fn = solver.device_function

    scratch_size = 6 * n

    @cuda.jit
    def kernel(flag, h):
        state = cuda.local.array(3, precision)
        params = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        base = cuda.local.array(1, precision)
        shared = cuda.shared.array(scratch_size, precision)
        persistent_local = cuda.local.array(scratch_size, precision)
        counters = cuda.local.array(1, np.int32)
        rhs = cuda.local.array(3, precision)
        x = cuda.local.array(3, precision)
        # The weighted norm scales against ``state``; an
        # uninitialised array makes the convergence check depend on
        # leftover local-memory contents.
        params[0] = precision(0.0)
        drivers[0] = precision(0.0)
        base[0] = precision(0.0)
        for i in range(3):
            state[i] = precision(1.0)
            rhs[i] = precision(1.0)
            x[i] = precision(0.0)
        time_scalar = precision(0.0)
        flag[0] = solver_fn(
            state,
            params,
            drivers,
            base,
            time_scalar,
            h,
            precision(1.0),
            rhs,
            x,
            shared,
            persistent_local,
            counters,
        )

    out_flag = cuda.to_device(np.array([0], dtype=np.int32))
    stream = default_memmgr.get_group_stream()
    kernel[1, 1, stream](out_flag, precision(0.01))
    stream.synchronize()
    status_code = int(out_flag.copy_to_host()[0]) & 0xFF
    # Zero operator: v = A(p) = 0 with a nonzero residual, so the
    # pivot quotient rho/<r0_hat, v> overflows on the first
    # iteration and must be labelled as breakdown, not as an
    # exhausted iteration budget.
    assert status_code == CUBIE_RESULT_CODES.BICGSTAB_BREAKDOWN


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
    assert settings["r0_hat_location"] in ("local", "shared")
    assert settings["p_location"] == solver.compile_settings.p_location
    assert settings["v_location"] == solver.compile_settings.v_location
    assert settings["tmp_location"] == solver.compile_settings.tmp_location
    assert (
        settings["s_hat_location"]
        == solver.compile_settings.s_hat_location
    )


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


def test_bicgstab_r0_hat_auto_placement():
    """Witness vector auto-selects shared in the DRAM-bound window.

    The window is 512 <= n*itemsize <= 1024 bytes: below it the
    working set is served on-chip and shared placement loses; above
    it the 32 KiB dynamic-shared cap collapses the block size.
    """
    cases = [
        (np.float32, 8, "local"),
        (np.float32, 100, "local"),
        (np.float32, 128, "shared"),
        (np.float32, 200, "shared"),
        (np.float32, 256, "shared"),
        (np.float32, 300, "local"),
        (np.float64, 50, "local"),
        (np.float64, 64, "shared"),
        (np.float64, 128, "shared"),
        (np.float64, 200, "local"),
    ]
    for prec, n, expected in cases:
        config = BiCGSTABSolverConfig(precision=prec, solver_width=n)
        assert config.resolved_r0_hat_location == expected, (
            f"solver_width={n}, precision={prec.__name__}"
        )


def test_bicgstab_r0_hat_override_respected():
    """Explicit r0_hat_location bypasses the auto heuristic."""
    config = BiCGSTABSolverConfig(
        precision=np.float32, solver_width=200, r0_hat_location="local"
    )
    assert config.resolved_r0_hat_location == "local"
    config = BiCGSTABSolverConfig(
        precision=np.float32, solver_width=8, r0_hat_location="shared"
    )
    assert config.resolved_r0_hat_location == "shared"


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
    t, h, a_ij, rhs, out, temp, scratch, chain_scratch,
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
        use_cached_auxiliaries=True,
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
