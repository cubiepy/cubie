"""Tests for the filtered (smoothed) embedded error estimate."""

from fractions import Fraction

import numpy as np
import pytest

from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import cuda, numba_from_dtype as from_dtype
from cubie.memory import default_memmgr
from cubie.integrators.algorithms.crank_nicolson import CrankNicolsonStep
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    KVAERNO3_TABLEAU,
)
from cubie.integrators.algorithms.generic_firk import FIRKStep
from cubie.integrators.algorithms.generic_firk_tableaus import (
    GAUSS_LEGENDRE_2_TABLEAU,
    GAUSS_LEGENDRE_4_TABLEAU,
    RADAU_IIA_3_TABLEAU,
    RADAU_IIA_5_TABLEAU,
    RADAU_IIA_9_TABLEAU,
)

from tests.system_fixtures import (
    TORN_DRIVER_CONSTANTS,
    TORN_TIME_CONSTANTS,
)

# gamma0 and DD from Hairer & Wanner's radau5.f.
RADAU5_GAMMA0 = 0.27488882959567734
RADAU5_DD = (
    -(13.0 + 7.0 * np.sqrt(6.0)) / 3.0,
    (-13.0 + 7.0 * np.sqrt(6.0)) / 3.0,
    -1.0 / 3.0,
)

# gamma0 and DD for the five-stage tableau, from radau.f.
RADAU9_GAMMA0 = 1.0 / 6.286704751729276645173
RADAU9_DD = (
    -2.778093394406463730479e1,
    3.641478498049213152712,
    -1.252547721169118720491,
    5.920031671845428725662e-1,
    -2.0e-1,
)

# Derived mass of the torn oracle systems.
ORACLE_MASS = np.diag([1.0, 0.0])


def _radau_reference_weights(tableau, dd):
    """Return exact-rational ``-gamma * dd @ a``, rounded once."""

    gamma = Fraction(tableau.smoothing_gamma)
    return [
        float(
            -gamma
            * sum(
                Fraction(dd_entry) * Fraction(a_entry)
                for dd_entry, a_entry in zip(dd, column)
            )
        )
        for column in zip(*tableau.a)
    ]


def test_radau_smoothed_weights_match_radau5():
    """The derived estimator reproduces Hairer & Wanner's RADAU5."""

    tableau = RADAU_IIA_5_TABLEAU
    # The eigensolver varies by an ulp across numpy builds.
    assert tableau.smoothing_gamma == pytest.approx(
        RADAU5_GAMMA0, abs=1e-13
    )

    # Exact-rational -gamma * DD @ a, rounded once at the end.
    expected = _radau_reference_weights(tableau, RADAU5_DD)
    weights = np.asarray(tableau.smoothed_error_weights(np.float64))
    assert weights == pytest.approx(expected, abs=1e-15)

    # The f(y_n) weight is -gamma, so the estimator is consistent.
    assert weights.sum() == pytest.approx(
        tableau.smoothing_gamma, abs=1e-15
    )


def test_radau9_smoothed_weights_match_radau():
    """The five-stage estimator reproduces the published constants."""

    tableau = RADAU_IIA_9_TABLEAU
    assert tableau.smoothing_gamma == pytest.approx(
        RADAU9_GAMMA0, abs=1e-13
    )

    expected = _radau_reference_weights(tableau, RADAU9_DD)
    weights = np.asarray(tableau.smoothed_error_weights(np.float64))
    assert weights == pytest.approx(expected, abs=1e-14)

    assert weights.sum() == pytest.approx(
        tableau.smoothing_gamma, abs=1e-15
    )


def test_even_stage_tableaus_have_no_smoothing_operator():
    """``inv(a)`` needs a sole real eigenvalue, which needs odd s."""

    assert not GAUSS_LEGENDRE_2_TABLEAU.supports_smoothed_error
    assert not GAUSS_LEGENDRE_4_TABLEAU.supports_smoothed_error
    assert not RADAU_IIA_3_TABLEAU.supports_smoothed_error
    assert RADAU_IIA_5_TABLEAU.supports_smoothed_error
    assert RADAU_IIA_9_TABLEAU.supports_smoothed_error


def test_dirk_smoothing_gamma_is_the_last_diagonal():
    """DIRK smooths against the matrix its last stage already solves."""

    assert KVAERNO3_TABLEAU.supports_smoothed_error
    assert KVAERNO3_TABLEAU.smoothing_gamma == pytest.approx(
        KVAERNO3_TABLEAU.a[-1][-1]
    )


def test_unsupported_request_warns_and_stays_off():
    """A step without tableau support warns and leaves smoothing off."""

    with pytest.warns(UserWarning, match="use_smoothed_error"):
        step = CrankNicolsonStep(
            precision=np.float64, n=2, use_smoothed_error=True
        )
    assert not step.smooth_error

    step = CrankNicolsonStep(precision=np.float64, n=2)
    with pytest.warns(UserWarning, match="use_smoothed_error"):
        step.update(use_smoothed_error=True)
    assert not step.smooth_error
    assert step.compile_settings.use_smoothed_error


def test_smoothing_default_follows_tableau_capability():
    """Radau defaults smoothing on; DIRK and gauss-legendre stay off."""

    assert FIRKStep(
        precision=np.float64, n=2, tableau=RADAU_IIA_5_TABLEAU
    ).smooth_error
    assert FIRKStep(
        precision=np.float64, n=2, tableau=RADAU_IIA_9_TABLEAU
    ).smooth_error
    assert not FIRKStep(
        precision=np.float64, n=2, tableau=RADAU_IIA_3_TABLEAU
    ).smooth_error
    assert not FIRKStep(
        precision=np.float64, n=2, tableau=GAUSS_LEGENDRE_2_TABLEAU
    ).smooth_error
    assert not FIRKStep(
        precision=np.float64, n=2, tableau=GAUSS_LEGENDRE_4_TABLEAU
    ).smooth_error
    assert not DIRKStep(
        precision=np.float64, n=2, tableau=KVAERNO3_TABLEAU
    ).smooth_error


def test_request_survives_tableau_swap():
    """A stored request enables smoothing once the tableau can."""

    with pytest.warns(UserWarning, match="use_smoothed_error"):
        step = FIRKStep(
            precision=np.float64,
            n=2,
            tableau=GAUSS_LEGENDRE_2_TABLEAU,
            use_smoothed_error=True,
        )
    assert not step.smooth_error
    step.update(tableau=RADAU_IIA_5_TABLEAU)
    assert step.smooth_error


@pytest.mark.parametrize("enabled", [False, True])
def test_firk_error_solver_costs_nothing_when_disabled(enabled):
    """The smoothing solver is built and registered only when the
    toggle is on."""

    step = FIRKStep(
        precision=np.float64,
        n=3,
        tableau=RADAU_IIA_5_TABLEAU,
        use_smoothed_error=enabled,
        stage_state_location="shared",
    )
    registered = buffer_registry._groups[step].entries
    assert ("error_solver_shared" in registered) is enabled
    assert (step.error_solver is not None) is enabled
    assert step.smooth_error is enabled


def test_dirk_error_solver_and_rhs_alias_the_newton_window():
    """Smoothing scratch and rhs pack into the Newton window."""

    shared_locations = {
        "preconditioned_vec_location": "shared",
        "temp_location": "shared",
        "delta_location": "shared",
        "residual_location": "shared",
    }
    baseline = DIRKStep(
        precision=np.float64,
        n=3,
        tableau=KVAERNO3_TABLEAU,
        **shared_locations,
    )
    smoothed = DIRKStep(
        precision=np.float64,
        n=3,
        tableau=KVAERNO3_TABLEAU,
        use_smoothed_error=True,
        **shared_locations,
    )
    assert buffer_registry.shared_buffer_size(
        smoothed
    ) == buffer_registry.shared_buffer_size(baseline)


def test_firk_error_solver_aliases_the_coupled_solver_window():
    """The smoothing scratch overlaps the coupled solve's shared
    window rather than adding to it."""

    shared_locations = {
        "stage_increment_location": "shared",
        "preconditioned_vec_location": "shared",
        "temp_location": "shared",
        "delta_location": "shared",
        "residual_location": "shared",
    }
    baseline = FIRKStep(
        precision=np.float64,
        n=3,
        tableau=RADAU_IIA_5_TABLEAU,
        use_smoothed_error=False,
        **shared_locations,
    )
    smoothed = FIRKStep(
        precision=np.float64,
        n=3,
        tableau=RADAU_IIA_5_TABLEAU,
        use_smoothed_error=True,
        **shared_locations,
    )
    assert buffer_registry.shared_buffer_size(
        smoothed
    ) == buffer_registry.shared_buffer_size(baseline)


@pytest.mark.parametrize(
    "step_class, tableau",
    [(FIRKStep, RADAU_IIA_5_TABLEAU), (DIRKStep, KVAERNO3_TABLEAU)],
)
def test_toggle_survives_update(step_class, tableau):
    """``update`` builds the gated solver and registers its buffers."""

    step = step_class(
        precision=np.float64,
        n=2,
        tableau=tableau,
        use_smoothed_error=False,
    )
    assert not step.smooth_error
    assert step.error_solver is None
    step.update(use_smoothed_error=True)
    assert step.smooth_error
    assert step.error_solver is not None
    entries = buffer_registry._groups[step].entries
    assert entries["error_solve_iters"].size == 1


# Dense numpy oracles for the at-state helper family.

TORN_DRIVER_SETTINGS = {
    "system_type": "torn_driver",
    "precision": np.float64,
}


def _oracle_jacobian(state, driver):
    """Dense Jacobian of the torn driver oracle at ``state``."""
    x0, x1 = float(state[0]), float(state[1])
    k = TORN_DRIVER_CONSTANTS
    return np.array(
        [
            [k["a"] * x1 + driver, k["a"] * x0 + k["b"]],
            [
                2.0 * k["c"] * x0,
                k["d"] + driver + 5.0 * x1**4,
            ],
        ]
    )


def _dense_columns(kernel, n):
    """Apply a kernel to basis vectors and return its dense matrix."""
    stream = default_memmgr.get_group_stream()
    columns = []
    for column in range(n):
        vec = np.zeros(n)
        vec[column] = 1.0
        vec_dev = cuda.to_device(vec, stream=stream)
        out_dev = cuda.to_device(np.zeros(n), stream=stream)
        kernel[1, 1, stream](vec_dev, out_dev)
        out = out_dev.copy_to_host(stream=stream)
        stream.synchronize()
        columns.append(out)
    return np.column_stack(columns)


def _helper_columns(device_fn, state, drivers, t, h, sigma, shape):
    """Return a helper's dense matrix from basis-vector applies."""
    n = state.shape[0]
    garbage = np.full(n, 97.0)

    if shape == "operator":

        @cuda.jit
        def kernel(vec, out):
            params = cuda.local.array(1, np.float64)
            cached_aux = cuda.local.array(1, np.float64)
            device_fn(
                state, params, drivers, cached_aux, garbage, t, h,
                sigma, vec, out,
            )

    elif shape == "preconditioner":

        @cuda.jit
        def kernel(vec, out):
            params = cuda.local.array(1, np.float64)
            cached_aux = cuda.local.array(1, np.float64)
            jvp = cuda.local.array(n, np.float64)
            device_fn(
                state, params, drivers, cached_aux, garbage, t, h,
                sigma, vec, out, jvp,
            )

    else:

        @cuda.jit
        def kernel(vec, out):
            device_fn(vec, out)

    return _dense_columns(kernel, n)


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_DRIVER_SETTINGS], indirect=True
)
def test_at_state_operator_and_apply_mass_match_dense(system):
    """The at-state operator is M - sigma*h*J(state) and apply_mass
    is M, independent of base_state."""

    system.build()
    operator = system.get_solver_helper(
        role="linear_operator",
        jacobian_at="state",
    ).device_function
    apply_mass = system.get_solver_helper(role="apply_mass").device_function

    state = np.array([0.3, -1.2])
    drivers = np.array([0.7])
    h, sigma, t = 0.05, 0.274888, 0.0
    expected = ORACLE_MASS - sigma * h * _oracle_jacobian(
        state, drivers[0]
    )

    dense = _helper_columns(
        operator, state, drivers, t, h, sigma, "operator"
    )
    np.testing.assert_allclose(dense, expected, atol=1e-13)

    dense_mass = _helper_columns(
        apply_mass, state, drivers, t, h, sigma, "mass"
    )
    np.testing.assert_allclose(dense_mass, ORACLE_MASS, atol=1e-15)


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_DRIVER_SETTINGS], indirect=True
)
def test_evaluate_inv_mass_f_rejected_on_torn_system(system):
    """M**-1 does not exist for a zero mass row, so the fused
    effective-derivative helper refuses to generate."""

    system.build()
    with pytest.raises(ValueError, match="singular"):
        system.get_solver_helper(role="evaluate_inv_mass_f")


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_DRIVER_SETTINGS], indirect=True
)
def test_at_state_jacobi_linearizes_at_state(system):
    """Jacobi at-state evaluates J at ``state``; a_ij scales only."""

    system.build()
    jacobi = system.get_solver_helper(
        role="jacobi_preconditioner",
        jacobian_at="state",
    ).device_function

    state = np.array([0.3, -1.2])
    drivers = np.array([0.7])
    h, sigma, t = 0.05, 0.274888, 0.0
    jac = _oracle_jacobian(state, drivers[0])

    # Jacobi: v / diag(M - sigma*h*J).
    dense_jacobi = _helper_columns(
        jacobi, state, drivers, t, h, sigma, "preconditioner"
    )
    diagonal = np.diag(ORACLE_MASS) - sigma * h * np.diag(jac)
    np.testing.assert_allclose(
        dense_jacobi, np.diag(1.0 / diagonal), atol=1e-13
    )


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_DRIVER_SETTINGS], indirect=True
)
def test_at_state_jacobi_series_expands_about_the_diagonal(system):
    """Order one adds the operator's off-diagonal on a torn system."""

    system.build()
    jacobi = system.get_solver_helper(
        role="jacobi_preconditioner",
        jacobian_at="state",
        preconditioner_order=1,
    ).device_function

    state = np.array([0.3, -1.2])
    drivers = np.array([0.7])
    h, sigma, t = 0.05, 0.274888, 0.0
    jac = _oracle_jacobian(state, drivers[0])

    dense_jacobi = _helper_columns(
        jacobi, state, drivers, t, h, sigma, "preconditioner"
    )
    operator = ORACLE_MASS - sigma * h * jac
    diagonal = np.diag(np.diag(operator))
    inverse_diagonal = np.diag(1.0 / np.diag(operator))
    off_diagonal = diagonal - operator
    expected = inverse_diagonal + (
        inverse_diagonal @ off_diagonal @ inverse_diagonal
    )
    np.testing.assert_allclose(dense_jacobi, expected, atol=1e-13)


@pytest.mark.parametrize(
    "solver_settings_override", [TORN_DRIVER_SETTINGS], indirect=True
)
def test_neumann_rejected_on_torn_system(system):
    """Every Neumann variant refuses a system with a mass matrix."""

    system.build()
    for axis_kwargs in (
        {},
        {"jacobian_at": "state"},
        {"jacobian_at": "step"},
    ):
        with pytest.raises(ValueError, match="identity mass"):
            system.get_solver_helper(
                role="neumann_preconditioner",
                **axis_kwargs,
            )


# Tightly-solved oracle chains; jacobi because the mass is singular.
ORACLE_STEP_COMMON = {
    "system_type": "torn_time",
    "precision": np.float64,
    "saved_state_indices": [0, 1],
    "saved_observable_indices": [],
    "summarised_state_indices": [0, 1],
    "summarised_observable_indices": [],
    "output_types": ["state", "time"],
    "step_controller": "pi",
    "use_smoothed_error": True,
    "attempt_dense_prediction": False,
    "newton_atol": 1e-13,
    "newton_rtol": 0.0,
    "krylov_atol": 1e-13,
    "krylov_rtol": 0.0,
    "newton_max_iters": 100,
    "krylov_max_iters": 400,
    "preconditioner_type": "jacobi",
}

# SDIRK keeps every stage implicit; one correction type per family.
DIRK_ORACLE_SETTINGS = dict(
    ORACLE_STEP_COMMON,
    algorithm="l_stable_sdirk_4",
    linear_correction_type="minimal_residual",
)
FIRK_ORACLE_SETTINGS = dict(
    ORACLE_STEP_COMMON,
    algorithm="radau",
    linear_correction_type="bicgstab",
)

SWEEP_CASES = [
    pytest.param(DIRK_ORACLE_SETTINGS, id="dirk-mr-jacobi"),
    pytest.param(FIRK_ORACLE_SETTINGS, id="firk-bicgstab-jacobi"),
]


@pytest.mark.parametrize(
    "solver_settings_override", SWEEP_CASES, indirect=True
)
def test_error_solver_solves_the_at_state_dense_system(
    step_object, system
):
    """The compiled error solver converges to the dense solution of
    (M - sigma*h*J(state)) x = rhs."""

    step = step_object
    assert step.smooth_error
    step.step_function
    error_solver = step.error_solver.device_function

    state = np.array([0.3, -1.2])
    drivers = np.array([0.7])
    tableau = step.tableau
    h, sigma, t = 0.05, float(tableau.smoothing_gamma), 0.4
    rhs = np.array([0.11, -0.045])
    matrix = ORACLE_MASS - sigma * h * _step_oracle_jacobian(state, t)
    expected = np.linalg.solve(matrix, rhs)

    stream = default_memmgr.get_group_stream()
    solution_dev = cuda.to_device(np.zeros(2), stream=stream)
    rhs_dev = cuda.to_device(rhs.copy(), stream=stream)
    shared_dev = cuda.to_device(np.zeros(256), stream=stream)
    persistent_dev = cuda.to_device(np.zeros(256), stream=stream)
    iters_dev = cuda.to_device(np.zeros(1, dtype=np.int32), stream=stream)
    status_dev = cuda.to_device(
        np.zeros(1, dtype=np.int32), stream=stream
    )

    @cuda.jit
    def kernel(rhs_io, x_io, shared_mem, persistent_mem, iters_out,
               status_out):
        params = cuda.local.array(1, np.float64)
        cached_aux = cuda.local.array(1, np.float64)
        status_out[0] = error_solver(
            state,
            params,
            drivers,
            state,
            cached_aux,
            t,
            h,
            sigma,
            rhs_io,
            x_io,
            shared_mem,
            persistent_mem,
            iters_out,
        )

    kernel[1, 1, stream](
        rhs_dev, solution_dev, shared_dev, persistent_dev, iters_dev,
        status_dev,
    )
    solution = solution_dev.copy_to_host(stream=stream)
    status = status_dev.copy_to_host(stream=stream)
    stream.synchronize()
    assert status[0] == 0
    np.testing.assert_allclose(solution, expected, atol=1e-7)


# One device step against a numpy replication of the algorithm.


def _step_oracle_f(state, time):
    """Right-hand side of the torn time-dependent oracle.

    Row 0 is the differential right-hand side; row 1 is the torn
    algebraic residual (constrained to zero by the mass structure).
    """
    x0, x1 = float(state[0]), float(state[1])
    k = TORN_TIME_CONSTANTS
    return np.array(
        [
            k["a"] * x0 * x1 + k["b"] * x1 + k["e"] * time * x0,
            k["c"] * x0 * x0 + k["d"] * x1 + x1**5,
        ]
    )


def _step_oracle_jacobian(state, time):
    """Dense Jacobian of the torn time-dependent oracle."""
    x0, x1 = float(state[0]), float(state[1])
    k = TORN_TIME_CONSTANTS
    return np.array(
        [
            [k["a"] * x1 + k["e"] * time, k["a"] * x0 + k["b"]],
            [2.0 * k["c"] * x0, k["d"] + 5.0 * x1**4],
        ]
    )


def _torn_time_consistent_x1(x0):
    """Solve the torn_time residual for x1 at the given x0."""
    k = TORN_TIME_CONSTANTS
    z = 0.0
    for _ in range(100):
        residual = k["c"] * x0 * x0 + k["d"] * z + z**5
        z = z - residual / (k["d"] + 5.0 * z**4)
    return z


def _run_one_device_step(step, state, dt, time_value):
    """Run one accepted device step; return (proposed, error)."""
    n = state.shape[0]
    numba_precision = from_dtype(np.float64)
    persistent_len = max(1, int(step.persistent_local_buffer_size))
    shared_elems = max(1, int(step.shared_buffer_size))
    shared_bytes = np.float64(0).itemsize * shared_elems
    step_fn = step.step_function

    @cuda.jit
    def kernel(state_in, proposed_out, error_out, status_out):
        shared = cuda.shared.array(0, dtype=numba_precision)
        persistent = cuda.local.array(
            persistent_len, dtype=numba_precision
        )
        params = cuda.local.array(1, dtype=numba_precision)
        driver_coeffs = cuda.local.array(
            (1, 1, 1), dtype=numba_precision
        )
        drivers = cuda.local.array(1, dtype=numba_precision)
        proposed_drivers = cuda.local.array(1, dtype=numba_precision)
        observables = cuda.local.array(1, dtype=numba_precision)
        proposed_observables = cuda.local.array(
            1, dtype=numba_precision
        )
        counters = cuda.local.array(2, dtype=np.int32)
        for i in range(persistent_len):
            persistent[i] = numba_precision(0.0)
        counters[0] = 0
        counters[1] = 0
        status_out[0] = step_fn(
            state_in,
            proposed_out,
            params,
            driver_coeffs,
            drivers,
            proposed_drivers,
            observables,
            proposed_observables,
            error_out,
            numba_precision(dt),
            numba_precision(time_value),
            np.int32(1),
            np.int32(1),
            shared,
            persistent,
            counters,
        )

    stream = default_memmgr.get_group_stream()
    state_dev = cuda.to_device(
        np.asarray(state, dtype=np.float64), stream=stream
    )
    proposed_dev = cuda.to_device(np.zeros(n), stream=stream)
    error_dev = cuda.to_device(np.zeros(n), stream=stream)
    status_dev = cuda.to_device(
        np.zeros(1, dtype=np.int32), stream=stream
    )
    kernel[1, 1, stream, int(shared_bytes)](
        state_dev, proposed_dev, error_dev, status_dev
    )
    proposed = proposed_dev.copy_to_host(stream=stream)
    error = error_dev.copy_to_host(stream=stream)
    status = status_dev.copy_to_host(stream=stream)
    stream.synchronize()
    assert status[0] == 0
    return proposed, error


def _newton_dense(residual_fn, jacobian_fn, guess):
    """Solve residual(u) = 0 by dense Newton iteration."""
    iterate = guess.copy()
    for _ in range(100):
        residual = residual_fn(iterate)
        if np.linalg.norm(residual) < 1e-14:
            break
        iterate = iterate - np.linalg.solve(
            jacobian_fn(iterate), residual
        )
    return iterate


@pytest.mark.parametrize(
    "solver_settings_override", [DIRK_ORACLE_SETTINGS], indirect=True
)
def test_dirk_step_smoothed_error_matches_dense_oracle(step_object):
    """One smoothed SDIRK step on the torn system filters
    M @ raw_error through the final stage's W: J at the converged
    final stage state and time."""

    step = step_object
    tableau = step.tableau
    assert step.smooth_error

    x0 = 0.3
    state = np.array([x0, _torn_time_consistent_x1(x0)])
    dt, time_value = 0.05, 0.4
    proposed, error = _run_one_device_step(
        step, state, dt, time_value
    )

    # Stage i solves M @ K = dt * f(base + a_ii * K, t_i).
    a_matrix = np.array(tableau.a)
    c_nodes = np.array(tableau.c)
    stage_count = a_matrix.shape[0]
    stage_increments = np.zeros((stage_count, 2))
    stage_states = np.zeros((stage_count, 2))
    for stage in range(stage_count):
        stage_time = time_value + c_nodes[stage] * dt
        base = state + (
            a_matrix[stage, :stage] @ stage_increments[:stage]
        )
        diag = a_matrix[stage, stage]
        assert diag != 0.0
        increment = _newton_dense(
            lambda u: ORACLE_MASS @ u
            - dt * _step_oracle_f(base + diag * u, stage_time),
            lambda u: ORACLE_MASS
            - dt
            * diag
            * _step_oracle_jacobian(base + diag * u, stage_time),
            np.zeros(2),
        )
        stage_increments[stage] = increment
        stage_states[stage] = base + diag * increment

    # Solution = final stage state; raw error = (b - b_hat) @ K.
    assert tableau.b_matches_a_row == stage_count - 1
    assert tableau.b_hat_matches_a_row is None
    expected_state = stage_states[-1]
    np.testing.assert_allclose(
        proposed, expected_state, rtol=1e-8, atol=1e-10
    )

    error_weights = np.array(tableau.b) - np.array(tableau.b_hat)
    raw_error = error_weights @ stage_increments
    final_time = time_value + c_nodes[-1] * dt
    filter_matrix = (
        ORACLE_MASS
        - tableau.smoothing_gamma
        * dt
        * _step_oracle_jacobian(stage_states[-1], final_time)
    )
    expected_error = np.linalg.solve(
        filter_matrix, ORACLE_MASS @ raw_error
    )
    np.testing.assert_allclose(
        error, expected_error, rtol=1e-6, atol=1e-10
    )


@pytest.mark.parametrize(
    "solver_settings_override", [FIRK_ORACLE_SETTINGS], indirect=True
)
def test_firk_step_smoothed_error_matches_dense_oracle(step_object):
    """One smoothed radau step builds the RADAU5 estimator:
    M @ (weighted stage sum) - gamma*h*f(y_n) filtered through
    M - gamma*h*J at the step-start state."""

    step = step_object
    tableau = step.tableau
    assert step.smooth_error

    x0 = 0.3
    state = np.array([x0, _torn_time_consistent_x1(x0)])
    dt, time_value = 0.05, 0.4
    proposed, error = _run_one_device_step(
        step, state, dt, time_value
    )

    # K_i satisfies M K_i = dt * f(y + sum_j a_ij K_j, t_i).
    a_matrix = np.array(tableau.a)
    c_nodes = np.array(tableau.c)
    stage_count = a_matrix.shape[0]
    n = 2

    def stage_states_of(flat):
        stages = flat.reshape(stage_count, n)
        return state + a_matrix @ stages

    def coupled_residual(flat):
        stages = flat.reshape(stage_count, n)
        points = stage_states_of(flat)
        residual = np.zeros_like(stages)
        for stage in range(stage_count):
            stage_time = time_value + c_nodes[stage] * dt
            residual[stage] = ORACLE_MASS @ stages[
                stage
            ] - dt * _step_oracle_f(points[stage], stage_time)
        return residual.ravel()

    def coupled_jacobian(flat):
        points = stage_states_of(flat)
        blocks = np.zeros((stage_count * n, stage_count * n))
        for row in range(stage_count):
            stage_time = time_value + c_nodes[row] * dt
            row_jac = _step_oracle_jacobian(points[row], stage_time)
            for col in range(stage_count):
                block = -dt * a_matrix[row, col] * row_jac
                if row == col:
                    block = block + ORACLE_MASS
                blocks[
                    row * n : (row + 1) * n, col * n : (col + 1) * n
                ] = block
        return blocks

    increments = _newton_dense(
        coupled_residual, coupled_jacobian, np.zeros(stage_count * n)
    ).reshape(stage_count, n)

    b_weights = np.array(tableau.b)
    expected_state = state + b_weights @ increments
    np.testing.assert_allclose(
        proposed, expected_state, rtol=1e-8, atol=1e-10
    )

    gamma = tableau.smoothing_gamma
    smoothed_weights = np.array(
        tableau.smoothed_error_weights(np.float64)
    )
    comb = smoothed_weights @ increments
    rhs = ORACLE_MASS @ comb - gamma * dt * _step_oracle_f(
        state, time_value
    )
    filter_matrix = ORACLE_MASS - gamma * dt * _step_oracle_jacobian(
        state, time_value
    )
    expected_error = np.linalg.solve(filter_matrix, rhs)
    np.testing.assert_allclose(
        error, expected_error, rtol=1e-6, atol=1e-10
    )
