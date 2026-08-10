"""Tests for the filtered (smoothed) embedded error estimate."""

from fractions import Fraction

import numpy as np
import pytest

from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import cuda, numba_from_dtype as from_dtype
from cubie.integrators.algorithms.crank_nicolson import CrankNicolsonStep
from cubie.integrators.algorithms.generic_dirk import DIRKStep
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    KVAERNO3_TABLEAU,
)
from cubie.integrators.algorithms.generic_firk import FIRKStep
from cubie.integrators.algorithms.generic_firk_tableaus import (
    GAUSS_LEGENDRE_2_TABLEAU,
    RADAU_IIA_5_TABLEAU,
)
from cubie.odesystems.solver_helpers import SolverHelperRequest
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system

# gamma0 and DD from Hairer & Wanner's radau5.f.
RADAU5_GAMMA0 = 0.27488882959567734
RADAU5_DD = (
    -(13.0 + 7.0 * np.sqrt(6.0)) / 3.0,
    (-13.0 + 7.0 * np.sqrt(6.0)) / 3.0,
    -1.0 / 3.0,
)


def test_radau_smoothed_weights_match_radau5():
    """The derived estimator reproduces Hairer & Wanner's RADAU5."""

    tableau = RADAU_IIA_5_TABLEAU
    # The eigensolver varies by an ulp across numpy builds.
    assert tableau.smoothing_gamma == pytest.approx(
        RADAU5_GAMMA0, abs=1e-13
    )

    # Exact-rational -gamma * DD @ a, rounded once at the end.
    gamma = Fraction(tableau.smoothing_gamma)
    expected = [
        float(
            -gamma
            * sum(
                Fraction(dd) * Fraction(a_entry)
                for dd, a_entry in zip(RADAU5_DD, column)
            )
        )
        for column in zip(*tableau.a)
    ]
    weights = np.asarray(tableau.smoothed_error_weights(np.float64))
    assert weights == pytest.approx(expected, abs=1e-15)

    # The f(y_n) weight is -gamma, so the estimator is consistent.
    assert weights.sum() == pytest.approx(
        tableau.smoothing_gamma, abs=1e-15
    )


def test_gauss_legendre_has_no_smoothing_operator():
    """A tableau without a sole real eigenvalue of inv(a) opts out."""

    assert not GAUSS_LEGENDRE_2_TABLEAU.supports_smoothed_error
    assert RADAU_IIA_5_TABLEAU.supports_smoothed_error


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
    """The smoothing solver enters the step's buffer group only when
    the toggle is on."""

    step = FIRKStep(
        precision=np.float64,
        n=3,
        tableau=RADAU_IIA_5_TABLEAU,
        use_smoothed_error=enabled,
        stage_state_location="shared",
    )
    registered = buffer_registry._groups[step].entries
    assert ("error_solver_shared" in registered) is enabled
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
    """``update`` turns smoothing on and registers its buffers."""

    step = step_class(precision=np.float64, n=2, tableau=tableau)
    assert not step.smooth_error
    step.update(use_smoothed_error=True)
    assert step.smooth_error
    entries = buffer_registry._groups[step].entries
    assert entries["error_solve_iters"].size == 1


# Dense numpy oracles for the at-state helper family.

ORACLE_MASS = np.array([[2.0, 0.5], [0.0, 1.5]])
ORACLE_CONSTANTS = {"a": 0.5, "b": 1.3, "c": -0.7, "d": 0.9}


def _oracle_jacobian(state, driver):
    """Dense Jacobian of the oracle system at ``state``/``driver``."""
    x0, x1 = float(state[0]), float(state[1])
    a = ORACLE_CONSTANTS["a"]
    b = ORACLE_CONSTANTS["b"]
    c = ORACLE_CONSTANTS["c"]
    d = ORACLE_CONSTANTS["d"]
    return np.array(
        [
            [a * x1 + driver, a * x0 + b],
            [2.0 * c * x0, d + driver],
        ]
    )


@pytest.fixture(scope="session")
def oracle_system():
    """Nonlinear system with a driver-dependent Jacobian and an
    off-diagonal mass matrix."""
    dxdt = [
        "dx0 = a*x0*x1 + b*x1 + d0*x0",
        "dx1 = c*x0*x0 + d*x1 + d0*x1",
    ]
    system = create_ODE_system(
        dxdt,
        states=["x0", "x1"],
        constants=ORACLE_CONSTANTS,
        drivers=["d0"],
        precision=np.float64,
        mass=ORACLE_MASS,
    )
    system.build()
    return system


def _dense_columns(kernel, n):
    """Apply a kernel to basis vectors and return its dense matrix."""
    columns = []
    for column in range(n):
        vec = np.zeros(n)
        vec[column] = 1.0
        out = np.zeros(n)
        kernel[1, 1](vec, out)
        columns.append(out.copy())
    return np.column_stack(columns)


def _helper_columns(device_fn, state, drivers, t, h, sigma, shape):
    """Return a helper's dense matrix from basis-vector applies.

    ``shape`` selects the call signature: ``"operator"`` for the
    9-argument operator, ``"preconditioner"`` for the 12-argument
    preconditioner family, ``"mass"`` for the mass product.
    """
    n = state.shape[0]
    garbage = np.full(n, 97.0)

    if shape == "operator":

        @cuda.jit
        def kernel(vec, out):
            params = cuda.local.array(1, np.float64)
            device_fn(
                state, params, drivers, garbage, t, h, sigma, vec, out
            )

    elif shape == "preconditioner":

        @cuda.jit
        def kernel(vec, out):
            params = cuda.local.array(1, np.float64)
            jvp = cuda.local.array(n, np.float64)
            scratch = cuda.local.array(n, np.float64)
            chain = cuda.local.array(n, np.float64)
            device_fn(
                state, params, drivers, garbage, t, h, sigma, vec,
                out, jvp, scratch, chain,
            )

    else:

        @cuda.jit
        def kernel(vec, out):
            device_fn(vec, out)

    return _dense_columns(kernel, n)


def test_at_state_operator_and_mass_apply_match_dense(oracle_system):
    """The at-state operator is M - sigma*h*J(state) and mass_apply
    is M, independent of base_state."""

    operator = oracle_system.get_solver_helper(
        SolverHelperRequest(kind="linear_operator_at_state")
    ).device_function
    mass_apply = oracle_system.get_solver_helper(
        SolverHelperRequest(kind="mass_apply")
    ).device_function

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
        mass_apply, state, drivers, t, h, sigma, "mass"
    )
    np.testing.assert_allclose(dense_mass, ORACLE_MASS, atol=1e-15)


def test_at_state_preconditioners_linearize_at_state(oracle_system):
    """Neumann and Jacobi at-state preconditioners evaluate J at the
    state argument, with a_ij scaling the matrix only."""

    order = 3
    neumann = oracle_system.get_solver_helper(
        SolverHelperRequest(
            kind="neumann_preconditioner_at_state",
            preconditioner_order=order,
        )
    ).device_function
    jacobi = oracle_system.get_solver_helper(
        SolverHelperRequest(kind="jacobi_preconditioner_at_state")
    ).device_function

    state = np.array([0.3, -1.2])
    drivers = np.array([0.7])
    h, sigma, t = 0.05, 0.274888, 0.0
    jac = _oracle_jacobian(state, drivers[0])

    # Truncated Neumann series in Horner form: S = v + T S.
    shift = sigma * h * jac
    dense_neumann = _helper_columns(
        neumann, state, drivers, t, h, sigma, "preconditioner"
    )
    expected = np.eye(2)
    for _ in range(order):
        expected = np.eye(2) + shift @ expected
    np.testing.assert_allclose(dense_neumann, expected, atol=1e-13)

    # Jacobi: v / diag(M - sigma*h*J).
    dense_jacobi = _helper_columns(
        jacobi, state, drivers, t, h, sigma, "preconditioner"
    )
    diagonal = np.diag(ORACLE_MASS) - sigma * h * np.diag(jac)
    np.testing.assert_allclose(
        dense_jacobi, np.diag(1.0 / diagonal), atol=1e-13
    )


@pytest.mark.parametrize(
    "step_class, tableau",
    [(DIRKStep, KVAERNO3_TABLEAU), (FIRKStep, RADAU_IIA_5_TABLEAU)],
)
@pytest.mark.parametrize(
    "linear_correction_type", ["minimal_residual", "bicgstab"]
)
@pytest.mark.parametrize(
    "preconditioner_type",
    ["neumann", "jacobi", ("neumann", "jacobi")],
    ids=["neumann", "jacobi", "chained"],
)
def test_error_solver_solves_the_at_state_dense_system(
    oracle_system,
    step_class,
    tableau,
    linear_correction_type,
    preconditioner_type,
):
    """The compiled error solver converges to the dense solution of
    (M - sigma*h*J(state)) x = rhs for every correction strategy and
    preconditioner."""

    step = step_class(
        precision=np.float64,
        n=2,
        n_drivers=1,
        evaluate_f=oracle_system.evaluate_f,
        evaluate_observables=oracle_system.evaluate_observables,
        get_solver_helper_fn=oracle_system.get_solver_helper,
        tableau=tableau,
        use_smoothed_error=True,
        linear_correction_type=linear_correction_type,
        preconditioner_type=preconditioner_type,
        krylov_atol=1e-13,
        krylov_rtol=0.0,
        krylov_max_iters=200,
    )
    assert step.smooth_error
    step.build_implicit_helpers()
    error_solver = step.error_solver.device_function

    state = np.array([0.3, -1.2])
    drivers = np.array([0.7])
    h, sigma, t = 0.05, float(tableau.smoothing_gamma), 0.0
    rhs = np.array([0.11, -0.045])
    matrix = ORACLE_MASS - sigma * h * _oracle_jacobian(
        state, drivers[0]
    )
    expected = np.linalg.solve(matrix, rhs)

    solution = np.zeros(2)
    rhs_arg = rhs.copy()
    shared = np.zeros(256)
    persistent = np.zeros(256)
    iters = np.zeros(1, dtype=np.int32)
    status = np.zeros(1, dtype=np.int32)

    @cuda.jit
    def kernel(rhs_io, x_io, shared_mem, persistent_mem, iters_out,
               status_out):
        params = cuda.local.array(1, np.float64)
        status_out[0] = error_solver(
            state,
            params,
            drivers,
            state,
            t,
            h,
            sigma,
            rhs_io,
            x_io,
            shared_mem,
            persistent_mem,
            iters_out,
        )

    kernel[1, 1](rhs_arg, solution, shared, persistent, iters, status)
    assert status[0] == 0
    np.testing.assert_allclose(solution, expected, atol=1e-9)


# One device step against a numpy replication of the algorithm.

STEP_ORACLE_CONSTANTS = {
    "a": 0.5, "b": 1.3, "c": -0.7, "d": 0.9, "e": 0.8,
}


def _step_oracle_f(state, time):
    """Right-hand side of the step-oracle system."""
    x0, x1 = float(state[0]), float(state[1])
    k = STEP_ORACLE_CONSTANTS
    return np.array(
        [
            k["a"] * x0 * x1 + k["b"] * x1 + k["e"] * time * x0,
            k["c"] * x0 * x0 + k["d"] * x1,
        ]
    )


def _step_oracle_jacobian(state, time):
    """Dense Jacobian of the step-oracle system."""
    x0, x1 = float(state[0]), float(state[1])
    k = STEP_ORACLE_CONSTANTS
    return np.array(
        [
            [k["a"] * x1 + k["e"] * time, k["a"] * x0 + k["b"]],
            [2.0 * k["c"] * x0, k["d"]],
        ]
    )


@pytest.fixture(scope="session")
def step_oracle_system():
    """Driverless nonlinear system with a time-dependent Jacobian
    and an off-diagonal mass matrix."""
    dxdt = [
        "dx0 = a*x0*x1 + b*x1 + e*t*x0",
        "dx1 = c*x0*x0 + d*x1",
    ]
    system = create_ODE_system(
        dxdt,
        states=["x0", "x1"],
        constants=STEP_ORACLE_CONSTANTS,
        precision=np.float64,
        mass=ORACLE_MASS,
    )
    system.build()
    return system


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

    proposed = np.zeros(n)
    error = np.zeros(n)
    status = np.zeros(1, dtype=np.int32)
    kernel[1, 1, 0, int(shared_bytes)](
        np.asarray(state, dtype=np.float64), proposed, error, status
    )
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


def test_dirk_step_smoothed_error_matches_dense_oracle(
    step_oracle_system,
):
    """One smoothed DIRK step filters M @ raw_error through the
    final stage's W: J at the converged final stage state and time."""

    tableau = KVAERNO3_TABLEAU
    step = DIRKStep(
        precision=np.float64,
        n=2,
        evaluate_f=step_oracle_system.evaluate_f,
        evaluate_observables=(
            step_oracle_system.evaluate_observables
        ),
        get_solver_helper_fn=step_oracle_system.get_solver_helper,
        tableau=tableau,
        use_smoothed_error=True,
        attempt_dense_prediction=False,
        newton_atol=1e-13,
        newton_rtol=0.0,
        krylov_atol=1e-13,
        krylov_rtol=0.0,
        newton_max_iters=100,
        krylov_max_iters=200,
    )
    assert step.smooth_error

    state = np.array([0.3, -1.2])
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
        if diag == 0.0:
            stage_increments[stage] = dt * np.linalg.solve(
                ORACLE_MASS, _step_oracle_f(base, stage_time)
            )
            stage_states[stage] = base
        else:
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

    # Kvaerno3 takes both A-row shortcuts: solution = final stage
    # state, raw error = solution - b_hat row's stage state.
    assert tableau.b_matches_a_row == stage_count - 1
    assert tableau.b_hat_matches_a_row is not None
    expected_state = stage_states[-1]
    np.testing.assert_allclose(
        proposed, expected_state, rtol=1e-8, atol=1e-10
    )

    raw_error = (
        expected_state - stage_states[tableau.b_hat_matches_a_row]
    )
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


def test_firk_step_smoothed_error_matches_dense_oracle(
    step_oracle_system,
):
    """One smoothed radau step builds the RADAU5 estimator:
    M @ (weighted stage sum) - gamma*h*f(y_n) filtered through
    M - gamma*h*J at the step-start state."""

    tableau = RADAU_IIA_5_TABLEAU
    step = FIRKStep(
        precision=np.float64,
        n=2,
        evaluate_f=step_oracle_system.evaluate_f,
        evaluate_observables=(
            step_oracle_system.evaluate_observables
        ),
        get_solver_helper_fn=step_oracle_system.get_solver_helper,
        tableau=tableau,
        use_smoothed_error=True,
        attempt_dense_prediction=False,
        newton_atol=1e-13,
        newton_rtol=0.0,
        krylov_atol=1e-13,
        krylov_rtol=0.0,
        newton_max_iters=100,
        krylov_max_iters=400,
    )
    assert step.smooth_error

    state = np.array([0.3, -1.2])
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


# Nonidentity mass with J = 0: the filter matrix is exactly M.

ZERO_J_CONSTANTS = {"a": 0.7, "b": -0.3, "c": 1.1}


def _zero_jacobian_f(time):
    """Quadratic-in-time, state-independent right-hand side."""
    k = ZERO_J_CONSTANTS
    return np.array(
        [k["a"] * time * time + k["b"], k["c"] * time * time]
    )


@pytest.fixture(scope="session")
def zero_jacobian_system():
    """Time-only right-hand side with an off-diagonal mass matrix."""
    dxdt = [
        "dx0 = a*t*t + b",
        "dx1 = c*t*t",
    ]
    system = create_ODE_system(
        dxdt,
        states=["x0", "x1"],
        constants=ZERO_J_CONSTANTS,
        precision=np.float64,
        mass=ORACLE_MASS,
    )
    system.build()
    return system


def test_dirk_zero_jacobian_smoothing_is_identity(
    zero_jacobian_system,
):
    """With J = 0 the filter solves M @ x = M @ raw, so the smoothed
    error equals the raw embedded estimate."""

    common = dict(
        precision=np.float64,
        n=2,
        evaluate_f=zero_jacobian_system.evaluate_f,
        evaluate_observables=(
            zero_jacobian_system.evaluate_observables
        ),
        get_solver_helper_fn=zero_jacobian_system.get_solver_helper,
        tableau=KVAERNO3_TABLEAU,
        attempt_dense_prediction=False,
        newton_atol=1e-13,
        newton_rtol=0.0,
        krylov_atol=1e-13,
        krylov_rtol=0.0,
        newton_max_iters=100,
        krylov_max_iters=200,
    )
    smoothed_step = DIRKStep(use_smoothed_error=True, **common)
    raw_step = DIRKStep(**common)

    state = np.array([0.4, -0.9])
    dt, time_value = 0.05, 0.3
    _, smoothed = _run_one_device_step(
        smoothed_step, state, dt, time_value
    )
    _, raw = _run_one_device_step(raw_step, state, dt, time_value)

    assert np.any(raw != 0.0)
    np.testing.assert_allclose(smoothed, raw, rtol=1e-6, atol=1e-12)


def test_firk_zero_jacobian_smoothing_matches_closed_form(
    zero_jacobian_system,
):
    """With J = 0 the radau estimator has the closed form
    M^-1 @ (M @ (w @ K) - gamma*h*f(t_n)) with K_i = h*M^-1@f(t_i)."""

    tableau = RADAU_IIA_5_TABLEAU
    step = FIRKStep(
        precision=np.float64,
        n=2,
        evaluate_f=zero_jacobian_system.evaluate_f,
        evaluate_observables=(
            zero_jacobian_system.evaluate_observables
        ),
        get_solver_helper_fn=zero_jacobian_system.get_solver_helper,
        tableau=tableau,
        use_smoothed_error=True,
        attempt_dense_prediction=False,
        newton_atol=1e-13,
        newton_rtol=0.0,
        krylov_atol=1e-13,
        krylov_rtol=0.0,
        newton_max_iters=100,
        krylov_max_iters=400,
    )

    state = np.array([0.4, -0.9])
    dt, time_value = 0.05, 0.3
    _, error = _run_one_device_step(step, state, dt, time_value)

    c_nodes = np.array(tableau.c)
    increments = np.stack(
        [
            dt
            * np.linalg.solve(
                ORACLE_MASS, _zero_jacobian_f(time_value + node * dt)
            )
            for node in c_nodes
        ]
    )
    weights = np.array(tableau.smoothed_error_weights(np.float64))
    comb = weights @ increments
    rhs = ORACLE_MASS @ comb - tableau.smoothing_gamma * (
        dt * _zero_jacobian_f(time_value)
    )
    expected = np.linalg.solve(ORACLE_MASS, rhs)
    np.testing.assert_allclose(error, expected, rtol=1e-8, atol=1e-12)
