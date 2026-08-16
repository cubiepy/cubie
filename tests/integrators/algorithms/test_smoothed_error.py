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
    GAUSS_LEGENDRE_4_TABLEAU,
    RADAU_IIA_3_TABLEAU,
    RADAU_IIA_5_TABLEAU,
    RADAU_IIA_9_TABLEAU,
)
from cubie.odesystems.solver_helpers import SolverHelperRequest

from tests.system_fixtures import (
    MASS_MATRIX_DRIVER_CONSTANTS,
    MASS_MATRIX_MASS,
    MASS_MATRIX_TIME_CONSTANTS,
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

ORACLE_MASS = np.asarray(MASS_MATRIX_MASS)


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

MASS_DRIVER_SETTINGS = {
    "system_type": "mass_matrix_driver",
    "precision": np.float64,
}


def _oracle_jacobian(state, driver):
    """Dense Jacobian of the driver oracle at ``state``/``driver``."""
    x0, x1 = float(state[0]), float(state[1])
    k = MASS_MATRIX_DRIVER_CONSTANTS
    return np.array(
        [
            [k["a"] * x1 + driver, k["a"] * x0 + k["b"]],
            [2.0 * k["c"] * x0, k["d"] + driver],
        ]
    )


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


@pytest.mark.parametrize(
    "solver_settings_override", [MASS_DRIVER_SETTINGS], indirect=True
)
def test_at_state_operator_and_apply_mass_match_dense(system):
    """The at-state operator is M - sigma*h*J(state) and apply_mass
    is M, independent of base_state."""

    system.build()
    operator = system.get_solver_helper(
        SolverHelperRequest(kind="linear_operator_at_state")
    ).device_function
    apply_mass = system.get_solver_helper(
        SolverHelperRequest(kind="apply_mass")
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
        apply_mass, state, drivers, t, h, sigma, "mass"
    )
    np.testing.assert_allclose(dense_mass, ORACLE_MASS, atol=1e-15)


@pytest.mark.parametrize(
    "solver_settings_override", [MASS_DRIVER_SETTINGS], indirect=True
)
def test_evaluate_inv_mass_f_matches_dense(system):
    """The fused effective derivative equals M**-1 @ f."""

    system.build()
    device_fn = system.get_solver_helper(
        SolverHelperRequest(kind="evaluate_inv_mass_f")
    ).device_function

    state = np.array([0.3, -1.2])
    drivers = np.array([0.7])
    k = MASS_MATRIX_DRIVER_CONSTANTS
    f_value = np.array(
        [
            k["a"] * state[0] * state[1]
            + k["b"] * state[1]
            + drivers[0] * state[0],
            k["c"] * state[0] * state[0]
            + k["d"] * state[1]
            + drivers[0] * state[1],
        ]
    )
    expected = np.linalg.solve(ORACLE_MASS, f_value)

    out = np.zeros(2)

    @cuda.jit
    def kernel(state_in, out_vec):
        params = cuda.local.array(1, np.float64)
        observables = cuda.local.array(1, np.float64)
        device_fn(state_in, params, drivers, observables, out_vec, 0.0)

    kernel[1, 1](state, out)
    np.testing.assert_allclose(out, expected, atol=1e-14)


@pytest.mark.parametrize(
    "solver_settings_override", [MASS_DRIVER_SETTINGS], indirect=True
)
def test_at_state_preconditioners_linearize_at_state(system):
    """Neumann and Jacobi at-state preconditioners evaluate J at the
    state argument, with a_ij scaling the matrix only."""

    system.build()
    order = 3
    neumann = system.get_solver_helper(
        SolverHelperRequest(
            kind="neumann_preconditioner_at_state",
            preconditioner_order=order,
        )
    ).device_function
    jacobi = system.get_solver_helper(
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


# One case per correction type and per preconditioner, spread over
# both step families.

SWEEP_COMMON = {
    "system_type": "mass_matrix_driver",
    "precision": np.float64,
    "saved_state_indices": [0, 1],
    "saved_observable_indices": [],
    "summarised_state_indices": [0, 1],
    "summarised_observable_indices": [],
    "output_types": ["state", "time"],
    "use_smoothed_error": True,
    "krylov_atol": 1e-13,
    "krylov_rtol": 0.0,
    "krylov_max_iters": 200,
    "newton_atol": 1e-13,
    "newton_rtol": 0.0,
    "newton_max_iters": 100,
    "attempt_dense_prediction": False,
}

SWEEP_CASES = [
    pytest.param(
        dict(
            SWEEP_COMMON,
            algorithm="kvaerno3",
            linear_correction_type="minimal_residual",
            preconditioner_type="neumann",
        ),
        id="dirk-mr-neumann",
    ),
    pytest.param(
        dict(
            SWEEP_COMMON,
            algorithm="radau",
            linear_correction_type="bicgstab",
            preconditioner_type="jacobi",
        ),
        id="firk-bicgstab-jacobi",
    ),
    pytest.param(
        dict(
            SWEEP_COMMON,
            algorithm="radau",
            linear_correction_type="minimal_residual",
            preconditioner_type=["neumann", "jacobi"],
        ),
        id="firk-chained",
    ),
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


def _step_oracle_f(state, time):
    """Right-hand side of the time-dependent mass oracle."""
    x0, x1 = float(state[0]), float(state[1])
    k = MASS_MATRIX_TIME_CONSTANTS
    return np.array(
        [
            k["a"] * x0 * x1 + k["b"] * x1 + k["e"] * time * x0,
            k["c"] * x0 * x0 + k["d"] * x1,
        ]
    )


def _step_oracle_jacobian(state, time):
    """Dense Jacobian of the time-dependent mass oracle."""
    x0, x1 = float(state[0]), float(state[1])
    k = MASS_MATRIX_TIME_CONSTANTS
    return np.array(
        [
            [k["a"] * x1 + k["e"] * time, k["a"] * x0 + k["b"]],
            [2.0 * k["c"] * x0, k["d"]],
        ]
    )


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


# Session-chain settings for the tightly-solved oracle steps.
ORACLE_STEP_COMMON = {
    "system_type": "mass_matrix_time",
    "precision": np.float64,
    "saved_state_indices": [0, 1],
    "saved_observable_indices": [],
    "summarised_state_indices": [0, 1],
    "summarised_observable_indices": [],
    "output_types": ["state", "time"],
    "use_smoothed_error": True,
    "attempt_dense_prediction": False,
    "newton_atol": 1e-13,
    "newton_rtol": 0.0,
    "newton_max_iters": 100,
    "krylov_atol": 1e-13,
    "krylov_rtol": 0.0,
    "krylov_max_iters": 400,
}

DIRK_ORACLE_SETTINGS = dict(ORACLE_STEP_COMMON, algorithm="kvaerno3")
FIRK_ORACLE_SETTINGS = dict(ORACLE_STEP_COMMON, algorithm="radau")


@pytest.mark.parametrize(
    "solver_settings_override", [DIRK_ORACLE_SETTINGS], indirect=True
)
def test_dirk_step_smoothed_error_matches_dense_oracle(step_object):
    """One smoothed DIRK step filters M @ raw_error through the
    final stage's W: J at the converged final stage state and time."""

    step = step_object
    tableau = step.tableau
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
