from math import copysign as math_copysign
import os

import numpy as np
import pytest
from cubie.cuda_simsafe import cuda
from cubie.memory import default_memmgr

from cubie.integrators.matrix_free_solvers import CUBIE_RESULT_CODES
from cubie.integrators.matrix_free_solvers.bicgstab_solver import (
    BiCGSTABSolver,
)
from cubie.integrators.matrix_free_solvers.linear_solver import (
    MRLinearSolver,
)
from cubie.integrators.matrix_free_solvers.lu_solver import (
    LUSolver,
)
from cubie.integrators.matrix_free_solvers.newton_krylov import (
    NewtonKrylov,
)
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system

# Keys the matrix-free overrides may carry that must be cast through
# ``precision``, matching the float handling in tests/conftest.py.
MATRIXFREE_FLOAT_KEYS = frozenset(
    {
        "krylov_atol",
        "krylov_rtol",
        "krylov_residual_reduction",
        "krylov_residual_floor",
        "newton_atol",
        "newton_rtol",
    }
)

# Each case runs two sequential solves in one thread sharing the
# solver's persistent scratch, so warm-started contraction history
# carries from the first solve into the second. ``initials`` and
# ``expected_finals`` hold one state vector per solve.
NEWTON_CONVERGENCE_EDGE_CASES = {
    "small-first-step": dict(
        kind="zero",
        n=1,
        newton_atol=1e-6,
        newton_rtol=0.0,
        newton_max_iters=4,
        krylov_atol=1e-6,
        krylov_max_iters=8,
        initials=(3.0, 3.0),
        expected_statuses=(
            CUBIE_RESULT_CODES.SUCCESS,
            CUBIE_RESULT_CODES.SUCCESS,
        ),
        expected_counts=(1, 1),
        expected_finals=(3.0, 3.0),
        final_tolerance=0.0,
    ),
    "warm-start": dict(
        kind="linear",
        n=1,
        newton_atol=1e-2,
        newton_rtol=0.0,
        newton_max_iters=8,
        krylov_atol=1e-6,
        krylov_max_iters=8,
        initials=(3.0, 3.9999),
        expected_statuses=(
            CUBIE_RESULT_CODES.SUCCESS,
            CUBIE_RESULT_CODES.SUCCESS,
        ),
        expected_counts=(2, 1),
        expected_finals=(4.0, 4.0),
        final_tolerance=1e-3,
    ),
    # Constant residual: the stagnant solve runs to the cap.
    "stagnation-max-iters": dict(
        kind="constant",
        n=1,
        newton_atol=1e-2,
        newton_rtol=0.0,
        newton_max_iters=4,
        krylov_atol=1e-6,
        krylov_max_iters=8,
        initials=(0.0, 0.0),
        expected_statuses=(
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED,
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED,
        ),
        expected_counts=(4, 4),
        expected_finals=(-4.0, -4.0),
        final_tolerance=1e-6,
    ),
    # A growing update under tolerance accepts the iterate.
    "growth-under-tolerance-accepts": dict(
        kind="floor-bounce",
        n=1,
        newton_atol=1.0,
        newton_rtol=0.0,
        newton_max_iters=8,
        krylov_atol=1e-6,
        krylov_max_iters=8,
        initials=(0.0, 0.0),
        expected_statuses=(
            CUBIE_RESULT_CODES.SUCCESS,
            CUBIE_RESULT_CODES.SUCCESS,
        ),
        expected_counts=(3, 3),
        expected_finals=(0.54, 0.54),
        final_tolerance=1e-6,
    ),
    # Tripling updates run to the cap and flag divergence.
    "theta-growth-divergence": dict(
        kind="root",
        n=1,
        newton_atol=1e-2,
        newton_rtol=0.0,
        newton_max_iters=4,
        krylov_atol=1e-6,
        krylov_max_iters=8,
        initials=(1.0, 1.0),
        expected_statuses=(
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
            | CUBIE_RESULT_CODES.NEWTON_DIVERGENCE,
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
            | CUBIE_RESULT_CODES.NEWTON_DIVERGENCE,
        ),
        expected_counts=(4, 4),
        expected_finals=(81.0, 81.0),
        final_tolerance=1e-3,
    ),
    "linear-failure-gates-commit": dict(
        kind="mixed-diag",
        n=2,
        newton_atol=1e-3,
        newton_rtol=0.0,
        newton_max_iters=32,
        krylov_atol=1e-12,
        krylov_max_iters=1,
        initials=((0.0, 0.0), (0.0, 0.0)),
        expected_statuses=(
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
            | CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED,
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
            | CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED,
        ),
        expected_counts=(32, 32),
        expected_finals=((0.0, 0.0), (0.0, 0.0)),
        final_tolerance=0.0,
    ),
    "max-iters-exceeded": dict(
        kind="cubic",
        n=1,
        newton_atol=1e-20,
        newton_rtol=0.0,
        newton_max_iters=1,
        krylov_atol=1e-8,
        krylov_max_iters=20,
        initials=(0.0, 0.0),
        expected_statuses=(
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED,
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED,
        ),
        expected_counts=(1, 1),
        expected_finals=(1.0 / 3.0, 1.0 / 3.0),
        final_tolerance=1e-6,
    ),
    "linear-failure-blocks-accept": dict(
        kind="zero-operator",
        n=1,
        newton_atol=1e-8,
        newton_rtol=0.0,
        newton_max_iters=4,
        krylov_atol=1e-20,
        krylov_max_iters=8,
        initials=(0.0, 0.0),
        expected_statuses=(
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
            | CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED,
            CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
            | CUBIE_RESULT_CODES.MAX_LINEAR_ITERATIONS_EXCEEDED,
        ),
        expected_counts=(4, 4),
        expected_finals=(0.0, 0.0),
        final_tolerance=0.0,
    ),
    # rtol floors at 4 ULP: the repeated 2-ULP update is under tolerance.
    "tolerance-floor-accept": dict(
        kind="noise",
        n=1,
        newton_atol=1e-30,
        newton_rtol=1e-30,
        newton_max_iters=4,
        krylov_atol=1e-20,
        krylov_max_iters=8,
        initials=(1.0, 1.0),
        expected_statuses=(
            CUBIE_RESULT_CODES.SUCCESS,
            CUBIE_RESULT_CODES.SUCCESS,
        ),
        expected_counts=(2, 2),
        expected_finals=(1.0, 1.0),
        final_tolerance=1e-5,
    ),
}


@pytest.fixture(scope="session")
def newton_edge_case(request):
    """Return one named Newton convergence edge case."""
    return NEWTON_CONVERGENCE_EDGE_CASES[request.param]


@pytest.fixture(scope="session")
def newton_edge_system(newton_edge_case, precision):
    """Compile the residual and operator for one edge case."""
    kind = newton_edge_case["kind"]
    target = precision(4.0)
    noise = precision(2.0 * float(np.finfo(precision).eps))

    @cuda.jit(device=True)
    def residual(
        state, parameters, drivers, t, h, a_ij, base_state, out
    ):
        if kind == "zero":
            out[0] = precision(0.0)
        elif kind == "noise":
            out[0] = noise
        elif kind == "linear":
            out[0] = target - state[0]
        elif kind == "constant" or kind == "zero-operator":
            out[0] = precision(1.0)
        elif kind == "cubic":
            diff = state[0] - precision(1.0)
            out[0] = diff * diff * diff
        elif kind == "mixed-diag":
            for index in range(out.shape[0]):
                out[index] = (
                    precision(index + 1) * state[index]
                    - precision(1.0)
                )
        elif kind == "floor-bounce":
            if state[0] < precision(0.25):
                out[0] = precision(-0.5)
            elif state[0] < precision(0.52):
                out[0] = precision(-0.04)
            elif state[0] < precision(0.6):
                out[0] = precision(-0.1)
            else:
                out[0] = precision(-0.001)
        else:
            magnitude = abs(state[0]) ** precision(0.25)
            out[0] = math_copysign(magnitude, state[0])

    @cuda.jit(device=True)
    def operator(
        state, parameters, drivers, cached_aux, base_state, t, h, a_ij,
        vec, out,
    ):
        if kind == "linear":
            out[0] = -vec[0]
        elif kind == "root":
            out[0] = (
                precision(0.25)
                * abs(state[0]) ** precision(-0.75)
                * vec[0]
            )
        elif kind == "cubic":
            diff = state[0] - precision(1.0)
            out[0] = precision(3.0) * diff * diff * vec[0]
        elif kind == "zero-operator":
            out[0] = precision(0.0)
        elif kind == "mixed-diag":
            for index in range(out.shape[0]):
                out[index] = precision(index + 1) * vec[index]
        else:
            out[0] = vec[0]

    return {"residual": residual, "operator": operator}


@pytest.fixture(scope="session")
def newton_edge_solver(newton_edge_case, newton_edge_system, precision):
    """Build the Newton solver for one edge case."""
    case = newton_edge_case
    linear_solver = MRLinearSolver(
        precision=precision,
        solver_width=case["n"],
        krylov_atol=case["krylov_atol"],
        krylov_rtol=0.0,
        krylov_max_iters=case["krylov_max_iters"],
        zero_initial_guess=True,
    )
    linear_solver.update(operator_apply=newton_edge_system["operator"])
    newton = NewtonKrylov(
        precision=precision,
        solver_width=case["n"],
        linear_solver=linear_solver,
        newton_atol=case["newton_atol"],
        newton_rtol=case["newton_rtol"],
        newton_max_iters=case["newton_max_iters"],
    )
    newton.update(residual_function=newton_edge_system["residual"])
    return newton


@pytest.fixture(scope="session")
def newton_edge_kernel(newton_edge_case, newton_edge_solver, precision):
    """Compile the two-solve kernel once per parameter set."""
    solver = newton_edge_solver.device_function
    n_states = newton_edge_case["n"]
    shared_size = max(newton_edge_solver.shared_buffer_size, 1)
    persistent_size = max(
        newton_edge_solver.persistent_local_buffer_size, 1
    )

    @cuda.jit
    def kernel(states, statuses, counts):
        parameters = cuda.local.array(1, precision)
        drivers = cuda.local.array(1, precision)
        base_state = cuda.local.array(n_states, precision)
        counters = cuda.local.array(2, np.int32)
        cached_aux = cuda.local.array(1, precision)
        shared = cuda.shared.array(shared_size, precision)
        persistent = cuda.local.array(persistent_size, precision)
        for index in range(shared_size):
            shared[index] = precision(0.0)
        for index in range(persistent_size):
            persistent[index] = precision(0.0)
        parameters[0] = precision(0.0)
        drivers[0] = precision(0.0)
        for index in range(n_states):
            base_state[index] = precision(0.0)
        for solve in range(2):
            counters[0] = np.int32(0)
            counters[1] = np.int32(0)
            statuses[solve] = solver(
                states[solve],
                parameters,
                drivers,
                cached_aux,
                precision(0.0),
                precision(1.0),
                precision(1.0),
                base_state,
                base_state,
                shared,
                persistent,
                counters,
            )
            counts[solve] = counters[0]

    return kernel


@pytest.fixture(scope="function")
def newton_edge_outcome(newton_edge_case, newton_edge_kernel, precision):
    """Run two sequential solves and return finals/statuses/counts."""
    case = newton_edge_case
    states = cuda.to_device(
        np.array(case["initials"], dtype=precision).reshape(
            2, case["n"]
        )
    )
    statuses = cuda.to_device(np.zeros(2, dtype=np.int32))
    counts = cuda.to_device(np.zeros(2, dtype=np.int32))
    stream = default_memmgr.get_group_stream()
    newton_edge_kernel[1, 1, stream](states, statuses, counts)
    stream.synchronize()
    return (
        states.copy_to_host(),
        statuses.copy_to_host(),
        counts.copy_to_host(),
    )


@pytest.fixture(scope="session")
def system_setup(request, precision):
    """Generate symbolic systems for solver tests.

    The returned device arrays are shared across the session, so any
    test passing ``state_init`` to a solver that writes its iterate in
    place must upload its own copy first.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Provides the system identifier via ``param``.
    precision : np.dtype
        Floating point precision for the system.

    Returns
    -------
    dict
        Problem definition including helper functions and reference
        solutions computed for a small implicit Euler step.
    """
    system = request.param
    if system == "linear":
        dxdt = [
            "dx0 = 0.5*x0 - 1.0",
            "dx1 = 0.5*x1 - 2.0",
            "dx2 = 0.5*x2 - 3.0",
        ]
        mr_rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
    elif system == "graded":
        # Diagonal Jacobian graded so h*J has eigenvalues 0.1, 0.5
        # and 0.9 at the default h=0.01: inside the Neumann series'
        # convergence radius but spread enough that each extra
        # preconditioner order visibly improves the conditioning.
        dxdt = [
            "dx0 = 10.0*x0 - 1.0",
            "dx1 = 50.0*x1 - 1.0",
            "dx2 = 90.0*x2 - 1.0",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
    elif system == "stiff":
        dxdt = [
            "dx0 = 1e-6*x0 - 1e-6",
            "dx1 = 0.5*x1 - 0.5",
            "dx2 = 1e6*x2 - 1e6",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
    elif system == "coupled_linear":
        dxdt = [
            "dx0 = 0.5*x0 + 0.1*x1 - 1.0",
            "dx1 = 0.2*x0 + 0.3*x1 - 1.0",
            "dx2 = 0.1*x0 + 0.2*x1 + 0.4*x2 - 1.0",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
    elif system == "coupled_nonlinear":
        dxdt = [
            "dx0 = 0.5*x0 - x1**2 - 1.0",
            "dx1 = x0*x1 - x1**3 - 1.0",
            "dx2 = x0 + x1**2 - x2**2 - 1.0",
        ]
        mr_rhs = np.array([1.0, 1.0, 1.0], dtype=precision)
    else:
        raise ValueError(f"Unknown system: {system}")

    # Construct system, generate helper functions
    sym_system = create_ODE_system(dxdt,
                                   states=[f"x{i}" for i in range(3)],
                                   precision=precision)
    dxdt_func = sym_system.evaluate_f
    operator = sym_system.get_solver_helper(
        role="linear_operator"
    ).device_function
    # Use helper interface for residual and preconditioner generation
    residual_func = sym_system.get_solver_helper(
        role="residual"
    ).device_function

    def make_precond(order):
        return sym_system.get_solver_helper(
            role="neumann_preconditioner",
            preconditioner_order=order,
        ).device_function

    # start system from a non-equilibrium position, generate initial guesses
    # using Euler
    if system == "stiff":
        h = precision(1e-4)
        base_host = np.ones(3, dtype=precision)
    else:
        h = precision(0.01)
        base_host = np.zeros(3, dtype=precision)

    base_state = cuda.to_device(base_host)
    params = np.zeros(1, dtype=precision)
    drivers = np.zeros(1, dtype=precision)
    observables = np.zeros(3, dtype=precision)
    deriv = np.zeros(3, dtype=precision)

    @cuda.jit()
    def dxdt_kernel(state, params, drivers, observables, deriv, time_scalar):
        dxdt_func(state, params, drivers, observables, deriv, time_scalar)

    zero_time = precision(0.0)
    stream = default_memmgr.get_group_stream()
    dxdt_kernel[1, 1, stream](
        base_host, params, drivers, observables, deriv, zero_time
    )
    stream.synchronize()
    state_init_host = base_host + h * deriv * precision(1.05)

    # Step forward until we converge onto the solution
    state_fp = state_init_host.copy()
    for _ in range(32):
        dxdt_kernel[1, 1, stream](
            state_fp, params, drivers, observables, deriv, zero_time
        )
        stream.synchronize()
        new_state = base_host + h * deriv
        if np.max(np.abs(new_state - state_fp)) < precision(1e-7):
            state_fp = new_state
            break
        state_fp = new_state
    nk_expected = state_fp

    F = np.zeros((3, 3), dtype=precision)
    temp_in = np.zeros(3, dtype=precision)
    temp_out = np.zeros(3, dtype=precision)

    @cuda.jit()
    def operator_kernel(
        state,
        params,
        drivers,
        base_state,
        time_scalar,
        h,
        in_vec,
        out_vec,
    ):
        cached_aux = cuda.local.array(1, precision)
        operator(
            state,
            params,
            drivers,
            cached_aux,
            base_state,
            time_scalar,
            h,
            precision(1.0),
            in_vec,
            out_vec,
        )

    for j in range(3):
        temp_in.fill(0)
        temp_in[j] = precision(1.0)
        operator_kernel[1, 1, stream](
            state_fp,
            params,
            drivers,
            base_state,
            zero_time,
            h,
            temp_in,
            temp_out,
        )
        stream.synchronize()
        F[:, j] = temp_out
    if os.environ.get("CUBIE_TARGET_CC", "").strip():
        # Headless precompile launches return zero-filled results,
        # leaving F singular; keep collecting so later kernels still
        # compile. Real runs must surface a singular operator.
        try:
            mr_expected = np.linalg.solve(F, mr_rhs)
        except np.linalg.LinAlgError:
            mr_expected = np.full_like(mr_rhs, np.nan)
    else:
        mr_expected = np.linalg.solve(F, mr_rhs)

    return {
        "id": system,
        "n": 3,
        "h": h,
        "operator": operator,
        "residual": residual_func,
        "base_state": base_state,
        "state_init": cuda.to_device(state_init_host - base_host),
        "preconditioner": make_precond,
        "mr_rhs": mr_rhs,
        "mr_expected": mr_expected,
        "nk_expected": nk_expected,
        "sym_system": sym_system,
    }


@pytest.fixture(scope="function")
def neumann_kernel(precision):
    """Compile a kernel for the Neumann preconditioner.

    Parameters
    ----------
    precision : np.dtype
        Floating point precision used for arrays.

    Returns
    -------
    callable
        Factory producing kernels of the form
        ``(state_init, residual, out)``.
    """

    def factory(precond, n, h):
        scratch_size = n

        @cuda.jit
        def kernel(state_init, residual, base_state, out):
            time_scalar = precision(0.0)
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            temp = cuda.shared.array(scratch_size, dtype=precision)
            cached_aux = cuda.local.array(1, precision)
            precond(
                state,
                parameters,
                drivers,
                cached_aux,
                base_state,
                time_scalar,
                h,
                precision(1.0),
                residual,
                out,
                temp,
            )

        return kernel

    return factory


@pytest.fixture(scope="session")
def counting_solver_kernel():
    """Compile a solver kernel taking ``parameters`` as an argument.

    Returns
    -------
    callable
        Factory producing kernels executing
        ``(state_init, parameters, rhs, base_state, x, flag)``;
        ``flag`` receives the status code and iteration count.
    """
    def factory(linear_solver, n, h, precision):
        solver = linear_solver.device_function
        shared_size = max(linear_solver.shared_buffer_size, 1)
        persistent_size = max(
            linear_solver.persistent_local_buffer_size, 1
        )

        @cuda.jit
        def kernel(state_init, parameters, rhs, base_state, x, flag):
            time_scalar = precision(0.0)
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            drivers = cuda.local.array(1, precision)
            shared = cuda.shared.array(shared_size, dtype=precision)
            persistent_local = cuda.local.array(
                persistent_size, dtype=precision
            )
            counters = cuda.local.array(1, np.int32)
            cached_aux = cuda.local.array(1, precision)
            flag[0] = solver(
                state,
                parameters,
                drivers,
                base_state,
                cached_aux,
                time_scalar,
                h,
                precision(1.0),
                rhs,
                x,
                shared,
                persistent_local,
                counters
            )
            flag[1] = counters[0]

        return kernel

    return factory


@pytest.fixture(scope="session")
def solver_kernel():
    """Compile a kernel around a linear solver instance.

    Returns a factory taking the solver instance; buffer sizes come
    from the instance's registered buffers, so the kernel matches the
    solver it wraps.

    Returns
    -------
    callable
        Factory producing kernels executing
        ``(state_init, rhs, base_state, x, flag)``; ``flag`` is a
        length-2 int32 array receiving the status code and the
        iteration count.
    """
    def factory(linear_solver, n, h, precision):
        solver = linear_solver.device_function
        shared_size = max(linear_solver.shared_buffer_size, 1)
        persistent_size = max(
            linear_solver.persistent_local_buffer_size, 1
        )

        @cuda.jit
        def kernel(state_init, rhs, base_state, x, flag):
            time_scalar = precision(0.0)
            state = cuda.local.array(n, precision)
            for i in range(n):
                state[i] = state_init[i]
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            # Allocate shared memory for solver buffers
            shared = cuda.shared.array(shared_size, dtype=precision)
            persistent_local = cuda.local.array(
                persistent_size, dtype=precision
            )
            counters = cuda.local.array(1, np.int32)
            cached_aux = cuda.local.array(1, precision)
            flag[0] = solver(
                state,
                parameters,
                drivers,
                base_state,
                cached_aux,
                time_scalar,
                h,
                precision(1.0),
                rhs,
                x,
                shared,
                persistent_local,
                counters
            )
            flag[1] = counters[0]

        return kernel

    return factory


@pytest.fixture(scope="session")
def zero_operator(precision):
    """Device operator mapping every vector to zero.

    ``F z = 0`` for all ``z``, so no correction makes progress.
    """

    @cuda.jit(device=True)
    def operator(
        state, parameters, drivers, cached_aux, base_state, t, h, a_ij,
        vec, out,
    ):
        for index in range(out.shape[0]):
            out[index] = precision(0.0)

    return operator


@pytest.fixture(scope="session")
def degenerate_linear_solver(
    request, zero_operator, solver_settings, precision
):
    """Build a linear solver that cannot converge on any right side.

    The tolerances are far below the reachable residual and the
    iteration budget is short, so each solver class reports its own
    failure mode. ``solver_settings`` is requested so
    ``buffer_registry.reset()`` runs before the solver registers its
    buffers.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Supplies ``linear_correction_type`` through ``param``.
    zero_operator : callable
        Device operator returning zeros.
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
        "krylov_atol": 1e-20,
        "krylov_rtol": 1e-20,
        "krylov_max_iters": 16,
    }
    if request.param == "bicgstab":
        solver = BiCGSTABSolver(**common)
    else:
        solver = MRLinearSolver(
            linear_correction_type=request.param, **common
        )
    solver.update(operator_apply=zero_operator)
    return solver


@pytest.fixture(scope="function")
def matrixfree_settings_override(request):
    """Per-test override for the matrix-free solver settings.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Supplies the override dict through ``param`` when the test
        parameterizes this fixture indirectly.

    Returns
    -------
    dict
        Settings to layer over the session ``solver_settings``.
    """
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def matrixfree_settings(
    matrixfree_settings_override, solver_settings, precision
):
    """Merge the per-test override onto the session solver settings.

    Solver tolerances and iteration limits are read only by the
    function-scoped solver fixtures, so carrying them here rebuilds
    nothing beyond the solver itself. The dependency on
    ``solver_settings`` is load-bearing: it runs
    ``buffer_registry.reset()`` before any solver registers buffers.

    Parameters
    ----------
    matrixfree_settings_override : dict
        Per-test settings to apply.
    solver_settings : dict
        Session-wide solver configuration.
    precision : np.dtype
        Floating point precision the float settings are cast to.

    Returns
    -------
    dict
        The merged settings.
    """
    settings = dict(solver_settings)
    for key, value in matrixfree_settings_override.items():
        if key in MATRIXFREE_FLOAT_KEYS and value is not None:
            settings[key] = precision(value)
        else:
            settings[key] = value
    return settings


def _build_linear_solver(
    matrixfree_settings, system_setup, precision, zero_initial_guess=False
):
    """Build the linear solver selected by the merged solver settings."""
    correction_type = matrixfree_settings["linear_correction_type"]
    common = {
        "precision": precision,
        "solver_width": system_setup["n"],
        "krylov_atol": matrixfree_settings["krylov_atol"],
        "krylov_rtol": matrixfree_settings["krylov_rtol"],
        "krylov_max_iters": matrixfree_settings["krylov_max_iters"],
        "zero_initial_guess": zero_initial_guess,
    }
    if correction_type == "lu":
        solver = LUSolver(**common)
        lu_result = system_setup["sym_system"].get_solver_helper(
            "lu_solve"
        )
        solver.update(
            lu_solve_function=lu_result.device_function,
            lu_nnz=lu_result.lu_nnz,
        )
        return solver

    order = matrixfree_settings["preconditioner_order"]
    if order == 0:
        preconditioner = None
    else:
        preconditioner = system_setup["preconditioner"](order)

    if correction_type == "bicgstab":
        solver = BiCGSTABSolver(**common)
    else:
        solver = MRLinearSolver(
            linear_correction_type=correction_type, **common
        )
    solver.update(
        operator_apply=system_setup["operator"],
        preconditioner=preconditioner,
    )
    return solver


@pytest.fixture(scope="function")
def linear_solver_instance(matrixfree_settings, system_setup, precision):
    """Build the linear solver selected by the merged solver settings.

    Parameterizing ``matrixfree_settings_override`` with
    ``"bicgstab"`` exercises :class:`BiCGSTABSolver` through the same
    tests as the minimal-residual and steepest-descent solvers.
    """
    return _build_linear_solver(
        matrixfree_settings, system_setup, precision
    )


@pytest.fixture(scope="function")
def newton_kernel(precision):
    """Compile a kernel around a Newton solver instance.

    Returns a factory taking the solver instance; buffer sizes come
    from the instance's registered buffers, so the kernel matches the
    solver it wraps. The kernel signature is
    ``(state, base_state, flag, h)``.
    """

    def factory(newton_solver):
        solver = newton_solver.device_function
        shared_size = max(newton_solver.shared_buffer_size, 1)
        persistent_size = max(
            newton_solver.persistent_local_buffer_size, 1
        )

        @cuda.jit
        def kernel(state, base, flag, h):
            params = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            cached_aux = cuda.local.array(1, precision)
            counters = cuda.local.array(2, np.int32)
            a_ij = precision(1.0)
            shared = cuda.shared.array(shared_size, precision)
            persistent_local = cuda.local.array(
                persistent_size, precision
            )
            for index in range(shared_size):
                shared[index] = precision(0.0)
            for index in range(persistent_size):
                persistent_local[index] = precision(0.0)
            time_scalar = precision(0.0)
            flag[0] = solver(
                state,
                params,
                drivers,
                cached_aux,
                time_scalar,
                h,
                a_ij,
                base,
                base,
                shared,
                persistent_local,
                counters,
            )

        return kernel

    return factory


@pytest.fixture(scope="function")
def newton_solver_instance(
    matrixfree_settings, system_setup, precision
):
    """Wrap a zero-guess linear solver in a NewtonKrylov instance."""
    child = _build_linear_solver(
        matrixfree_settings,
        system_setup,
        precision,
        zero_initial_guess=True,
    )
    solver = NewtonKrylov(
        precision=precision,
        solver_width=system_setup["n"],
        linear_solver=child,
        newton_atol=matrixfree_settings["newton_atol"],
        newton_rtol=matrixfree_settings["newton_rtol"],
        newton_max_iters=matrixfree_settings["newton_max_iters"],
    )
    solver.update(residual_function=system_setup["residual"])
    return solver
