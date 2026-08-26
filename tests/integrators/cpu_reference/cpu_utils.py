"""Shared utilities for CPU reference integrator implementations."""

import math
import attrs
from typing import Callable, Optional, Sequence, Union

import numpy as np
from numba import njit
from numpy.typing import NDArray

from cubie.integrators import CUBIE_RESULT_CODES


Array = NDArray[np.floating]
STATUS_MASK = 0xFFFF


def _ensure_array(
    vector: Union[Sequence[float], Array], dtype: np.dtype
) -> Array:
    """Return ``vector`` as a one-dimensional array with the desired dtype."""

    array = np.atleast_1d(vector).astype(dtype)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array of samples.")
    return array


def resolve_precision_signature(
    precision: np.dtype,
) -> tuple[np.dtype, type[np.floating]]:
    """Return a canonical ``(dtype, scalar_type)`` tuple for ``precision``."""

    dtype = np.dtype(precision)
    scalar_type = dtype.type  # type: ignore[assignment]
    return dtype, scalar_type


@njit(cache=True)
def _dot_product_impl(left: Array, right: Array) -> np.floating:
    """Return the dot product of ``left`` and ``right``."""

    size = left.shape[0]
    total = left.dtype.type(0.0)
    for index in range(size):
        total = total + left[index] * right[index]
    return total


def dot_product(left: Array, right: Array, precision: np.dtype) -> np.floating:
    """Return the dot product of ``left`` and ``right`` in ``precision``."""

    left_array = np.asarray(left, dtype=precision)
    right_array = np.asarray(right, dtype=precision)
    return _dot_product_impl(left_array, right_array)


@njit(cache=True)
def _squared_norm_impl(vector: Array) -> np.floating:
    """Return the squared Euclidean norm of ``vector``."""

    size = vector.shape[0]
    total = vector.dtype.type(0.0)
    for index in range(size):
        value = vector[index]
        total = total + value * value
    return total


def squared_norm(vector: Array, precision: np.dtype) -> np.floating:
    """Return the squared Euclidean norm of ``vector`` in ``precision``."""

    array = np.asarray(vector, dtype=precision)
    return _squared_norm_impl(array)


@njit(cache=True)
def _euclidean_norm_impl(vector: Array) -> np.floating:
    """Return the Euclidean norm of ``vector``."""

    return np.sqrt(_squared_norm_impl(vector))


def euclidean_norm(
    vector: Union[Sequence[float], Array], precision: np.dtype
) -> np.floating:
    """Return the Euclidean norm of ``vector`` in ``precision``."""

    array = np.asarray(vector, dtype=precision)
    return _euclidean_norm_impl(array)


def floored_rtol(rtol: np.floating, dtype: np.dtype) -> np.floating:
    """Raise a nonzero rtol below 4 ULPs of ``dtype`` to that floor."""

    scalar_type = np.dtype(dtype).type
    rtol_floor = scalar_type(4.0 * np.finfo(dtype).eps)
    if rtol > 0.0:
        return scalar_type(max(rtol, rtol_floor))
    return scalar_type(rtol)


@njit(cache=True)
def _scaled_norm_impl(
    values: Array,
    reference: Array,
    atol: np.floating,
    rtol: np.floating,
) -> np.floating:
    """Mean squared scaled norm; tol_i = max(atol+rtol*|ref_i|, 1e-16)."""

    size = values.shape[0]
    zero = values.dtype.type(0.0)
    nrm2 = zero
    floor = values.dtype.type(1e-16)
    inv_n = values.dtype.type(1.0 / size)
    for index in range(size):
        ref_value = reference[index]
        abs_ref = ref_value if ref_value >= zero else -ref_value
        tol = atol + rtol * abs_ref
        tol = tol if tol > floor else floor
        value = values[index]
        abs_val = value if value >= zero else -value
        ratio = abs_val / tol
        nrm2 = nrm2 + ratio * ratio
    return nrm2 * inv_n


def correction_norm_reference(
    update: Array,
    stage_state: Array,
    step_start: Array,
    atol: np.floating,
    rtol: np.floating,
) -> np.floating:
    """Return the scaled correction norm against the stage state.

    Mirrors the device correction norms: the update is scaled by
    ``atol + rtol * max(|stage_state|, |step_start|)``.
    """

    reference = np.maximum(np.abs(stage_state), np.abs(step_start))
    rtol = floored_rtol(rtol, update.dtype)
    return _scaled_norm_impl(update, reference, atol, rtol)


def scaled_norm(
    values: Union[Sequence[float], Array],
    reference: Union[Sequence[float], Array],
    atol: np.floating,
    rtol: np.floating,
    precision: np.dtype,
) -> np.floating:
    """Return the mean squared scaled norm in ``precision``."""

    dtype, scalar_type = resolve_precision_signature(precision)
    values_array = np.asarray(values, dtype=dtype)
    reference_array = np.asarray(reference, dtype=dtype)
    return _scaled_norm_impl(
        values_array,
        reference_array,
        scalar_type(atol),
        scalar_type(rtol),
    )


@njit(cache=True)
def _matrix_vector_product(matrix: Array, vector: Array, out: Array) -> None:
    """Store ``matrix @ vector`` into ``out`` without allocating."""

    rows, cols = matrix.shape
    for row in range(rows):
        total = matrix.dtype.type(0.0)
        for col in range(cols):
            total = total + matrix[row, col] * vector[col]
        out[row] = total


@njit(cache=True)
def _compute_neumann_preconditioner(matrix: Array, order: int) -> Array:
    """Return the truncated Neumann series for ``matrix``."""

    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    neumann = identity.copy()
    if order <= 0:
        return neumann

    residual = identity - matrix
    power = identity.copy()
    for _ in range(order):
        power = power @ residual
        neumann = neumann + power
    return neumann


@attrs.define
class StepResult:
    """Container describing the outcome of a single integration step."""

    state: Array
    observables: Array
    error: Array
    status: int = 0
    niters: int = 0


StepResultLike = StepResult


@njit(cache=True)
def _krylov_solve_dense_impl(
    rhs: Array,
    operator_matrix: Array,
    tolerance: np.floating,
    rtol: np.floating,
    max_iterations: int,
    initial_guess: Array,
    has_initial_guess: bool,
    neumann_order: int,
    minimal_residual: bool,
    norm_reference: Array,
    residual_reduction: np.floating,
    residual_floor: np.floating,
) -> tuple[Array, bool, int]:
    """Return the Krylov solution for a dense operator matrix.

    Converged when the weighted residual norm falls below
    ``residual_floor + residual_reduction * ||rhs||``, with the
    target fixed from the untouched right-hand side.
    """

    solution = np.empty_like(rhs)
    if has_initial_guess:
        solution[:] = initial_guess
    else:
        zero = operator_matrix.dtype.type(0.0)
        for index in range(solution.shape[0]):
            solution[index] = zero

    iteration_limit = max_iterations

    rhs_norm2 = _scaled_norm_impl(rhs, norm_reference, tolerance, rtol)
    tol = residual_floor + residual_reduction * np.sqrt(rhs_norm2)
    tol2 = np.fmin(tol * tol, np.finfo(rhs.dtype).max)

    # No guess: residual is rhs and the operator is not applied.
    operator_buffer = np.empty_like(rhs)
    residual = np.empty_like(rhs)
    if has_initial_guess:
        _matrix_vector_product(operator_matrix, solution, operator_buffer)
        for index in range(residual.shape[0]):
            residual[index] = rhs[index] - operator_buffer[index]
        residual_norm2 = _scaled_norm_impl(
            residual, norm_reference, tolerance, rtol
        )
    else:
        for index in range(residual.shape[0]):
            residual[index] = rhs[index]
        residual_norm2 = rhs_norm2
    if residual_norm2 <= tol2:
        return solution, True, 0

    preconditioner_matrix = _compute_neumann_preconditioner(
        operator_matrix,
        neumann_order,
    )
    direction = np.empty_like(rhs)

    converged = False
    iteration = 0

    while iteration < iteration_limit:
        iteration += 1

        _matrix_vector_product(preconditioner_matrix, residual, direction)
        _matrix_vector_product(operator_matrix, direction, operator_buffer)

        if minimal_residual:
            numerator = _dot_product_impl(operator_buffer, residual)
            denominator = _dot_product_impl(operator_buffer, operator_buffer)
        else:
            numerator = _dot_product_impl(residual, direction)
            denominator = _dot_product_impl(operator_buffer, direction)

        if denominator == operator_matrix.dtype.type(0.0):
            return solution, False, iteration

        alpha = operator_matrix.dtype.type(numerator / denominator)
        if converged:
            alpha = operator_matrix.dtype.type(0.0)

        for index in range(solution.shape[0]):
            solution[index] = solution[index] + alpha * direction[index]
            residual[index] = residual[index] - alpha * operator_buffer[index]
        residual_norm2 = _scaled_norm_impl(
            residual, norm_reference, tolerance, rtol
        )

        if residual_norm2 <= tol2:
            converged = True
            break

    return solution, converged, iteration


@njit(cache=True)
def _bicgstab_solve_dense_impl(
    rhs: Array,
    operator_matrix: Array,
    tolerance: np.floating,
    rtol: np.floating,
    max_iterations: int,
    initial_guess: Array,
    has_initial_guess: bool,
    neumann_order: int,
    norm_reference: Array,
    residual_reduction: np.floating,
    residual_floor: np.floating,
) -> tuple[Array, bool, int]:
    """Return the BiCGSTAB solution for a dense operator matrix.

    Mirrors the device solver's preconditioned recurrences: the
    search direction and intermediate residual are preconditioned
    before each operator application, and a vanished recurrence
    scalar (pivot, omega, or rho) exits unconverged as breakdown.
    Converged when the weighted residual norm falls below
    ``residual_floor + residual_reduction * ||rhs||``.
    """

    dtype = operator_matrix.dtype
    zero = dtype.type(0.0)
    solution = np.empty_like(rhs)
    if has_initial_guess:
        solution[:] = initial_guess
    else:
        for index in range(solution.shape[0]):
            solution[index] = zero

    rhs_norm2 = _scaled_norm_impl(rhs, norm_reference, tolerance, rtol)
    tol = residual_floor + residual_reduction * np.sqrt(rhs_norm2)
    tol2 = np.fmin(tol * tol, np.finfo(rhs.dtype).max)

    # No guess: residual is rhs and the operator is not applied.
    operator_buffer = np.empty_like(rhs)
    residual = np.empty_like(rhs)
    if has_initial_guess:
        _matrix_vector_product(operator_matrix, solution, operator_buffer)
        for index in range(residual.shape[0]):
            residual[index] = rhs[index] - operator_buffer[index]
        residual_norm2 = _scaled_norm_impl(
            residual, norm_reference, tolerance, rtol
        )
    else:
        for index in range(residual.shape[0]):
            residual[index] = rhs[index]
        residual_norm2 = rhs_norm2
    if residual_norm2 <= tol2:
        return solution, True, 0

    preconditioner_matrix = _compute_neumann_preconditioner(
        operator_matrix,
        neumann_order,
    )

    witness = residual.copy()
    direction = residual.copy()
    direction_hat = np.empty_like(rhs)
    s_hat = np.empty_like(rhs)
    v_vector = np.empty_like(rhs)
    t_vector = np.empty_like(rhs)

    rho_prev = _dot_product_impl(witness, residual)
    converged = False
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        _matrix_vector_product(
            preconditioner_matrix, direction, direction_hat
        )
        _matrix_vector_product(
            operator_matrix, direction_hat, v_vector
        )
        pivot = _dot_product_impl(witness, v_vector)
        if pivot == zero:
            return solution, False, iteration
        alpha = dtype.type(rho_prev / pivot)

        for index in range(solution.shape[0]):
            solution[index] = (
                solution[index] + alpha * direction_hat[index]
            )
            residual[index] = residual[index] - alpha * v_vector[index]
        residual_norm2 = _scaled_norm_impl(
            residual, norm_reference, tolerance, rtol
        )
        if residual_norm2 <= tol2:
            converged = True
            break

        _matrix_vector_product(preconditioner_matrix, residual, s_hat)
        _matrix_vector_product(operator_matrix, s_hat, t_vector)
        t_squared = _dot_product_impl(t_vector, t_vector)
        if t_squared == zero:
            return solution, False, iteration
        omega = dtype.type(
            _dot_product_impl(t_vector, residual) / t_squared
        )

        for index in range(solution.shape[0]):
            solution[index] = solution[index] + omega * s_hat[index]
            residual[index] = residual[index] - omega * t_vector[index]
        residual_norm2 = _scaled_norm_impl(
            residual, norm_reference, tolerance, rtol
        )
        if residual_norm2 <= tol2:
            converged = True
            break

        rho_new = _dot_product_impl(witness, residual)
        if rho_new == zero or omega == zero:
            return solution, False, iteration
        beta = dtype.type((rho_new / rho_prev) * (alpha / omega))
        for index in range(solution.shape[0]):
            direction[index] = residual[index] + beta * (
                direction[index] - omega * v_vector[index]
            )
        rho_prev = rho_new

    return solution, converged, iteration


def newton_solve(
    initial_guess: Array,
    precision: np.dtype,
    residual_fn: Callable[[Array], Array],
    jacobian_fn: Callable[[Array], Array],
    linear_solver: Callable[..., tuple[Array, bool, int]],
    newton_tol: np.floating,
    newton_max_iters: int,
    newton_rtol: np.floating = 0.0,
    correction_norm: Optional[Callable[[Array, Array], np.floating]] = None,
    prev_theta_store: Optional[Array] = None,
) -> tuple[Array, bool, int]:
    """Solve the nonlinear system on the CPU.

    Mirrors the device Newton solver: the update-error bound with
    warm-started contraction estimates decides convergence.
    ``correction_norm`` computes the scaled update norm against the
    stage context; when absent the update is scaled against the
    iterate. ``prev_theta_store`` is a one-element array persisting
    the contraction estimate between solves.
    """

    dtype, scalar_type = resolve_precision_signature(precision)
    atol_value = scalar_type(newton_tol)
    rtol_value = scalar_type(newton_rtol)
    typed_one = scalar_type(1.0)
    iteration_limit = int(newton_max_iters)

    state = np.asarray(initial_guess, dtype=dtype).copy()
    residual = np.zeros_like(state)
    direction = np.zeros_like(state)

    typed_zero = scalar_type(0.0)
    typed_tiny = scalar_type(np.finfo(dtype).tiny)
    typed_huge = scalar_type(np.finfo(dtype).max)
    kappa = scalar_type(0.01)
    rtol_value = floored_rtol(rtol_value, dtype)
    first_iteration_bound = scalar_type(1.0e-5)
    theta_decay = scalar_type(0.3)

    if correction_norm is None:
        def correction_norm(update, iterate):
            return _scaled_norm_impl(
                update, iterate, atol_value, rtol_value
            )

    # Warm-started contraction estimate; zero marks a fresh store or
    # a failed previous solve. Callers zero the store before the
    # first solve.
    stored_theta = typed_zero
    if prev_theta_store is not None:
        stored_theta = scalar_type(prev_theta_store[0])
    prev_theta = stored_theta if stored_theta > typed_zero else typed_one

    # RMS norm of the previous accepted full-step correction.
    ndz_prev = typed_zero

    converged = False
    failed = False
    iterations_used = 0
    for iteration in range(iteration_limit):
        if converged or failed:
            break
        iterations_used = iteration + 1

        residual = np.asarray(residual_fn(state), dtype=dtype)
        jacobian = np.asarray(jacobian_fn(state), dtype=dtype)

        # No guess: the correction starts from zero every solve.
        direction, linear_converged, _ = linear_solver(
            jacobian,
            -residual,
            initial_guess=None,
        )

        step = np.asarray(direction, dtype=dtype)

        norm2_dz = scalar_type(correction_norm(step, state))
        ndz = scalar_type(np.sqrt(norm2_dz))

        # A failed linear solve yields no usable correction: nothing
        # commits and no contraction evidence accrues.
        judged = bool(linear_converged)
        history = ndz_prev > typed_zero
        if history:
            theta = scalar_type(
                max(
                    theta_decay * prev_theta,
                    ndz / max(ndz_prev, typed_tiny),
                )
            )
        else:
            theta = prev_theta
        small_first_step = iteration == 0 and ndz < first_iteration_bound
        eta_accept = bool(
            theta < typed_one
            and theta * ndz < kappa * (typed_one - theta)
        )

        # Exit if non-finite.
        nonfinite = judged and not (norm2_dz <= typed_huge)
        # Accept a non-contracting update under tolerance.
        converged_floor = (
            judged
            and history
            and ndz >= ndz_prev
            and ndz <= typed_one
        )
        failed = failed or nonfinite

        commit = judged and not nonfinite and not converged_floor
        if commit:
            state = np.asarray(state + step, dtype=dtype)
        converged = converged or converged_floor or (
            commit and (eta_accept or small_first_step)
        )
        ndz_prev = ndz if commit else typed_zero
        if judged and history:
            prev_theta = theta

    # Store contraction history; failed solves reset it to 1.
    if prev_theta_store is not None:
        prev_theta_store[0] = (
            min(prev_theta, typed_one) if converged else typed_one
        )

    return state, converged, iterations_used


def make_step_result(
    state: Array,
    observables: Array,
    error: Array,
    status: int,
    niters: int,
) -> StepResultLike:
    """Return a step result container."""

    iter_count = max(0, min(int(niters) + 1, STATUS_MASK))
    return StepResult(state, observables, error, status, iter_count)


def _encode_solver_status(converged: bool, niters: int) -> int:
    """Return a solver status word with the Newton iteration count encoded."""

    base_code = (
        CUBIE_RESULT_CODES.SUCCESS
        if converged
        else CUBIE_RESULT_CODES.MAX_NEWTON_ITERATIONS_EXCEEDED
    )
    iter_count = max(0, min(int(niters) + 1, STATUS_MASK))
    return (iter_count << 16) | (int(base_code) & STATUS_MASK)


class DriverEvaluator:
    """Evaluate spline coefficients to recover driver samples on the CPU."""

    def __init__(
        self,
        coefficients: Optional[Array],
        dt: float,
        t0: float,
        wrap: bool,
        precision: np.dtype,
        boundary_condition: Optional[str] = "not-a-knot",
    ) -> None:
        coeffs = (
            np.zeros((0, 0, 1), dtype=precision)
            if coefficients is None
            else np.array(coefficients, dtype=precision, copy=True)
        )
        if coeffs.ndim != 3:
            raise ValueError(
                "Driver coefficients must have shape (segments, drivers, "
                "order + 1)."
            )
        self.precision = precision
        self.coefficients = coeffs
        self.dt = precision(dt)
        self.t0 = precision(t0)
        self.wrap = bool(wrap)
        self.boundary_condition = boundary_condition
        self._segments = coeffs.shape[0]
        self._width = coeffs.shape[1]
        self._order = coeffs.shape[2] - 1 if coeffs.size else -1
        self._zero = np.zeros(self._width, dtype=precision)
        self._zero_value = precision(0.0)
        self._pad_clamped = (not self.wrap) and (
            self.boundary_condition == "clamped"
        )
        self._inv_dt = precision(precision(1.0) / self.dt)
        offset = self.dt if self._pad_clamped else self._zero_value
        self._evaluation_start = precision(self.t0 - offset)

    def _evaluate(
        self,
        time: float,
        coefficients: Array,
        *,
        segments: Optional[int] = None,
        order: Optional[int] = None,
    ) -> tuple[Array, bool]:
        """Return Horner evaluations and range flag for ``coefficients``.

        Busybody AI over-checking left in place."""

        segment_count = self._segments if segments is None else int(segments)
        if segment_count <= 0 or self._width == 0:
            return self._zero.copy(), False

        poly_order = (
            coefficients.shape[-1] - 1 if order is None else int(order)
        )
        if poly_order < 0:
            return self._zero.copy(), False

        precision = self.precision
        time_value = precision(time)
        scaled = precision(
            (time_value - self._evaluation_start) * self._inv_dt
        )
        scaled_floor = precision(math.floor(scaled))
        idx = np.int32(scaled_floor)

        if self.wrap:
            segment = idx % segment_count
            tau = scaled - scaled_floor
            in_range = True
        else:
            max_segment = segment_count - 1
            if idx < 0:
                segment = 0
            elif idx >= segment_count:
                segment = max_segment
            else:
                segment = idx
            tau = precision(scaled - precision(segment))
            in_range = (
                scaled >= self._zero_value
                and scaled <= precision(segment_count)
            )

        limit = poly_order + 1
        values = self._zero.copy()
        for driver_idx in range(self._width):
            segment_coeffs = coefficients[segment, driver_idx, :limit]
            acc = self._zero_value
            for coeff in reversed(segment_coeffs):
                acc = acc * tau + precision(coeff)
            values[driver_idx] = acc
        return values, in_range

    def evaluate(self, time: float) -> Array:
        """Return driver values interpolated at ``time``."""

        values, in_range = self._evaluate(
            time,
            self.coefficients,
            segments=self._segments,
            order=self._order,
        )
        if self.wrap or in_range:
            return values
        return self._zero.copy()

    def __call__(self, time: float) -> Array:
        """Alias for :meth:`evaluate` so instances are callable."""

        return self.evaluate(time)

    def derivative(self, time: float) -> Array:
        """Return time derivatives of the drivers evaluated at ``time``."""

        if self._segments == 0 or self._width == 0 or self._order <= 0:
            return self._zero.copy()

        derivative_coeffs = self.coefficients[..., 1:].copy()
        powers = np.arange(
            1, derivative_coeffs.shape[-1] + 1, dtype=self.precision
        ).reshape(1, 1, -1)
        derivative_coeffs *= powers

        values, in_range = self._evaluate(
            time,
            derivative_coeffs,
            segments=self._segments,
            order=self._order - 1,
        )
        if not self.wrap and not in_range:
            return self._zero.copy()
        return values * self._inv_dt

    def with_coefficients(
        self, coefficients: Optional[Array]
    ) -> "DriverEvaluator":
        """Return a new evaluator with ``coefficients`` but shared timing."""

        return DriverEvaluator(
            coefficients=coefficients,
            dt=self.dt,
            t0=self.t0,
            wrap=self.wrap,
            precision=self.precision,
            boundary_condition=self.boundary_condition,
        )


def _lu_solve_dense_impl(
    rhs: Array,
    operator_matrix: Array,
    dtype: np.dtype,
) -> tuple[Array, bool, int]:
    """Solve by dense partial-pivot LU; pivots divide exactly."""
    scalar_type = dtype.type
    size = rhs.shape[0]
    factor = np.asarray(operator_matrix, dtype=dtype).copy()
    order = np.arange(size)
    for k in range(size):
        lead = k + int(np.argmax(np.abs(factor[k:, k])))
        if lead != k:
            factor[[k, lead], :] = factor[[lead, k], :]
            order[[k, lead]] = order[[lead, k]]
        pivot = factor[k, k]
        for i in range(k + 1, size):
            multiplier = scalar_type(factor[i, k] / pivot)
            factor[i, k] = multiplier
            factor[i, k + 1:] = (
                factor[i, k + 1:] - multiplier * factor[k, k + 1:]
            ).astype(dtype)

    solution = np.asarray(rhs, dtype=dtype)[order].copy()
    for i in range(size):
        solution[i] = solution[i] - np.dot(
            factor[i, :i], solution[:i]
        )
    for i in range(size - 1, -1, -1):
        solution[i] = scalar_type(
            (
                solution[i]
                - np.dot(factor[i, i + 1:], solution[i + 1:])
            )
            / factor[i, i]
        )
    return solution.astype(dtype), True, 1


def krylov_solve(
    operator_matrix: Array,
    rhs: Array,
    tolerance: np.floating,
    max_iterations: int,
    precision: np.dtype,
    rtol: np.floating = 0.0,
    neumann_order: int = 2,
    correction_type: str = "minimal_residual",
    initial_guess: Optional[Array] = None,
    norm_reference: Optional[Array] = None,
    residual_reduction: Optional[float] = None,
    residual_floor: Optional[float] = None,
) -> tuple[Array, bool, int]:
    """Solve ``operator_matrix @ x = rhs`` using dense Krylov iterations.

    Parameters
    ----------
    operator_matrix
        Dense linear operator expressed as a matrix.
    rhs
        Right-hand side vector.
    tolerance
        Absolute tolerance of the scaled convergence norm.
    max_iterations
        Maximum iteration count for the descent loop. Zero is permitted.
    precision
        Floating-point precision to use for the iteration.
    rtol
        Relative tolerance of the scaled convergence norm, scaled by
        ``norm_reference``.
    neumann_order
        Order of the truncated Neumann-series left preconditioner.
    correction_type
        Linear solve to apply. ``"steepest_descent"``,
        ``"minimal_residual"``, ``"bicgstab"``, or ``"lu"`` (direct
        dense LU without pivoting, one reported iteration).
    initial_guess
        Optional starting iterate. When absent the solve starts
        from zero and skips the initial operator application, so
        the first residual is the right-hand side.
    norm_reference
        Vector the weighted norm scales against. Defaults to the zero
        vector, which reduces the weights to the absolute tolerance.
    residual_reduction
        Relative factor the weighted residual must fall below, against
        the weighted right-hand side. ``None`` derives machine epsilon
        so the floor criterion governs.
    residual_floor
        Absolute term of the stopping rule, in weighted-norm units.
        ``None`` derives ``sqrt(eps)`` of the precision.

    Returns
    -------
    tuple[Array, bool, int]
        Solution vector, convergence flag, and iteration count.
    """

    if correction_type not in (
        "steepest_descent",
        "minimal_residual",
        "bicgstab",
        "lu",
    ):
        raise ValueError(
            "Correction type must be 'steepest_descent', "
            "'minimal_residual', 'bicgstab', or 'lu'."
        )

    dtype, scalar_type = resolve_precision_signature(precision)
    matrix = np.asarray(operator_matrix, dtype=dtype)
    vector = np.asarray(rhs, dtype=dtype)

    if correction_type == "lu":
        solution, converged, iteration = _lu_solve_dense_impl(
            vector, matrix, dtype
        )
        return (
            np.asarray(solution, dtype=dtype),
            bool(converged),
            int(iteration),
        )

    minimal_residual = correction_type == "minimal_residual"
    tol_value = scalar_type(tolerance)
    rtol_value = scalar_type(rtol)
    iteration_limit = int(max_iterations)
    order = int(neumann_order)

    if initial_guess is None:
        guess = np.zeros_like(vector)
    else:
        guess = np.asarray(initial_guess, dtype=dtype)

    if norm_reference is None:
        reference = np.zeros_like(vector)
    else:
        reference = np.asarray(norm_reference, dtype=dtype)
    if residual_reduction is None:
        reduction_value = scalar_type(np.finfo(dtype).eps)
    else:
        reduction_value = scalar_type(residual_reduction)
    if residual_floor is None:
        floor_value = scalar_type(float(np.finfo(dtype).eps) ** 0.5)
    else:
        floor_value = scalar_type(residual_floor)

    initial_provided = initial_guess is not None
    if correction_type == "bicgstab":
        solution, converged, iteration = _bicgstab_solve_dense_impl(
            vector,
            matrix,
            tol_value,
            rtol_value,
            iteration_limit,
            guess,
            initial_provided,
            order,
            reference,
            reduction_value,
            floor_value,
        )
    else:
        solution, converged, iteration = _krylov_solve_dense_impl(
            vector,
            matrix,
            tol_value,
            rtol_value,
            iteration_limit,
            guess,
            initial_provided,
            order,
            minimal_residual,
            reference,
            reduction_value,
            floor_value,
        )
    return (
        np.asarray(solution, dtype=dtype),
        bool(converged),
        int(iteration),
    )


__all__ = [
    "Array",
    "DriverEvaluator",
    "STATUS_MASK",
    "resolve_precision_signature",
    "newton_solve",
    "dot_product",
    "euclidean_norm",
    "StepResult",
    "_encode_solver_status",
    "_ensure_array",
    "squared_norm",
    "krylov_solve",
]
