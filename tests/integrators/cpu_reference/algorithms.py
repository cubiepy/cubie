"""CPU reference implementations for integrator step algorithms."""

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from cubie.integrators.algorithms import (
    BackwardsEulerPCStep,
    BackwardsEulerStep,
    CrankNicolsonStep,
    DIRKStep,
    ERKStep,
    ExplicitEulerStep,
    FIRKStep,
    GenericRosenbrockWStep,
    resolve_alias,
    resolve_supplied_tableau,
)
from cubie.integrators.algorithms.base_algorithm_step import ButcherTableau
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DIRKTableau,
    DEFAULT_DIRK_TABLEAU,
)
from cubie.integrators.algorithms.generic_firk_tableaus import (
    FIRKTableau,
    DEFAULT_FIRK_TABLEAU,
)
from cubie.integrators.stage_predictors import (
    dense_predictor_matrix,
    tableau_supports_dense_prediction,
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DEFAULT_ERK_TABLEAU,
    ERKTableau,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    RosenbrockTableau,
)

from .cpu_ode_system import CPUODESystem
from .cpu_utils import (
    Array,
    DriverEvaluator,
    StepResultLike,
    correction_norm_reference,
    make_step_result,
    newton_solve,
    _encode_solver_status,
    krylov_solve,
)


class CPUStep:
    """Base class for CPU reference integrator steps."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        newton_rtol: float = 0.0,
        linear_rtol: float = 0.0,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        residual_reduction: Optional[float] = None,
        residual_floor: Optional[float] = None,
        tableau: Optional[ButcherTableau] = None,
        use_smoothed_error: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.driver_evaluator = driver_evaluator
        self.precision = evaluator.precision
        self._state_size = evaluator.system.sizes.states
        self._identity = np.eye(self._state_size, dtype=self.precision)
        self._newton_tol = self.precision(newton_tol)
        self._newton_rtol = self.precision(newton_rtol)
        self._newton_max_iters = int(newton_max_iters)
        self._linear_tol = self.precision(linear_tol)
        self._linear_rtol = self.precision(linear_rtol)
        self._linear_max_iters = int(linear_max_iters)
        # Derive-unless-overridden, mirroring LinearSolverBaseConfig.
        if residual_reduction is None:
            self._residual_reduction = self.precision(
                np.finfo(self.precision).eps
            )
        else:
            self._residual_reduction = self.precision(residual_reduction)
        if residual_floor is None:
            self._residual_floor = self.precision(
                float(np.finfo(self.precision).eps) ** 0.5
            )
        else:
            self._residual_floor = self.precision(residual_floor)
        # Weighted-norm reference for linear solves; steps stash the
        # stage base state (or the model state for direct solves)
        # here before each solve.
        self._linear_norm_reference = np.zeros(
            self._state_size, dtype=self.precision
        )
        correction = str(linear_correction_type)
        if correction not in {
            "steepest_descent",
            "minimal_residual",
            "bicgstab",
        }:
            raise ValueError(
                "Correction type must be 'steepest_descent', "
                "'minimal_residual', or 'bicgstab'."
            )
        self._linear_correction_type = correction
        self._preconditioner_order = int(preconditioner_order)
        if self._preconditioner_order < 0:
            raise ValueError("Preconditioner order must be non-negative.")
        # Contraction history persisted between Newton solves.
        self._newton_prev_theta = np.zeros(1, dtype=self.precision)
        self.tableau = tableau
        self._use_smoothed_error = (
            use_smoothed_error
            and tableau is not None
            and tableau.supports_smoothed_error
        )

        # Cached tableau-derived values (computed once at init)
        self._stage_count: Optional[int] = None
        self._first_same_as_last: Optional[bool] = None
        self._a_matrix: Optional[Array] = None
        self._b_weights: Optional[Array] = None
        self._c_nodes: Optional[Array] = None
        self._error_weights: Optional[Array] = None
        # Optional fields for Rosenbrock-W style tableaus
        self._C_matrix: Optional[Array] = None
        self._gamma: Optional[np.floating[Any]] = None
        self._gamma_stages: Optional[Array] = None

        tb = self.tableau
        if tb is not None:
            self._stage_count = tb.stage_count
            self._first_same_as_last = tb.first_same_as_last
            self._a_matrix = np.asarray(
                tb.typed_rows(tb.a, self.precision), dtype=self.precision
            )
            self._b_weights = np.asarray(
                tb.typed_vector(tb.b, self.precision), dtype=self.precision
            )
            self._c_nodes = np.asarray(
                tb.typed_vector(tb.c, self.precision), dtype=self.precision
            )
            error_weights = tb.error_weights(self.precision)
            if self._use_smoothed_error and isinstance(tb, FIRKTableau):
                error_weights = tb.smoothed_error_weights(self.precision)
            if error_weights is not None:
                self._error_weights = np.asarray(
                    error_weights, dtype=self.precision
                )
            if isinstance(tb, RosenbrockTableau):
                self._C_matrix = np.asarray(
                    tb.typed_rows(tb.C, self.precision), dtype=self.precision
                )
                self._gamma = self.precision(tb.gamma)
                gamma_stages = tb.typed_gamma_stages(self.precision)
                if gamma_stages:
                    self._gamma_stages = np.asarray(
                        gamma_stages, dtype=self.precision
                    )

    def _tableau_rows(
        self, values: Optional[Sequence[Sequence[float]]]
    ) -> Optional[Array]:
        """Return ``values`` typed to the step precision when available."""

        if values is None or self.tableau is None:
            return None
        typed_rows = self.tableau.typed_rows(values, self.precision)
        return np.asarray(typed_rows, dtype=self.precision)

    def _tableau_vector(
        self, values: Optional[Sequence[float]]
    ) -> Optional[Array]:
        """Return ``values`` typed to the step precision when available."""

        if values is None or self.tableau is None:
            return None
        typed_values = self.tableau.typed_vector(values, self.precision)
        return np.asarray(typed_values, dtype=self.precision)

    @property
    def stage_count(self) -> Optional[int]:
        """Return the number of stages described by the tableau."""

        if self.tableau is None:
            return None
        return self._stage_count

    @property
    def first_same_as_last(self) -> Optional[bool]:
        """Return whether the tableau shares its first and last stage."""

        if self.tableau is None:
            return None
        return self._first_same_as_last

    @property
    def a_matrix(self) -> Optional[Array]:
        """Return the stage coupling coefficients."""

        if self.tableau is None:
            return None
        return self._a_matrix

    @property
    def b_weights(self) -> Optional[Array]:
        """Return the weights for combining stage derivatives."""

        if self.tableau is None:
            return None
        return self._b_weights

    @property
    def c_nodes(self) -> Optional[Array]:
        """Return the substage time nodes."""

        if self.tableau is None:
            return None
        return self._c_nodes

    @property
    def error_weights(self) -> Optional[Array]:
        """Return embedded error weights when available."""

        if self.tableau is None:
            return None
        return self._error_weights

    # Optional Rosenbrock-W helpers exposed for subclasses
    @property
    def C_matrix(self) -> Optional[Array]:
        if self.tableau is None:
            return None
        return self._C_matrix

    @property
    def gamma(self) -> Optional[np.floating[Any]]:
        if self.tableau is None:
            return None
        return self._gamma

    def step(self, **_: object) -> StepResultLike:
        """Execute a single integration step."""

        raise NotImplementedError

    def _make_result(
        self,
        *,
        state: Array,
        observables: Array,
        error: Array,
        status: int,
        niters: int,
    ) -> StepResultLike:
        """Return a step result."""

        return make_step_result(
            state=state,
            observables=observables,
            error=error,
            status=status,
            niters=niters,
        )

    def ensure_array(
        self,
        vector: Optional[Union[Sequence[float], Array]],
        *,
        copy: bool = False,
    ) -> Array:
        if vector is None:
            return np.zeros(0, dtype=self.precision)
        array = np.asarray(vector, dtype=self.precision)
        if array.ndim != 1:
            raise ValueError("Expected a one-dimensional array.")
        if copy:
            return array.copy()
        return array

    def drivers(self, time: float) -> Array:
        return self.driver_evaluator(self.precision(time))

    def driver_time_derivative(self, time: float) -> Array:
        """Return driver time derivatives evaluated at ``time``."""

        return self.driver_evaluator.derivative(self.precision(time))

    def mass_matrix_apply(self, vector: Array) -> Array:
        return vector

    def mass_matrix_solve(self, vector: Array) -> Array:
        """Return ``M^-1 @ vector``; identity on the base class."""
        return vector

    def observables(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        time: float,
    ) -> Array:
        return self.evaluator.observables(state, params, drivers, time)

    def rhs(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        observables: Array,
        time: float,
    ) -> Array:
        derivative, _ = self.evaluator.rhs(
            state,
            params,
            drivers,
            observables,
            time,
        )
        return derivative

    def observables_and_jac(
        self,
        state: Array,
        params: Array,
        drivers: Array,
        time: float,
    ) -> tuple[Array, Array]:
        observables = self.evaluator.observables(
            state,
            params,
            drivers,
            time,
        )
        jacobian = self.evaluator.jacobian(
            state,
            params,
            drivers,
            observables,
            time,
        )
        return observables, jacobian

    def linear_solve(
        self,
        matrix: Array,
        rhs: Array,
        *,
        initial_guess: Optional[Array] = None,
    ) -> tuple[Array, bool, int]:
        coefficients = np.asarray(matrix, dtype=self.precision)
        vector = np.asarray(rhs, dtype=self.precision)

        solution, converged, niters = krylov_solve(
            coefficients,
            vector,
            tolerance=self._linear_tol,
            max_iterations=self._linear_max_iters,
            precision=self.precision,
            rtol=self._linear_rtol,
            neumann_order=self._preconditioner_order,
            correction_type=self._linear_correction_type,
            initial_guess=initial_guess,
            norm_reference=self._linear_norm_reference,
            residual_reduction=self._residual_reduction,
            residual_floor=self._residual_floor,
        )
        return np.asarray(solution, dtype=self.precision), converged, niters

    def dense_mass_matrix(self) -> Array:
        """Return the mass matrix as a dense array."""

        columns = [
            self.mass_matrix_apply(column)
            for column in self._identity.T
        ]
        return np.column_stack(columns).astype(self.precision)

    def smooth_error(
        self,
        error: Array,
        eval_state: Array,
        params: Array,
        drivers: Array,
        time: float,
        gamma: float,
        dt: float,
        norm_reference: Array,
        initial_guess: Optional[Array] = None,
    ) -> Array:
        """Filter ``error`` through ``(M - gamma*dt*J)^-1``."""

        _, jacobian = self.observables_and_jac(
            eval_state, params, drivers, time
        )
        scale = self.precision(gamma) * self.precision(dt)
        matrix = self.dense_mass_matrix() - scale * jacobian
        self._linear_norm_reference = norm_reference
        if initial_guess is None:
            initial_guess = error.copy()
        solution, _, _ = self.linear_solve(
            matrix, error.copy(), initial_guess=initial_guess.copy()
        )
        return solution

    @property
    def linear_correction_type(self) -> str:
        """Return the Krylov correction strategy used by the step."""

        return self._linear_correction_type

    @property
    def preconditioner_order(self) -> int:
        """Return the Neumann-series preconditioner order."""

        return self._preconditioner_order

    @property
    def newton_atol(self) -> np.floating:
        """Return the Newton iteration absolute tolerance."""

        return self._newton_tol

    @property
    def max_newton_iterations(self) -> int:
        """Return the Newton iteration cap."""

        return self._newton_max_iters

    @property
    def linear_tolerance(self) -> np.floating:
        """Return the Krylov solver convergence tolerance."""

        return self._linear_tol

    @property
    def max_linear_iterations(self) -> int:
        """Return the Krylov iteration cap."""

        return self._linear_max_iters

    def residual(self, candidate: Array) -> Array:
        raise NotImplementedError

    def jacobian(self, candidate: Array) -> Array:
        raise NotImplementedError

    def _status(self, converged: bool, niters: int) -> int:
        return _encode_solver_status(converged, niters)


class CPUExplicitEulerStep(CPUStep):
    """Explicit Euler step implementation."""

    def residual(self, candidate: Array) -> Array:
        raise NotImplementedError

    def jacobian(self, candidate: Array) -> Array:
        raise NotImplementedError

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
        prev_accepted: bool = True,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        drivers_now = self.drivers(current_time)
        observables_now = self.observables(
            state_vector,
            params_array,
            drivers_now,
            current_time,
        )
        derivative = self.rhs(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )
        new_state = state_vector + dt_value * derivative

        next_time = current_time + dt_value
        drivers_next = self.drivers(next_time)
        observables = self.observables(
            new_state,
            params_array,
            drivers_next,
            next_time,
        )
        error = np.zeros_like(state_vector, dtype=self.precision)
        status = self._status(True, 0)
        return self._make_result(
            state=new_state,
            observables=observables,
            error=error,
            status=status,
            niters=0,
        )


class CPUBackwardEulerStep(CPUStep):
    """Backward Euler step solved with Newton iterations."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        *,
        newton_rtol: float = 0.0,
        linear_rtol: float = 0.0,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        residual_reduction: Optional[float] = None,
        residual_floor: Optional[float] = None,
    ) -> None:
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_rtol=newton_rtol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_rtol=linear_rtol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            residual_reduction=residual_reduction,
            residual_floor=residual_floor,
        )
        self._be_state = np.zeros(self._state_size, dtype=self.precision)
        self._be_params = np.zeros(0, dtype=self.precision)
        self._be_drivers = np.zeros(0, dtype=self.precision)
        self._be_time = self.precision(0.0)
        self._be_dt = self.precision(0.0)
        self._be_increment = np.zeros(self._state_size, dtype=self.precision)

    def residual(self, candidate: Array) -> Array:
        base_state = self._be_state
        increment = candidate
        stage_state = base_state + increment
        observables = self.observables(
            stage_state,
            self._be_params,
            self._be_drivers,
            self._be_time,
        )
        derivative = self.rhs(
            stage_state,
            self._be_params,
            self._be_drivers,
            observables,
            self._be_time,
        )
        beta = self.precision(1.0)
        gamma = self.precision(1.0)
        mass_term = self.mass_matrix_apply(increment)
        return beta * mass_term - gamma * self._be_dt * derivative

    def jacobian(self, candidate: Array) -> Array:
        stage_state = self._be_state + candidate
        _, jacobian = self.observables_and_jac(
            stage_state,
            self._be_params,
            self._be_drivers,
            self._be_time,
        )
        return self._identity - self._be_dt * jacobian

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        initial_guess: Optional[Array] = None,
        time: float = 0.0,
        prev_accepted: bool = True,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)
        next_time = current_time + dt_value

        drivers_next = self.drivers(next_time)
        self._be_state = state_vector
        self._be_params = params_array
        self._be_drivers = drivers_next
        self._be_time = next_time
        self._be_dt = dt_value
        self._linear_norm_reference = state_vector

        if initial_guess is None:
            guess = self._be_increment
        else:
            guess = self.ensure_array(initial_guess, copy=True)

        def correction_norm(update, iterate):
            stage_state = state_vector + iterate
            return correction_norm_reference(
                update,
                stage_state,
                state_vector,
                self._newton_tol,
                self._newton_rtol,
            )

        increment, converged, niters = newton_solve(
            guess,
            precision=self.precision,
            residual_fn=self.residual,
            jacobian_fn=self.jacobian,
            linear_solver=self.linear_solve,
            newton_tol=self._newton_tol,
            newton_rtol=self._newton_rtol,
            newton_max_iters=self._newton_max_iters,
            correction_norm=correction_norm,
            prev_theta_store=self._newton_prev_theta,
        )
        next_state = state_vector + increment

        observables = self.observables(
            next_state,
            params_array,
            drivers_next,
            next_time,
        )
        error = np.zeros_like(next_state, dtype=self.precision)
        status = self._status(converged, niters)
        self._be_increment = increment
        return self._make_result(
            state=next_state,
            observables=observables,
            error=error,
            status=status,
            niters=niters,
        )


class CPUCrankNicolsonStep(CPUStep):
    """Crank--Nicolson step with backward Euler error estimate."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        newton_rtol: float = 0.0,
        linear_rtol: float = 0.0,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        residual_reduction: Optional[float] = None,
        residual_floor: Optional[float] = None,
        backward_step: Optional[CPUBackwardEulerStep] = None,
    ) -> None:
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_rtol=newton_rtol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_rtol=linear_rtol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            residual_reduction=residual_reduction,
            residual_floor=residual_floor,
        )
        self._cn_previous_state = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._cn_params = np.zeros(0, dtype=self.precision)
        self._cn_drivers_next = np.zeros(0, dtype=self.precision)
        self._cn_next_time = self.precision(0.0)
        self._cn_dt = self.precision(0.0)
        self._cn_stage_coefficient = self.precision(0.0)
        self._cn_base_state = np.zeros(self._state_size, dtype=self.precision)
        if backward_step is None:
            backward_step = CPUBackwardEulerStep(
                evaluator,
                driver_evaluator,
                newton_tol=newton_tol,
                newton_rtol=newton_rtol,
                newton_max_iters=newton_max_iters,
                linear_tol=linear_tol,
                linear_rtol=linear_rtol,
                linear_max_iters=linear_max_iters,
                linear_correction_type=linear_correction_type,
                preconditioner_order=preconditioner_order,
                residual_reduction=residual_reduction,
                residual_floor=residual_floor,
            )
            # The device solver shares one contraction history across
            # the trapezoidal and backward Euler stages.
            backward_step._newton_prev_theta = self._newton_prev_theta
            self._backward = backward_step

    def residual(self, candidate: Array) -> Array:
        increment = candidate
        stage_state = (
            self._cn_base_state
            + self._cn_stage_coefficient * increment
        )
        observables = self.observables(
            stage_state,
            self._cn_params,
            self._cn_drivers_next,
            self._cn_next_time,
        )
        derivative = self.rhs(
            stage_state,
            self._cn_params,
            self._cn_drivers_next,
            observables,
            self._cn_next_time,
        )
        beta = self.precision(1.0)
        gamma = self.precision(1.0)
        mass_term = self.mass_matrix_apply(increment)
        scale = self._cn_dt
        return beta * mass_term - gamma * scale * derivative

    def jacobian(self, candidate: Array) -> Array:
        stage_state = (
            self._cn_base_state
            + self._cn_stage_coefficient * candidate
        )
        _, jacobian = self.observables_and_jac(
            stage_state,
            self._cn_params,
            self._cn_drivers_next,
            self._cn_next_time,
        )
        scale = self._cn_dt * self._cn_stage_coefficient
        return self._identity - scale * jacobian

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
        prev_accepted: bool = True,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)
        next_time = current_time + dt_value

        drivers_now = self.drivers(current_time)
        observables_now = self.observables(
            state_vector,
            params_array,
            drivers_now,
            current_time,
        )
        derivative_now = self.rhs(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )

        drivers_next = self.drivers(next_time)
        self._cn_previous_state = state_vector
        self._cn_params = params_array
        self._cn_drivers_next = drivers_next
        self._cn_next_time = next_time
        self._cn_dt = dt_value
        self._cn_stage_coefficient = self.precision(0.5)
        base_increment = self._cn_dt * self._cn_stage_coefficient
        self._cn_base_state = state_vector + base_increment * derivative_now
        self._linear_norm_reference = self._cn_base_state

        guess = dt_value * derivative_now

        def correction_norm(update, iterate):
            stage_state = (
                self._cn_base_state
                + self._cn_stage_coefficient * iterate
            )
            return correction_norm_reference(
                update,
                stage_state,
                state_vector,
                self._newton_tol,
                self._newton_rtol,
            )

        increment, converged, niters = newton_solve(
            guess,
            precision=self.precision,
            residual_fn=self.residual,
            jacobian_fn=self.jacobian,
            linear_solver=self.linear_solve,
            newton_tol=self._newton_tol,
            newton_rtol=self._newton_rtol,
            newton_max_iters=self._newton_max_iters,
            correction_norm=correction_norm,
            prev_theta_store=self._newton_prev_theta,
        )
        stage_increment = self._cn_stage_coefficient * increment
        next_state = self._cn_base_state + stage_increment
        full_increment = next_state - self._cn_previous_state

        observables = self.observables(
            next_state,
            params_array,
            drivers_next,
            next_time,
        )
        backward_result = self._backward.step(
            state=state_vector,
            params=params_array,
            dt=dt_value,
            initial_guess=full_increment,
            time=current_time,
        )
        error = next_state - backward_result.state
        status = self._status(converged, niters) | backward_result.status
        return self._make_result(
            state=next_state,
            observables=observables,
            error=error,
            status=status,
            niters=niters,
        )


class CPUERKStep(CPUStep):
    """Explicit Runge--Kutta step implementation."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        newton_rtol: float = 0.0,
        linear_rtol: float = 0.0,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        residual_reduction: Optional[float] = None,
        residual_floor: Optional[float] = None,
        tableau: Optional[ERKTableau] = None,
    ) -> None:
        resolved = DEFAULT_ERK_TABLEAU if tableau is None else tableau
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_rtol=newton_rtol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_rtol=linear_rtol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            residual_reduction=residual_reduction,
            residual_floor=residual_floor,
            tableau=resolved,
        )

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
        prev_accepted: bool = True,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        stage_count = self.stage_count
        a_matrix = self.a_matrix
        b_weights = self.b_weights
        c_nodes = self.c_nodes
        error_weights = self.error_weights


        state_dim = state_vector.shape[0]
        stage_derivatives = np.zeros(
            (stage_count, state_dim),
            dtype=self.precision,
        )
        stage_states = np.zeros(
            (stage_count, state_dim),
            dtype=self.precision,
        )

        for stage_index in range(stage_count):
            # Rounding-sensitive: scale by dt once, after accumulating.
            stage_accum = np.zeros_like(state_vector)
            for dependency in range(stage_index):
                stage_accum = stage_accum + (
                    a_matrix[stage_index, dependency]
                    * stage_derivatives[dependency]
                )
            stage_state = stage_accum * dt_value + state_vector
            stage_states[stage_index, :] = stage_state
            stage_time = current_time + c_nodes[stage_index] * dt_value
            drivers_stage = self.drivers(stage_time)
            observables_stage = self.observables(
                stage_state,
                params_array,
                drivers_stage,
                stage_time,
            )
            derivative = self.rhs(
                stage_state,
                params_array,
                drivers_stage,
                observables_stage,
                stage_time,
            )
            stage_derivatives[stage_index, :] = derivative

        # A stage whose A row equals b (or b_hat) already holds the
        # output (or embedded) solution.
        b_row = self.tableau.b_matches_a_row
        b_hat_row = self.tableau.b_hat_matches_a_row
        if b_row is not None:
            new_state = stage_states[b_row].copy()
        else:
            state_update = np.zeros_like(state_vector)
            for stage_index in range(stage_count):
                state_update = state_update + (
                    b_weights[stage_index] * stage_derivatives[stage_index]
                )
            new_state = state_vector + dt_value * state_update

        error = np.zeros_like(state_vector)
        if error_weights is not None:
            if b_hat_row is not None:
                error = new_state - stage_states[b_hat_row]
            else:
                error_update = np.zeros_like(state_vector)
                for stage_index in range(stage_count):
                    error_update = error_update + (
                        error_weights[stage_index]
                        * stage_derivatives[stage_index]
                    )
                error = error_update * dt_value

        end_time = current_time + dt_value
        drivers_next = self.drivers(end_time)
        observables = self.observables(
            new_state,
            params_array,
            drivers_next,
            end_time,
        )
        status = self._status(True, 0)
        return self._make_result(
            state=new_state,
            observables=observables,
            error=error,
            status=status,
            niters=0,
        )


class CPUDIRKStep(CPUStep):
    """Diagonally implicit Runge--Kutta step implementation."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        newton_rtol: float = 0.0,
        linear_rtol: float = 0.0,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        residual_reduction: Optional[float] = None,
        residual_floor: Optional[float] = None,
        tableau: Optional[DIRKTableau] = None,
        use_smoothed_error: bool = False,
        attempt_dense_prediction: bool = True,
    ) -> None:
        resolved = DEFAULT_DIRK_TABLEAU if tableau is None else tableau
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_rtol=newton_rtol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_rtol=linear_rtol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            residual_reduction=residual_reduction,
            residual_floor=residual_floor,
            tableau=resolved,
            use_smoothed_error=use_smoothed_error,
        )
        self._dirk_reference = np.zeros(
            self._state_size, dtype=self.precision
        )
        self._dirk_params = np.zeros(0, dtype=self.precision)
        self._dirk_drivers = np.zeros(0, dtype=self.precision)
        self._dirk_time = self.precision(0.0)
        self._dirk_dt = self.precision(0.0)
        self._dirk_diag_coeff = self.precision(0.0)
        self._dirk_increment = np.zeros(self._state_size, dtype=self.precision)
        self._dirk_stage_increment_history = None
        self._dirk_previous_dt = None
        # Mirrors the device gate: prediction requires the request, a
        # tableau meeting the transform preconditions, and a positive
        # calibrated ratio ceiling at the configured precision.
        self._dirk_max_ratio = self.precision(
            resolved.dense_prediction_ratio_limit(self.precision)
        )
        self._dirk_dense_prediction = bool(
            attempt_dense_prediction
            and float(self._dirk_max_ratio) > 0.0
            and tableau_supports_dense_prediction(resolved)
        )
        # Derived independently of the device mapping: a repeated
        # stage time starts from the earlier same-time stage's row.
        latest_stage_at_node = {}
        prediction_sources = []
        for stage, node in enumerate(resolved.c):
            prediction_sources.append(
                latest_stage_at_node.get(node, stage)
            )
            latest_stage_at_node[node] = stage
        self._dirk_prediction_sources = tuple(prediction_sources)

    def residual(self, candidate: Array) -> Array:
        base_state = self._dirk_reference
        increment = candidate
        stage_state = base_state + self._dirk_diag_coeff * increment
        observables = self.observables(
            stage_state,
            self._dirk_params,
            self._dirk_drivers,
            self._dirk_time,
        )
        derivative = self.rhs(
            stage_state,
            self._dirk_params,
            self._dirk_drivers,
            observables,
            self._dirk_time,
        )
        beta = self.precision(1.0)
        mass_term = self.mass_matrix_apply(increment)
        return beta * mass_term - self._dirk_dt * derivative

    def jacobian(self, candidate: Array) -> Array:
        stage_state = self._dirk_reference + self._dirk_diag_coeff * candidate
        _, jacobian = self.observables_and_jac(
            stage_state,
            self._dirk_params,
            self._dirk_drivers,
            self._dirk_time,
        )
        scale = self._dirk_dt * self._dirk_diag_coeff
        return self._identity - scale * jacobian

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
        prev_accepted: bool = True,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        stage_count = self.stage_count
        a_matrix = self.a_matrix
        b_weights = self.b_weights
        c_nodes = self.c_nodes
        error_weights = self.error_weights

        state_dim = state_vector.shape[0]
        stage_derivatives = np.zeros(
            (stage_count, state_dim),
            dtype=self.precision,
        )
        stage_states = np.zeros(
            (stage_count, state_dim),
            dtype=self.precision,
        )

        all_converged = True
        total_iters = 0

        # Mirrors the device: the persistent history is transformed
        # in place on accepted steps within the ratio bound and
        # carried unchanged otherwise; stages a solver never touches
        # (explicit stages) keep the transform's read-ahead values.
        history = None
        if self._dirk_dense_prediction:
            if self._dirk_stage_increment_history is None:
                self._dirk_stage_increment_history = np.zeros(
                    (stage_count, state_dim), dtype=self.precision
                )
            history = self._dirk_stage_increment_history
            if prev_accepted and self._dirk_previous_dt is not None:
                # Ratio and ceiling compare in configured precision
                # so the boundary agrees with the device exactly.
                ratio = self.precision(dt_value) / self.precision(
                    self._dirk_previous_dt
                )
                if ratio <= self._dirk_max_ratio:
                    predictor = np.asarray(
                        dense_predictor_matrix(self.tableau, ratio),
                        dtype=self.precision,
                    )
                    history[:] = (predictor @ history).astype(
                        self.precision
                    )
        self._dirk_previous_dt = dt_value

        for stage_index in range(stage_count):
            # A repeated stage time starts from the earlier
            # same-time stage's row, mirroring the device.
            if history is not None:
                source_stage = self._dirk_prediction_sources[
                    stage_index
                ]
                guess = history[source_stage].copy()
            else:
                guess = self._dirk_increment

            # Rounding-sensitive: scale by dt once, after accumulating.
            stage_accum = np.zeros_like(state_vector)
            for dependency in range(stage_index):
                stage_accum = stage_accum + (
                    a_matrix[stage_index, dependency]
                    * stage_derivatives[dependency]
                )
            stage_state = stage_accum * dt_value + state_vector

            stage_time = current_time + c_nodes[stage_index] * dt_value
            drivers_stage = self.drivers(stage_time)
            diag_coeff = a_matrix[stage_index, stage_index]

            if np.isclose(diag_coeff, self.precision(0.0)):
                observables_stage = self.observables(
                    stage_state,
                    params_array,
                    drivers_stage,
                    stage_time,
                )
                derivative = self.mass_matrix_solve(
                    self.rhs(
                        stage_state,
                        params_array,
                        drivers_stage,
                        observables_stage,
                        stage_time,
                    )
                )
                stage_derivatives[stage_index, :] = derivative
                stage_states[stage_index, :] = stage_state
                if history is not None:
                    # Explicit stage's free sample joins the history.
                    history[stage_index] = dt_value * derivative
                continue

            base_state = stage_state.copy()
            self._dirk_reference = base_state
            self._dirk_params = params_array
            self._dirk_drivers = drivers_stage
            self._dirk_time = stage_time
            self._dirk_dt = dt_value
            self._dirk_diag_coeff = diag_coeff
            self._linear_norm_reference = base_state

            def correction_norm(update, iterate):
                stage_value = (
                    self._dirk_reference
                    + self._dirk_diag_coeff * iterate
                )
                return correction_norm_reference(
                    update,
                    stage_value,
                    state_vector,
                    self._newton_tol,
                    self._newton_rtol,
                )

            increment, converged, niters = newton_solve(
                guess,
                precision=self.precision,
                residual_fn=self.residual,
                jacobian_fn=self.jacobian,
                linear_solver=self.linear_solve,
                newton_tol=self._newton_tol,
                newton_rtol=self._newton_rtol,
                newton_max_iters=self._newton_max_iters,
                correction_norm=correction_norm,
                prev_theta_store=self._newton_prev_theta,
            )

            solved_state = base_state + diag_coeff * increment
            stage_states[stage_index, :] = solved_state
            all_converged = all_converged and converged
            total_iters += niters
            # The derivative row is k = increment / dt.
            stage_derivatives[stage_index, :] = increment / dt_value
            self._dirk_increment = increment
            if history is not None:
                history[stage_index] = increment

        # A stage whose A row equals b (or b_hat) already holds the
        # output (or embedded) solution.
        b_row = self.tableau.b_matches_a_row
        b_hat_row = self.tableau.b_hat_matches_a_row
        if b_row is not None:
            new_state = stage_states[b_row].copy()
        else:
            state_accum = np.zeros_like(state_vector)
            for stage_index in range(stage_count):
                state_accum = state_accum + (
                    b_weights[stage_index] * stage_derivatives[stage_index]
                )
            new_state = state_vector + dt_value * state_accum

        error_accum = np.zeros_like(state_vector)
        if error_weights is not None:
            if b_hat_row is not None:
                error_accum = new_state - stage_states[b_hat_row]
            else:
                for stage_index in range(stage_count):
                    error_accum = error_accum + (
                        error_weights[stage_index]
                        * stage_derivatives[stage_index]
                    )
                error_accum = error_accum * dt_value

        if self._use_smoothed_error:
            # Solve (M - g*h*J) x = M @ raw at the final stage.
            gamma = self.precision(self.tableau.smoothing_gamma)
            final_stage_time = (
                current_time + c_nodes[stage_count - 1] * dt_value
            )
            # The device seeds the solve with the raw estimate.
            error_accum = self.smooth_error(
                self.mass_matrix_apply(error_accum),
                stage_states[stage_count - 1],
                params_array,
                self.drivers(final_stage_time),
                final_stage_time,
                gamma,
                dt_value,
                state_vector,
                initial_guess=error_accum,
            )

        end_time = current_time + dt_value
        drivers_next = self.drivers(end_time)
        observables = self.observables(
            new_state,
            params_array,
            drivers_next,
            end_time,
        )
        status = self._status(all_converged, total_iters)
        return self._make_result(
            state=new_state,
            observables=observables,
            error=error_accum,
            status=status,
            niters=total_iters,
        )


class CPUFIRKStep(CPUStep):
    """Fully implicit Runge--Kutta step implementation."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        newton_rtol: float = 0.0,
        linear_rtol: float = 0.0,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        residual_reduction: Optional[float] = None,
        residual_floor: Optional[float] = None,
        tableau: Optional[FIRKTableau] = None,
        use_smoothed_error: bool = False,
        attempt_dense_prediction: bool = True,
    ) -> None:
        resolved = DEFAULT_FIRK_TABLEAU if tableau is None else tableau
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_rtol=newton_rtol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_rtol=linear_rtol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            residual_reduction=residual_reduction,
            residual_floor=residual_floor,
            tableau=resolved,
            use_smoothed_error=use_smoothed_error,
        )
        self._firk_state = np.zeros(self._state_size, dtype=self.precision)
        self._firk_params = np.zeros(0, dtype=self.precision)
        self._firk_drivers = None
        self._firk_time = self.precision(0.0)
        self._firk_dt = self.precision(0.0)
        self._firk_stage_increments = None
        self._firk_previous_dt = None
        # Mirrors the device gate: prediction requires the request, a
        # tableau meeting the transform preconditions, and a positive
        # calibrated ratio ceiling at the configured precision.
        self._firk_max_ratio = self.precision(
            resolved.dense_prediction_ratio_limit(self.precision)
        )
        self._firk_dense_prediction = bool(
            attempt_dense_prediction
            and float(self._firk_max_ratio) > 0.0
            and tableau_supports_dense_prediction(resolved)
        )

    def residual(self, candidate: Array) -> Array:
        """Compute the residual for the fully implicit stage equations.

        For FIRK, all stages are coupled: candidate is a flattened vector
        of all stage increments k_i for i=1..s where s is the stage count.
        The residual for each stage i is:
            M * k_i - dt * f(x_0 + sum_j(a_ij * k_j), t_0 + c_i * dt)
        """
        stage_count = self.stage_count
        state_dim = self._state_size
        a_matrix = self.a_matrix
        c_nodes = self.c_nodes

        residual = np.zeros_like(candidate)

        for stage_idx in range(stage_count):
            # Extract this stage's increment from the flattened candidate
            k_start = stage_idx * state_dim
            k_end = (stage_idx + 1) * state_dim
            k_i = candidate[k_start:k_end]

            # Compute the stage state: x_0 + sum_j(a_ij * k_j)
            stage_state = self._firk_state.copy()
            for j in range(stage_count):
                j_start = j * state_dim
                j_end = (j + 1) * state_dim
                k_j = candidate[j_start:j_end]
                stage_state += a_matrix[stage_idx, j] * k_j

            # Evaluate the RHS at this stage
            stage_time = self._firk_time + c_nodes[stage_idx] * self._firk_dt
            drivers_stage = self._firk_drivers[stage_idx]
            observables_stage = self.observables(
                stage_state,
                self._firk_params,
                drivers_stage,
                stage_time,
            )
            derivative = self.rhs(
                stage_state,
                self._firk_params,
                drivers_stage,
                observables_stage,
                stage_time,
            )

            # Residual: M * k_i - dt * f(...)
            mass_term = self.mass_matrix_apply(k_i)
            residual[k_start:k_end] = mass_term - self._firk_dt * derivative

        return residual

    def jacobian(self, candidate: Array) -> Array:
        """Compute the Jacobian of the residual for the fully implicit system.

        The Jacobian is block-structured:
            J[i,j] = delta_ij * M - dt * a_ij * df/dx
        where delta_ij is the Kronecker delta.
        """
        stage_count = self.stage_count
        state_dim = self._state_size
        a_matrix = self.a_matrix
        c_nodes = self.c_nodes

        all_dim = stage_count * state_dim
        jac = np.zeros((all_dim, all_dim), dtype=self.precision)

        for stage_idx in range(stage_count):
            # Compute the stage state to evaluate the Jacobian
            stage_state = self._firk_state.copy()
            for j in range(stage_count):
                j_start = j * state_dim
                j_end = (j + 1) * state_dim
                k_j = candidate[j_start:j_end]
                stage_state += a_matrix[stage_idx, j] * k_j

            stage_time = self._firk_time + c_nodes[stage_idx] * self._firk_dt
            drivers_stage = self._firk_drivers[stage_idx]
            _, df_dx = self.observables_and_jac(
                stage_state,
                self._firk_params,
                drivers_stage,
                stage_time,
            )

            # Fill in the block row for this stage
            i_start = stage_idx * state_dim
            i_end = (stage_idx + 1) * state_dim

            for dep_idx in range(stage_count):
                j_start = dep_idx * state_dim
                j_end = (dep_idx + 1) * state_dim

                if stage_idx == dep_idx:
                    # Diagonal block: M - dt * a_ii * df/dx
                    jac[i_start:i_end, j_start:j_end] = (
                        self._identity - self._firk_dt * a_matrix[stage_idx, dep_idx] * df_dx
                    )
                else:
                    # Off-diagonal block: -dt * a_ij * df/dx
                    jac[i_start:i_end, j_start:j_end] = (
                        -self._firk_dt * a_matrix[stage_idx, dep_idx] * df_dx
                    )

        return jac

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
        prev_accepted: bool = True,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        stage_count = self.stage_count
        a_matrix = self.a_matrix
        b_weights = self.b_weights
        c_nodes = self.c_nodes
        error_weights = self.error_weights

        state_dim = state_vector.shape[0]
        all_dim = stage_count * state_dim

        # Pre-compute driver values for all stages
        stage_drivers = []
        for stage_idx in range(stage_count):
            stage_time = current_time + c_nodes[stage_idx] * dt_value
            drivers_stage = self.drivers(stage_time)
            stage_drivers.append(drivers_stage)

        # Set up the fully implicit solve
        self._firk_state = state_vector.copy()
        self._firk_params = params_array
        self._firk_drivers = stage_drivers
        self._firk_time = current_time
        self._firk_dt = dt_value
        # Mirrors the device TiledScaledNorm: the single-stage base
        # state weights every stage block of the coupled residual.
        self._linear_norm_reference = np.tile(state_vector, stage_count)

        # Accepted steps read the previous step's stage curve ahead
        # over the next step; the first step and rejected proposals
        # keep the carried increments unchanged.
        guess = np.zeros(all_dim, dtype=self.precision)
        if self._firk_stage_increments is not None:
            # Ratio and ceiling compare in configured precision so
            # the boundary agrees with the device exactly.
            ratio = self.precision(dt_value) / self.precision(
                self._firk_previous_dt
            )
            if (
                self._firk_dense_prediction
                and prev_accepted
                and ratio <= self._firk_max_ratio
            ):
                predictor = np.asarray(
                    dense_predictor_matrix(self.tableau, ratio),
                    dtype=self.precision,
                )
                previous = self._firk_stage_increments.reshape(
                    stage_count,
                    state_dim,
                )
                guess = (predictor @ previous).reshape(all_dim)
            else:
                guess = self._firk_stage_increments.copy()
        self._firk_previous_dt = dt_value

        def correction_norm(update, iterate):
            stage_values = np.zeros(all_dim, dtype=self.precision)
            for stage_idx in range(stage_count):
                stage_state = state_vector.copy()
                for j in range(stage_count):
                    j_start = j * state_dim
                    stage_state = stage_state + (
                        a_matrix[stage_idx, j]
                        * iterate[j_start:j_start + state_dim]
                    )
                i_start = stage_idx * state_dim
                stage_values[i_start:i_start + state_dim] = stage_state
            step_start_tiled = np.tile(state_vector, stage_count)
            return correction_norm_reference(
                update,
                stage_values,
                step_start_tiled,
                self._newton_tol,
                self._newton_rtol,
            )

        # Solve the fully implicit system for all stage increments simultaneously
        stage_increments_flat, converged, niters = newton_solve(
            guess,
            precision=self.precision,
            residual_fn=self.residual,
            jacobian_fn=self.jacobian,
            linear_solver=self.linear_solve,
            newton_tol=self._newton_tol,
            newton_rtol=self._newton_rtol,
            newton_max_iters=self._newton_max_iters,
            correction_norm=correction_norm,
            prev_theta_store=self._newton_prev_theta,
        )

        # Reconstruct the stage states from the solved increments; the
        # A-row shortcut below reads them directly.
        stage_states = np.zeros(
            (stage_count, state_dim),
            dtype=self.precision,
        )
        for stage_idx in range(stage_count):
            stage_state = state_vector.copy()
            for j in range(stage_count):
                j_start = j * state_dim
                j_end = (j + 1) * state_dim
                k_j = stage_increments_flat[j_start:j_end]
                stage_state += a_matrix[stage_idx, j] * k_j
            stage_states[stage_idx, :] = stage_state

        def kahan_weighted_increment_sum(weights):
            """Kahan-sum the weighted stage increments."""
            accumulator = np.zeros(state_dim, dtype=self.precision)
            compensation = np.zeros(state_dim, dtype=self.precision)
            for summed_idx in range(stage_count):
                increment_slice = stage_increments_flat[
                    summed_idx * state_dim : (summed_idx + 1) * state_dim
                ]
                weighted = weights[summed_idx] * increment_slice
                term = weighted - compensation
                temp = accumulator + term
                compensation = (temp - accumulator) - term
                accumulator = temp
            return accumulator

        # Same A-row shortcut; sums use the solved increments, which
        # already include dt.
        b_row = self.tableau.b_matches_a_row
        b_hat_row = self.tableau.b_hat_matches_a_row
        if b_row is not None:
            new_state = stage_states[b_row].copy()
        else:
            new_state = state_vector + kahan_weighted_increment_sum(
                b_weights
            )

        error_accum = np.zeros_like(state_vector)
        if error_weights is not None:
            if b_hat_row is not None and not self._use_smoothed_error:
                error_accum = new_state - stage_states[b_hat_row]
            else:
                error_accum = kahan_weighted_increment_sum(error_weights)

        if self._use_smoothed_error:
            # rhs = M @ (weighted stage sum) - gamma*h*f(y_n),
            # solved at the step-start state.
            gamma = self.precision(self.tableau.smoothing_gamma)
            drivers_start = self.drivers(current_time)
            observables_start = self.observables(
                state_vector, params_array, drivers_start, current_time
            )
            f_start = self.rhs(
                state_vector,
                params_array,
                drivers_start,
                observables_start,
                current_time,
            )
            error_accum = self.smooth_error(
                self.mass_matrix_apply(error_accum)
                - gamma * dt_value * f_start,
                state_vector,
                params_array,
                drivers_start,
                current_time,
                gamma,
                dt_value,
                state_vector,
            )

        end_time = current_time + dt_value
        drivers_next = self.drivers(end_time)
        observables = self.observables(
            new_state,
            params_array,
            drivers_next,
            end_time,
        )

        # Cache the increments for next step's initial guess
        self._firk_stage_increments = stage_increments_flat.copy()

        status = self._status(converged, niters)
        return self._make_result(
            state=new_state,
            observables=observables,
            error=error_accum,
            status=status,
            niters=niters,
        )


class CPURosenbrockWStep(CPUStep):
    """Rosenbrock--W step implementation."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        newton_rtol: float = 0.0,
        linear_rtol: float = 0.0,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        residual_reduction: Optional[float] = None,
        residual_floor: Optional[float] = None,
        tableau: Optional[RosenbrockTableau] = None,
        use_smoothed_error: bool = False,
    ) -> None:
        resolved = (
            DEFAULT_ROSENBROCK_TABLEAU if tableau is None else tableau
        )
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_rtol=newton_rtol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_rtol=linear_rtol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            residual_reduction=residual_reduction,
            residual_floor=residual_floor,
            tableau=resolved,
            use_smoothed_error=use_smoothed_error,
        )
        self._increment_cache = np.zeros(self._state_size, dtype=self.precision)

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        tol: Optional[float] = None,
        max_iters: Optional[int] = None,
        time: float = 0.0,
        prev_accepted: bool = True,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)
        end_time = current_time + dt_value
        idt = self.precision(1.0) / dt_value
        stage_count = int(self.stage_count)

        a_matrix = np.asarray(self.a_matrix, dtype=self.precision)
        b_weights = np.asarray(self.b_weights, dtype=self.precision)
        c_nodes = np.asarray(self.c_nodes, dtype=self.precision)
        C_matrix = np.asarray(self.C_matrix, dtype=self.precision)
        gamma = self.gamma
        stage_gammas = (
            self._gamma_stages
            if self._gamma_stages is not None
            else np.full(
                stage_count, self._gamma, dtype=self.precision
            )
        )
        stage_gammas = np.asarray(stage_gammas, dtype=self.precision)

        error_weights = self.error_weights
        if error_weights is not None:
            error_weights = np.asarray(error_weights, dtype=self.precision)

        state_dim = state_vector.shape[0]
        zero = self.precision(0.0)

        drivers_now = self.drivers(current_time)
        driver_rates_now = self.driver_time_derivative(current_time)
        observables_now = self.observables(
            state_vector,
            params_array,
            drivers_now,
            current_time,
        )
        f_now, symbol_values_now = self.evaluator.rhs(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )
        time_derivative_now = self.evaluator.time_derivative(
            symbol_values_now,
            driver_rates_now,
        ) * dt_value

        jacobian_now = self.evaluator.jacobian(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )

        # Direct solves weight the linear norm by the model state.
        self._linear_norm_reference = state_vector

        stage_increments = np.zeros(
            (stage_count, state_dim),
            dtype=self.precision,
        )
        state_accum = np.zeros(state_dim, dtype=self.precision)
        error_accum = np.zeros(state_dim, dtype=self.precision)

        rhs_vector = (
                f_now
                + stage_gammas[0] * time_derivative_now
        ) * gamma * dt_value

        lhs_matrix = (
            self._identity - dt_value * gamma * jacobian_now
        )
        stage_increment, converged, niters = self.linear_solve(
            lhs_matrix,
            rhs_vector,
            initial_guess=self._increment_cache,
        )
        stage_increments[0, :] = stage_increment
        state_accum = state_accum + b_weights[0] * stage_increment
        if error_weights is not None:
            error_accum = error_accum + error_weights[0] * stage_increment

        all_converged = bool(converged)
        total_iters = int(niters)

        for stage_index in range(1, stage_count):
            stage_state = state_vector.copy()
            for predecessor in range(stage_index):
                coeff = a_matrix[stage_index, predecessor]
                if coeff != zero:
                    stage_state = stage_state + (
                        coeff * stage_increments[predecessor]
                    )
            stage_time = current_time + c_nodes[stage_index] * dt_value
            drivers_stage = self.drivers(stage_time)
            observables_stage = self.observables(
                stage_state,
                params_array,
                drivers_stage,
                stage_time,
            )
            f_stage, _ = self.evaluator.rhs(
                stage_state,
                params_array,
                drivers_stage,
                observables_stage,
                stage_time,
            )
            rhs_vector = (
                    f_stage
                    + stage_gammas[stage_index]
                    * time_derivative_now
            )
            correction = np.zeros(state_dim, dtype=self.precision)
            for predecessor in range(stage_index):
                coeff = C_matrix[stage_index, predecessor]
                if coeff != zero:
                    correction += (
                        coeff * stage_increments[predecessor]
                    )
            rhs_vector = (rhs_vector + correction * idt) * gamma * dt_value

            lhs_matrix = (
                self._identity
                - dt_value * gamma * jacobian_now
            )
            initial_guess = stage_increments[stage_index - 1].copy()
            stage_increment, converged, niters = self.linear_solve(
                lhs_matrix,
                rhs_vector,
                initial_guess=initial_guess,
            )
            stage_increments[stage_index, :] = stage_increment
            state_accum = state_accum + b_weights[stage_index] * stage_increment
            if error_weights is not None:
                error_accum = error_accum + (
                    error_weights[stage_index] * stage_increment
                )
            all_converged = all_converged and bool(converged)
            total_iters += int(niters)

        new_state = state_vector + state_accum
        drivers_end = self.drivers(end_time)
        observables_end = self.observables(
            new_state,
            params_array,
            drivers_end,
            end_time,
        )
        self._increment_cache = stage_increments[-1].copy()
        status = self._status(all_converged, total_iters)
        error_vector = (
            error_accum if error_weights is not None else np.zeros_like(state_vector)
        )
        if self._use_smoothed_error:
            # The device seeds the solve with the raw estimate.
            error_vector = self.smooth_error(
                self.mass_matrix_apply(error_vector),
                state_vector,
                params_array,
                drivers_now,
                current_time,
                gamma,
                dt_value,
                state_vector,
                initial_guess=error_vector,
            )
        return self._make_result(
            state=new_state,
            observables=observables_end,
            error=error_vector,
            status=status,
            niters=total_iters,
        )


class CPUBackwardEulerPCStep(CPUStep):
    """Backward Euler predictor--corrector step."""

    def __init__(
        self,
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        newton_rtol: float = 0.0,
        linear_rtol: float = 0.0,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        residual_reduction: Optional[float] = None,
        residual_floor: Optional[float] = None,
        corrector: Optional[CPUBackwardEulerStep] = None,
    ) -> None:
        super().__init__(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_rtol=newton_rtol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_rtol=linear_rtol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            residual_reduction=residual_reduction,
            residual_floor=residual_floor,
        )
        if corrector is None:
            corrector = CPUBackwardEulerStep(
                evaluator,
                driver_evaluator,
                newton_tol=newton_tol,
                newton_rtol=newton_rtol,
                newton_max_iters=newton_max_iters,
                linear_tol=linear_tol,
                linear_rtol=linear_rtol,
                linear_max_iters=linear_max_iters,
                linear_correction_type=linear_correction_type,
                preconditioner_order=preconditioner_order,
                residual_reduction=residual_reduction,
                residual_floor=residual_floor,
            )
        self._corrector = corrector

    def step(
        self,
        *,
        state: Optional[Array] = None,
        params: Optional[Array] = None,
        dt: Optional[float] = None,
        time: float = 0.0,
        prev_accepted: bool = True,
    ) -> StepResultLike:
        state_vector = self.ensure_array(state, copy=True)
        params_array = self.ensure_array(params)
        dt_value = self.precision(dt)
        current_time = self.precision(time)

        drivers_now = self.drivers(current_time)
        observables_now = self.observables(
            state_vector,
            params_array,
            drivers_now,
            current_time,
        )
        derivative_now = self.rhs(
            state_vector,
            params_array,
            drivers_now,
            observables_now,
            current_time,
        )
        predictor = dt_value * derivative_now
        return self._corrector.step(
            state=state_vector,
            params=params_array,
            dt=dt_value,
            initial_guess=predictor,
            time=current_time,
        )


_STEP_CONSTRUCTOR_TO_CLASS = {
    ExplicitEulerStep: CPUExplicitEulerStep,
    BackwardsEulerStep: CPUBackwardEulerStep,
    BackwardsEulerPCStep: CPUBackwardEulerPCStep,
    CrankNicolsonStep: CPUCrankNicolsonStep,
    ERKStep: CPUERKStep,
    DIRKStep: CPUDIRKStep,
    FIRKStep: CPUFIRKStep,
    GenericRosenbrockWStep: CPURosenbrockWStep,
}

def _resolve_step_configuration(
    algorithm: str,
    tableau: Optional[Union[str, ButcherTableau]],
) -> tuple[type, Optional[ButcherTableau]]:
    """Resolve the step constructor and tableau for ``algorithm``."""

    try:
        step_constructor, resolved_tableau = resolve_alias(algorithm.lower())
    except KeyError as exc:
        raise ValueError(f"Unknown stepper algorithm: {algorithm}") from exc

    tableau_value = resolved_tableau
    if isinstance(tableau, str):
        try:
            override_constructor, tableau_override = resolve_alias(tableau)
        except KeyError as exc:
            raise ValueError(
                f"Unknown {step_constructor.__name__} tableau '{tableau}'."
            ) from exc
        if tableau_override is None:
            raise ValueError(
                f"Alias '{tableau}' does not reference a tableau."
            )
        if override_constructor is not step_constructor:
            raise ValueError(
                "Tableau alias does not match the requested algorithm type."
            )
        tableau_value = tableau_override
    elif isinstance(tableau, ButcherTableau):
        override_constructor, override_tableau = resolve_supplied_tableau(
            tableau
        )
        if override_constructor is not step_constructor:
            raise ValueError(
                "Tableau instance does not match the requested algorithm type."
            )
        tableau_value = override_tableau
    elif tableau is not None:
        raise TypeError(
            "Expected tableau alias or ButcherTableau instance, "
            f"received {type(tableau).__name__}."
        )

    return step_constructor, tableau_value


def get_ref_step_factory(
    algorithm: str,
    *,
    tableau: Optional[Union[str, ButcherTableau]] = None,
) -> Callable:
    """Return a factory binding the CPU stepper for ``algorithm``."""

    step_constructor, tableau_value = _resolve_step_configuration(
        algorithm,
        tableau,
    )
    try:
        step_class = _STEP_CONSTRUCTOR_TO_CLASS[step_constructor]
    except KeyError as exc:
        raise ValueError(
            f"No CPU reference implementation registered for {algorithm}."
        ) from exc

    def factory(
        evaluator: CPUODESystem,
        driver_evaluator: DriverEvaluator,
        *,
        newton_tol: float,
        newton_max_iters: int,
        linear_tol: float,
        linear_max_iters: int,
        newton_rtol: float = 0.0,
        linear_rtol: float = 0.0,
        linear_correction_type: str = "minimal_residual",
        preconditioner_order: int = 2,
        residual_reduction: Optional[float] = None,
        residual_floor: Optional[float] = None,
        attempt_dense_prediction: bool = True,
        use_smoothed_error: bool = False,
    ) -> Callable:
        extra_kwargs = {}
        if step_class in (CPUFIRKStep, CPUDIRKStep):
            extra_kwargs["attempt_dense_prediction"] = (
                attempt_dense_prediction
            )
        if step_class in (CPUFIRKStep, CPUDIRKStep, CPURosenbrockWStep):
            extra_kwargs["use_smoothed_error"] = use_smoothed_error
        if tableau_value is None:
            return step_class(
                evaluator,
                driver_evaluator,
                newton_tol=newton_tol,
                newton_rtol=newton_rtol,
                newton_max_iters=newton_max_iters,
                linear_tol=linear_tol,
                linear_rtol=linear_rtol,
                linear_max_iters=linear_max_iters,
                linear_correction_type=linear_correction_type,
                preconditioner_order=preconditioner_order,
                residual_reduction=residual_reduction,
                residual_floor=residual_floor,
                **extra_kwargs,
            )
        return step_class(
            evaluator,
            driver_evaluator,
            newton_tol=newton_tol,
            newton_rtol=newton_rtol,
            newton_max_iters=newton_max_iters,
            linear_tol=linear_tol,
            linear_rtol=linear_rtol,
            linear_max_iters=linear_max_iters,
            linear_correction_type=linear_correction_type,
            preconditioner_order=preconditioner_order,
            residual_reduction=residual_reduction,
            residual_floor=residual_floor,
            tableau=tableau_value,
            **extra_kwargs,
        )

    return factory


def get_ref_stepper(
    evaluator: CPUODESystem,
    driver_evaluator: DriverEvaluator,
    algorithm: str,
    *,
    newton_tol: float,
    newton_max_iters: int,
    linear_tol: float,
    linear_max_iters: int,
    newton_rtol: float = 0.0,
    linear_rtol: float = 0.0,
    linear_correction_type: str = "minimal_residual",
    preconditioner_order: int = 2,
    residual_reduction: Optional[float] = None,
    residual_floor: Optional[float] = None,
    tableau: Optional[Union[str, ButcherTableau]] = None,
    attempt_dense_prediction: bool = True,
    use_smoothed_error: bool = False,
) -> CPUStep:
    """Return a configured CPU reference stepper for ``algorithm``."""

    factory = get_ref_step_factory(algorithm, tableau=tableau)
    return factory(
        evaluator,
        driver_evaluator,
        newton_tol=newton_tol,
        newton_rtol=newton_rtol,
        newton_max_iters=newton_max_iters,
        linear_tol=linear_tol,
        linear_rtol=linear_rtol,
        linear_max_iters=linear_max_iters,
        linear_correction_type=linear_correction_type,
        preconditioner_order=preconditioner_order,
        residual_reduction=residual_reduction,
        residual_floor=residual_floor,
        attempt_dense_prediction=attempt_dense_prediction,
        use_smoothed_error=use_smoothed_error,
    )
