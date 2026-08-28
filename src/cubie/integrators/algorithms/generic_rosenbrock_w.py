"""Rosenbrock-W integration step as described in Lang & Verwer (2001).

Published Classes
-----------------
:class:`RosenbrockWStepConfig`
    Configuration container for the Rosenbrock-W step.

:class:`GenericRosenbrockWStep`
    Multi-stage linearly implicit step using a cached Jacobian
    approximation and linear (not Newton) solvers.

Constants
---------
:data:`ROSENBROCK_ADAPTIVE_DEFAULTS`
    Default Gustafsson controller settings for adaptive tableaus.

:data:`ROSENBROCK_FIXED_DEFAULTS`
    Default fixed-step settings for errorless tableaus.

Notes
-----
The step controller defaults are selected dynamically based on whether
the tableau has an embedded error estimate. Rosenbrock methods
linearise the ODE around the current state, avoiding iterative Newton
solves.

See Also
--------
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Abstract parent managing the solver lifecycle.
:class:`~cubie.integrators.algorithms.generic_rosenbrockw_tableaus.RosenbrockTableau`
    Tableau class describing Rosenbrock-W coefficients.
:class:`RosenbrockWStepConfig`
    Configuration for this step.

References
----------
Lang, J., Verwer, J. ROS3P—An Accurate Third-Order Rosenbrock Solver
Designed for Parabolic Problems. *BIT Numerical Mathematics* 41,
731–738 (2001).
"""

from typing import Callable, Optional

from attrs import field, validators, frozen
from cubie.cuda_simsafe import cuda, int32

from cubie.result_codes import CUBIE_RESULT_CODES
from numpy import int32 as np_int32

from cubie._utils import PrecisionDType, build_config, is_device_validator
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    AlgorithmDefaults,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
    DEFAULT_ROSENBROCK_TABLEAU,
    RosenbrockTableau,
)
from cubie.buffer_registry import buffer_registry


ROSENBROCK_SOLVER_DEFAULTS = {
    "linear_correction_type": "lu",
    "preconditioner_type": "jacobi",
}

ROSENBROCK_ADAPTIVE_DEFAULTS = AlgorithmDefaults(
    settings={
        **ROSENBROCK_SOLVER_DEFAULTS,
        "step_controller": "gustafsson",
        "min_step_shrink": 0.2,
        "max_step_growth": 8.0,
        "safety": 0.9,
    }
)
"""Defaults for adaptive Rosenbrock tableaus."""

ROSENBROCK_FIXED_DEFAULTS = AlgorithmDefaults(
    settings={
        **ROSENBROCK_SOLVER_DEFAULTS,
        "step_controller": "fixed",
    }
)
"""Defaults for errorless Rosenbrock tableaus."""


@frozen
class RosenbrockWStepConfig(ImplicitStepConfig):
    """Configuration describing the Rosenbrock-W integrator."""

    tableau: RosenbrockTableau = field(default=DEFAULT_ROSENBROCK_TABLEAU)
    time_derivative_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    driver_del_t: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    stage_rhs_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    stage_store_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    base_state_placeholder_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    krylov_iters_out_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    apply_mass_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )


class GenericRosenbrockWStep(ODEImplicitStep):
    """Rosenbrock-W step with an embedded error estimate."""

    is_linear = True

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        evaluate_f: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        driver_del_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: RosenbrockTableau = DEFAULT_ROSENBROCK_TABLEAU,
        **kwargs,
    ) -> None:
        """Initialise the Rosenbrock-W step configuration.

        This constructor creates a Rosenbrock-W step object and automatically
        selects appropriate default step controller settings based on whether
        the tableau has an embedded error estimate. Tableaus with error
        estimates default to adaptive stepping (Gustafsson controller),
        while errorless tableaus default to fixed stepping.

        Parameters
        ----------
        precision
            Floating-point precision for CUDA computations.
        n
            Number of state variables in the ODE system.
        evaluate_f
            Device function for evaluating f(t, y) right-hand side.
        evaluate_observables
            Device function computing system observables.
        evaluate_driver_at_t
            Optional device function evaluating drivers at arbitrary times.
        driver_del_t
            Optional compiled CUDA device function computing time derivatives
            of drivers (required for some Rosenbrock formulations).
        get_solver_helper_fn
            Factory function returning solver helper for Jacobian operations.
        tableau
            Rosenbrock tableau describing the coefficients and gamma values.
            Defaults to :data:`DEFAULT_ROSENBROCK_TABLEAU`.
        **kwargs
            Optional parameters passed to config classes. See
            RosenbrockWStepConfig, ImplicitStepConfig, and solver config
            classes for available parameters. None values are ignored.

        Notes
        -----
        The step controller defaults are selected dynamically:

        - If ``tableau.has_error_estimate`` is ``True``:
          Uses :data:`ROSENBROCK_ADAPTIVE_DEFAULTS` (Gustafsson
          controller)
        - If ``tableau.has_error_estimate`` is ``False``:
          Uses :data:`ROSENBROCK_FIXED_DEFAULTS` (fixed-step controller)

        This automatic selection prevents incompatible configurations where
        an adaptive controller is paired with an errorless tableau.

        Rosenbrock methods linearize the ODE around the current state,
        avoiding the need for iterative Newton solves. This makes them
        efficient for moderately stiff problems. The gamma parameter from the
        tableau controls the implicit treatment of the linearized system.
        """
        tableau_value = tableau

        config = build_config(
            RosenbrockWStepConfig,
            required={
                "precision": precision,
                "n": n,
                "evaluate_f": evaluate_f,
                "evaluate_observables": evaluate_observables,
                "evaluate_driver_at_t": evaluate_driver_at_t,
                "driver_del_t": driver_del_t,
                "get_solver_helper_fn": get_solver_helper_fn,
                "tableau": tableau_value,
                "beta": 1.0,
                "gamma": tableau_value.gamma,
            },
            **kwargs,
        )

        # Select defaults based on error estimate
        if tableau_value.has_error_estimate:
            defaults = ROSENBROCK_ADAPTIVE_DEFAULTS
        else:
            defaults = ROSENBROCK_FIXED_DEFAULTS

        super().__init__(config, defaults, **kwargs)

        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        config = self.compile_settings
        n = config.n
        tableau = config.tableau

        # Calculate buffer sizes
        stage_store_elements = tableau.stage_count * n

        # Register algorithm buffers using config values
        buffer_registry.register(
            "stage_rhs",
            self,
            n,
            config.stage_rhs_location,
        )
        buffer_registry.register(
            "stage_store",
            self,
            stage_store_elements,
            config.stage_store_location,
        )
        # cached_auxiliaries registered with 0 size; updated in
        # build_implicit_helpers
        buffer_registry.register(
            "cached_auxiliaries",
            self,
            0,
            config.cached_auxiliaries_location,
        )

        # Persists across steps; its lifetime bars aliasing stage_store.
        buffer_registry.register(
            "stage_increment",
            self,
            n,
            config.stage_store_location,
            persistent=True,
        )

        buffer_registry.register(
            "base_state_placeholder",
            self,
            1,
            config.base_state_placeholder_location,
            dtype=np_int32,
        )
        buffer_registry.register(
            "krylov_iters_out",
            self,
            1,
            config.krylov_iters_out_location,
            dtype=np_int32,
        )

    def build_implicit_helpers(
        self,
    ) -> None:
        """Construct the linear solver used by Rosenbrock methods."""
        config = self.compile_settings
        request_kwargs = self._helper_request_kwargs()

        get_fn = config.get_solver_helper_fn

        # A cached member carries prepare_jac and the aux size.
        if self.uses_direct_solver:
            lu_result = get_fn(
                "lu_solve", jacobian_at="step", **request_kwargs
            )
            prepare_jacobian = lu_result.prepare_jac
            cached_auxiliary_count = lu_result.cached_auxiliary_count
            self.solver.update(
                lu_solve_function=lu_result.device_function,
                lu_nnz=lu_result.lu_nnz,
            )
        else:
            preconditioner = get_fn(
                config.preconditioner_type,
                jacobian_at="step",
                **request_kwargs,
            ).device_function
            operator_result = get_fn(
                "linear_operator",
                jacobian_at="step",
                **request_kwargs,
            )
            prepare_jacobian = operator_result.prepare_jac
            cached_auxiliary_count = (
                operator_result.cached_auxiliary_count
            )
            self.solver.update(
                operator_apply=operator_result.device_function,
                preconditioner=preconditioner,
            )

        # Resize the zero-registered auxiliary cache to the real count.
        buffer_registry.update_buffer(
            "cached_auxiliaries",
            self,
            size=cached_auxiliary_count,
        )

        time_derivative_function = get_fn(
            "time_derivative_rhs"
        ).device_function

        apply_mass_function = None
        if self.smooth_error:
            # The smoothing rhs is M @ raw_error.
            apply_mass_function = get_fn("apply_mass").device_function

        # Return linear solver device function
        self.update_compile_settings(
            {
                "solver_function": self.solver.device_function,
                "time_derivative_function": time_derivative_function,
                "prepare_jacobian_function": prepare_jacobian,
                "apply_mass_function": apply_mass_function,
            }
        )

    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        solver_function: Callable,
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the Rosenbrock-W device step."""

        config = self.compile_settings
        tableau = config.tableau

        # Access solver from parameter
        linear_solver = solver_function
        prepare_jacobian = config.prepare_jacobian_function
        time_derivative_rhs = config.time_derivative_function
        driver_del_t = config.driver_del_t

        n = int32(n)
        stage_count = int32(self.stage_count)
        stages_except_first = stage_count - int32(1)
        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        has_error = self.is_adaptive
        use_smoothed_error = self.smooth_error
        apply_mass = config.apply_mass_function
        typed_zero = numba_precision(0.0)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)

        a_coeffs = tableau.typed_columns(tableau.a, numba_precision)
        C_coeffs = tableau.typed_columns(tableau.C, numba_precision)
        gamma_stages = tableau.typed_gamma_stages(numba_precision)
        gamma = numba_precision(tableau.gamma)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

        # Replace streaming accumulation with direct assignment when
        # stage matches b or b_hat row in coupling matrix.
        accumulates_output = tableau.accumulates_output
        accumulates_error = tableau.accumulates_error
        b_row = tableau.b_matches_a_row
        b_hat_row = tableau.b_hat_matches_a_row
        if b_row is not None:
            b_row = int32(b_row)
        if b_hat_row is not None:
            b_hat_row = int32(b_hat_row)

        # Get allocators from buffer registry
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(
                self, self.solver, name="solver"
            )
        )
        getalloc = buffer_registry.get_allocator
        alloc_stage_rhs = getalloc("stage_rhs", self)
        alloc_stage_store = getalloc("stage_store", self)
        alloc_cached_auxiliaries = getalloc("cached_auxiliaries", self)
        alloc_stage_increment = getalloc("stage_increment", self)
        alloc_base_state_placeholder = getalloc("base_state_placeholder", self)
        alloc_krylov_iters_out = getalloc("krylov_iters_out", self)

        # no cover: start
        @cuda.jit(
            # (
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision,
            #     numba_precision,
            #     int32,
            #     int32,
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     int32[::1],
            # ),
            device=True,
            inline=False,
            **self.jit_kwargs,
        )
        def step(
            state,
            proposed_state,
            parameters,
            driver_coeffs,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
            counters,
        ):
            # Allocate buffers
            stage_rhs = alloc_stage_rhs(shared, persistent_local)
            stage_store = alloc_stage_store(shared, persistent_local)
            cached_auxiliaries = alloc_cached_auxiliaries(
                shared, persistent_local
            )
            stage_increment = alloc_stage_increment(shared, persistent_local)
            base_state_placeholder = alloc_base_state_placeholder(
                shared, persistent_local
            )
            krylov_iters_out = alloc_krylov_iters_out(shared, persistent_local)
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(
                shared, persistent_local
            )
            # ----------------------------------------------------------- #

            current_time = time_scalar
            end_time = current_time + dt_scalar
            final_stage_base = n * (stage_count - int32(1))
            time_derivative = stage_store[
                final_stage_base : final_stage_base + n
            ]

            inv_dt = numba_precision(1.0) / dt_scalar

            prepare_jacobian(
                state,
                parameters,
                drivers_buffer,
                current_time,
                dt_scalar,
                cached_auxiliaries,
            )

            # Evaluate del_t term at t_n, y_n
            if has_evaluate_driver_at_t:
                driver_del_t(
                    current_time,
                    driver_coeffs,
                    proposed_drivers,
                )
            else:
                for i in range(n_drivers):
                    proposed_drivers[i] = numba_precision(0.0)

            time_derivative_rhs(
                state,
                parameters,
                drivers_buffer,
                proposed_drivers,
                observables,
                time_derivative,
                current_time,
            )

            for idx in range(n):
                proposed_state[idx] = state[idx]
                time_derivative[idx] *= dt_scalar
                if has_error:
                    error[idx] = typed_zero

            status_code = success
            stage_time = current_time + dt_scalar * stage_time_fractions[0]

            # --------------------------------------------------------------- #
            #            Stage 0: uses starting values                        #
            # --------------------------------------------------------------- #

            evaluate_f(
                state,
                parameters,
                drivers_buffer,
                observables,
                stage_rhs,
                current_time,
            )

            for idx in range(n):
                # No accumulated contributions at stage 0.
                f_value = stage_rhs[idx]
                rhs_value = (
                    f_value + gamma_stages[0] * time_derivative[idx]
                ) * dt_scalar
                stage_rhs[idx] = rhs_value * gamma

            krylov_iters_out[0] = int32(0)

            # Use stored copy as the initial guess for the first stage.
            status_code |= linear_solver(
                state,
                parameters,
                drivers_buffer,
                base_state_placeholder,
                cached_auxiliaries,
                stage_time,
                dt_scalar,
                numba_precision(1.0),
                stage_rhs,
                stage_increment,
                solver_shared,
                solver_persistent,
                krylov_iters_out,
            )

            for idx in range(n):
                stage_store[idx] = stage_increment[idx]

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] += (
                        stage_increment[idx] * solution_weights[int32(0)]
                    )
                if has_error and accumulates_error:
                    error[idx] += (
                        stage_increment[idx] * error_weights[int32(0)]
                    )

            # --------------------------------------------------------------- #
            #            Stages 1-s: must refresh all values                  #
            # --------------------------------------------------------------- #
            for prev_idx in range(stages_except_first):
                stage_idx = prev_idx + int32(1)
                stage_offset = stage_idx * n
                stage_gamma = gamma_stages[stage_idx]
                stage_time = (
                    current_time + dt_scalar * stage_time_fractions[stage_idx]
                )

                # Get base state for F(t + c_i * dt, Y_n + sum(a_ij * K_j))
                for idx in range(n):
                    stage_store[stage_offset + idx] = state[idx]

                # Accumulate contributions from predecessor stages Loop over
                # all stages for static loop bounds (better unrolling) Zero
                # coefficients from strict lower triangular structure
                for predecessor_idx in range(stages_except_first):
                    a_col = a_coeffs[predecessor_idx]
                    a_coeff = a_col[stage_idx]
                    # Only accumulate valid predecessors (coefficient will be
                    # zero for predecessor_idx >= stage_idx due to strict
                    # lower triangular structure)
                    if predecessor_idx < stage_idx:
                        base_idx = predecessor_idx * n
                        for idx in range(n):
                            prior_val = stage_store[base_idx + idx]
                            stage_store[stage_offset + idx] += (
                                a_coeff * prior_val
                            )

                for idx in range(n):
                    stage_increment[idx] = stage_store[stage_offset + idx]

                # Get t + c_i * dt parts
                if has_evaluate_driver_at_t:
                    evaluate_driver_at_t(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                evaluate_observables(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_time,
                )

                evaluate_f(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                # Capture precalculated outputs here, before overwrite
                if b_row == stage_idx:
                    for idx in range(n):
                        proposed_state[idx] = stage_increment[idx]
                if b_hat_row == stage_idx:
                    for idx in range(n):
                        error[idx] = stage_increment[idx]

                # Overwrite the final accumulator slice with time-derivative
                if stage_idx == stage_count - int32(1):
                    if has_evaluate_driver_at_t:
                        driver_del_t(
                            current_time,
                            driver_coeffs,
                            proposed_drivers,
                        )
                    time_derivative_rhs(
                        state,
                        parameters,
                        drivers_buffer,
                        proposed_drivers,
                        observables,
                        time_derivative,
                        current_time,
                    )
                    for idx in range(n):
                        time_derivative[idx] *= dt_scalar

                # Add C_ij*K_j/dt + dt * gamma_i * d/dt terms to rhs
                for idx in range(n):
                    correction = numba_precision(0.0)
                    # Loop over all stages for static loop bounds
                    for predecessor_idx in range(stages_except_first):
                        c_col = C_coeffs[predecessor_idx]
                        c_coeff = c_col[stage_idx]
                        # Only accumulate valid predecessors
                        if predecessor_idx < stage_idx:
                            prior_idx = predecessor_idx * n + idx
                            prior_val = stage_store[prior_idx]
                            correction += c_coeff * prior_val

                    f_stage_val = stage_rhs[idx]
                    deriv_val = stage_gamma * time_derivative[idx]
                    rhs_value = f_stage_val + correction * inv_dt + deriv_val
                    stage_rhs[idx] = rhs_value * dt_scalar * gamma

                # Use previous stage's solution as a guess for this stage
                previous_base = prev_idx * n

                for idx in range(n):
                    stage_increment[idx] = stage_store[previous_base + idx]

                status_code |= linear_solver(
                    state,
                    parameters,
                    drivers_buffer,
                    base_state_placeholder,
                    cached_auxiliaries,
                    stage_time,
                    dt_scalar,
                    numba_precision(1.0),
                    stage_rhs,
                    stage_increment,
                    solver_shared,
                    solver_persistent,
                    krylov_iters_out,
                )
                for idx in range(n):
                    stage_store[stage_offset + idx] = stage_increment[idx]

                if accumulates_output:
                    # Standard accumulation path for proposed_state
                    solution_weight = solution_weights[stage_idx]
                    for idx in range(n):
                        increment = stage_increment[idx]
                        proposed_state[idx] += solution_weight * increment

                if has_error:
                    if accumulates_error:
                        # Standard accumulation path for error
                        error_weight = error_weights[stage_idx]
                        for idx in range(n):
                            increment = stage_increment[idx]
                            error[idx] += error_weight * increment

            # ----------------------------------------------------------- #
            if not accumulates_error:
                for idx in range(n):
                    error[idx] = proposed_state[idx] - error[idx]

            if use_smoothed_error:
                # Dead stage_rhs holds the rhs M @ raw_error.
                apply_mass(error, stage_rhs)
                krylov_iters_out[0] = int32(0)
                status_code |= linear_solver(
                    state,
                    parameters,
                    drivers_buffer,
                    base_state_placeholder,
                    cached_auxiliaries,
                    current_time,
                    dt_scalar,
                    numba_precision(1.0),
                    stage_rhs,
                    error,
                    solver_shared,
                    solver_persistent,
                    krylov_iters_out,
                )

            if has_evaluate_driver_at_t:
                evaluate_driver_at_t(
                    end_time,
                    driver_coeffs,
                    proposed_drivers,
                )

            evaluate_observables(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

            return status_code

        # no cover: end
        return StepCache(step=step)

    @property
    def baked_stage_diagonal(self) -> float:
        """Return the 1.0 the step passes at every solver call."""
        return 1.0

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` as the method has multiple stages."""
        return self.tableau.stage_count > 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` if algorithm calculates an error estimate."""
        return self.tableau.has_error_estimate

    @property
    def is_implicit(self) -> bool:
        """Return ``True`` because the method solves linear systems."""
        return True

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""
        return self.tableau.order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""
        return 1


__all__ = [
    "GenericRosenbrockWStep",
    "RosenbrockWStepConfig",
    "RosenbrockTableau",
]
