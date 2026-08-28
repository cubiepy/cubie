"""Diagonally implicit Runge–Kutta integration step implementation.

Published Classes
-----------------
:class:`DIRKStepConfig`
    Configuration container for the DIRK step.

:class:`DIRKStep`
    Multi-stage implicit step supporting configurable DIRK Butcher
    tableaus with FSAL and stage-skipping compile-time optimisations.

Constants
---------
:data:`DIRK_ADAPTIVE_DEFAULTS`
    Default order-dependent PI controller settings for adaptive
    tableaus.

:data:`DIRK_FIXED_DEFAULTS`
    Default fixed-step settings for errorless tableaus.

Notes
-----
The step controller defaults are selected dynamically based on whether
the tableau has an embedded error estimate. Tableaus with error
estimates default to adaptive stepping with order-dependent PI
controller defaults, while errorless tableaus default to fixed
stepping.

See Also
--------
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Abstract parent managing the Newton–Krylov solver lifecycle.
:class:`~cubie.integrators.algorithms.generic_dirk_tableaus.DIRKTableau`
    Tableau class describing DIRK coefficients.
:class:`DIRKStepConfig`
    Configuration for this step.
"""

from typing import Callable, Optional

from attrs import field, validators, frozen
from numpy import int32 as np_int32
from cubie.cuda_simsafe import cuda, int32

from cubie._utils import (
    PrecisionDType,
    build_config,
    is_device_validator,
)
from cubie.cuda_simsafe import activemask, all_sync
from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    AlgorithmDefaults,
)
from cubie.integrators.algorithms.generic_dirk_tableaus import (
    DEFAULT_DIRK_TABLEAU,
    DIRKTableau,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from cubie.integrators.norms import ScaledNorm
from cubie.integrators.stage_predictors import DenseStagePredictor
from cubie.buffer_registry import buffer_registry


def dirk_default_integral_gain(order: int) -> float:
    """Return the DIRK integral gain for an algorithm order."""
    return 0.3 * (order + 1) / order


def dirk_default_proportional_gain(order: int) -> float:
    """Return the DIRK proportional gain for an algorithm order."""
    return 0.4 * (order + 1) / order


DIRK_SOLVER_DEFAULTS = {
    "linear_correction_type": "lu",
    "inexact_newton": True,
    "prefactored": True,
    "preconditioner_type": "jacobi",
}

DIRK_ADAPTIVE_DEFAULTS = AlgorithmDefaults(
    settings={
        **DIRK_SOLVER_DEFAULTS,
        "attempt_dense_prediction": False,
        "step_controller": "pi",
        "integral_gain": dirk_default_integral_gain,
        "proportional_gain": dirk_default_proportional_gain,
        "min_step_shrink": 0.2,
        "max_step_growth": 10.0,
        "safety": 0.9,
    }
)
"""Defaults for adaptive DIRK tableaus."""


DIRK_FIXED_DEFAULTS = AlgorithmDefaults(
    settings={
        **DIRK_SOLVER_DEFAULTS,
        "step_controller": "fixed",
    }
)
"""Defaults for errorless DIRK tableaus."""


@frozen
class DIRKStepConfig(ImplicitStepConfig):
    """Configuration describing the DIRK integrator.

    Attributes
    ----------
    tableau : DIRKTableau
        Butcher tableau describing the diagonally implicit method.
    attempt_dense_prediction : bool
        Request dense stage prediction: accepted steps warm-start
        each stage's Newton solve by reading the previous step's
        stage curve ahead over the next step. Ignored when the
        tableau does not meet the transform's preconditions.
    predictor_function : Callable or None
        Compiled dense-prediction device function, piped through
        compile settings so predictor rebuilds invalidate the step.
    stage_increment_location : str
        Buffer location for the working stage-increment vector.
    stage_increment_history_location : str
        Buffer location for the previous step's per-stage increment
        history consumed by dense prediction.
    previous_step_size_location : str
        Buffer location for the previous-step-size scalar consumed
        by dense prediction.
    stage_base_location : str
        Buffer location for the stage base-state vector.
    accumulator_location : str
        Buffer location for the explicit stage accumulator.
    stage_rhs_location : str
        Buffer location for the cached stage effective derivative.
    apply_mass_function : Callable or None
        Compiled mass-matrix product used by error smoothing.
    evaluate_inv_mass_f_function : Callable or None
        Compiled effective derivative ``M**-1 @ f`` for explicit
        stages.
    """

    tableau: DIRKTableau = field(
        default=DEFAULT_DIRK_TABLEAU,
    )
    attempt_dense_prediction: bool = field(
        default=True, validator=validators.instance_of(bool)
    )
    predictor_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    stage_increment_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )
    stage_increment_history_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )
    previous_step_size_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )
    stage_base_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )
    accumulator_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )
    stage_rhs_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )
    apply_mass_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    evaluate_inv_mass_f_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )


class DIRKStep(ODEImplicitStep):
    """Diagonally implicit Runge–Kutta step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        evaluate_f: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
        n_drivers: int = 0,
        attempt_dense_prediction: bool = True,
        **kwargs,
    ) -> None:
        """Initialise the DIRK step configuration.

        This constructor creates a DIRK step object and automatically selects
        appropriate default step controller settings based on whether the
        tableau has an embedded error estimate. Tableaus with error estimates
        default to adaptive stepping with order-dependent PI controller
        defaults, while errorless tableaus default to fixed stepping.

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
        get_solver_helper_fn
            Factory function returning solver helper for Jacobian operations.
        tableau
            DIRK tableau describing the coefficients. Defaults to
            :data:`DEFAULT_DIRK_TABLEAU`.
        n_drivers
            Number of driver variables in the system.
        attempt_dense_prediction
            Request dense stage prediction; ignored when the tableau
            does not meet the transform's preconditions.
        **kwargs
            Optional parameters passed to config classes. See
            DIRKStepConfig, ImplicitStepConfig, and solver config classes
            for available parameters. None values are ignored.

        Notes
        -----
        The step controller defaults are selected dynamically:

        - If ``tableau.has_error_estimate`` is ``True``:
          Uses :data:`DIRK_ADAPTIVE_DEFAULTS` (order-dependent PI
          controller defaults)
        - If ``tableau.has_error_estimate`` is ``False``:
          Uses :data:`DIRK_FIXED_DEFAULTS` (fixed-step controller)

        This automatic selection prevents incompatible configurations where
        an adaptive controller is paired with an errorless tableau.
        """
        config = build_config(
            DIRKStepConfig,
            required={
                'precision': precision,
                'n': n,
                'n_drivers': n_drivers,
                'evaluate_f': evaluate_f,
                'evaluate_observables': evaluate_observables,
                'evaluate_driver_at_t': evaluate_driver_at_t,
                'get_solver_helper_fn': get_solver_helper_fn,
                'tableau': tableau,
                'attempt_dense_prediction': attempt_dense_prediction,
                'beta': 1.0,
                'gamma': 1.0,
            },
            **kwargs
        )

        # Select defaults based on error estimate
        if tableau.has_error_estimate:
            defaults = DIRK_ADAPTIVE_DEFAULTS
        else:
            defaults = DIRK_FIXED_DEFAULTS

        super().__init__(config, defaults, **kwargs)

        settings = self.compile_settings
        self.dense_predictor = DenseStagePredictor(
            precision=settings.precision,
            n=n,
            tableau=settings.tableau,
            **kwargs,
        )
        self.register_buffers()

    def _build_error_solver(self) -> None:
        """Construct the width-n smoothing solver from live settings."""
        config = self.compile_settings
        # Smoothing solves with the at-state operator family.
        carried = {
            key: value
            for key, value in self.linear_solver.settings_dict.items()
            if key in self._LINEAR_SOLVER_PARAMS and value is not None
        }
        norm_kwargs = {
            key: carried[key]
            for key in ("krylov_atol", "krylov_rtol")
            if key in carried
        }
        # Smoothing solves warm-start from the raw error estimate.
        self.error_solver = self._construct_linear_solver(
            precision=config.precision,
            solver_width=config.n,
            norm=ScaledNorm(
                precision=config.precision,
                solver_width=config.n,
                n=config.n,
                instance_label="krylov",
                **norm_kwargs,
            ),
            norm_reference="base_state",
            zero_initial_guess=False,
            **carried,
        )

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        config = self.compile_settings
        n = config.n
        tableau = config.tableau
        if self.smooth_error and self.error_solver is None:
            self._build_error_solver()

        # Clear this step's own registrations only: child factories
        # keep their still-valid declarations, and register_child
        # below re-records them with fresh sizes.
        buffer_registry.clear_own(self)

        # Calculate buffer sizes
        accumulator_length = max(tableau.stage_count - 1, 0) * n
        history_length = (
            tableau.stage_count * n if self.dense_prediction else 0
        )
        previous_step_size_length = 1 if self.dense_prediction else 0

        # Register solver scratch and solver persistent buffers so they can
        # be aliased
        buffer_registry.register_child(
                self,
                self.solver,
                name='solver'
        )
        buffer_registry.register_child(
            self, self.dense_predictor, name='dense_predictor'
        )
        buffer_registry.register(
            'stage_increment_history',
            self,
            history_length,
            config.stage_increment_history_location,
            persistent=True,
        )
        buffer_registry.register(
            'previous_step_size',
            self,
            previous_step_size_length,
            config.previous_step_size_location,
            persistent=True,
        )
        buffer_registry.register(
            'stage_increment',
            self,
            n,
            config.stage_increment_location,
            persistent=True,
        )
        buffer_registry.register(
            'accumulator',
            self,
            accumulator_length,
            config.accumulator_location,
        )

        buffer_registry.register(
            'stage_base',
            self,
            n,
            config.stage_base_location,
            aliases='accumulator',
        )

        buffer_registry.register(
            'stage_rhs',
            self,
            n,
            config.stage_rhs_location,
            persistent=True,
        )

        # Frozen-Jacobian cache; resized in build_implicit_helpers.
        buffer_registry.register(
            'cached_auxiliaries',
            self,
            0,
            config.cached_auxiliaries_location,
        )

        buffer_registry.register(
            'error_solve_iters',
            self,
            1 if self.smooth_error else 0,
            'local',
            dtype=np_int32,
        )
        if self.smooth_error:
            # Reuse solver_shared, which is unused after the solve
            # completes.
            buffer_registry.register_child(
                self,
                self.error_solver,
                name='error_solver',
                aliases='solver_shared',
            )
        # error_rhs packs in after the error solver's window.
        buffer_registry.register(
            'error_rhs',
            self,
            n if self.smooth_error else 0,
            'local',
            aliases='solver_shared' if self.smooth_error else None,
        )

    def build_implicit_helpers(
        self,
    ) -> None:
        """Construct the nonlinear solver chain used by implicit methods."""

        super().build_implicit_helpers()

        config = self.compile_settings
        request_kwargs = self._helper_request_kwargs()
        get_fn = config.get_solver_helper_fn

        apply_mass_function = None
        if self.smooth_error:
            # Smoothing solves at the accepted state, not an increment.
            if self.uses_direct_solver:
                lu_at_state = get_fn(
                    "lu_solve", jacobian_at="state", **request_kwargs
                )
                self.error_solver.update(
                    lu_solve_function=lu_at_state.device_function,
                    lu_nnz=lu_at_state.lu_nnz,
                )
            else:
                self.error_solver.update(
                    operator_apply=get_fn(
                        "linear_operator",
                        jacobian_at="state",
                        **request_kwargs,
                    ).device_function,
                    preconditioner=get_fn(
                        config.preconditioner_type,
                        jacobian_at="state",
                        **request_kwargs,
                    ).device_function,
                )
            # The smoothing rhs is M @ raw_error.
            apply_mass_function = get_fn("apply_mass").device_function

        # Explicit stages evaluate k = M**-1 @ f in one call.
        evaluate_inv_mass_f_function = None
        if config.tableau.has_explicit_stage:
            evaluate_inv_mass_f_function = get_fn(
                "evaluate_inv_mass_f"
            ).device_function

        self.update_compile_settings(
            {
                'predictor_function': (
                    self.dense_predictor.device_function
                    if self.dense_prediction
                    else None
                ),
                'error_solver_function': (
                    self.error_solver.device_function
                    if self.smooth_error
                    else None
                ),
                'apply_mass_function': apply_mass_function,
                'evaluate_inv_mass_f_function': (
                    evaluate_inv_mass_f_function
                ),
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
        """Compile the DIRK device step."""

        config = self.compile_settings
        tableau = config.tableau
        nonlinear_solver = solver_function

        use_dense_prediction = self.dense_prediction
        predict_stages = config.predictor_function
        use_smoothed_error = self.smooth_error
        error_solver = config.error_solver_function
        smoothing_gamma = config.smoothing_gamma
        apply_mass = config.apply_mass_function
        evaluate_inv_mass_f = config.evaluate_inv_mass_f_function
        has_explicit_stage = evaluate_inv_mass_f is not None
        use_cached_solve = self.uses_cached_solve
        prepare_jacobian = config.prepare_jacobian_function

        n = int32(n)
        stage_count = int32(tableau.stage_count)
        stages_except_first = stage_count - int32(1)

        # Compile-time toggles
        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        has_error = self.is_adaptive
        multistage = stage_count > 1
        first_same_as_last = self.first_same_as_last
        can_reuse_accepted_start = self.can_reuse_accepted_start

        explicit_a_coeffs = tableau.explicit_terms(numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)
        diagonal_coeffs = tableau.typed_vector(
            tableau.diagonal, numba_precision
        )

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

        stage_implicit = tuple(coeff != numba_precision(0.0)
                               for coeff in diagonal_coeffs)
        first_stage_implicit = bool(stage_implicit[0])
        prediction_source_stages = tableau.prediction_source_stages
        max_step_ratio = tableau.dense_prediction_ratio_limit(
            config.precision
        )
        accumulator_length = int32(max(stage_count - 1, 0) * n)

        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, self.solver,
                                                 name='solver')
        )

        # Get allocators from buffer registry
        getalloc = buffer_registry.get_allocator
        alloc_stage_increment = getalloc('stage_increment', self)
        alloc_accumulator = getalloc('accumulator', self)
        alloc_stage_base = getalloc('stage_base', self)
        alloc_stage_rhs = getalloc('stage_rhs', self)
        alloc_stage_increment_history = getalloc(
            'stage_increment_history', self
        )
        alloc_previous_step_size = getalloc(
            'previous_step_size', self
        )
        alloc_predictor_shared, alloc_predictor_persistent = (
            buffer_registry.get_child_allocators(
                self, self.dense_predictor, name='dense_predictor'
            )
        )
        alloc_error_solve_iters = getalloc('error_solve_iters', self)
        alloc_error_rhs = getalloc('error_rhs', self)
        alloc_cached_aux = getalloc('cached_auxiliaries', self)
        alloc_error_shared = None
        alloc_error_persistent = None
        if use_smoothed_error:
            # Duplicate the nonlinear solver's linear-solver allocators.
            alloc_error_shared, alloc_error_persistent = (
                buffer_registry.get_child_allocators(
                    self,
                    self.error_solver,
                    name='error_solver',
                    aliases='solver_shared',
                )
            )

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

            stage_increment = alloc_stage_increment(shared, persistent_local)
            stage_accumulator = alloc_accumulator(shared, persistent_local)
            stage_base = alloc_stage_base(shared, persistent_local)
            cached_aux = alloc_cached_aux(shared, persistent_local)
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(
                shared,
                persistent_local,
            )
            stage_rhs = alloc_stage_rhs(shared, persistent_local)
            stage_increment_history = alloc_stage_increment_history(
                shared, persistent_local
            )
            previous_step_size = alloc_previous_step_size(
                shared, persistent_local
            )
            predictor_shared = alloc_predictor_shared(
                shared, persistent_local
            )
            predictor_persistent = alloc_predictor_persistent(
                shared, persistent_local
            )
            if use_smoothed_error:
                error_solve_iters = alloc_error_solve_iters(
                    shared, persistent_local
                )
                error_rhs = alloc_error_rhs(shared, persistent_local)
                error_shared = alloc_error_shared(
                    shared, persistent_local
                )
                error_persistent = alloc_error_persistent(
                    shared, persistent_local
                )

            for _i in range(accumulator_length):
                stage_accumulator[_i] = typed_zero
            # --------------------------------------------------------------- #

            current_time = time_scalar
            end_time = current_time + dt_scalar

            for idx in range(n):
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            status_code = success

            if use_cached_solve:
                # Every stage reads one step-start preparation.
                status_code |= prepare_jacobian(
                    state,
                    parameters,
                    drivers_buffer,
                    current_time,
                    dt_scalar,
                    cached_aux,
                )
            # --------------------------------------------------------------- #
            #            Stage 0: may reuse cached values                     #
            # --------------------------------------------------------------- #

            first_step = first_step_flag != int32(0)

            if use_dense_prediction:
                previous_dt = previous_step_size[0]
                # Safe to store on a rejected attempt: prediction is
                # skipped after a rejection, so a rejected size is
                # never consumed.
                previous_step_size[0] = dt_scalar
                # Zeroed first-step storage keeps the ratio finite.
                safe_previous_dt = (
                    previous_dt
                    if previous_dt > typed_zero
                    else dt_scalar
                )
                step_ratio = dt_scalar / safe_previous_dt
                # Predict only from an accepted step within the ceiling.
                previous_accepted = accepted_flag != int32(0)
                apply_prediction = (
                    (not first_step)
                    and previous_accepted
                    and (step_ratio <= max_step_ratio)
                )
                predict_stages(
                    stage_increment_history,
                    step_ratio,
                    apply_prediction,
                    predictor_shared,
                    predictor_persistent,
                )

            # Only use cache if all threads in warp can - otherwise no gain
            use_cached_rhs = False
            # Compile-time branch: guarded by static configuration flags
            if first_same_as_last and multistage:
                # Runtime branch: depends on previous step acceptance
                if not first_step:
                    mask = activemask()
                    all_threads_accepted = all_sync(
                        mask,
                        accepted_flag != int32(0),
                    )
                    use_cached_rhs = all_threads_accepted
            else:
                use_cached_rhs = False

            stage_time = current_time + dt_scalar * stage_time_fractions[0]
            diagonal_coeff = diagonal_coeffs[0]

            for idx in range(n):
                stage_base[idx] = state[idx]
                if accumulates_output:
                    proposed_state[idx] = typed_zero

            # Recompute if not FSAL cached
            if not use_cached_rhs:
                if can_reuse_accepted_start:
                    for idx in range(int32(drivers_buffer.shape[0])):
                        # Use step-start driver values
                        proposed_drivers[idx] = drivers_buffer[idx]

                else:
                    if has_evaluate_driver_at_t:
                        evaluate_driver_at_t(
                            stage_time,
                            driver_coeffs,
                            proposed_drivers,
                        )

                if stage_implicit[0]:
                    if use_dense_prediction:
                        for idx in range(n):
                            stage_increment[idx] = (
                                stage_increment_history[idx]
                            )
                    solver_status = nonlinear_solver(
                        stage_increment,
                        parameters,
                        proposed_drivers,
                        cached_aux,
                        stage_time,
                        dt_scalar,
                        diagonal_coeffs[0],
                        stage_base,
                        state,
                        solver_shared,
                        solver_persistent,
                        counters,
                    )
                    status_code = int32(status_code | solver_status)

                    if use_dense_prediction:
                        for idx in range(n):
                            stage_increment_history[idx] = (
                                stage_increment[idx]
                            )

                    # stage_rhs holds the derivative k = K / dt.
                    for idx in range(n):
                        stage_base[idx] += (
                            diagonal_coeff * stage_increment[idx]
                        )
                        stage_rhs[idx] = (
                            stage_increment[idx] / dt_scalar
                        )
                elif has_explicit_stage:
                    evaluate_observables(
                        stage_base,
                        parameters,
                        proposed_drivers,
                        proposed_observables,
                        stage_time,
                    )
                    evaluate_inv_mass_f(
                        stage_base,
                        parameters,
                        proposed_drivers,
                        proposed_observables,
                        stage_rhs,
                        stage_time,
                    )

            if use_dense_prediction and not first_stage_implicit:
                # An explicit first stage's history row is dt * k.
                for idx in range(n):
                    stage_increment_history[idx] = (
                        dt_scalar * stage_rhs[idx]
                    )

            solution_weight = solution_weights[0]
            error_weight = error_weights[0]
            for idx in range(n):
                rhs_value = stage_rhs[idx]
                # Accumulate if required; save directly if tableau allows
                if accumulates_output:
                    # Standard accumulation
                    proposed_state[idx] += solution_weight * rhs_value
                elif b_row == int32(0):
                    # Direct assignment when stage 0 matches b_row
                    proposed_state[idx] = stage_base[idx]
                if has_error:
                    if accumulates_error:
                        # Standard accumulation
                        error[idx] += error_weight * rhs_value
                    elif b_hat_row == int32(0):
                        # Direct assignment for error
                        error[idx] = stage_base[idx]

            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            # --------------------------------------------------------------- #
            #            Stages 1-s: must refresh all qtys                    #
            # --------------------------------------------------------------- #
            mask = activemask()
            for prev_idx in range(stages_except_first):

                stage_offset = prev_idx * n
                stage_idx = prev_idx + int32(1)
                matrix_col = explicit_a_coeffs[prev_idx]

                # Stream previous stage's RHS into accumulators for successors
                for successor_idx in range(stages_except_first):
                    coeff = matrix_col[successor_idx + int32(1)]
                    row_offset = successor_idx * n
                    for idx in range(n):
                        contribution = coeff * stage_rhs[idx]
                        stage_accumulator[row_offset + idx] += contribution

                stage_time = (
                    current_time + dt_scalar * stage_time_fractions[stage_idx]
                )

                if has_evaluate_driver_at_t:
                    evaluate_driver_at_t(
                        stage_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

                # Convert accumulator slice to state by adding y_n
                for idx in range(n):
                    stage_base[idx] = (stage_accumulator[stage_offset + idx]
                                       * dt_scalar + state[idx])

                diagonal_coeff = diagonal_coeffs[stage_idx]

                if stage_implicit[stage_idx]:
                    if use_dense_prediction:
                        history_offset = stage_idx * n
                        source_offset = (
                            prediction_source_stages[stage_idx] * n
                        )
                        for idx in range(n):
                            stage_increment[idx] = (
                                stage_increment_history[
                                    source_offset + idx
                                ]
                            )
                    solver_status = nonlinear_solver(
                        stage_increment,
                        parameters,
                        proposed_drivers,
                        cached_aux,
                        stage_time,
                        dt_scalar,
                        diagonal_coeffs[stage_idx],
                        stage_base,
                        state,
                        solver_shared,
                        solver_persistent,
                        counters,
                    )
                    status_code = int32(status_code | solver_status)

                    if use_dense_prediction:
                        for idx in range(n):
                            stage_increment_history[
                                history_offset + idx
                            ] = stage_increment[idx]

                    # stage_rhs holds the derivative k = K / dt.
                    for idx in range(n):
                        stage_base[idx] += (
                            diagonal_coeff * stage_increment[idx]
                        )
                        stage_rhs[idx] = (
                            stage_increment[idx] / dt_scalar
                        )
                elif has_explicit_stage:
                    evaluate_observables(
                        stage_base,
                        parameters,
                        proposed_drivers,
                        proposed_observables,
                        stage_time,
                    )
                    evaluate_inv_mass_f(
                        stage_base,
                        parameters,
                        proposed_drivers,
                        proposed_observables,
                        stage_rhs,
                        stage_time,
                    )

                    if use_dense_prediction:
                        # Store the explicit stage's free sample.
                        history_offset = stage_idx * n
                        for idx in range(n):
                            stage_increment_history[
                                history_offset + idx
                            ] = dt_scalar * stage_rhs[idx]

                solution_weight = solution_weights[stage_idx]
                error_weight = error_weights[stage_idx]

                # Accumulate output/error or write directly if possible
                for idx in range(n):
                    increment = stage_rhs[idx]
                    if accumulates_output:
                        proposed_state[idx] += solution_weight * increment
                    elif b_row == stage_idx:
                        proposed_state[idx] = stage_base[idx]

                    if has_error:
                        if accumulates_error:
                            error[idx] += error_weight * increment
                        elif b_hat_row == stage_idx:
                            error[idx] = stage_base[idx]

            # --------------------------------------------------------------- #

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] *= dt_scalar
                    proposed_state[idx] += state[idx]
                if has_error:
                    if accumulates_error:
                        error[idx] *= dt_scalar
                    else:
                        error[idx] = proposed_state[idx] - error[idx]

            if use_smoothed_error:
                # Solve (M - g*h*J) x = M @ raw at the final stage.
                apply_mass(error, error_rhs)
                error_solve_iters[0] = int32(0)
                # Don't keep the error solve's status, it doesn't hurt the step
                error_solver(
                    stage_base,
                    parameters,
                    proposed_drivers,
                    state,
                    cached_aux,
                    stage_time,
                    dt_scalar,
                    smoothing_gamma,
                    error_rhs,
                    error,
                    error_shared,
                    error_persistent,
                    error_solve_iters,
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

            return int32(status_code)
        # no cover: end
        return StepCache(step=step, nonlinear_solver=nonlinear_solver)

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` as the method has multiple stages."""
        return self.tableau.stage_count > 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` because an embedded error estimate is produced."""
        return self.tableau.has_error_estimate

    @property
    def is_implicit(self) -> bool:
        """Return ``True`` because the method solves nonlinear systems."""
        return True

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""
        return self.tableau.order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""
        return 1
