"""Fully implicit Runge–Kutta integration step implementation.

Published Classes
-----------------
:class:`FIRKStepConfig`
    Configuration container for the FIRK step.

:class:`FIRKStep`
    Multi-stage fully implicit step solving all stages as a coupled
    nonlinear system. Uses Kahan summation for output accumulation.

Constants
---------
:data:`FIRK_ADAPTIVE_DEFAULTS`
    Default Gustafsson controller settings for adaptive tableaus.

:data:`FIRK_FIXED_DEFAULTS`
    Default fixed-step settings for errorless tableaus.

Notes
-----
The step controller defaults are selected dynamically based on whether
the tableau has an embedded error estimate. FIRK methods require
solving a coupled system of all stages simultaneously, which is more
expensive than DIRK methods but can achieve higher orders for stiff
systems.

See Also
--------
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Abstract parent managing the Newton-Krylov solver lifecycle.
:class:`~cubie.integrators.algorithms.generic_firk_tableaus.FIRKTableau`
    Tableau class describing FIRK coefficients.
:class:`FIRKStepConfig`
    Configuration for this step.
"""

from typing import Callable, Optional

from attrs import field, validators, frozen
from numpy import int32 as np_int32
from cubie.cuda_simsafe import cuda, int32

from cubie.result_codes import CUBIE_RESULT_CODES

from cubie._utils import (
    PrecisionDType,
    build_config,
    is_device_validator,
)
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    AlgorithmDefaults,
)
from cubie.integrators.algorithms.generic_firk_tableaus import (
    DEFAULT_FIRK_TABLEAU,
    FIRKTableau,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from cubie.integrators.norms import (
    FIRKCorrectionNorm,
    ScaledNorm,
    TiledScaledNorm,
)
from cubie.integrators.stage_predictors import DenseStagePredictor
from cubie.buffer_registry import buffer_registry


FIRK_SOLVER_DEFAULTS = {
    "linear_correction_type": "lu",
    "inexact_newton": True,
    "prefactored": True,
    "preconditioner_type": "jacobi",
    "preconditioner_order": 0,
}

FIRK_ADAPTIVE_DEFAULTS = AlgorithmDefaults(
    settings={
        **FIRK_SOLVER_DEFAULTS,
        "step_controller": "gustafsson",
        "min_step_factor": 0.2,
        "max_step_factor": 8.0,
        "safety": 0.9,
    }
)
"""Defaults for adaptive FIRK tableaus."""

FIRK_FIXED_DEFAULTS = AlgorithmDefaults(
    settings={
        **FIRK_SOLVER_DEFAULTS,
        "step_controller": "fixed",
    }
)
"""Defaults for errorless FIRK tableaus."""


@frozen
class FIRKStepConfig(ImplicitStepConfig):
    """Configuration describing the FIRK integrator.

    Attributes
    ----------
    tableau : FIRKTableau
        Butcher tableau describing the fully implicit method.
    attempt_dense_prediction : bool
        Request dense stage prediction: accepted steps warm-start
        Newton by reading the previous step's stage curve ahead over
        the next step. Ignored when the tableau does not meet the
        transform's preconditions.
    predictor_function : Callable or None
        Compiled dense-prediction device function, piped through
        compile settings so predictor rebuilds invalidate the step.
    stage_increment_location : str
        Buffer location for the coupled stage-increment vector.
    previous_step_size_location : str
        Buffer location for the previous-step-size scalar consumed
        by dense prediction.
    stage_driver_stack_location : str
        Buffer location for the per-stage driver samples.
    stage_state_location : str
        Buffer location for the stage-state scratch vector.
    """

    tableau: FIRKTableau = field(
        default=DEFAULT_FIRK_TABLEAU,
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
        default="local", validator=validators.in_(["local", "shared"])
    )
    previous_step_size_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    stage_driver_stack_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    stage_state_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    apply_mass_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""

        return self.tableau.stage_count

    @property
    def smoothed_error_weights(self) -> tuple:
        """Return the smoothed error weights cast to precision."""
        return self.tableau.smoothed_error_weights(self.precision)

    @property
    def solver_width(self) -> int:
        """Return the coupled solver width across all stages."""

        return self.stage_count * self.n


class FIRKStep(ODEImplicitStep):
    """Fully implicit Runge--Kutta step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        evaluate_f: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: FIRKTableau = DEFAULT_FIRK_TABLEAU,
        n_drivers: int = 0,
        attempt_dense_prediction: bool = True,
        **kwargs,
    ) -> None:
        """Initialise the FIRK step configuration.

        This constructor creates a FIRK step object and automatically selects
        appropriate default step controller settings based on whether the
        tableau has an embedded error estimate. Tableaus with error estimates
        default to adaptive stepping (Gustafsson controller), while
        errorless tableaus default to fixed stepping.

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
            FIRK tableau describing the coefficients. Defaults to
            :data:`DEFAULT_FIRK_TABLEAU`.
        n_drivers
            Number of driver variables in the system.
        attempt_dense_prediction
            Request dense stage prediction; ignored when the tableau
            does not meet the transform's preconditions.
        **kwargs
            Optional parameters passed to config classes. See
            FIRKStepConfig, ImplicitStepConfig, and solver config classes
            for available parameters. None values are ignored.

        Notes
        -----
        The step controller defaults are selected dynamically:

        - If ``tableau.has_error_estimate`` is ``True``:
          Uses :data:`FIRK_ADAPTIVE_DEFAULTS` (Gustafsson controller)
        - If ``tableau.has_error_estimate`` is ``False``:
          Uses :data:`FIRK_FIXED_DEFAULTS` (fixed-step controller)

        This automatic selection prevents incompatible configurations where
        an adaptive controller is paired with an errorless tableau.

        FIRK methods require solving a coupled system of all stages
        simultaneously, which is more computationally expensive than DIRK
        methods but can achieve higher orders of accuracy for stiff systems.

        ``use_smoothed_error`` defaults on when the tableau supports it.
        """

        # Default to smoothed error true if the tableau supports it.
        if (
            kwargs.get("use_smoothed_error") is None
            and tableau.supports_smoothed_error
        ):
            kwargs["use_smoothed_error"] = True
        config = build_config(
            FIRKStepConfig,
            required={
                "precision": precision,
                "n": n,
                "n_drivers": n_drivers,
                "evaluate_f": evaluate_f,
                "evaluate_observables": evaluate_observables,
                "evaluate_driver_at_t": evaluate_driver_at_t,
                "get_solver_helper_fn": get_solver_helper_fn,
                "tableau": tableau,
                "attempt_dense_prediction": attempt_dense_prediction,
                "beta": 1.0,
                "gamma": 1.0,
            },
            **kwargs,
        )

        # Select defaults based on error estimate
        if tableau.has_error_estimate:
            defaults = FIRK_ADAPTIVE_DEFAULTS
        else:
            defaults = FIRK_FIXED_DEFAULTS

        newton_norm = FIRKCorrectionNorm(
            precision=precision,
            solver_width=config.solver_width,
            n=n,
            stage_coefficients=tableau.a_flat(float),
            instance_label="newton",
            **kwargs,
        )
        # The coupled solve stacks all stages, but callers pass the
        # single-stage base state; the tiled norm reuses it per stage.
        krylov_norm = TiledScaledNorm(
            precision=precision,
            solver_width=config.solver_width,
            n=n,
            instance_label="krylov",
            **kwargs,
        )
        super().__init__(
            config,
            defaults,
            newton_norm=newton_norm,
            krylov_norm=krylov_norm,
            **kwargs,
        )
        self.dense_predictor = DenseStagePredictor(
            precision=self.compile_settings.precision,
            n=n,
            tableau=self.compile_settings.tableau,
        )
        self.register_buffers()

    def _build_error_solver(self) -> None:
        """Construct the width-n smoothing solver from live settings."""
        config = self.compile_settings
        # Build a second, n-wide solver for the smoothed error
        # estimation.
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
        all_stages_n = tableau.stage_count * n
        stage_driver_stack_elements = tableau.stage_count * config.n_drivers

        buffer_registry.register_child(
            self, self.solver, name="solver"
        )
        buffer_registry.register_child(
            self, self.dense_predictor, name="dense_predictor"
        )
        buffer_registry.register(
            "stage_increment",
            self,
            all_stages_n,
            config.stage_increment_location,
            persistent=True,
        )
        buffer_registry.register(
            "previous_step_size",
            self,
            1 if self.dense_prediction else 0,
            config.previous_step_size_location,
            persistent=True,
        )
        buffer_registry.register(
            "stage_driver_stack",
            self,
            stage_driver_stack_elements,
            config.stage_driver_stack_location,
        )
        buffer_registry.register(
            "stage_state",
            self,
            n,
            config.stage_state_location,
        )
        # Frozen-Jacobian cache; resized in build_implicit_helpers.
        buffer_registry.register(
            "cached_auxiliaries",
            self,
            0,
            config.cached_auxiliaries_location,
        )
        buffer_registry.register(
            "error_solve_iters",
            self,
            1 if self.smooth_error else 0,
            "local",
            dtype=np_int32,
        )
        if self.smooth_error:
            # Reuse solver_shared, which is unused after the solve completes.
            buffer_registry.register_child(
                self,
                self.error_solver,
                name="error_solver",
                aliases="solver_shared",
            )

    def build_implicit_helpers(
        self,
    ) -> None:
        """Construct the nonlinear solver chain used by implicit methods."""

        config = self.compile_settings
        tableau = config.tableau

        get_fn = config.get_solver_helper_fn

        stage_kwargs = dict(
            self._helper_request_kwargs(),
            stage_coefficients=tableau.stage_coefficients,
            stage_nodes=tableau.stage_nodes,
        )

        residual = get_fn(
            "residual",
            jacobian_at="stage",
            stacked=True,
            **stage_kwargs,
        ).device_function

        prepare_function = None
        cached_count = 0
        if self.uses_cached_solve:
            if self.uses_direct_solver:
                # Eigenvalue block-transform solve on frozen J.
                lu_result = get_fn(
                    "lu_solve",
                    jacobian_at="step",
                    prefactored=True,
                    stacked=True,
                    **stage_kwargs,
                )
                prepare_function = lu_result.prepare_jac
                cached_count = lu_result.cached_auxiliary_count
                self.solver.update(
                    lu_solve_function=lu_result.device_function,
                    lu_nnz=lu_result.lu_nnz,
                    residual_function=residual,
                    use_cached_auxiliaries=True,
                    solver_width=config.solver_width,
                )
            else:
                operator_result = get_fn(
                    "linear_operator",
                    jacobian_at="step",
                    stacked=True,
                    **stage_kwargs,
                )
                preconditioner = get_fn(
                    config.preconditioner_type,
                    jacobian_at="step",
                    stacked=True,
                    **stage_kwargs,
                ).device_function
                prepare_function = operator_result.prepare_jac
                cached_count = operator_result.cached_auxiliary_count
                self.solver.update(
                    operator_apply=operator_result.device_function,
                    preconditioner=preconditioner,
                    residual_function=residual,
                    use_cached_auxiliaries=True,
                    solver_width=config.solver_width,
                )
        elif self.uses_direct_solver:
            # Coupled all-stages factorisation per Newton iteration.
            lu_result = get_fn(
                "lu_solve",
                jacobian_at="stage",
                stacked=True,
                **stage_kwargs,
            )
            self.solver.update(
                lu_solve_function=lu_result.device_function,
                lu_nnz=lu_result.lu_nnz,
                residual_function=residual,
                use_cached_auxiliaries=False,
                solver_width=config.solver_width,
            )
        else:
            operator = get_fn(
                "linear_operator",
                jacobian_at="stage",
                stacked=True,
                **stage_kwargs,
            ).device_function

            preconditioner = get_fn(
                config.preconditioner_type,
                jacobian_at="stage",
                stacked=True,
                **stage_kwargs,
            ).device_function

            self.solver.update(
                operator_apply=operator,
                preconditioner=preconditioner,
                residual_function=residual,
                use_cached_auxiliaries=False,
                solver_width=config.solver_width,
            )

        buffer_registry.update_buffer(
            "cached_auxiliaries", self, size=cached_count
        )

        if self.smooth_error:
            if self.uses_direct_solver:
                if self.uses_cached_solve:
                    # The smoothing solve substitutes against the
                    # transform's real-eigenvalue block factors.
                    smoothing = get_fn(
                        "lu_smoothing_solve",
                        jacobian_at="step",
                        prefactored=True,
                        stacked=True,
                        **stage_kwargs,
                    )
                    self.error_solver.update(
                        lu_solve_function=smoothing.device_function,
                        lu_nnz=smoothing.lu_nnz,
                        solver_width=config.n,
                    )
                else:
                    lu_at_state = get_fn(
                        "lu_solve",
                        jacobian_at="state",
                        **stage_kwargs,
                    )
                    self.error_solver.update(
                        lu_solve_function=lu_at_state.device_function,
                        lu_nnz=lu_at_state.lu_nnz,
                        solver_width=config.n,
                    )
            else:
                self.error_solver.update(
                    operator_apply=get_fn(
                        "linear_operator",
                        jacobian_at="state",
                        **stage_kwargs,
                    ).device_function,
                    preconditioner=get_fn(
                        config.preconditioner_type,
                        jacobian_at="state",
                        **stage_kwargs,
                    ).device_function,
                    solver_width=config.n,
                )

        self.update_compile_settings(
            {
                "solver_function": self.solver.device_function,
                "prepare_jacobian_function": prepare_function,
                "predictor_function": (
                    self.dense_predictor.device_function
                    if self.dense_prediction
                    else None
                ),
                "error_solver_function": (
                    self.error_solver.device_function
                    if self.smooth_error
                    else None
                ),
                "apply_mass_function": (
                    get_fn("apply_mass").device_function
                    if self.smooth_error
                    else None
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
        """Compile the FIRK device step."""

        config = self.compile_settings
        tableau = config.tableau

        use_dense_prediction = self.dense_prediction
        predict_stages = config.predictor_function
        use_smoothed_error = self.smooth_error
        error_solver = config.error_solver_function
        apply_mass = config.apply_mass_function
        use_cached_solve = self.uses_cached_solve
        prepare_jacobian = config.prepare_jacobian_function

        nonlinear_solver = solver_function

        n = int32(n)
        n_drivers = int32(n_drivers)
        stage_count = int32(self.stage_count)

        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        has_error = self.is_adaptive

        stage_rhs_coeffs = tableau.a_flat(numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
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

        smoothing_gamma = config.smoothing_gamma
        if use_smoothed_error:
            # Smoothed error always accumulates.
            error_weights = config.smoothed_error_weights
            accumulates_error = True

        ends_at_one = stage_time_fractions[-1] == numba_precision(1.0)
        max_step_ratio = tableau.dense_prediction_ratio_limit(
            config.precision
        )

        # Get allocators from buffer registry
        getalloc = buffer_registry.get_allocator
        alloc_stage_increment = getalloc("stage_increment", self)
        alloc_stage_driver_stack = getalloc("stage_driver_stack", self)
        alloc_stage_state = getalloc("stage_state", self)
        alloc_previous_step_size = getalloc("previous_step_size", self)
        alloc_error_solve_iters = getalloc("error_solve_iters", self)
        alloc_cached_aux = getalloc("cached_auxiliaries", self)

        # Re-register the solver child under the same name as
        # register_buffers so the size snapshot reflects the solver's
        # fully built buffer group (the instance owns the group; the
        # compiled device function does not).
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(
                self, self.solver, name="solver"
            )
        )
        alloc_predictor_shared, alloc_predictor_persistent = (
            buffer_registry.get_child_allocators(
                self, self.dense_predictor, name="dense_predictor"
            )
        )
        alloc_error_shared = None
        alloc_error_persistent = None
        if use_smoothed_error:
            alloc_error_shared, alloc_error_persistent = (
                buffer_registry.get_child_allocators(
                    self,
                    self.error_solver,
                    name="error_solver",
                    aliases="solver_shared",
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
            inline=True,
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
            # ----------------------------------------------------------- #
            # Selective allocation from local or shared memory
            # ----------------------------------------------------------- #
            stage_state = alloc_stage_state(shared, persistent_local)
            cached_aux = alloc_cached_aux(shared, persistent_local)
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(
                shared, persistent_local
            )
            stage_increment = alloc_stage_increment(shared, persistent_local)
            stage_driver_stack = alloc_stage_driver_stack(
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
                error_shared = alloc_error_shared(shared, persistent_local)
                error_persistent = alloc_error_persistent(
                    shared, persistent_local
                )

            # ----------------------------------------------------------- #

            current_time = time_scalar
            end_time = current_time + dt_scalar
            status_code = success

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
                first_step = first_step_flag != int32(0)
                previous_accepted = accepted_flag != int32(0)
                apply_prediction = (
                    (not first_step)
                    and previous_accepted
                    and (step_ratio <= max_step_ratio)
                )
                predict_stages(
                    stage_increment,
                    step_ratio,
                    apply_prediction,
                    predictor_shared,
                    predictor_persistent,
                )

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] = state[idx]
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            # Fill stage_drivers_stack if driver arrays provided
            if has_evaluate_driver_at_t:
                for stage_idx in range(stage_count):
                    stage_time = (
                        current_time
                        + dt_scalar * stage_time_fractions[stage_idx]
                    )
                    driver_offset = stage_idx * n_drivers
                    driver_slice = stage_driver_stack[
                        driver_offset : driver_offset + n_drivers
                    ]
                    evaluate_driver_at_t(
                        stage_time, driver_coeffs, driver_slice
                    )

            if use_cached_solve:
                # Freeze the Jacobian at the step-start state.
                status_code |= prepare_jacobian(
                    state,
                    parameters,
                    drivers_buffer,
                    current_time,
                    dt_scalar,
                    cached_aux,
                )
            # Solve n-stage nonlinear problem for all stages
            solver_status = nonlinear_solver(
                stage_increment,
                parameters,
                stage_driver_stack,
                cached_aux,
                current_time,
                dt_scalar,
                typed_zero,
                state,
                state,
                solver_shared,
                solver_persistent,
                counters,
            )
            status_code = int32(status_code | solver_status)

            for stage_idx in range(stage_count):
                if has_evaluate_driver_at_t:
                    stage_base = stage_idx * n_drivers
                    for idx in range(n_drivers):
                        proposed_drivers[idx] = stage_driver_stack[
                            stage_base + idx
                        ]

                for idx in range(n):
                    value = state[idx]
                    for contrib_idx in range(stage_count):
                        flat_idx = stage_idx * stage_count + contrib_idx
                        increment_idx = contrib_idx * n
                        coeff = stage_rhs_coeffs[flat_idx]
                        if coeff != typed_zero:
                            value += (
                                coeff * stage_increment[increment_idx + idx]
                            )
                    stage_state[idx] = value

                # Capture precalculated outputs if tableau allows
                if not accumulates_output:
                    if b_row == stage_idx:
                        for idx in range(n):
                            proposed_state[idx] = stage_state[idx]
                if not accumulates_error:
                    if b_hat_row == stage_idx:
                        for idx in range(n):
                            error[idx] = stage_state[idx]

            # Kahan summation to reduce floating point errors
            # see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            if accumulates_output:
                for idx in range(n):
                    solution_acc = typed_zero
                    compensation = typed_zero
                    for stage_idx in range(stage_count):
                        increment_value = stage_increment[stage_idx * n + idx]
                        weighted = (
                            solution_weights[stage_idx] * increment_value
                        )
                        term = weighted - compensation
                        temp = solution_acc + term
                        compensation = (temp - solution_acc) - term
                        solution_acc = temp
                    proposed_state[idx] = state[idx] + solution_acc

            if has_error and accumulates_error:
                # Standard accumulation path for error
                for idx in range(n):
                    error_acc = typed_zero
                    compensation = typed_zero
                    for stage_idx in range(stage_count):
                        increment_value = stage_increment[stage_idx * n + idx]
                        weighted = error_weights[stage_idx] * increment_value
                        term = weighted - compensation
                        temp = error_acc + term
                        compensation = (temp - error_acc) - term
                        error_acc = temp
                    error[idx] = error_acc

            if use_smoothed_error:
                # rhs = M @ (weighted stage sum) - gamma*h*f(y_n),
                # stored in the unused stage_state buffer.
                apply_mass(error, stage_state)
                evaluate_f(
                    state,
                    parameters,
                    drivers_buffer,
                    observables,
                    error,
                    current_time,
                )
                for idx in range(n):
                    stage_state[idx] = (
                        stage_state[idx]
                        - smoothing_gamma * dt_scalar * error[idx]
                    )
                    error[idx] = stage_state[idx]
                error_solve_iters[0] = int32(0)

                # Solve error at step-start jacobian, discard status.
                error_solver(
                    state,
                    parameters,
                    drivers_buffer,
                    state,
                    cached_aux,
                    current_time,
                    dt_scalar,
                    smoothing_gamma,
                    stage_state,
                    error,
                    error_shared,
                    error_persistent,
                    error_solve_iters,
                )

            if not ends_at_one:
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

            if not accumulates_error:
                for idx in range(n):
                    error[idx] = proposed_state[idx] - error[idx]

            return status_code

        # no cover: end
        return StepCache(step=step, nonlinear_solver=nonlinear_solver)

    @property
    def baked_stage_diagonal(self) -> None:
        """Return ``None``: smoothing solves use transform eigenvalues."""
        return None

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` as the method has multiple stages."""

        return self.stage_count > 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` when the tableau supplies an error estimate."""

        return self.tableau.has_error_estimate

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""
        return self.compile_settings.stage_count

    @property
    def is_implicit(self) -> bool:
        """Return ``True`` because the method solves nonlinear systems."""
        return True

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""
        return self.tableau.order

    @property
    def controller_order(self) -> int:
        """Return the order of accuracy used for step-size control."""
        if self.smooth_error:
            tableau = self.compile_settings.tableau
            return min(self.order, tableau.smoothed_embedded_order)
        return super().controller_order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""
        return 1
