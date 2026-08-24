"""Generic explicit Runge--Kutta integration step with streamed accumulators.

This module provides :class:`ERKStep`, a configurable explicit
Runge--Kutta integrator compiled as a CUDA device function through
the :class:`CUDAFactory` pattern. The tableau is supplied at
construction time; adaptive or fixed-step controller defaults are
selected automatically based on whether the tableau includes an
embedded error estimate.

Published Classes
-----------------
:class:`ERKStepConfig`
    Configuration container for the ERK step.

    >>> from numpy import float32
    >>> from cubie.integrators.algorithms.generic_erk_tableaus import (
    ...     CLASSICAL_RK4_TABLEAU,
    ... )
    >>> config = ERKStepConfig(
    ...     precision=float32, n=3, tableau=CLASSICAL_RK4_TABLEAU,
    ... )
    >>> config.stage_count
    4

:class:`ERKStep`
    Concrete explicit Runge--Kutta step factory.

    >>> from numpy import float32
    >>> step = ERKStep(precision=float32, n=3)
    >>> step.order
    5
    >>> step.is_adaptive
    True

Module-Level Constants
----------------------
:data:`ERK_ADAPTIVE_DEFAULTS`
    Default integral controller settings applied when the tableau has
    an embedded error estimate.

:data:`ERK_FIXED_DEFAULTS`
    Default fixed-step settings applied when the tableau lacks an
    error estimate.

See Also
--------
:class:`~cubie.integrators.algorithms.ode_explicitstep.ODEExplicitStep`
    Abstract base class for explicit integration steps.
:class:`~cubie.integrators.algorithms.generic_erk_tableaus.ERKTableau`
    Butcher tableau subclass accepted by this step.
:data:`~cubie.integrators.algorithms.generic_erk_tableaus.ERK_TABLEAU_REGISTRY`
    Name-based lookup of available ERK tableaus.
"""

from typing import Callable, Optional

from attrs import field, validators, frozen
from cubie.cuda_simsafe import cuda, int32

from cubie._utils import PrecisionDType, build_config
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import all_sync, activemask
from cubie.result_codes import CUBIE_RESULT_CODES
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    AlgorithmDefaults,
)
from cubie.integrators.algorithms.ode_explicitstep import (
    ExplicitStepConfig,
    ODEExplicitStep,
)
from cubie.integrators.algorithms.generic_erk_tableaus import (
    DEFAULT_ERK_TABLEAU,
    ERKTableau,
)

ERK_ADAPTIVE_DEFAULTS = AlgorithmDefaults(
    settings={
        "step_controller": "i",
        "kp": 1.2,
        "deadband_min": 1.0,
        "deadband_max": 1.0,
        "min_gain": 0.2,
        "max_gain": 10.0,
        "safety": 0.9,
    }
)
"""Default step controller settings for adaptive ERK tableaus.

Applied when ``tableau.has_error_estimate`` is ``True``: an integral
controller with gain ``safety * err_rms^(-1.2 / (order + 1))``.
"""

ERK_FIXED_DEFAULTS = AlgorithmDefaults(
    settings={
        "step_controller": "fixed",
    }
)
"""Default step controller settings for errorless ERK tableaus.

Applied when ``tableau.has_error_estimate`` is ``False``.
"""


@frozen
class ERKStepConfig(ExplicitStepConfig):
    """Configuration describing an explicit Runge--Kutta integrator.

    Parameters
    ----------
    tableau
        Butcher tableau coefficients for the ERK method.
    stage_rhs_location
        Memory location for the stage RHS buffer (``'local'`` or
        ``'shared'``).
    stage_accumulator_location
        Memory location for the stage accumulator buffer (``'local'``
        or ``'shared'``).
    """

    tableau: ERKTableau = field(default=DEFAULT_ERK_TABLEAU)
    stage_rhs_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )
    stage_accumulator_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )

    @property
    def first_same_as_last(self) -> bool:
        """Return ``True`` when the tableau shares the first and last stage."""

        return self.tableau.first_same_as_last


class ERKStep(ODEExplicitStep):
    """Generic explicit Runge--Kutta step with configurable tableaus."""

    _ADAPTIVE_DEFAULTS = ERK_ADAPTIVE_DEFAULTS
    _FIXED_DEFAULTS = ERK_FIXED_DEFAULTS
    _DEFAULT_TABLEAU = DEFAULT_ERK_TABLEAU

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        evaluate_f: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: ERKTableau = DEFAULT_ERK_TABLEAU,
        n_drivers: int = 0,
        **kwargs,
    ) -> None:
        """Initialise the Runge--Kutta step configuration.

        This constructor creates an ERK step object and automatically selects
        appropriate default step controller settings based on whether the
        tableau has an embedded error estimate. Tableaus with error estimates
        default to adaptive stepping (integral controller), while errorless
        tableaus default to fixed stepping.

        Parameters
        ----------
        precision
            Floating-point precision for CUDA computations (np.float32 or
            np.float64).
        n
            Number of state variables in the ODE system.
        evaluate_f
            Compiled CUDA device function computing state derivatives. Should
            match signature expected by the integration kernel.
        evaluate_observables
            Optional compiled CUDA device function computing observable
            quantities from the state.
        evaluate_driver_at_t
            Optional compiled CUDA device function computing time-varying
            driver inputs.
        get_solver_helper_fn
            Factory function returning solver helper for implicit stages (not
            used in explicit methods but included for interface consistency).
        tableau
            Explicit Runge--Kutta tableau describing the coefficients used by
            the integrator. Defaults to :data:`DEFAULT_ERK_TABLEAU`
            (Dormand-Prince 5(4)).
        n_drivers
            Number of driver variables in the system.
        **kwargs
            Optional parameters passed to config classes. See ERKStepConfig
            for available parameters. None values are ignored.

        Notes
        -----
        The step controller defaults are selected dynamically:

        - If ``tableau.has_error_estimate`` is ``True``:
          Uses :data:`ERK_ADAPTIVE_DEFAULTS` (integral controller)
        - If ``tableau.has_error_estimate`` is ``False``:
          Uses :data:`ERK_FIXED_DEFAULTS` (fixed-step controller)

        This automatic selection prevents incompatible configurations where
        an adaptive controller is paired with an errorless tableau. Users
        can still override these defaults by explicitly specifying step
        controller settings when creating a :class:`Solver` or calling
        :func:`solve_ivp`.

        Examples
        --------
        Create an ERK step with the default Dormand-Prince tableau:

        >>> from cubie.integrators.algorithms.generic_erk import ERKStep
        >>> import numpy as np
        >>> step = ERKStep(precision=np.float32,n=3)
        >>> step.algorithm_defaults["step_controller"]
        'i'

        Create an ERK step with Classical RK4 (errorless):

        >>> from cubie.integrators.algorithms.generic_erk_tableaus import (
        ...     CLASSICAL_RK4_TABLEAU
        ... )
        >>> step = ERKStep(
        ...     precision=np.float32, n=3, tableau=CLASSICAL_RK4_TABLEAU
        ... )
        >>> step.algorithm_defaults["step_controller"]
        'fixed'
        """
        config = build_config(
            ERKStepConfig,
            required={
                'precision': precision,
                'n': n,
                'n_drivers': n_drivers,
                'evaluate_f': evaluate_f,
                'evaluate_observables': evaluate_observables,
                'evaluate_driver_at_t': evaluate_driver_at_t,
                'get_solver_helper_fn': get_solver_helper_fn,
                'tableau': tableau,
            },
            **kwargs
        )

        if tableau.has_error_estimate:
            defaults = ERK_ADAPTIVE_DEFAULTS
        else:
            defaults = ERK_FIXED_DEFAULTS

        super().__init__(config, defaults)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers with buffer_registry."""
        config = self.compile_settings
        n = config.n
        tableau = config.tableau

        # Calculate buffer sizes
        accumulator_length = max(tableau.stage_count - 1, 0) * n

        # Register algorithm buffers using config values
        buffer_registry.register(
            'stage_rhs',
            self,
            n,
            config.stage_rhs_location,
            persistent=True,
        )
        buffer_registry.register(
            'stage_accumulator',
            self,
            accumulator_length,
            config.stage_accumulator_location,
        )

    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the explicit Runge--Kutta device step."""

        config = self.compile_settings
        tableau = config.tableau

        typed_zero = numba_precision(0.0)
        n = int32(n)
        stage_count = int32(tableau.stage_count)
        stages_except_first = stage_count - int32(1)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)

        accumulator_length = (tableau.stage_count - 1) * n

        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        first_same_as_last = self.first_same_as_last
        multistage = stage_count > 1
        has_error = self.is_adaptive

        stage_rhs_coeffs = tableau.typed_columns(tableau.a, numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        stage_nodes = tableau.typed_vector(tableau.c, numba_precision)

        if has_error:
            error_weights = tableau.error_weights(numba_precision)
        else:
            error_weights = tuple(typed_zero for _ in range(stage_count))

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
        getalloc = buffer_registry.get_allocator
        alloc_stage_rhs = getalloc('stage_rhs', self)
        alloc_stage_accumulator = getalloc('stage_accumulator', self)

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
            """Advance the state with an explicit Runge--Kutta update.

            Parameters
            ----------
            state : array of numba_precision
                Current state vector, length ``n``.
            proposed_state : array of numba_precision
                Output buffer for the proposed state, length ``n``.
            parameters : array of numba_precision
                Static model parameters.
            driver_coeffs : array of numba_precision
                Spline driver coefficients.
            drivers_buffer : array of numba_precision
                Time-dependent driver values at current time.
            proposed_drivers : array of numba_precision
                Output buffer for driver values at proposed time.
            observables : array of numba_precision
                Accepted observable outputs.
            proposed_observables : array of numba_precision
                Output buffer for proposed observable outputs.
            error : array of numba_precision
                Output buffer for the embedded error estimate.
                Non-adaptive tableaus receive a zero-length slice.
            dt_scalar : numba_precision
                Proposed step size.
            time_scalar : numba_precision
                Current simulation time.
            first_step_flag : int32
                Non-zero on the first step of the integration.
            accepted_flag : int32
                Non-zero when the previous step was accepted.
            shared : array
                Shared memory pool.
            persistent_local : array
                Persistent local memory pool.
            counters : int32 array
                Diagnostic counter array (unused here).

            Returns
            -------
            status_code : int32
                Always ``0`` (success).
            """

            stage_rhs = alloc_stage_rhs(shared, persistent_local)
            stage_accumulator = alloc_stage_accumulator(
                shared,
                persistent_local,
            )

            current_time = time_scalar
            end_time = current_time + dt_scalar

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] = typed_zero
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            # ----------------------------------------------------------- #
            #            Stage 0: may use cached values                   #
            # ----------------------------------------------------------- #
            # Only use cache if all threads in warp can - otherwise no gain.
            use_cached_rhs = False
            if first_same_as_last and multistage:
                if not first_step_flag:
                    mask = activemask()
                    all_threads_accepted = all_sync(
                            mask,
                            accepted_flag != int32(0))
                    use_cached_rhs = all_threads_accepted
            else:
                use_cached_rhs = False

            # Keep cached rhs if able to, otherwise recalculate.
            if not multistage or not use_cached_rhs:
                evaluate_f(
                    state,
                    parameters,
                    drivers_buffer,
                    observables,
                    stage_rhs,
                    current_time,
                )

            # b weights can't match a rows for erk, as they would return 0
            # So we include ifs to skip accumulating but do no direct assign.
            for idx in range(n):
                increment = stage_rhs[idx]
                if accumulates_output:
                    proposed_state[idx] += solution_weights[0] * increment
                if has_error:
                    if accumulates_error:
                        error[idx] = error[idx] + error_weights[0] * increment

            for idx in range(accumulator_length):
                stage_accumulator[idx] = typed_zero

            # ----------------------------------------------------------- #
            #            Stages 1-s: refresh observables and drivers       #
            # ----------------------------------------------------------- #
            for prev_idx in range(stages_except_first):
                stage_offset = prev_idx * n
                stage_idx = prev_idx + int32(1)
                matrix_col = stage_rhs_coeffs[prev_idx]

                for successor_idx in range(stages_except_first):
                    coeff = matrix_col[successor_idx + int32(1)]
                    row_offset = successor_idx * n
                    for idx in range(n):
                        increment = stage_rhs[idx]
                        stage_accumulator[row_offset + idx] += (
                            coeff * increment
                        )

                base = stage_offset
                dt_stage = dt_scalar * stage_nodes[stage_idx]
                stage_time = current_time + dt_stage

                # Convert accumulated gradients sum(f(y_nj) into a state y_j
                for idx in range(n):
                    stage_accumulator[base] = (
                        stage_accumulator[base] * dt_scalar + state[idx]
                    )
                    base += int32(1)

                # get rhs for next stage
                stage_drivers = proposed_drivers
                if has_evaluate_driver_at_t:
                    evaluate_driver_at_t(
                        stage_time,
                        driver_coeffs,
                        stage_drivers,
                    )

                evaluate_observables(
                    stage_accumulator[stage_offset : stage_offset + n],
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_time,
                )

                evaluate_f(
                    stage_accumulator[stage_offset : stage_offset + n],
                    parameters,
                    stage_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                # Accumulate f(y_jn) terms or capture direct stage state
                solution_weight = solution_weights[stage_idx]
                error_weight = error_weights[stage_idx]
                for idx in range(n):
                    if accumulates_output:
                        increment = stage_rhs[idx]
                        proposed_state[idx] += solution_weight * increment

                    if has_error:
                        if accumulates_error:
                            increment = stage_rhs[idx]
                            error[idx] += error_weight * increment

            if b_row is not None:
                for idx in range(n):
                    proposed_state[idx] = stage_accumulator[
                        (b_row - 1) * n + idx
                    ]
            if b_hat_row is not None:
                for idx in range(n):
                    error[idx] = stage_accumulator[(b_hat_row - 1) * n + idx]
            # ----------------------------------------------------------- #
            for idx in range(n):
                # Scale and shift f(Y_n) value if accumulated
                if accumulates_output:
                    proposed_state[idx] = (
                        proposed_state[idx] * dt_scalar + state[idx]
                    )
                if has_error:
                    # Scale error if accumulated
                    if accumulates_error:
                        error[idx] *= dt_scalar
                    # Or form error from difference if captured from a-row
                    else:
                        error[idx] = proposed_state[idx] - error[idx]

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

            return success

        # no cover: end
        return StepCache(step=step)

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` when the method has multiple stages."""
        return self.tableau.stage_count > 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` if algorithm calculates an error estimate."""
        return self.tableau.has_error_estimate

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""

        return self.tableau.order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""

        return 1
