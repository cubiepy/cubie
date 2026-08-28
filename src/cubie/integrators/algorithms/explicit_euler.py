"""Explicit Euler step implementation.

This module provides :class:`ExplicitEulerStep`, a single-stage,
non-adaptive forward Euler integrator compiled as a CUDA device
function through the :class:`CUDAFactory` pattern.

Published Classes
-----------------
:class:`ExplicitEulerStep`
    Forward Euler integration step.

    >>> from numpy import float32
    >>> step = ExplicitEulerStep(precision=float32, n=4)
    >>> step.order
    1
    >>> step.is_adaptive
    False

See Also
--------
:class:`~cubie.integrators.algorithms.ode_explicitstep.ODEExplicitStep`
    Abstract base class for explicit integration steps.
:class:`~cubie.integrators.algorithms.base_algorithm_step.BaseAlgorithmStep`
    Factory base managing compilation and caching.
"""

from typing import Callable, Optional

from cubie.cuda_simsafe import cuda, int32
from cubie.cuda_simsafe import consteval

from cubie.result_codes import CUBIE_RESULT_CODES

from cubie._utils import PrecisionDType, build_config
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    AlgorithmDefaults
from cubie.integrators.algorithms.ode_explicitstep import (
    ExplicitStepConfig,
    ODEExplicitStep,
)

EE_DEFAULTS = AlgorithmDefaults(
    settings={
        "step_controller": "fixed",
    }
)


class ExplicitEulerStep(ODEExplicitStep):
    """Forward Euler integration step for explicit ODE updates."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        evaluate_f: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """Initialise the explicit Euler step configuration.

        Parameters
        ----------
        precision
            Precision applied to device buffers.
        n
            Number of state entries advanced per step.
        evaluate_f
            Device function for evaluating f(t, y) right-hand side.
        evaluate_observables
            Device function computing system observables.
        evaluate_driver_at_t
            Optional device function evaluating drivers at arbitrary times.
        get_solver_helper_fn
            Present for interface parity with implicit steps and ignored here.
        **kwargs
            Optional parameters passed to config classes. See
            ExplicitStepConfig for available parameters. None values are
            ignored.
        """
        config = build_config(
            ExplicitStepConfig,
            required={
                'precision': precision,
                'n': n,
                'evaluate_f': evaluate_f,
                'evaluate_observables': evaluate_observables,
                'evaluate_driver_at_t': evaluate_driver_at_t,
            },
            **kwargs
        )

        super().__init__(config, EE_DEFAULTS.copy())

    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:
        """Build the device function for an explicit Euler step.

        Parameters
        ----------
        evaluate_f
            Device function for evaluating f(t, y).
        evaluate_observables
            Device function for computing observables.
        evaluate_driver_at_t
            Optional device function for evaluating drivers at time t.
        numba_precision
            Numba type for device buffers.
        n
            State vector dimension.
        n_drivers
            Number of driver signals.

        Returns
        -------
        StepCache
            Compiled step function.
        """

        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        n = int32(n)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)

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
            driver_coefficients,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,  # Non-adaptive algorithms receive a zero-length slice.
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
            counters,
        ):
            """Advance the state with a single explicit Euler update.

            Parameters
            ----------
            state
                Device array storing the current state.
            proposed_state
                Device array receiving the updated state.
            parameters
                Device array of static model parameters.
            driver_coefficients
                Device array containing spline driver coefficients.
            drivers_buffer
                Device array of time-dependent drivers.
            proposed_drivers
                Device array receiving proposed driver samples.
            observables
                Device array storing accepted observable outputs.
            proposed_observables
                Device array receiving proposed observable outputs.
            error
                Device array reserved for error estimates. Non-adaptive
                algorithms receive a zero-length slice that can be reused
                as scratch.
            dt_scalar
                Scalar containing the proposed step size.
            time_scalar
                Scalar containing the current simulation time.
            first_step_flag : int32
                Non-zero on the first step of the integration.
            accepted_flag : int32
                Non-zero when the previous step was accepted.
            shared
                Device array providing shared scratch buffers.
            persistent_local
                Device array for persistent local storage (unused here).
            counters : int32 array
                Diagnostic counter array (unused here).

            Returns
            -------
            int
                Status code indicating successful completion.
            """

            # error buffer unused; stage dx/dt in proposed_state instead.
            dxdt_buffer = proposed_state
            evaluate_f(
                state,
                parameters,
                drivers_buffer,
                observables,
                dxdt_buffer,
                time_scalar,
            )
            for i in consteval(range(n)):
                proposed_state[i] = state[i] + dt_scalar * dxdt_buffer[i]

            next_time = time_scalar + dt_scalar
            if has_evaluate_driver_at_t:
                evaluate_driver_at_t(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )
            evaluate_observables(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )
            return success
        # no cover: end

        return StepCache(step=step, nonlinear_solver=None)

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""
        return 1

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because explicit Euler is a single-stage method."""
        return False

    is_adaptive = False

    @property
    def order(self) -> int:
        """Return the classical order of the explicit Euler method."""
        return 1
