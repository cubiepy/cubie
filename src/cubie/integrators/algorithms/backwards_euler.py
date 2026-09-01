"""Backward Euler step implementation using Newton–Krylov.

Published Classes
-----------------
:class:`BackwardsEulerStepConfig`
    Configuration container for the backward Euler step.

:class:`BackwardsEulerStep`
    Single-stage, first-order implicit step with persistent increment
    cache for warm-starting Newton iterations.

Constants
---------
:data:`ALGO_CONSTANTS`
    Beta and gamma values for backward Euler (unity).

:data:`BE_DEFAULTS`
    Default step controller settings (fixed-step, dt=1e-3).

See Also
--------
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Abstract parent managing the Newton–Krylov solver lifecycle.
:class:`BackwardsEulerStepConfig`
    Configuration for this step.
"""

from typing import Callable, Optional

from attrs import field, validators, frozen
from cubie.cuda_simsafe import cuda, int32
from cubie.cuda_simsafe import unroll_if

from cubie._utils import PrecisionDType, build_config
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    AlgorithmDefaults
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig, ODEImplicitStep
)


@frozen
class BackwardsEulerStepConfig(ImplicitStepConfig):
    """Configuration for Backwards Euler step with buffer location."""

    increment_cache_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )


ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0}

BE_DEFAULTS = AlgorithmDefaults(
    settings={
        "step_controller": "fixed",
        "preconditioner_type": "jacobi",
    }
)


class BackwardsEulerStep(ODEImplicitStep):
    """Backward Euler step solved with matrix-free Newton–Krylov."""

    # The single stage solves with a_ij = 1.
    _PREFACTOR_STAGE_DATA = (((1.0,),), (1.0,))
    _BAKED_STAGE_DIAGONAL = 1.0

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
        """Initialise the backward Euler step configuration.

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
            Callable returning device helpers used by the nonlinear solver.
        **kwargs
            Optional parameters passed to config classes. See
            BackwardsEulerStepConfig, ImplicitStepConfig, and solver config
            classes for available parameters. None values are ignored.
        """
        beta = ALGO_CONSTANTS['beta']
        gamma = ALGO_CONSTANTS['gamma']

        config = build_config(
            BackwardsEulerStepConfig,
            required={
                'precision': precision,
                'n': n,
                'evaluate_f': evaluate_f,
                'evaluate_observables': evaluate_observables,
                'evaluate_driver_at_t': evaluate_driver_at_t,
                'get_solver_helper_fn': get_solver_helper_fn,
                'beta': beta,
                'gamma': gamma,
            },
            **kwargs
        )

        super().__init__(config, BE_DEFAULTS.copy(), **kwargs)

        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers with buffer_registry."""
        config = self.compile_settings

        # Register solver child buffers
        buffer_registry.register_child(
            self, self.solver, name='solver_scratch'
        )

        # Register increment cache buffer
        buffer_registry.register(
            'increment_cache',
            self,
            config.n,
            config.increment_cache_location,
            persistent=True,
        )

        # Frozen-Jacobian cache; resized in build_implicit_helpers.
        buffer_registry.register(
            'cached_auxiliaries',
            self,
            0,
            config.cached_auxiliaries_location,
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
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for a backward Euler step.

        Parameters
        ----------
        evaluate_f
            Device function for evaluating f(t, y).
        evaluate_observables
            Device function for computing observables.
        evaluate_driver_at_t
            Optional device function for evaluating drivers at time t.
        solver_function
            Device function for the Newton-Krylov nonlinear solver.
        numba_precision
            Numba precision corresponding to the configured precision.
        n
            Dimension of the state vector.
        n_drivers
            Number of driver signals provided to the system.

        Returns
        -------
        StepCache
            Container holding the compiled step function and solver.
        """
        a_ij = numba_precision(1.0)
        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        n = int32(n)
        unroll_step_element = self.compile_settings.unroll.step_element

        use_cached_solve = self.uses_cached_solve
        prepare_jacobian = (
            self.compile_settings.prepare_jacobian_function
        )

        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, self.solver,
                                                 name='solver_scratch')
        )

        # Get increment cache allocator from buffer_registry
        alloc_increment_cache = buffer_registry.get_allocator(
            'increment_cache', self
        )
        alloc_cached_aux = buffer_registry.get_allocator(
            'cached_auxiliaries', self
        )

        solver_fn = solver_function

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
            """Perform one backward Euler update.

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
                Device array capturing solver diagnostics. Fixed-step
                algorithms receive a zero-length slice that can be repurposed
                as scratch when available.
            dt_scalar
                Scalar containing the proposed step size.
            time_scalar
                Scalar containing the current simulation time.
            shared
                Device array providing shared scratch buffers.
            first_step_flag
                Non-zero on the first integration step.
            accepted_flag
                Non-zero when the previous step was accepted.
            persistent_local
                Device array for persistent local storage (unused here).
            counters
                Integer array for Newton iteration counters.

            Returns
            -------
            int
                Status code returned by the nonlinear solver.
            """
            solver_scratch = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(
                shared,
                persistent_local,
            )
            increment_cache = alloc_increment_cache(shared, persistent_local)
            cached_aux = alloc_cached_aux(shared, persistent_local)

            for i in unroll_if(range(n), unroll_step_element):
                proposed_state[i] = increment_cache[i]

            next_time = time_scalar + dt_scalar
            if has_evaluate_driver_at_t:
                evaluate_driver_at_t(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            status = int32(0)
            if use_cached_solve:
                # Freeze the Jacobian at the step-start state.
                status = prepare_jacobian(
                    state,
                    parameters,
                    proposed_drivers,
                    next_time,
                    dt_scalar,
                    cached_aux,
                )
            status |= solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                cached_aux,
                next_time,
                dt_scalar,
                a_ij,
                state,
                state,
                solver_scratch,
                solver_persistent,
                counters,
            )

            for i in unroll_if(range(n), unroll_step_element):
                increment_cache[i] = proposed_state[i]
                proposed_state[i] += state[i]

            evaluate_observables(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            return status

        # no cover: end
        return StepCache(step=step, nonlinear_solver=solver_fn)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because backward Euler is a single-stage method."""

        return False

    # Class attribute so alias-level queries can read adaptivity
    # without an instance; backward Euler has no error estimate.
    is_adaptive = False

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""

        return 1

    @property
    def order(self) -> int:
        """Return the classical order of the backward Euler method."""
        return 1
