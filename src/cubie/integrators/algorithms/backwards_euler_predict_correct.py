"""Backward Euler step with an explicit predictor and implicit corrector.

Published Classes
-----------------
:class:`BackwardsEulerPCStep`
    Subclass of :class:`~backwards_euler.BackwardsEulerStep` that
    evaluates an explicit forward-Euler predictor before running the
    implicit Newton–Krylov corrector.

See Also
--------
:class:`~cubie.integrators.algorithms.backwards_euler.BackwardsEulerStep`
    Parent class providing configuration, buffer registration, and
    default controller settings.
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Abstract base managing the Newton–Krylov solver lifecycle.
"""

from typing import Callable, Optional

from cubie.cuda_simsafe import cuda, int32

from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.backwards_euler import BackwardsEulerStep
from cubie.integrators.algorithms.base_algorithm_step import StepCache


class BackwardsEulerPCStep(BackwardsEulerStep):
    """Backward Euler with a predictor-corrector refinement."""

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
        """Build the device function for the predictor-corrector scheme.

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
            Container holding the compiled predictor-corrector step.
        """
        a_ij = numba_precision(1.0)
        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        n = int32(n)

        use_cached_solve = self.uses_cached_solve
        prepare_jacobian = (
            self.compile_settings.prepare_jacobian_function
        )

        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, self.solver,
                                                 name='solver')
        )
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
            """Advance the state using an explicit predictor and implicit
            corrector.

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

            predictor = alloc_increment_cache(shared, persistent_local)
            solver_scratch = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared,
                                                        persistent_local)
            cached_aux = alloc_cached_aux(shared, persistent_local)
            evaluate_f(
                state,
                parameters,
                drivers_buffer,
                observables,
                predictor,
                time_scalar,
            )
            for i in range(n):
                proposed_state[i] = dt_scalar * predictor[i]

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

            for i in range(n):
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
