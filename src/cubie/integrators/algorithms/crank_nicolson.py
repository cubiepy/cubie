"""Crank–Nicolson step with embedded backward Euler error estimation.

Published Classes
-----------------
:class:`CrankNicolsonStepConfig`
    Configuration container for the Crank–Nicolson step.

:class:`CrankNicolsonStep`
    Second-order adaptive implicit step. The error estimate is the
    difference between the Crank–Nicolson and backward Euler solutions,
    computed by solving two implicit systems per step.

Constants
---------
:data:`CN_DEFAULTS`
    Default Gustafsson adaptive controller settings.

See Also
--------
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Abstract parent managing the Newton–Krylov solver lifecycle.
:class:`CrankNicolsonStepConfig`
    Configuration for this step.
"""

from typing import Callable, Optional

from attrs import field, validators, frozen
from cubie.cuda_simsafe import cuda, int32

from cubie._utils import PrecisionDType, build_config
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms import ImplicitStepConfig
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    AlgorithmDefaults
from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep

ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0}

CN_DEFAULTS = AlgorithmDefaults(
    settings={
        "step_controller": "gustafsson",
        "preconditioner_type": "jacobi",
        "min_step_shrink": 0.2,
        "max_step_growth": 8.0,
        "safety": 0.9,
    }
)
"""Gustafsson controller defaults for Crank--Nicolson."""


@frozen
class CrankNicolsonStepConfig(ImplicitStepConfig):
    """Configuration for Crank-Nicolson step."""

    dxdt_location: str = field(
        default='local',
        validator=validators.in_(['local', 'shared'])
    )


class CrankNicolsonStep(ODEImplicitStep):
    """Crank–Nicolson step with embedded backward Euler error estimation."""

    # Diagonals 0.5 (trapezoidal) and 1.0 (backward Euler companion).
    _PREFACTOR_STAGE_DATA = (((0.5, 0.0), (0.0, 1.0)), (1.0, 1.0))

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
        """Initialise the Crank–Nicolson step configuration.

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
            CrankNicolsonStepConfig, ImplicitStepConfig, and solver config
            classes for available parameters. None values are ignored.
        """
        beta = ALGO_CONSTANTS['beta']
        gamma = ALGO_CONSTANTS['gamma']

        config = build_config(
            CrankNicolsonStepConfig,
            required={
                'precision': precision,
                'n': n,
                'get_solver_helper_fn': get_solver_helper_fn,
                'beta': beta,
                'gamma': gamma,
                'evaluate_f': evaluate_f,
                'evaluate_observables': evaluate_observables,
                'evaluate_driver_at_t': evaluate_driver_at_t,
            },
            **kwargs
        )

        super().__init__(config, CN_DEFAULTS.copy(), **kwargs)

        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers with buffer_registry."""
        config = self.compile_settings
        # Register solver child buffers

        buffer_registry.register_child(
            self, self.solver, name='solver'
        )

        buffer_registry.register(
            'cn_dxdt',
            self,
            config.n,
            config.dxdt_location,
            aliases='solver_shared',
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
        """Build the device function for the Crank–Nicolson step.

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
        stage_coefficient = numba_precision(0.5)
        be_coefficient = numba_precision(1.0)
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
        alloc_dxdt = buffer_registry.get_allocator('cn_dxdt', self)
        alloc_cached_aux = buffer_registry.get_allocator(
            'cached_auxiliaries', self
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
            driver_coefficients,
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
            """Advance the state using Crank–Nicolson with embedded error
            check.

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
                Device array capturing embedded error estimates.
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

            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(
                shared,
                persistent_local,
            )
            dxdt = alloc_dxdt(shared, persistent_local)
            cached_aux = alloc_cached_aux(shared, persistent_local)

            # base_state aliases error as their lifetimes are disjoint
            base_state = error

            # Evaluate f(state)
            evaluate_f(
                state,
                parameters,
                drivers_buffer,
                observables,
                dxdt,
                time_scalar,
            )

            half_dt = dt_scalar * numba_precision(0.5)
            end_time = time_scalar + dt_scalar

            # Form the Crank-Nicolson stage base
            for i in range(n):
                base_state[i] = state[i] + half_dt * dxdt[i]
                proposed_state[i] = dt_scalar * dxdt[i]

            # Solve Crank-Nicolson step (main solution)
            if has_evaluate_driver_at_t:
                evaluate_driver_at_t(
                    end_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            status = int32(0)
            if use_cached_solve:
                # Both solves share one step-start preparation.
                status = prepare_jacobian(
                    state,
                    parameters,
                    proposed_drivers,
                    end_time,
                    dt_scalar,
                    cached_aux,
                )
            status |= solver_function(
                proposed_state,
                parameters,
                proposed_drivers,
                cached_aux,
                end_time,
                dt_scalar,
                stage_coefficient,
                base_state,
                state,
                solver_shared,
                solver_persistent,
                counters,
            )

            for i in range(n):
                increment = proposed_state[i]
                proposed_state[i] = (
                    base_state[i] + stage_coefficient * increment
                )
                base_state[i] = increment

            status |= solver_function(
                base_state,
                parameters,
                proposed_drivers,
                cached_aux,
                end_time,
                dt_scalar,
                be_coefficient,
                state,
                state,
                solver_shared,
                solver_persistent,
                counters,
            )

            # Compute error as difference between Crank-Nicolson and Backward
            # Euler
            for i in range(n):
                error[i] = proposed_state[i] - (state[i] + base_state[i])

            evaluate_observables(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

            return status

        # no cover: end
        return StepCache(step=step, nonlinear_solver=solver_function)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because Crank–Nicolson is a single-stage method."""

        return False

    # Class attribute so alias-level queries can read adaptivity
    # without an instance; the embedded estimate enables adaptivity.
    is_adaptive = True

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""

        return 1

    @property
    def order(self) -> int:
        """Return the classical order of the Crank–Nicolson method."""

        return 2
