"""Backward differentiation formula (BDF) integration step.

Published Classes
-----------------
:class:`BDFStepConfig`
    Configuration container for the BDF step.

:class:`BDFStep`
    Variable-step BDF; one Newton solve per step.

Constants
---------
:data:`BDF_DEFAULTS`
    Default controller and solver settings for BDF tableaus.

See Also
--------
:class:`~cubie.integrators.step_history.StepHistory`
    Owned history that supplies the corrector and predictor.
:class:`~cubie.integrators.algorithms.generic_bdf_tableaus.BDFTableau`
    Tableau class describing the formula's order.
"""

from typing import Any, Callable, Dict, Optional, Set

from attrs import field, frozen, validators
from numba_cuda_mlir.types import int32
from cubie._cudasim_extensions import cuda
from cubie.backend.intrinsics import unroll_if

from cubie._utils import PrecisionDType, device_function_field
from cubie.CUDAFactory import build_config
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.base_algorithm_step import (
    AlgorithmDefaults,
    StepCache,
)
from cubie.integrators.algorithms.generic_bdf_tableaus import (
    DEFAULT_BDF_TABLEAU,
    BDFTableau,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from cubie.integrators.step_history import StepHistory


BDF_DEFAULTS = AlgorithmDefaults(
    settings={
        "linear_correction_type": "lu",
        "inexact_newton": True,
        "prefactored": True,
        "preconditioner_type": "jacobi",
        "step_controller": "i",
        "integral_gain": 1.0,
        "min_step_shrink": 0.2,
        "max_step_growth": 2.0,
        "safety": 0.9,
    }
)
"""Defaults for BDF tableaus; each tableau caps ``max_step_growth``."""


@frozen
class BDFStepConfig(ImplicitStepConfig):
    """Configuration describing the BDF integrator.

    Attributes
    ----------
    tableau : BDFTableau
        Formula of the step.
    stage_increment_location : str
        Buffer location of the Newton increment.
    stage_base_location : str
        Buffer location of the corrector base state.
    history_fn : Callable or None
        Compiled history device function.
    """

    tableau: BDFTableau = field(default=DEFAULT_BDF_TABLEAU)
    stage_increment_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    stage_base_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    history_fn: Optional[Callable] = device_function_field()


class BDFStep(ODEImplicitStep):
    """Variable-step BDF with an owned history of accepted states."""

    default_tableau = DEFAULT_BDF_TABLEAU

    # The step-end stage solves with a_ij = 1.
    _PREFACTOR_STAGE_DATA = (((1.0,),), (1.0,))
    _BAKED_STAGE_DIAGONAL = 1.0

    @classmethod
    def family_defaults(cls, tableau=None) -> AlgorithmDefaults:
        """Return the BDF defaults."""
        return BDF_DEFAULTS.copy()

    def __init__(
        self,
        precision: PrecisionDType,
        n_states: int,
        dxdt_fn: Optional[Callable] = None,
        observables_fn: Optional[Callable] = None,
        drivers_fn: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: BDFTableau = DEFAULT_BDF_TABLEAU,
        n_drivers: int = 0,
        **kwargs,
    ) -> None:
        """Initialise the BDF step configuration.

        Parameters
        ----------
        precision
            Floating-point precision for CUDA computations.
        n_states
            Number of state variables in the ODE system.
        dxdt_fn
            Device function for evaluating f(t, y) right-hand side.
        observables_fn
            Device function computing system observables.
        drivers_fn
            Optional device function evaluating drivers at arbitrary times.
        get_solver_helper_fn
            Factory function returning solver helpers.
        tableau
            BDF tableau giving the maximum order. Defaults to
            :data:`DEFAULT_BDF_TABLEAU`.
        n_drivers
            Number of driver variables in the system.
        **kwargs
            Optional parameters passed to config classes. See
            BDFStepConfig, ImplicitStepConfig, StepHistoryConfig and
            solver config classes. None values are ignored.
        """
        config = build_config(
            BDFStepConfig,
            required={
                "precision": precision,
                "n_states": n_states,
                "n_drivers": n_drivers,
                "dxdt_fn": dxdt_fn,
                "observables_fn": observables_fn,
                "drivers_fn": drivers_fn,
                "get_solver_helper_fn": get_solver_helper_fn,
                "tableau": tableau,
                "operator_beta": 1.0,
                "operator_gamma": 1.0,
            },
            **kwargs,
        )

        super().__init__(config, self.family_defaults(tableau), **kwargs)

        settings = self.compile_settings
        self.history = StepHistory(
            precision=settings.precision,
            n_states=n_states,
            tableau=settings.tableau,
            **kwargs,
        )
        self.register_buffers()
        self.build_implicit_helpers()

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        config = self.compile_settings
        n = config.n_states

        buffer_registry.clear_own(self)
        buffer_registry.register_child(self, self.solver, name="solver")
        buffer_registry.register_child(self, self.history, name="history")
        buffer_registry.register(
            "stage_increment",
            self,
            n,
            config.stage_increment_location,
        )
        buffer_registry.register(
            "stage_base",
            self,
            n,
            config.stage_base_location,
        )
        # Frozen-Jacobian cache; resized by build_implicit_helpers.
        buffer_registry.register(
            "cached_auxiliaries",
            self,
            0,
            config.cached_auxiliaries_location,
        )

    def _apply_updates(self, updates: Dict[str, Any]) -> Set[str]:
        """Update the history, then the step and its solvers.

        Parameters
        ----------
        updates
            Setting names to new values.

        Returns
        -------
        set[str]
            Names the history, step and solvers recognised.
        """
        history_recognised = self.history.update(updates, silent=True)
        recognised = super()._apply_updates(updates)
        if history_recognised and not recognised:
            self.register_buffers()
            self.build_implicit_helpers()
        return recognised | history_recognised

    def build_implicit_helpers(self) -> None:
        """Request the helpers and wire in the history function."""

        super().build_implicit_helpers()
        self.update_compile_settings(
            {"history_fn": self.history.device_function}
        )

    def build_step(
        self,
        dxdt_fn: Callable,
        observables_fn: Callable,
        drivers_fn: Optional[Callable],
        solver_function: Callable,
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the BDF device step."""

        config = self.compile_settings
        a_ij = numba_precision(1.0)
        n = int32(n)
        unroll_step_element = config.unroll.unroll_step_element
        has_evaluate_driver_at_t = drivers_fn is not None
        has_error = self.uses_error
        history_fn = config.history_fn
        use_cached_solve = self.uses_cached_solve
        prepare_jacobian = config.prepare_jacobian_fn
        solver_fn = solver_function

        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(
                self, self.solver, name="solver"
            )
        )
        alloc_history_shared, alloc_history_persistent = (
            buffer_registry.get_child_allocators(
                self, self.history, name="history"
            )
        )
        getalloc = buffer_registry.get_allocator
        alloc_stage_increment = getalloc("stage_increment", self)
        alloc_stage_base = getalloc("stage_base", self)
        alloc_cached_aux = getalloc("cached_auxiliaries", self)

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
            error,
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
            counters,
        ):
            """Perform one BDF update.

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
                Device array of drivers at the current time.
            proposed_drivers
                Device array receiving drivers at the step end.
            observables
                Device array of observables at the current state.
            proposed_observables
                Device array receiving observables at the step end.
            error
                Device array receiving the local error estimate;
                zero-length without an adaptive controller.
            dt_scalar
                Scalar containing the proposed step size.
            time_scalar
                Scalar containing the current simulation time.
            first_step_flag
                Non-zero on the first integration step.
            accepted_flag
                Non-zero when the previous step was accepted.
            shared
                Device array providing shared scratch buffers.
            persistent_local
                Device array for persistent local storage.
            counters
                Integer array for Newton and Krylov iteration counters.

            Returns
            -------
            int
                Status code returned by the nonlinear solver.
            """
            solver_scratch = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(
                shared, persistent_local
            )
            history_shared = alloc_history_shared(shared, persistent_local)
            history_persistent = alloc_history_persistent(
                shared, persistent_local
            )
            stage_increment = alloc_stage_increment(shared, persistent_local)
            stage_base = alloc_stage_base(shared, persistent_local)
            cached_aux = alloc_cached_aux(shared, persistent_local)

            corrector_step, error_scale, restart = history_fn(
                state,
                dt_scalar,
                first_step_flag,
                accepted_flag,
                stage_base,
                stage_increment,
                history_shared,
                history_persistent,
            )

            next_time = time_scalar + dt_scalar
            if has_evaluate_driver_at_t:
                drivers_fn(next_time, driver_coefficients, proposed_drivers)

            # A restart predicts with explicit Euler.
            if cuda.any_sync(cuda.activemask(), restart):
                dxdt_fn(
                    state,
                    parameters,
                    drivers_buffer,
                    observables,
                    proposed_state,
                    time_scalar,
                )
                for i in unroll_if(range(n), unroll_step_element):
                    stage_increment[i] = cuda.selp(
                        restart,
                        stage_increment[i] + dt_scalar * proposed_state[i],
                        stage_increment[i],
                    )
            # The error buffer holds the starting increment.
            for i in unroll_if(range(n), unroll_step_element):
                stage_increment[i] -= stage_base[i]
                if has_error:
                    error[i] = stage_increment[i]

            status = int32(0)
            if use_cached_solve:
                # Freeze the Jacobian at the step-start state.
                status = prepare_jacobian(
                    state,
                    parameters,
                    proposed_drivers,
                    next_time,
                    corrector_step,
                    cached_aux,
                )
            status |= solver_fn(
                stage_increment,
                parameters,
                proposed_drivers,
                cached_aux,
                next_time,
                corrector_step,
                a_ij,
                stage_base,
                state,
                solver_scratch,
                solver_persistent,
                counters,
            )

            for i in unroll_if(range(n), unroll_step_element):
                proposed_state[i] = stage_base[i] + stage_increment[i]
                if has_error:
                    error[i] = error_scale * (stage_increment[i] - error[i])

            observables_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            return status

        # no cover: end
        return StepCache(step_fn=step, nonlinear_solver_fn=solver_fn)

    @property
    def is_multistage(self) -> bool:
        """Return ``False``: one implicit solve per step."""
        return False

    # Class attribute so alias queries need no instance.
    has_error_estimate = True

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""
        return 1

    @property
    def order(self) -> int:
        """Return the maximum order of the formula."""
        return self.tableau.order
