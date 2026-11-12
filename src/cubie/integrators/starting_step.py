"""Starting-step estimate for adaptive integrations.

Build a device function that returns a run's first step by Hairer's
method. The function takes the step controller's error norm of the
state, of its derivative and of the change in the derivative over one
explicit Euler step, and returns a step from those three norms and the
algorithm's order. On a mass-matrix system the function reads the
differential rows only.

Published Classes
-----------------
:class:`StartingStepConfig`
    Attrs configuration for the estimator factory.

:class:`StartingStepCache`
    Cache container holding the compiled device function.

:class:`StartingStepEstimator`
    CUDAFactory compiling the starting-step device function.

Constants
---------
:data:`ALL_STARTING_STEP_PARAMETERS`
    Names of the Solver settings :class:`StartingStepEstimator` takes.
"""

from math import sqrt as math_sqrt
from typing import Any, Callable, Dict, Optional, Set, Tuple

from attrs import define, field, frozen, validators
from numpy import array as np_array
from numpy import bool_ as np_bool

from cubie._cudasim_extensions import cuda, fmax, fmin
from cubie._utils import (
    PrecisionDType,
    clamp_factory,
    device_function_field,
    getype_validator,
    opt_getype_validator,
)
from cubie.backend.intrinsics import unroll_if
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import (
    ALL_JIT_PARAMETERS,
    ALL_UNROLL_PARAMETERS,
    CUDADispatcherCache,
    CUDAFactory,
    CUDAFactoryConfig,
    build_config,
)

ALL_STARTING_STEP_PARAMETERS = (
    {"dt", "initial_derivative_location"}
    | ALL_UNROLL_PARAMETERS
    | ALL_JIT_PARAMETERS
)
""":class:`StartingStepEstimator` takes these settings from the Solver."""


@frozen
class StartingStepConfig(CUDAFactoryConfig):
    """Compile settings for the starting-step estimate.

    Parameters
    ----------
    n_states
        Number of state variables.
    mass_flags
        Per-state mass-diagonal flags, ``True`` for a differential row.
    n_drivers
        Number of driver variables.
    dt
        Initial step the user gave, or ``None`` when not given.
    is_adaptive
        Whether the step controller adapts the step.
    algorithm_order
        Order the step controller assumes for the algorithm.
    dt_min
        Lower bound on the estimate.
    dt_max
        Upper bound on the estimate.
    initial_derivative_location
        Memory location (``'local'`` or ``'shared'``) of the buffer for
        the derivative at the initial state.
    dxdt_fn
        System right-hand-side device function.
    observables_fn
        System observables device function.
    drivers_fn
        Driver evaluation device function, or ``None`` without drivers.
    norm_fn
        Step controller's error-norm device function.
    """

    n_states: int = field(validator=getype_validator(int, 1))
    mass_flags: Tuple[bool, ...] = field(
        converter=tuple,
        validator=validators.deep_iterable(
            validators.instance_of(bool),
            validators.instance_of(tuple),
        ),
    )
    n_drivers: int = field(default=0, validator=getype_validator(int, 0))
    _dt: Optional[float] = field(
        default=None, validator=opt_getype_validator(float, 0)
    )
    is_adaptive: bool = field(
        default=False, validator=validators.instance_of(bool)
    )
    algorithm_order: int = field(default=1, validator=getype_validator(int, 1))
    _dt_min: float = field(default=1e-9, validator=getype_validator(float, 0))
    _dt_max: float = field(default=1.0, validator=getype_validator(float, 0))
    initial_derivative_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    dxdt_fn: Optional[Callable] = device_function_field()
    observables_fn: Optional[Callable] = device_function_field()
    drivers_fn: Optional[Callable] = device_function_field()
    norm_fn: Optional[Callable] = device_function_field()

    @property
    def estimates_dt(self) -> bool:
        """Return ``True`` for an adaptive controller given no ``dt``."""
        return self.is_adaptive and self._dt is None

    @property
    def initial_derivative_size(self) -> int:
        """Return the derivative buffer's length, or zero when the
        estimator compiles no estimate."""
        return self.n_states if self.estimates_dt else 0

    @property
    def dt_min(self) -> float:
        """Return the lower bound on the estimate."""
        return self.precision(self._dt_min)

    @property
    def dt_max(self) -> float:
        """Return the upper bound on the estimate."""
        return self.precision(self._dt_max)


@define
class StartingStepCache(CUDADispatcherCache):
    """Cache container for the compiled starting-step function.

    Attributes
    ----------
    starting_step_fn
        Compiled device function that returns the starting step, or
        ``None`` when the estimator compiled no estimate.
    estimates_dt
        Whether the estimator compiled an estimate.
    """

    starting_step_fn: Optional[Callable] = field(default=None, eq=False)
    estimates_dt: bool = field(default=False)


class StartingStepEstimator(CUDAFactory):
    """Factory for the starting-step device function.

    Parameters
    ----------
    precision
        Numerical precision for computations.
    n_states
        Number of state variables.
    mass_flags
        Per-state mass-diagonal flags, ``True`` for a differential row.
    **kwargs
        Config fields of :class:`StartingStepConfig`; unrecognised keys
        are ignored.
    """

    settings_keys = frozenset(ALL_STARTING_STEP_PARAMETERS)

    @classmethod
    def system_inputs(cls, system: Any) -> Dict[str, Any]:
        """Return the estimator settings read from a system object."""
        return dict(
            precision=system.precision,
            n_states=system.sizes.states,
            n_drivers=system.sizes.drivers,
            mass_flags=system.mass_diagonal_flags,
            dxdt_fn=system.dxdt_fn,
            observables_fn=system.observables_fn,
        )

    def __init__(
        self,
        precision: PrecisionDType,
        n_states: int,
        mass_flags: Tuple[bool, ...],
        **kwargs,
    ) -> None:
        super().__init__()
        config = build_config(
            StartingStepConfig,
            required={
                "precision": precision,
                "n_states": int(n_states),
                "mass_flags": mass_flags,
            },
            **kwargs,
        )
        self.setup_compile_settings(config)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register the buffer for the derivative at the initial state."""
        config = self.compile_settings
        buffer_registry.register(
            "initial_derivative",
            self,
            config.initial_derivative_size,
            config.initial_derivative_location,
        )

    def build(self) -> StartingStepCache:
        """Compile the starting-step device function.

        Returns
        -------
        StartingStepCache
            The compiled device function, or ``None`` in its place for
            a fixed controller or a given ``dt``.
        """
        config = self.compile_settings
        if not config.estimates_dt:
            return StartingStepCache(starting_step_fn=None, estimates_dt=False)

        precision = config.numba_precision
        dxdt_fn = config.dxdt_fn
        observables_fn = config.observables_fn
        drivers_fn = config.drivers_fn
        norm_fn = config.norm_fn
        clamp = clamp_factory(config.precision)
        n_states = config.n_states
        has_drivers = drivers_fn is not None and config.n_drivers > 0
        unroll_step_element = config.unroll.unroll_step_element

        differential = np_array(config.mass_flags, dtype=np_bool)
        all_differential = bool(differential.all())

        dt_min = config.dt_min
        dt_max = config.dt_max
        # We square each of Hairer's norm thresholds and halve his order
        # exponent, because norm_fn returns a mean square.
        guess_fraction = precision(0.01)
        zero_norm2 = precision(1e-5**2)
        zero_norm_guess = precision(1e-6)
        target_norm2 = precision(0.01**2)
        guess_growth = precision(100.0)
        half_inverse_order = precision(0.5 / (config.algorithm_order + 1))

        alloc_derivative = buffer_registry.get_allocator(
            "initial_derivative", self
        )

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def starting_step_fn(
            state,
            proposed_state,
            parameters,
            driver_coefficients,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            time_scalar,
            shared,
            persistent_local,
        ):
            """Return the starting step estimated at ``state``.

            Overwrite ``proposed_state``, ``proposed_drivers``,
            ``proposed_observables`` and ``error``.
            """
            derivative = alloc_derivative(shared, persistent_local)
            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                derivative,
                time_scalar,
            )
            state_norm2 = norm_fn(state, state, state)
            derivative_norm2 = norm_fn(derivative, state, state)

            # Guess the step that changes the state by 1% of its norm.
            # We guess 1e-6 for a zero state or a zero derivative.
            no_scale = (state_norm2 < zero_norm2) | (
                derivative_norm2 < zero_norm2
            )
            first_guess = cuda.selp(
                no_scale,
                zero_norm_guess,
                precision(
                    guess_fraction * math_sqrt(state_norm2 / derivative_norm2)
                ),
            )
            first_guess = clamp(first_guess, dt_min, dt_max)

            # Take one explicit Euler step of the first guess. Hold the
            # algebraic rows, because dxdt_fn returns their residuals.
            for i in unroll_if(range(n_states), unroll_step_element):
                euler_value = state[i] + first_guess * derivative[i]
                if all_differential:
                    proposed_state[i] = euler_value
                else:
                    proposed_state[i] = cuda.selp(
                        differential[i], euler_value, state[i]
                    )

            end_time = precision(time_scalar + first_guess)
            if has_drivers:
                drivers_fn(end_time, driver_coefficients, proposed_drivers)
            observables_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )
            dxdt_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                error,
                end_time,
            )
            for i in unroll_if(range(n_states), unroll_step_element):
                error[i] = error[i] - derivative[i]

            # Estimate the second derivative from the derivative's change
            # per unit step. We divide by the first guess twice so that
            # we never square a very small guess.
            second_derivative_norm2 = (
                norm_fn(error, state, state) / first_guess / first_guess
            )

            # Solve h^(order + 1) * norm = 0.01 for the second guess h,
            # with the larger of the two derivative norms. For a system
            # at rest we divide by zero here and rely on fmin to return
            # its first operand.
            largest_norm2 = fmax(derivative_norm2, second_derivative_norm2)
            second_guess = precision(
                (target_norm2 / largest_norm2) ** half_inverse_order
            )
            estimate = fmin(guess_growth * first_guess, second_guess)
            return clamp(estimate, dt_min, dt_max)

        # no cover: end
        return StartingStepCache(
            starting_step_fn=starting_step_fn, estimates_dt=True
        )

    def _update(self, updates: Dict[str, Any], silent: bool) -> Set[str]:
        """Apply the estimator settings and the buffer location.

        Parameters
        ----------
        updates
            Setting names to new values.
        silent
            Whether :meth:`update` ignores unrecognised names.

        Returns
        -------
        set[str]
            Recognised setting names.
        """
        recognised = self.update_compile_settings(updates, silent=True)
        recognised |= buffer_registry.update(self, updates, silent=True)
        self.register_buffers()
        return recognised

    @property
    def device_function(self) -> Optional[Callable]:
        """Return the starting-step device function, or ``None``."""
        return self.get_cached_output("starting_step_fn")

    @property
    def estimates_dt(self) -> bool:
        """Return whether the estimator compiled an estimate."""
        return self.get_cached_output("estimates_dt")
