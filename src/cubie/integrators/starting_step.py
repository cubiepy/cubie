"""Starting-step estimate for adaptive integrations.

Build a one-shot device function that returns a run's first step when
the user gives no initial step. The function scales the state, its
derivative and the change in the derivative over one explicit Euler
step with the step controller's error norm, and returns the step whose
leading error term just meets the tolerances. Because the error norm
skips the algebraic rows of a mass-matrix system, the function
estimates the step from the differential rows only.

Published Classes
-----------------
:class:`StartingStepConfig`
    Attrs configuration for the estimator factory.

:class:`StartingStepCache`
    Cache container holding the compiled device function.

:class:`StartingStepEstimator`
    CUDAFactory compiling the starting-step device function.
"""

from math import sqrt as math_sqrt
from typing import Any, Callable, Dict, Optional, Set, Tuple

from attrs import define, field, frozen, validators
from numpy import array as np_array
from numpy import bool_ as np_bool
from numpy import finfo as np_finfo

from cubie.CUDAFactory import (
    CUDAFactory,
    CUDAFactoryConfig,
    CUDADispatcherCache,
    build_config,
)
from cubie._utils import (
    PrecisionDType,
    device_function_field,
    getype_validator,
    opt_getype_validator,
)
from cubie.buffer_registry import buffer_registry
from cubie.backend.intrinsics import unroll_if
from cubie._cudasim_extensions import cuda, fmax, fmin

FLAT_NORM_THRESHOLD = 1e-5
"""Scaled state or derivative norm below which the first guess is fixed."""

FLAT_FIRST_GUESS = 1e-6
"""First guess used when the state or its derivative is near zero."""

FIRST_GUESS_FRACTION = 0.01
"""Fraction of the state-to-derivative norm ratio taken as first guess."""

NEGLIGIBLE_CURVATURE = 1e-15
"""Derivative scale below which the order-based step is not trusted."""

FIRST_GUESS_GROWTH = 100.0
"""Largest multiple of the first guess to return."""

NEGLIGIBLE_CURVATURE_SHRINK = 1e-3
"""First-guess multiple used when the derivative scale is negligible."""


@frozen
class StartingStepConfig(CUDAFactoryConfig):
    """Compile settings for the starting-step estimate.

    Parameters
    ----------
    n_states
        Number of state variables.
    n_drivers
        Number of driver variables.
    mass_flags
        Per-state mass-diagonal flags, ``True`` for a differential row.
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
    dxdt_fn
        System right-hand-side device function.
    observables_fn
        System observables device function.
    drivers_fn
        Driver evaluation device function, or ``None`` without drivers.
    norm_fn
        Step controller's error-norm device function.
    """

    n_states: int = field(default=1, validator=getype_validator(int, 1))
    n_drivers: int = field(default=0, validator=getype_validator(int, 0))
    mass_flags: Tuple[bool, ...] = field(
        default=(),
        validator=validators.deep_iterable(
            validators.instance_of(bool),
            validators.instance_of(tuple),
        ),
    )
    _dt: Optional[float] = field(
        default=None, validator=opt_getype_validator(float, 0)
    )
    is_adaptive: bool = field(
        default=False, validator=validators.instance_of(bool)
    )
    algorithm_order: int = field(
        default=1, validator=getype_validator(int, 1)
    )
    _dt_min: float = field(default=1e-6, validator=getype_validator(float, 0))
    _dt_max: float = field(default=1.0, validator=getype_validator(float, 0))
    dxdt_fn: Optional[Callable] = device_function_field()
    observables_fn: Optional[Callable] = device_function_field()
    drivers_fn: Optional[Callable] = device_function_field()
    norm_fn: Optional[Callable] = device_function_field()

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.mass_flags and len(self.mass_flags) != self.n_states:
            raise ValueError(
                "mass_flags must carry one flag per state: got "
                f"{len(self.mass_flags)} flags for n_states={self.n_states}."
            )

    @property
    def estimates_dt(self) -> bool:
        """Return whether to estimate the starting step.

        Return ``True`` for an adaptive controller given no initial
        step.
        """
        return self.is_adaptive and self._dt is None

    @property
    def dt_min(self) -> float:
        """Return the lower bound on the estimate."""
        return self.precision(self._dt_min)

    @property
    def dt_max(self) -> float:
        """Return the upper bound on the estimate."""
        return self.precision(self._dt_max)

    @property
    def differential_rows(self) -> Tuple[bool, ...]:
        """Return the per-state flags, all ``True`` when none are given."""
        if not self.mass_flags:
            return (True,) * self.n_states
        return self.mass_flags


@define
class StartingStepCache(CUDADispatcherCache):
    """Cache container for the compiled starting-step function.

    Attributes
    ----------
    starting_step_fn
        Compiled device function that returns the starting step, or
        ``None`` when no estimate is compiled.
    estimates_dt
        Whether the build estimates each run's starting step.
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
    **kwargs
        Config fields of :class:`StartingStepConfig`; unrecognised keys
        are ignored.
    """

    settings_keys = frozenset({"dt"})

    @classmethod
    def system_inputs(cls, system: Any) -> Dict[str, Any]:
        """Return the estimator settings a system object provides."""
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
        **kwargs,
    ) -> None:
        super().__init__()
        config = build_config(
            StartingStepConfig,
            required={"precision": precision, "n_states": int(n_states)},
            **kwargs,
        )
        self.setup_compile_settings(config)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register the derivative buffer, sized zero without an estimate."""
        config = self.compile_settings
        size = config.n_states if config.estimates_dt else 0
        buffer_registry.clear_own(self)
        buffer_registry.register("starting_derivative", self, size, "local")

    def build(self) -> StartingStepCache:
        """Compile the starting-step device function.

        Returns
        -------
        StartingStepCache
            Cache holding the compiled device function, or ``None`` in
            its place when no estimate is compiled.
        """
        config = self.compile_settings
        if not config.estimates_dt:
            return StartingStepCache(
                starting_step_fn=None, estimates_dt=False
            )

        precision = config.numba_precision
        dxdt_fn = config.dxdt_fn
        observables_fn = config.observables_fn
        drivers_fn = config.drivers_fn
        norm_fn = config.norm_fn
        n_states = config.n_states
        has_drivers = drivers_fn is not None and config.n_drivers > 0
        unroll_step_element = config.unroll.unroll_step_element

        differential = np_array(config.differential_rows, dtype=np_bool)
        all_differential = bool(differential.all())

        dt_min = config.dt_min
        dt_max = config.dt_max
        # The controller's norm is the mean square of the scaled error,
        # so we raise it to half the order exponent.
        half_inverse_order = precision(0.5 / (config.algorithm_order + 1))
        flat_threshold = precision(FLAT_NORM_THRESHOLD)
        flat_guess = precision(FLAT_FIRST_GUESS)
        guess_fraction = precision(FIRST_GUESS_FRACTION)
        curvature_floor2 = precision(NEGLIGIBLE_CURVATURE**2)
        curvature_target = precision(FIRST_GUESS_FRACTION**2)
        growth = precision(FIRST_GUESS_GROWTH)
        curvature_shrink = precision(NEGLIGIBLE_CURVATURE_SHRINK)
        typed_zero = precision(0.0)
        typed_huge = precision(float(np_finfo(config.precision).max))

        alloc_derivative = buffer_registry.get_allocator(
            "starting_derivative", self
        )

        # no cover: start
        @cuda.jit(device=True, inline=True, **self.jit_kwargs)
        def starting_step_fn(
            state,
            parameters,
            driver_coefficients,
            drivers,
            observables,
            probe_state,
            probe_drivers,
            probe_observables,
            probe_derivative,
            t,
            shared_scratch,
            persistent_scratch,
        ):
            """Return the starting step estimated at ``state``.

            Overwrites the probe buffers; leaves ``state``, ``drivers``
            and ``observables`` unchanged.
            """
            derivative = alloc_derivative(shared_scratch, persistent_scratch)
            dxdt_fn(state, parameters, drivers, observables, derivative, t)

            state_norm2 = norm_fn(state, state, state)
            derivative_norm2 = norm_fn(derivative, state, state)
            state_scale = math_sqrt(state_norm2)
            derivative_scale = math_sqrt(derivative_norm2)

            flat = (state_scale < flat_threshold) | (
                derivative_scale < flat_threshold
            )
            first_guess = cuda.selp(
                flat,
                flat_guess,
                precision(guess_fraction * state_scale / derivative_scale),
            )
            first_guess = fmin(first_guess, dt_max)

            # Take one explicit Euler step. Hold the algebraic rows at
            # their values, because their right-hand side is a
            # constraint residual.
            for i in unroll_if(range(n_states), unroll_step_element):
                euler_value = state[i] + first_guess * derivative[i]
                if all_differential:
                    probe_state[i] = euler_value
                else:
                    probe_state[i] = cuda.selp(
                        differential[i], euler_value, state[i]
                    )

            probe_time = precision(t + first_guess)
            if has_drivers:
                drivers_fn(probe_time, driver_coefficients, probe_drivers)
            observables_fn(
                probe_state,
                parameters,
                probe_drivers,
                probe_observables,
                probe_time,
            )
            dxdt_fn(
                probe_state,
                parameters,
                probe_drivers,
                probe_observables,
                probe_derivative,
                probe_time,
            )
            for i in unroll_if(range(n_states), unroll_step_element):
                probe_derivative[i] = probe_derivative[i] - derivative[i]

            # We divide by the first guess after taking the norm so the
            # squared terms inside the norm stay in range.
            change_norm2 = norm_fn(probe_derivative, state, state)
            curvature_scale = math_sqrt(change_norm2) / first_guess

            largest_norm2 = fmax(
                derivative_norm2, curvature_scale * curvature_scale
            )
            negligible = largest_norm2 <= curvature_floor2
            order_step = cuda.selp(
                negligible,
                fmax(flat_guess, first_guess * curvature_shrink),
                precision(
                    (curvature_target / largest_norm2) ** half_inverse_order
                ),
            )
            estimate = fmin(growth * first_guess, order_step)
            # When the derivative does not change over the Euler step,
            # start at the largest multiple of the first guess.
            estimate = cuda.selp(
                change_norm2 == typed_zero, growth * first_guess, estimate
            )
            estimate = fmax(fmin(estimate, dt_max), dt_min)

            # Return dt_min when a norm is not finite. We test
            # "<= huge", which is false for both NaN and infinity.
            finite = (derivative_norm2 <= typed_huge) & (
                change_norm2 <= typed_huge
            ) & (curvature_scale <= typed_huge)
            return cuda.selp(finite, estimate, dt_min)

        # no cover: end
        return StartingStepCache(
            starting_step_fn=starting_step_fn, estimates_dt=True
        )

    def _update(self, updates: Dict[str, Any], silent: bool) -> Set[str]:
        """Apply the estimator settings and resize its buffer.

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
        self.register_buffers()
        return recognised

    @property
    def device_function(self) -> Optional[Callable]:
        """Return the starting-step device function, or ``None``."""
        return self.get_cached_output("starting_step_fn")

    @property
    def estimates_dt(self) -> bool:
        """Return whether the build estimates each run's starting step."""
        return self.get_cached_output("estimates_dt")
