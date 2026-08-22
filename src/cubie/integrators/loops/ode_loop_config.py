"""Configuration container for CUDA-based integration loops.

Published Classes
-----------------
:class:`ODELoopConfig`
    Attrs container holding system sizes, buffer locations, output compile
    flags, timing intervals, and device function references required to
    compile an :class:`~cubie.integrators.loops.ode_loop.IVPLoop`.

    >>> from numpy import float32
    >>> from cubie.outputhandling.output_config import OutputCompileFlags
    >>> config = ODELoopConfig(
    ...     precision=float32, n_states=4,
    ...     compile_flags=OutputCompileFlags(),
    ... )
    >>> config.n_states
    4

See Also
--------
:class:`~cubie.integrators.loops.ode_loop.IVPLoop`
    Loop factory that consumes this configuration.
:class:`~cubie.CUDAFactory.CUDAFactoryConfig`
    Parent configuration class providing precision and hashing.
:class:`~cubie.outputhandling.output_config.OutputCompileFlags`
    Compile flags governing save and summary cadence.
"""

from typing import Callable, Optional

from attrs import field, validators, frozen
from cubie.CUDAFactory import CUDAFactoryConfig
from warnings import warn

from cubie._utils import (
    getype_validator,
    is_device_validator,
    opt_gttype_validator,
)
from cubie.outputhandling.output_config import OutputCompileFlags


@frozen
class ODELoopConfig(CUDAFactoryConfig):
    """Compile-critical settings for an integrator loop.

    Parameters
    ----------
    n_states
        Number of state variables.
    n_parameters
        Number of parameters.
    n_drivers
        Number of driver variables.
    n_observables
        Number of observable variables.
    n_error
        Number of error elements (typically equals ``n_states`` for
        adaptive methods).
    n_counters
        Number of counter elements.
    state_summaries_buffer_height
        Height of state summary buffer.
    observable_summaries_buffer_height
        Height of observable summary buffer.
    compile_flags
        Output configuration governing save and summary cadence.
    save_every
        Interval between accepted saves, or ``None`` when auto-derived.
    summarise_every
        Interval between summary accumulations, or ``None`` when
        auto-derived.
    sample_summaries_every
        Interval between summary metric updates, or ``None`` when
        auto-derived.
    save_last
        When ``True``, the loop saves the final state regardless of
        ``save_every`` alignment.
    save_regularly
        When ``True``, state saves occur at ``save_every`` intervals.
    summarise_regularly
        When ``True``, summary accumulations occur at
        ``summarise_every`` intervals.
    save_state_fn
        Device function that records state and observable snapshots.
    update_summaries_fn
        Device function that accumulates summary statistics.
    save_summaries_fn
        Device function that writes summary statistics to output buffers.
    step_controller_fn
        Device function that updates the timestep and acceptance flag.
    step_function
        Device function that advances the solution by one tentative step.
    evaluate_driver_at_t
        Device function that evaluates driver signals for a given time.
    evaluate_observables
        Device function that evaluates observables for the current state.
    initialise_state_fn
        Device function correcting the state at ``t0`` to a
        consistent DAE starting point; the owning
        :class:`~cubie.integrators.dae_initialiser.DAEInitialiser`
        compiles a no-op for non-DAE configurations.
    dt
        Initial timestep prior to controller feedback.
    is_adaptive
        Whether the loop operates with an adaptive controller.
    state_location, proposed_state_location, parameters_location, \
    drivers_location, proposed_drivers_location, observables_location, \
    proposed_observables_location, error_location, counters_location, \
    state_summary_location, observable_summary_location, dt_location, \
    accept_step_location, proposed_counters_location
        Memory location for the corresponding buffer (``'local'`` or
        ``'shared'``).
    """

    # System size parameters
    n_states: int = field(default=0, validator=getype_validator(int, 0))
    n_parameters: int = field(default=0, validator=getype_validator(int, 0))
    n_drivers: int = field(default=0, validator=getype_validator(int, 0))
    n_observables: int = field(default=0, validator=getype_validator(int, 0))
    n_error: int = field(default=0, validator=getype_validator(int, 0))
    n_counters: int = field(default=0, validator=getype_validator(int, 0))

    # Array sizes
    state_summaries_buffer_height: int = field(
        default=0, validator=getype_validator(int, 0)
    )
    observable_summaries_buffer_height: int = field(
        default=0, validator=getype_validator(int, 0)
    )

    # Buffer location settings
    state_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    proposed_state_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    parameters_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    drivers_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    proposed_drivers_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    observables_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    proposed_observables_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    error_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    counters_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    state_summary_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    observable_summary_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    dt_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    accept_step_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )
    proposed_counters_location: str = field(
        default="local", validator=validators.in_(["shared", "local"])
    )

    compile_flags: OutputCompileFlags = field(
        factory=OutputCompileFlags,
        validator=validators.instance_of(OutputCompileFlags),
    )

    # Loop timing parameters
    _save_every: Optional[float] = field(
        default=None, validator=opt_gttype_validator(float, 0)
    )
    _summarise_every: Optional[float] = field(
        default=None, validator=opt_gttype_validator(float, 0)
    )
    _sample_summaries_every: Optional[float] = field(
        default=None, validator=opt_gttype_validator(float, 0)
    )

    # Flags for end-of-run behavior
    save_last: bool = field(
        default=False, validator=validators.instance_of(bool)
    )
    save_regularly: bool = field(
        default=False, validator=validators.instance_of(bool)
    )
    summarise_regularly: bool = field(
        default=False, validator=validators.instance_of(bool)
    )

    save_state_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    update_summaries_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    save_summaries_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    step_controller_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    step_function: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    evaluate_driver_at_t: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    evaluate_observables: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    initialise_state_fn: Optional[Callable] = field(
        default=None,
        validator=validators.optional(is_device_validator),
        eq=False,
    )
    _dt: Optional[float] = field(
        default=0.01,
        validator=opt_gttype_validator(float, 0),
    )
    is_adaptive: Optional[bool] = field(
        default=False,
        validator=validators.optional(validators.instance_of(bool)),
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    @property
    def samples_per_summary(self):
        """Return the number of update samples per summary interval.

        When both ``summarise_every`` and ``sample_summaries_every`` are
        set, the ratio must be close to an integer.  A small deviation
        (≤ 0.01) is accepted with a warning; larger deviations raise
        ``ValueError``.

        Returns
        -------
        int
            Number of samples per summary, or ``0`` when either timing
            parameter is ``None``.

        Raises
        ------
        ValueError
            If ``summarise_every`` is not an integer multiple of
            ``sample_summaries_every``.
        """
        summarise_every = self.summarise_every
        sample_summaries_every = self.sample_summaries_every

        if summarise_every is None or sample_summaries_every is None:
            return 0

        raw_ratio = summarise_every / sample_summaries_every
        samples_per_summary = int(round(raw_ratio))

        # How close is this to an integer multiple? Warn if it needs slight
        # adjustment, raise if the arguments aren't even multiples.
        deviation = abs(raw_ratio - samples_per_summary)
        if deviation <= 0.01:
            adjusted = samples_per_summary * self.sample_summaries_every
            if adjusted != self._summarise_every:
                warn(
                    f"summarise_every adjusted from "
                    f"{self._summarise_every} to {adjusted}, the nearest "
                    f" integer multiple of sample_summaries_every "
                    f"({self.sample_summaries_every})"
                )
            return samples_per_summary
        else:
            raise ValueError(
                f"summarise_every ({self._summarise_every}) must be an "
                f"integer multiple of sample_summaries_every "
                f"({self.sample_summaries_every}). Under these "
                f"settings, summaries are calculated every "
                f"{raw_ratio:.2f} updates, and the calculation can't run "
                f"between samples."
            )

    @property
    def save_every(self) -> Optional[float]:
        """Return the output save interval, or None if not configured."""
        if self._save_every is None:
            return None
        return self.precision(self._save_every)

    @property
    def summarise_every(self) -> Optional[float]:
        """Return the summary interval, or None if not configured."""
        if self._summarise_every is None:
            return None
        return self.precision(self._summarise_every)

    @property
    def sample_summaries_every(self) -> Optional[float]:
        """Return the summary sampling interval, or None if not configured."""
        if self._sample_summaries_every is None:
            return None
        return self.precision(self._sample_summaries_every)

    @property
    def dt(self) -> float:
        """Return the initial timestep."""
        return self.precision(self._dt)
