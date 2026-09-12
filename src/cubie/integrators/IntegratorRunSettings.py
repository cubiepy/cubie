"""Settings of a single integrator run.

Published Classes
-----------------
:class:`IntegratorRunSettings`
    The run's given settings and the inputs they resolve against.

    >>> from numpy import float32
    >>> settings = IntegratorRunSettings(
    ...     precision=float32, algorithm="erk", step_controller="pid"
    ... )
    >>> settings.algorithm
    'erk'

Constants
---------
:data:`ALL_RUN_PARAMETERS`
    Keys the run owns.

See Also
--------
:class:`~cubie.CUDAFactory.CUDAFactoryConfig`
    Parent class providing precision and numba type conversion.
:class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`
    Consumer that writes the resolved values into its children.
"""

from typing import Any, Callable, Dict, Optional

import attrs
from attrs import cmp_using, converters, field, validators
from numpy import array_equal, asarray, finfo as np_finfo, ndarray

from cubie._utils import (
    device_function_field,
    opt_getype_validator,
)
from cubie.CUDAFactory import CUDAFactoryConfig
from cubie.integrators.algorithms.base_algorithm_step import (
    AlgorithmDefaults,
    LINEAR_SOLVER_VARIANT_PARAMETERS,
    RUN_RESOLVED_STEP_PARAMETERS,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    DAE_SOLVER_DEFAULTS,
)
from cubie.integrators.step_control.adaptive_step_controller import (
    gain_converter,
)
from cubie.integrators.step_control.base_step_controller import (
    CONTROLLER_GAIN_NAMES,
    promoted_gain_controller,
)


RUN_TIMING_PARAMETERS = frozenset(
    {"save_every", "summarise_every", "sample_summaries_every"}
)
"""Loop schedule keys the run owns."""

RUN_INNER_TOLERANCE_PARAMETERS = frozenset(
    {
        "krylov_atol",
        "krylov_rtol",
        "krylov_residual_reduction",
        "newton_atol",
        "newton_rtol",
    }
)
"""Inner-solver tolerances resolved from the controller's."""

RUN_STEP_DEFAULT_PARAMETERS = frozenset(
    {
        "preconditioner_type",
        "preconditioner_order",
        "linear_correction_type",
        "inexact_newton",
        "prefactored",
        "attempt_dense_prediction",
    }
)
"""Step keys resolved from the family, tableau and DAE defaults."""

RUN_CONTROLLER_PARAMETERS = frozenset(
    {
        *CONTROLLER_GAIN_NAMES,
        "filter_coefficients",
        "min_step_shrink",
        "max_step_growth",
        "safety",
        "deadband_min",
        "deadband_max",
    }
)
"""Controller keys resolved from the family defaults."""

ALL_RUN_PARAMETERS = frozenset(
    {"algorithm", "step_controller", "dae_initialisation"}
    | RUN_TIMING_PARAMETERS
    | RUN_INNER_TOLERANCE_PARAMETERS
    | RUN_STEP_DEFAULT_PARAMETERS
    | RUN_CONTROLLER_PARAMETERS
)
"""Keys the run owns."""

assert RUN_INNER_TOLERANCE_PARAMETERS | RUN_STEP_DEFAULT_PARAMETERS | {
    "dae_initialisation"
} == RUN_RESOLVED_STEP_PARAMETERS

DERIVED_SAMPLES_PER_SUMMARY = 100
"""Summary samples per window derived from the duration."""

GIVEN_SAMPLES_PER_SUMMARY = 10
"""Summary samples per window when only the window is given."""


def _filter_converter(value):
    """Keep a preset name; freeze a coefficient sequence as a tuple."""
    if value is None or isinstance(value, str):
        return value
    return tuple(value)


def _tolerance_converter(value):
    """Keep a scalar as a float and a vector as a float64 array."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, ndarray)):
        return asarray(value, dtype=float)
    return float(value)


def _tolerance_field():
    return field(
        default=None,
        converter=_tolerance_converter,
        eq=cmp_using(eq=array_equal),
    )


def _optional_float():
    return field(default=None, validator=opt_getype_validator(float, 0))


def _optional_bool():
    return field(
        default=None,
        validator=validators.optional(validators.instance_of(bool)),
    )


def _optional_str():
    return field(
        default=None,
        converter=converters.optional(str.lower),
        validator=validators.optional(validators.instance_of(str)),
    )


@attrs.frozen
class IntegratorRunSettings(CUDAFactoryConfig):
    """Given settings of a run and the inputs they resolve against.

    Each owned key is stored as given (``None`` unset) and read through
    a resolving property; the inputs come from the children's products.

    Attributes
    ----------
    precision
        Numerical precision of every child.
    algorithm
        Name of the integration step algorithm.
    step_controller
        Controller name as given.
    dae_initialisation
        Consistent-initialisation mode for singular-mass systems.
    save_every, summarise_every, sample_summaries_every
        The loop schedule as given.
    krylov_atol, krylov_rtol, krylov_residual_reduction, newton_atol,
    newton_rtol
        Inner-solver tolerances as given.
    preconditioner_type, preconditioner_order, linear_correction_type,
    inexact_newton, prefactored, attempt_dense_prediction
        Step keys as given.
    integral_gain, proportional_gain, derivative_gain,
    filter_coefficients, min_step_shrink, max_step_growth, safety,
    deadband_min, deadband_max
        Controller keys as given.
    algorithm_defaults
        The step's merged family and tableau defaults.
    has_error_estimate, is_implicit, is_linear, has_mass
        Flags of the step and the system; a step without an error
        estimate replaces an adaptive request with ``fixed``.
    controller_atol, controller_rtol
        The controller's tolerance vectors.
    has_summary_outputs, has_time_domain_outputs
        Flags of the output functions.
    summary_window
        The batch's duration while the schedule derives from it.
    loop_fn
        The loop's compiled device function.
    """

    algorithm: str = field(
        default="euler",
        converter=str.lower,
        validator=validators.instance_of(str),
    )
    _step_controller: Optional[str] = _optional_str()
    dae_initialisation: Optional[str] = _optional_str()

    _save_every: Optional[float] = _optional_float()
    _summarise_every: Optional[float] = _optional_float()
    _sample_summaries_every: Optional[float] = _optional_float()

    _krylov_atol: Optional[Any] = _tolerance_field()
    _krylov_rtol: Optional[Any] = _tolerance_field()
    _krylov_residual_reduction: Optional[float] = _optional_float()
    _newton_atol: Optional[Any] = _tolerance_field()
    _newton_rtol: Optional[Any] = _tolerance_field()

    _preconditioner_type: Optional[str] = _optional_str()
    _preconditioner_order: Optional[int] = field(
        default=None, validator=opt_getype_validator(int, 0)
    )
    _linear_correction_type: Optional[str] = _optional_str()
    _inexact_newton: Optional[bool] = _optional_bool()
    _prefactored: Optional[bool] = _optional_bool()
    _attempt_dense_prediction: Optional[bool] = _optional_bool()

    _integral_gain: Optional[Any] = field(
        default=None, converter=converters.optional(gain_converter)
    )
    _proportional_gain: Optional[Any] = field(
        default=None, converter=converters.optional(gain_converter)
    )
    _derivative_gain: Optional[Any] = field(
        default=None, converter=converters.optional(gain_converter)
    )
    filter_coefficients: Optional[Any] = field(
        default=None, converter=_filter_converter
    )
    _min_step_shrink: Optional[float] = _optional_float()
    _max_step_growth: Optional[float] = _optional_float()
    _safety: Optional[float] = _optional_float()
    _deadband_min: Optional[float] = _optional_float()
    _deadband_max: Optional[float] = _optional_float()

    algorithm_defaults: AlgorithmDefaults = field(
        factory=AlgorithmDefaults, eq=False
    )
    has_error_estimate: bool = field(default=True)
    is_implicit: bool = field(default=False)
    is_linear: bool = field(default=False)
    has_mass: bool = field(default=False)
    controller_atol: Optional[ndarray] = _tolerance_field()
    controller_rtol: Optional[ndarray] = _tolerance_field()
    has_summary_outputs: bool = field(default=False)
    has_time_domain_outputs: bool = field(default=False)
    summary_window: Optional[float] = _optional_float()
    loop_fn: Optional[Callable] = device_function_field()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.has_mass and self.preconditioner_type == "neumann":
            raise ValueError(
                "Neumann preconditioners assume an identity mass "
                "matrix and cannot precondition a system with torn "
                "algebraic rows. Use preconditioner_type='jacobi'."
            )

    def _family(self, key: str) -> Any:
        """Return the family or tableau default of ``key``."""
        return self.algorithm_defaults.settings.get(key)

    def _given(self, key: str) -> Any:
        """Return the given value of a run-owned key."""
        return getattr(self, f"_{key}")

    # ------------------------------------------------------------------
    # Controller selection
    # ------------------------------------------------------------------
    @property
    def requested_controller(self) -> str:
        """Given, else the family default promoted to carry given gains."""
        if self._step_controller is not None:
            return self._step_controller
        family = self._family("step_controller") or "fixed"
        given = {
            name: self._given(name)
            for name in CONTROLLER_GAIN_NAMES
            if self._given(name) is not None
        }
        if self.filter_coefficients is not None:
            given["filter_coefficients"] = self.filter_coefficients
        return promoted_gain_controller(family, given) or family

    @property
    def step_controller(self) -> str:
        """The request, or ``fixed`` when the step has no error estimate."""
        name = self.requested_controller
        if name != "fixed" and not self.has_error_estimate:
            return "fixed"
        return name

    @property
    def controller_replaced(self) -> bool:
        """Return whether the request was replaced by ``fixed``."""
        return self.step_controller != self.requested_controller

    @property
    def is_adaptive(self) -> bool:
        """Return whether the controller in effect adapts the step."""
        return self.step_controller != "fixed"

    @property
    def controller_settings(self) -> Dict[str, Any]:
        """Resolved controller keys; family gains only for the family's
        controller without a filter."""
        settings = {}
        if self.filter_coefficients is not None:
            settings["filter_coefficients"] = self.filter_coefficients
        else:
            family_gains = (
                self.step_controller == self._family("step_controller")
            )
            for name in CONTROLLER_GAIN_NAMES:
                value = self._given(name)
                if value is None and family_gains:
                    value = self._family(name)
                if value is not None:
                    settings[name] = value
        for name in (
            "min_step_shrink",
            "max_step_growth",
            "safety",
            "deadband_min",
            "deadband_max",
        ):
            value = self._given(name)
            if value is None:
                value = self._family(name)
            if value is not None:
                settings[name] = value
        return settings

    # ------------------------------------------------------------------
    # Step keys
    # ------------------------------------------------------------------
    def _resolved_default(self, key: str) -> Any:
        """Given, else the DAE default on a mass system, else the family's."""
        value = self._given(key)
        if value is not None:
            return value
        if self.has_mass and key in DAE_SOLVER_DEFAULTS:
            return DAE_SOLVER_DEFAULTS[key]
        return self._family(key)

    @property
    def preconditioner_type(self) -> Optional[str]:
        """Return the preconditioner in effect."""
        return self._resolved_default("preconditioner_type")

    @property
    def linear_correction_type(self) -> Optional[str]:
        """Return the linear correction in effect."""
        return self._resolved_default("linear_correction_type")

    @property
    def step_settings(self) -> Dict[str, Any]:
        """Resolved step keys; Newton-variant defaults only with the
        family's linear solver."""
        settings = {}
        for key in (
            "preconditioner_type",
            "preconditioner_order",
            "linear_correction_type",
            "attempt_dense_prediction",
        ):
            value = self._resolved_default(key)
            if value is not None:
                settings[key] = value
        family_solver = self.linear_correction_type == self._family(
            "linear_correction_type"
        )
        for key in LINEAR_SOLVER_VARIANT_PARAMETERS:
            value = self._given(key)
            if value is None and family_solver:
                value = self._family(key)
            if value is not None:
                settings[key] = value
        settings.update(self.inner_tolerances)
        return settings

    @property
    def inner_tolerances(self) -> Dict[str, Any]:
        """Given tolerances, else Newton at a tenth of the controller's,
        Krylov at the controller's, reduction at the tightest adaptive
        ``rtol`` (a hundredth on a linear step) or machine epsilon."""
        if not self.is_implicit or self.controller_atol is None:
            return {}
        atol = asarray(self.controller_atol)
        rtol = asarray(self.controller_rtol)
        derived = {
            "krylov_atol": atol.copy(),
            "krylov_rtol": rtol.copy(),
            "newton_atol": atol / 10.0,
            "newton_rtol": rtol / 10.0,
        }
        rtol_floor = float(rtol.min())
        if self.is_adaptive and rtol_floor > 0.0:
            if self.is_linear:
                rtol_floor *= 0.01
            derived["krylov_residual_reduction"] = rtol_floor
        else:
            derived["krylov_residual_reduction"] = float(
                np_finfo(self.precision).eps
            )
        return {
            key: value if self._given(key) is None else self._given(key)
            for key, value in derived.items()
        }

    # ------------------------------------------------------------------
    # Loop schedule
    # ------------------------------------------------------------------
    @property
    def save_every(self) -> Optional[float]:
        """Return the save interval as given."""
        return self._save_every

    @property
    def summarise_every(self) -> Optional[float]:
        """Given, else the batch's window; ``None`` without summaries."""
        if not self.has_summary_outputs:
            return None
        if self._summarise_every is not None:
            return self._summarise_every
        return self.summary_window

    @property
    def sample_summaries_every(self) -> Optional[float]:
        """A hundredth of a derived window; else given, else a tenth."""
        window = self.summarise_every
        if window is None:
            return None
        if self._summarise_every is None:
            return window / DERIVED_SAMPLES_PER_SUMMARY
        if self._sample_summaries_every is not None:
            return self._sample_summaries_every
        return window / GIVEN_SAMPLES_PER_SUMMARY

    @property
    def summary_window_derived(self) -> bool:
        """Whether the summary window follows the duration."""
        return self.has_summary_outputs and self._summarise_every is None

    @property
    def save_last(self) -> bool:
        """Return whether only the final state is saved."""
        return self.has_time_domain_outputs and self._save_every is None

    @property
    def save_regularly(self) -> bool:
        """Return whether states are saved on the interval."""
        return self.has_time_domain_outputs and self._save_every is not None

    @property
    def summarise_regularly(self) -> bool:
        """Return whether summaries are committed on the window."""
        return self.summarise_every is not None

    @property
    def loop_timing(self) -> Dict[str, Any]:
        """Return the schedule to write into the loop."""
        return dict(
            save_every=self.save_every,
            summarise_every=self.summarise_every,
            sample_summaries_every=self.sample_summaries_every,
            save_last=self.save_last,
            save_regularly=self.save_regularly,
            summarise_regularly=self.summarise_regularly,
        )
