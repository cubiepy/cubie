"""User-given solver settings and the effective values resolved from them.

Published Classes
-----------------
:class:`SolverSettings`
    ``user_given`` and ``effective``.
:class:`ResolvedSettings`
    One resolution's effective settings and notices.

Module-Level Functions
----------------------
:func:`resolve_settings`
    Effective settings for a system.
:func:`resolve_loop_timing`
    Loop schedule and flags.
:func:`resolve_step_bounds`
    ``dt``, ``dt_min`` and ``dt_max``.
:func:`resolve_inner_tolerances`
    Inner-solver tolerances.
"""

from math import sqrt
from typing import Any, Dict, Mapping, Optional, Set

from attrs import define, field, fields_dict, frozen
from numpy import asarray, finfo as np_finfo

from cubie.cuda_simsafe import ALL_UNROLL_PARAMETERS, UnrollFlags
from cubie.integrators.algorithms import algorithm_facts
from cubie.integrators.algorithms.base_algorithm_step import (
    LINEAR_SOLVER_VARIANT_PARAMETERS,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    DAE_SOLVER_DEFAULTS,
)
from cubie.integrators.step_control import _CONTROLLER_REGISTRY
from cubie.integrators.step_control.adaptive_step_controller import (
    DEFAULT_DT_MAX,
    DEFAULT_DT_MIN,
)
from cubie.integrators.step_control.base_step_controller import (
    BaseStepControllerConfig,
    CONTROLLER_GAIN_NAMES,
    promoted_gain_controller,
)
from cubie.integrators.step_control.fixed_step_controller import (
    DEFAULT_FIXED_DT,
)
from cubie.outputhandling.output_config import OutputConfig


STEP_DEFAULT_KEYS = (
    "preconditioner_type",
    "preconditioner_order",
    "linear_correction_type",
    "attempt_dense_prediction",
)
"""Step keys the family, tableau and DAE defaults fill."""

CONTROLLER_LIMIT_KEYS = (
    "min_step_shrink",
    "max_step_growth",
    "safety",
    "deadband_min",
    "deadband_max",
)
"""Adaptive-controller keys the family defaults fill."""

INNER_TOLERANCE_KEYS = (
    "krylov_atol",
    "krylov_rtol",
    "krylov_residual_reduction",
    "newton_atol",
    "newton_rtol",
)
"""Inner-solver tolerances derived from the controller's."""

DERIVED_SAMPLES_PER_SUMMARY = 100
"""Summary samples per window derived from the duration."""

GIVEN_SAMPLES_PER_SUMMARY = 10
"""Summary samples per window when only the window is given."""

DEFAULT_TOLERANCE = float(
    asarray(fields_dict(BaseStepControllerConfig)["atol"].default).flat[0]
)
"""Controller tolerance when none is given."""


def controller_gain_names(step_controller: str) -> tuple:
    """Return the gain keys ``step_controller``'s config carries."""
    config_class = _CONTROLLER_REGISTRY[step_controller]._config_class
    config_fields = fields_dict(config_class)
    return tuple(
        name for name in CONTROLLER_GAIN_NAMES if f"_{name}" in config_fields
    )


def resolve_step_bounds(
    dt: Optional[float],
    dt_min: Optional[float],
    dt_max: Optional[float],
    is_adaptive: bool,
) -> Dict[str, float]:
    """Fill missing step bounds from ``dt`` and a missing ``dt`` from them."""
    if not is_adaptive:
        if dt is None:
            if dt_min is not None and dt_max is not None:
                dt = sqrt(dt_min * dt_max)
            elif dt_min is not None:
                dt = dt_min
            elif dt_max is not None:
                dt = dt_max
            else:
                dt = DEFAULT_FIXED_DT
        return {"dt": dt}
    if dt_min is None:
        if dt is not None:
            dt_min = dt / 100
        else:
            dt_min = DEFAULT_DT_MIN
            if dt_max is not None and dt_max < dt_min:
                dt_min = dt_max / 100
    if dt_max is None:
        if dt is not None:
            dt_max = dt * 100
        else:
            dt_max = DEFAULT_DT_MAX
            if dt_min > dt_max:
                dt_max = dt_min * 100
    if dt is None:
        dt = sqrt(dt_min * dt_max)
    return {"dt": dt, "dt_min": dt_min, "dt_max": dt_max}


def _scalar_or_array(value):
    """Return a 0-d array as a float, any other array as is."""
    if value.ndim == 0:
        return float(value)
    return value


def resolve_inner_tolerances(
    given: Mapping[str, Any],
    atol: Any,
    rtol: Any,
    is_adaptive: bool,
    is_linear: bool,
    precision: type,
) -> Dict[str, Any]:
    """Fill unset inner tolerances from the controller's."""
    atol = asarray(atol, dtype=float)
    rtol = asarray(rtol, dtype=float)
    derived = {
        "krylov_atol": _scalar_or_array(atol),
        "krylov_rtol": _scalar_or_array(rtol),
        "newton_atol": _scalar_or_array(atol / 10.0),
        "newton_rtol": _scalar_or_array(rtol / 10.0),
    }
    rtol_floor = float(rtol.min())
    if is_adaptive and rtol_floor > 0.0:
        if is_linear:
            rtol_floor *= 0.01
        derived["krylov_residual_reduction"] = rtol_floor
    else:
        derived["krylov_residual_reduction"] = float(np_finfo(precision).eps)
    return {
        key: given[key] if given.get(key) is not None else value
        for key, value in derived.items()
    }


def resolve_loop_timing(
    save_every: Optional[float],
    summarise_every: Optional[float],
    sample_summaries_every: Optional[float],
    has_time_domain_outputs: bool,
    has_summary_outputs: bool,
    duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the loop schedule and flags for ``duration``."""
    save_last = has_time_domain_outputs and save_every is None
    save_regularly = has_time_domain_outputs and save_every is not None
    if not has_summary_outputs:
        summarise_every = None
        sample_summaries_every = None
    elif summarise_every is None:
        summarise_every = duration
        if duration is not None:
            sample_summaries_every = duration / DERIVED_SAMPLES_PER_SUMMARY
        else:
            sample_summaries_every = None
    elif sample_summaries_every is None:
        sample_summaries_every = summarise_every / GIVEN_SAMPLES_PER_SUMMARY
    return {
        "save_every": save_every,
        "summarise_every": summarise_every,
        "sample_summaries_every": sample_summaries_every,
        "save_last": save_last,
        "save_regularly": save_regularly,
        "summarise_regularly": summarise_every is not None,
    }


def _in_range(indices: Any, size: int) -> Any:
    """Return ``indices`` without those the system does not have."""
    if indices is None:
        return None
    return [index for index in asarray(indices).ravel() if 0 <= index < size]


def output_flags(
    given: Mapping[str, Any], n_states: int, n_observables: int, precision
) -> Dict[str, bool]:
    """Return which output kinds the given output settings produce."""
    config = OutputConfig.from_loop_settings(
        output_types=given.get("output_types") or ["state"],
        precision=precision,
        saved_state_indices=_in_range(
            given.get("saved_state_indices"), n_states
        ),
        saved_observable_indices=_in_range(
            given.get("saved_observable_indices"), n_observables
        ),
        summarised_state_indices=_in_range(
            given.get("summarised_state_indices"), n_states
        ),
        summarised_observable_indices=_in_range(
            given.get("summarised_observable_indices"), n_observables
        ),
        n_states=n_states,
        n_observables=n_observables,
    )
    return {
        "has_time_domain_outputs": bool(
            config.save_time or config.save_state or config.save_observables
        ),
        "has_summary_outputs": bool(
            config.summarise_state or config.summarise_observables
        ),
    }


@frozen
class ResolvedSettings:
    """One resolution's effective settings and notices.

    Attributes
    ----------
    effective
        Settings to push down.
    replaced_controller
        Requested controller replaced by ``fixed``, else ``None``.
    summary_window_derived
        Whether the summary window follows the duration.
    """

    effective: Dict[str, Any]
    replaced_controller: Optional[str]
    summary_window_derived: bool


def resolve_settings(
    user_given: Mapping[str, Any],
    system: Any,
    duration: Optional[float] = None,
) -> ResolvedSettings:
    """Resolve the effective settings for ``system``.

    Raises
    ------
    ValueError
        Neumann preconditioner on a mass-matrix system.
    """
    given = {
        key: value for key, value in user_given.items() if value is not None
    }
    effective = dict(given)
    precision = system.precision
    sizes = system.sizes
    has_mass = system.mass is not None

    algorithm = given.get("algorithm", "euler")
    facts = algorithm_facts(algorithm, given.get("tableau"))
    defaults = facts.defaults.settings
    effective["algorithm"] = algorithm

    family_controller = defaults.get("step_controller", "fixed")
    if "step_controller" in given:
        step_controller = given["step_controller"].lower()
    else:
        step_controller = (
            promoted_gain_controller(family_controller, given)
            or family_controller
        )
    replaced_controller = None
    if step_controller != "fixed" and not facts.has_error_estimate:
        replaced_controller = step_controller
        step_controller = "fixed"
    is_adaptive = step_controller != "fixed"
    effective["step_controller"] = step_controller
    effective["is_adaptive"] = is_adaptive

    gain_names = controller_gain_names(step_controller)
    family_gains = (
        step_controller == family_controller
        and given.get("filter_coefficients") is None
    )
    for name in CONTROLLER_GAIN_NAMES:
        value = given.get(name)
        if value is None and family_gains:
            value = defaults.get(name)
        if value is not None and name in gain_names:
            effective[name] = value
        else:
            effective.pop(name, None)
    if is_adaptive:
        for key in CONTROLLER_LIMIT_KEYS:
            value = given.get(key, defaults.get(key))
            if value is not None:
                effective[key] = value
    effective.update(
        resolve_step_bounds(
            given.get("dt"),
            given.get("dt_min"),
            given.get("dt_max"),
            is_adaptive,
        )
    )

    for key in STEP_DEFAULT_KEYS:
        value = given.get(key)
        if value is None and has_mass and key in DAE_SOLVER_DEFAULTS:
            value = DAE_SOLVER_DEFAULTS[key]
        if value is None:
            value = defaults.get(key)
        if value is not None:
            effective[key] = value
    if has_mass and effective.get("preconditioner_type") == "neumann":
        raise ValueError(
            "Neumann preconditioners assume an identity mass matrix and "
            "cannot precondition a system with torn algebraic rows. Use "
            "preconditioner_type='jacobi'."
        )
    family_solver = effective.get("linear_correction_type") == defaults.get(
        "linear_correction_type"
    )
    for key in LINEAR_SOLVER_VARIANT_PARAMETERS:
        value = given.get(key)
        if value is None and family_solver:
            value = defaults.get(key)
        if value is not None:
            effective[key] = value
    effective["atol"] = given.get("atol", DEFAULT_TOLERANCE)
    effective["rtol"] = given.get("rtol", DEFAULT_TOLERANCE)
    if facts.is_implicit:
        effective.update(
            resolve_inner_tolerances(
                given,
                effective["atol"],
                effective["rtol"],
                is_adaptive,
                facts.is_linear,
                precision,
            )
        )

    flags = output_flags(
        given, int(sizes.states), int(sizes.observables), precision
    )
    effective.update(
        resolve_loop_timing(
            given.get("save_every"),
            given.get("summarise_every"),
            given.get("sample_summaries_every"),
            duration=duration,
            **flags,
        )
    )
    summary_window_derived = (
        flags["has_summary_outputs"] and "summarise_every" not in given
    )
    return ResolvedSettings(
        effective=effective,
        replaced_controller=replaced_controller,
        summary_window_derived=summary_window_derived,
    )


def unroll_flags_as_settings(unroll: Optional[UnrollFlags]) -> Dict[str, Any]:
    """Return an ``UnrollFlags`` as the loose ``unroll_*`` keys it sets."""
    if unroll is None:
        return {}
    return {key: getattr(unroll, key) for key in ALL_UNROLL_PARAMETERS}


@define
class SolverSettings:
    """User-given settings and the last resolved effective settings.

    Attributes
    ----------
    user_given
        Settings as passed, without ``None`` values.
    effective
        Settings from the last resolution.
    """

    user_given: Dict[str, Any] = field(factory=dict)
    effective: Dict[str, Any] = field(factory=dict)

    def give(self, updates: Mapping[str, Any]) -> None:
        """Record ``updates``; ``None`` unsets; gains and a filter exclude."""
        for key, value in updates.items():
            if value is None:
                self.user_given.pop(key, None)
            else:
                self.user_given[key] = value
        gains_given = set(updates) & set(CONTROLLER_GAIN_NAMES)
        if gains_given and "filter_coefficients" not in updates:
            self.user_given.pop("filter_coefficients", None)
        elif "filter_coefficients" in updates and not gains_given:
            for name in CONTROLLER_GAIN_NAMES:
                self.user_given.pop(name, None)

    def resolve(
        self, system: Any, duration: Optional[float] = None
    ) -> ResolvedSettings:
        """Resolve and record the effective settings for ``system``."""
        resolved = resolve_settings(self.user_given, system, duration)
        self.effective = resolved.effective
        return resolved

    def given_keys(self, keys: Set[str]) -> Set[str]:
        """Return the members of ``keys`` the user gave."""
        return set(self.user_given) & set(keys)
