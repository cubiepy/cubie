"""Resolve subcomponent defaults from user-given arguments.

All user-intent interpretation and translation to low-level settings.

Published Classes
-----------------
:class:`VariableSelection`
    Variables chosen by label and by index.

Module-Level Functions
----------------------
:func:`resolve`
    Resolve user-given arguments to the low-level settings in effect.
"""

from math import sqrt
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

from attrs import field, fields, fields_dict, frozen, validators
from numpy import asarray, finfo as np_finfo, int32 as np_int32, ndarray

from cubie.batchsolving.solver_settings import EffectiveSettings
from cubie.integrators.algorithms import algorithm_facts
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
    LINEAR_SOLVER_VARIANT_PARAMETERS,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    DAE_SOLVER_DEFAULTS,
    ImplicitStepConfig,
)
from cubie.integrators.step_control import _CONTROLLER_REGISTRY
from cubie.integrators.step_control.adaptive_step_controller import (
    DEFAULT_DT_MAX,
    DEFAULT_DT_MIN,
)
from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
    BaseStepControllerConfig,
    CONTROLLER_GAIN_NAMES,
    promoted_gain_controller,
)
from cubie.integrators.step_control.fixed_step_controller import (
    DEFAULT_FIXED_DT,
)
from cubie.outputhandling.output_config import OutputConfig


DEFAULT_TOLERANCE = float(
    asarray(fields(BaseStepControllerConfig).atol.default).flat[0]
)
"""Controller tolerance used when none is given."""

STEP_BOUND_DECADES = 3
"""How many decades either side of dt to set the min and max step bounds."""

def given_or(given: Any, name: str, default: Any) -> Any:
    """Return the given value of ``name``, or ``default`` if not given."""
    value = getattr(given, name)
    return default if value is None else value


def _optional_index_array(value: Any) -> Optional[ndarray]:
    """Return ``value`` as an int32 index array; ``None`` stays ``None``."""
    if value is None:
        return None
    return asarray(value, dtype=np_int32).reshape(-1)


@frozen
class VariableSelection:
    """Variables chosen by label and by index; ``None`` means unselected.

    Attributes
    ----------
    variables
        State and observable labels.
    state_indices
        Indices into the system's states.
    observable_indices
        Indices into the system's observables.
    """

    variables: Optional[List[str]] = field(
        default=None,
        validator=validators.optional(
            validators.deep_iterable(validators.instance_of(str))
        ),
    )
    state_indices: Optional[ndarray] = field(
        default=None, converter=_optional_index_array
    )
    observable_indices: Optional[ndarray] = field(
        default=None, converter=_optional_index_array
    )

    @property
    def is_given(self) -> bool:
        """Return whether any label or index was given."""
        return (
            self.variables is not None
            or self.state_indices is not None
            or self.observable_indices is not None
        )

    def resolve(self, interface: Any) -> "VariableSelection":
        """Return the selection as index arrays.

        Labels resolve through ``interface``; ``None`` selects every
        variable.

        Raises
        ------
        ValueError
            A label or index the system does not have.
        """
        states, observables = interface.merge_variable_inputs(
            self.variables, self.state_indices, self.observable_indices
        )
        return VariableSelection(None, states, observables)


def resolve_controller(
    given: Any, facts: Any
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Return the controller name, its gains and its limits.

    Parameters
    ----------
    given
        The given settings.
    facts
        The algorithm's :class:`AlgorithmFacts`.

    Returns
    -------
    tuple[dict, str or None]
        The controller settings, and the adaptive controller replaced
        by ``fixed`` if the algorithm has no error estimate.

    Raises
    ------
    ValueError
        Gains given together with ``filter_coefficients``.
    """
    gains = [name for name in CONTROLLER_GAIN_NAMES if given.is_given(name)]
    if gains and given.filter_coefficients is not None:
        raise ValueError(
            f"filter_coefficients cannot be combined with {gains}; "
            "give one or the other."
        )
    defaults = facts.defaults.settings
    family_default_controller = defaults.get("step_controller", "fixed")
    if given.step_controller is not None:
        step_controller = given.step_controller.lower()
    else:
        # Promote the default controller to one carrying the given gains.
        step_controller = (
            promoted_gain_controller(
                family_default_controller, given.as_kwargs()
            )
            or family_default_controller
        )
    replaced = None
    if step_controller != "fixed" and not facts.has_error_estimate:
        # No error estimate: use fixed steps.
        replaced = step_controller
        step_controller = "fixed"
    resolved = {
        "step_controller": step_controller,
        "is_adaptive": step_controller != "fixed",
    }
    use_family_gains = step_controller == family_default_controller
    config_fields = fields_dict(
        _CONTROLLER_REGISTRY[step_controller]._config_class
    )
    if given.filter_coefficients is None:
        for name in CONTROLLER_GAIN_NAMES:
            if f"_{name}" not in config_fields:
                # The controller has no such gain; a given one is unused.
                if given.is_given(name):
                    resolved[name] = None
                continue
            default = config_fields[f"_{name}"].default
            if use_family_gains:
                default = defaults.get(name, default)
            resolved[name] = given_or(given, name, default)
    if resolved["is_adaptive"]:
        # Fill the family's other controller settings where none is given.
        for key, value in defaults.items():
            if (
                key in ALL_STEP_CONTROLLER_PARAMETERS
                and key not in CONTROLLER_GAIN_NAMES
                and key != "step_controller"
            ):
                resolved[key] = given_or(given, key, value)
    return resolved, replaced


def resolve_step_bounds(
    dt: Optional[float],
    dt_min: Optional[float],
    dt_max: Optional[float],
    is_adaptive: bool,
) -> Dict[str, Any]:
    """Fill the step bounds from ``dt``, or ``dt`` from the bounds.

    Parameters
    ----------
    dt, dt_min, dt_max
        The given step and bounds; ``None`` when not given.
    is_adaptive
        Whether the controller adapts the step.

    Returns
    -------
    dict
        ``dt`` for a fixed step; ``dt``, ``dt_min`` and ``dt_max``
        for an adaptive one.
    """
    if not is_adaptive:
        # Fixed: dt, else the bounds' geometric mean, else a lone bound.
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
    factor = 10.0**STEP_BOUND_DECADES
    # Missing bounds sit STEP_BOUND_DECADES either side of a given dt.
    if dt_min is None:
        if dt is not None:
            dt_min = dt / factor
        else:
            dt_min = DEFAULT_DT_MIN
            if dt_max is not None and dt_max < dt_min:
                dt_min = dt_max / factor
    if dt_max is None:
        if dt is not None:
            dt_max = dt * factor
        else:
            dt_max = DEFAULT_DT_MAX
            if dt_min > dt_max:
                dt_max = dt_min * factor
    # No dt: start at the bounds' geometric mean.
    if dt is None:
        dt = sqrt(dt_min * dt_max)
    return {"dt": dt, "dt_min": dt_min, "dt_max": dt_max}


def resolve_step_defaults(
    given: Any, facts: Any, has_mass: bool
) -> Dict[str, Any]:
    """Fill unset step settings from the family, tableau and DAE defaults.

    Parameters
    ----------
    given
        The given settings.
    facts
        The algorithm's :class:`AlgorithmFacts`.
    has_mass
        Whether the system has a mass matrix.

    Returns
    -------
    dict
        The step settings in effect.

    Raises
    ------
    ValueError
        A Neumann preconditioner on a mass-matrix system.
    """
    family_defaults = facts.defaults.settings
    defaults = dict(family_defaults)
    if has_mass:
        # A mass matrix needs the DAE linear solve.
        defaults.update(DAE_SOLVER_DEFAULTS)
    resolved = {}
    for key in sorted(defaults):
        if (
            key in ALL_ALGORITHM_STEP_PARAMETERS
            and key not in LINEAR_SOLVER_VARIANT_PARAMETERS
        ):
            resolved[key] = given_or(given, key, defaults[key])
    if has_mass and resolved.get("preconditioner_type") == "neumann":
        raise ValueError(
            "Neumann preconditioners assume an identity mass matrix and "
            "cannot precondition a system with torn algebraic rows. Use "
            "preconditioner_type='jacobi'."
        )
    # Family Newton-variant defaults apply only with the family solver.
    linear_solver = given_or(
        given,
        "linear_correction_type",
        defaults.get("linear_correction_type"),
    )
    use_family_variants = linear_solver == family_defaults.get(
        "linear_correction_type"
    )
    step_fields = fields_dict(ImplicitStepConfig)
    for key in LINEAR_SOLVER_VARIANT_PARAMETERS:
        default = step_fields[key].default
        if use_family_variants:
            default = family_defaults.get(key, default)
        resolved[key] = given_or(given, key, default)
    return resolved


def resolve_inner_tolerances(
    given: Any,
    atol: Any,
    rtol: Any,
    is_adaptive: bool,
    is_linear: bool,
    precision: type,
) -> Dict[str, Any]:
    """Fill the unset inner-solver tolerances from the controller's.

    Parameters
    ----------
    given
        The given settings.
    atol, rtol
        The controller tolerances in effect.
    is_adaptive
        Whether the controller adapts the step.
    is_linear
        Whether the step solves one linear system without Newton.
    precision
        The working precision.

    Returns
    -------
    dict
        The Krylov and Newton tolerances and the Krylov reduction.
    """
    atol = asarray(atol, dtype=float)
    rtol = asarray(rtol, dtype=float)
    # Krylov solves to the step tolerance; Newton one decade tighter.
    derived = {
        "krylov_atol": atol,
        "krylov_rtol": rtol,
        "newton_atol": atol / 10.0,
        "newton_rtol": rtol / 10.0,
    }
    rtol_floor = float(rtol.min())
    if is_adaptive and rtol_floor > 0.0:
        # Set the reduction 100x lower for a no-Newton step.
        if is_linear:
            rtol_floor *= 0.01
        derived["krylov_residual_reduction"] = rtol_floor
    else:
        derived["krylov_residual_reduction"] = float(np_finfo(precision).eps)
    return {
        key: given_or(given, key, value) for key, value in derived.items()
    }


def resolve_output_selection(
    given: Any, interface: Any
) -> Tuple[VariableSelection, VariableSelection]:
    """Return the saved and summarised selections as index arrays.

    Parameters
    ----------
    given
        The given settings.
    interface
        The :class:`SystemInterface` resolving labels to indices.

    Returns
    -------
    tuple[VariableSelection, VariableSelection]
        The saved and the summarised selection.
    """
    saved = VariableSelection(
        given.save_variables,
        given.saved_state_indices,
        given.saved_observable_indices,
    ).resolve(interface)
    summarised = VariableSelection(
        given.summarise_variables,
        given.summarised_state_indices,
        given.summarised_observable_indices,
    )
    # Set summaries to match the saved variables if none are given.
    if summarised.is_given:
        summarised = summarised.resolve(interface)
    else:
        summarised = saved
    return saved, summarised


def output_flags(
    output_types: List[str],
    saved: VariableSelection,
    summarised: VariableSelection,
    n_states: int,
    n_observables: int,
    precision: type,
) -> Tuple[bool, bool]:
    """Return whether the outputs include time-domain and summary arrays.

    Parameters
    ----------
    output_types
        The output type names.
    saved, summarised
        The resolved selections.
    n_states, n_observables
        The system's sizes.
    precision
        The working precision.

    Returns
    -------
    tuple[bool, bool]
        Whether time-domain outputs and whether summaries are produced.
    """
    config = OutputConfig.from_loop_settings(
        output_types=output_types,
        precision=precision,
        saved_state_indices=saved.state_indices,
        saved_observable_indices=saved.observable_indices,
        summarised_state_indices=summarised.state_indices,
        summarised_observable_indices=summarised.observable_indices,
        n_states=n_states,
        n_observables=n_observables,
    )
    time_domain = bool(
        config.save_time or config.save_state or config.save_observables
    )
    summaries = bool(config.summarise_state or config.summarise_observables)
    return time_domain, summaries


def resolve_loop_timing(
    save_every: Optional[float],
    summarise_every: Optional[float],
    sample_summaries_every: Optional[float],
    has_time_domain_outputs: bool,
    has_summary_outputs: bool,
) -> Dict[str, Any]:
    """Return the save and summary intervals and the flags selecting them.

    Parameters
    ----------
    save_every, summarise_every, sample_summaries_every
        The given intervals; ``None`` when not given.
    has_time_domain_outputs, has_summary_outputs
        Which output arrays are produced.

    Returns
    -------
    dict
        The intervals and ``save_last``, ``save_regularly``,
        ``summarise_last`` and ``summarise_regularly``.

    Raises
    ------
    ValueError
        Summary outputs requested without ``sample_summaries_every``.
    """
    # Outputs with no interval fire once at the end of the run.
    save_last = has_time_domain_outputs and save_every is None
    save_regularly = has_time_domain_outputs and save_every is not None
    summarise_last = has_summary_outputs and summarise_every is None
    summarise_regularly = has_summary_outputs and summarise_every is not None
    if not has_summary_outputs:
        summarise_every = None
        sample_summaries_every = None
    elif sample_summaries_every is None:
        raise ValueError(
            "When summary metrics are requested, you must provide a "
            "sampling period for the loop to collect summary samples by "
            "setting sample_summaries_every"
        )
    return {
        "save_every": save_every,
        "summarise_every": summarise_every,
        "sample_summaries_every": sample_summaries_every,
        "save_last": save_last,
        "save_regularly": save_regularly,
        "summarise_last": summarise_last,
        "summarise_regularly": summarise_regularly,
    }


def _newton_rtol_inverted(
    newton_rtol: Any, controller_rtol: Any, precision: type
) -> bool:
    """Return whether the floored Newton rtol reaches the controller's."""
    floor = 4.0 * float(np_finfo(precision).eps)
    newton = asarray(newton_rtol, dtype=float).copy()
    newton[(newton > 0.0) & (newton < floor)] = floor
    controller = asarray(controller_rtol, dtype=float)
    newton = newton.reshape(-1, controller.size)
    return bool(((controller > 0.0) & (newton >= controller)).any())


def check_loop_timing(
    timing: Dict[str, Any],
    duration: Optional[float],
    dt_min: float,
    precision: type,
) -> None:
    """Raise when the loop schedule would produce no output.

    ``duration`` plus ``dt_min`` is the end-time tolerance.

    Raises
    ------
    ValueError
        An interval longer than the run, or a sample interval that is
        not shorter than its window.
    """
    save_every = timing["save_every"]
    summarise_every = timing["summarise_every"]
    sample_every = timing["sample_summaries_every"]
    if timing["summarise_regularly"] and sample_every >= summarise_every:
        raise ValueError(
            f"sample_summaries_every ({sample_every}) >= summarise_every "
            f"({summarise_every}); The saved summary will be based on 0 "
            f"samples, so will result in 0/inf/NaN values."
        )
    if duration is None:
        return
    end_time = precision(duration) + dt_min
    if timing["save_regularly"] and save_every > end_time:
        raise ValueError(
            f"save_every ({save_every}) > duration ({duration}) so this "
            f"loop will produce no outputs"
        )
    if timing["summarise_last"] and sample_every > end_time:
        raise ValueError(
            f"sample_summaries_every ({sample_every}) > duration "
            f"({duration}), so the summary at the end will be based on 0 "
            f"samples"
        )
    if timing["summarise_regularly"] and summarise_every > end_time:
        raise ValueError(
            f"summarise_every ({summarise_every}) > duration ({duration}), "
            f"so this loop will produce no summary outputs"
        )


def resolve(given: Any, system: Any, interface: Any) -> EffectiveSettings:
    """Resolve user-given arguments to the low-level settings in effect.

    Parameters
    ----------
    given
        The given settings, a :class:`SolverSettings`.
    system
        The system to solve; its precision is the one in effect.
    interface
        The :class:`SystemInterface` resolving labels to indices.

    Returns
    -------
    EffectiveSettings
        The given settings with the resolved ones filled in.

    Raises
    ------
    ValueError
        A Neumann preconditioner on a mass-matrix system, gains given
        with a filter, an output index the system does not have, summary
        metrics without ``sample_summaries_every``, or an output interval
        the run cannot fit.
    """
    precision = system.precision
    resolved = {"precision": precision}
    sizes = system.sizes
    has_mass = system.mass is not None

    algorithm = given_or(given, "algorithm", "euler")
    facts = algorithm_facts(algorithm, given.tableau)
    resolved["algorithm"] = algorithm

    controller, replaced = resolve_controller(given, facts)
    if replaced is not None:
        warn(
            f"Adaptive step controller '{replaced}' cannot be used with "
            f"fixed-step algorithm '{algorithm}'. The algorithm does not "
            "provide an error estimate required for adaptive stepping. "
            "Replacing with fixed-step controller.",
            UserWarning,
            stacklevel=2,
        )
    resolved.update(controller)
    resolved.update(
        resolve_step_bounds(
            given.dt, given.dt_min, given.dt_max, resolved["is_adaptive"]
        )
    )
    resolved.update(resolve_step_defaults(given, facts, has_mass))

    resolved["atol"] = given_or(given, "atol", DEFAULT_TOLERANCE)
    resolved["rtol"] = given_or(given, "rtol", DEFAULT_TOLERANCE)
    if facts.is_implicit:
        resolved.update(
            resolve_inner_tolerances(
                given,
                resolved["atol"],
                resolved["rtol"],
                resolved["is_adaptive"],
                facts.is_linear,
                precision,
            )
        )
        if not facts.is_linear and _newton_rtol_inverted(
            resolved["newton_rtol"], resolved["rtol"], precision
        ):
            warn(
                "newton_rtol is at or above the step controller rtol: "
                "the requested rtol is below what the working precision "
                "resolves in the stage solves.",
                UserWarning,
                stacklevel=2,
            )

    saved, summarised = resolve_output_selection(given, interface)
    output_types = given_or(given, "output_types", ["state"])
    resolved.update(
        output_types=output_types,
        saved_state_indices=saved.state_indices,
        saved_observable_indices=saved.observable_indices,
        summarised_state_indices=summarised.state_indices,
        summarised_observable_indices=summarised.observable_indices,
    )
    time_domain, summaries = output_flags(
        output_types,
        saved,
        summarised,
        int(sizes.states),
        int(sizes.observables),
        precision,
    )
    timing = resolve_loop_timing(
        given.save_every,
        given.summarise_every,
        given.sample_summaries_every,
        time_domain,
        summaries,
    )
    # Fixed control has no dt_min; its step is the tolerance.
    check_loop_timing(
        timing,
        given.duration,
        resolved.get("dt_min", resolved["dt"]),
        precision,
    )
    resolved.update(timing)
    return EffectiveSettings(**{**given.as_kwargs(), **resolved})
