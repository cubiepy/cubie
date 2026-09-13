"""User-provided solver settings and the effective values resolved from them.

Published Classes
-----------------
:class:`SolverSettings`
    One attrs record of every user-facing setting. The ``Solver`` holds
    one instance as provided and one as resolved.
:class:`Resolution`
    A resolved :class:`SolverSettings` and the notices to warn about.

Module-Level Functions
----------------------
:func:`resolve`
    Effective settings for a system.
:func:`resolve_performance`
    Unroll and placement values from the built step.
"""

from math import sqrt
from typing import Any, Dict, List, Optional, Set, Tuple

from attrs import NOTHING, define, evolve, field, fields
from numpy import asarray, finfo as np_finfo

from cubie._utils import unpack_dict_values
from cubie.backend.utils import SASS_INSTRUCTION_BYTES, device_hardware
from cubie.batchsolving.BatchSolverConfig import (
    ALL_CACHE_PARAMETERS,
    cache_settings_converter,
)
from cubie.cuda_simsafe import (
    ALL_UNROLL_PARAMETERS,
    JITFlags,
    UnrollChoice,
    UnrollFlags,
)
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


DEFAULT_SAMPLES_PER_SUMMARY = 10
"""Summary samples per window when only the window is known."""

DEFAULT_TOLERANCE = float(
    asarray(fields(BaseStepControllerConfig).atol.default).flat[0]
)
"""Controller tolerance when none is provided."""

RENAMED_TIMING_KWARGS = {
    "dt_save": "save_every",
    "dt_summarise": "summarise_every",
    "dt_update_summaries": "sample_summaries_every",
}
"""Legacy timing keyword spellings mapped to their current names."""


def _setting(**kwargs) -> Any:
    """A user-facing setting; ``None`` means not provided."""
    return field(default=None, **kwargs)


def _local(**kwargs) -> Any:
    """A setting the Solver keeps to itself, never pushed to the kernel."""
    return field(default=None, metadata={"push": False}, **kwargs)


@define
class SolverSettings:
    """Every user-facing solver setting; ``None`` is not provided.

    The Solver keeps one instance as the user provided it and one as
    :func:`resolve` filled it. Grouped by the component that consumes
    each setting.
    """

    # The system: applied to it before anything else resolves.
    precision: Optional[type] = _setting()
    operation_ordering: Optional[str] = _setting()
    system_constants: Optional[Dict[str, float]] = _local()

    # The algorithm step.
    algorithm: Any = _setting()
    tableau: Any = _setting()
    attempt_dense_prediction: Optional[bool] = _setting()
    beta: Optional[float] = _setting()
    gamma: Optional[float] = _setting()
    preconditioner_order: Optional[int] = _setting()
    preconditioner_type: Optional[str] = _setting()
    linear_correction_type: Optional[str] = _setting()
    inexact_newton: Optional[bool] = _setting()
    prefactored: Optional[bool] = _setting()
    use_smoothed_error: Optional[bool] = _setting()
    dae_initialisation: Optional[str] = _setting()
    krylov_atol: Any = _setting()
    krylov_rtol: Any = _setting()
    krylov_max_iters: Optional[int] = _setting()
    krylov_residual_reduction: Optional[float] = _setting()
    krylov_residual_floor: Optional[float] = _setting()
    newton_atol: Any = _setting()
    newton_rtol: Any = _setting()
    newton_max_iters: Optional[int] = _setting()
    error_atol: Any = _setting()
    error_rtol: Any = _setting()
    error_max_iters: Optional[int] = _setting()
    error_residual_reduction: Optional[float] = _setting()
    error_residual_floor: Optional[float] = _setting()

    # The step's buffer placements.
    stage_increment_location: Optional[str] = _setting()
    stage_increment_history_location: Optional[str] = _setting()
    stage_base_location: Optional[str] = _setting()
    accumulator_location: Optional[str] = _setting()
    previous_step_size_location: Optional[str] = _setting()
    predictor_transform_location: Optional[str] = _setting()
    predictor_previous_values_location: Optional[str] = _setting()
    stage_rhs_location: Optional[str] = _setting()
    stage_accumulator_location: Optional[str] = _setting()
    stage_driver_stack_location: Optional[str] = _setting()
    stage_state_location: Optional[str] = _setting()
    stage_store_location: Optional[str] = _setting()
    cached_auxiliaries_location: Optional[str] = _setting()
    increment_cache_location: Optional[str] = _setting()
    dxdt_location: Optional[str] = _setting()
    preconditioned_vec_location: Optional[str] = _setting()
    temp_location: Optional[str] = _setting()
    r0_hat_location: Optional[str] = _setting()
    p_location: Optional[str] = _setting()
    v_location: Optional[str] = _setting()
    tmp_location: Optional[str] = _setting()
    s_hat_location: Optional[str] = _setting()
    lu_factor_location: Optional[str] = _setting()
    delta_location: Optional[str] = _setting()
    residual_location: Optional[str] = _setting()
    krylov_iters_local_location: Optional[str] = _setting()
    prev_theta_location: Optional[str] = _setting()
    base_state_placeholder_location: Optional[str] = _setting()
    krylov_iters_out_location: Optional[str] = _setting()

    # The step controller.
    step_controller: Optional[str] = _setting()
    is_adaptive: Optional[bool] = _setting()
    dt: Optional[float] = _setting()
    dt_min: Optional[float] = _setting()
    dt_max: Optional[float] = _setting()
    atol: Any = _setting()
    rtol: Any = _setting()
    min_step_shrink: Optional[float] = _setting()
    max_step_growth: Optional[float] = _setting()
    safety: Optional[float] = _setting()
    integral_gain: Any = _setting()
    proportional_gain: Any = _setting()
    derivative_gain: Any = _setting()
    filter_coefficients: Any = _setting()
    deadband_min: Optional[float] = _setting()
    deadband_max: Optional[float] = _setting()
    newton_target_iters: Optional[int] = _setting()
    timestep_memory_location: Optional[str] = _setting()

    # The loop's schedule and buffer placements.
    duration: Optional[float] = _local()
    save_every: Optional[float] = _setting()
    summarise_every: Optional[float] = _setting()
    sample_summaries_every: Optional[float] = _setting()
    save_last: Optional[bool] = _setting()
    save_regularly: Optional[bool] = _setting()
    summarise_regularly: Optional[bool] = _setting()
    state_location: Optional[str] = _setting()
    proposed_state_location: Optional[str] = _setting()
    parameters_location: Optional[str] = _setting()
    drivers_location: Optional[str] = _setting()
    proposed_drivers_location: Optional[str] = _setting()
    observables_location: Optional[str] = _setting()
    proposed_observables_location: Optional[str] = _setting()
    error_location: Optional[str] = _setting()
    counters_location: Optional[str] = _setting()
    state_summary_location: Optional[str] = _setting()
    observable_summary_location: Optional[str] = _setting()
    dt_location: Optional[str] = _setting()
    accept_step_location: Optional[str] = _setting()
    proposed_counters_location: Optional[str] = _setting()

    # The outputs: labels resolve to the index arrays.
    output_types: Optional[List[str]] = _setting()
    save_variables: Optional[List[str]] = _local()
    summarise_variables: Optional[List[str]] = _local()
    saved_state_indices: Any = _setting()
    saved_observable_indices: Any = _setting()
    summarised_state_indices: Any = _setting()
    summarised_observable_indices: Any = _setting()

    # The kernel.
    blocksize: Optional[int] = _setting()
    max_registers: Optional[int] = _setting()
    kernel_name: Optional[str] = _setting()
    cache: Any = _setting()
    cache_enabled: Optional[bool] = _setting()
    cache_mode: Optional[str] = _setting()
    max_cache_entries: Optional[int] = _setting()
    cache_dir: Any = _setting()
    auto_performance: Optional[bool] = _setting()
    lineinfo: Optional[bool] = _setting()

    # Loop unrolling, one flag per loop group.
    unroll_stage: Any = _setting()
    unroll_step_element: Any = _setting()
    unroll_accumulator: Any = _setting()
    unroll_solver_element: Any = _setting()
    unroll_norms: Any = _setting()
    unroll_other_small: Any = _setting()
    unroll_newton_exits: Any = _setting()
    unroll_krylov_exits: Any = _setting()

    # Driver interpolation.
    order: Optional[int] = _setting()
    wrap: Optional[bool] = _setting()
    boundary_condition: Optional[str] = _setting()

    # Memory.
    memory_manager: Any = _setting()
    stream_group: Optional[str] = _setting()
    mem_proportion: Optional[float] = _setting()
    host_spill_threshold: Optional[int] = _setting()
    spill_directory: Any = _setting()

    def __attrs_post_init__(self) -> None:
        """Reject gains given together with a filter."""
        gains = [name for name in CONTROLLER_GAIN_NAMES if self.given(name)]
        if gains and self.filter_coefficients is not None:
            raise ValueError(
                f"filter_coefficients cannot be combined with {gains}; "
                "give one or the other."
            )

    # ------------------------------------------------------------------
    # Construction and update
    # ------------------------------------------------------------------
    @classmethod
    def names(cls) -> Tuple[str, ...]:
        """Return every setting name."""
        return tuple(fld.name for fld in fields(cls))

    @staticmethod
    def _flatten(
        kwargs: Dict[str, Any], strict: bool = True
    ) -> Tuple[Dict[str, Any], Set[str]]:
        """Flatten grouped dicts and an ``unroll`` object; check names."""
        kwargs = dict(kwargs)
        # The constants dict is a value, not a group.
        constants = kwargs.pop("system_constants", NOTHING)
        flat, grouped = unpack_dict_values(kwargs)
        if constants is not NOTHING:
            flat["system_constants"] = constants
        renamed = [key for key in flat if key in RENAMED_TIMING_KWARGS]
        if renamed:
            hints = ", ".join(
                f"'{key}' is now '{RENAMED_TIMING_KWARGS[key]}'"
                for key in renamed
            )
            raise KeyError(f"Renamed keyword argument(s): {hints}.")
        recognised = set(grouped)
        if "unroll" in flat:
            unroll = flat.pop("unroll")
            recognised.add("unroll")
            if unroll is not None:
                for key in ALL_UNROLL_PARAMETERS:
                    flat.setdefault(key, getattr(unroll, key))
        names = set(SolverSettings.names())
        unknown = set(flat) - names
        if unknown and strict:
            raise KeyError(f"Unrecognized keyword arguments: {unknown}")
        flat = {key: value for key, value in flat.items() if key in names}
        return flat, recognised | set(flat)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "SolverSettings":
        """Record ``kwargs`` as provided; grouped dicts are flattened."""
        flat, _ = cls._flatten(kwargs)
        return cls(**flat)

    def updated(
        self, updates: Dict[str, Any], strict: bool = True
    ) -> Tuple["SolverSettings", Set[str]]:
        """Return a copy with ``updates`` recorded and the names taken.

        ``None`` unsets a setting. Unknown names raise ``KeyError``
        unless ``strict`` is ``False``.
        """
        flat, recognised = self._flatten(updates, strict=strict)
        return evolve(self, **flat), recognised

    def update(self, updates: Dict[str, Any]) -> "SolverSettings":
        """Return a copy with ``updates`` recorded; ``None`` unsets."""
        return self.updated(updates)[0]

    def given(self, name: str) -> bool:
        """Whether ``name`` was provided."""
        return getattr(self, name) is not None

    @property
    def given_names(self) -> Set[str]:
        """The names provided."""
        return {name for name in self.names() if self.given(name)}

    def as_kwargs(self) -> Dict[str, Any]:
        """The provided settings as ``Solver`` keyword arguments."""
        return {name: getattr(self, name) for name in self.given_names}

    def as_updates(self) -> Dict[str, Any]:
        """The settings the kernel takes, with the unroll and jit objects."""
        updates = {
            fld.name: getattr(self, fld.name)
            for fld in fields(type(self))
            if fld.metadata.get("push", True)
            and getattr(self, fld.name) is not None
        }
        unroll = {
            key: updates.pop(key)
            for key in ALL_UNROLL_PARAMETERS
            if key in updates
        }
        if unroll:
            updates["unroll"] = UnrollFlags(**unroll)
        if "lineinfo" in updates:
            updates["jit_flags"] = JITFlags(lineinfo=updates.pop("lineinfo"))
        cache_keys = {
            key: updates.pop(key)
            for key in ALL_CACHE_PARAMETERS
            if key in updates
        }
        if cache_keys:
            cache = cache_settings_converter(updates.get("cache", True))
            updates["cache"], _, _ = cache.update(cache_keys)
        return updates


@define
class Resolution:
    """A resolved :class:`SolverSettings` and the notices to warn about."""

    effective: SolverSettings
    notices: Tuple[str, ...] = ()


# ----------------------------------------------------------------------
# Resolution stages
# ----------------------------------------------------------------------
def _controller_gain_names(step_controller: str) -> Tuple[str, ...]:
    """Return the gain keys ``step_controller``'s config carries."""
    config_class = _CONTROLLER_REGISTRY[step_controller]._config_class
    names = {fld.name for fld in fields(config_class)}
    return tuple(name for name in CONTROLLER_GAIN_NAMES if f"_{name}" in names)


def resolve_controller(
    given: SolverSettings, facts: Any
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Return the controller name, gains and limits; the name replaced."""
    defaults = facts.defaults.settings
    family_controller = defaults.get("step_controller", "fixed")
    if given.step_controller is not None:
        step_controller = given.step_controller.lower()
    else:
        # Provided gains promote the family controller to carry them.
        step_controller = (
            promoted_gain_controller(family_controller, given.as_kwargs())
            or family_controller
        )
    replaced = None
    # A step with no error estimate can only run fixed steps.
    if step_controller != "fixed" and not facts.has_error_estimate:
        replaced = step_controller
        step_controller = "fixed"
    resolved = {
        "step_controller": step_controller,
        "is_adaptive": step_controller != "fixed",
    }
    # Family gains apply to the family controller without a filter.
    family_gains = (
        step_controller == family_controller
        and given.filter_coefficients is None
    )
    carried = _controller_gain_names(step_controller)
    for name in CONTROLLER_GAIN_NAMES:
        value = getattr(given, name)
        if value is None and family_gains:
            value = defaults.get(name)
        # A controller drops the gains it does not carry.
        resolved[name] = value if name in carried else None
    if resolved["is_adaptive"]:
        for key in (
            "min_step_shrink",
            "max_step_growth",
            "safety",
            "deadband_min",
            "deadband_max",
        ):
            value = getattr(given, key)
            if value is None:
                value = defaults.get(key)
            if value is not None:
                resolved[key] = value
    return resolved, replaced


def resolve_step_bounds(
    dt: Optional[float],
    dt_min: Optional[float],
    dt_max: Optional[float],
    is_adaptive: bool,
) -> Dict[str, float]:
    """Fill the missing step bounds from ``dt``, or ``dt`` from them."""
    if not is_adaptive:
        # A fixed step takes dt, else the bounds' geometric mean.
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
    # A lone dt bounds itself two decades either side.
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
    # Bounds alone start at their geometric mean.
    if dt is None:
        dt = sqrt(dt_min * dt_max)
    return {"dt": dt, "dt_min": dt_min, "dt_max": dt_max}


def resolve_step_defaults(
    given: SolverSettings, facts: Any, has_mass: bool
) -> Dict[str, Any]:
    """Fill the unset step keys from the family, tableau and DAE tables."""
    defaults = facts.defaults.settings
    resolved = {}
    for key in (
        "preconditioner_type",
        "preconditioner_order",
        "linear_correction_type",
        "attempt_dense_prediction",
    ):
        value = getattr(given, key)
        # A mass matrix takes the DAE linear solve over the family's.
        if value is None and has_mass and key in DAE_SOLVER_DEFAULTS:
            value = DAE_SOLVER_DEFAULTS[key]
        if value is None:
            value = defaults.get(key)
        if value is not None:
            resolved[key] = value
    if has_mass and resolved.get("preconditioner_type") == "neumann":
        raise ValueError(
            "Neumann preconditioners assume an identity mass matrix and "
            "cannot precondition a system with torn algebraic rows. Use "
            "preconditioner_type='jacobi'."
        )
    # Newton-variant defaults belong to the family's linear solver.
    family_solver = resolved.get("linear_correction_type") == defaults.get(
        "linear_correction_type"
    )
    for key in LINEAR_SOLVER_VARIANT_PARAMETERS:
        value = getattr(given, key)
        if value is None and family_solver:
            value = defaults.get(key)
        if value is not None:
            resolved[key] = value
    return resolved


def resolve_inner_tolerances(
    given: SolverSettings,
    atol: Any,
    rtol: Any,
    is_adaptive: bool,
    is_linear: bool,
    precision: type,
) -> Dict[str, Any]:
    """Fill the unset inner-solver tolerances from the controller's."""
    atol = asarray(atol, dtype=float)
    rtol = asarray(rtol, dtype=float)

    def scalar_or_array(value):
        return float(value) if value.ndim == 0 else value

    # Newton converges a decade tighter than the step it feeds.
    derived = {
        "krylov_atol": scalar_or_array(atol),
        "krylov_rtol": scalar_or_array(rtol),
        "newton_atol": scalar_or_array(atol / 10.0),
        "newton_rtol": scalar_or_array(rtol / 10.0),
    }
    # The Krylov reduction tracks the tightest adaptive rtol.
    rtol_floor = float(rtol.min())
    if is_adaptive and rtol_floor > 0.0:
        if is_linear:
            rtol_floor *= 0.01
        derived["krylov_residual_reduction"] = rtol_floor
    else:
        derived["krylov_residual_reduction"] = float(np_finfo(precision).eps)
    return {
        key: getattr(given, key) if given.given(key) else value
        for key, value in derived.items()
    }


def resolve_output_selection(
    given: SolverSettings, interface: Any
) -> Dict[str, Any]:
    """Return the output types and the index arrays the labels select."""
    resolved = {"output_types": given.output_types or ["state"]}
    saved_state, saved_observable = interface.merge_variable_inputs(
        given.save_variables,
        given.saved_state_indices,
        given.saved_observable_indices,
    )
    # Summaries follow the saved selection unless selected themselves.
    if (
        given.summarise_variables is None
        and given.summarised_state_indices is None
        and given.summarised_observable_indices is None
    ):
        summarised_state = saved_state.copy()
        summarised_observable = saved_observable.copy()
    else:
        summarised_state, summarised_observable = (
            interface.merge_variable_inputs(
                given.summarise_variables,
                given.summarised_state_indices,
                given.summarised_observable_indices,
            )
        )
    resolved.update(
        saved_state_indices=saved_state,
        saved_observable_indices=saved_observable,
        summarised_state_indices=summarised_state,
        summarised_observable_indices=summarised_observable,
    )
    return resolved


def output_flags(
    selection: Dict[str, Any],
    n_states: int,
    n_observables: int,
    precision: type,
) -> Tuple[bool, bool]:
    """Return whether the selection saves time-domain and summary outputs."""
    config = OutputConfig.from_loop_settings(
        output_types=selection["output_types"],
        precision=precision,
        saved_state_indices=selection["saved_state_indices"],
        saved_observable_indices=selection["saved_observable_indices"],
        summarised_state_indices=selection["summarised_state_indices"],
        summarised_observable_indices=selection[
            "summarised_observable_indices"
        ],
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
    duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the loop schedule and its flags for ``duration``."""
    # Time-domain outputs with no interval save the final state only.
    save_last = has_time_domain_outputs and save_every is None
    save_regularly = has_time_domain_outputs and save_every is not None
    if not has_summary_outputs:
        summarise_every = None
        sample_summaries_every = None
    else:
        # An unset window is the whole run.
        if summarise_every is None:
            summarise_every = duration
        if sample_summaries_every is None and summarise_every is not None:
            sample_summaries_every = (
                summarise_every / DEFAULT_SAMPLES_PER_SUMMARY
            )
    return {
        "save_every": save_every,
        "summarise_every": summarise_every,
        "sample_summaries_every": sample_summaries_every,
        "save_last": save_last,
        "save_regularly": save_regularly,
        "summarise_regularly": summarise_every is not None,
    }


def _newton_rtol_inverted(
    newton_rtol: Any, controller_rtol: Any, precision: type
) -> bool:
    """Whether the floored Newton rtol reaches the controller's."""
    floor = 4.0 * float(np_finfo(precision).eps)
    newton = asarray(newton_rtol, dtype=float)
    newton = newton.copy()
    newton[(newton > 0.0) & (newton < floor)] = floor
    controller = asarray(controller_rtol, dtype=float)
    newton = newton.reshape(-1, controller.size)
    return bool(((controller > 0.0) & (newton >= controller)).any())


def resolve(
    given: SolverSettings, system: Any, interface: Any
) -> Resolution:
    """Resolve every effective setting for ``system``.

    Raises
    ------
    ValueError
        Neumann preconditioner on a mass-matrix system, or an output
        index the system does not have.
    """
    notices = []
    resolved = {"precision": given.precision or system.precision}
    precision = resolved["precision"]
    sizes = system.sizes
    has_mass = system.mass is not None

    algorithm = given.algorithm if given.algorithm is not None else "euler"
    facts = algorithm_facts(algorithm, given.tableau)
    resolved["algorithm"] = algorithm

    controller, replaced = resolve_controller(given, facts)
    if replaced is not None:
        notices.append(
            f"Adaptive step controller '{replaced}' cannot be used with "
            f"fixed-step algorithm '{algorithm}'. The algorithm does not "
            "provide an error estimate required for adaptive stepping. "
            "Replacing with fixed-step controller."
        )
    resolved.update(controller)
    resolved.update(
        resolve_step_bounds(
            given.dt, given.dt_min, given.dt_max, resolved["is_adaptive"]
        )
    )
    resolved.update(resolve_step_defaults(given, facts, has_mass))

    resolved["atol"] = (
        given.atol if given.atol is not None else DEFAULT_TOLERANCE
    )
    resolved["rtol"] = (
        given.rtol if given.rtol is not None else DEFAULT_TOLERANCE
    )
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
            notices.append(
                "newton_rtol is at or above the step controller rtol: "
                "the requested rtol is below what the working precision "
                "resolves in the stage solves."
            )

    selection = resolve_output_selection(given, interface)
    resolved.update(selection)
    time_domain, summaries = output_flags(
        selection, int(sizes.states), int(sizes.observables), precision
    )
    resolved.update(
        resolve_loop_timing(
            given.save_every,
            given.summarise_every,
            given.sample_summaries_every,
            time_domain,
            summaries,
            given.duration,
        )
    )
    if summaries and given.summarise_every is None:
        notices.append(
            "Summary metrics were requested with no summarise_every "
            "timing; the summary window is the solve's duration, so a "
            "change of duration recompiles the kernel. Set "
            "summarise_every to avoid this."
        )

    # Every unroll flag is pushed so the children share one object.
    for key in ALL_UNROLL_PARAMETERS:
        if getattr(given, key) is None:
            resolved[key] = getattr(UnrollFlags(), key)

    effective = evolve(given, **resolved)
    return Resolution(effective=effective, notices=tuple(notices))


def resolve_performance(
    given: SolverSettings,
    effective: SolverSettings,
    step: Any,
    system: Any,
    previous: Optional[SolverSettings] = None,
    hardware: Any = None,
) -> SolverSettings:
    """Fill the unset unroll and placement values from the built step.

    With ``auto_performance`` off the values ``previous`` derived stay.
    """
    defaults = dict(step.performance_defaults)
    if effective.auto_performance is False:
        if previous is None:
            return effective
        carried = {
            key: getattr(previous, key)
            for key in set(defaults) | {"unroll_newton_exits"}
            if not given.given(key) and getattr(previous, key) is not None
        }
        return evolve(effective, **carried)
    # A Newton loop that overflows the instruction cache stays rolled.
    if step.is_implicit and step.newton_solves_per_step > 0:
        if hardware is None:
            hardware = device_hardware()
        unrolled = system.operation_count + step.step_operation_count
        capacity = hardware.instruction_cache_bytes // SASS_INSTRUCTION_BYTES
        defaults["unroll_newton_exits"] = (
            UnrollChoice.ROLLED if unrolled > capacity else UnrollChoice.FULL
        )
    updates = {
        key: value for key, value in defaults.items() if not given.given(key)
    }
    return evolve(effective, **updates)
