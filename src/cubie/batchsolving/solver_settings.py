"""Provided solver settings and the effective values resolved from them.

Published Classes
-----------------
:class:`SolverSettings`
    Every user-facing setting.
:class:`EffectiveSettings`
    Those plus the resolution-only names.
:class:`Resolution`
    A resolved record and its notices.

Module-Level Functions
----------------------
:func:`resolve`
    Effective settings for a system.
:func:`resolve_performance`
    Unroll and placement values from the built step.
"""

from math import sqrt
from typing import Any, Dict, List, Optional, Set, Tuple

from attrs import (
    NOTHING,
    asdict,
    define,
    evolve,
    field,
    fields,
    make_class,
)
from numpy import asarray, finfo as np_finfo

from cubie._utils import unpack_dict_values
from cubie.array_interpolator import ALL_INTERPOLATOR_PARAMETERS
from cubie.backend.utils import SASS_INSTRUCTION_BYTES, device_hardware
from cubie.batchsolving.BatchSolverConfig import ALL_KERNEL_PARAMETERS
from cubie.cuda_simsafe import (
    ALL_UNROLL_PARAMETERS,
    JITFlags,
    UnrollChoice,
)
from cubie.integrators.algorithms import algorithm_facts
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
    LINEAR_SOLVER_VARIANT_PARAMETERS,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    DAE_SOLVER_DEFAULTS,
)
from cubie.integrators.loops.ode_loop import ALL_LOOP_SETTINGS
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
from cubie.memory.mem_manager import ALL_MEMORY_MANAGER_PARAMETERS
from cubie.odesystems.ODEData import ALL_ODE_PARAMETERS
from cubie.outputhandling.output_config import OutputConfig
from cubie.outputhandling.output_functions import (
    ALL_OUTPUT_FUNCTION_PARAMETERS,
)


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

DERIVED_NAMES = frozenset(
    {"is_adaptive", "save_last", "save_regularly", "summarise_regularly"}
)
"""Names only resolution sets."""


def _child_names() -> Set[str]:
    """Names the kernel and the run's children take as settings."""
    return set().union(
        ALL_ALGORITHM_STEP_PARAMETERS,
        ALL_STEP_CONTROLLER_PARAMETERS,
        ALL_LOOP_SETTINGS,
        ALL_OUTPUT_FUNCTION_PARAMETERS,
        ALL_KERNEL_PARAMETERS,
        ALL_MEMORY_MANAGER_PARAMETERS,
        ALL_ODE_PARAMETERS,
        ALL_UNROLL_PARAMETERS,
        ALL_INTERPOLATOR_PARAMETERS,
        (fld.name for fld in fields(JITFlags)),
    )


def _local(**kwargs) -> Any:
    """A setting the Solver keeps to itself, never pushed to the kernel."""
    return field(default=None, metadata={"push": False}, **kwargs)


@define(eq=False)
class _SolverOwnSettings:
    """The settings the Solver itself reads; ``None`` is not provided."""

    tableau: Any = field(default=None)
    system_constants: Optional[Dict[str, float]] = _local()
    duration: Optional[float] = _local()
    save_variables: Optional[List[str]] = _local()
    summarise_variables: Optional[List[str]] = _local()

    def __attrs_post_init__(self) -> None:
        """Reject gains given together with a filter."""
        gains = [name for name in CONTROLLER_GAIN_NAMES if self.given(name)]
        if gains and self.filter_coefficients is not None:
            raise ValueError(
                f"filter_coefficients cannot be combined with {gains}; "
                "give one or the other."
            )

    @classmethod
    def names(cls) -> Tuple[str, ...]:
        """Return every setting name."""
        return tuple(fld.name for fld in fields(cls))

    @classmethod
    def _flatten(
        cls, kwargs: Dict[str, Any], strict: bool = True
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
        names = set(cls.names())
        unknown = set(flat) - names
        if unknown and strict:
            raise KeyError(f"Unrecognized keyword arguments: {unknown}")
        flat = {key: value for key, value in flat.items() if key in names}
        return flat, recognised | set(flat)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "_SolverOwnSettings":
        """Record ``kwargs`` as provided; grouped dicts are flattened."""
        flat, _ = cls._flatten(kwargs)
        return cls(**flat)

    def updated(
        self, updates: Dict[str, Any], strict: bool = True
    ) -> Tuple["_SolverOwnSettings", Set[str]]:
        """Return a copy with ``updates`` recorded (``None`` unsets) and
        the names taken; unknown names raise unless not ``strict``."""
        flat, recognised = self._flatten(updates, strict=strict)
        return evolve(self, **flat), recognised

    def update(self, updates: Dict[str, Any]) -> "_SolverOwnSettings":
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
        """The provided settings the kernel takes."""
        return {
            fld.name: getattr(self, fld.name)
            for fld in fields(type(self))
            if fld.metadata.get("push", True)
            and getattr(self, fld.name) is not None
        }


SolverSettings = make_class(
    "SolverSettings",
    {
        name: field(default=None)
        for name in sorted(_child_names() - DERIVED_NAMES)
    },
    bases=(_SolverOwnSettings,),
    eq=False,
    slots=True,
)
SolverSettings.__doc__ = (
    "Every user-facing solver setting; ``None`` is not provided."
)

EffectiveSettings = make_class(
    "EffectiveSettings",
    {name: field(default=None) for name in sorted(DERIVED_NAMES)},
    bases=(SolverSettings,),
    eq=False,
    slots=True,
)
EffectiveSettings.__doc__ = "The settings in effect, provided and derived."


def settings_differ(old: Any, new: Any) -> bool:
    """Whether any field of two records differs in value."""
    for name in type(new).names():
        before, after = getattr(old, name, None), getattr(new, name)
        if before is after:
            continue
        equal = before == after
        if not isinstance(equal, bool):
            equal = bool(asarray(equal).all()) and (
                asarray(before).shape == asarray(after).shape
            )
        if not equal:
            return True
    return False


@define
class Resolution:
    """A resolved :class:`EffectiveSettings` and the notices to warn about."""

    effective: Any
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
    given: Any, facts: Any
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
        # The family's other controller keys fill the unset ones.
        for key, value in defaults.items():
            if (
                key in ALL_STEP_CONTROLLER_PARAMETERS
                and key not in CONTROLLER_GAIN_NAMES
                and key != "step_controller"
            ):
                provided = getattr(given, key)
                resolved[key] = value if provided is None else provided
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
    given: Any, facts: Any, has_mass: bool
) -> Dict[str, Any]:
    """Fill the unset step keys from the family, tableau and DAE tables."""
    defaults = facts.defaults.settings
    resolved = {}
    step_keys = {
        key
        for key in (*defaults, *DAE_SOLVER_DEFAULTS)
        if key in ALL_ALGORITHM_STEP_PARAMETERS
        and key not in LINEAR_SOLVER_VARIANT_PARAMETERS
    }
    for key in sorted(step_keys):
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
    given: Any,
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


def resolve_output_selection(given: Any, interface: Any) -> Dict[str, Any]:
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


def resolve(given: Any, system: Any, interface: Any) -> Resolution:
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

    effective = EffectiveSettings(
        **{**asdict(given, recurse=False), **resolved}
    )
    return Resolution(effective=effective, notices=tuple(notices))


def resolve_performance(
    given: Any,
    effective: Any,
    step: Any,
    system: Any,
    previous: Any = None,
    hardware: Any = None,
) -> Any:
    """Fill the unset unroll and placement values from the built step;
    with ``auto_performance`` off the values in ``previous`` stay."""
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
