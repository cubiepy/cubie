"""Coordination layer for a single CUDA-based ODE integration.

Published Classes
-----------------
:class:`SingleIntegratorRunCache`
    Attrs cache container for the compiled integrator device function.

:class:`SingleIntegratorRunCore`
    CUDAFactory that owns and wires the algorithm step, step controller,
    output functions, and IVP loop into a single compilable unit.

See Also
--------
:class:`~cubie.integrators.SingleIntegratorRun.SingleIntegratorRun`
    Property-aggregation subclass exposing read-only access.
:class:`~cubie.integrators.loops.ode_loop.IVPLoop`
    Loop factory owned by this class.
:class:`~cubie.outputhandling.output_functions.OutputFunctions`
    Output function factory owned by this class.
:class:`~cubie.integrators.IntegratorRunSettings.IntegratorRunSettings`
    Compile settings container used by this class.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple
from warnings import warn

from attrs import define, field, fields
from numpy import asarray, finfo as np_finfo

from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie._utils import unpack_dict_values
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import (
    ALL_UNROLL_PARAMETERS,
    UnrollFlag,
    unroll_flag_converter,
)
from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings
from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
    LINEAR_SOLVER_VARIANT_PARAMETERS,
    PerformanceSettings,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    DAE_SOLVER_DEFAULTS,
)
from cubie.integrators.dae_initialiser import DAEInitialiser
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.odesystems.solver_helpers import OperationCounts
from cubie.outputhandling import OutputArrayHeights, OutputCompileFlags
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.integrators.step_control import (
    CONTROLLER_GAIN_PARAMETERS,
    get_controller,
    promoted_gain_controller,
)


if TYPE_CHECKING:  # pragma: no cover - imported for static typing only
    from cubie.odesystems.baseODE import BaseODE


def warn_on_newton_rtol_inversion(newton_rtol, controller_rtol) -> None:
    """Warn when the Newton rtol reaches the step controller's rtol."""
    controller = asarray(controller_rtol)
    newton = asarray(newton_rtol).reshape(-1, controller.size)
    inverted = (controller > 0.0) & (newton >= controller)
    if inverted.any():
        warn(
            "newton_rtol is at or above the step controller rtol: the "
            "requested rtol is below what the working precision "
            "resolves in the stage solves.",
            UserWarning,
            stacklevel=2,
        )


@define
class SingleIntegratorRunCache(CUDADispatcherCache):
    """Cache for SingleIntegratorRunCore device function.

    Attributes
    ----------
    loop_fn
        Compiled CUDA loop callable ready for execution on device.
    compile_flags, threads_per_step, output_array_heights, is_implicit
        The children's flags and sizes.
    shared_memory_elements, persistent_local_elements
        The loop's buffer sizes.
    operation_counts, performance_defaults
        The system's operator counts and the step's defaults.
    newton_solves_per_step, step_operation_count, unroll_newton_exits
        The step's Newton solves, unrolled operation count and loop flag.
    """
    loop_fn: Callable = field(eq=False)
    compile_flags: Optional[OutputCompileFlags] = field(default=None)
    threads_per_step: int = field(default=1)
    shared_memory_elements: int = field(default=0)
    persistent_local_elements: int = field(default=0)
    output_array_heights: Optional[OutputArrayHeights] = field(default=None)
    operation_counts: OperationCounts = field(factory=OperationCounts)
    performance_defaults: PerformanceSettings = field(
        factory=PerformanceSettings
    )
    is_implicit: bool = field(default=False)
    newton_solves_per_step: int = field(default=0)
    step_operation_count: int = field(default=0)
    unroll_newton_exits: UnrollFlag = field(
        default=(True, None), converter=unroll_flag_converter
    )


class SingleIntegratorRunCore(CUDAFactory):
    """Coordinate a single ODE integration loop and its dependencies.

    Parameters
    ----------
    system
        ODE system whose device functions drive the integration.
    loop_settings
        Mapping of compile-critical loop configuration forwarded to the
        :class:`cubie.integrators.loops.ode_loop.IVPLoop`.  Recognised
        keys include ``"save_every"`` and ``"summarise_every"``.  When
        ``None`` the loop falls back to built-in defaults.
    output_settings
        Mapping forwarded to :class:`cubie.outputhandling.output_functions.
        OutputFunctions`.  Recognised keys include ``"output_types"`` and
        the saved or summarised selector fields:
        ``"saved_state_indices"``, ``"saved_observable_indices"``,
        ``"summarised_state_indices"``, and
        ``"summarised_observable_indices"``.
    drivers_fn
        Optional device function that interpolates driver inputs for use
        by step algorithms.
    driver_derivative_fn
        Optional device function providing the time derivative of the
        driver signal, used by Rosenbrock-W methods.
    algorithm_settings
        Mapping forwarded to
        :func:`cubie.integrators.algorithms.get_algorithm_step`
        containing ``"algorithm"`` and any additional parameters required
        by the selected step factory.  When ``None`` the algorithm
        defaults are used.
    step_control_settings
        Mapping merged with the algorithm defaults before calling
        :func:`cubie.integrators.step_control.get_controller`.  Include
        ``"step_controller"`` to select a controller family and provide
        bounds such as ``"dt_min"`` and ``"dt_max"`` when configuring
        adaptive controllers.  Supported identifiers include ``"fixed"``,
        ``"i"``, ``"pi"``, ``"pid"``, and ``"gustafsson"``.  When
        ``None`` the algorithm defaults are used.
    """

    settings_keys = frozenset({"algorithm", "step_controller"})

    _INNER_TOLERANCE_KEYS = (
        "krylov_atol",
        "krylov_rtol",
        "krylov_residual_reduction",
        "newton_atol",
        "newton_rtol",
    )

    # Child keys this run writes from the system, step or controller.
    _INJECTED_KEYS = frozenset(
        {
            "precision",
            "n_states",
            "n_drivers",
            "mass_flags",
            "algorithm_order",
            "is_adaptive",
            "save_last",
            "save_regularly",
            "summarise_regularly",
        }
    )
    _TIMING_KEYS = ("save_every", "summarise_every", "sample_summaries_every")
    # Summary samples per window when the schedule derives from duration.
    _DERIVED_SAMPLES_PER_SUMMARY = 100

    def __init__(
        self,
        system: "BaseODE",
        loop_settings: Optional[Dict[str, Any]] = None,
        output_settings: Optional[Dict[str, Any]] = None,
        drivers_fn: Optional[Callable] = None,
        driver_derivative_fn: Optional[Callable] = None,
        algorithm_settings: Optional[Dict[str, Any]] = None,
        step_control_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        step_control_settings = dict(step_control_settings or {})
        algorithm_settings = dict(algorithm_settings or {})
        output_settings = dict(output_settings or {})
        loop_settings = dict(loop_settings or {})

        self._user_given_inner_tols = set()
        self._user_given_keys = set()
        self._user_timing = dict.fromkeys(self._TIMING_KEYS)
        self._derived_duration = None
        self.is_duration_dependent = False
        self._record_givenness({**algorithm_settings, **loop_settings})

        self._system = system
        products = system.products
        precision = products["precision"]

        output_settings.pop("precision", None)
        self._output_functions = OutputFunctions(
            n_states=products["n_states"],
            n_observables=products["n_observables"],
            precision=precision,
            **output_settings,
        )

        algorithm_settings.update(
            n_states=products["n_states"],
            n_drivers=products["n_drivers"],
            dxdt_fn=products["dxdt_fn"],
            observables_fn=products["observables_fn"],
            get_solver_helper_fn=products["get_solver_helper_fn"],
            drivers_fn=drivers_fn,
            driver_derivative_fn=driver_derivative_fn,
        )
        self._algo_step = get_algorithm_step(
            precision=precision,
            settings=algorithm_settings,
        )
        self._check_algorithm_consumes_mass(algorithm_settings["algorithm"])
        self._apply_algorithm_step_defaults()
        self._apply_dae_linear_solve_defaults()

        controller_name = self._requested_controller_name(
            step_control_settings, given_algorithm=True
        )
        controller_settings = self._family_controller_defaults(
            controller_name, step_control_settings
        )
        controller_settings.update(step_control_settings)
        controller_settings.update(
            step_controller=controller_name,
            n_states=products["n_states"],
            algorithm_order=self._algo_step.algorithm_order,
            mass_flags=products["mass_flags"],
        )
        self._step_controller = get_controller(
            precision=precision,
            settings=controller_settings,
        )
        self._algo_step.update(
            {"is_adaptive": self._step_controller.is_adaptive}, silent=True
        )
        self._apply_inner_tolerance_defaults()

        self.setup_compile_settings(
            IntegratorRunSettings(
                precision=precision,
                algorithm=algorithm_settings["algorithm"],
                step_controller=controller_name,
            )
        )

        outputs = self._output_functions.products
        step = self._algo_step.products
        controller = self._step_controller.products
        loop_settings.update(
            precision=precision,
            n_states=products["n_states"],
            n_parameters=products["n_parameters"],
            n_observables=products["n_observables"],
            n_drivers=products["n_drivers"],
            compile_flags=outputs["compile_flags"],
            n_counters=outputs["n_counters"],
            state_summaries_buffer_height=(
                outputs["state_summaries_buffer_height"]
            ),
            observable_summaries_buffer_height=(
                outputs["observable_summaries_buffer_height"]
            ),
            n_error=step["n_error"],
            drivers_fn=drivers_fn,
            dt=controller["dt"],
            dt_min=controller["dt_min"],
            dt_max=controller["dt_max"],
            is_adaptive=controller["is_adaptive"],
        )
        self._loop = IVPLoop(**loop_settings)

        init_settings = self._algo_step.settings_dict
        init_settings["dae_initialisation"] = algorithm_settings.get(
            "dae_initialisation"
        )
        init_settings["mass_flags"] = products["mass_flags"]
        init_settings["get_solver_helper_fn"] = products[
            "get_solver_helper_fn"
        ]
        self._dae_initialiser = DAEInitialiser(**init_settings)

        self._distribute({})
        self._warn_if_summary_timing_derived()

    def _record_givenness(self, updates: Dict[str, Any]) -> None:
        """Record the tolerance, step and timing keys the user gave.

        The family and DAE defaults never overwrite a step key the user
        gave; a parent's derived settings arrive as a
        ``performance_settings`` object and are never recorded.
        """
        for key in self._INNER_TOLERANCE_KEYS:
            if updates.get(key) is not None:
                self._user_given_inner_tols.add(key)
        self._user_given_keys |= {
            key
            for key in set(updates) & ALL_ALGORITHM_STEP_PARAMETERS
            if updates[key] is not None
        }
        for key in self._TIMING_KEYS:
            if key in updates:
                self._user_timing[key] = updates[key]

    def _loop_timing(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Return the loop schedule; ``duration`` sets a derived window."""
        has_time_domain_outputs = self.time_domain_outputs_requested
        has_summary_outputs = self.summary_outputs_requested
        save_every = self._user_timing["save_every"]
        summarise_every = self._user_timing["summarise_every"]
        sample_summaries_every = self._user_timing["sample_summaries_every"]

        save_last = has_time_domain_outputs and save_every is None
        self.is_duration_dependent = False
        if has_summary_outputs:
            if summarise_every is None:
                self.is_duration_dependent = True
                if updates.get("duration") is not None:
                    self._derived_duration = float(updates["duration"])
                if self._derived_duration is not None:
                    summarise_every = self._derived_duration
                    sample_summaries_every = (
                        summarise_every / self._DERIVED_SAMPLES_PER_SUMMARY
                    )
            elif sample_summaries_every is None:
                sample_summaries_every = summarise_every / 10.0
        else:
            summarise_every = None
            sample_summaries_every = None

        return dict(
            save_every=save_every,
            summarise_every=summarise_every,
            sample_summaries_every=sample_summaries_every,
            save_last=save_last,
            save_regularly=save_every is not None and has_time_domain_outputs,
            summarise_regularly=(
                summarise_every is not None and has_summary_outputs
            ),
        )

    def _warn_if_summary_timing_derived(self):
        if self.is_duration_dependent:
            warn(
                "Summary metrics were requested with no "
                "summarise_every or sample_summaries_every timing. "
                "Sample_summaries_every was set to duration / 100 by "
                "default. If duration changes, the kernel will need "
                "to recompile, which will cause a slow integration "
                "(once). Set timing parameters explicitly to avoid "
                "this.",
                UserWarning,
                stacklevel=3,
            )

    def _apply_inner_tolerance_defaults(self) -> set:
        """Derive unset inner-solver tolerances from the controller.

        Unset ``newton_atol``/``newton_rtol`` default to the
        controller's ``atol``/``rtol`` divided by ten, so every stage
        solve converges tighter than the error estimate it feeds.
        Unset ``krylov_atol``/``krylov_rtol`` default to the
        controller's ``atol``/``rtol`` directly: they weight the
        linear stopping norm, placing its absolute floor at the step
        tolerance envelope.  Unset ``krylov_residual_reduction``
        defaults to the adaptive controller's tightest ``rtol`` entry,
        divided by one hundred for linearly-implicit (``is_linear``)
        steps; non-adaptive runs default to machine epsilon, leaving
        the floor governing.  Values the user set explicitly (tracked in
        ``_user_given_inner_tols``) are preserved.  Solver-norm
        tolerances take the controller's per-state length, coupled
        FIRK solves included, so a non-uniform vector carries through
        unchanged.

        Every controller carries ``atol``/``rtol`` — fixed-step
        included — so the defaults apply whenever the algorithm is
        implicit (it then owns inner solvers).

        Returns
        -------
        set of str
            The inner-tolerance keys forwarded to the algorithm step;
            keys its solvers do not use are ignored there.
        """
        if not self._algo_step.is_implicit:
            return set()

        controller_atol = self._step_controller.atol
        controller_rtol = self._step_controller.rtol
        derived_source = {
            "krylov_atol": controller_atol.copy(),
            "krylov_rtol": controller_rtol.copy(),
            "newton_atol": controller_atol / 10.0,
            "newton_rtol": controller_rtol / 10.0,
        }
        # Non-adaptive runs and pure-absolute controllers (rtol of
        # zero) offer no relative target; an epsilon reduction leaves
        # the floor governing.
        controller_rtol_floor = float(controller_rtol.min())
        if self._step_controller.is_adaptive and controller_rtol_floor > 0.0:
            if self._algo_step.is_linear:
                controller_rtol_floor *= 0.01
            derived_source["krylov_residual_reduction"] = (
                controller_rtol_floor
            )
        else:
            derived_source["krylov_residual_reduction"] = float(
                np_finfo(self._algo_step.precision).eps
            )
        derived = {
            key: value
            for key, value in derived_source.items()
            if key not in self._user_given_inner_tols
        }
        if derived:
            self._algo_step.update(derived, silent=True)
        if not self._algo_step.is_linear:
            warn_on_newton_rtol_inversion(
                self._algo_step.solver.rtol,
                self._step_controller.rtol,
            )
        return set(derived)

    @property
    def n_error(self) -> int:
        """Return the length of the shared error buffer."""

        if self._algo_step.uses_error:
            return int(self._system.sizes.states)
        return 0

    @property
    def device_function(self):
        """Return the compiled CUDA solver kernel.

        Returns
        -------
        callable
            Compiled CUDA device function.
        """
        return self.get_cached_output("loop_fn")

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> set[str]:
        """Update parameters across all components.

        Parameters
        ----------
        updates_dict
            Dictionary of parameters to update. A ``performance_settings``
            entry carries a :class:`PerformanceSettings` a parent derived;
            its fields reach the children without being recorded as
            user-given.
        silent
            If ``True``, suppress errors about unrecognised parameters.
        **kwargs
            Additional updates provided as keyword arguments.

        Returns
        -------
        set[str]
            Names of parameters that were recognised and applied.

        Raises
        ------
        KeyError
            Raised when unrecognised parameters remain and ``silent`` is
            ``False``.

        Notes
        -----
        Children update in order, each receiving the dict plus the
        earlier children's products; a new ``algorithm`` or
        ``step_controller`` swaps the child first.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        updates_dict, unpacked_keys = unpack_dict_values(updates_dict)
        user_keys = set(updates_dict)
        self._record_givenness(updates_dict)

        recognized = self._distribute(updates_dict)
        if user_keys & (set(self._TIMING_KEYS) | {"output_types", "duration"}):
            self._warn_if_summary_timing_derived()

        unrecognized = user_keys - recognized
        if unrecognized and not silent:
            raise KeyError(f"Unrecognized parameters: {unrecognized}")
        return recognized | unpacked_keys

    def _distribute(self, updates: Dict[str, Any]) -> set[str]:
        """Update every child in order, merging each child's products."""
        recognized = set()
        derived = updates.pop("performance_settings", None)
        if derived is not None:
            updates.update(derived.as_updates())
            recognized.add("performance_settings")
        if "duration" in updates:
            recognized.add("duration")
        user_keys = set(updates)

        recognized |= self._system.update(updates, silent=True)
        updates.update(self._system.products)
        recognized |= self._output_functions.update(updates, silent=True)
        updates.update(self._loop_timing(updates))
        recognized |= self._output_functions.update(updates, silent=True)
        updates.update(self._output_functions.products)

        switched = self._switch_algos(updates)
        recognized |= self._algo_step.update(updates, silent=True)
        switched |= self._switch_controllers(updates)
        recognized |= switched

        recognized |= self._step_controller.update(updates, silent=True)
        updates.update(self._step_controller.products)
        recognized |= self._algo_step.update(updates, silent=True)
        recognized |= self._apply_algorithm_step_defaults()
        recognized |= self._apply_dae_linear_solve_defaults()
        if switched or user_keys & {"atol", "rtol"}:
            recognized |= self._apply_inner_tolerance_defaults()
        updates.update(self._algo_step.products)
        recognized |= self._step_controller.update(updates, silent=True)
        updates.update(self._step_controller.products)

        recognized |= self._dae_initialiser.update(updates, silent=True)
        updates.update(self._dae_initialiser.products)
        self._register_loop_children()
        recognized |= self._loop.update(updates, silent=True)
        updates.update(self._loop.products)
        recognized |= self.update_compile_settings(updates, silent=True)
        return recognized

    def _register_loop_children(self) -> None:
        """Register the step, controller and initialiser under the loop."""
        buffer_registry.register_child(
            self._loop, self._algo_step, name="algorithm"
        )
        buffer_registry.register_child(
            self._loop, self._step_controller, name="controller"
        )
        buffer_registry.register_child(
            self._loop,
            self._dae_initialiser,
            name="initialiser",
            aliases="algorithm_shared",
        )

    def _switch_algos(self, updates_dict):
        """Swap the step on a new ``algorithm``; merge the family defaults."""
        if "algorithm" not in updates_dict:
            return set()
        precision = updates_dict.get("precision", self.precision)

        new_algo = updates_dict.get("algorithm").lower()
        if new_algo != self.compile_settings.algorithm:
            buffer_registry.clear_parent(self._algo_step)
            old_config = self._algo_step.compile_settings
            old_settings = self._algo_step.settings_dict
            old_settings["algorithm"] = new_algo
            # The old step's device functions carry over.
            old_settings.update(
                {
                    fld.name: getattr(old_config, fld.name)
                    for fld in fields(type(old_config))
                    if fld.metadata.get("device_function")
                    and fld.name in ALL_ALGORITHM_STEP_PARAMETERS
                }
            )
            self._algo_step = get_algorithm_step(
                precision=precision,
                settings=old_settings,
            )
            self.update_compile_settings(algorithm=new_algo)
            self._check_algorithm_consumes_mass(new_algo)
        updates_dict["algorithm"] = new_algo

        controller_name = self._requested_controller_name(
            updates_dict, given_algorithm=True
        )
        defaults = self._family_controller_defaults(
            controller_name, updates_dict
        )
        for key, value in defaults.items():
            if key not in updates_dict:
                updates_dict[key] = value
        updates_dict["step_controller"] = controller_name
        return {"algorithm"}

    def _family_controller_defaults(
        self, controller_name: str, settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return the family's controller defaults minus dropped gains."""
        defaults = self._algo_step.controller_default_settings
        skip_gains = (
            controller_name != defaults["step_controller"]
            or settings.get("filter_coefficients") is not None
        )
        if skip_gains:
            for key in CONTROLLER_GAIN_PARAMETERS:
                defaults.pop(key, None)
        return defaults

    def _requested_controller_name(
        self, settings: Dict[str, Any], given_algorithm: bool = False
    ) -> str:
        """Return the named controller, else the base promoted for gains."""
        requested = settings.get("step_controller")
        if requested is not None:
            return requested.lower()
        if given_algorithm or self._compile_settings is None:
            base = self._algo_step.controller_default_settings[
                "step_controller"
            ]
        else:
            base = self.compile_settings.step_controller
        promoted = promoted_gain_controller(base, settings)
        return promoted or base

    def _compatible_controller_name(
        self, controller_name: str, algorithm_name: str
    ) -> str:
        """Return ``fixed`` when the step has no error estimate."""
        if (
            controller_name == "fixed"
            or self._algo_step.has_error_estimate
        ):
            return controller_name
        warn(
            f"Adaptive step controller '{controller_name}' cannot be "
            f"used with fixed-step algorithm '{algorithm_name}'. "
            f"The algorithm does not provide an error estimate "
            f"required for adaptive stepping. "
            f"Replacing with fixed-step controller.",
            UserWarning,
            stacklevel=4,
        )
        return "fixed"

    def _check_algorithm_consumes_mass(self, algorithm_name: str) -> None:
        """Reject explicit algorithms on systems with a mass matrix.

        Parameters
        ----------
        algorithm_name
            Name of the algorithm being installed, used in the error
            message.

        Raises
        ------
        ValueError
            If the system defines a mass matrix and the current
            algorithm is not implicit. Explicit steps cannot consume
            a mass matrix and would integrate algebraic constraint
            residuals as derivatives.
        """
        if self._system.mass is None or self._algo_step.is_implicit:
            return
        raise ValueError(
            "The system defines a mass matrix and requires an "
            f"implicit algorithm; '{algorithm_name}' does not "
            "consume a mass matrix and would integrate the "
            "constraint residuals as derivatives."
        )

    def _apply_algorithm_step_defaults(self) -> set:
        """Apply family and tableau step defaults to unset keys.

        Newton-variant defaults
        (:data:`LINEAR_SOLVER_VARIANT_PARAMETERS`) apply when the
        linear solver in use is the family's default one and drop
        when the user picks a different ``linear_correction_type``.

        Returns
        -------
        set of str
            The default keys forwarded to the algorithm step.
        """
        defaults = self._algo_step.step_default_settings
        user_given = self._user_given_keys
        if (
            "linear_correction_type" in user_given
            and self._algo_step.is_implicit
            and self._algo_step.linear_correction_type
            != defaults.get("linear_correction_type")
        ):
            for key in LINEAR_SOLVER_VARIANT_PARAMETERS:
                defaults.pop(key, None)
        updates = {
            key: value
            for key, value in defaults.items()
            if key not in user_given
        }
        if not updates:
            return set()
        return self._algo_step.update(updates, silent=True)

    def _apply_dae_linear_solve_defaults(self) -> set:
        """Fill unset linear solve keys from ``DAE_SOLVER_DEFAULTS``."""
        if self._system.mass is None or not self._algo_step.is_implicit:
            return set()
        user_given = self._user_given_keys
        updates = {
            key: value
            for key, value in DAE_SOLVER_DEFAULTS.items()
            if key not in user_given
        }
        effective = updates.get(
            "preconditioner_type", self._algo_step.preconditioner_type
        )
        if effective == "neumann":
            raise ValueError(
                "Neumann preconditioners assume an identity mass "
                "matrix and cannot precondition a system with torn "
                "algebraic rows. Use preconditioner_type='jacobi'."
            )
        if not updates:
            return set()
        return self._algo_step.update(updates)

    def _switch_controllers(self, updates_dict):
        """Resolve the controller name and swap the controller on change."""
        given = "step_controller" in updates_dict
        precision = updates_dict.get("precision", self.precision)
        new_controller = self._compatible_controller_name(
            self._requested_controller_name(updates_dict),
            self.compile_settings.algorithm,
        )

        if new_controller != self.compile_settings.step_controller:
            buffer_registry.clear_parent(self._step_controller)
            old_settings = self._step_controller.settings_dict
            # A new controller starts from its own gain defaults.
            for key in CONTROLLER_GAIN_PARAMETERS:
                old_settings.pop(key, None)
            old_settings["step_controller"] = new_controller
            old_settings["dt"] = self._step_controller.dt
            old_settings["algorithm_order"] = self._algo_step.algorithm_order
            self._step_controller = get_controller(
                precision=precision,
                settings=old_settings,
                warn_on_unused=False,
            )
            self.update_compile_settings(step_controller=new_controller)
            given = True
        updates_dict["step_controller"] = new_controller
        return {"step_controller"} if given else set()

    def build(self) -> SingleIntegratorRunCache:
        """Return the captured loop function with the children's sizes."""
        loop = self._loop.products
        step = self._algo_step.products
        outputs = self._output_functions.products
        return SingleIntegratorRunCache(
            loop_fn=self.compile_settings.loop_fn,
            compile_flags=outputs["compile_flags"],
            threads_per_step=step["threads_per_step"],
            shared_memory_elements=loop["shared_memory_elements"],
            persistent_local_elements=loop["persistent_local_elements"],
            output_array_heights=outputs["output_array_heights"],
            operation_counts=self._system.products["operation_counts"],
            performance_defaults=step["performance_defaults"],
            is_implicit=step["is_implicit"],
            newton_solves_per_step=step["newton_solves_per_step"],
            step_operation_count=step["step_operation_count"],
            unroll_newton_exits=step["unroll_newton_exits"],
        )

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Return the keys rebuilding this run; derived ones only as given."""
        settings = super().settings_dict
        # The run writes the loop's dt and schedule itself.
        loop_settings = self._loop.settings_dict
        for key in ("dt", *self._TIMING_KEYS):
            loop_settings.pop(key, None)
        settings.update(loop_settings)
        for child in (
            self._output_functions, self._step_controller, self._algo_step
        ):
            settings.update(child.settings_dict)
        for key in self._INJECTED_KEYS:
            settings.pop(key, None)
        settings.pop("sample_summaries_every", None)
        settings.update(
            {
                key: value
                for key, value in self._user_timing.items()
                if value is not None
            }
        )
        for key in self._INNER_TOLERANCE_KEYS:
            if key not in self._user_given_inner_tols:
                settings.pop(key, None)
        # The flags every child compiled with, as one object.
        settings["unroll"] = self.compile_settings.unroll
        return settings

    def grouped_settings(self) -> Dict[str, Dict[str, Any]]:
        """Return ``settings_dict`` split into the constructor's groups."""
        settings = self.settings_dict
        groups = {
            "loop_settings": self._loop.settings_keys,
            "output_settings": self._output_functions.settings_keys,
            "step_control_settings": self._step_controller.settings_keys,
            "algorithm_settings": self._algo_step.settings_keys,
            "unroll_settings": ALL_UNROLL_PARAMETERS | {"unroll"},
        }
        grouped = {
            name: {
                key: value
                for key, value in settings.items()
                if key in keys
            }
            for name, keys in groups.items()
        }
        return grouped

    def copy(self) -> "SingleIntegratorRunCore":
        """Return a new run with these settings on a copy of the system."""
        step_config = self._algo_step.compile_settings
        grouped = self.grouped_settings()
        unroll_settings = grouped.pop("unroll_settings")
        twin = type(self)(
            self._system.copy(),
            drivers_fn=step_config.drivers_fn,
            driver_derivative_fn=getattr(
                step_config, "driver_derivative_fn", None
            ),
            **grouped,
        )
        if unroll_settings:
            twin.update(unroll_settings, silent=True)
        return twin

    @property
    def algorithm_candidates(self) -> Tuple[Dict[str, Any], ...]:
        """Return the step's candidate settings for ``Solver.optimize``."""
        return self._algo_step.optimisation_candidates

    @property
    def time_domain_outputs_requested(self) -> bool:
        """Return True if time-domain outputs are requested in output_types."""
        return self._output_functions.has_time_domain_outputs

    @property
    def summary_outputs_requested(self) -> bool:
        """Return True if summary outputs are requested in output_types."""
        return self._output_functions.has_summary_outputs

    @property
    def has_time_domain_outputs(self) -> bool:
        """Return True if time-domain outputs will be produced by the loop"""
        has_time_domain_types = self.time_domain_outputs_requested
        has_save_timing = (
            self._loop.compile_settings._save_every is not None
            or self._loop.compile_settings.save_last
        )
        return has_time_domain_types and has_save_timing

    @property
    def has_summary_outputs(self) -> bool:
        """Return True if summary outputs will be produced by the loop"""
        has_summaries_types = self.summary_outputs_requested
        has_summarise_timing = (
            self._loop.compile_settings._summarise_every is not None
        )
        return has_summaries_types and has_summarise_timing
