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

from attrs import define, field
from numpy import asarray

from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie._utils import product_field, unpack_dict_values
from cubie.buffer_registry import buffer_registry
from cubie.integrators.IntegratorRunSettings import (
    ALL_RUN_PARAMETERS,
    IntegratorRunSettings,
    RUN_CONTROLLER_PARAMETERS,
    RUN_TIMING_PARAMETERS,
)
from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
    BaseAlgorithmStep,
    KERNEL_RESOLVED_STEP_PARAMETERS,
    RUN_RESOLVED_STEP_PARAMETERS,
)
from cubie.integrators.dae_initialiser import DAEInitialiser
from cubie.integrators.loops.ode_loop import ALL_LOOP_SETTINGS, IVPLoop
from cubie.outputhandling import OutputCompileFlags
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.integrators.step_control import (
    _CONTROLLER_REGISTRY,
    get_controller,
)
from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
    CONTROLLER_GAIN_NAMES,
    BaseStepController,
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
    compile_flags, n_states, threads_per_step, is_implicit,
    algorithm_family, newton_solves_per_step, step_operation_count,
    uses_direct_solver, stage_count, accumulates_output, local_elements
        Product fields the batch kernel reads.
    """

    loop_fn: Callable = field(eq=False)
    compile_flags: Optional[OutputCompileFlags] = product_field()
    n_states: int = product_field()
    threads_per_step: int = product_field()
    is_implicit: bool = product_field()
    algorithm_family: str = product_field()
    newton_solves_per_step: int = product_field()
    step_operation_count: int = product_field()
    uses_direct_solver: bool = product_field()
    stage_count: int = product_field()
    accumulates_output: bool = product_field()
    local_elements: int = product_field()


# Keys every child takes from the parent unchanged.
BROADCAST_KEYS = frozenset({"precision", "unroll", "jit_flags", "lineinfo"})

_STEP_KEYS = (
    frozenset(ALL_ALGORITHM_STEP_PARAMETERS) | {"tableau"}
) - frozenset(BaseAlgorithmStep.injected_keys)
_CONTROLLER_KEYS = frozenset(ALL_STEP_CONTROLLER_PARAMETERS) - frozenset(
    BaseStepController.injected_keys
)
_LOOP_KEYS = frozenset(ALL_LOOP_SETTINGS) - frozenset(IVPLoop.injected_keys)


def _controller_name(controller: BaseStepController) -> str:
    """Return the registry name of ``controller``'s class."""
    for name, cls in _CONTROLLER_REGISTRY.items():
        if type(controller) is cls:
            return name
    raise ValueError(f"{type(controller).__name__} is not registered.")


class SingleIntegratorRunCore(CUDAFactory):
    """Coordinate a single ODE integration loop and its dependencies.

    Owns the keys in ``ALL_RUN_PARAMETERS``, resolves them from the
    children's products and writes the results into the children.

    Parameters
    ----------
    system
        ODE system whose device functions drive the integration.
    loop_settings
        Mapping forwarded to the loop; the schedule keys are the run's.
    output_settings
        Mapping forwarded to :class:`cubie.outputhandling.output_functions.
        OutputFunctions`.
    drivers_fn
        Optional device function that interpolates driver inputs for use
        by step algorithms.
    driver_derivative_fn
        Optional device function providing the time derivative of the
        driver signal, used by Rosenbrock-W methods.
    algorithm_settings
        ``"algorithm"`` and the step's keys; tolerances and
        family-defaulted keys are the run's.
    step_control_settings
        Controller keys; ``step_controller`` and family-defaulted keys
        are the run's.
    """

    settings_keys = frozenset(ALL_RUN_PARAMETERS)
    injected_keys = frozenset({"precision"})

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

        self._system = system
        self._drivers_fn = drivers_fn
        self._driver_derivative_fn = driver_derivative_fn
        products = system.products
        precision = products["precision"]

        output_settings.pop("precision", None)
        self._output_functions = OutputFunctions(
            n_states=products["n_states"],
            n_observables=products["n_observables"],
            precision=precision,
            **output_settings,
        )

        if algorithm_settings.get("algorithm") is None:
            raise ValueError("Algorithm settings must include 'algorithm'.")
        given = {
            **loop_settings,
            **step_control_settings,
            **algorithm_settings,
        }
        self.setup_compile_settings(
            IntegratorRunSettings(
                precision=precision,
                **self._run_settings(given),
            )
        )
        self._algo_step = self._new_step(algorithm_settings)
        self._record_step_products()
        if self.compile_settings.controller_replaced:
            self._warn_controller_replaced()
        self._step_controller = self._new_controller(step_control_settings)
        self._record_controller_products()
        self._push_resolved_step_settings()
        self._dae_initialiser = self._new_initialiser()
        self._output_functions.update(
            sample_summaries_every=(
                self.compile_settings.sample_summaries_every
            ),
            silent=True,
        )
        self._loop = self._new_loop(loop_settings)
        self._register_loop_children()
        self.update_compile_settings(
            loop_fn=self._loop.device_function, silent=True
        )
        self._warn_if_summary_timing_derived()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
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
            Dictionary of parameters to update; nested dicts are
            flattened one level.
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
        Order: system, outputs, the run's settings, then every child
        with its user keys and the resolved values; ``loop_fn`` is
        captured last.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        updates, unpacked_keys = unpack_dict_values(updates_dict)
        user_keys = set(updates)
        if "drivers_fn" in updates:
            self._drivers_fn = updates["drivers_fn"]
        if "driver_derivative_fn" in updates:
            self._driver_derivative_fn = updates["driver_derivative_fn"]

        recognised = self._system.update(updates, silent=True)
        products = self._system.products
        output_updates = dict(updates)
        output_updates.update(
            precision=products["precision"],
            n_states=products["n_states"],
            n_observables=products["n_observables"],
        )
        recognised |= self._output_functions.update(
            output_updates, silent=True
        )

        run_updates = self._run_settings(updates)
        run_updates.update(
            {key: updates[key] for key in BROADCAST_KEYS if key in updates}
        )
        run_updates["precision"] = products["precision"]
        # A new algorithm returns the controller to the family default
        # unless the same update names one.
        if "algorithm" in updates and "step_controller" not in updates:
            run_updates["step_controller"] = None
        # The last of a filter or loose gains given wins.
        gains_given = set(updates) & set(CONTROLLER_GAIN_NAMES)
        if gains_given and "filter_coefficients" not in updates:
            run_updates["filter_coefficients"] = None
        if "filter_coefficients" in updates and not gains_given:
            for name in CONTROLLER_GAIN_NAMES:
                run_updates[name] = None
        if "summary_window" in updates:
            run_updates["summary_window"] = updates["summary_window"]
        recognised |= self.update_compile_settings(run_updates, silent=True)
        recognised |= self._sync(updates)

        if user_keys & (
            RUN_TIMING_PARAMETERS | {"output_types", "summary_window"}
        ):
            self._warn_if_summary_timing_derived()

        unrecognised = user_keys - recognised
        if unrecognised and not silent:
            raise KeyError(f"Unrecognized parameters: {unrecognised}")
        return recognised | unpacked_keys

    def _run_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Run-owned keys of ``updates`` plus the system and output flags."""
        settings = {
            key: value
            for key, value in updates.items()
            if key in ALL_RUN_PARAMETERS
        }
        settings.update(
            has_mass=self._system.mass is not None,
            has_summary_outputs=self._output_functions.has_summary_outputs,
            has_time_domain_outputs=(
                self._output_functions.has_time_domain_outputs
            ),
        )
        return settings

    def _sync(self, updates: Dict[str, Any]) -> set[str]:
        """Write user keys and resolved settings into every child."""
        recognised = set()
        broadcast = {
            key: updates[key] for key in BROADCAST_KEYS if key in updates
        }
        config = self.compile_settings

        if config.algorithm != self._algo_step_algorithm:
            self._swap_step(updates)
        step_updates = dict(broadcast)
        step_updates.update(
            {
                k: v
                for k, v in updates.items()
                if k in _STEP_KEYS or k in KERNEL_RESOLVED_STEP_PARAMETERS
            }
        )
        step_updates.update(self._step_inputs())
        recognised |= self._algo_step.update(step_updates, silent=True)
        self._record_step_products()

        config = self.compile_settings
        if config.controller_replaced and (
            set(updates) & {"algorithm", "step_controller"}
        ):
            self._warn_controller_replaced()
        if config.step_controller != _controller_name(self._step_controller):
            self._swap_controller(updates)
        controller_updates = dict(broadcast)
        controller_updates.update(
            {k: v for k, v in updates.items() if k in _CONTROLLER_KEYS}
        )
        controller_updates.update(self._controller_inputs())
        recognised |= self._step_controller.update(
            controller_updates, silent=True
        )
        self._record_controller_products()

        self._push_resolved_step_settings()

        init_updates = dict(broadcast)
        init_updates.update(
            {k: v for k, v in updates.items() if k in _STEP_KEYS}
        )
        init_updates.update(self._initialiser_inputs())
        recognised |= self._dae_initialiser.update(init_updates, silent=True)

        self._output_functions.update(
            sample_summaries_every=(
                self.compile_settings.sample_summaries_every
            ),
            silent=True,
        )

        self._register_loop_children()
        loop_updates = dict(broadcast)
        loop_updates.update(
            {
                k: v
                for k, v in updates.items()
                if k in _LOOP_KEYS or k == "state_location"
            }
        )
        loop_updates.update(self._loop_inputs())
        recognised |= self._loop.update(loop_updates, silent=True)
        self.update_compile_settings(
            loop_fn=self._loop.device_function, silent=True
        )
        return recognised

    def _push_resolved_step_settings(self) -> None:
        """Write the resolved step keys and ``is_adaptive`` into the step."""
        config = self.compile_settings
        step_settings = dict(config.step_settings)
        step_settings["is_adaptive"] = config.is_adaptive
        self._algo_step.update(step_settings, silent=True)
        if config.is_implicit and not config.is_linear:
            warn_on_newton_rtol_inversion(
                self._algo_step.solver.rtol, config.controller_rtol
            )

    # ------------------------------------------------------------------
    # Children: construction and inputs
    # ------------------------------------------------------------------
    def _new_step(self, settings: Dict[str, Any]) -> BaseAlgorithmStep:
        """Build the named step from its given keys and the system."""
        step_settings = {
            key: value for key, value in settings.items() if key in _STEP_KEYS
        }
        step_settings.update(self._step_inputs())
        step_settings["algorithm"] = self.compile_settings.algorithm
        step = get_algorithm_step(
            precision=self.precision, settings=step_settings
        )
        self._algo_step_algorithm = self.compile_settings.algorithm
        self._check_algorithm_consumes_mass(step)
        return step

    def _swap_step(self, updates: Dict[str, Any]) -> None:
        """Swap the step, carrying its given keys and compile flags."""
        old = self._algo_step
        buffer_registry.clear_parent(old)
        carried = old.settings_dict
        carried.update(
            jit_flags=old.compile_settings.jit_flags,
            unroll=old.compile_settings.unroll,
        )
        carried.update({k: v for k, v in updates.items() if k in _STEP_KEYS})
        self._algo_step = self._new_step(carried)

    def _step_inputs(self) -> Dict[str, Any]:
        """Return the system's products the step takes."""
        products = self._system.products
        return dict(
            precision=products["precision"],
            n_states=products["n_states"],
            n_drivers=products["n_drivers"],
            dxdt_fn=products["dxdt_fn"],
            observables_fn=products["observables_fn"],
            get_solver_helper_fn=products["get_solver_helper_fn"],
            drivers_fn=self._drivers_fn,
            driver_derivative_fn=self._driver_derivative_fn,
        )

    def _record_step_products(self) -> None:
        """Write the step's flags and defaults into the run settings."""
        step = self._algo_step
        self.update_compile_settings(
            algorithm_defaults=step.algorithm_defaults,
            has_error_estimate=step.has_error_estimate,
            is_implicit=step.is_implicit,
            is_linear=step.is_linear,
            silent=True,
        )

    def _new_controller(
        self, settings: Dict[str, Any]
    ) -> BaseStepController:
        """Build the controller in effect from its given keys."""
        config = self.compile_settings
        controller_settings = {
            key: value
            for key, value in settings.items()
            if key in _CONTROLLER_KEYS
        }
        controller_settings.update(self._controller_inputs())
        controller_settings["step_controller"] = config.step_controller
        return get_controller(
            precision=self.precision,
            settings=controller_settings,
            warn_on_unused=False,
        )

    def _swap_controller(self, updates: Dict[str, Any]) -> None:
        """Swap the controller, carrying its given keys, dt and flags."""
        old = self._step_controller
        buffer_registry.clear_parent(old)
        carried = old.settings_dict
        carried.update(
            dt=old.dt,
            jit_flags=old.compile_settings.jit_flags,
            unroll=old.compile_settings.unroll,
        )
        carried.update(
            {k: v for k, v in updates.items() if k in _CONTROLLER_KEYS}
        )
        self._step_controller = self._new_controller(carried)

    def _controller_inputs(self) -> Dict[str, Any]:
        """Sizes, order, flags and resolved keys the controller takes."""
        products = self._system.products
        inputs = dict(
            precision=products["precision"],
            n_states=products["n_states"],
            mass_flags=products["mass_flags"],
            algorithm_order=self._algo_step.algorithm_order,
        )
        inputs.update(self.compile_settings.controller_settings)
        return inputs

    def _record_controller_products(self) -> None:
        """Write the controller's tolerances into the run settings."""
        controller = self._step_controller.products
        self.update_compile_settings(
            controller_atol=controller["atol"],
            controller_rtol=controller["rtol"],
            silent=True,
        )

    def _warn_controller_replaced(self) -> None:
        config = self.compile_settings
        warn(
            f"Adaptive step controller '{config.requested_controller}' "
            f"cannot be used with fixed-step algorithm "
            f"'{config.algorithm}'. The algorithm does not provide an "
            "error estimate required for adaptive stepping. Replacing "
            "with fixed-step controller.",
            UserWarning,
            stacklevel=4,
        )

    def _new_initialiser(self) -> DAEInitialiser:
        """Build the initialiser from the step's given keys."""
        settings = self._algo_step.settings_dict
        settings.update(self._initialiser_inputs())
        return DAEInitialiser(**settings)

    def _initialiser_inputs(self) -> Dict[str, Any]:
        """Sizes, helper and tolerances the initialiser takes."""
        products = self._system.products
        config = self.compile_settings
        inputs = dict(
            precision=products["precision"],
            n_states=products["n_states"],
            mass_flags=products["mass_flags"],
            get_solver_helper_fn=products["get_solver_helper_fn"],
        )
        if config.dae_initialisation is not None:
            inputs["dae_initialisation"] = config.dae_initialisation
        tolerances = config.inner_tolerances
        for key in ("newton_atol", "newton_rtol"):
            if key in tolerances:
                inputs[key] = tolerances[key]
        return inputs

    def _new_loop(self, settings: Dict[str, Any]) -> IVPLoop:
        """Build the loop from its given placements and the inputs."""
        loop_settings = {
            key: value for key, value in settings.items() if key in _LOOP_KEYS
        }
        loop_settings.update(self._loop_inputs())
        return IVPLoop(**loop_settings)

    def _loop_inputs(self) -> Dict[str, Any]:
        """Sizes, schedule, dt and device functions the loop takes."""
        products = self._system.products
        outputs = self._output_functions.products
        step = self._algo_step.products
        controller = self._step_controller.products
        config = self.compile_settings
        inputs = dict(
            precision=products["precision"],
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
            dt=controller["dt"],
            is_adaptive=controller["is_adaptive"],
            save_state_fn=outputs["save_state_fn"],
            update_summaries_fn=outputs["update_summaries_fn"],
            save_summaries_fn=outputs["save_summaries_fn"],
            step_controller_fn=controller["step_controller_fn"],
            step_fn=step["step_fn"],
            observables_fn=products["observables_fn"],
            drivers_fn=self._drivers_fn,
        )
        inputs.update(config.loop_timing)
        inputs["initialise_state_fn"] = self._dae_initialiser.products[
            "initialise_state_fn"
        ]
        return inputs

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

    def _check_algorithm_consumes_mass(self, step: BaseAlgorithmStep) -> None:
        """Reject an explicit step on a system with a mass matrix.

        Raises
        ------
        ValueError
            If the system has a mass matrix and the step is explicit.
        """
        if self._system.mass is None or step.is_implicit:
            return
        raise ValueError(
            "The system defines a mass matrix and requires an "
            f"implicit algorithm; '{self.compile_settings.algorithm}' "
            "does not consume a mass matrix and would integrate the "
            "constraint residuals as derivatives."
        )

    def _warn_if_summary_timing_derived(self):
        if self.compile_settings.summary_window_derived:
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

    # ------------------------------------------------------------------
    # Build and settings
    # ------------------------------------------------------------------
    @property
    def device_function(self):
        """Return the compiled CUDA loop function."""
        return self.get_cached_output("loop_fn")

    def build(self) -> SingleIntegratorRunCache:
        """Return the captured loop function."""
        return SingleIntegratorRunCache(loop_fn=self.compile_settings.loop_fn)

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Return the output compile flags."""
        return self._output_functions.compile_flags

    @property
    def n_states(self) -> int:
        """Return the system's state count."""
        return self._system.products["n_states"]

    @property
    def threads_per_step(self) -> int:
        """Return the threads the step function needs per run."""
        return self._algo_step.threads_per_step

    @property
    def is_implicit(self) -> bool:
        """Return whether the step is implicit."""
        return self._algo_step.is_implicit

    @property
    def algorithm_family(self) -> str:
        """Return the step's family name."""
        return self._algo_step.algorithm_family

    @property
    def newton_solves_per_step(self) -> int:
        """Return the Newton solves one step runs."""
        return self._algo_step.newton_solves_per_step

    @property
    def step_operation_count(self) -> int:
        """Return the operator count of one fully unrolled step."""
        return self._algo_step.step_operation_count

    @property
    def uses_direct_solver(self) -> bool:
        """Return whether the step solves its stages with a direct LU."""
        return self._algo_step.uses_direct_solver

    @property
    def stage_count(self) -> int:
        """Return the step's stage count."""
        return self._algo_step.stage_count

    @property
    def accumulates_output(self) -> bool:
        """Return whether the step accumulates its output over stages."""
        return self._algo_step.accumulates_output

    @property
    def local_elements(self) -> int:
        """Return the elements the step declares in local memory."""
        return self._algo_step.local_elements

    @property
    def summary_window_derived(self) -> bool:
        """Return whether the summary window follows the duration."""
        return self.compile_settings.summary_window_derived

    @property
    def n_error(self) -> int:
        """Return the length of the shared error buffer."""
        return self._algo_step.n_error

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Given keys of this run and its children."""
        settings = super().settings_dict
        for child in (
            self._loop,
            self._output_functions,
            self._step_controller,
            self._algo_step,
        ):
            settings.update(self.child_settings(child))
        return settings

    def grouped_settings(self) -> Dict[str, Dict[str, Any]]:
        """Return ``settings_dict`` split into the constructor's groups."""
        settings = self.settings_dict
        groups = {
            "loop_settings": _LOOP_KEYS | RUN_TIMING_PARAMETERS,
            "output_settings": self._output_functions.settings_keys,
            "step_control_settings": (
                _CONTROLLER_KEYS | RUN_CONTROLLER_PARAMETERS
                | {"step_controller"}
            ),
            "algorithm_settings": (
                _STEP_KEYS | RUN_RESOLVED_STEP_PARAMETERS | {"algorithm"}
            ),
        }
        return {
            name: {
                key: value
                for key, value in settings.items()
                if key in keys
            }
            for name, keys in groups.items()
        }

    def copy(self) -> "SingleIntegratorRunCore":
        """Return a new run with these settings on a copy of the system."""
        return type(self)(
            self._system.copy(),
            drivers_fn=self._drivers_fn,
            driver_derivative_fn=self._driver_derivative_fn,
            **self.grouped_settings(),
        )

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
