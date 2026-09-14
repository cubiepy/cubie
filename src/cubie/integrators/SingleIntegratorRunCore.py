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

from attrs import define, field

from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie._utils import (
    build_config,
    merge_kwargs_into_settings,
    unpack_dict_values,
)
from cubie.buffer_registry import buffer_registry
from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings
from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
)
from cubie.integrators.dae_initialiser import DAEInitialiser
from cubie.integrators.loops.ode_loop import ALL_LOOP_SETTINGS, IVPLoop
from cubie.outputhandling.output_functions import (
    ALL_OUTPUT_FUNCTION_PARAMETERS,
    OutputFunctions,
)
from cubie.integrators.step_control import (
    _CONTROLLER_REGISTRY,
    get_controller,
)
from cubie.integrators.step_control.base_step_controller import (
    BaseStepController,
)


if TYPE_CHECKING:  # pragma: no cover - imported for static typing only
    from cubie.odesystems.baseODE import BaseODE


@define
class SingleIntegratorRunCache(CUDADispatcherCache):
    """Cache for SingleIntegratorRunCore device function.

    Attributes
    ----------
    loop_fn
        Compiled CUDA loop callable ready for execution on device.
    """

    loop_fn: Callable = field(eq=False)


def _controller_name(controller: BaseStepController) -> str:
    """Return the registry name of ``controller``'s class."""
    for name, cls in _CONTROLLER_REGISTRY.items():
        if type(controller) is cls:
            return name
    raise ValueError(f"{type(controller).__name__} is not registered.")


class SingleIntegratorRunCore(CUDAFactory):
    """Coordinate a single ODE integration loop and its dependencies.

    Parameters
    ----------
    system
        ODE system whose device functions drive the integration.
    drivers_fn
        Device function that interpolates driver inputs.
    driver_derivative_fn
        Device function giving the drivers' time derivative.
    **settings
        The resolved settings; ``algorithm`` is required.
    """

    settings_keys = frozenset({"algorithm", "step_controller"})

    def __init__(
        self,
        system: "BaseODE",
        drivers_fn: Optional[Callable] = None,
        driver_derivative_fn: Optional[Callable] = None,
        **settings: Any,
    ) -> None:
        super().__init__()
        settings, _ = unpack_dict_values(settings)
        if settings.get("algorithm") is None:
            raise ValueError("Settings must include 'algorithm'.")
        # The run and its children take the system's precision.
        settings["precision"] = system.precision

        self._system = system
        config = build_config(
            IntegratorRunSettings,
            required={
                "precision": system.precision,
                "drivers_fn": drivers_fn,
                "driver_derivative_fn": driver_derivative_fn,
            },
            **settings,
        )
        self.setup_compile_settings(config)
        # Every child and the system take the same compile flags.
        settings.update(unroll=config.unroll, jit_flags=config.jit_flags)
        system.update_compile_settings(
            {"unroll": config.unroll, "jit_flags": config.jit_flags},
            silent=True,
        )

        output_settings, _ = merge_kwargs_into_settings(
            settings,
            ALL_OUTPUT_FUNCTION_PARAMETERS | {"unroll", "jit_flags"},
        )
        output_settings.update(self._output_inputs())
        self._output_functions = OutputFunctions(**output_settings)

        self._algo_step = self._new_step(settings)
        self._step_controller = self._new_controller(settings)
        self._dae_initialiser = self._new_initialiser(settings)
        self._loop = self._new_loop(settings)
        self._register_loop_children()
        self.update_compile_settings(
            loop_fn=self._loop.device_function, silent=True
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> set[str]:
        """Write ``updates`` into every child and recapture ``loop_fn``.

        Parameters
        ----------
        updates_dict
            Parameters to update; nested dicts flatten one level.
        silent
            Ignore unrecognised parameters instead of raising.
        **kwargs
            Further updates.

        Returns
        -------
        set[str]
            The recognised names.

        Raises
        ------
        KeyError
            Unrecognised parameters when not ``silent``.
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

        # Constant values reach the system by name from the Solver only.
        recognised = self._system.update_compile_settings(
            updates, silent=True
        )
        recognised |= self.update_compile_settings(updates, silent=True)
        recognised |= self._output_functions.update(
            {**updates, **self._output_inputs()}, silent=True
        )

        config = self.compile_settings
        if config.algorithm != self._algo_step_algorithm:
            self._swap_step(updates)
        recognised |= self._algo_step.update(
            {**updates, **self._step_inputs()}, silent=True
        )

        if config.step_controller != _controller_name(self._step_controller):
            self._swap_controller(updates)
        recognised |= self._step_controller.update(
            {**updates, **self._controller_inputs()}, silent=True
        )

        recognised |= self._dae_initialiser.update(
            {**updates, **self._initialiser_inputs()}, silent=True
        )

        self._register_loop_children()
        recognised |= self._loop.update(
            {**updates, **self._loop_inputs()}, silent=True
        )
        self.update_compile_settings(
            loop_fn=self._loop.device_function, silent=True
        )

        unrecognised = user_keys - recognised
        if unrecognised and not silent:
            raise KeyError(f"Unrecognized parameters: {unrecognised}")
        return recognised | unpacked_keys

    # ------------------------------------------------------------------
    # Children: construction and inputs
    # ------------------------------------------------------------------
    def _output_inputs(self) -> Dict[str, Any]:
        """Return the system's sizes and precision the outputs take."""
        system = self._system
        return dict(
            precision=system.precision,
            n_states=system.sizes.states,
            n_observables=system.sizes.observables,
        )

    def _new_step(self, settings: Dict[str, Any]) -> BaseAlgorithmStep:
        """Build the named step from ``settings`` and the system."""
        step_settings = {**settings, **self._step_inputs()}
        step_settings["algorithm"] = self.compile_settings.algorithm
        step = get_algorithm_step(
            precision=self.precision, settings=step_settings
        )
        self._algo_step_algorithm = self.compile_settings.algorithm
        self._check_algorithm_consumes_mass(step)
        return step

    def _swap_step(self, updates: Dict[str, Any]) -> None:
        """Swap the step, carrying the old one's compile flags."""
        old = self._algo_step
        buffer_registry.clear_parent(old)
        settings = dict(
            jit_flags=old.compile_settings.jit_flags,
            unroll=old.compile_settings.unroll,
        )
        settings.update(updates)
        self._algo_step = self._new_step(settings)

    def _step_inputs(self) -> Dict[str, Any]:
        """Return the system's products and flags the step takes."""
        system = self._system
        config = self.compile_settings
        return dict(
            precision=system.precision,
            n_states=system.sizes.states,
            n_drivers=system.sizes.drivers,
            dxdt_fn=system.dxdt_fn,
            observables_fn=system.observables_fn,
            get_solver_helper_fn=system.get_solver_helper,
            drivers_fn=config.drivers_fn,
            driver_derivative_fn=config.driver_derivative_fn,
            is_adaptive=config.step_controller != "fixed",
        )

    def _new_controller(
        self, settings: Dict[str, Any]
    ) -> BaseStepController:
        """Build the named controller from ``settings``."""
        controller_settings = {**settings, **self._controller_inputs()}
        controller_settings["step_controller"] = (
            self.compile_settings.step_controller
        )
        return get_controller(
            precision=self.precision,
            settings=controller_settings,
            warn_on_unused=False,
        )

    def _swap_controller(self, updates: Dict[str, Any]) -> None:
        """Swap the controller, carrying the old one's compile flags."""
        old = self._step_controller
        buffer_registry.clear_parent(old)
        settings = dict(
            jit_flags=old.compile_settings.jit_flags,
            unroll=old.compile_settings.unroll,
        )
        settings.update(updates)
        self._step_controller = self._new_controller(settings)

    def _controller_inputs(self) -> Dict[str, Any]:
        """Return the sizes, order and mass flags the controller takes."""
        system = self._system
        return dict(
            precision=system.precision,
            n_states=system.sizes.states,
            mass_flags=system.mass_diagonal_flags,
            algorithm_order=self._algo_step.algorithm_order,
        )

    def _new_initialiser(self, settings: Dict[str, Any]) -> DAEInitialiser:
        """Build the initialiser from the step's settings and ``settings``."""
        init_settings = self._algo_step.settings_dict
        init_settings.update(settings)
        init_settings.update(self._initialiser_inputs())
        init_settings["unroll"] = self._algo_step.compile_settings.unroll
        init_settings["jit_flags"] = self._algo_step.compile_settings.jit_flags
        return DAEInitialiser(**init_settings)

    def _initialiser_inputs(self) -> Dict[str, Any]:
        """Return the sizes and helper getter the initialiser takes."""
        system = self._system
        return dict(
            precision=system.precision,
            n_states=system.sizes.states,
            mass_flags=system.mass_diagonal_flags,
            get_solver_helper_fn=system.get_solver_helper,
        )

    def _new_loop(self, settings: Dict[str, Any]) -> IVPLoop:
        """Build the loop from ``settings`` and the children's products."""
        loop_settings, _ = merge_kwargs_into_settings(
            settings, ALL_LOOP_SETTINGS | {"unroll", "jit_flags"}
        )
        loop_settings.update(self._loop_inputs())
        return IVPLoop(**loop_settings)

    def _loop_inputs(self) -> Dict[str, Any]:
        """Return the sizes, dt and device functions the loop takes."""
        system = self._system
        sizes = system.sizes
        outputs = self._output_functions.products
        step = self._algo_step.products
        controller = self._step_controller.products
        return dict(
            precision=system.precision,
            n_states=sizes.states,
            n_parameters=sizes.parameters,
            n_observables=sizes.observables,
            n_drivers=sizes.drivers,
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
            observables_fn=system.observables_fn,
            drivers_fn=self.compile_settings.drivers_fn,
            initialise_state_fn=self._dae_initialiser.products[
                "initialise_state_fn"
            ],
        )

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

    # ------------------------------------------------------------------
    # Build and settings
    # ------------------------------------------------------------------
    @property
    def device_function(self):
        """Return the compiled CUDA loop function."""
        return self.get_cached_output("loop_fn")

    def build(self) -> SingleIntegratorRunCache:
        """Return the loop function captured by the last update."""
        return SingleIntegratorRunCache(loop_fn=self.compile_settings.loop_fn)

    @property
    def n_error(self) -> int:
        """Return the length of the shared error buffer."""
        return self._algo_step.n_error

    @property
    def settings_dict(self) -> Dict[str, Any]:
        """Settings of this run and its children."""
        settings = super().settings_dict
        for child in (
            self._loop,
            self._output_functions,
            self._dae_initialiser,
            self._step_controller,
            self._algo_step,
        ):
            settings.update(child.settings_dict)
        return settings

    def copy(self) -> "SingleIntegratorRunCore":
        """Return a new run with these settings on a copy of the system."""
        config = self.compile_settings
        return type(self)(
            self._system.copy(),
            drivers_fn=config.drivers_fn,
            driver_derivative_fn=config.driver_derivative_fn,
            unroll=config.unroll,
            jit_flags=config.jit_flags,
            **self.settings_dict,
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
