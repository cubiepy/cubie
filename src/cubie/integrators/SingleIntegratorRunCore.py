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
from cubie._utils import build_config, merge_kwargs_into_settings
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
    """Return the string name of the active controller."""
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
        The settings in effect, as one flat dict; every child takes
        its own keys from it.
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

        output_settings, _ = merge_kwargs_into_settings(
            settings, ALL_OUTPUT_FUNCTION_PARAMETERS
        )
        self._output_functions = OutputFunctions(
            **{**output_settings, **OutputFunctions.system_inputs(system)}
        )
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
    def _update(self, updates: Dict[str, Any], silent: bool) -> set[str]:
        """Update every child and recapture ``loop_fn``."""
        system = self._system
        recognised = self.update_compile_settings(updates, silent=True)
        recognised |= self._output_functions.update(
            {**updates, **OutputFunctions.system_inputs(system)}, silent=True
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
            {
                **updates,
                **BaseStepController.system_inputs(
                    system, algorithm_order=self._algo_step.algorithm_order
                ),
            },
            silent=True,
        )

        recognised |= self._dae_initialiser.update(
            {**updates, **DAEInitialiser.system_inputs(system)}, silent=True
        )

        self._register_loop_children()
        recognised |= self._loop.update(
            {**updates, **self._loop_inputs()}, silent=True
        )
        self.update_compile_settings(
            loop_fn=self._loop.device_function, silent=True
        )
        return recognised

    # ------------------------------------------------------------------
    # Children
    # ------------------------------------------------------------------
    def _new_step(self, settings: Dict[str, Any]) -> BaseAlgorithmStep:
        """Build the algorithm step named in the compile settings."""
        step_settings = {
            **settings,
            **self._step_inputs(),
            "algorithm": self.compile_settings.algorithm,
        }
        step = get_algorithm_step(
            precision=self.precision, settings=step_settings
        )
        self._algo_step_algorithm = self.compile_settings.algorithm
        self._check_algorithm_consumes_mass(step)
        return step

    def _swap_step(self, updates: Dict[str, Any]) -> None:
        """Replace the step with one built from ``updates``."""
        buffer_registry.clear_parent(self._algo_step)
        self._algo_step = self._new_step(
            {**self.compile_settings.init_kwargs, **updates}
        )

    def _step_inputs(self) -> Dict[str, Any]:
        """Return what the step takes from the system and the drivers."""
        config = self.compile_settings
        return BaseAlgorithmStep.system_inputs(
            self._system,
            drivers_fn=config.drivers_fn,
            driver_derivative_fn=config.driver_derivative_fn,
            is_adaptive=config.step_controller != "fixed",
        )

    def _new_controller(
        self, settings: Dict[str, Any]
    ) -> BaseStepController:
        """Build the controller named in the compile settings."""
        controller_settings = {
            **settings,
            **BaseStepController.system_inputs(
                self._system, algorithm_order=self._algo_step.algorithm_order
            ),
            "step_controller": self.compile_settings.step_controller,
        }
        return get_controller(
            precision=self.precision,
            settings=controller_settings,
            warn_on_unused=False,
        )

    def _swap_controller(self, updates: Dict[str, Any]) -> None:
        """Replace the controller with one built from ``updates``."""
        buffer_registry.clear_parent(self._step_controller)
        self._step_controller = self._new_controller(
            {**self.compile_settings.init_kwargs, **updates}
        )

    def _new_initialiser(self, settings: Dict[str, Any]) -> DAEInitialiser:
        """Build the initialiser from the step's settings and ``settings``."""
        return DAEInitialiser(
            **{
                **self._algo_step.settings_dict,
                **settings,
                **DAEInitialiser.system_inputs(self._system),
            }
        )

    def _new_loop(self, settings: Dict[str, Any]) -> IVPLoop:
        """Build the loop from ``settings`` and the children's products."""
        loop_settings, _ = merge_kwargs_into_settings(
            settings, ALL_LOOP_SETTINGS
        )
        return IVPLoop(**{**loop_settings, **self._loop_inputs()})

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

    @property
    def optimisation_candidates(self) -> Tuple[Dict[str, Any], ...]:
        """Return the setting combinations ``Solver.optimize`` times."""
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
        config = self._loop.compile_settings
        return self.time_domain_outputs_requested and (
            config.save_regularly or config.save_last
        )

    @property
    def has_summary_outputs(self) -> bool:
        """Return True if summary outputs will be produced by the loop"""
        config = self._loop.compile_settings
        return self.summary_outputs_requested and (
            config.summarise_regularly or config.summarise_last
        )
