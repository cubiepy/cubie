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

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from warnings import warn

from attrs import define, field
from numpy import asarray, finfo as np_finfo

from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie._utils import PrecisionDType, unpack_dict_values
from cubie.buffer_registry import buffer_registry
from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings
from cubie.integrators.algorithms import get_algorithm_step
from cubie.integrators.loops.ode_loop import IVPLoop
from cubie.outputhandling import OutputCompileFlags
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.integrators.step_control import get_controller


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
    single_integrator_function
        Compiled CUDA loop callable ready for execution on device.
    """
    single_integrator_function: Callable = field(eq=False)

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
    evaluate_driver_at_t
        Optional device function that interpolates driver inputs for use
        by step algorithms.
    driver_del_t
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
    solver_helper_fn
        Callable used to fetch solver helper device functions, with
        the same signature as :meth:`BaseODE.get_solver_helper`. The
        owning batch solver kernel passes its own binding; ``None``
        falls back to ``system.get_solver_helper``.
    """

    _INNER_TOLERANCE_KEYS = (
        "krylov_atol",
        "krylov_rtol",
        "krylov_residual_reduction",
        "newton_atol",
        "newton_rtol",
    )

    def __init__(
        self,
        system: "BaseODE",
        loop_settings: Optional[Dict[str, Any]] = None,
        output_settings: Optional[Dict[str, Any]] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        driver_del_t: Optional[Callable] = None,
        algorithm_settings: Optional[Dict[str, Any]] = None,
        step_control_settings: Optional[Dict[str, Any]] = None,
        solver_helper_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self._solver_helper_fn = (
            solver_helper_fn or system.get_solver_helper
        )

        if step_control_settings is None:
            step_control_settings = {}
        if algorithm_settings is None:
            algorithm_settings = {}
        if output_settings is None:
            output_settings = {}
        if loop_settings is None:
            loop_settings = {}

        # Track which inner-solver tolerances the user set explicitly so
        # the derived controller-scaled defaults never overwrite them.
        self._user_given_inner_tols = {
            key
            for key in self._INNER_TOLERANCE_KEYS
            if algorithm_settings.get(key) is not None
        }

        precision = system.precision

        self._system = system
        system_sizes = system.sizes
        n = system_sizes.states

        # Outputsettings may/may not include precision, so we pop it here to
        # ensure that it gets passed a precision matching system's
        _ = output_settings.pop("precision", None)
        self._output_functions = OutputFunctions(
            max_states=system_sizes.states,
            max_observables=system_sizes.observables,
            precision=precision,
            **output_settings,
        )

        dt = step_control_settings.get("dt", None)
        algorithm_settings["n"] = n
        algorithm_settings["n_drivers"] = system_sizes.drivers
        # The mass matrix belongs to the ODE system; algorithms read
        # it from the system when building solver helpers.
        if "M" in algorithm_settings:
            raise ValueError(
                "'M' is not an algorithm setting: the mass matrix is "
                "part of the system definition. Pass mass= to "
                "create_ODE_system instead."
            )
        if dt is not None:
            algorithm_settings["dt"] = dt
        algorithm_settings["evaluate_driver_at_t"] = evaluate_driver_at_t
        # Thread the driver time-derivative through to algorithm factories
        algorithm_settings["driver_del_t"] = driver_del_t
        self._algo_step = get_algorithm_step(
                precision=precision,
                settings=algorithm_settings,
        )
        self._check_algorithm_consumes_mass(algorithm_settings["algorithm"])
        # Fetch and override controller defaults from algorithm settings
        controller_settings = (
            self._algo_step.controller_defaults.step_controller.copy())
        controller_settings.update(step_control_settings)
        controller_settings["n"] = system_sizes.states
        controller_settings["algorithm_order"] = self._algo_step.order

        self._step_controller = get_controller(
            precision=precision,
            settings=controller_settings,
        )

        self.check_compatibility(
            algorithm_settings["algorithm"],
            controller_settings["step_controller"],
            precision,
        )

        # Default any unset inner-solver tolerances from the controller.
        self._apply_inner_tolerance_defaults()

        loop_settings["dt"] = self._step_controller.dt
        loop_settings["dt_min"] = self._step_controller.dt_min
        loop_settings["dt_max"] = self._step_controller.dt_max
        loop_settings["is_adaptive"] = self._step_controller.is_adaptive

        config = IntegratorRunSettings(
            precision=system.precision,
            algorithm=algorithm_settings["algorithm"],
            step_controller=controller_settings["step_controller"],
        )

        self.setup_compile_settings(config)
        self._loop = self.instantiate_loop(
            precision=precision,
            n_states=system_sizes.states,
            n_parameters=system_sizes.parameters,
            n_observables=system_sizes.observables,
            n_drivers=system_sizes.drivers,
            compile_flags=self._output_functions.compile_flags,
            state_summaries_buffer_height=self._output_functions.state_summaries_buffer_height,
            observable_summaries_buffer_height=self._output_functions.observable_summaries_buffer_height,
            loop_settings=loop_settings,
            evaluate_driver_at_t=evaluate_driver_at_t,
        )

        # Keep the timing parameters explicitly set by the user at run level
        # Only pass the loop values to implement.
        self._user_timing = {
            "save_every": None,
            "summarise_every": None,
            "sample_summaries_every": None,
        }
        self.is_duration_dependent = False
        self._process_loop_timing(loop_settings)


        # Register algorithm step and controller buffers with loop as parent
        buffer_registry.register_child(
            self._loop, self._algo_step, name="algorithm"
        )
        buffer_registry.register_child(
                self._loop, self._step_controller, name='controller'
        )

    def _process_loop_timing(self, settings_dict: Dict[str, Any]):
        """Derive and apply timing parameters from *settings_dict*.

        Resolves ``save_every``, ``summarise_every``, and
        ``sample_summaries_every`` from user intent and output
        configuration, then forwards the derived values to the loop
        and output functions.

        Parameters
        ----------
        settings_dict
            Mapping containing timing overrides.  Recognised keys are
            ``save_every``, ``summarise_every``, and
            ``sample_summaries_every``.
        """
        timing_params = (
            "save_every",
            "summarise_every",
            "sample_summaries_every",
        )
        # 1. Overwrite "user intent" with incoming values
        for p in timing_params:
            if p in settings_dict:
                self._user_timing[p] = settings_dict[p]

        has_time_domain_outputs = self.time_domain_outputs_requested
        has_summary_outputs = self.summary_outputs_requested

        # 2. Get provided values from user intent
        save_every = self._user_timing["save_every"]
        summarise_every = self._user_timing["summarise_every"]
        sample_summaries_every = self._user_timing["sample_summaries_every"]

        save_last = False
        self.is_duration_dependent = False

        # 3. Time-domain outputs
        if has_time_domain_outputs and save_every is None:
            save_last = True

        # 4. Summary outputs
        if has_summary_outputs:
            if summarise_every is None:
                # There is no `summarise_last`, we simulate
                # summarise_regularly once we get a duration.
                self.is_duration_dependent = True
            else:
                if sample_summaries_every is None:
                    sample_summaries_every = summarise_every / 10.0
        else:
            summarise_every = None
            sample_summaries_every = None

        save_regularly = save_every is not None and has_time_domain_outputs
        summarise_regularly = (summarise_every is not None and
                               has_summary_outputs)
        values = dict(
            save_every=save_every,
            summarise_every=summarise_every,
            sample_summaries_every=sample_summaries_every,
            save_last=save_last,
            save_regularly=save_regularly,
            summarise_regularly=summarise_regularly,
        )

        # Update loop and output functions with derived timing values.
        self._warn_if_summary_timing_derived()
        self._loop.update(values)
        self._output_functions.update(values, silent=True)

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

    def set_summary_timing_from_duration(self,
                                         duration: float):
        """Set summary timing from *duration* when no explicit timing
        was provided.

        Parameters
        ----------
        duration
            Total integration duration used to derive
            ``sample_summaries_every`` and ``summarise_every``.
        """

        if self.is_duration_dependent:
            samples_per_summary = 100
            sample_summaries_every = duration / samples_per_summary

            self._loop.update(
                summarise_every=duration,
                sample_summaries_every=sample_summaries_every,
            )
            self._output_functions.update(
                sample_summaries_every=sample_summaries_every,
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
        ``_user_given_inner_tols``) are preserved.  The solver norms'
        tolerance converter broadcasts uniform arrays to their own
        vector length; a non-uniform per-state vector must match the
        solver vector exactly.

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

        if self._algo_step.is_adaptive:
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
        return self.get_cached_output("single_integrator_function")

    def check_compatibility(
        self,
        algorithm_name: str = None,
        controller_name: str = None,
        precision: PrecisionDType = None,
    ) -> None:
        """Validate algorithm and controller compatibility.

        This method checks whether the chosen integration algorithm and step
        controller are compatible. When an adaptive controller is paired with
        a fixed-step (errorless) algorithm, this method replaces the adaptive
        controller with a fixed-step controller and issues a warning.

        The validation is performed during integrator initialization, after
        both the algorithm and controller have been instantiated but before
        the CUDA loop is compiled.

        Parameters
        ----------
        algorithm_name : str, optional
            Name of the algorithm being used. If not provided, retrieved from
            compile_settings.
        controller_name : str, optional
            Name of the controller being used. If not provided, retrieved from
            compile_settings.
        precision : PrecisionDType, optional
            Numerical precision for the controller. If not provided, retrieved
            from system.

        Notes
        -----
        When an incompatible configuration is detected (adaptive controller
        with errorless algorithm), the controller is automatically replaced
        with a fixed-step controller using dt from the original controller.
        A warning is issued to inform the user of this automatic correction.

        Valid combinations:
        - Adaptive algorithm + adaptive controller: Valid and recommended
        - Errorless algorithm + fixed controller: Valid
        - Adaptive algorithm + fixed controller: Valid (uses fixed step)
        - Errorless algorithm + adaptive controller: Auto-corrected with warning
        """

        if (not self._algo_step.is_adaptive and
                self._step_controller.is_adaptive):
            dt = self._step_controller.dt
            
            # Get names from arguments or compile_settings
            if algorithm_name is None:
                algorithm_name = self.compile_settings.algorithm
            if controller_name is None:
                controller_name = self.compile_settings.step_controller
            if precision is None:
                precision = self._system.precision
            
            warn(
                f"Adaptive step controller '{controller_name}' cannot be "
                f"used with fixed-step algorithm '{algorithm_name}'. "
                f"The algorithm does not provide an error estimate "
                f"required for adaptive stepping. "
                f"Replacing with fixed-step controller (dt={dt}).",
                UserWarning,
                stacklevel=3
            )
            
            # Replace with a fixed step controller, keeping the outgoing
            # controller's atol/rtol so implicit algorithms still derive
            # their inner-solver tolerances from the user's request.
            self._step_controller = get_controller(
                precision=precision,
                settings={
                    "step_controller": "fixed",
                    "dt": dt,
                    "n": self._system.sizes.states,
                    "atol": self._step_controller.atol,
                    "rtol": self._step_controller.rtol,
                },
                warn_on_unused=False,
            )

    def instantiate_loop(
        self,
        precision: PrecisionDType,
        n_states: int,
        n_parameters: int,
        n_observables: int,
        n_drivers: int,
        state_summaries_buffer_height: int,
        observable_summaries_buffer_height: int,
        compile_flags: OutputCompileFlags,
        loop_settings: Dict[str, Any],
        evaluate_driver_at_t: Optional[Callable] = None,
    ) -> IVPLoop:
        """Instantiate the integrator loop.

        Parameters
        ----------
        precision
            Numerical precision used when compiling the loop.
        n_states
            Number of state variables in the system.
        n_parameters
            Number of persistent parameters available to the loop.
        n_observables
            Number of observables emitted by the system.
        n_drivers
            Number of external driver signals consumed by the loop.
        state_summaries_buffer_height
            Height of the state summary buffer managed by the outputs.
        observable_summaries_buffer_height
            Height of the observable summary buffer managed by the outputs.
        compile_flags
            Output function compile flags generated by
            :class:`cubie.outputhandling.OutputFunctions`.
        loop_settings
            Mapping of loop configuration overrides forwarded directly to the
            :class:`~cubie.integrators.loops.ode_loop.IVPLoop` constructor.
        evaluate_driver_at_t
            Optional device function that evaluates drivers for proposed times.

        Returns
        -------
        IVPLoop
            Configured loop instance ready for CUDA compilation.
        """
        n_counters = 4 if compile_flags.save_counters else 0
        
        loop_kwargs = dict(loop_settings)

        # Build the loop with individual parameters (new API)
        loop_kwargs.update(
            precision=precision,
            n_states=n_states,
            compile_flags=compile_flags,
            n_parameters=n_parameters,
            n_drivers=n_drivers,
            n_observables=n_observables,
            n_error=self.n_error,
            n_counters=n_counters,
            state_summaries_buffer_height=state_summaries_buffer_height,
            observable_summaries_buffer_height=observable_summaries_buffer_height,
        )
        if "evaluate_driver_at_t" not in loop_kwargs:
            loop_kwargs["evaluate_driver_at_t"] = evaluate_driver_at_t

        loop = IVPLoop(**loop_kwargs)
        return loop

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
            Dictionary of parameters to update.
        silent
            If ``True``, suppress warnings about unrecognised parameters.
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
        When algorithm or controller selections change, new instances are
        created and primed with settings from their predecessors before
        applying ``updates_dict``. Parameters present only on the new
        instance are ignored unless explicitly provided in the update.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        # Flatten any nested dict values so that all parameters are
        # top-level keys before passing to sub-components. For example,
        # step_controller_settings={'dt_min': 0.01} becomes dt_min=0.01.
        # This ensures sub-components (algorithm, controller, output
        # functions) receive only flat parameter sets.
        updates_dict, unpacked_keys = unpack_dict_values(updates_dict)

        all_unrecognized = set(updates_dict.keys())
        recognized = set()

        if "solver_helper_fn" in updates_dict:
            self.set_solver_helper_fn(updates_dict["solver_helper_fn"])
            recognized.add("solver_helper_fn")

        system_recognized = self._system.update(updates_dict, silent=True)

        # Capture n and n_drivers whether or not system updated, in case
        # of an algo/step swap
        updates_dict.update({'n': self._system.sizes.states})
        updates_dict.update({'n_drivers': self._system.sizes.drivers})

        # Capture outputsettings-generated compile settings and pass on
        out_rcgnzd = self._output_functions.update(updates_dict, silent=True)
        if out_rcgnzd:
            updates_dict.update({**self._output_functions.buffer_sizes_dict})

        # Capture algorithm-generated compile settings and pass on
        step_recognized = self._switch_algos(updates_dict)
        step_recognized |= self._algo_step.update(updates_dict, silent=True)
        if step_recognized:
            updates_dict.update(
                {"threads_per_step": self._algo_step.threads_per_step}
            )

        updates_dict["algorithm_order"] = self._algo_step.order

        ctrl_rcgnzd = self._switch_controllers(updates_dict)
        ctrl_rcgnzd |= self._step_controller.update(updates_dict, silent=True)
        if ctrl_rcgnzd:
            updates_dict.update(
                {
                    "is_adaptive": self._step_controller.is_adaptive,
                    "dt_min": self._step_controller.dt_min,
                    "dt_max": self._step_controller.dt_max,
                    "dt": self._step_controller.dt,
                }
            )

        # Record any inner-solver tolerances the user set explicitly so the
        # derived defaults never overwrite them on this or a later update.
        for key in self._INNER_TOLERANCE_KEYS:
            if updates_dict.get(key) is not None:
                self._user_given_inner_tols.add(key)

        # Re-derive unset inner-solver tolerances when the controller
        # tolerances change or the algorithm is swapped, so they keep
        # tracking the controller atol/rtol.
        rederive = bool(
            ctrl_rcgnzd & {"atol", "rtol", "step_controller"}
            or "algorithm" in step_recognized
        )
        if rederive:
            step_recognized |= self._apply_inner_tolerance_defaults()

        # Re-register algo and controller buffers to refresh sizing in loop
        buffer_registry.register_child(
                self._loop, self._algo_step, name='algorithm'
        )
        buffer_registry.register_child(
                self._loop, self._step_controller, name='controller'
        )

        loop_recognized = self._loop.update(updates_dict, silent=True)
        self._process_loop_timing(updates_dict)

        recognized |= self.update_compile_settings(updates_dict, silent=True)
        recognized |= (out_rcgnzd | ctrl_rcgnzd | step_recognized |
                       system_recognized | loop_recognized)

        all_unrecognized -= recognized
        if all_unrecognized and not silent:
            raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        if recognized:
            self._invalidate_cache()

        self.check_compatibility()

        # Include unpacked dict keys in recognized set
        return recognized | unpacked_keys

    def _switch_algos(self, updates_dict):
        """Replace the algorithm step when ``updates_dict`` contains a
        new ``"algorithm"`` key and propagate defaults.

        Parameters
        ----------
        updates_dict
            Mutable mapping of pending updates.  Modified in-place to
            include algorithm defaults for the new step.

        Returns
        -------
        set of str
            ``{"algorithm"}`` if a swap occurred, otherwise empty.
        """
        if "algorithm" not in updates_dict:
            return set()
        precision = updates_dict.get('precision', self.precision)

        new_algo = updates_dict.get("algorithm").lower()
        if new_algo != self.compile_settings.algorithm:
            buffer_registry.clear_parent(self._algo_step)
            old_settings = self._algo_step.settings_dict
            old_settings["algorithm"] = new_algo
            self._algo_step = get_algorithm_step(
                    precision=precision,
                    settings=old_settings,
            )
            self.update_compile_settings(algorithm=new_algo)
            self._check_algorithm_consumes_mass(new_algo)
        updates_dict["algorithm"] = new_algo

        # Update any not-deliberately-updated controller settings with defaults
        algo_defaults = self._algo_step.controller_defaults.step_controller
        for key, value in algo_defaults.items():
            if key not in updates_dict:
                updates_dict[key] = value
        updates_dict["algorithm_order"] = self._algo_step.order
        return {"algorithm"}

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

    def _switch_controllers(self, updates_dict):
        """Replace the step controller when ``updates_dict`` contains a
        new ``"step_controller"`` key.

        Parameters
        ----------
        updates_dict
            Mutable mapping of pending updates.  Modified in-place to
            normalise the controller name.

        Returns
        -------
        set of str
            ``{"step_controller"}`` if a swap occurred, otherwise empty.
        """
        if "step_controller" not in updates_dict:
            return set()
        precision = updates_dict.get('precision', self.precision)

        new_controller = updates_dict.get("step_controller").lower()

        if new_controller != self.compile_settings.step_controller:
            buffer_registry.clear_parent(self._step_controller)
            old_settings = self._step_controller.settings_dict
            old_settings["step_controller"] = new_controller
            old_settings["algorithm_order"] = updates_dict.get(
                "algorithm_order", self._algo_step.order)
            self._step_controller = get_controller(
                    precision=precision,
                    settings=old_settings,
            )
            self.update_compile_settings(
                step_controller=new_controller
            )
        updates_dict["step_controller"] = new_controller
        return {"step_controller"}

    def set_solver_helper_fn(self, solver_helper_fn: Callable) -> None:
        """Replace the helper getter wired into the algorithm step.

        Parameters
        ----------
        solver_helper_fn
            Replacement getter with the ``get_solver_helper``
            contract.

        Notes
        -----
        Helper requests only fire during builds, so the replacement
        is picked up at the next build with no extra invalidation.
        """
        self._solver_helper_fn = solver_helper_fn

    def build(self) -> SingleIntegratorRunCache:
        """Compile the integration loop and its dependencies.

        Returns
        -------
        SingleIntegratorRunCache
            Cache containing the compiled loop device function.
        """

        # Lowest level - check for changes in evaluate_f, get_solver_helper_fn
        evaluate_f = self._system.evaluate_f
        evaluate_observables = self._system.evaluate_observables
        get_solver_helper_fn = self._solver_helper_fn
        compiled_fns_dict = {}
        if evaluate_f != self._algo_step.evaluate_f:
            compiled_fns_dict["evaluate_f"] = evaluate_f
        if evaluate_observables != self._algo_step.evaluate_observables:
            compiled_fns_dict["evaluate_observables"] = evaluate_observables
        if get_solver_helper_fn != self._algo_step.get_solver_helper_fn:
            compiled_fns_dict['get_solver_helper_fn'] = get_solver_helper_fn

        #Build algorithm fn after change made
        self._algo_step.update(compiled_fns_dict)

        # Building the step and controller functions must precede the
        # child-buffer registration below: an implicit step refreshes
        # its nested solver buffer sizes during build_step, so a size
        # snapshot taken before the build undersizes the loop's pool.
        compiled_functions = {
            'save_state_fn': self._output_functions.save_state_func,
            'update_summaries_fn': self._output_functions.update_summaries_func,
            'save_summaries_fn': self._output_functions.save_summary_metrics_func,
            'step_controller_fn': self._step_controller.device_function,
            'step_function': self._algo_step.step_function,
            'evaluate_observables': evaluate_observables}

        # Re-register algo and controller buffers to refresh sizing in loop
        buffer_registry.register_child(
                self._loop, self._algo_step, name='algorithm'
        )
        buffer_registry.register_child(
                self._loop, self._step_controller, name='controller'
        )

        self._loop.update(compiled_functions)
        loop_fn = self._loop.device_function

        return SingleIntegratorRunCache(single_integrator_function=loop_fn)

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
