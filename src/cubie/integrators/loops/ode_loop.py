"""Outer integration loop factory for CUDA-based ODE solvers.

Published Classes
-----------------
:class:`IVPLoopCache`
    Attrs cache container for the compiled loop device function.

:class:`IVPLoop`
    CUDAFactory that registers loop buffers and compiles a CUDA device
    function coordinating step, controller, save, and summary callbacks.

    >>> from numpy import float32
    >>> from cubie.outputhandling.output_config import OutputCompileFlags
    >>> loop = IVPLoop(
    ...     precision=float32, n_states=4,
    ...     compile_flags=OutputCompileFlags(),
    ... )
    >>> loop.save_every is None
    True

Constants
---------
:data:`ALL_LOOP_SETTINGS`
    Set of recognised loop configuration parameter names accepted by
    :class:`IVPLoop` and forwarded through ``**kwargs``.

See Also
--------
:class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
    Configuration container consumed by this factory.
:class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`
    Parent coordinator that owns and configures the loop.
:class:`~cubie.CUDAFactory.CUDAFactory`
    Base factory class providing compilation and caching.
"""

from typing import Callable, Optional, Set

from attrs import define, field
from numpy import int32 as np_int32
from cubie.cuda_simsafe import cuda, int32, float32, float64, bool_
from cubie.cuda_simsafe import unroll_if

from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import activemask, all_sync, narrow_f64, selp
from cubie.result_codes import CUBIE_RESULT_CODES
from cubie._utils import PrecisionDType, unpack_dict_values, build_config
from cubie.integrators.loops.ode_loop_config import ODELoopConfig
from cubie.outputhandling import OutputCompileFlags


@define
class IVPLoopCache(CUDADispatcherCache):
    """Cache for IVP loop device function.

    Attributes
    ----------
    loop_function
        Compiled CUDA device function that executes the integration loop.
    """

    loop_function: Callable = field()


ALL_LOOP_SETTINGS = {
    "save_every",
    "summarise_every",
    "sample_summaries_every",
    "dt",
    "is_adaptive",
    "save_last",
    "save_regularly",
    "summarise_regularly",
    "state_location",
    "proposed_state_location",
    "parameters_location",
    "drivers_location",
    "proposed_drivers_location",
    "observables_location",
    "proposed_observables_location",
    "error_location",
    "counters_location",
    "state_summary_location",
    "observable_summary_location",
    "dt_location",
    "accept_step_location",
    "proposed_counters_location",
}
"""Compile-critical loop configuration parameters accepted by
:class:`IVPLoop`.

These parameters can be passed as keyword arguments to :class:`IVPLoop`
or to :meth:`IVPLoop.update`.  Parent components use this set to filter
``**kwargs`` before forwarding.

.. list-table:: Parameter Summary
   :header-rows: 1

   * - Parameter
     - Accepted By
     - Description
   * - ``save_every``
     - :class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
     - Interval between accepted state saves.
   * - ``summarise_every``
     - :class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
     - Interval between summary accumulations.
   * - ``sample_summaries_every``
     - :class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
     - Interval between summary metric updates.
   * - ``dt``
     - :class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
     - Initial timestep.
   * - ``is_adaptive``
     - :class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
     - Whether the loop uses adaptive stepping.
   * - ``save_last``
     - :class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
     - Save the final state regardless of interval alignment.
   * - ``save_regularly``
     - :class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
     - Enable periodic state saving.
   * - ``summarise_regularly``
     - :class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
     - Enable periodic summary accumulation.
   * - ``state_location`` … ``proposed_counters_location``
     - :class:`~cubie.integrators.loops.ode_loop_config.ODELoopConfig`
     - Memory location (``'local'`` or ``'shared'``) for each buffer.
"""


class IVPLoop(CUDAFactory):
    """Factory for CUDA device loops that advance an IVP integration.

    Parameters
    ----------
    precision
        Precision used for state and observable updates.
    n_states
        Number of state variables.
    compile_flags
        Output configuration that drives save and summary behaviour.
    n_parameters
        Number of parameters.
    n_drivers
        Number of driver variables.
    n_observables
        Number of observable variables.
    n_error
        Number of error elements (typically equals n_states for adaptive).
    n_counters
        Number of counter elements.
    state_summaries_buffer_height
        Height of state summary buffer.
    observable_summaries_buffer_height
        Height of observable summary buffer.
    save_every
        Interval between accepted saves. Defaults to None (auto-configured).
    summarise_every
        Interval between summary accumulations. Defaults to None
        (auto-configured).
    sample_summaries_every
        Interval between summary metric updates. Must be an integer divisor
        of ``summarise_every``. Defaults to None (auto-configured).
    save_state_func
        Device function that writes state and observable snapshots.
    update_summaries_func
        Device function that accumulates summary statistics.
    save_summaries_func
        Device function that commits summary statistics to output buffers.
    step_controller_fn
        Device function that updates the timestep and accept flag.
    step_function
        Device function that advances the solution by one tentative step.
    evaluate_driver_at_t
        Device function that evaluates drivers for a given time.
    evaluate_observables
        Device function that computes observables for proposed states.
    **kwargs
        Optional parameters passed to ODELoopConfig. Available parameters
        include dt, is_adaptive, and buffer location
        parameters (state_location, proposed_state_location,
        parameters_location, drivers_location, proposed_drivers_location,
        observables_location, proposed_observables_location, error_location,
        counters_location, state_summary_location,
        observable_summary_location, dt_location, accept_step_location).
        None values are ignored.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n_states: int,
        compile_flags: OutputCompileFlags,
        n_parameters: int = 0,
        n_drivers: int = 0,
        n_observables: int = 0,
        n_error: int = 0,
        n_counters: int = 0,
        state_summaries_buffer_height: int = 0,
        observable_summaries_buffer_height: int = 0,
        save_every: Optional[float] = None,
        summarise_every: Optional[float] = None,
        sample_summaries_every: Optional[float] = None,
        save_state_func: Optional[Callable] = None,
        update_summaries_func: Optional[Callable] = None,
        save_summaries_func: Optional[Callable] = None,
        step_controller_fn: Optional[Callable] = None,
        step_function: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """Initialise the IVP loop configuration.

        Parameters
        ----------
        precision
            Precision used for state and observable updates.
        n_states
            Number of state variables.
        compile_flags
            Output configuration that drives save and summary behaviour.
        n_parameters
            Number of parameters.
        n_drivers
            Number of driver variables.
        n_observables
            Number of observable variables.
        n_error
            Number of error elements (typically equals n_states for adaptive).
        n_counters
            Number of counter elements.
        state_summaries_buffer_height
            Height of state summary buffer.
        observable_summaries_buffer_height
            Height of observable summary buffer.
        save_every
            Interval between accepted saves. Defaults to None
            (auto-configured).
        summarise_every
            Interval between summary accumulations. Defaults to None
            (auto-configured).
        sample_summaries_every
            Interval between summary metric updates. Must be an integer divisor
            of ``summarise_every``. Defaults to None (auto-configured).
        save_state_func
            Device function that writes state and observable snapshots.
        update_summaries_func
            Device function that accumulates summary statistics.
        save_summaries_func
            Device function that commits summary statistics to output buffers.
        step_controller_fn
            Device function that updates the timestep and accept flag.
        step_function
            Device function that advances the solution by one tentative step.
        evaluate_driver_at_t
            Device function that evaluates drivers for a given time.
        evaluate_observables
            Device function that computes observables for proposed states.
        **kwargs
            Optional parameters passed to ODELoopConfig. See ODELoopConfig
            for available parameters including dt, is_adaptive, and buffer
            location parameters (state_location, proposed_state_location,
            etc.). None values are ignored.
        """
        super().__init__()

        config = build_config(
            ODELoopConfig,
            required={
                "n_states": n_states,
                "n_parameters": n_parameters,
                "n_drivers": n_drivers,
                "n_observables": n_observables,
                "n_error": n_error,
                "n_counters": n_counters,
                "state_summaries_buffer_height": state_summaries_buffer_height,
                "observable_summaries_buffer_height": (
                    observable_summaries_buffer_height
                ),
                "precision": precision,
                "compile_flags": compile_flags,
                "save_every": save_every,
                "summarise_every": summarise_every,
                "sample_summaries_every": sample_summaries_every,
                "save_state_fn": save_state_func,
                "update_summaries_fn": update_summaries_func,
                "save_summaries_fn": save_summaries_func,
                "step_controller_fn": step_controller_fn,
                "step_function": step_function,
                "evaluate_driver_at_t": evaluate_driver_at_t,
                "evaluate_observables": evaluate_observables,
            },
            **kwargs,
        )
        self.setup_compile_settings(config)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        config = self.compile_settings
        n_states = config.n_states
        n_parameters = config.n_parameters
        n_drivers = config.n_drivers
        n_observables = config.n_observables
        n_error = config.n_error
        n_counters = config.n_counters
        state_summaries_buffer_height = config.state_summaries_buffer_height
        observable_summaries_buffer_height = (
            config.observable_summaries_buffer_height
        )

        # Register all loop buffers with central registry

        buffer_registry.register(
            "state", self, n_states, config.state_location
        )
        buffer_registry.register(
            "proposed_state",
            self,
            n_states,
            config.proposed_state_location,
        )
        buffer_registry.register(
            "parameters",
            self,
            n_parameters,
            config.parameters_location,
        )
        buffer_registry.register(
            "drivers",
            self,
            n_drivers,
            config.drivers_location,
        )
        buffer_registry.register(
            "proposed_drivers",
            self,
            n_drivers,
            config.proposed_drivers_location,
        )
        buffer_registry.register(
            "observables",
            self,
            n_observables,
            config.observables_location,
        )
        buffer_registry.register(
            "proposed_observables",
            self,
            n_observables,
            config.proposed_observables_location,
        )
        buffer_registry.register(
            "error", self, n_error, config.error_location
        )
        buffer_registry.register(
            "counters",
            self,
            n_counters,
            config.counters_location,
            dtype=np_int32,
        )
        buffer_registry.register(
            "state_summary",
            self,
            state_summaries_buffer_height,
            config.state_summary_location,
        )
        buffer_registry.register(
            "observable_summary",
            self,
            observable_summaries_buffer_height,
            config.observable_summary_location,
        )
        buffer_registry.register(
            "dt", self, 1, config.dt_location
        )
        buffer_registry.register(
            "accept_step",
            self,
            1,
            config.accept_step_location,
            dtype=np_int32,
        )
        buffer_registry.register(
            "proposed_counters",
            self,
            2,
            config.proposed_counters_location,
            dtype=np_int32,
        )

    def build(self) -> IVPLoopCache:
        """Compile the CUDA device loop.

        Returns
        -------
        IVPLoopCache
            Cache containing the compiled loop device function.
        """
        config = self.compile_settings

        precision = config.numba_precision
        # narrow_f64 skips the ftz subnormal guard on the t narrowing.
        narrow_time = narrow_f64 if precision == float32 else precision

        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        step_too_small = int32(CUBIE_RESULT_CODES.STEP_TOO_SMALL)
        stagnation = int32(CUBIE_RESULT_CODES.STAGNATION)

        save_state = config.save_state_fn
        update_summaries = config.update_summaries_fn
        save_summaries = config.save_summaries_fn
        step_controller = config.step_controller_fn
        step_function = config.step_function
        evaluate_driver_at_t = config.evaluate_driver_at_t
        evaluate_observables = config.evaluate_observables
        initialise_state = config.initialise_state_fn

        flags = config.compile_flags
        save_obs_bool = flags.save_observables
        save_state_bool = flags.save_state
        summarise_obs_bool = flags.summarise_observables
        summarise_state_bool = flags.summarise_state
        summarise = summarise_obs_bool or summarise_state_bool
        save_counters_bool = flags.save_counters

        # Get allocators from buffer registry
        getalloc = buffer_registry.get_allocator
        alloc_state = getalloc("state", self, zero=True)
        alloc_proposed_state = getalloc("proposed_state", self, zero=True)
        alloc_parameters = getalloc("parameters", self, zero=True)
        alloc_drivers = getalloc("drivers", self, zero=True)
        alloc_proposed_drivers = getalloc("proposed_drivers", self, zero=True)
        alloc_observables = getalloc("observables", self, zero=True)
        alloc_proposed_observables = getalloc(
            "proposed_observables", self, zero=True
        )
        alloc_error = getalloc("error", self, zero=True)
        alloc_counters = getalloc("counters", self, zero=True)
        alloc_state_summary = getalloc("state_summary", self, zero=True)
        alloc_observable_summary = getalloc(
            "observable_summary", self, zero=True
        )
        alloc_algo_shared = getalloc("algorithm_shared", self, zero=True)
        alloc_algo_persistent = getalloc(
            "algorithm_persistent", self, zero=True
        )
        alloc_controller_shared = getalloc(
            "controller_shared", self, zero=True
        )
        alloc_controller_persistent = getalloc(
            "controller_persistent", self, zero=True
        )
        alloc_dt = getalloc("dt", self, zero=True)
        alloc_accept_step = getalloc("accept_step", self, zero=True)
        alloc_proposed_counters = getalloc("proposed_counters", self)
        alloc_initialiser_shared = getalloc(
            "initialiser_shared", self, zero=True
        )
        alloc_initialiser_persistent = getalloc(
            "initialiser_persistent", self, zero=True
        )
        n_shared = int32(self.shared_buffer_size)
        n_persistent_local = int32(self.persistent_local_buffer_size)

        # Timing values
        initial_dt = precision(config.dt)
        typed_zero = precision(0.0)
        save_every = config.save_every
        sample_summaries_every = config.sample_summaries_every
        samples_per_summary = int32(config.samples_per_summary)

        # Boolean control-flow constants
        save_last = config.save_last
        save_regularly = config.save_regularly
        summarise_regularly = config.summarise_regularly

        # Loop sizes from config (sizes also used for iteration bounds)
        n_states = int32(config.n_states)
        unroll_step_element = config.unroll_step_element
        unroll_other_small = config.unroll_other_small
        n_parameters = int32(config.n_parameters)
        n_observables = int32(config.n_observables)
        n_drivers = int32(config.n_drivers)
        n_counters = int32(config.n_counters)
        n_error = int32(config.n_error)

        fixed_mode = not config.is_adaptive

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def loop_fn(
            initial_states,
            parameters,
            driver_coefficients,
            shared_scratch,
            persistent_local,
            state_output,
            observables_output,
            state_summaries_output,
            observable_summaries_output,
            iteration_counters_output,
            duration,
            settling_time,
            t0,
            save_stop,
            summary_stop,
        ):  # pragma: no cover - CUDA fns not marked in coverage
            """Advance an integration using a compiled CUDA device loop.

            The loop terminates when every output schedule passes its
            stop time, or when the maximum number of iterations is
            reached.

            Parameters
            ----------
            initial_states
                1d Device array containing the initial state vector.
            parameters
                1d Device array containing static parameters.
            driver_coefficients
                3d Device array containing precomputed spline coefficients.
            shared_scratch
                1d Device array providing shared-memory work buffers.
            persistent_local
                1d Device array providing persistent local memory buffers.
            state_output
                2d Device array storing accepted state snapshots.
            observables_output
                2d Device array storing accepted observable snapshots.
            state_summaries_output
                Device array storing aggregated state summaries.
            observable_summaries_output
                Device array storing aggregated observable summaries.
            iteration_counters_output
                Device array storing iteration counter values at each save.
            duration
                Total integration duration.
            settling_time
                Lead-in time before samples are collected.
            t0
                Initial integration time.
            save_stop
                Time half a save interval past the final scheduled
                save event; the save schedule is complete once
                ``next_save`` exceeds it. Computed host-side with
                the same arithmetic as the output allocation
                (:meth:`SingleIntegratorRun.save_stop_time`).
            summary_stop
                Time half a sample interval past the final
                scheduled summary-update event; the summary
                schedule is complete once ``next_update_summary``
                exceeds it
                (:meth:`SingleIntegratorRun.summary_stop_time`).

            Returns
            -------
            int
                Status code aggregating errors and iteration counts.
            """
            t = float64(t0)
            t_prec = narrow_time(t)
            t_end = precision(settling_time + t0 + duration)

            # Clear inherited arrays on entry
            for i in unroll_if(range(n_persistent_local), unroll_other_small):
                persistent_local[i] = typed_zero
            for i in unroll_if(range(n_shared), unroll_other_small):
                shared_scratch[i] = typed_zero
            # ----------------------------------------------------------- #
            # Allocate buffers using registry allocators
            # ----------------------------------------------------------- #
            state_buffer = alloc_state(shared_scratch, persistent_local)
            state_proposal_buffer = alloc_proposed_state(
                shared_scratch, persistent_local
            )
            observables_buffer = alloc_observables(
                shared_scratch, persistent_local
            )
            observables_proposal_buffer = alloc_proposed_observables(
                shared_scratch, persistent_local
            )
            parameters_buffer = alloc_parameters(
                shared_scratch, persistent_local
            )
            drivers_buffer = alloc_drivers(shared_scratch, persistent_local)
            drivers_proposal_buffer = alloc_proposed_drivers(
                shared_scratch, persistent_local
            )
            state_summary_buffer = alloc_state_summary(
                shared_scratch, persistent_local
            )
            observable_summary_buffer = alloc_observable_summary(
                shared_scratch, persistent_local
            )
            counters_since_save = alloc_counters(
                shared_scratch, persistent_local
            )
            error = alloc_error(shared_scratch, persistent_local)

            # Allocate child buffers for algorithm step
            algo_shared = alloc_algo_shared(shared_scratch, persistent_local)
            algo_persistent = alloc_algo_persistent(
                shared_scratch, persistent_local
            )
            ctrl_shared = alloc_controller_shared(
                shared_scratch, persistent_local
            )
            ctrl_persistent = alloc_controller_persistent(
                shared_scratch, persistent_local
            )
            dt = alloc_dt(shared_scratch, persistent_local)
            accept_step = alloc_accept_step(shared_scratch, persistent_local)

            proposed_counters = alloc_proposed_counters(
                shared_scratch, persistent_local
            )
            initialiser_shared = alloc_initialiser_shared(
                shared_scratch, persistent_local
            )
            initialiser_persistent = alloc_initialiser_persistent(
                shared_scratch, persistent_local
            )
            # --------------------------------------------------------------- #

            first_step_flag = True
            prev_step_accepted_flag = True
            stagnant_counts = int32(0)
            save_idx = int32(0)
            summary_idx = int32(0)
            update_idx = int32(0)
            next_save = precision(settling_time + t0)
            next_update_summary = precision(settling_time + t0)
            # --------------------------------------------------------------- #
            #                       Seed t=0 values                           #
            # --------------------------------------------------------------- #
            for k in unroll_if(range(n_states), unroll_other_small):
                state_buffer[k] = initial_states[k]
            for k in unroll_if(range(n_parameters), unroll_other_small):
                parameters_buffer[k] = parameters[k]

            # Seed initial observables from initial state.
            if evaluate_driver_at_t is not None and n_drivers > int32(0):
                evaluate_driver_at_t(
                    t_prec,
                    driver_coefficients,
                    drivers_buffer,
                )

            # Solve for a consistent DAE start before the t0 save.
            proposed_counters[0] = int32(0)
            proposed_counters[1] = int32(0)
            init_status = initialise_state(
                state_buffer,
                parameters_buffer,
                drivers_buffer,
                t_prec,
                initial_dt,
                initialiser_shared,
                initialiser_persistent,
                proposed_counters,
            )
            init_failed = bool_(init_status != int32(0))

            # The initialiser's iterations land in the t0 save row.
            if save_counters_bool:
                for i in unroll_if(range(n_counters), unroll_other_small):
                    if i < int32(2):
                        counters_since_save[i] += proposed_counters[i]

            if n_observables > int32(0):
                evaluate_observables(
                    state_buffer,
                    parameters_buffer,
                    drivers_buffer,
                    observables_buffer,
                    t_prec,
                )

            # Set next save for `settling_time`, or save first value if
            # starting at t0
            if settling_time == 0.0:
                # Save initial state at t0, then advance to first interval save
                if save_regularly:
                    next_save = precision(next_save + save_every)
                if summarise_regularly:
                    next_update_summary = precision(
                        sample_summaries_every + next_update_summary
                    )

                save_state(
                    state_buffer,
                    observables_buffer,
                    counters_since_save,
                    t_prec,
                    state_output[save_idx * save_state_bool, :],
                    observables_output[save_idx * save_obs_bool, :],
                    iteration_counters_output[
                        save_idx * save_counters_bool, :
                    ],
                )
                save_idx += int32(1)

                # Call save_summaries only to reset buffer values
                if summarise:
                    statesumm_idx = summary_idx * summarise_state_bool
                    obsumm_idx = summary_idx * summarise_obs_bool
                    save_summaries(
                        state_summary_buffer,
                        observable_summary_buffer,
                        state_summaries_output[statesumm_idx, :],
                        observable_summaries_output[obsumm_idx, :],
                        samples_per_summary,
                    )

            status = int32(success | init_status)
            iteration_status = int32(0)
            dt[0] = initial_dt
            dt_raw = initial_dt
            accept_step[0] = int32(0)

            # Initialize iteration counters
            for i in unroll_if(range(n_counters), unroll_other_small):
                counters_since_save[i] = int32(0)
                if i < int32(2):
                    proposed_counters[i] = int32(0)

            mask = activemask()
            # A failed initialisation ends the run at the t0 save.
            irrecoverable = init_failed
            at_end = False
            save_finished = False
            summary_finished = False
            # --------------------------------------------------------------- #
            #                        Main Loop                                #
            # --------------------------------------------------------------- #
            while True:
                # ----------------------------------------------------------- #
                #               Events due - end, update, save                #
                # ----------------------------------------------------------- #
                # Compile-time branching: save_regularly and
                # summarise_regularly are constants, allowing Numba to
                # eliminate dead branches
                end_of_step = t_prec + dt_raw
                if save_regularly or summarise_regularly:
                    # Loop continues until all scheduled outputs are
                    # complete. Each stop time sits half an interval
                    # past that schedule's last event, so a schedule
                    # running slightly late still fires its last
                    # event, and one running slightly early cannot
                    # fire an extra one.
                    finished = True
                    if save_regularly:
                        save_finished = bool_(next_save > save_stop)
                        finished &= save_finished
                    if summarise_regularly:
                        summary_finished = bool_(
                            next_update_summary > summary_stop
                        )
                        finished &= summary_finished
                else:
                    # No scheduled outputs; finish when time reaches t_end.
                    # >= keeps a step that lands exactly on t_end inside the
                    # save_last window below.
                    finished = bool_(end_of_step >= t_end)

                if save_last:
                    # Save final state even if not aligned with save_every
                    # at_end triggers when we're in the last step before t_end
                    at_end = bool_(t_prec < t_end) & finished
                    finished = finished & ~at_end

                finished = finished or irrecoverable

                if all_sync(mask, finished):
                    return status

                if not finished:
                    # Determine output actions for this step
                    # Compile-time constants enable branch elimination
                    if save_regularly:
                        do_save = (
                            bool_(end_of_step >= next_save) & ~save_finished
                        )
                    else:
                        do_save = False

                    if summarise_regularly:
                        do_update_summary = (
                            bool_(end_of_step >= next_update_summary)
                            & ~summary_finished
                        )
                    else:
                        do_update_summary = False

                    if save_last:
                        do_save |= at_end

                    # Adjust step size to hit output boundaries exactly.
                    # Only positive gaps clamp the step: downward
                    # accumulation drift or rounding can put the event
                    # time at or behind t_prec, and a non-positive
                    # dt_eff traps the loop. The due event fires at the
                    # next accepted step instead. truncated tells the
                    # controller the step length was schedule-forced.
                    dt_eff = dt_raw
                    truncated = False
                    if do_save or do_update_summary:
                        next_event = t_end
                        if do_save and save_regularly:
                            next_event = precision(min(next_event, next_save))
                        if do_update_summary and summarise_regularly:
                            next_event = precision(
                                min(next_event, next_update_summary)
                            )
                        gap = precision(next_event - t_prec)
                        dt_eff = selp(gap > typed_zero, gap, dt_raw)
                        truncated = bool_(dt_eff != dt_raw)

                    # ------------------------------------------------------- #
                    # Take a step
                    step_status = int32(
                        step_function(
                            state_buffer,
                            state_proposal_buffer,
                            parameters_buffer,
                            driver_coefficients,
                            drivers_buffer,
                            drivers_proposal_buffer,
                            observables_buffer,
                            observables_proposal_buffer,
                            error,
                            dt_eff,
                            t_prec,
                            first_step_flag,
                            prev_step_accepted_flag,
                            algo_shared,
                            algo_persistent,
                            proposed_counters,
                        )
                    )

                    # Convert times before the controller to hide
                    # f64 latency.
                    t_proposal = t + float64(dt_eff)
                    t_prec_proposal = narrow_time(t_proposal)
                    # Land the final save_last step exactly on t_end.
                    if save_last:
                        t_prec_proposal = selp(
                            at_end, t_end, t_prec_proposal
                        )
                    time_advances = bool_(t_proposal != t)

                    first_step_flag = False
                    niters = proposed_counters[0]
                    iteration_status = int32(iteration_status | step_status)

                    # A nonzero step status indicates step failure (e.g.,
                    # solver convergence failure). In adaptive mode this should
                    # reject the step and trigger a timestep reduction; in
                    # fixed mode it is irrecoverable.
                    step_failed = bool_(step_status != int32(0))
                    irrecoverable = bool_(
                        irrecoverable or (fixed_mode and step_failed)
                    )
                    for i in unroll_if(range(n_error), unroll_other_small):
                        error[i] = selp(step_failed, precision(1e16), error[i])

                    # Adjust dt based on calculated error if adaptive
                    if not fixed_mode:
                        controller_status = step_controller(
                            dt,
                            state_proposal_buffer,
                            state_buffer,
                            error,
                            niters,
                            truncated,
                            accept_step,
                            ctrl_shared,
                            ctrl_persistent,
                        )

                        accept = bool_(accept_step[0] != int32(0))
                        accept = bool_(accept and (not step_failed))
                        iteration_status = int32(
                            iteration_status | controller_status
                        )

                        # Controller may signal irrecoverable error via status
                        # bit
                        irrecoverable = bool_(
                            irrecoverable
                            or ((controller_status & step_too_small)
                                != success)
                        )
                    else:
                        accept = bool_(not step_failed)

                    dt_raw = dt[0]

                    # Accumulate iteration counters if active
                    if save_counters_bool:
                        for i in unroll_if(
                            range(n_counters), unroll_other_small
                        ):
                            if i < int32(2):
                                # Write newton, krylov iterations from buffer
                                counters_since_save[i] += proposed_counters[i]
                            elif i == int32(2):
                                # Increment total steps counter
                                counters_since_save[i] += int32(1)
                            elif not accept:
                                # Increment rejected steps counter
                                counters_since_save[i] += int32(1)

                    # test for stagnation - we might have one small step
                    # which doesn't nudge t if we're right up against a save
                    # boundary, so we call 2 stale t values in a row "stagnant"
                    stagnant_counts = selp(
                        time_advances,
                        int32(0),
                        int32(stagnant_counts + int32(1)),
                    )

                    stagnant = bool_(stagnant_counts >= int32(2))
                    iteration_status = selp(
                        stagnant,
                        int32(iteration_status | stagnation),
                        iteration_status,
                    )
                    irrecoverable = bool_(irrecoverable or stagnant)

                    # Fold the iteration's accumulated status bits into the
                    # persistent status word only when the run ends
                    # irrecoverably.  The accumulator is cleared whenever a
                    # step is accepted, so a save event (which requires an
                    # accepted step) delivers the cleared value and transient
                    # bits from rejected-then-recovered attempts never reach
                    # the persistent word.  The fatal iteration's bits are
                    # committed before the reset, preserving diagnosability.
                    status = selp(
                        irrecoverable,
                        int32(status | iteration_status),
                        status,
                    )
                    if accept:
                        iteration_status = int32(0)

                    t = selp(accept, t_proposal, t)
                    t_prec = selp(accept, t_prec_proposal, t_prec)

                    for i in unroll_if(range(n_states), unroll_step_element):
                        newv = state_proposal_buffer[i]
                        oldv = state_buffer[i]
                        state_buffer[i] = selp(accept, newv, oldv)

                    for i in unroll_if(range(n_drivers), unroll_step_element):
                        new_drv = drivers_proposal_buffer[i]
                        old_drv = drivers_buffer[i]
                        drivers_buffer[i] = selp(accept, new_drv, old_drv)

                    for i in unroll_if(
                        range(n_observables), unroll_step_element
                    ):
                        new_obs = observables_proposal_buffer[i]
                        old_obs = observables_buffer[i]
                        observables_buffer[i] = selp(accept, new_obs, old_obs)

                    prev_step_accepted_flag = selp(
                        accept,
                        int32(1),
                        int32(0),
                    )

                    # Predicated output execution: only perform outputs
                    # if step was accepted (avoids warp divergence)
                    do_save &= accept
                    do_update_summary &= accept

                    if do_save:
                        # Increment next_save if it's in use
                        if save_regularly:
                            next_save += save_every

                        save_state(
                            state_buffer,
                            observables_buffer,
                            counters_since_save,
                            t_prec,
                            state_output[save_idx * save_state_bool, :],
                            observables_output[save_idx * save_obs_bool, :],
                            iteration_counters_output[
                                save_idx * save_counters_bool, :
                            ],
                        )
                        save_idx += int32(1)

                        # Reset iteration counters after save
                        if save_counters_bool:
                            for i in unroll_if(
                                range(n_counters), unroll_other_small
                            ):
                                counters_since_save[i] = int32(0)

                    if do_update_summary:
                        if summarise_regularly:
                            next_update_summary += sample_summaries_every

                        if summarise:
                            statesumm_idx = summary_idx * summarise_state_bool
                            obssumm_idx = summary_idx * summarise_obs_bool
                            update_summaries(
                                state_buffer,
                                observables_buffer,
                                state_summary_buffer,
                                observable_summary_buffer,
                                update_idx,
                            )
                            update_idx += int32(1)

                            # Save summary when enough updates collected
                            if update_idx % samples_per_summary == int32(0):
                                save_summaries(
                                    state_summary_buffer,
                                    observable_summary_buffer,
                                    state_summaries_output[statesumm_idx, :],
                                    observable_summaries_output[
                                        obssumm_idx, :
                                    ],
                                    samples_per_summary,
                                )
                                summary_idx += int32(1)

        # no cover: end
        return IVPLoopCache(loop_function=loop_fn)

    @property
    def save_every(self) -> Optional[float]:
        """Return the save interval, or None if not configured."""
        return self.compile_settings.save_every

    @property
    def summarise_every(self) -> Optional[float]:
        """Return the summary interval, or None if not configured."""
        return self.compile_settings.summarise_every

    @property
    def sample_summaries_every(self) -> Optional[float]:
        """Return the summary sampling interval, or None if not configured."""
        return self.compile_settings.sample_summaries_every

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Return the output compile flags associated with the loop."""

        return self.compile_settings.compile_flags

    @property
    def device_function(self):
        """Return the compiled CUDA loop function.

        Returns
        -------
        callable
            Compiled CUDA device function.
        """
        return self.get_cached_output("loop_function")

    @property
    def save_state_fn(self) -> Optional[Callable]:
        """Return the cached state saving device function."""

        return self.compile_settings.save_state_fn

    @property
    def update_summaries_fn(self) -> Optional[Callable]:
        """Return the cached summary update device function."""

        return self.compile_settings.update_summaries_fn

    @property
    def save_summaries_fn(self) -> Optional[Callable]:
        """Return the cached summary saving device function."""

        return self.compile_settings.save_summaries_fn

    @property
    def step_controller_fn(self) -> Optional[Callable]:
        """Return the device function implementing step control."""

        return self.compile_settings.step_controller_fn

    @property
    def step_function(self) -> Optional[Callable]:
        """Return the algorithm step device function used by the loop."""

        return self.compile_settings.step_function

    @property
    def evaluate_driver_at_t(self) -> Optional[Callable]:
        """Return the driver evaluation device function used by the loop."""

        return self.compile_settings.evaluate_driver_at_t

    @property
    def evaluate_observables(self) -> Optional[Callable]:
        """Return the observables device function used by the loop."""

        return self.compile_settings.evaluate_observables

    @property
    def dt(self) -> Optional[float]:
        """Return the initial step size provided to the loop."""

        return self.compile_settings.dt

    @property
    def is_adaptive(self) -> Optional[bool]:
        """Return whether the loop operates in adaptive mode."""

        return self.compile_settings.is_adaptive

    def update(
        self,
        updates_dict: Optional[dict[str, object]] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> Set[str]:
        """Update compile settings through the CUDAFactory interface.

        Parameters
        ----------
        updates_dict
            Mapping of configuration names to replacement values.
        silent
            When True, suppress warnings about unrecognized parameters.
        **kwargs
            Additional configuration updates applied as keyword arguments.

        Returns
        -------
        set
            Set of parameter names that were recognized and updated.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        # Flatten nested dict values (e.g., loop_settings={'save_every': 0.01})
        # into top-level parameters before distributing to compile settings.
        # This ensures all configuration options are recognized and updated.
        # Example: {'loop_settings': {'save_every': 0.01}, 'other': 5}
        #       -> {'save_every': 0.01, 'other': 5}
        updates_dict, unpacked_keys = unpack_dict_values(updates_dict)

        recognised = self.update_compile_settings(updates_dict, silent=True)

        # Update buffer locations in registry
        recognised |= buffer_registry.update(self, updates_dict, silent=True)
        self.register_buffers()

        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        # Include unpacked dict keys in recognized set
        return recognised | unpacked_keys
