"""Read-only property aggregation for single integrator runs.

Published Classes
-----------------
:class:`SingleIntegratorRun`
    Subclass of
    :class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`
    that exposes compiled loop artifacts, controller settings, and
    algorithm step metadata as read-only properties.

See Also
--------
:class:`~cubie.integrators.SingleIntegratorRunCore.SingleIntegratorRunCore`
    Parent class providing initialisation, update, and compilation.
:class:`~cubie.batchsolving.BatchSolverKernel.BatchSolverKernel`
    Primary consumer of these properties.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

from numpy import (
    dtype as np_dtype,
    finfo as np_finfo,
    float64 as np_float64,
    floating,
    floor as np_floor,
)

from cubie.integrators.SingleIntegratorRunCore import (
    SingleIntegratorRunCore,
)
from cubie.odesystems.ODEData import SystemSizes


if TYPE_CHECKING:  # pragma: no cover - type checking import only
    from cubie.odesystems.baseODE import BaseODE


class SingleIntegratorRun(SingleIntegratorRunCore):
    """Expose aggregated read-only properties for integrator runs.

    Notes
    -----
    Instantiation, updates, and compilation are provided by
    :class:`SingleIntegratorRunCore`. This subclass intentionally limits
    itself to property-based access so that front-end utilities can inspect
    compiled CUDA components without mutating internal state.
    """

    # ------------------------------------------------------------------
    # Compile settings
    # ------------------------------------------------------------------
    @property
    def algorithm(self) -> str:
        """Return the configured algorithm identifier."""

        return self.compile_settings.algorithm

    @property
    def step_controller(self) -> str:
        """Return the configured step-controller identifier."""

        return self.compile_settings.step_controller

    # ------------------------------------------------------------------
    # Aggregated memory usage
    # ------------------------------------------------------------------
    @property
    def shared_memory_elements(self) -> int:
        """Return total shared-memory elements required by the loop."""
        return self._loop.shared_buffer_size

    @property
    def shared_memory_bytes(self) -> int:
        """Return total shared-memory usage in bytes."""

        element_count = self.shared_memory_elements
        itemsize = np_dtype(self.precision).itemsize
        return element_count * itemsize

    @property
    def persistent_local_elements(self) -> int:
        """Return total persistent local-memory elements required by the loop.
        """
        return self._loop.persistent_local_buffer_size

    @property
    def dt_min(self) -> float:
        """Return the minimum allowable step size."""

        return self._step_controller.dt_min

    @property
    def dt_max(self) -> float:
        """Return the maximum allowable step size."""

        return self._step_controller.dt_max

    @property
    def is_adaptive(self) -> bool:
        """Return whether adaptive stepping is active."""

        return self._step_controller.is_adaptive

    @property
    def system(self) -> "BaseODE":
        """Return the underlying ODE system."""

        return self._system

    @property
    def system_sizes(self) -> SystemSizes:
        """Return the size descriptor for the ODE system."""

        return self._system.sizes

    @property
    def save_summaries_func(self) -> Callable:
        """Return the summary saving function from the output handlers."""

        return self.save_summary_metrics_func

    @property
    def evaluate_f(self) -> Callable:
        """Return the derivative function used by the integration step."""

        return self._algo_step.evaluate_f

    # ------------------------------------------------------------------
    # Loop properties
    # ------------------------------------------------------------------
    @property
    def save_every(self) -> Optional[float]:
        """Return the loop save interval, or None if not configured."""

        return self._loop.save_every

    @property
    def summarise_every(self) -> Optional[float]:
        """Return the loop summary interval, or None if not configured."""

        return self._loop.summarise_every

    @property
    def sample_summaries_every(self) -> Optional[float]:
        """Return the loop sample summaries interval, or None if not
        configured.
        """

        return self._loop.sample_summaries_every

    @property
    def save_last(self) -> bool:
        """Return True if end-of-run-only state saving is configured."""
        return self._loop.compile_settings.save_last

    def _regular_event_count(self, duration: float, interval: float) -> int:
        """Count how many scheduled events fit inside a duration.

        Casting ``duration`` and ``interval`` to the working
        precision rounds each of them slightly, so the division can
        land just below a whole number when the user asked for a
        whole number of events: float32 turns 10.0 / 0.001 into
        9999.9993, which would floor to 9999 and lose the event at
        the end time. A small allowance is added before flooring so
        a result this close to a whole number counts as that whole
        number.

        The allowance covers only the one-off rounding of the two
        cast values. Drift that builds up on the device as it
        repeatedly adds ``interval`` is handled separately, by the
        half-interval margin in the stop times. The allocation and
        the stop times both come from this count, so the host and
        the device always agree with each other; see
        :meth:`save_stop_time` and :meth:`summary_stop_time`.

        Parameters
        ----------
        duration
            Integration duration in time units.
        interval
            Scheduling interval in time units.

        Returns
        -------
        int
            Number of scheduled events, excluding the initial sample.
        """
        precision = self.precision
        total_events = np_float64(precision(duration)) / np_float64(
            precision(interval)
        )
        # Each cast moves its value by at most half a relative eps,
        # so the ratio is off by at most about one eps of itself;
        # allow four of them for headroom. The cap keeps the
        # allowance below one half so a deliberately fractional
        # duration (say 10.6 intervals) never gains an event. The
        # cap engages beyond ~1e6 events in float32 (5e14 in
        # float64), where the input rounding alone is worth a whole
        # event and the count can be off by one in either
        # direction; host and device still share whatever count
        # this returns.
        allowance = min(
            4.0 * float(np_finfo(precision).eps) * total_events, 0.49
        )
        return int(np_floor(total_events + allowance))

    def output_length(self, duration: float) -> int:
        """Calculate number of time-domain output samples for a duration.

        Parameters
        ----------
        duration
            Integration duration in time units.

        Returns
        -------
        int
            Number of output samples including initial and optionally final.
        """
        save_every = self.save_every

        regular_samples = 0
        final_samples = 1 if self.save_last else 0
        initial_sample = 1
        if save_every is not None:
            regular_samples = self._regular_event_count(
                duration, save_every
            )
        return regular_samples + initial_sample + final_samples

    def save_stop_time(
        self, duration: float, settling_time: float, t0: float
    ) -> floating:
        """Return the time when the regular save schedule is done.

        The device stops saving once its accumulated schedule
        passes this time. The stop sits half an interval past the
        last counted save, so a device schedule running slightly
        late still reaches its last save, and one running slightly
        early cannot squeeze in an extra one. Without a regular
        save schedule the stop is the end time.

        Parameters
        ----------
        duration
            Integration duration in time units.
        settling_time
            Lead-in time before samples are collected.
        t0
            Initial integration time.

        Returns
        -------
        floating
            Save-schedule stop time in the run precision.
        """
        start = np_float64(settling_time) + np_float64(t0)
        save_every = self.save_every
        if save_every is None:
            return self.precision(
                start + np_float64(self.precision(duration))
            )
        events = self._regular_event_count(duration, save_every)
        return self.precision(
            start + (events + 0.5) * np_float64(save_every)
        )

    def summary_stop_time(
        self, duration: float, settling_time: float, t0: float
    ) -> floating:
        """Return the time when the summary-update schedule is done.

        The device stops taking summary measurements once its
        accumulated schedule passes this time. The stop sits half a
        sampling interval past the last counted measurement, for
        the same reasons as :meth:`save_stop_time`. Without a
        summary schedule the stop is the end time.

        Parameters
        ----------
        duration
            Integration duration in time units.
        settling_time
            Lead-in time before samples are collected.
        t0
            Initial integration time.

        Returns
        -------
        floating
            Summary-schedule stop time in the run precision.
        """
        start = np_float64(settling_time) + np_float64(t0)
        sample_every = self.sample_summaries_every
        if sample_every is None:
            return self.precision(
                start + np_float64(self.precision(duration))
            )
        events = self._regular_event_count(duration, sample_every)
        return self.precision(
            start + (events + 0.5) * np_float64(sample_every)
        )

    def summaries_length(self, duration: float) -> int:
        """Calculate number of summary output rows for a duration.

        The device writes one summary row after every
        ``samples_per_summary`` summary measurements. The number of
        measurements in a run follows the same counting rule as
        saves (:meth:`_regular_event_count`), and only a complete
        window produces a row, so the row count is the measurement
        count divided by the measurements per window, rounded down.
        The measurements-per-window value is read from the loop
        configuration, the same place the device code gets it.

        Parameters
        ----------
        duration
            Integration duration in time units.

        Returns
        -------
        int
            Number of summary rows.
        """
        summarise_every = self.summarise_every

        regular_summaries = 0
        if summarise_every is not None:
            sample_every = self.sample_summaries_every
            updates = self._regular_event_count(duration, sample_every)
            samples_per_summary = (
                self._loop.compile_settings.samples_per_summary
            )
            regular_summaries = updates // max(samples_per_summary, 1)
        return regular_summaries

    @property
    def compile_flags(self) -> Any:
        """Return loop compile flags."""

        return self._loop.compile_flags

    @property
    def save_state_fn(self) -> Callable:
        """Return the loop state-save function."""

        return self._loop.save_state_fn

    @property
    def update_summaries_fn(self) -> Callable:
        """Return the loop summary-update function."""

        return self._loop.update_summaries_fn

    @property
    def save_summaries_fn(self) -> Callable:
        """Return the loop summary-save function."""

        return self._loop.save_summaries_fn

    # ------------------------------------------------------------------
    # Step controller properties
    # ------------------------------------------------------------------
    @property
    def atol(self) -> Optional[Any]:
        """Return the absolute tolerance array."""

        controller = self._step_controller
        return controller.atol if hasattr(controller, "atol") else None

    @property
    def rtol(self) -> Optional[Any]:
        """Return the relative tolerance array."""

        controller = self._step_controller
        return controller.rtol if hasattr(controller, "rtol") else None

    @property
    def dt(self) -> Optional[float]:
        """Return the fixed step size for fixed controllers."""
        controller = self._step_controller
        return controller.dt if hasattr(controller, "dt") else None

    # ------------------------------------------------------------------
    # Algorithm step properties
    # ------------------------------------------------------------------
    @property
    def threads_per_step(self) -> int:
        """Return the number of threads required by the step function."""

        return self._algo_step.threads_per_step

    # ------------------------------------------------------------------
    # Output function properties
    # ------------------------------------------------------------------
    @property
    def save_state_func(self) -> Callable:
        """Return the compiled state saving function."""

        return self._output_functions.save_state_func

    @property
    def update_summaries_func(self) -> Callable:
        """Return the compiled summary update function."""

        return self._output_functions.update_summaries_func

    @property
    def save_summary_metrics_func(self) -> Callable:
        """Return the compiled summary saving function."""

        return self._output_functions.save_summary_metrics_func

    @property
    def output_types(self) -> Any:
        """Return the configured output types."""

        return self._output_functions.output_types

    @property
    def output_compile_flags(self) -> Any:
        """Return the output compile flags."""

        return self._output_functions.compile_flags

    @property
    def save_time(self) -> bool:
        """Return whether time saving is enabled."""

        return self._output_functions.save_time

    @property
    def save_counters(self) -> bool:
        """Return whether iteration-counter saving is enabled."""

        return self._output_functions.save_counters

    @property
    def saved_state_indices(self) -> Any:
        """Return the saved state indices."""

        return self._output_functions.saved_state_indices

    @property
    def saved_observable_indices(self) -> Any:
        """Return the saved observable indices."""

        return self._output_functions.saved_observable_indices

    @property
    def summarised_state_indices(self) -> Any:
        """Return the summarised state indices."""

        return self._output_functions.summarised_state_indices

    @property
    def summarised_observable_indices(self) -> Any:
        """Return the summarised observable indices."""

        return self._output_functions.summarised_observable_indices

    @property
    def output_array_heights(self) -> Any:
        """Return the output array height descriptor."""

        return self._output_functions.output_array_heights

    @property
    def summary_legend_per_variable(self) -> Any:
        """Return the summary legend per variable."""

        return self._output_functions.summary_legend_per_variable

    @property
    def summary_unit_modifications(self) -> Any:
        """Return the summary unit modifications."""

        return self._output_functions.summary_unit_modifications


__all__ = ["SingleIntegratorRun"]
