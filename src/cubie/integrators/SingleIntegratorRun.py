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

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from numpy import (
    dtype as np_dtype,
    finfo as np_finfo,
    float64 as np_float64,
    floor as np_floor,
)

from cubie.integrators.SingleIntegratorRunCore import (
    SingleIntegratorRunCore,
)
from cubie.odesystems.ODEData import SystemSizes


if TYPE_CHECKING:  # pragma: no cover - type checking import only
    from cubie.odesystems.baseODE import BaseODE


def regular_event_count(
    duration: float, interval: float, precision: type
) -> int:
    """Count the events of ``interval`` in ``duration`` at ``precision``.

    Parameters
    ----------
    duration
        Integration duration in time units.
    interval
        Scheduling interval in time units.
    precision
        Working float type of the run.

    Returns
    -------
    int
        Number of scheduled events, excluding the initial sample.
    """
    total_events = np_float64(precision(duration)) / np_float64(
        precision(interval)
    )
    allowance = min(
        4.0 * float(np_finfo(precision).eps) * total_events, 0.49
    )
    return int(np_floor(total_events + allowance))


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

    @property
    def solver_diagnostics(self) -> Dict[str, Any]:
        """Return the step's solver settings reported when runs fail."""

        return self._algo_step.solver_diagnostics

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
    def dxdt_fn(self) -> Callable:
        """Return the derivative function used by the integration step."""

        return self._algo_step.dxdt_fn

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

    @property
    def save_regularly(self) -> bool:
        """Return True if states save every ``save_every``."""
        return self._loop.compile_settings.save_regularly

    @property
    def summarise_last(self) -> bool:
        """Return True if one summary over the whole run is configured."""
        return self._loop.compile_settings.summarise_last

    @property
    def summarise_regularly(self) -> bool:
        """Return True if a summary is written every ``summarise_every``."""
        return self._loop.compile_settings.summarise_regularly

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
        regular_samples = 0
        final_samples = 1 if self.save_last else 0
        initial_sample = 1
        if self.save_regularly:
            regular_samples = regular_event_count(
                duration, self.save_every, self.precision
            )
        return regular_samples + initial_sample + final_samples

    def save_event_count(self, duration: float) -> int:
        """Return the number of scheduled save rows, initial included."""
        if not self.save_regularly:
            return 1
        return regular_event_count(
            duration, self.save_every, self.precision
        ) + 1

    def summary_sample_count(self, duration: float) -> int:
        """Return the number of scheduled summary samples."""
        if not (self.summarise_regularly or self.summarise_last):
            return 0
        return regular_event_count(
            duration, self.sample_summaries_every, self.precision
        )

    def summaries_length(self, duration: float) -> int:
        """Calculate number of summary output rows for a duration.

        ``summarise_last`` is one row. Otherwise the device writes one
        summary row after every
        ``samples_per_summary`` summary measurements. The number of
        measurements in a run follows the same counting rule as
        saves (:func:`regular_event_count`), and only a complete
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
        if self.summarise_last:
            return 1

        regular_summaries = 0
        if self.summarise_regularly:
            sample_every = self.sample_summaries_every
            updates = regular_event_count(
                duration, sample_every, self.precision
            )
            samples_per_summary = (
                self._loop.compile_settings.samples_per_summary
            )
            regular_summaries = updates // max(samples_per_summary, 1)
        return regular_summaries

    @property
    def compile_flags(self) -> Any:
        """Return loop compile flags."""

        return self._loop.compile_flags

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
    def save_state_fn(self) -> Callable:
        """Return the compiled state saving function."""

        return self._output_functions.save_state_fn

    @property
    def update_summaries_fn(self) -> Callable:
        """Return the compiled summary update function."""

        return self._output_functions.update_summaries_fn

    @property
    def save_summaries_fn(self) -> Callable:
        """Return the compiled summary saving function."""

        return self._output_functions.save_summaries_fn

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
