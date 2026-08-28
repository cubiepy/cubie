"""Peak (local maximum) detection over a summary interval.

Published Classes
-----------------
:class:`Peaks`
    Peak (local maximum) detection over a summary interval.

See Also
--------
:class:`~cubie.outputhandling.summarymetrics.metrics.SummaryMetric`
    Abstract base class for summary metrics.
:data:`~cubie.outputhandling.summarymetrics.summary_metrics`
    Global registry where this metric is registered.
"""

from numpy import dtype as np_dtype, int32 as np_int32

from cubie.cuda_simsafe import cuda, int32

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class Peaks(SummaryMetric):
    """Summary metric that records the indices of detected peaks.

    Notes
    -----
    The buffer stores the two previous values, a peak counter, and slots for
    the recorded peak indices. The algorithm assumes ``0.0`` does not occur in
    valid data so it can serve as an initial sentinel.
    """

    def __init__(self, precision) -> None:
        """Initialise the Peaks summary metric with parameterised sizes."""
        super().__init__(
            name="peaks",
            precision=precision,
            buffer_size=lambda n: 3 + n,
            output_size=lambda n: n,
            unit_modification="s",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for peak detection.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback compares the current value against stored history
        to identify peaks, while the save callback copies stored indices and
        resets the buffer for the next period.
        """

        precision = self.compile_settings.precision
        jit_kwargs = self.jit_kwargs
        # The counter and peak-index slots hold integers; use an
        # int32 view when the elements are wide enough to hold one.
        use_int_view = np_dtype(precision).itemsize >= 4

        # no cover: start
        @cuda.jit(
            # [
            #     "float32, float32[::1], int32, int32",
            #     "float64, float64[::1], int32, int32",
            # ],
            device=True,
            inline=False,
            **jit_kwargs,
        )
        def update(
            value,
            buffer,
            current_index,
            customisable_variable,
        ):
            """Update peak detection with a new value.

            Parameters
            ----------
            value
                float. New value to analyse for peak detection.
            buffer
                device array. Layout ``[prev, prev_prev, counter,
                times...]``.
            current_index
                int. Current integration step index, used to record peaks.
            customisable_variable
                int. Maximum number of peaks to detect.

            Notes
            -----
            Detects peaks when the prior value exceeds both the current and
            second-prior values. Peak indices are stored after the
            counter.
            """
            npeaks = customisable_variable
            prev = buffer[0]
            prev_prev = buffer[1]
            if use_int_view:
                int_slots = buffer[2:].view(np_int32)
            else:
                int_slots = buffer[2:]
            peak_counter = int32(int_slots[0])

            if (
                (current_index >= 2)
                and (peak_counter < npeaks)
                and (prev_prev != precision(0.0))
            ):
                if prev > value and prev_prev < prev:
                    # Bingo
                    int_slots[1 + peak_counter] = current_index - int32(1)
                    int_slots[0] = peak_counter + int32(1)
            buffer[0] = value  # Update previous value
            buffer[1] = prev  # Update previous previous value

        @cuda.jit(
            # [
            #     "float32[::1], float32[::1], int32, int32",
            #     "float64[::1], float64[::1], int32, int32",
            # ],
            device=True,
            inline=False,
            **jit_kwargs,
        )
        def save(
            buffer,
            output_array,
            summarise_every,
            customisable_variable,
        ):
            """Save detected peak time indices and reset the buffer.

            Parameters
            ----------
            buffer
                device array. Buffer containing detected peak time indices.
            output_array
                device array. Output array for saving peak time indices.
            summarise_every
                int. Number of steps between saves (unused for peak detection).
            customisable_variable
                int. Maximum number of peaks to detect.

            Notes
            -----
            Copies peak indices to the output array then clears the
            storage for the next summary interval.
            """
            n_peaks = int32(customisable_variable)
            if use_int_view:
                int_slots = buffer[2:].view(np_int32)
            else:
                int_slots = buffer[2:]
            for p in range(n_peaks):
                output_array[p] = precision(int_slots[1 + p])
                int_slots[1 + p] = int32(0)
            int_slots[0] = int32(0)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
