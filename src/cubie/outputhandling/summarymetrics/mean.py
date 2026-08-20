"""Arithmetic mean over a summary interval.

Published Classes
-----------------
:class:`Mean`
    Arithmetic mean over a summary interval.

See Also
--------
:class:`~cubie.outputhandling.summarymetrics.metrics.SummaryMetric`
    Abstract base class for summary metrics.
:data:`~cubie.outputhandling.summarymetrics.summary_metrics`
    Global registry where this metric is registered.
"""

from cubie.cuda_simsafe import cuda


from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class Mean(SummaryMetric):
    """Summary metric that calculates the arithmetic mean of a variable.

    Notes
    -----
    The metric uses a single buffer slot per variable to accumulate the sum
    of values and divides by the number of integration steps when the results
    are saved.
    """

    def __init__(self, precision) -> None:
        """Initialise the Mean summary metric with fixed buffer sizes."""
        super().__init__(
            name="mean",
            precision=precision,
            buffer_size=1,
            output_size=1,
            unit_modification="[unit]",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for mean value calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback accumulates the running sum while the save
        callback divides by ``summarise_every`` and resets the buffer.
        """

        precision = self.compile_settings.precision

        # no cover: start
        @cuda.jit(
            # [
            #     "float32, float32[::1], int32, int32",
            #     "float64, float64[::1], int32, int32",
            # ],
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def update(
            value,
            buffer,
            current_index,
            customisable_variable,
        ):
            """Update the running sum with a new value.

            Parameters
            ----------
            value
                float. New value to add to the running sum.
            buffer
                device array. Location containing the running sum.
            current_index
                int. Current integration step index (unused for mean).
            customisable_variable
                int. Metric parameter placeholder (unused for mean).

            Notes
            -----
            Adds the new value to ``buffer[0]`` to maintain the running sum.
            """
            buffer[0] += value

        @cuda.jit(
            # [
            #     "float32[::1], float32[::1], int32, int32",
            #     "float64[::1], float64[::1], int32, int32",
            # ],
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def save(
            buffer,
            output_array,
            summarise_every,
            customisable_variable,
        ):
            """Calculate the mean and reset the buffer.

            Parameters
            ----------
            buffer
                device array. Location containing the running sum of values.
            output_array
                device array. Location for saving the mean value.
            summarise_every
                int. Number of integration steps contributing to each summary.
            customisable_variable
                int. Metric parameter placeholder (unused for mean).

            Notes
            -----
            Divides the accumulated sum by ``summarise_every`` and saves the
            result to ``output_array[0]`` before resetting ``buffer[0]``.
            """
            output_array[0] = buffer[0] / summarise_every
            buffer[0] = precision(0.0)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
