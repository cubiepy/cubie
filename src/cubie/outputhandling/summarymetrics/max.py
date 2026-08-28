"""Maximum value over a summary interval.

Published Classes
-----------------
:class:`Max`
    Maximum value over a summary interval.

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
class Max(SummaryMetric):
    """Summary metric that tracks the maximum value of a variable.

    Notes
    -----
    A single buffer slot stores the running maximum. The buffer resets to
    ``-1.0e30`` after each save so any new value can replace it.
    """

    def __init__(self, precision) -> None:
        """Initialise the Max summary metric with fixed buffer sizes."""
        super().__init__(
            name="max",
            precision=precision,
            buffer_size=1,
            output_size=1,
            unit_modification="[unit]",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for maximum value calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback keeps the running maximum while the save callback
        writes the result and resets the buffer sentinel.
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
            """Update the running maximum with a new value.

            Parameters
            ----------
            value
                float. New value to compare against the current maximum.
            buffer
                device array. Storage for the current maximum value.
            current_index
                int. Current integration step index (unused for this metric).
            customisable_variable
                int. Metric parameter placeholder (unused for max).

            Notes
            -----
            Updates ``buffer[0]`` if the new value exceeds the current maximum.
            """
            if value > buffer[0]:
                buffer[0] = value

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
            """Save the maximum value to output and reset the buffer.

            Parameters
            ----------
            buffer
                device array. Buffer containing the current maximum value.
            output_array
                device array. Output location for saving the maximum value.
            summarise_every
                int. Number of steps between saves (unused for max).
            customisable_variable
                int. Metric parameter placeholder (unused for max).

            Notes
            -----
            Copies ``buffer[0]`` to ``output_array[0]`` and resets the buffer
            sentinel to ``-1.0e30`` for the next period.
            """
            output_array[0] = buffer[0]
            buffer[0] = precision(-1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
