"""Combined maximum and minimum over a summary interval.

Published Classes
-----------------
:class:`Extrema`
    Combined maximum and minimum over a summary interval.

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
class Extrema(SummaryMetric):
    """Summary metric that tracks both maximum and minimum values.

    Notes
    -----
    Uses two buffer slots: buffer[0] for maximum and buffer[1] for minimum.
    Outputs two values in the same order.
    """

    def __init__(self, precision) -> None:
        """Initialise the Extrema summary metric."""
        super().__init__(
            name="extrema",
            precision=precision,
            buffer_size=2,
            output_size=2,
            unit_modification="[unit]",
            output_names=["max", "min"],
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for extrema calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback maintains both max and min while the save callback
        writes both results and resets the buffers.
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
            """Update the running maximum and minimum with a new value.

            Parameters
            ----------
            value
                float. New value to compare against current extrema.
            buffer
                device array. Storage for [max, min] values.
            current_index
                int. Current integration step index (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Updates ``buffer[0]`` (max) if value exceeds it, and
            ``buffer[1]`` (min) if value is less than it.
            """
            if value > buffer[0]:
                buffer[0] = value
            if value < buffer[1]:
                buffer[1] = value

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
            """Save both extrema to output and reset the buffers.

            Parameters
            ----------
            buffer
                device array. Buffer containing [max, min] values.
            output_array
                device array. Output location for [max, min] values.
            summarise_every
                int. Number of steps between saves (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Saves max to ``output_array[0]`` and min to ``output_array[1]``,
            then resets buffers to their sentinel values.
            """
            output_array[0] = buffer[0]
            output_array[1] = buffer[1]
            buffer[0] = precision(-1.0e30)
            buffer[1] = precision(1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
