"""Maximum absolute value over a summary interval.

Published Classes
-----------------
:class:`MaxMagnitude`
    Maximum absolute value over a summary interval.

See Also
--------
:class:`~cubie.outputhandling.summarymetrics.metrics.SummaryMetric`
    Abstract base class for summary metrics.
:data:`~cubie.outputhandling.summarymetrics.summary_metrics`
    Global registry where this metric is registered.
"""

from cubie.cuda_simsafe import cuda

from math import fabs

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class MaxMagnitude(SummaryMetric):
    """Summary metric that tracks the maximum absolute value of a variable.

    Notes
    -----
    A single buffer slot stores the running maximum magnitude. The buffer
    resets to ``0.0`` after each save.
    """

    def __init__(self, precision) -> None:
        """Initialise the MaxMagnitude summary metric."""
        super().__init__(
            name="max_magnitude",
            precision=precision,
            buffer_size=1,
            output_size=1,
            unit_modification="[unit]",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for maximum magnitude calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback keeps the running maximum of absolute values while
        the save callback writes the result and resets the buffer.
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
            """Update the running maximum magnitude with a new value.

            Parameters
            ----------
            value
                float. New value whose absolute value is compared.
            buffer
                device array. Storage for the current maximum magnitude.
            current_index
                int. Current integration step index (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Updates ``buffer[0]`` if ``abs(value)`` exceeds the current
            maximum magnitude.
            """
            abs_value = fabs(value)
            if abs_value > buffer[0]:
                buffer[0] = abs_value

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
            """Save the maximum magnitude to output and reset the buffer.

            Parameters
            ----------
            buffer
                device array. Buffer containing the current max magnitude.
            output_array
                device array. Output location for saving the max magnitude.
            summarise_every
                int. Number of steps between saves (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Copies ``buffer[0]`` to ``output_array[0]`` and resets the buffer
            to ``0.0`` for the next period.
            """
            output_array[0] = buffer[0]
            buffer[0] = precision(0.0)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
