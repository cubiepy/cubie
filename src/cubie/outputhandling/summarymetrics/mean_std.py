"""Combined mean and standard deviation from shared running sums.

Published Classes
-----------------
:class:`MeanStd`
    Combined mean and standard deviation from shared running sums.

See Also
--------
:class:`~cubie.outputhandling.summarymetrics.metrics.SummaryMetric`
    Abstract base class for summary metrics.
:data:`~cubie.outputhandling.summarymetrics.summary_metrics`
    Global registry where this metric is registered.
"""

from cubie.cuda_simsafe import cuda

from math import sqrt

from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class MeanStd(SummaryMetric):
    """Composite metric that calculates mean and std together.

    Notes
    -----
    Uses three buffer slots: shift (first value), sum of shifted values, and
    sum of squares of shifted values. The shift technique improves numerical
    stability for the variance calculation.

    The output array contains [mean, std] in that order.
    """

    def __init__(self, precision) -> None:
        """Initialise the MeanStd composite metric."""
        super().__init__(
            name="mean_std",
            precision=precision,
            buffer_size=3,
            output_size=2,
            unit_modification="[unit]",
            output_names=["mean", "std"],
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for composite calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback accumulates sum and sum_of_squares while the
        save callback computes both metrics from these running sums
        and clears the buffer.
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
            """Update running sums with a new value using shifted data.

            Parameters
            ----------
            value
                float. New value to add to the running statistics.
            buffer
                device array. Storage containing [shift, sum_shifted,
                sum_sq_shifted].
            current_index
                int. Current integration step index within summary period.
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            On first sample (current_index == 0), stores value as shift.
            Computes shifted_value = value - shift and adds it to buffer[1]
            (sum) and shifted_value^2 to buffer[2] (sum of squares).
            """
            if current_index == 0:
                buffer[0] = value
                buffer[1] = precision(0.0)
                buffer[2] = precision(0.0)

            shifted_value = value - buffer[0]
            buffer[1] += shifted_value
            buffer[2] += shifted_value * shifted_value

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
            """Calculate mean and std from shifted running sums.

            Parameters
            ----------
            buffer
                device array. Buffer containing [shift, sum_shifted,
                sum_sq_shifted].
            output_array
                device array. Output location for [mean, std].
            summarise_every
                int. Number of steps contributing to each summary window.
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Calculates:
            - mean = shift + sum_shifted / n
            - variance = (sum_sq_shifted/n) - (sum_shifted/n)^2
            - std = sqrt(variance)

            Saves to output_array[0:2] and resets buffer for next period.
            """
            shift = buffer[0]
            mean_shifted = buffer[1] / summarise_every
            mean_of_squares_shifted = buffer[2] / summarise_every

            mean = shift + mean_shifted
            variance = mean_of_squares_shifted - (mean_shifted * mean_shifted)

            output_array[0] = mean
            output_array[1] = sqrt(variance)

            buffer[0] = mean
            buffer[1] = precision(0.0)
            buffer[2] = precision(0.0)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
