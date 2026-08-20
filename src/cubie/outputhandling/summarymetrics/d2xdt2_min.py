"""Minimum second derivative via central finite differences.

Published Classes
-----------------
:class:`D2xdt2Min`
    Minimum second derivative via central finite differences.

See Also
--------
:class:`~cubie.outputhandling.summarymetrics.metrics.SummaryMetric`
    Abstract base class for summary metrics.
:data:`~cubie.outputhandling.summarymetrics.summary_metrics`
    Global registry where this metric is registered.
"""

from cubie.cuda_simsafe import cuda

from cubie.cuda_simsafe import selp
from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.summarymetrics.metrics import (
    SummaryMetric,
    register_metric,
    MetricFuncCache,
)


@register_metric(summary_metrics)
class D2xdt2Min(SummaryMetric):
    """Summary metric that tracks minimum second derivative values.

    Notes
    -----
    Uses three buffer slots: buffer[0] for previous value, buffer[1] for
    previous-previous value, and buffer[2] for minimum unscaled second
    derivative. The derivative is computed using central differences and
    scaled by sample_summaries_every² in the save function.
    """

    def __init__(self, precision) -> None:
        """Initialise the D2xdt2Min summary metric."""
        super().__init__(
            name="d2xdt2_min",
            precision=precision,
            buffer_size=3,
            output_size=1,
            unit_modification="[unit]*s^-2",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for minimum second derivative.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback computes central finite differences and tracks
        the minimum unscaled second derivative. The save callback scales
        by sample_summaries_every² and resets the buffers.
        """

        sample_summaries_every = self.compile_settings.sample_summaries_every
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
            """Update the minimum second derivative with a new value.

            Parameters
            ----------
            value
                float. New value to compute second derivative from.
            buffer
                device array. Storage for [prev_value, prev_prev_value,
                min_unscaled].
            current_index
                int. Monotonic summary-sample counter; gates updates until
                two previous values exist.
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Computes unscaled second derivative using central difference
            formula (value - 2*buffer[0] + buffer[1]) and updates buffer[2] if
            smaller. Uses predicated commit pattern to avoid warp divergence.
            The current_index guard skips samples with incomplete history
            rather than testing buffer[1] against zero, so exact-zero samples
            are handled correctly.
            """
            second_derivative_unscaled = (
                value - precision(2.0) * buffer[0] + buffer[1]
            )
            update_flag = (second_derivative_unscaled < buffer[2]) and (
                current_index >= 2
            )
            buffer[2] = selp(
                update_flag, second_derivative_unscaled, buffer[2]
            )
            buffer[1] = buffer[0]
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
            """Save scaled minimum second derivative and reset buffers.

            Parameters
            ----------
            buffer
                device array. Buffer containing [prev_value, prev_prev_value,
                min_unscaled].
            output_array
                device array. Output location for minimum second derivative.
            summarise_every
                int. Number of steps between saves (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Scales the minimum unscaled second derivative by
            sample_summaries_every² and saves to output_array[0], then resets
            buffers to sentinel values.
            """
            sample_interval_sq = (precision(sample_summaries_every)
                                  * precision(sample_summaries_every))
            output_array[0] = buffer[2] / sample_interval_sq
            buffer[2] = precision(1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
