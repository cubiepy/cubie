"""Maximum first derivative via finite differences.

Published Classes
-----------------
:class:`DxdtMax`
    Maximum first derivative via finite differences.

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
class DxdtMax(SummaryMetric):
    """Summary metric that tracks maximum first derivative values.

    Notes
    -----
    Uses two buffer slots: buffer[0] for previous value and buffer[1] for
    maximum unscaled derivative. The derivative is computed using finite
    differences and scaled by sample_summaries_every in the save function.
    """

    def __init__(self, precision) -> None:
        """Initialise the DxdtMax summary metric."""
        super().__init__(
            name="dxdt_max",
            precision=precision,
            buffer_size=2,
            output_size=1,
            unit_modification="[unit]*s^-1",
        )

    def build(self) -> MetricFuncCache:
        """Generate CUDA device functions for maximum derivative calculation.

        Returns
        -------
        MetricFuncCache
            Cache containing the device update and save callbacks.

        Notes
        -----
        The update callback computes finite differences and tracks the
        maximum unscaled derivative. The save callback scales by
        sample_summaries_every and resets the buffers.
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
            """Update the maximum first derivative with a new value.

            Parameters
            ----------
            value
                float. New value to compute derivative from.
            buffer
                device array. Storage for [prev_value, max_unscaled].
            current_index
                int. Monotonic summary-sample counter; gates the first
                update until one previous value exists.
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Computes unscaled derivative as (value - buffer[0]) and updates
            buffer[1] if larger. Uses predicated commit pattern to avoid
            warp divergence. The current_index guard skips the sample with
            no history rather than testing the value against zero, so
            exact-zero samples are handled correctly.
            """
            derivative_unscaled = value - buffer[0]
            update_flag = (derivative_unscaled > buffer[1]) and (
                current_index >= 1
            )
            buffer[1] = selp(update_flag, derivative_unscaled, buffer[1])
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
            """Save scaled maximum derivative and reset buffers.

            Parameters
            ----------
            buffer
                device array. Buffer containing [prev_value, max_unscaled].
            output_array
                device array. Output location for maximum derivative.
            summarise_every
                int. Number of steps between saves (unused).
            customisable_variable
                int. Metric parameter placeholder (unused).

            Notes
            -----
            Scales the maximum unscaled derivative by sample_summaries_every
            and saves to output_array[0], then resets buffers to sentinel
            values.
            """
            output_array[0] = buffer[1] / precision(sample_summaries_every)
            buffer[1] = precision(-1.0e30)

        # no cover: end
        return MetricFuncCache(update=update, save=save)
