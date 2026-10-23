Adding Summary Metrics
======================

CuBIE's summary metrics system is extensible via the
``@register_metric`` decorator.  This page walks through implementing a
custom metric.

File Location
-------------

Built-in metrics live in
``src/cubie/outputhandling/summarymetrics/``.  Each metric is a
separate module.

The ``@register_metric`` Decorator
-----------------------------------

Register a new metric by subclassing ``SummaryMetric`` and decorating
with ``@register_metric``:

.. code-block:: python

   from cubie.outputhandling.summarymetrics.metrics import (
       SummaryMetric,
       MetricFuncCache,
       register_metric,
       summary_metrics,
   )

   @register_metric(summary_metrics)
   class MyCustomMetric(SummaryMetric):
       def __init__(self, precision):
           super().__init__(
               buffer_size=2,
               output_size=1,
               name="my_custom",
               precision=precision,
               unit_modification="[custom]",
               sample_summaries_every=None,
           )

       def build(self) -> MetricFuncCache:
           # Return a MetricFuncCache with update and save callables
           ...

``__init__`` Parameters
^^^^^^^^^^^^^^^^^^^^^^^

``buffer_size``
   Number of scratch elements per variable needed during accumulation.
   Can be an ``int`` or a callable that receives the number of
   summarised variables.

``output_size``
   Number of output elements per variable written to the result.  Can
   also be a callable.

``name``
   String identifier used in ``output_types`` lists.

``unit_modification``
   Label suffix appended to variable names in the summary legend
   (e.g. ``"[mean]"``, ``"[max]"``).

``sample_summaries_every``
   If not ``None``, the metric requests sub-step sampling at this
   interval.

The ``build()`` Method
----------------------

``build()`` must return a ``MetricFuncCache`` containing two Numba CUDA
device functions:

``update(buffer, value, t, dt)``
   Called at every summary sample point during integration.  Accumulates
   data into ``buffer``.

``save(buffer, output, n_samples)``
   Called at the end of the solve.  Finalises the accumulated data and
   writes to ``output``.

Both functions operate on per-variable slices and must be compatible with
Numba's CUDA compilation.

Example: Implementing a "Count" Metric
----------------------------------------

A metric that counts how many samples were accumulated:

.. code-block:: python

   from numba import cuda

   @register_metric(summary_metrics)
   class CountMetric(SummaryMetric):
       def __init__(self, precision):
           super().__init__(
               buffer_size=1,
               output_size=1,
               name="count",
               precision=precision,
               unit_modification="[count]",
           )

       def build(self) -> MetricFuncCache:
           @cuda.jit(device=True)
           def update(buffer, value, t, dt):
               buffer[0] += 1.0

           @cuda.jit(device=True)
           def save(buffer, output, n_samples):
               output[0] = buffer[0]

           return MetricFuncCache(update=update, save=save)

After adding the file, the metric is available as
``output_types=["count"]``.
