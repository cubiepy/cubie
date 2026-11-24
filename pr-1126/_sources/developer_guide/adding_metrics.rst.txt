Adding Summary Metrics
======================

You add a summary metric by subclassing ``SummaryMetric`` and registering
the class with the ``@register_metric`` decorator.  Once the module that
defines it has been imported, users request your metric by name in
``output_types``, just like a built-in one.

File Location
-------------

cubie keeps its built-in metrics in
``src/cubie/outputhandling/summarymetrics/``, one module per metric.
cubie imports every module in that package's ``__init__.py`` to
register each metric, so add an import line there for a new built-in
metric.  cubie registers a metric defined in your own code when you
import its module, so import it before you create a ``Solver`` or call
``solve_ivp``.

The ``@register_metric`` Decorator
-----------------------------------

Subclass ``SummaryMetric`` and decorate the class with
``@register_metric(summary_metrics)``:

.. code-block:: python

   from cubie.outputhandling.summarymetrics import (
       register_metric,
       summary_metrics,
   )
   from cubie.outputhandling.summarymetrics.metrics import (
       MetricFuncCache,
       SummaryMetric,
   )

   @register_metric(summary_metrics)
   class MyCustomMetric(SummaryMetric):
       def __init__(self, precision):
           super().__init__(
               name="my_custom",
               precision=precision,
               buffer_size=2,
               output_size=1,
               unit_modification="[unit]",
           )

       def build(self) -> MetricFuncCache:
           # Return a MetricFuncCache with the update and save functions
           ...

When the decorator runs, cubie creates one instance of the class,
passing the registry's precision as the only argument, so ``__init__``
takes ``precision`` and nothing else.

``__init__`` Parameters
^^^^^^^^^^^^^^^^^^^^^^^

``name``
   The name users put in ``output_types``.

``buffer_size``
   The number of working values cubie stores for each summarised
   variable while it calculates the metric.  For a metric that takes a
   parameter in its request, such as ``"peaks[5]"``, pass a callable
   that receives the parameter and returns the size.

``output_size``
   The number of values cubie writes for each variable in each summary
   window.  This can also be a callable of the parameter.

``unit_modification``
   The unit of the metric's output for the results legend, written in
   terms of the variable's unit.  Write ``[unit]`` where the variable's
   unit goes: use ``"[unit]"`` for a mean and ``"[unit]*s^-1"`` for a
   first derivative.

``output_names``
   Optional column names, one for each output value.  Without them,
   cubie names a single output after the metric and numbers the columns
   of a longer output (``peaks_1``, ``peaks_2``, ...).

Leave ``sample_summaries_every`` at its default.  Before cubie compiles
a solver, it sets the precision and ``sample_summaries_every`` of every
registered metric to the solver's values.

The ``build()`` Method
----------------------

Return a ``MetricFuncCache(update_fn=update, save_fn=save)`` from
``build()``, holding two CUDA device functions.  cubie passes both
functions this metric's slice of the buffer for one variable, starting
at index 0.

``update(value, buffer, current_index, customisable_variable)``
   cubie calls ``update`` at every summary sample with the variable's
   current value.  ``current_index`` is the number of the summary
   sample, counted from 0 at the first sample of the run; cubie does not
   restart the count at each window.
   ``customisable_variable`` is the parameter from the request (``5``
   for ``"peaks[5]"``), or ``0`` when there is none.

``save(buffer, output_array, summarise_every, customisable_variable)``
   cubie calls ``save`` at the end of each summary window.
   ``summarise_every`` is the number of samples in the window.  Write
   ``output_size`` values to ``output_array``, then put ``buffer`` back
   into its starting state, ready for the next window.

Compile both functions with
``@cuda.jit(device=True, inline=True, **self.jit_kwargs)``, importing
``cuda`` from ``cubie._cudasim_extensions`` so that you can also run the
metric under the CUDA simulator.  Read the precision and
``sample_summaries_every`` from ``self.compile_settings`` inside
``build()``, and cast constants with ``precision(...)`` so that cubie can
compile the functions for both ``float32`` and ``float64``.  cubie
rebuilds the metric whenever it changes either setting.

Combined Metrics
----------------

When a user requests every metric in certain sets, cubie computes them
together from one shared buffer: for example, cubie calculates
``"mean"`` and ``"std"`` together as ``"mean_std"``.  To add a
combination for built-in metrics, register a combined metric with
``output_names`` set to the names of its parts, so that users still see
the names they asked for, and add the set of names to
``_combined_metrics`` in ``SummaryMetrics.__attrs_post_init__`` in
``metrics.py``.

Example: Implementing a "Count" Metric
--------------------------------------

With the ``Count`` metric below, you get the number of summary samples
in each window:

.. code-block:: python

   from cubie._cudasim_extensions import cuda
   from cubie.outputhandling.summarymetrics import (
       register_metric,
       summary_metrics,
   )
   from cubie.outputhandling.summarymetrics.metrics import (
       MetricFuncCache,
       SummaryMetric,
   )

   @register_metric(summary_metrics)
   class Count(SummaryMetric):
       def __init__(self, precision):
           super().__init__(
               name="count",
               precision=precision,
               buffer_size=1,
               output_size=1,
               unit_modification="1",
           )

       def build(self) -> MetricFuncCache:
           precision = self.compile_settings.precision

           @cuda.jit(device=True, inline=True, **self.jit_kwargs)
           def update(value, buffer, current_index, customisable_variable):
               buffer[0] += precision(1.0)

           @cuda.jit(device=True, inline=True, **self.jit_kwargs)
           def save(buffer, output_array, summarise_every,
                    customisable_variable):
               output_array[0] = buffer[0]
               buffer[0] = precision(0.0)

           return MetricFuncCache(update_fn=update, save_fn=save)

Save the metric in ``count_metric.py``, import that module, and request
the metric like any other:

.. code-block:: python

   import numpy as np
   import cubie as qb

   import count_metric  # import to register "count"

   LV = qb.create_ODE_system(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )

   result = qb.solve_ivp(
       LV,
       parameters={"b": np.linspace(0.01, 0.05, 1000)},
       method="ode45",
       duration=50.0,
       summarise_every=10.0,
       sample_summaries_every=0.5,
       output_types=["count"],
   )
   result.as_numpy_per_summary["count"][:, 0, 0]  # 20.0 in each window
