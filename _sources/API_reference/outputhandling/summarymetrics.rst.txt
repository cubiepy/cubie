Summary metrics
===============

``cubie.outputhandling.summarymetrics``
---------------------------------------

.. currentmodule:: cubie.outputhandling.summarymetrics

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   summary_metrics
   register_metric
   summarymetrics/metrics/summary_metric
   summarymetrics/metrics/summary_metrics
   summarymetrics/metrics/metric_func_cache
   summarymetrics/metrics/mean
   summarymetrics/metrics/max
   summarymetrics/metrics/rms
   summarymetrics/metrics/peaks
   summarymetrics/metrics/std
   summarymetrics/metrics/min
   summarymetrics/metrics/max_magnitude
   summarymetrics/metrics/extrema
   summarymetrics/metrics/negative_peaks
   summarymetrics/metrics/mean_std_rms
   summarymetrics/metrics/dxdt_max
   summarymetrics/metrics/dxdt_min
   summarymetrics/metrics/dxdt_extrema
   summarymetrics/metrics/d2xdt2_max
   summarymetrics/metrics/d2xdt2_min
   summarymetrics/metrics/d2xdt2_extrema

The ``summarymetrics`` package houses the summary metric registry used by output
handling to accumulate reductions during integration. Importing the package
creates :data:`summary_metrics` and eagerly imports the built-in metrics so that
each registers its CUDA device update and save functions. External packages
extend the system by decorating new metric classes with :func:`register_metric`.

Public interface
----------------

* :doc:`summary_metrics <summary_metrics>` – registry storing metric factories and compiled device callables.
* :doc:`register_metric <register_metric>` – decorator used by metric modules to register implementations.
* :doc:`SummaryMetric <summarymetrics/metrics/summary_metric>` – base class describing summary metric interfaces.
* :doc:`SummaryMetrics <summarymetrics/metrics/summary_metrics>` – registry container that stores metrics and compiled functions.
* :doc:`MetricFuncCache <summarymetrics/metrics/metric_func_cache>` – caches compiled CUDA functions per metric.

Built-in metrics
~~~~~~~~~~~~~~~~

* :doc:`Mean <summarymetrics/metrics/mean>` – arithmetic average.
* :doc:`Max <summarymetrics/metrics/max>` – maximum value.
* :doc:`Min <summarymetrics/metrics/min>` – minimum value.
* :doc:`RMS <summarymetrics/metrics/rms>` – root-mean-square.
* :doc:`Std <summarymetrics/metrics/std>` – standard deviation.
* :doc:`MaxMagnitude <summarymetrics/metrics/max_magnitude>` – maximum absolute value.
* :doc:`Extrema <summarymetrics/metrics/extrema>` – both maximum and minimum.
* :doc:`Peaks <summarymetrics/metrics/peaks>` – peak detection (local maxima).
* :doc:`NegativePeaks <summarymetrics/metrics/negative_peaks>` – negative peak detection (local minima).
* :doc:`MeanStdRms <summarymetrics/metrics/mean_std_rms>` – composite metric computing mean, std, and rms efficiently.
* :doc:`DxdtMax <summarymetrics/metrics/dxdt_max>` – maximum first derivative (slope).
* :doc:`DxdtMin <summarymetrics/metrics/dxdt_min>` – minimum first derivative (slope).
* :doc:`DxdtExtrema <summarymetrics/metrics/dxdt_extrema>` – both maximum and minimum first derivative.
* :doc:`D2xdt2Max <summarymetrics/metrics/d2xdt2_max>` – maximum second derivative (acceleration).
* :doc:`D2xdt2Min <summarymetrics/metrics/d2xdt2_min>` – minimum second derivative (acceleration).
* :doc:`D2xdt2Extrema <summarymetrics/metrics/d2xdt2_extrema>` – both maximum and minimum second derivative.

Dependencies
------------

* Compiles device functions via :class:`cubie.CUDAFactory` and :mod:`numba.cuda`.
* Consumes save/update cadence configuration from :mod:`cubie.outputhandling`.
