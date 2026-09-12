Output handling
===============

``cubie.outputhandling``
------------------------

.. currentmodule:: cubie.outputhandling

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   output_functions
   output_config
   output_compile_flags
   output_array_heights
   single_run_output_sizes
   batch_input_sizes
   batch_output_sizes
   output_function_cache
   summary_metrics
   register_metric

The output handling package manages CUDA device callbacks that save state
trajectories and summary calculations from integration loops. It turns loop
settings into checked configuration, builds the device functions through the
CUDA factory tools, and provides sizing helpers so callers can allocate host and
device buffers without repeating dimension logic.

.. toctree::
   :maxdepth: 2
   :caption: Subpackages
   :titlesonly:

   summarymetrics

Entry point
-----------

:doc:`OutputFunctions <output_functions>` is the main interface. Create it with loop
dimensions and requested outputs to compile CUDA functions that save solver
state, refresh summary metrics, and write reductions back to host arrays. The
factory keeps an :class:`OutputConfig` instance and rebuilds compiled callbacks
when configuration changes.

Configuration
-------------

* :doc:`OutputConfig <output_config>` – attrs container describing requested outputs and summaries.
* :doc:`OutputCompileFlags <output_compile_flags>` – compile-time flags for CUDA output kernels.

Sizing utilities
----------------

* :doc:`OutputArrayHeights <output_array_heights>` – calculates height metadata for output arrays.
* :doc:`SingleRunOutputSizes <single_run_output_sizes>` – shapes for single-run solver outputs.
* :doc:`BatchInputSizes <batch_input_sizes>` – input buffer sizes for batch runs.
* :doc:`BatchOutputSizes <batch_output_sizes>` – output buffer sizes for batch runs.

Runtime factories and registries
--------------------------------

* :doc:`OutputFunctions <output_functions>` – compiles CUDA callbacks for saving state and summaries.
* :doc:`OutputFunctionCache <output_function_cache>` – caches compiled device callables keyed by configuration.
* :doc:`summary_metrics <summary_metrics>` – shared registry of registered summary metrics.
* :doc:`register_metric <register_metric>` – decorator used by metric implementations to register with the shared registry.

Dependencies
------------

* Compiles CUDA callables through :class:`cubie.CUDAFactory` and
  :mod:`numba.cuda`.
* Loop buffers and output slices align with expectations from
  :mod:`cubie.integrators.loops` and related algorithm factories.
