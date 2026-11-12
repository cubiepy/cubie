Arrays
======

``cubie.batchsolving.arrays``
-----------------------------

.. currentmodule:: cubie.batchsolving.arrays

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   array_container
   base_array_manager
   managed_array
   input_array_container
   input_arrays
   output_array_container
   active_outputs
   output_arrays

The ``batchsolving.arrays`` package coordinates host and device array
management for batch solver runs. ``InputArrays`` and ``OutputArrays`` are the
key classes: they collect stride metadata, request GPU allocations via
:mod:`cubie.memory`, and expose helpers for copying data between the CPU and
CUDA kernels. ``OutputArrays`` mirrors requested state, observable, and summary
buffers on the host and GPU for every solver launch.

Base utilities
^^^^^^^^^^^^^^

* :doc:`ArrayContainer <array_container>` – attrs container shared by input and output managers.
* :doc:`BaseArrayManager <base_array_manager>` – helper that manages host/device array bindings and synchronisation.

Input arrays
^^^^^^^^^^^^

* :doc:`InputArrayContainer <input_array_container>` – checked attrs container describing solver inputs.
* :doc:`InputArrays <input_arrays>` – handles allocation, host buffers, and CUDA copies for input data.

Output arrays
^^^^^^^^^^^^^

* :doc:`OutputArrayContainer <output_array_container>` – attrs container for state, observables, and summary outputs.
* :doc:`ActiveOutputs <active_outputs>` – boolean flags describing which outputs and summaries are requested.
* :doc:`OutputArrays <output_arrays>` – prepares host/device arrays, handles CUDA transfers, and exposes host views.

Dependencies
------------

* :mod:`cubie.outputhandling.output_sizes` supplies ``BatchOutputSizes``
  metadata for array shapes and active buffers.
* :mod:`cubie.batchsolving.ArrayTypes` defines the array typing shared with
  solver kernels.
* :mod:`cubie._utils` offers ``slice_variable_dimension`` to convert chunk
  indices into multidimensional slices.
