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
   staging_blocks
   input_array_container
   input_arrays
   output_array_container
   output_arrays

The classes in this package hold the arrays of a batch solve and move
them between the host and the GPU. You don't create them yourself: each
solver has one :class:`~cubie.batchsolving.arrays.BatchInputArrays.InputArrays`
for the initial values, parameters and driver coefficients, and one
:class:`~cubie.batchsolving.arrays.BatchOutputArrays.OutputArrays` for
the outputs.

If you pass host arrays in the solver's precision, cubie uses them as
they are and copies them to the GPU when it solves. If you pass them in
another precision, cubie makes one converted copy of each. If a batch
does not fit in GPU memory, cubie splits it into chunks of runs and
copies each chunk's inputs and outputs in turn. Cubie hands the host
outputs of a solve to the
:class:`~cubie.batchsolving.solveresult.SolveResult` it returns.

Each manager keeps a host and a device
:class:`~cubie.batchsolving.arrays.BaseArrayManager.ArrayContainer`,
which hold one
:class:`~cubie.batchsolving.arrays.BaseArrayManager.ManagedArray` for
each array. Cubie allocates the managers' device memory through
:mod:`cubie.memory`, and decides there how many chunks a batch needs.

Base utilities
^^^^^^^^^^^^^^

* :doc:`ArrayContainer <array_container>` – the base class for the
  containers of host or device arrays.
* :doc:`BaseArrayManager <base_array_manager>` – the base class of the
  input and output managers, which handles registration, size checks,
  allocation and copies.
* :doc:`ManagedArray <managed_array>` – the shape, dtype, memory type
  and chunking of one array.
* :doc:`staging_blocks <staging_blocks>` – cuts a copy into blocks that
  fit a pinned staging buffer.

Input arrays
^^^^^^^^^^^^

* :doc:`InputArrayContainer <input_array_container>` – the container
  for the initial values, parameters and driver coefficients.
* :doc:`InputArrays <input_arrays>` – attaches a solve's inputs and
  copies them to the GPU.

Output arrays
^^^^^^^^^^^^^

* :doc:`OutputArrayContainer <output_array_container>` – the container
  for the state and observable trajectories, their summaries, the
  status codes and the iteration counters.
* :doc:`OutputArrays <output_arrays>` – allocates the outputs and
  copies the ones the solve records back to the host.

Dependencies
------------

* :mod:`cubie.outputhandling.output_sizes`: ``BatchInputSizes`` and
  ``BatchOutputSizes``, the expected shape of each array.
* :mod:`cubie.memory`: the allocation of the device arrays, pinned host
  buffers and staging buffers, and the choice of chunks.
* :mod:`cubie.batchsolving.writeback_watcher`: the background thread
  that finishes the staged copies.
