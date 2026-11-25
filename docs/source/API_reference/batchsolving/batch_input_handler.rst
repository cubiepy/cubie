BatchInputHandler
=================

.. currentmodule:: cubie.batchsolving.BatchInputHandler

:class:`~cubie.batchsolving.solver.Solver` uses a
:class:`BatchInputHandler` to turn the initial values and parameters you
pass to ``solve`` into the arrays the GPU reads, one row per variable and
one column per run. The grid functions below build those arrays from
dicts of values.

.. autoclass:: BatchInputHandler
    :members:
    :show-inheritance:

Grid functions
--------------

.. autofunction:: generate_grid

.. autofunction:: combinatorial_grid

.. autofunction:: verbatim_grid

.. autofunction:: unique_cartesian_product

.. autofunction:: combine_grids

.. autofunction:: extend_grid_to_array
