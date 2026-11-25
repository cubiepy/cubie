BatchInputHandler
=================

.. currentmodule:: cubie.batchsolving.BatchInputHandler

Cubie uses a :class:`BatchInputHandler` to turn the initial values and
parameters you pass to ``solve`` into arrays with one row per variable
and one column per run. The grid functions below build those arrays
from dicts of values, either from every combination of the values or
by pairing them run for run.

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
