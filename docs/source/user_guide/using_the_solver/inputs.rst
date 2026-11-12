Batch Inputs
============

Each run is defined by the initial values of its states and its parameter values.
You can provide these in a number of ways. All examples in this page
will follow the same Lotka-Volterra system as the rest of the docs, defined
here:

.. code-block:: python

   from cubie import create_ODE_system

   LV = create_ODE_system(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )

.. _parameter-inputs:

Input formats
-------------

Dicts
~~~~~

The simplest way to give cubie your values is a dict for each input,
keyed by name:

.. code-block:: python

   import numpy as np
   from cubie import Solver

   solver = Solver(LV, algorithm="ode45")
   result = solver.solve(
       initial_values={"x": np.linspace(0.1, 2.0, 100)},
       parameters={"a": 0.2, "b": np.linspace(0.01, 0.05, 100)},
       duration=100.0,
   )

Give a single value for anything that is the same in every run, and a list
or array of values for anything you want to vary. If you've already given
a default value when you created the system and you don't want to modify it,
you can leave it out, like ``y`` above. If we wanted to start all trajectories
from the default state, or under default parameters, we could leave the whole
``initial_values`` or ``parameters`` arguments out completely. If you didn't
set defaults when you built the system, you can set them on the system with
``system.set_default_parameters({"a": 0.2})`` for parameters and
``system.set_initial_values({"y": 0.3})`` for states.

If you include a constant value for a parameter or initial value that
is different to the default, the default doesn't change. Cubie presumes
that input arguments to ``solve`` are intended for that run only.

Arrays
~~~~~~

If you're looking for something faster (by ~tens of ms per solve) or are
generating values programmatically, you can pass arrays instead. Lay each
array out as ``(n_variables, n_runs)``, with its rows ordered by variable
name. If the array is already in the same data type as ``solver.precision``,
then you can also save an additional array copy.

.. note::

   It may seem odd that the states array ends up in a different order
   than the one you gave. When we tear DAE systems, or only select a few
   outputs, name-ordering makes it easier to reconstruct and index the
   resulting system.

When you pass a dict, cubie compiles every parameter you give a single value
directly into the GPU code to make it go faster. It does the same for an
array under the default ``fix_constant_parameters=True`` argument. This
will usually make the solve faster, but costs a few ms per solve call.
To get around this, you can:

- Pass ``fix_constant_parameters=False``. All parameters, even the constant
  ones, are part of the input array that gets passed to the GPU, and get loaded
  from memory before use. The GPU can usually hold fewer runs at once,
  and the solve goes a little slower.
- Tell the solver which parameters will be swept with
  ``solver.set_swept_parameters(names)``, where names is a list of parameter names.
  After this you can pass an array with rows for those parameters only, saving
  both the scan for constant parameters and the cost of keeping them in memory.

For the system above, we can pass the smaller parameters array like this.
When you sweep a single parameter, as here, you can pass its values as a 1D
array:

.. code-block:: python

   inits = np.array(
       [np.linspace(0.1, 2.0, 10000), np.full(10000, 0.3)], dtype=np.float32
   )
   solver.set_swept_parameters(["b"])
   result = solver.solve(
       inits,
       np.linspace(0.01, 0.05, 10000, dtype=np.float32),
       duration=100.0,
   )

``solver.prepare_grid_arrays`` can generate arrays from dicts of variable
names, to avoid doing it on every solve. It sets the swept parameters to
the parameters in your dict and gives each of them a row, so if you give a
parameter a single value, cubie fills its whole row with that value. To
compile a value into the code instead, set it as the parameter's default and
leave it out of the dict:

.. code-block:: python

   LV.set_default_parameters({"a": 0.2})
   inits, params = solver.prepare_grid_arrays(
       {"x": np.linspace(0.1, 2.0, 100), "y": 0.3},
       {"b": np.linspace(0.01, 0.05, 100)},
   )
   result = solver.solve(inits, params, duration=100.0)

Arrays of swept parameters only are intended as a convenience: some models
have hundreds of parameters, and if you're only sweeping two of them you
probably don't want to throw hundreds-high arrays around.

Device arrays
~~~~~~~~~~~~~

If you want to go *really* fast and you are comfortable with the CUDA
environment, you can pass arrays on the GPU (device arrays) directly to cut
out the few milliseconds of transfer time as well. Use the same layout as host
arrays, and ensure they're the same datatype as ``solver.precision``.
Pair them with ``on_device=True`` to keep an entire solve → post-process
pipeline on the GPU (see :doc:`results`), never bringing them back to the host.

.. code-block:: python

   from numba_cuda_mlir import cuda

   d_inits = cuda.to_device(inits)
   d_params = cuda.to_device(params)
   result = solver.solve(d_inits, d_params, duration=100.0, on_device=True)
   result.stream.synchronize()

Combining values into runs
--------------------------

When you give arrays of values in dicts, cubie builds the batch of runs
from them in one of two ways, set by ``grid_type``.

Combinatorial grids
~~~~~~~~~~~~~~~~~~~

With ``grid_type="combinatorial"``, the default, cubie solves every
combination of the values you supply:

.. code-block:: python

   from cubie import solve_ivp

   result = solve_ivp(
       LV,
       parameters={
           "b": np.linspace(0.01, 0.05, 500),
           "d": np.linspace(0.005, 0.02, 400),
       },
       method="ode45",
       duration=100.0,
   )

Cubie builds every combination of parameters in this case; it creates a run
for every ``b`` value combined with every ``d`` value. This creates
:math:`500 \times 400 = 200000` IVPs in this case.

Verbatim grids
~~~~~~~~~~~~~~
With ``grid_type="verbatim"``, cubie pairs your values run by run, so
every array must have the same length:

.. code-block:: python

   rng = np.random.default_rng()
   b_values = rng.uniform(0.01, 0.05, 10000)
   d_values = rng.uniform(0.005, 0.02, 10000)

   result = Solver(LV, algorithm="ode45").solve(
       parameters={"b": b_values, "d": d_values},
       duration=100.0,
       grid_type="verbatim",
   )

This creates 10,000 runs, one for each same-index pair of ``b`` and ``d`` values.

Example: heatmap from a two-parameter sweep
-------------------------------------------

Because cubie lays combinatorial runs out in C order, a ``reshape`` turns
the results of the combinatorial example above back into a grid:

.. code-block:: python

   import matplotlib.pyplot as plt

   result = solve_ivp(
       LV,
       parameters={
           "b": np.linspace(0.01, 0.05, 500),
           "d": np.linspace(0.005, 0.02, 400),
       },
       method="ode45",
       duration=100.0,
   )
   final_x_grid = result.time_domain_array[-1, 0, :].reshape(500, 400)

   fig, ax = plt.subplots()
   ax.imshow(
       final_x_grid,
       origin="lower",
       extent=[0.005, 0.02, 0.01, 0.05],
       aspect="auto",
   )
   ax.set_xlabel("d")
   ax.set_ylabel("b")
   ax.set_title("Final prey population")
   plt.show()
