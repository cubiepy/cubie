Inputs and Parameter Sweeps
===========================

You give every solve two inputs: the initial values of the states, and the
parameter values. Here we show the formats you can give each input in, then
how cubie combines your values into a batch of runs. We use this system in
the examples:

.. code-block:: python

   import numpy as np
   from cubie import Solver, create_ODE_system, solve_ivp

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

   solver = Solver(LV, algorithm="ode45")
   result = solver.solve(
       initial_values={"x": np.linspace(0.1, 2.0, 10000), "y": 0.3},
       parameters={"a": 0.2, "b": np.linspace(0.01, 0.05, 10000)},
       duration=100.0,
   )

Give a single value for anything that is the same in every run, and a list
or array of values for anything you want to vary. Cubie compiles ``a``
above into the code, because you gave it one value, and gives ``b`` a row of
the parameters array. If you set default values for the states or
parameters, you can leave them out of the dicts and cubie will use their
default values. You can leave either dict out entirely, too. If you didn't
set defaults when you built the system, you can set them on the system with
``system.set_default_parameters({"a": 0.2})`` for parameters and
``system.set_initial_values({"y": 0.3})`` for states. If you include a
constant value for a parameter or initial value that has a different
default, the default doesn't change: Cubie presumes that arguments to
``solve`` are intended for that run only. Pass
``fix_constant_parameters=False`` to give every constant value in the dict
a row instead. ``solve_ivp``, ``Solver.compile``, ``Solver.calibrate`` and
``Solver.optimize`` treat their inputs the same way.

Arrays
~~~~~~

If you're looking for something faster (by ~tens of ms per solve) or are
generating values programmatically, you can pass arrays instead. Lay each
array out as ``(n_variables, n_runs)``, in the system's precision, with
its rows ordered by variable name. Give an initial values array one row
per state in your system. Give the parameters array either one row for
every parameter in your system (a full-height parameter array), or one row
per parameter that varies across the batch. Read
``solver.parameter_legend`` to see each row's parameter name.

To pass just the varying parameters, you will need to tell the system which
parameters to expect. To do this, call ``solver.set_swept_parameters`` with
the parameter names you want to vary, or pass the same names to the
``swept_parameters`` argument when building the system. When you sweep a
single parameter, you can pass its values as a 1D array:

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
parameter a single value, cubie fills its whole row with that value. To compile a value into the code instead, set it as the parameter's
default and leave it out of the dict:

.. code-block:: python

   LV.set_default_parameters({"a": 0.2})
   inits, params = solver.prepare_grid_arrays(
       {"x": np.linspace(0.1, 2.0, 10000), "y": 0.3},
       {"b": np.linspace(0.01, 0.05, 10000)},
   )
   result = solver.solve(inits, params, duration=100.0)

Arrays of swept parameters only are intended as a convenience: some models
have hundreds of parameters, and if you're only sweeping two of them you
probably don't want to throw hundreds-high arrays around. If it ends up
being easier to pass full-height parameter arrays because of how you
generate the values, Cubie will spend a few ms scanning the array for
values that are the same across every run, and it will remove those from
the array it passes to the GPU and compile them into your code instead.
You can save a few ms per solve by disabling this with
``fix_constant_parameters=False``, but it may slow down the solve itself if
you have a lot of constant parameters.

Device arrays
~~~~~~~~~~~~~

If you want to go *really* fast and you are comfortable with the CUDA
environment, you can pass arrays on the GPU (device arrays) directly to cut
out the few milliseconds of transfer time as well. Use the same layout and
precision as host arrays. Pair them with ``on_device=True`` to keep an
entire solve → post-process pipeline on the GPU (see :doc:`results`):

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

With ``grid_type="combinatorial"``, the default for ``solve_ivp``, cubie
solves every combination of the values you supply:

.. code-block:: python

   result = solve_ivp(
       LV,
       parameters={
           "b": np.linspace(0.01, 0.05, 500),
           "d": np.linspace(0.005, 0.02, 400),
       },
       method="ode45",
       duration=100.0,
   )

Cubie solves :math:`500 \times 400 = 200000` IVPs here. It lays the runs
out in C order over the combinations, with the last parameter varying
fastest.

Verbatim grids
~~~~~~~~~~~~~~

With ``grid_type="verbatim"``, the default for ``Solver.solve``, cubie
pairs your values run by run, so every array must have the same length:

.. code-block:: python

   rng = np.random.default_rng()
   b_values = rng.uniform(0.01, 0.05, 10000)
   d_values = rng.uniform(0.005, 0.02, 10000)

   result = Solver(LV, algorithm="ode45").solve(
       parameters={"b": b_values, "d": d_values},
       duration=100.0,
   )

Cubie gives run ``i`` the values ``b_values[i]`` and ``d_values[i]``.

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
