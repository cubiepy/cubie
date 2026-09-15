Batching and Parameter Sweeps
=============================

CuBIE's main strength is solving many IVPs in parallel.  This page
explains how to set up parameter sweeps and map results back to inputs.

Combinatorial Grids
-------------------

By default, :func:`~cubie.solve_ivp` uses ``grid_type="combinatorial"``:
every combination of the supplied parameter arrays is solved.

.. code-block:: python

   import cubie as qb
   import numpy as np

   def lotka_volterra(t, y, p):
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -p.c * y.y + p.d * y.x * y.y
       return [dx, dy]

   LV = qb.create_ODE_system(
       lotka_volterra,
       constants={"a": 0.1, "c": 0.3},
       parameters={"b": 0.02, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )

   result = qb.solve_ivp(
       LV,
       y0={"x": np.array([0.5]), "y": np.array([0.3])},
       parameters={
           "b": np.linspace(0.01, 0.05, 50),
           "d": np.linspace(0.005, 0.02, 40),
       },
       method="dormand-prince-54",
       duration=100.0,
       grid_type="combinatorial",
   )

This produces :math:`50 \times 40 = 2000` IVPs.  The result arrays are
indexed in C-order over the Cartesian product of the input arrays.

Verbatim Grids
--------------

When ``grid_type="verbatim"`` (the default for ``Solver.solve``), all
parameter arrays must have the same length and are paired element-wise:

.. code-block:: python

   solver = qb.Solver(LV, algorithm="dormand-prince-54")

   b_vals = np.random.uniform(0.01, 0.05, 2000)
   d_vals = np.random.uniform(0.005, 0.02, 2000)

   result = solver.solve(
       initial_values={"x": np.array([0.5]), "y": np.array([0.3])},
       parameters={"b": b_vals, "d": d_vals},
       duration=100.0,
       grid_type="verbatim",
   )

Run ``i`` uses ``b_vals[i]`` and ``d_vals[i]``.

Mapping Results to Parameters
-----------------------------

Results are ordered to match the input grid.  For verbatim grids the
mapping is one-to-one.  For combinatorial grids the runs are laid out in
C-order (last parameter varies fastest).

Example: Heatmap from a 2-Parameter Sweep
------------------------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   # Assuming result from the combinatorial example above
   # Get the final value of state 'x' for each run
   final_x = result.time_domain_array[-1, 0, :]  # last t, first var
   final_x_grid = final_x.reshape(50, 40)

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
