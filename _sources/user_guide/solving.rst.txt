Solving ODEs
============

Once you have defined an ODE system (see :doc:`systems`), CuBIE provides
two ways to solve it: the convenience function
:func:`~cubie.batchsolving.solver.solve_ivp` and the reusable
:class:`~cubie.batchsolving.solver.Solver` class.

Quick Start with ``solve_ivp``
------------------------------

The fastest way to get results is :func:`~cubie.solve_ivp`.  Here we
solve the Lotka--Volterra system for a single initial condition:

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
       parameters={"b": np.array([0.02]), "d": np.array([0.01])},
       method="dormand-prince-54",
       duration=100.0,
   )

``result`` is a :class:`~cubie.batchsolving.solveresult.SolveResult` that
holds the time-domain output, summaries, and metadata.  See
:doc:`results` for how to inspect it.

.. note::

   CuBIE is a *batch* solver---it is designed to solve many IVPs at once.
   A single-system call works, but the real performance gains come from
   passing arrays of initial values or parameters.  See :doc:`batching`.

Changing Initial States
-----------------------

Pass arrays for any states you want to vary across the batch:

.. code-block:: python

   result = qb.solve_ivp(
       LV,
       y0={
           "x": np.linspace(0.1, 2.0, 500),
           "y": np.array([0.3]),  # same for all runs
       },
       parameters={"b": np.array([0.02]), "d": np.array([0.01])},
       method="dormand-prince-54",
       duration=100.0,
   )

States provided as length-1 arrays are broadcast to all runs.

The ``Solver`` Class
--------------------

If you plan to solve the same system multiple times (e.g. exploring
different parameter sets interactively), create a
:class:`~cubie.Solver` once and call
:meth:`~cubie.batchsolving.solver.Solver.solve` repeatedly.  This avoids
recompiling the CUDA kernel on every call:

.. code-block:: python

   solver = qb.Solver(LV, algorithm="dormand-prince-54")

   result_a = solver.solve(
       initial_values={"x": np.array([0.5]), "y": np.array([0.3])},
       parameters={"b": np.linspace(0.01, 0.05, 1000),
                   "d": np.full(1000, 0.01)},
       duration=100.0,
   )

   result_b = solver.solve(
       initial_values={"x": np.array([1.0]), "y": np.array([0.5])},
       parameters={"b": np.linspace(0.01, 0.05, 1000),
                   "d": np.full(1000, 0.01)},
       duration=200.0,
   )

The first call compiles the kernel; subsequent calls reuse it.

Two further ``Solver`` methods are useful in interactive loops:
:meth:`~cubie.batchsolving.solver.Solver.update` reconfigures a live
solver (tolerances, algorithm, output settings) without rebuilding it
from scratch, and
:meth:`~cubie.batchsolving.solver.Solver.build_grid` pre-builds the
input grid once so repeated ``solve`` calls with the same batch layout
skip grid construction.

Initial values and parameters that already live on the GPU (CuPy or
Numba device arrays in ``(n_variables, n_runs)`` layout, matching the
system precision) are wired directly into the kernel with no
host-to-device transfer; pair them with ``on_device=True`` to keep an
entire solve → post-process pipeline on the GPU (see :doc:`results`).

The ``duration`` Parameter
--------------------------

``duration`` sets the total integration time.  You can also specify
``t0`` (default 0) and ``settling_time``.  When ``settling_time`` is
non-zero, the solver integrates for that period *before* it starts
recording output, useful for discarding transients.

Next Steps
----------

- :doc:`batching` --- parameter sweeps and grid types.
- :doc:`results` --- inspecting and post-processing solver output.
- :doc:`choosing_algorithms` --- which algorithm to pick.
