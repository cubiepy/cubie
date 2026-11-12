Solving ODEs
============

Quick Start with ``solve_ivp``
------------------------------

The fastest way to get started is :func:`~cubie.solve_ivp`. For a simple
system, you can just write the equations straight into this call as strings.
Here we solve the Lotka--Volterra system for a batch of parameter values:

.. code-block:: python

   import numpy as np
   from cubie import solve_ivp

   result = solve_ivp(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       y0={"x": 0.5, "y": 0.3},
       parameters={
           "a": 0.1,
           "b": np.linspace(0.01, 0.05, 100),
           "c": 0.3,
           "d": np.linspace(0.005, 0.02, 100),
       },
       method="ode45",
       duration=100.0,
   )

Cubie solves every combination of the 100 values of ``b`` and the 100
values of ``d``: 10,000 runs in all. ``result`` is a
:class:`~cubie.batchsolving.solveresult.SolveResult`, from which you can
read the time-domain output, summaries, and metadata. See :doc:`results` for how to
inspect it, and :doc:`inputs` for every way to give cubie your values.

The ``Solver`` Class
--------------------

If you plan to solve the same system multiple times (e.g. exploring
different parameter sets interactively), build the system once, create a
:class:`~cubie.Solver` for it and call
:meth:`~cubie.batchsolving.solver.Solver.solve` repeatedly. You avoid
some overhead on each solve this way, and you don't clutter up your solve
calls with definitions of the system every time:

.. code-block:: python

   import numpy as np
   from cubie import Solver, create_ODE_system

   LV = create_ODE_system(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )
   solver = Solver(LV, algorithm="ode45")
   b_values = np.linspace(0.01, 0.05, 10000)

   result_a = solver.solve(
       initial_values={"x": 0.5, "y": 0.3},
       parameters={"b": b_values},
       duration=100.0,
   )

   result_b = solver.solve(
       initial_values={"x": 1.0, "y": 0.5},
       parameters={"b": b_values},
       duration=200.0,
   )

Cubie compiles the kernel on the first call, and reuses it on later calls.
In a large
system with a complex algorithm (higher-order DIRK and FIRK algorithms are
the worst culprits, with large CellML systems) this can take up to several
minutes, so it is the only part of the cubie process that requires some
patience. You can set ``time_logging_level="verbose"`` to watch the steps
in your terminal output. Cubie compiles the Lotka--Volterra system above in
about a second, so this is a big-system problem only.

If you want to change a setting in the live solver, you can pass it as a
keyword argument to the ``solve`` method, or call
:meth:`~cubie.batchsolving.solver.Solver.update`. Here we tighten ``rtol``
with ``update``, then pass a tighter ``atol`` to ``solve``:

.. code-block:: python

   solver.update(rtol=1e-8)
   result_c = solver.solve(
       parameters={"b": b_values},
       duration=100.0,
       atol=1e-10,
   )

Cubie keeps both settings for later solves on this solver.

Configuring a solve
-------------------

Set how long cubie integrates for with ``duration``, when it starts with
``t0``, and how long it integrates before it starts recording with
``settling_time``; see :doc:`timing` for these and for how often cubie
saves output. To
choose an integration algorithm and its step controller, see
:doc:`/user_guide/algorithms_and_tuning/choosing_algorithms`. Every other
setting, from tolerances to output selection, is a keyword argument; see
:doc:`configuration` for where each one goes and
:doc:`/user_guide/algorithms_and_tuning/optional_arguments` for what each
one does.
