Coming from MATLAB
==================

cubie's interface follows the same shape as MATLAB's ODE suite: a
right-hand-side function of ``(t, y)`` returning the derivatives, a
solver call naming the method, and tolerances as options. This page
ports an ``ode45`` script and points out each translation.

A direct port
-------------

.. code-block:: matlab
   :caption: MATLAB

   mu = 1.5;
   f = @(t, y) [y(2); mu*(1 - y(1)^2)*y(2) - y(1)];
   opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
   [t, y] = ode45(f, [0 20], [1; 0], opts);

.. code-block:: python
   :caption: cubie

   import cubie as qb

   def f(t, y, mu):
       return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

   result = qb.solve_ivp(
       f,
       y0={"x": [1.0], "v": [0.0]},
       parameters={"mu": [1.5]},
       duration=20.0,
       method="ode45",
       rtol=1e-6,
       atol=1e-8,
       save_every=0.1,
   )
   trajectories = result.as_numpy["time_domain_array"]

The translations, line by line:

* **Indexing starts at 0.** MATLAB's ``y(1)``, ``y(2)`` become
  ``y[0]``, ``y[1]``. If you prefer names to indices, cubie also
  accepts ``y.x`` / ``y.v`` in the function body.
* **Parameters are arguments, not workspace variables.** The anonymous
  function captured ``mu`` from the workspace; cubie requires it as a
  named function argument declared in ``parameters=``. A workspace-style
  bare name in the body raises an error at build time telling you what
  to declare.
* **The return is a list, not a column vector.** ``[dx; dv]`` becomes
  ``[dx, dv]`` (or a dict ``{"x": dx, "v": dv}``, which also names the
  states).
* **``tspan`` becomes a duration.** ``[0 20]`` becomes
  ``duration=20.0`` with an optional ``t0``.
* **``odeset`` options are keyword arguments.** ``RelTol``/``AbsTol``
  are ``rtol``/``atol``; ``MaxStep`` is ``dt_max``.
* **Outputs are batch arrays.** Instead of ``[t, y]`` for one run,
  ``result`` holds every run in the batch — see :doc:`results`.

Method names
------------

===================== ==============================================
MATLAB                cubie ``method=``
===================== ==============================================
``ode45``             ``"ode45"``
``ode23``             ``"ode23"``
``ode89``             ``"dop853"`` (closest high-order explicit)
``ode15s``, ``ode23s`` no direct equivalent; for stiff systems use
                      ``"rosenbrock"``, ``"backwards_euler"``,
                      ``"crank_nicolson"``, or ``"firk"``
===================== ==============================================

See :doc:`choosing_algorithms` for guidance on the stiff solvers.

Replacing the parfor sweep
--------------------------

The MATLAB pattern cubie replaces is the sweep loop:

.. code-block:: matlab

   mus = linspace(0.5, 4, 1024);
   parfor i = 1:numel(mus)
       f = @(t, y) [y(2); mus(i)*(1 - y(1)^2)*y(2) - y(1)];
       [t, y] = ode45(f, [0 20], [1; 0]);
       results{i} = y;
   end

In cubie the sweep is the input, and one call integrates every run in
parallel on the GPU:

.. code-block:: python

   import numpy as np

   result = qb.solve_ivp(
       f,
       y0={"x": [1.0], "v": [0.0]},
       parameters={"mu": np.linspace(0.5, 4.0, 1024)},
       duration=20.0,
       method="ode45",
       save_every=0.1,
   )

Supplying arrays for several parameters (or initial values) runs every
combination by default; see :doc:`batching` for pairing inputs
run-for-run instead.

Repeated solves: build the system once
--------------------------------------

``qb.solve_ivp(f, ...)`` rebuilds the system each call. When you solve
the same equations many times with different inputs, create the system
once and keep a :class:`~cubie.batchsolving.solver.Solver` so repeat
solves reuse the compiled GPU kernel:

.. code-block:: python

   system = qb.create_ODE_system(
       f,
       states={"x": 1.0, "v": 0.0},
       parameters={"mu": 1.5},
   )
   solver = qb.Solver(system, algorithm="ode45", save_every=0.1)
   result = solver.solve(
       {"x": [1.0], "v": [0.0]},
       {"mu": np.linspace(0.5, 4.0, 1024)},
       duration=20.0,
       grid_type="combinatorial",
   )

Equation strings
----------------

If you would rather keep the equations looking like the mathematics,
cubie also accepts them as strings — no indexing at all:

.. code-block:: python

   result = qb.solve_ivp(
       ["dx = v", "dv = mu*(1 - x**2)*v - x"],
       y0={"x": [1.0], "v": [0.0]},
       parameters={"mu": [1.5]},
       duration=20.0,
       method="ode45",
   )

Note the power operator is Python's ``**``, not MATLAB's ``^``. See
:doc:`systems` for everything the string form supports.
