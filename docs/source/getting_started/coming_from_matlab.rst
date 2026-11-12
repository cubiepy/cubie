Coming from MATLAB
==================

# Weird robot language. Instead you'd write something from the voice of: Most parts of a MATLAB script just need translated into Python, then they'll work. The main difference is that cubie has a single "solve" function and takes the integration method as an argument, instead of one function to call per algorithm.

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
# Is there any python style guidance on using import acronyms? I imagine it's cleaner just to type cubie throughout. In this case you'd probablyjust do form cubie import solve_ivp

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
# Does save_every cause a match to the MATLAB settings? is that default?
# I can't see where trajectories matches matlab here. I think we'd assign directly to t, y.
   trajectories = result.as_numpy["time_domain_array"]

Key changes:

- Python indexing: square brackets, count from 0 instead of 1.
- Parameters as arguments: cubie can't capture local variables from inside the script like MATLAB does. You need to pass parameter values as `parameters=` when you call the solve function. 
- Time is different: the total integration span is given as a starting time t0 and a duration is given as duration and save frequency (save_every), rather than a span
- The "odeset" options are given as individual keyword arguments, see [optional arguments]
# Is that return point valid? can they not return an array?
- Results come back as one array for all batches, as part of a solveResult object, see: results
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
  ``result`` holds every run in the batch — see :doc:`/user_guide/results`.

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
                      ``"crank_nicolson"``, or ``"firk"`` # is ode23s not a rosenbrock23 solver?
===================== ==============================================

See :doc:`/user_guide/choosing_algorithms` for guidance on the stiff solvers.

Replacing a parfor sweep
--------------------------
Cubie directly replaces a parfor or for loop if you're already doing batch integration in MATLAB.

.. code-block:: matlab

   mus = linspace(0.5, 4, 1024);
   parfor i = 1:numel(mus)
       f = @(t, y) [y(2); mus(i)*(1 - y(1)^2)*y(2) - y(1)];
       [t, y] = ode45(f, [0 20], [1; 0]);
       results{i} = y;
   end

In cubie the solve call takes all of the input values instead:

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
combination of the supplied values by default, see :doc:`/user_guide/batching` other options.

# This looks like a restatement not required.
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

# This looks like it's got nothing to do with MATLAB.
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
:doc:`/user_guide/systems` for everything the string form supports.
