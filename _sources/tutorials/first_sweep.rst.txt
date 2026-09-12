Tutorial 1: Your First Parameter Sweep
======================================

This tutorial walks from an ODE definition to a heatmap showing how
the system's behaviour varies across a grid of parameter values.  The
whole workflow is about 30 lines of code.

The example system is the Lotka--Volterra predator--prey model, in
which prey ``x`` grows and gets eaten while predators ``y`` eat and
die off.

Step 1: Define the system
-------------------------

:func:`~cubie.odesystems.symbolic.symbolicODE.create_ODE_system`
accepts either a plain Python function or a string of equations.  The
function form will feel familiar if you have used
``scipy.integrate.solve_ivp``: it takes time, a state object, and a
container of named values, and returns the derivatives in state
order.  States and named values can be accessed by attribute
(``y.x``), by name (``y["x"]``), or by index (``y[0]``):

.. code-block:: python

   import numpy as np
   import cubie as qb

   def lotka_volterra(t, y, p):
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -p.c * y.y + p.d * y.x * y.y
       return [dx, dy]

   LV = qb.create_ODE_system(
       lotka_volterra,
       constants={"a": 0.1, "c": 0.3},     # fixed for the whole batch
       parameters={"b": 0.02, "d": 0.01},  # can vary per run
       states={"x": 0.5, "y": 0.3},        # initial values
       name="LotkaVolterra",
   )

CuBIE reads the function's source code to build the GPU kernel, so
the function form needs to live somewhere Python can read it back: a
script, a module, or a notebook cell all work, but a string passed
to ``exec`` or ``python -c`` does not.

The same system can be written as equation strings, which is handy
when the model already exists as equations on paper.  Anything of the
form ``dx = ...`` defines a state variable ``x``:

.. code-block:: python

   LV = qb.create_ODE_system(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       constants={"a": 0.1, "c": 0.3},
       parameters={"b": 0.02, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )

The keyword arguments sort your symbols into roles:

- ``states`` are the variables the solver integrates.  The dict
  values are default initial conditions; the ``y0`` argument of
  :func:`~cubie.batchsolving.solver.solve_ivp` overrides them per
  run.  For the string form the state names can also be inferred
  from the ``dx = ...`` left-hand sides; the function form requires
  ``states`` so CuBIE knows what the returned derivatives refer to.
- ``parameters`` can take a different value in every run of the
  batch.  Sweeps operate on parameters and initial conditions.
- ``constants`` hold one value for the whole batch and are baked
  into the compiled GPU code, which makes the kernel faster.  A
  value you will never sweep belongs here.
- Any right-hand-side symbol you never declared is inferred as a
  parameter with a default value of 0.0, and CuBIE emits a warning
  naming it.  Declaring everything explicitly keeps the warning
  noise down and catches typos early.

Step 2: Sweep two parameters
----------------------------

Pass an array for each parameter you want to sweep.  With the default
``grid_type="combinatorial"``, CuBIE solves every combination.  Here
that is 1000 x 1000 = 1,000,000 initial value problems, a batch size
that a GPU handles comfortably:

.. code-block:: python

   b_values = np.linspace(0.01, 0.05, 1000)
   d_values = np.linspace(0.005, 0.02, 1000)

   result = qb.solve_ivp(
       LV,
       y0={"x": 0.5, "y": 0.3},
       parameters={"b": b_values, "d": d_values},
       method="ode45",
       duration=50.0,
       save_every=0.5,
   )

Initial values accept plain floats; pass an array only when you want
to sweep the initial state as well.  ``method`` selects the
integration algorithm, and ``"ode45"`` selects the adaptive 5th-order
Dormand--Prince pair, the same method behind MATLAB's ``ode45`` and a
good default for non-stiff problems.  ``save_every=0.5`` records a
snapshot every half time-unit.  If the batch outgrows GPU memory,
CuBIE splits it into chunks along the run axis and solves the chunks
in sequence, so batch size is limited by host memory rather than
device memory.

Step 3: Look at the results
---------------------------

The returned :class:`~cubie.batchsolving.solveresult.SolveResult`
holds a 3-D array indexed ``[time, variable, run]``:

.. code-block:: python

   trajectories = result.time_domain_array
   print(trajectories.shape)        # (101, 2, 1000000)
   print(result.time_domain_legend) # which variable is which index

   # Trajectory of prey (variable 0) in the first run:
   prey_run0 = trajectories[:, 0, 0]

Before trusting the numbers, check that every run integrated cleanly:

.. code-block:: python

   assert np.all(result.status_codes == 0), result.status_messages

Step 4: Map runs back to parameters
-----------------------------------

Combinatorial runs are laid out in C-order: the *last* parameter you
passed (``d``) varies fastest.  A ``reshape`` therefore recovers the
2-D parameter grid directly:

.. code-block:: python

   final_prey = trajectories[-1, 0, :]              # final x, per run
   final_prey_grid = final_prey.reshape(1000, 1000)  # rows: b, cols: d

Step 5: Plot
------------

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()
   im = ax.imshow(
       final_prey_grid,
       origin="lower",
       extent=[d_values[0], d_values[-1], b_values[0], b_values[-1]],
       aspect="auto",
   )
   ax.set_xlabel("d (predator growth per prey eaten)")
   ax.set_ylabel("b (predation rate)")
   ax.set_title("Final prey population")
   fig.colorbar(im)
   fig.savefig("lv_sweep.png", dpi=150)

That is the whole workflow: define the system once, sweep it in a
single call, reshape, and plot.

Where to go next
----------------

- Sweep initial values too by passing arrays in ``y0`` exactly like
  parameters (:doc:`/user_guide/batching`).
- Record statistics (mean, peaks, ...) instead of full trajectories
  to save memory: :doc:`extracting_summaries`.
- :doc:`stiff_systems` covers implicit methods for systems that
  mix fast and slow dynamics.
