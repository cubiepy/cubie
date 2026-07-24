Tutorial 2: Summaries Instead of Trajectories
=============================================

A million-run batch that saves full trajectories will exhaust your
GPU memory long before it runs out of compute.  Usually you do not
want the trajectories anyway; you want a number per run, such as the
mean, the peak count, or the oscillation amplitude.  This tutorial
shows how to compute those statistics *on the GPU during
integration*, so the full time series never exists anywhere.

Step 1: A system worth summarising
----------------------------------

The same Lotka--Volterra system as :doc:`first_sweep`.  Its
populations oscillate, so per-run statistics are meaningful:

.. code-block:: python

   import numpy as np
   import cubie as qb

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

Step 2: Ask for statistics, not trajectories
--------------------------------------------

``output_types`` controls what is recorded.  Skip ``"state"`` and
list summary metrics instead:

.. code-block:: python

   result = qb.solve_ivp(
       LV,
       y0={"x": 0.5, "y": 0.3},
       parameters={
           "b": np.linspace(0.01, 0.05, 20),
           "d": np.linspace(0.005, 0.02, 20),
       },
       method="ode45",
       duration=50.0,
       output_types=["mean", "max"],
       summarise_every=50.0,
   )

Summaries run on two clocks.  ``summarise_every`` sets the window
length: each metric produces one value per window per variable.
``sample_summaries_every`` sets the measurement cadence inside each
window: the metric is computed from the trajectory sampled at that
interval, and it defaults to a tenth of the window length.  With
``summarise_every=10.0`` and ``sample_summaries_every=1.0``, the
first ``"mean"`` is the mean of the measurements at t = 1, 2, ...,
10 time-units, the second covers t = 11 through 20, and so on.

Here ``summarise_every=50.0`` makes one window spanning the whole
run, so each metric reduces to a single number per variable per run.
Shorter windows give a statistic per window instead, which is handy
for tracking slow drift.  If you omit ``summarise_every``, CuBIE
defaults to one whole-run window but warns you, because the derived
timing forces a recompile whenever ``duration`` changes.

There are 18 built-in metrics, including ``"rms"``, ``"std"``,
``"peaks"``, and first/second-derivative extrema; the full table is
in :doc:`/user_guide/results`.

Step 3: Read the summaries
--------------------------

Summaries come out as a 3-D array indexed
``[window, summary, run]``.  The friendliest accessor is
``as_numpy_per_summary``, which splits it into one array per metric,
each indexed ``[window, variable, run]``:

.. code-block:: python

   per_metric = result.as_numpy_per_summary
   print(per_metric["max"].shape)      # (1, 2, 400)
   peak_prey = per_metric["max"][0, 0, :]   # window 0, x, all runs
   mean_prey = per_metric["mean"][0, 0, :]

Metrics that share an accumulator are computed together on the GPU:
requesting both ``"max"`` and ``"min"`` runs one combined pass over
the data instead of two.  This fusion is transparent — the result
keys are always the metric names you requested.

Step 4: Trim what you don't need
--------------------------------

Two more levers cut memory and time further:

.. code-block:: python

   result = qb.solve_ivp(
       LV,
       y0={"x": 0.5, "y": 0.3},
       parameters={
           "b": np.linspace(0.01, 0.05, 20),
           "d": np.linspace(0.005, 0.02, 20),
       },
       method="ode45",
       duration=50.0,
       settling_time=20.0,                # discard the transient
       output_types=["mean", "max"],
       summarise_every=50.0,
       summarise_variables=["x"],         # only summarise prey
   )

``summarise_variables`` (and its time-domain sibling
``save_variables``) restricts recording to the variables you name.
``settling_time`` integrates for 20 time-units *before* recording
starts, so start-up transients do not pollute your statistics.
Settling extends the run rather than eating into it: the solver
integrates for ``settling_time + duration`` in total, so the
recorded window is still the full 50 time-units and
``summarise_every=50.0`` produces exactly one summary window per
run.

When to use which output
------------------------

- **Full trajectories** (``"state"``): exploring, debugging, and
  plotting a handful of runs.
- **Summaries**: large sweeps, likelihood-free inference, and
  anything where each run reduces to features.  Memory per run drops
  from ``n_saves x n_variables`` values to a handful.
- **Both at once** works too; the two records run at independent
  cadences (see :doc:`/user_guide/timing`).

The next tutorial, :doc:`stiff_systems`, covers systems that need
implicit methods and time-dependent forcing.
