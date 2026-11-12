Working with Results
====================

Every call to :func:`~cubie.solve_ivp` or
:meth:`~cubie.batchsolving.solver.Solver.solve` returns a
:class:`~cubie.batchsolving.solveresult.SolveResult`.

When integrating large batches, you end up with large data. Unless you're
checking the method against some known reference as a sanity check, you probably
don't need the full time-domain series across the whole integration. To
cut down the output data, and speed up the solver, you can save only certain
variables, at only certain times, or you can never save the time-domain signals
and calculate summary variables (like averages, or peak times) on the GPU
while integrating.

The examples on this page use the Lotka-Volterra system with one
observable:

.. code-block:: python

   import numpy as np
   from cubie import create_ODE_system, solve_ivp

   LV = create_ODE_system(
       """
       predator_death_rate = c*y
       dx = a*x - b*x*y
       dy = -predator_death_rate + d*x*y
       """,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       observables=["predator_death_rate"],
   )

Output types
------------

Choose what cubie records with the ``output_types`` list. If you leave it
out, cubie records ``["state"]``.

``"state"``
   Time-domain state trajectories.

``"observables"``
   Time-domain observable trajectories.

``"time"``
   The time of each saved point.

``"iteration_counters"``
   The solver's work between saved points: Newton iterations, Krylov
   iterations, accepted steps and rejected steps.

Any summary metric name, such as ``"mean"`` or ``"max"``, adds that metric
to the output; see `Summary metrics`_ below. ``save_every`` sets how often
cubie saves a time-domain point; see :doc:`timing`.

.. code-block:: python

   result = solve_ivp(
       LV,
       parameters={"b": np.linspace(0.01, 0.05, 10000)},
       method="ode45",
       duration=100.0,
       save_every=1.0,
       output_types=["state", "observables", "time", "iteration_counters"],
   )

   counters = result.iteration_counters
   newton_iters = counters[:, 0, :]    # per save point, per run
   krylov_iters = counters[:, 1, :]
   accepted_steps = counters[:, 2, :]
   rejected_steps = counters[:, 3, :]

The iteration counters help you spot runs where an implicit algorithm is
struggling. See :doc:`/theory/solvers` for background on the Newton
solver.

Summary metrics
---------------

Summary metrics let you extract statistics such as the mean, the maximum or
the peaks of a variable without ever writing its full trajectory to VRAM.
cubie updates each metric on the GPU as it integrates. Set how often cubie
samples the variables with ``sample_summaries_every``, which every solve
with summaries needs, and the length of each summary window with
``summarise_every``. If you leave ``summarise_every`` out, cubie calculates
one summary over the whole integration. See :doc:`timing` for how the two
intervals fit together.

.. code-block:: python

   result = solve_ivp(
       LV,
       parameters={"b": np.linspace(0.01, 0.05, 10000)},
       method="ode45",
       duration=100.0,
       sample_summaries_every=0.5,
       output_types=["mean", "max"],
   )

   per_summary = result.as_numpy_per_summary
   mean_prey = per_summary["mean"][0, 0, :]   # window 0, x, every run
   peak_prey = per_summary["max"][0, 0, :]

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Description
   * - ``"mean"``
     - Time-averaged value.
   * - ``"std"``
     - Standard deviation over time.
   * - ``"rms"``
     - Root-mean-square value.
   * - ``"max"``
     - Maximum value.
   * - ``"min"``
     - Minimum value.
   * - ``"extrema"``
     - Both max and min.
   * - ``"max_magnitude"``
     - Maximum absolute value.
   * - ``"peaks"``
     - Positive peaks (local maxima).
   * - ``"negative_peaks"``
     - Negative peaks (local minima).
   * - ``"dxdt_max"``
     - Maximum of the first derivative.
   * - ``"dxdt_min"``
     - Minimum of the first derivative.
   * - ``"dxdt_extrema"``
     - Both max and min of the first derivative.
   * - ``"d2xdt2_max"``
     - Maximum of the second derivative.
   * - ``"d2xdt2_min"``
     - Minimum of the second derivative.
   * - ``"d2xdt2_extrema"``
     - Both max and min of the second derivative.
   * - ``"mean_std"``
     - Mean and standard deviation combined.
   * - ``"std_rms"``
     - Standard deviation and RMS combined.
   * - ``"mean_std_rms"``
     - Mean, standard deviation, and RMS combined.

If you ask for a set of metrics that share running sums, such as
``["max", "min"]`` or ``["mean", "std", "rms"]``, cubie calculates them
together in one combined metric. You still get each result under the name
you asked for.

Selecting variables
-------------------

By default cubie saves and summarises every state and observable. To
reduce memory and improve speed, name only the variables you need with
``save_variables`` for the time-domain output and ``summarise_variables``
for the summaries:

.. code-block:: python

   result = solve_ivp(
       LV,
       parameters={"b": np.linspace(0.01, 0.05, 10000)},
       method="ode45",
       duration=100.0,
       save_every=1.0,
       sample_summaries_every=0.5,
       save_variables=["x"],
       summarise_variables=["x", "y"],
       output_types=["state", "mean", "max"],
   )

If you leave ``summarise_variables`` out, cubie summarises the variables
you save.

Result formats
--------------

A ``SolveResult`` holds each output as a NumPy array, with time (or the
summary window) first, then the variable, then the run:

``state`` and ``observables``
   The state and observable trajectories.

``time_domain_array``
   The states followed by the observables, in one array.
   ``time_domain_legend`` names each of its columns.

``time``
   The time of each saved point, one column per run.

``summaries_array``
   Every summary metric, one row per summary window.
   ``summaries_legend`` names each of its columns, such as ``"x mean"``.

``iteration_counters``
   The solver work between saved points, as described above.

``status_codes`` and ``status_messages``
   One status word per run, ``0`` for success. ``status_messages`` maps
   the index of each failed run to the names of its status codes; see
   :doc:`/troubleshooting`.

``parameter_legend`` and ``fixed_parameter_values``
   ``parameter_legend`` names the parameter of each row of the swept
   parameters array, and ``fixed_parameter_values`` gives the value of
   every parameter cubie compiled into the code.

You can also convert the result:

- ``result.as_numpy`` returns a dict of the arrays and legends.
- ``result.as_numpy_per_summary`` returns the same, with the summaries
  split into one array per metric.
- ``result.as_pandas`` returns pandas DataFrames of the time-domain output
  and the summaries.

Device-resident results
~~~~~~~~~~~~~~~~~~~~~~~

A normal solve copies the outputs from the GPU to the host before it
returns. If you feed the outputs straight into more GPU work, pass
``on_device=True`` to ``Solver.solve`` to skip that copy. You get a
:class:`~cubie.batchsolving.solveresult.DeviceSolveResult` of device
arrays instead:

.. code-block:: python

   from cubie import Solver

   solver = Solver(LV, algorithm="ode45")
   result = solver.solve(
       parameters={"b": np.linspace(0.01, 0.05, 10000)},
       duration=100.0,
       on_device=True,
   )
   result.stream.synchronize()      # the outputs are ready after this
   device_state = result.state      # GPU array, no host copy made

Keep these points in mind:

- The solve returns without waiting for the GPU. You can queue more work
  on ``result.stream`` straight away; to read the outputs on the host,
  synchronise that stream first and then copy them, for example with
  ``result.state.copy_to_host()``.
- The next ``solve()`` on the same solver overwrites the arrays.
- cubie skips the host-side work, so you get no output legends and no NaN
  masking of failed runs.
- The batch must fit on the GPU in one chunk, otherwise the solve raises
  ``ValueError``.
- ``solve_ivp`` closes its solver before it returns, so it has no
  ``on_device`` option.
