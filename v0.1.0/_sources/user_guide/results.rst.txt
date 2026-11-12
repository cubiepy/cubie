Working with Results
====================

Every call to :func:`~cubie.solve_ivp` or
:meth:`~cubie.batchsolving.solver.Solver.solve` returns a
:class:`~cubie.batchsolving.solveresult.SolveResult`.

The ``SolveResult`` Object
--------------------------

Key attributes:

``time_domain_array``
   3-D NumPy array with shape ``(n_time_points, n_variables, n_runs)``.

``summaries_array``
   3-D NumPy array with shape ``(n_summary_windows, n_summaries,
   n_runs)`` — one row per summary window.

``time``
   1-D array of time values corresponding to the first axis of
   ``time_domain_array``.

``iteration_counters``
   3-D integer array with shape ``(n_time_points, 4, n_runs)``.  The
   four channels are, in order: Newton iterations, Krylov iterations,
   accepted steps, and rejected steps since the previous save point.

``status_codes`` / ``status_messages``
   Per-run integration status.  ``status_codes`` is an ``int32`` array
   of bit flags (one per run, ``0`` means success);
   ``status_messages`` decodes each into a list of human-readable
   flag names such as ``MAX_NEWTON_ITERATIONS_EXCEEDED``.

``time_domain_legend``
   Dictionary mapping column indices to variable names.

``summaries_legend``
   Dictionary mapping row indices to summary labels.

Convenience accessors:

- ``result.as_numpy`` --- returns a dict of NumPy arrays.
- ``result.as_pandas`` --- returns a dict of pandas DataFrames.
- ``result.as_numpy_per_summary`` --- splits summaries by metric type.

Output Types
------------

Control what is saved with the ``output_types`` list (or via
``output_settings`` on the ``Solver``):

``"state"``
   Time-domain state trajectories.

``"observables"``
   Time-domain observable trajectories.

``"time"``
   Time point array.

``"iteration_counters"``
   Newton/Krylov iteration counts and accepted/rejected step counts
   per save interval.

Any summary metric name (e.g. ``"mean"``, ``"max"``, ``"peaks"``) adds
that metric to the output.

Time-Domain vs Summary Output
-----------------------------

Time-domain saves and summary metrics operate at independent cadences:

- ``dt_save`` (or ``save_every``) controls how often state snapshots are
  written.
- ``dt_summarise`` (or ``summarise_every``) controls how often the
  summary accumulators are updated.
- ``sample_summaries_every`` controls the sub-step sampling rate for
  metrics that interpolate between steps.

Using summaries lets you extract statistics (mean, max, peaks, etc.)
without ever writing the full trajectory to VRAM.

Built-in Summary Metrics
------------------------

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

Requesting a full set of constituent metrics that share running sums
(for example ``["max", "min"]`` or ``["mean", "std", "rms"]``) is
automatically computed using the matching combined metric internally,
but each result is still reported under its requested name (``"max"``,
``"min"``, and so on) — the fusion is a transparent performance
optimisation, not a change to the reported legend.

Selecting Variables to Save
---------------------------

By default all states and observables are saved.  To reduce memory and
improve speed, select only the variables you need:

.. code-block:: python

   result = qb.solve_ivp(
       system,
       y0=y0,
       parameters=params,
       method="dormand-prince-54",
       duration=10.0,
       save_variables=["x"],
       summarise_variables=["x", "y"],
       output_types=["state", "mean", "max"],
   )

Iteration Counters and Run Status
---------------------------------

Requesting ``"iteration_counters"`` in ``output_types`` records solver
effort at every save point — useful for spotting runs where an
implicit algorithm is struggling:

.. code-block:: python

   counters = result.iteration_counters
   newton_iters = counters[:, 0, :]    # per save point, per run
   krylov_iters = counters[:, 1, :]
   accepted_steps = counters[:, 2, :]
   rejected_steps = counters[:, 3, :]

Separately, every result carries a per-run status word.  A run that
hit a solver limit is flagged there:

.. code-block:: python

   failed = result.status_codes != 0
   print(result.status_messages)  # e.g. ["MAX_NEWTON_ITERATIONS_EXCEEDED"]

See :doc:`/theory/solvers` for background on the Newton solver.

Example: Requesting Summaries
-----------------------------

.. code-block:: python

   import cubie as qb
   import numpy as np

   # ... (system definition omitted for brevity)

   result = qb.solve_ivp(
       system,
       y0=y0,
       parameters=params,
       method="dormand-prince-54",
       duration=50.0,
       output_types=["mean", "std"],
       summarise_variables=["x"],
   )

   per_summary = result.as_numpy_per_summary
   print("Mean of x:", per_summary["mean"])
   print("Std of x:", per_summary["std"])
