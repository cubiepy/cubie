Timing Parameters
=================

CuBIE has three groups of timing parameters: **integration bounds** define
the overall time span, **output timing** parameters control when results
are saved, and **step size** parameters control how the integrator advances
through time.

We use this system in the examples on this page:

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
   )
   b_values = np.linspace(0.01, 0.05, 10000)

Integration Bounds
------------------

Three parameters define the time span of integration:

duration
~~~~~~~~

Total length of the recorded integration in simulation time units.
This is the time span over which outputs are saved.

- **Required**: Yes
- **Type**: ``float`` (must be positive)

.. code-block:: python

   # Integrate for 100 time units
   result = solve_ivp(
       LV, parameters={"b": b_values}, method="ode45", duration=100.0
   )

t0
~~

Starting time of the integration. Affects time-dependent drivers and
the timestamps in output arrays.

- **Default**: ``0.0``
- **Type**: ``float``

.. code-block:: python

   # Start integration at t=5.0
   result = solve_ivp(
       LV,
       parameters={"b": b_values},
       method="ode45",
       t0=5.0,
       duration=10.0,
   )
   # Cubie records output from t=5.0 to t=15.0

settling_time
~~~~~~~~~~~~~

Lead-in period before recording begins. The integrator runs for this
duration starting from ``t0``, but no output is saved. This is useful
for allowing transients to decay before collecting data.

- **Default**: ``0.0``
- **Type**: ``float`` (non-negative)

.. code-block:: python

   # Let the system settle for 50 time units, then record 100
   result = solve_ivp(
       LV,
       parameters={"b": b_values},
       method="ode45",
       settling_time=50.0,
       duration=100.0,
   )
   # Cubie integrates from t=0 to t=150 and records from t=50 to t=150

The total integration time is ``settling_time + duration``. The
integration starts at ``t0`` and runs until ``t0 + settling_time + duration``.

.. code-block:: text

   Example: t0=5, settling_time=20, duration=50

   t=5                 t=25                                    t=75
   |                    |                                       |
   |<-- settling_time ->|<------------ duration --------------->|
   |       (20)         |                 (50)                  |
   |                    |                                       |
   +--------------------+---------------------------------------+
   ^                    ^                                       ^
   |                    |                                       |
   Integration      Recording                            Integration
   starts           starts                               ends
   (no output)      (output saved)

Output Timing Parameters
------------------------

Output timing controls how often CuBIE records results during integration.

save_every
~~~~~~~~~~

Time interval between saved state and observable snapshots. Each saved
point includes the current time, state values, and observable values
(depending on your ``output_types`` setting).

- **Default**: If not specified but state/observable outputs are requested,
  CuBIE saves only at the start (t=0 or t=settling_time) and end of the
  integration ("save_last" mode).
- **Type**: ``float`` (seconds of simulation time)

.. code-block:: python

   # Save state every 0.1 seconds of simulation time
   result = solve_ivp(
       LV,
       parameters={"b": b_values},
       method="ode45",
       save_every=0.1,
       duration=10.0,
   )
   # Cubie saves 101 points, including t=0

   # Without save_every, cubie saves only the first and last points
   result = solve_ivp(
       LV,
       parameters={"b": b_values},
       method="ode45",
       duration=10.0,
       output_types=["state"],
   )

summarise_every
~~~~~~~~~~~~~~~

Length of each summary window. CuBIE uses **fixed, non-overlapping windows**
for summary statistics. At the end of each window, metrics (mean, max, RMS,
etc.) are computed from all samples taken during that window, written to
output, and the accumulator resets for the next window.

- **Default**: If not specified but summary outputs are requested, one
  summary is calculated and saved at the end of the integration.
- **Type**: ``float`` (seconds of simulation time)

.. code-block:: python

   # Summarise over 10 windows, each 1 second long
   result = solve_ivp(
       LV,
       parameters={"b": b_values},
       method="ode45",
       summarise_every=1.0,
       sample_summaries_every=0.1,
       duration=10.0,
       output_types=["mean"],
   )

   # Without summarise_every, cubie saves one summary at the end
   result = solve_ivp(
       LV,
       parameters={"b": b_values},
       method="ode45",
       sample_summaries_every=0.1,
       duration=10.0,
       output_types=["mean"],
   )

sample_summaries_every
~~~~~~~~~~~~~~~~~~~~~~

Sampling interval within each summary window. This controls how many
samples contribute to each summary calculation. At each sample point,
the current state/observable values are fed into the running accumulator
(e.g., added to a running sum for mean calculation, compared against
current max, etc.).

- **Required** whenever summary outputs are requested; a solve without
  it raises ``ValueError``.
- **Type**: ``float`` (seconds of simulation time)
- **Constraint**: ``summarise_every``, when set, must be an integer
  multiple of ``sample_summaries_every``; ``sample_summaries_every``
  must fit inside ``duration``.

.. code-block:: python

   # Take 100 samples in each 1-second window
   result = solve_ivp(
       LV,
       parameters={"b": b_values},
       method="ode45",
       sample_summaries_every=0.01,
       summarise_every=1.0,
       duration=10.0,
       output_types=["mean", "max"],
   )

Higher sampling rates give more accurate summaries but increase
computational overhead.

Summary Window Behaviour
~~~~~~~~~~~~~~~~~~~~~~~~

The summary system uses fixed, non-overlapping windows:

1. At each ``sample_summaries_every`` interval, current values are sampled
   and fed into accumulators (running sum, running max, etc.)
2. At each ``summarise_every`` interval, final metrics are computed from
   the accumulators, written to output, and accumulators reset
3. The next window starts fresh with no memory of previous windows

This differs from sliding-window approaches where windows overlap.

.. code-block:: text

   Timeline with summarise_every=1.0, sample_summaries_every=0.25:

   t=0.0   t=0.25  t=0.5   t=0.75  t=1.0   t=1.25  ...
     |       |       |       |       |       |
     S       S       S       S      S+W      S      ...

   S = sample taken, W = window closes (summary written, reset)

Relationship Between Output Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Purpose
   * - ``save_every``
     - Interval for storing full state/observable snapshots
   * - ``sample_summaries_every``
     - Interval for sampling values into summary accumulators
   * - ``summarise_every``
     - Window length; summaries computed and reset at this interval

These are independent: you can save states at high frequency while
computing summaries over longer windows, or vice versa.

Step Size Parameters
--------------------

Adaptive controllers use three related parameters to control step sizes:

- ``dt``: Initial step size (starting point for adaptation)
- ``dt_min``: Minimum allowable step size (floor)
- ``dt_max``: Maximum allowable step size (ceiling)

Fixed-step controllers use only ``dt``, which becomes the constant step
size throughout integration.

Default Behaviour
~~~~~~~~~~~~~~~~~

When you don't specify any step size parameters, CuBIE uses these defaults:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Controller
     - Parameter
     - Default Value
   * - Adaptive
     - ``dt_min``
     - ``1e-6``
   * - Adaptive
     - ``dt_max``
     - ``1.0``
   * - Adaptive
     - ``dt``
     - ``sqrt(dt_min * dt_max)`` = ``1e-3``
   * - Fixed
     - ``dt``
     - ``1e-3``

Derivation Rules
~~~~~~~~~~~~~~~~

When you provide some parameters but not others, CuBIE derives the
missing values. The rules differ for adaptive and fixed controllers:

**Adaptive controllers:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - You Provide
     - CuBIE Derives
   * - ``dt`` only
     - ``dt_min = dt / 1000``, ``dt_max = dt * 1000``
   * - ``dt_min`` only
     - ``dt_max = 1.0`` (default, or ``dt_min * 1000`` if ``dt_min``
       is above it), ``dt = sqrt(dt_min * dt_max)``
   * - ``dt_max`` only
     - ``dt_min = 1e-6`` (default, or ``dt_max / 1000`` if ``dt_max``
       is below it), ``dt = sqrt(dt_min * dt_max)``
   * - ``dt_min`` and ``dt_max``
     - ``dt = sqrt(dt_min * dt_max)``
   * - ``dt`` and ``dt_min``
     - ``dt_max = dt * 1000``
   * - ``dt`` and ``dt_max``
     - ``dt_min = dt / 1000``
   * - All three
     - Nothing derived (all values used as given)

**Fixed-step controllers:**

A fixed controller uses ``dt`` as its step if you give it. Otherwise it
uses ``sqrt(dt_min * dt_max)`` if you give both bounds, the one bound if
you give only one, and ``1e-3`` if you give none.

Build a ``Solver`` to read the step sizes cubie derives:

.. code-block:: python

   # Give an adaptive controller dt, and cubie derives the bounds
   solver = Solver(LV, algorithm="ode45", dt=0.01)
   print(solver.dt_min, solver.dt, solver.dt_max)
   # cubie derives dt_min=1e-5 and dt_max=10.0

   # Give the bounds, and cubie computes the initial dt
   solver = Solver(LV, algorithm="ode45", dt_min=1e-5, dt_max=0.1)
   print(solver.dt_min, solver.dt, solver.dt_max)
   # cubie computes dt=0.001

   # Set a fixed step of dt=0.005 either way
   solver = Solver(LV, algorithm="ode45", step_controller="fixed", dt=0.005)
   solver = Solver(
       LV, algorithm="ode45", step_controller="fixed", dt_min=0.005
   )
   print(solver.dt)

Update Behaviour
~~~~~~~~~~~~~~~~

When you call ``solver.update()`` to change step parameters mid-session,
CuBIE tracks which values you originally set versus which were derived:

- **User-set values** are preserved and never automatically adjusted
- **Derived values** may be adjusted if they become inconsistent with
  your changes

For example, if you give only ``dt=0.01``, cubie derives ``dt_min=1e-5``
and ``dt_max=10.0``. If you then update ``dt=0.001``, cubie derives the
bounds again from your new ``dt``, as ``1e-6`` and ``1.0``. If you had also
given ``dt_min=1e-4`` yourself, cubie keeps it at ``1e-4`` and derives only
``dt_max``.

Complete Example
----------------

This example shows all timing parameters working together:

.. code-block:: python

   solver = Solver(LV, algorithm="ode45")

   result = solver.solve(
       initial_values={"x": np.linspace(0.1, 2.0, 100)},
       parameters={"b": np.linspace(0.01, 0.05, 100)},

       # Integration bounds
       t0=0.0,
       settling_time=10.0,    # Discard first 10 time units
       duration=100.0,        # Record 100 time units after settling

       # Output timing
       save_every=0.1,        # Save states every 0.1 time units
       summarise_every=10.0,  # Compute summaries every 10 time units
       sample_summaries_every=0.05,  # Sample for summaries every 0.05 units

       # Step size (adaptive controller)
       dt=0.01,               # Start at 0.01; cubie derives bounds of 1e-5 to 10.0

       # What to save
       output_types=["state", "mean", "max"],
   )

   # Timeline:
   # t=0 to t=10: settling (no output saved)
   # t=10 to t=110: recording period
   #   - 1001 state saves (every 0.1 from t=10 to t=110)
   #   - 10 summary windows (every 10.0 from t=10 to t=110)
   #   - 200 samples per summary window (every 0.05 over 10.0)
