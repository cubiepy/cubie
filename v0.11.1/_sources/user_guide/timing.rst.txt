Timing Parameters
=================

CuBIE has three groups of timing parameters: **step size** parameters
control how the integrator advances through time, **output timing**
parameters control when results are saved, and **integration bounds**
define the overall time span.

.. contents:: On this page
   :local:
   :depth: 2

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
     - ``dt_min = dt / 100``, ``dt_max = dt * 100``
   * - ``dt_min`` only
     - ``dt_max = 1.0`` (default), ``dt = sqrt(dt_min * dt_max)``
   * - ``dt_max`` only
     - ``dt_min = 1e-6`` (default), ``dt = sqrt(dt_min * dt_max)``
   * - ``dt_min`` and ``dt_max``
     - ``dt = sqrt(dt_min * dt_max)``
   * - ``dt`` and ``dt_min``
     - ``dt_max = dt * 100``
   * - ``dt`` and ``dt_max``
     - ``dt_min = dt / 100``
   * - All three
     - Nothing derived (all values used as given)

**Fixed-step controllers:**

Fixed controllers collapse ``dt``, ``dt_min``, and ``dt_max`` to a single
value. If you provide any of them, the first non-None value in the order
``dt``, ``dt_min``, ``dt_max`` is used as the step size.

Example usage:

.. code-block:: python

   # Adaptive: provide dt, bounds derived automatically
   solver.solve(..., dt=0.01)
   # Results in: dt_min=0.0001, dt=0.01, dt_max=1.0

   # Adaptive: provide bounds, initial dt computed
   solver.solve(..., dt_min=1e-5, dt_max=0.1)
   # Results in: dt_min=1e-5, dt=0.001, dt_max=0.1

   # Fixed: any of these sets dt=0.005
   solver.solve(..., step_controller="fixed", dt=0.005)
   solver.solve(..., step_controller="fixed", dt_min=0.005)

Update Behaviour
~~~~~~~~~~~~~~~~

When you call ``solver.update()`` to change step parameters mid-session,
CuBIE tracks which values you originally set versus which were derived:

- **User-set values** are preserved and never automatically adjusted
- **Derived values** may be adjusted if they become inconsistent with
  your changes

For example, if you originally provided only ``dt=0.01`` (so ``dt_min``
and ``dt_max`` were derived), then later update ``dt=0.001``:

- ``dt`` changes to ``0.001`` (your new value)
- ``dt_min`` may be adjusted if ``0.001 < dt_min`` (derived value, can change)
- ``dt_max`` stays at ``1.0`` (derived value, still valid)

But if you originally provided ``dt_min=1e-4`` explicitly, that value
won't be changed even if it would make more sense to adjust it.

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
   solver.solve(..., save_every=0.1, duration=10.0)
   # Results in 101 saved points (including t=0)

   # No save_every specified: only initial and final states saved
   solver.solve(..., duration=10.0, output_types=["state"])
   # Results in 2 saved points

summarise_every
~~~~~~~~~~~~~~~

Length of each summary window. CuBIE uses **fixed, non-overlapping windows**
for summary statistics. At the end of each window, metrics (mean, max, RMS,
etc.) are computed from all samples taken during that window, written to
output, and the accumulator resets for the next window.

- **Default**: If not specified but summary outputs are requested, defaults
  to ``duration`` (one summary window covering the entire integration).
  This default is derived from the duration, so CuBIE emits a
  ``UserWarning``: if ``duration`` changes on a later solve, the kernel
  must recompile once.  Set ``summarise_every`` explicitly to avoid it.
- **Type**: ``float`` (seconds of simulation time)

.. code-block:: python

   # 10 summary windows, each 1 second long
   solver.solve(..., summarise_every=1.0, duration=10.0, output_types=["mean"])

   # No summarise_every: one summary over entire duration
   solver.solve(..., duration=10.0, output_types=["mean"])

sample_summaries_every
~~~~~~~~~~~~~~~~~~~~~~

Sampling interval within each summary window. This controls how many
samples contribute to each summary calculation. At each sample point,
the current state/observable values are fed into the running accumulator
(e.g., added to a running sum for mean calculation, compared against
current max, etc.).

- **Default**: ``summarise_every / 10`` (10 samples per window) when
  you set ``summarise_every``.  When you set neither,
  ``duration / 100`` (100 samples in the single whole-run window).
- **Type**: ``float`` (seconds of simulation time)
- **Constraint**: ``summarise_every`` must be an integer multiple of
  ``sample_summaries_every``

.. code-block:: python

   # 100 samples per 1-second window
   solver.solve(
       ...,
       sample_summaries_every=0.01,
       summarise_every=1.0,
       output_types=["mean", "max"]
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

   solver.solve(..., duration=100.0)  # Integrate for 100 time units

t0
~~

Starting time of the integration. Affects time-dependent drivers and
the timestamps in output arrays.

- **Default**: ``0.0``
- **Type**: ``float``

.. code-block:: python

   # Start integration at t=5.0
   solver.solve(..., t0=5.0, duration=10.0)
   # Recorded output spans t=5.0 to t=15.0

settling_time
~~~~~~~~~~~~~

Lead-in period before recording begins. The integrator runs for this
duration starting from ``t0``, but no output is saved. This is useful
for allowing transients to decay before collecting data.

- **Default**: ``0.0``
- **Type**: ``float`` (non-negative)

.. code-block:: python

   # Let the system settle for 50 time units, then record 100
   solver.solve(..., settling_time=50.0, duration=100.0)
   # Integration runs from t=0 to t=150
   # Output spans t=50 to t=150

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

Complete Example
----------------

This example shows all timing parameters working together:

.. code-block:: python

   result = solver.solve(
       initial_values={"x": x0_array},
       parameters={"k": k_array},

       # Integration bounds
       t0=0.0,
       settling_time=10.0,    # Discard first 10 time units
       duration=100.0,        # Record 100 time units after settling

       # Step size (adaptive controller)
       dt=0.01,               # Start at 0.01, bounds derived as 1e-4 to 1.0

       # Output timing
       save_every=0.1,        # Save states every 0.1 time units
       summarise_every=10.0,  # Compute summaries every 10 time units
       sample_summaries_every=0.05,  # Sample for summaries every 0.05 units

       # What to save
       output_types=["state", "mean", "max"],
   )

   # Timeline:
   # t=0 to t=10: settling (no output saved)
   # t=10 to t=110: recording period
   #   - 1001 state saves (every 0.1 from t=10 to t=110)
   #   - 10 summary windows (every 10.0 from t=10 to t=110)
   #   - 200 samples per summary window (every 0.05 over 10.0)
