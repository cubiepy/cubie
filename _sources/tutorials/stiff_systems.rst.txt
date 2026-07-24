Tutorial 3: Stiff Systems, Implicit Methods, and Drivers
========================================================

A *stiff* system mixes fast and slow dynamics.  The trouble
this causes is about stability rather than accuracy: an explicit
method's step size is capped by the fastest timescale in the system
even after that fast component has decayed away, because larger steps
make the method unstable.  An implicit method stays stable at large
steps, so its step size is set by the accuracy you asked for instead.
This tutorial solves a stiff oscillator with an implicit method,
shows how to recognise and fix a failed solve, and adds a forcing
signal.

Step 1: A stiff test problem
----------------------------

The Van der Pol oscillator with a large damping parameter ``mu`` is a
classic stiff benchmark.  It creeps along a slow branch, then relaxes
almost instantaneously:

.. code-block:: python

   import numpy as np
   import cubie as qb

   vdp = qb.create_ODE_system(
       """
       dx = v
       dv = mu * (1 - x*x) * v - x
       """,
       parameters={"mu": 50.0},
       states={"x": 2.0, "v": 0.0},
       name="VanDerPol",
   )

Step 2: Solve with an implicit method
-------------------------------------

A stiff solver is selected through the same ``method`` keyword as
any other.  ``method="radau"`` selects Radau IIA of order 5, a fully
implicit method that copes with the stiffest problems you are likely
to meet:

.. code-block:: python

   result = qb.solve_ivp(
       vdp,
       y0={"x": 2.0, "v": 0.0},
       parameters={"mu": np.linspace(20.0, 80.0, 16)},
       method="radau",
       duration=20.0,
       save_every=0.05,
       rtol=1e-5,
       atol=1e-8,
   )
   assert np.all(result.status_codes == 0), result.status_messages

CuBIE derives the Jacobian your implicit solver needs symbolically
from the equations, so there is nothing extra to supply.

This solve succeeds, but it also prints a warning that the default
Neumann preconditioner may diverge for this system.  The solve above
finished cleanly regardless (the status check confirms it), and
passing ``preconditioner_type="jacobi"`` selects a preconditioner
suited to systems like this one and silences the warning.

Radau is the big gun of the family, and its robustness costs
iterations.  For mildly stiff problems, cheaper options include the
linearly implicit Rosenbrock-W methods (``method="rosenbrock"``,
which skip Newton iteration entirely), ``"crank_nicolson"``, and the
DIRK family (:doc:`/user_guide/choosing_algorithms`).

Step 3: When the solver gives up
--------------------------------

An adaptive controller shrinks the step until the local error
estimate meets your tolerances.  When the tolerances cannot be met at
any allowed step size, the step collapses to the ``dt_min`` floor and
the run exits early with the ``STEP_TOO_SMALL`` status.  You can
provoke this by asking a single-precision solve for tighter accuracy
than single-precision arithmetic can deliver:

.. code-block:: python

   result = qb.solve_ivp(
       vdp,
       y0={"x": 2.0, "v": 0.0},
       parameters={"mu": np.linspace(20.0, 80.0, 16)},
       method="radau",
       duration=20.0,
       save_every=0.05,
       rtol=1e-10,
       atol=1e-12,
   )
   print(result.status_codes.max())  # nonzero: some runs bailed out
   print(result.status_messages[0])  # decoded flags for run 0

Status codes are bit flags, so a failing run can report several
conditions at once (a Newton failure and the step-size collapse it
caused, for example).  The flag to look for here is
``STEP_TOO_SMALL``.

Two levers fix a ``STEP_TOO_SMALL`` exit:

1. Loosen ``atol`` and ``rtol`` to values the arithmetic and the
   problem can deliver.  If your states span different magnitude
   scales, pass a vector-valued ``atol`` with one entry per state
   (``atol=np.array([1e-6, 1e-2])``) rather than tightening every
   state to suit the smallest one.
2. Lower ``dt_min`` (default ``1e-6``) when the dynamics genuinely
   need steps smaller than the floor, for example when resolving a
   relaxation spike at high ``mu``.

Failed runs are reported per run: ``result.status_codes`` holds one
code per run and ``result.status_messages`` decodes them, so a batch
where only the stiffest runs fail tells you which parameter values
need attention.

Step 4: Add a forcing signal (driver)
-------------------------------------

Real experiments force their systems.  A *driver* is a time-dependent
input; here we drive the oscillator with a sampled signal, as if
replaying a measurement.  In the ``drivers`` dict, ``"time"`` is a
reserved key holding the timestamps the signals are sampled at; every
other entry names a driver whose array must match ``"time"`` in
length:

.. code-block:: python

   forced = qb.create_ODE_system(
       """
       dx = v
       dv = mu * (1 - x*x) * v - x + forcing
       """,
       parameters={"mu": 50.0},
       states={"x": 2.0, "v": 0.0},
       drivers=["forcing"],
       name="ForcedVanDerPol",
   )

   t_samples = np.linspace(0.0, 20.0, 400)
   signal = 5.0 * np.sin(2.0 * np.pi * 0.25 * t_samples)

   result = qb.solve_ivp(
       forced,
       y0={"x": 2.0, "v": 0.0},
       parameters={"mu": np.linspace(20.0, 80.0, 16)},
       drivers={"forcing": signal, "time": t_samples},
       method="rosenbrock",
       duration=20.0,
       save_every=0.05,
       rtol=1e-5,
       atol=1e-8,
   )
   assert np.all(result.status_codes == 0), result.status_messages

This example uses the linearly implicit Rosenbrock-W method from the
quicker end of the stiff family, together with explicit tolerances.
Stating your tolerances is a good habit in general: the defaults are
not tuned to your problem, and the failure modes of Step 3 are
easier to reason about when you know what accuracy you asked for.

CuBIE fits a cubic spline through your samples so adaptive steppers
can evaluate the forcing at any time point, not just your sample
times.  Interpolation options (polynomial order, periodic wrapping,
boundary conditions) are covered in :doc:`/user_guide/drivers`.

A sine wave is not really a sampled signal, though.  When the forcing
has a closed form, you can write it directly into the equations using
the time symbol ``t``, and CuBIE substitutes the current simulation
time as it integrates.  This form needs no driver arrays and
introduces no interpolation error:

.. code-block:: python

   driven = qb.create_ODE_system(
       """
       dx = v
       dv = mu * (1 - x*x) * v - x + amp * sin(omega * t)
       """,
       constants={"amp": 5.0, "omega": 2.0 * np.pi * 0.25},
       parameters={"mu": 50.0},
       states={"x": 2.0, "v": 0.0},
       name="SineDrivenVanDerPol",
   )

Reserve sampled drivers for signals that only exist as data, such as
recorded measurements or stochastic inputs.

Recap
-----

- A stiff system is solved by passing ``method="radau"``; step down
  to ``"rosenbrock"`` or a DIRK method if radau proves more power
  than the problem needs.
- Runs that fail with ``STEP_TOO_SMALL`` are fixed by loosening
  ``atol``/``rtol`` (or supplying a per-state ``atol`` vector), and
  by lowering ``dt_min`` if the dynamics truly need smaller steps.
- Sampled forcing data enters as a driver, declared on the system
  and passed as ``{"name": values, "time": times}``; forcing with a
  closed form is written into the equations using ``t``.
