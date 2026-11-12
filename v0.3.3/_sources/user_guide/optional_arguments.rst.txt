Solver Options Reference
========================

Every option on this page is a keyword argument: pass it to
:func:`~cubie.solve_ivp`, to the :class:`~cubie.Solver` constructor, or
to :meth:`~cubie.batchsolving.solver.Solver.solve`, and CuBIE routes it
to the right internal component for you.  You never need to know which
component that is — the options below are grouped by what you are
trying to achieve.

Two conventions apply throughout:

- Leaving an option out (or passing ``None``) means "use the default".
- Misspelt or unknown option names raise ``KeyError`` rather than
  being silently ignored.

Controlling accuracy
--------------------

These are the two knobs most users touch.  Adaptive methods choose
their own step size to keep the estimated local error below
``atol + rtol * |value|`` for every state variable.

**atol** — absolute error tolerance.
    Dominates when a state is near zero.  A scalar applies to every
    state; an array gives one tolerance per state variable.  Must be
    non-negative.

    - Default: ``1e-6``
    - Type: ``float`` or array of ``float``

**rtol** — relative error tolerance.
    Scales with the magnitude of the state: ``rtol=1e-6`` asks for
    roughly six correct digits.  Scalar or per-state array; must be
    non-negative.

    - Default: ``1e-6``
    - Type: ``float`` or array of ``float``

Controlling the step size
-------------------------

**dt** — the step size (fixed methods) or initial step size (adaptive).
    For fixed-step methods this is *the* step size.  For adaptive
    methods it seeds the first step, and if you give ``dt`` alone the
    bounds are derived from it as ``dt_min = dt/100`` and
    ``dt_max = dt*100``.

    - Default: ``1e-3`` (fixed); ``sqrt(dt_min * dt_max)`` (adaptive)
    - Type: ``float``, positive

**dt_min** / **dt_max** — hard bounds on the adaptive step size.
    The controller never steps outside these.  Use ``dt_max`` to stop
    the solver skating over short-lived dynamics; if the error is
    still too large at ``dt_min``, the run is flagged in
    ``result.status_codes`` rather than silently continuing.
    Supplying ``dt_max < dt_min`` raises ``ValueError``.

    - Defaults: ``dt_min=1e-6``, ``dt_max=1.0``
    - Type: ``float``, positive

See :doc:`timing` for the full derivation rules, and for the output
timing options (``save_every``, ``summarise_every``,
``sample_summaries_every``, ``duration``, ``t0``, ``settling_time``).

Tuning the step controller
--------------------------

Adaptive step-size control is handled by a controller you select with
``step_controller``: one of ``"fixed"``, ``"i"``, ``"pi"``, ``"pid"``,
or ``"gustafsson"`` (see :doc:`choosing_algorithms`).  Each algorithm
picks a sensible default, so treat everything in this section as
expert tuning.

Options for all adaptive controllers:

**min_gain** / **max_gain** — limits on how fast the step changes.
    Per adjustment, the step size can shrink to no less than
    ``min_gain`` times and grow to no more than ``max_gain`` times its
    previous value.

    - Defaults: ``min_gain=0.3``, ``max_gain=2.0``

**safety** — conservatism factor.
    Step-size predictions are multiplied by this, so values below 1.0
    aim slightly smaller than the error estimate suggests, trading a
    little speed for fewer rejected steps.

    - Default: ``0.9``

**deadband_min** / **deadband_max** — "leave it alone" band.
    Proposed gains inside this band are snapped to 1.0, preventing
    twitchy step-size changes when the error hovers near tolerance.
    Set both to 1.0 to disable.

    - Defaults: ``deadband_min=1.0``, ``deadband_max=1.2``

Controller-specific gains:

**kp** / **ki** — proportional and integral gains (``pi`` and ``pid``).
    How strongly the controller reacts to the current error (``kp``)
    and to the error history (``ki``).

    - Defaults: ``kp=0.7``, ``ki=-0.4``

**kd** — derivative gain (``pid`` only).
    Reacts to the rate of change of the error.  Disabled at 0.

    - Default: ``0.0``

**newton_target_iters** — Newton-work reference (``gustafsson`` only).
    The Gustafsson controller scales its prediction by how hard the
    implicit solver is working; this reference count sets the strength
    of that damping.  It does not limit Newton solver execution.

    - Default: ``20``

Tuning the implicit (stiff) solvers
-----------------------------------

Implicit algorithms (Backward Euler, Crank--Nicolson, DIRK, FIRK)
solve a nonlinear system at every step with a Newton--Krylov method:
an outer Newton loop handles the nonlinearity, and an inner Krylov
loop solves a linear system for each Newton update.  Rosenbrock-W
methods are linearly implicit: they skip the Newton loop entirely, so
only the linear-solver and preconditioner options below apply to them.

Newton (outer loop) options:

**newton_atol** / **newton_rtol** — Newton convergence tolerances.
    Scale the Newton update in the convergence test: the solve
    accepts when the estimated update error drops below one percent
    of the scaled tolerance, matching OrdinaryDiffEq's Newton
    criterion.

    - Default: the step controller's ``atol``/``rtol`` divided by 10
      (so stage solves always converge tighter than the error
      estimate they feed).
    - Type: ``float`` or array, non-negative

**newton_max_iters** — Newton iteration limit.
    Steps that fail to converge within this many iterations are marked
    failed (and retried at a smaller step size under adaptive
    control).

    - Default: ``100``

Krylov (inner loop) options:

**krylov_atol** / **krylov_rtol** — linear-solve tolerances.
    These weight the linear solver's stopping norm: a weighted
    residual of one sits at this envelope.

    - Default: the step controller's ``atol``/``rtol``.

**krylov_max_iters** — linear iteration limit per Newton step.

    - Default: ``100``

**krylov_residual_reduction** — relative linear stopping term.
    Each linear solve stops once its weighted residual falls below
    ``floor + reduction * ||b||`` with ``||b||`` the weighted
    right-hand side at entry, so early Newton iterations avoid
    over-solving.

    - Default: the adaptive step controller's smallest ``rtol``
      entry; machine epsilon for non-adaptive runs.
    - Type: ``float`` in ``[0, 1]``

**krylov_residual_floor** — absolute linear stopping term.
    The absolute part of the stopping rule, in weighted-norm units
    (one sits at the ``krylov_atol``/``krylov_rtol`` envelope).

    - Default: ``sqrt(eps)`` of the precision.
    - Type: ``float``, non-negative

**linear_correction_type** — linear solver strategy.
    ``"minimal_residual"`` (default) minimises the residual along the
    search direction; ``"steepest_descent"`` is more robust but often
    slower; ``"bicgstab"`` switches to a BiCGSTAB solver, which can
    help on difficult non-symmetric systems.

Preconditioner options:

**preconditioner_order** — truncated Neumann-series order.
    Higher orders speed up Krylov convergence at more cost per
    iteration; 1--3 suits most problems.

    - Default: ``2``

**preconditioner_type** — preconditioner family.
    ``"neumann"`` (default) or ``"jacobi"``; pass a two-element list
    (e.g. ``["jacobi", "neumann"]``) to chain both.

Advanced implicit options: **beta** and **gamma** (implicit-integration
coefficients, default 1.0 each).  These change the equations being
solved — leave them alone unless you know you need them.  The mass
matrix is not a solver option: it is part of the system definition
(pass ``mass=`` to ``create_ODE_system`` for a hand-formulated
semi-explicit DAE, or let structural simplification derive it), and
systems carrying one require an implicit algorithm.

Choosing the method's coefficients
----------------------------------

**tableau** — the Butcher tableau for multi-stage methods.
    Selecting an algorithm by name (see :doc:`choosing_algorithms`)
    already picks a tableau.  The bare family names use these
    defaults:

    - ERK: ``dormand-prince-54`` (adaptive)
    - DIRK: ``l_stable_dirk_3`` (fixed-step)
    - FIRK: ``firk_gauss_legendre_2`` (fixed-step)
    - Rosenbrock-W: ``ros3p`` (adaptive)

    You can also pass a custom ``ButcherTableau`` instance as the
    algorithm itself.

Choosing what gets saved
------------------------

**output_types** — what to record.
    Any combination of the time-domain outputs ``"state"``,
    ``"observables"``, ``"time"``, and ``"iteration_counters"``, plus
    any of the 18 summary metric names (``"mean"``, ``"max"``,
    ``"peaks"``, ...) — the full metric table is in :doc:`results`.

    - Default: ``["state"]``

**save_variables** / **summarise_variables** — which variables, by name.
    Restrict time-domain saving and summary metrics to the listed
    state/observable names.  Saving less is the single easiest memory
    and speed win.  (The index-based equivalents
    ``saved_state_indices``, ``saved_observable_indices``,
    ``summarised_state_indices``, and ``summarised_observable_indices``
    are also accepted.)

    - Default: all states and observables; summaries follow the saved
      selection unless you specify them separately.

GPU tuning (advanced)
---------------------

**Buffer location options** (``state_location``,
``timestep_memory_location``, and about forty more ``*_location``
options across the loop, algorithm, and solver configs)
    Each working buffer can live in ``"local"`` (per-thread) or
    ``"shared"`` (per-block) GPU memory.  Everything defaults to
    ``"local"``, which profiling shows is the right choice on typical
    hardware — these exist for performance experiments on specific
    GPU architectures and never affect results.

Memory-manager settings (VRAM proportion, stream groups) are covered in
:doc:`memory`, and compilation caching in :doc:`caching`.
