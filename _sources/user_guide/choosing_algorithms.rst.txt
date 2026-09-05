Choosing an Algorithm
=====================

CuBIE ships several integration algorithm families.  This page helps you
pick the right one.  Pass the algorithm name as ``method=`` to
:func:`~cubie.solve_ivp` or ``algorithm=`` to :class:`~cubie.Solver`;
names are case-insensitive.

Decision Guide
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Problem type
     - Recommended family
     - Notes
   * - Non-stiff
     - ERK
     - Fast per step; ``dormand-prince-54`` is a good default.
   * - Mildly stiff
     - DIRK or Rosenbrock-W
     - DIRK is robust; Rosenbrock-W avoids Newton iteration.
   * - Very stiff
     - FIRK
     - ``radau_iia_5`` handles extreme stiffness well.
   * - Fixed step required
     - ``euler`` or any non-adaptive tableau
     - Forward Euler for explicit, ``backwards_euler`` for implicit.

The bare family names ``erk``, ``dirk``, ``firk``, and ``rosenbrock``
are also accepted and select each family's default tableau
(``dormand-prince-54``, ``l_stable_dirk_3``, ``firk_gauss_legendre_2``,
and ``ros3p`` respectively).

Adaptive or fixed?
------------------

"Adaptive: Yes" below means the method produces an embedded error
estimate, so it can drive an adaptive step-size controller.  Methods
without an error estimate always run at a fixed step: if you pair one
with an adaptive controller, CuBIE issues a ``UserWarning`` and
silently swaps in the fixed-step controller, because there is no error
signal to adapt on.

Available Algorithms
--------------------

**Explicit Runge--Kutta (ERK)**

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``heun-21``
     - 2
     - No
     - Heun's method.
   * - ``ralston-33``
     - 3
     - No
     - Ralston's third-order method.
   * - ``bogacki-shampine-32`` / ``rk23`` / ``ode23``
     - 3(2)
     - Yes
     - Low-order, cheap.
   * - ``classical-rk4`` / ``rk4``
     - 4
     - No
     - The classical Runge--Kutta method.
   * - ``fehlberg-45``
     - 5(4)
     - Yes
     - Fehlberg's method.
   * - ``cash-karp-54``
     - 5(4)
     - Yes
     - Cash--Karp coefficients; defaults to Gustafsson control.
   * - ``dormand-prince-54`` / ``dopri54`` / ``rk45`` / ``ode45``
     - 5(4)
     - Yes
     - Industry standard; good default and the ERK family default.
   * - ``tsit5``
     - 5(4)
     - Yes
     - Tsitouras 5(4); often slightly more efficient than
       Dormand--Prince.
   * - ``vern7``
     - 7(6)
     - Yes
     - Verner's high-order method; defaults to Gustafsson control.
   * - ``dormand-prince-853`` / ``dop853``
     - 8(5,3)
     - Yes
     - High order for smooth problems; defaults to Gustafsson
       control.

**Diagonally Implicit Runge--Kutta (DIRK)**

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``implicit_midpoint``
     - 2
     - No
     - Symmetric, energy-preserving.
   * - ``trapezoidal_dirk`` / ``ode23t``
     - 2
     - No
     - Trapezoidal rule.
   * - ``sdirk_2_2``
     - 2
     - No
     - L-stable SDIRK; has no embedded error estimate.
   * - ``l_stable_dirk_3``
     - 3
     - No
     - Default DIRK tableau; L-stable, stiffly accurate, 3 stages.
   * - ``kvaerno3``
     - 3
     - Yes
     - A-L stable ESDIRK, 4 stages.
   * - ``kvaerno5``
     - 5
     - Yes
     - A-L stable ESDIRK, 7 stages; defaults to exact Newton.
   * - ``l_stable_sdirk_4``
     - 4
     - Yes
     - L-stable, 5 stages; defaults to exact Newton.
   * - ``eldirk32_euler``
     - 2
     - Yes
     - ELDIRK: implicit Euler, two explicit stages, third-order estimate.
   * - ``eldirk32_trapezoidal``
     - 2
     - Yes
     - ELDIRK: trapezoidal rule, one explicit stage, third-order estimate.
   * - ``eldirk32_ellsiepen``
     - 2
     - Yes
     - ELDIRK: ``sdirk_2_2``, one explicit stage, third-order estimate.

**Fully Implicit Runge--Kutta (FIRK)**

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``firk_gauss_legendre_2``
     - 4
     - No
     - 2-stage Gauss--Legendre; default FIRK.
   * - ``firk_gauss_legendre_4``
     - 8
     - Yes
     - 4-stage Gauss--Legendre; conservative step control from a
       second-order error estimate.
   * - ``radau_iia_3``
     - 3
     - Yes
     - 2-stage Radau IIA; the cheapest of the family.
   * - ``radau_iia_5`` / ``radau``
     - 5
     - Yes
     - 3-stage Radau IIA; excellent for stiff problems.
   * - ``radau_iia_9``
     - 9
     - Yes
     - 5-stage Radau IIA; for tight tolerances, five coupled stages
       per step. Defaults to BiCGSTAB with exact Newton.

**Rosenbrock-W**

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``ros3p``
     - 3
     - Yes
     - Default Rosenbrock; linearly implicit, no Newton iteration.
   * - ``rodas3p``
     - 3
     - Yes
     - Stiffly accurate variant.
   * - ``rosenbrock23_sciml`` / ``rosenbrock23`` / ``ode23s``
     - 2(3)
     - Yes
     - MATLAB ``ode23s``-equivalent / SciML-compatible tableau.

**Simple methods**

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Name
     - Order
     - Adaptive
     - Notes
   * - ``euler``
     - 1
     - No
     - Forward Euler; explicit.
   * - ``backwards_euler``
     - 1
     - No
     - Backward Euler; implicit, L-stable.
   * - ``backwards_euler_pc``
     - 1
     - No
     - Predictor-corrector backward Euler.
   * - ``crank_nicolson``
     - 2
     - Yes
     - Implicit trapezoidal rule; a second backward-Euler solve
       provides the embedded error estimate.

.. _choosing-a-controller:

Choosing a Controller
---------------------

Adaptive algorithms use an error controller to adjust the step size.
Select one by name with ``step_controller`` inside
``step_control_settings``; the registered names are ``"fixed"``,
``"i"``, ``"pi"``, ``"pid"``, and ``"gustafsson"``.

**fixed**
   Constant step size; no error control.  Used automatically for
   methods without an error estimate.

**i, pi, pid**
   The ``i``, ``pi``, and ``pid`` controllers are all the same PID
   controller, but the ``i`` and ``pi`` device functions compile to
   more efficient code when you don't need the additional gain terms.
   After every step they compare the estimated error with the
   tolerance (``atol + rtol * max(|this_state|, |last_state|)``) and
   propose a next-step size to bring the ratio of error/tolerance to
   1.  Each gain term acts on the current error and a subset of
   previous errors:

   - ``integral_gain`` multiplies by the error/tolerance ratio of the
     current step.  It sets how much this error adds to the running
     correction.
   - ``proportional_gain`` multiplies by how much the ratio changed
     since the previous step.  It sets how much the running correction
     is adjusted by the change in error.
   - ``derivative_gain`` multiplies by how much the change in the
     ratio changed over two steps.  It acts against large changes in
     the rate of change of error, damping the controller.

   Passing gains without any ``step_controller`` selected will select
   the fastest of the ``i``, ``pi``, ``pid`` controllers that can
   provide your gains, or you can provide a string to select a common
   literature controller using the ``filter_coefficients`` argument.

**gustafsson**
   Predictive controller that accounts for the previous step's error
   ratio.  Widely used with implicit methods; useful when step
   rejections are frequent.

Each algorithm family picks a default controller (``i`` for ERK,
``pi`` for DIRK, ``gustafsson`` for FIRK and Rosenbrock, ``fixed``
without an error estimate); some tableaus override it.  To override:

.. code-block:: python

   solver = qb.Solver(
       LV,
       algorithm="dormand-prince-54",
       step_control_settings={"step_controller": "gustafsson"},
   )

Automatic calibration
---------------------

The fastest algorithm order, linear solver, preconditioner, and
Newton variant depend on the system.  Given a representative batch,
the solver can measure them directly:

.. code-block:: python

   solver = qb.Solver(LV, algorithm="ros3p", atol=1e-6, rtol=1e-6)
   report = solver.calibrate(
       {"x": x0_values},
       {"alpha": alpha_values},
       duration=10.0,
   )
   print(report.summary())

:meth:`Solver.calibrate <cubie.batchsolving.solver.Solver.calibrate>`
races different solver configurations on your problem: it compares a
few orders of each algorithm family and, for the implicit families,
the preconditioner, linear-solver, Newton-variant, smoothed-error,
and dense-predictor settings.  Candidates that fail to integrate the
grid are dropped before timing; survivors are ranked on a few
full-length solves, and the leaders are returned with solve times
and failure counts.  By default the winning configuration is applied
to the solver in place; pass ``apply=False`` to get the race results
without modifying the solver.

For the mathematical background behind these algorithms, see
:doc:`/theory/numerical_integration`.
