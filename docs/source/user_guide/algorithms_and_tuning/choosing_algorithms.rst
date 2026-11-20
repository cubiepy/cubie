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
     - Fast per step.
   * - Mildly stiff
     - DIRK or Rosenbrock-W
     - DIRK is robust but slower to compile and run; Rosenbrock-W
       methods are fast.
   * - Very stiff
     - FIRK
     - Use FIRK algorithms like ``radau_iia_5`` for extreme stiffness.

The bare family names ``erk``, ``dirk``, ``firk``, and ``rosenbrock``
are also accepted and select each family's default tableau
(``tsit5``, ``l_stable_dirk_3``, ``firk_gauss_legendre_2``,
and ``ros3p`` respectively).

Adaptive or fixed?
------------------

"Adaptive: Yes" in the tables below means that cubie can estimate the
error of each step of the method, so you can use it with an adaptive
step-size controller.  An order of ``p(q)`` means the method is of
order ``p`` and cubie estimates its error with an order ``q`` method.
Cubie always runs a method without an error estimate at a fixed step:
if you pair one with an adaptive controller, cubie warns with a
``UserWarning`` and uses the fixed-step controller instead.

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
     - Industry standard.
   * - ``tsit5``
     - 5(4)
     - Yes
     - Tsitouras 5(4); often slightly more efficient than
       Dormand--Prince. The default algorithm and the ERK family
       default.
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
     - 3(2)
     - Yes
     - A-L stable ESDIRK, 4 stages.
   * - ``kvaerno5``
     - 5(4)
     - Yes
     - A-L stable ESDIRK, 7 stages; defaults to exact Newton.
   * - ``l_stable_sdirk_4``
     - 4(3)
     - Yes
     - L-stable, 5 stages; defaults to exact Newton.
   * - ``eldirk32_euler``
     - 2(3)
     - Yes
     - ELDIRK: implicit Euler and two explicit stages.
   * - ``eldirk32_trapezoidal``
     - 2(3)
     - Yes
     - ELDIRK: trapezoidal rule and one explicit stage.
   * - ``eldirk32_ellsiepen``
     - 2(3)
     - Yes
     - ELDIRK: ``sdirk_2_2`` and one explicit stage.

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
     - 8(2)
     - Yes
     - 4-stage Gauss--Legendre. Cubie controls the step conservatively,
       because it estimates the error with a second-order method.
   * - ``radau_iia_3``
     - 3(1)
     - Yes
     - 2-stage Radau IIA; the cheapest of the family.
   * - ``radau_iia_5`` / ``radau``
     - 5(2)
     - Yes
     - 3-stage Radau IIA; excellent for stiff problems.
   * - ``radau_iia_9``
     - 9(4)
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
     - 3(2)
     - Yes
     - Lang and Verwer (2001), 3 stages. The default Rosenbrock-W tableau.
   * - ``rodas3p``
     - 3(2)
     - Yes
     - Steinebach (2024), 5 stages.
   * - ``rosenbrock23_sciml`` / ``rosenbrock23`` / ``ode23s``
     - 2(3)
     - Yes
     - Shampine and Reichelt (1997), 3 stages; the method of MATLAB's ``ode23s``.
   * - ``rosenbrock32``
     - 3(2)
     - Yes
     - Shampine and Reichelt (1997), 3 stages.
   * - ``ros2``
     - 2(1)
     - Yes
     - Verwer et al. (1999), 2 stages.
   * - ``ros2pr``
     - 2(1)
     - Yes
     - Rang (2014), 3 stages.
   * - ``ros2s``
     - 2(1)
     - Yes
     - Rang (2014), 3 stages.
   * - ``ros3``
     - 3(2)
     - Yes
     - Hairer and Wanner (1996), 3 stages.
   * - ``ros3pr``
     - 3(2)
     - Yes
     - Rang (2014), 3 stages.
   * - ``scholz4_7``
     - 3(2)
     - Yes
     - Rang (2014), 3 stages.
   * - ``ros34pw1a``
     - 3(2)
     - Yes
     - Rang and Angermann (2005), 4 stages.
   * - ``ros34pw1b``
     - 3(2)
     - Yes
     - Rang and Angermann (2005), 4 stages.
   * - ``ros34pw2``
     - 3(2)
     - Yes
     - Rang and Angermann (2005), 4 stages.
   * - ``ros34pw3``
     - 4(3)
     - Yes
     - Rang and Angermann (2005), 4 stages.
   * - ``ros34prw``
     - 3(2)
     - Yes
     - Rang (2015), 4 stages.
   * - ``ros3prl``
     - 3(2)
     - Yes
     - Rang (2014), 4 stages.
   * - ``ros3prl2``
     - 3(2)
     - Yes
     - Rang (2014), 4 stages.
   * - ``rok4a``
     - 4(3)
     - Yes
     - Tranquilli and Sandu (2014), 4 stages.
   * - ``rosshamp4``
     - 4(3)
     - Yes
     - Shampine (1982), 4 stages.
   * - ``veldd4``
     - 4(3)
     - Yes
     - van Veldhuizen (1984), 4 stages.
   * - ``velds4``
     - 4(3)
     - Yes
     - van Veldhuizen (1984), 4 stages.
   * - ``grk4t``
     - 4(3)
     - Yes
     - Kaps and Rentrop (1979), 4 stages.
   * - ``grk4a``
     - 4(3)
     - Yes
     - Kaps and Rentrop (1979), 4 stages.
   * - ``ros4lstab``
     - 4(3)
     - Yes
     - Hairer and Wanner (1996), 4 stages.
   * - ``rosenbrockw6s4os``
     - 4
     - No
     - 6 stages.
   * - ``rodas3``
     - 3(2)
     - Yes
     - Sandu et al. (1997), 4 stages.
   * - ``rodas3d``
     - 3(2)
     - Yes
     - Yu et al. (2024), 4 stages.
   * - ``rodas23w``
     - 2(3)
     - Yes
     - Steinebach (2024), 5 stages.
   * - ``rodas4``
     - 4(3)
     - Yes
     - Hairer and Wanner (1996), 6 stages.
   * - ``rodas42``
     - 4(3)
     - Yes
     - Hairer and Wanner (1996), 6 stages.
   * - ``rodas4p``
     - 4(3)
     - Yes
     - Steinebach (1995), 6 stages.
   * - ``rodas4p2``
     - 4(3)
     - Yes
     - Steinebach (2020), 6 stages.
   * - ``rodas4pw``
     - 4(3)
     - Yes
     - Steinebach, 9 stages.
   * - ``rodas5``
     - 5(4)
     - Yes
     - Di Marzo (1993), 8 stages.
   * - ``rodas5p``
     - 5(4)
     - Yes
     - Steinebach (2023), 8 stages.
   * - ``rodas5pe``
     - 5(4)
     - Yes
     - Steinebach (2024), 8 stages.
   * - ``rodas6p``
     - 6(5)
     - Yes
     - Steinebach (2025), 16 stages.

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
Select one by name with the ``step_controller`` keyword argument:
``"fixed"``, ``"i"``, ``"pi"``, ``"pid"``, or ``"gustafsson"``.

**fixed**
   Constant step size; no error control.  Used automatically for
   methods without an error estimate.

**i, pi, pid**
   The ``i``, ``pi``, and ``pid`` controllers are all the same PID
   controller, but the ``i`` and ``pi`` device functions compile to
   more efficient code when you don't need the additional gain terms.
   After every step they compare the estimated error with the
   tolerance (``atol + rtol * max(|this_state|, |last_state|)``) and
   propose a next-step size to bring the ratio of error/tolerance
   to 1.  Each gain term acts on the current error and a subset of
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
without an error estimate); some tableaus override it.  To choose a
different controller, pass its name:

.. code-block:: python

   solver = qb.Solver(
       LV,
       algorithm="dormand-prince-54",
       step_controller="gustafsson",
   )

Automatic calibration
---------------------

The fastest algorithm order, linear solver type, preconditioner, and
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
and dense-predictor settings.  Every candidate is timed on a few
full-length solves and reported with its solve times and failed-run
count.  Cubie resizes the batch it times so that a solve takes about
20 ms where it can; pass ``auto_size=False`` to time your exact batch
instead.  Cubie ranks candidates whose trajectories succeed (runs that
don't end in a step-size or convergence error) at a rate at least 95%
of the best candidate's rate by time.  Candidates below that threshold
are ranked afterward, also by time, with their failure rates shown.
By default cubie applies the winning configuration to the solver in
place; pass ``apply=False`` to get the results of the race without
modifying the solver.

``calibrate`` compiles a large number of kernels, so allow half an
hour for a bigger system.

With the algorithm settled, :meth:`Solver.optimize
<cubie.batchsolving.solver.Solver.optimize>` tries a few different
configurations of where buffers sit in memory, which loops get
unrolled, and how many threads run at once on the GPU to find the
fastest configuration for your system:

.. code-block:: python

   report = solver.optimize(
       {"x": x0_values},
       {"alpha": alpha_values},
       duration=10.0,
   )
   print(report.summary())

``optimize`` leaves the settings you gave explicitly, and the ones an
earlier ``optimize`` set, unchanged unless you pass ``force=True``.
Like ``calibrate``, it applies the fastest configuration in place
unless you pass ``apply=False``, and it resizes the batch to take
about 20 ms unless you pass ``auto_size=False``.

