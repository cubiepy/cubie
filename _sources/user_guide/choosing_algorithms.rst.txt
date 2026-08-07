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
     - Cash--Karp coefficients.
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
     - Verner's high-order method.
   * - ``dormand-prince-853`` / ``dop853``
     - 8(5,3)
     - Yes
     - High order; useful for smooth, high-accuracy problems.

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
   * - ``l_stable_sdirk_4``
     - 4
     - Yes
     - L-stable, 5 stages; the only adaptive DIRK tableau.

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
   * - ``radau_iia_5`` / ``radau``
     - 5
     - Yes
     - 3-stage Radau IIA; excellent for stiff problems.

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

Choosing a Controller
---------------------

Adaptive algorithms use an error controller to adjust the step size.
Select one by name with ``step_controller`` inside
``step_control_settings``; the registered names are ``"fixed"``,
``"i"``, ``"pi"``, ``"pid"``, and ``"gustafsson"``.

**fixed**
   Constant step size; no error control.  Used automatically for
   methods without an error estimate.

**i (integral)**
   The simplest adaptive controller: reacts to the current error only.

**pi (proportional-integral)**
   The most common choice.  Balances responsiveness with stability.

**pid (proportional-integral-derivative)**
   Adds a derivative term for smoother step-size histories; can reduce
   oscillations in the step size on problems with sharp transients.

**gustafsson**
   Predictive controller that accounts for the previous step's error
   ratio.  Widely used with implicit methods; useful when step
   rejections are frequent.

Each algorithm picks a sensible default controller (``pi`` for
explicit Runge--Kutta, ``gustafsson`` for the implicit families,
``fixed`` when there is no error estimate), so for most problems you
don't need to choose at all.  To override:

.. code-block:: python

   solver = qb.Solver(
       LV,
       algorithm="dormand-prince-54",
       step_control_settings={"step_controller": "gustafsson"},
   )

For the mathematical background behind these algorithms, see
:doc:`/theory/numerical_integration`.
