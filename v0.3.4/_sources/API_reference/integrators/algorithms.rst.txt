Algorithms
==========

``cubie.integrators.algorithms``
--------------------------------

.. currentmodule:: cubie.integrators.algorithms

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   algorithms/base_step_config
   algorithms/base_algorithm_step
   algorithms/step_cache
   algorithms/explicit_step_config
   algorithms/explicit_euler_step
   algorithms/implicit_step_config
   algorithms/backwards_euler_step
   algorithms/backwards_euler_pc_step
   algorithms/crank_nicolson_step
   algorithms/generic_erk_step
   algorithms/generic_erk_tableaus
   algorithms/generic_dirk_step
   algorithms/generic_dirk_tableaus
   algorithms/generic_firk_step
   algorithms/generic_firk_tableaus
   algorithms/generic_rosenbrock_step
   algorithms/generic_rosenbrock_tableaus
   algorithms/get_algorithm_step

Factories in :mod:`cubie.integrators.algorithms` assemble explicit and implicit
step implementations that plug into the CUDA-based integrator loop. Explicit
steps wrap direct right-hand-side evaluations, while implicit steps couple
user-supplied device callbacks with matrix-free Newton–Krylov helpers to
satisfy nonlinear solves. Precision handling is central: every factory
propagates the configured precision through compiled device helpers and the
shared linear and nonlinear solver stack.

.. _algorithm-defaults:

Default settings
----------------

Each algorithm ships with a default scheme and a default step
controller. Requesting an algorithm by family name
(``algorithm="erk"``) uses the defaults below; requesting a specific
tableau alias (``algorithm="radau"``) selects that scheme within the
same family.

.. list-table::
   :header-rows: 1
   :widths: 20 36 10 34

   * - Name
     - Default scheme
     - Order
     - Default step control
   * - ``euler``
     - Forward Euler (explicit, single stage)
     - 1
     - Fixed step
   * - ``backwards_euler``
     - Backward Euler (implicit, single stage)
     - 1
     - Fixed step
   * - ``backwards_euler_pc``
     - Backward Euler with a forward-Euler predictor
     - 1
     - Fixed step
   * - ``crank_nicolson``
     - Crank–Nicolson with an embedded backward-Euler error
       estimate
     - 2
     - Gustafsson predictive: gain clamp 0.2–8.0
   * - ``erk``
     - ``dormand-prince-54`` (explicit, seven stages)
     - 5(4)
     - PI: ``kp=0.7``, ``ki=-0.4``, gain clamp 0.2–10.0
   * - ``dirk``
     - ``l_stable_dirk_3`` (diagonally implicit, three stages)
     - 3
     - Fixed step — the default tableau has no error estimate
   * - ``firk``
     - ``firk_gauss_legendre_2`` (fully implicit, two coupled
       stages)
     - 4
     - Fixed step — the default tableau has no error estimate
   * - ``rosenbrock``
     - ``ros3p`` (linearly implicit, three stages)
     - 3
     - Gustafsson predictive: gain clamp 0.2–8.0

How the step controller is chosen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptive step control needs an embedded error estimate. When an
algorithm or tableau is named without setting ``step_controller``,
CuBIE selects the family's tuned controller if the scheme provides
an estimate — a PI controller for explicit Runge–Kutta, the
Gustafsson predictive controller for the implicit families — and a
fixed-step controller if it does not.
This is why ``dirk`` and ``firk`` run fixed-step out of the box:
their default tableaus carry no embedded estimate. Aliases that do
(``radau``, for example) enable the family's adaptive defaults
automatically. Explicitly pairing an adaptive controller with an
estimate-free scheme falls back to fixed stepping with a
``UserWarning``.

The implicit family defaults use a 1.0–1.2 deadband (step-size
increases smaller than 20% are skipped; decreases always apply),
matching RADAU5's step-freeze band; the explicit ``erk`` defaults
apply no deadband. Every value can be overridden per solve — see
:doc:`/user_guide/configuration` for how kwargs reach the controller
and :doc:`/user_guide/optional_arguments` for the full parameter
list.

Implicit solver defaults
~~~~~~~~~~~~~~~~~~~~~~~~

Implicit algorithms (``backwards_euler``, ``backwards_euler_pc``,
``crank_nicolson``, ``dirk``, ``firk``) solve their stage equations
with a matrix-free Newton–Krylov iteration. ``rosenbrock`` is
linearly implicit: it performs one linear solve per stage, so only
the Krylov and preconditioner settings below apply to it and the
``newton_*`` parameters do not.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Parameter
     - Default
   * - ``newton_atol`` / ``newton_rtol``
     - ``1e-6``
   * - ``newton_max_iters``
     - ``100``
   * - ``krylov_atol`` / ``krylov_rtol``
     - ``1e-6``
   * - ``krylov_max_iters``
     - ``100``
   * - ``linear_correction_type``
     - ``"minimal_residual"``
   * - ``preconditioner_type``
     - ``"neumann"``
   * - ``preconditioner_order``
     - ``2``

Base infrastructure
-------------------

* :doc:`BaseStepConfig <algorithms/base_step_config>` – attrs configuration shared by explicit and implicit steps.
* :doc:`BaseAlgorithmStep <algorithms/base_algorithm_step>` – base class that wires precision and CUDA compilation helpers.
* :doc:`StepCache <algorithms/step_cache>` – container storing compiled kernels and loop scratch buffers.

Explicit steps
--------------

* :doc:`ExplicitStepConfig <algorithms/explicit_step_config>` – configuration for explicit step factories.
* :doc:`ExplicitEulerStep <algorithms/explicit_euler_step>` – Euler step that invokes the RHS device function.
* :doc:`ERKStep <algorithms/generic_erk_step>` – multistage explicit Runge–Kutta step with adaptive control defaults.
* :doc:`ERK tableau registry <algorithms/generic_erk_tableaus>` – named explicit Runge–Kutta tableaus and references.

Implicit steps
--------------

* :doc:`ImplicitStepConfig <algorithms/implicit_step_config>` – configuration for implicit step factories.
* :doc:`BackwardsEulerStep <algorithms/backwards_euler_step>` – backward Euler implicit algorithm.
* :doc:`BackwardsEulerPCStep <algorithms/backwards_euler_pc_step>` – predictor-corrector backward Euler variant.
* :doc:`CrankNicolsonStep <algorithms/crank_nicolson_step>` – Crank–Nicolson implicit algorithm.
* :doc:`DIRKStep <algorithms/generic_dirk_step>` – diagonally implicit Runge–Kutta family with embedded error estimates.
* :doc:`DIRK tableau registry <algorithms/generic_dirk_tableaus>` – named diagonally implicit Runge–Kutta tableaus and references.
* :doc:`FIRKStep <algorithms/generic_firk_step>` – fully implicit Runge–Kutta family (Gauss–Legendre, Radau IIA).
* :doc:`FIRK tableau registry <algorithms/generic_firk_tableaus>` – named fully implicit Runge–Kutta tableaus and references.
* :doc:`GenericRosenbrockWStep <algorithms/generic_rosenbrock_step>` – linearly implicit Rosenbrock-W methods.
* :doc:`Rosenbrock tableau registry <algorithms/generic_rosenbrock_tableaus>` – named Rosenbrock-W tableaus and references.

Factory helpers
---------------

* :doc:`get_algorithm_step <algorithms/get_algorithm_step>` – resolves step factories by enum or name.

Dependencies
------------

Implicit steps depend on :mod:`cubie.integrators.matrix_free_solvers` for the
linear and Newton–Krylov factories and reuse :class:`cubie.CUDAFactory`
utilities for JIT compilation and caching. All algorithms expect caller
supplied device callbacks for time derivatives, operator assembly, and
observable evaluation, operating on preallocated device buffers whose precision
matches the configured integration precision.
