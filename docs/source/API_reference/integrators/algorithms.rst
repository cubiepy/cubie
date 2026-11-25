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

To choose an algorithm, pass its name as ``algorithm=`` to
:class:`~cubie.batchsolving.solver.Solver` or as ``method=`` to
:func:`~cubie.batchsolving.solver.solve_ivp`. To pick one, and to see
every name you can pass, read
:doc:`/user_guide/algorithms_and_tuning/choosing_algorithms`. Read on
here for the settings cubie uses with each algorithm unless you set
them yourself, and for the classes that implement the algorithms.

.. _algorithm-defaults:

Default settings
----------------

You can pass the general name of a group of methods, such as
``"erk"`` or ``"dirk"``, and cubie uses the method in the table below.
You can also pass the name of one method, such as ``"radau"``.

.. list-table::
   :header-rows: 1
   :widths: 20 40 10 30

   * - Name
     - Method
     - Order
     - Default step controller
   * - ``euler``
     - Forward Euler (explicit, one stage)
     - 1
     - ``fixed``
   * - ``backwards_euler``
     - Backward Euler (implicit, one stage)
     - 1
     - ``fixed``
   * - ``backwards_euler_pc``
     - Backward Euler with a forward Euler predictor
     - 1
     - ``fixed``
   * - ``crank_nicolson``
     - Crank--Nicolson, with a backward Euler step to estimate the
       error
     - 2
     - ``gustafsson``
   * - ``erk``
     - ``tsit5`` (explicit, seven stages)
     - 5(4)
     - ``i``
   * - ``dirk``
     - ``l_stable_dirk_3`` (diagonally implicit, three stages)
     - 3
     - ``fixed``
   * - ``firk``
     - ``firk_gauss_legendre_2`` (fully implicit, two stages)
     - 4
     - ``fixed``
   * - ``rosenbrock``
     - ``ros3p`` (Rosenbrock-W, three stages)
     - 3(2)
     - ``gustafsson``

Step controller
~~~~~~~~~~~~~~~

CuBIE has default controllers tuned for each algorithm, which may not
be ideal for your specific problem. You'll find the default controller
of every method in the tables in
:doc:`/user_guide/algorithms_and_tuning/choosing_algorithms`. To use a
different controller, pass ``step_controller``.

Cubie can only adapt the step size if it can estimate the error of
each step. With a method that has no error estimate, cubie steps at a
fixed size, and warns with a ``UserWarning`` if you ask for an
adaptive controller. Cubie steps at a fixed size with the default
``dirk`` and ``firk`` methods, because they have no error estimate. To
adapt the step, pick a method that has one, such as ``kvaerno3`` or
``radau``.

With an adaptive method, cubie limits how much the step can change
from one step to the next:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Algorithms
     - ``min_step_shrink``
     - ``max_step_growth``
   * - ERK and DIRK
     - 0.2
     - 10.0
   * - FIRK, Rosenbrock-W and Crank--Nicolson
     - 0.2
     - 8.0

When you use a method's default controller, cubie also uses gains
tuned for that method. With the ERK methods whose default controller
is ``i``, cubie uses ``integral_gain=1.2``. With a DIRK method and the
``pi`` controller, cubie uses ``integral_gain=0.3 * (q + 1) / q`` and
``proportional_gain=0.4 * (q + 1) / q``, where ``q`` is the lower of
the method's two orders. For ``kvaerno3``, of order 3(2), these gains
are 0.45 and 0.6. If you choose a different controller, cubie uses
that controller's own gains, listed in
:doc:`/user_guide/using_the_solver/configuration`.

Implicit solver settings
~~~~~~~~~~~~~~~~~~~~~~~~

With backward Euler, Crank--Nicolson, DIRK and FIRK methods, cubie
solves a nonlinear system at every step with Newton's method, and a
linear system for each Newton update. With Rosenbrock-W methods, cubie
solves only the linear systems, so cubie ignores the Newton settings
for them. See :doc:`/user_guide/using_the_solver/configuration` for what
each setting does.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Default
   * - ``newton_atol`` / ``newton_rtol``
     - A tenth of your ``atol`` and ``rtol``. In ``float32``, cubie
       raises a Newton ``rtol`` below ``4.8e-7`` to ``4.8e-7``.
   * - ``newton_max_iters``
     - ``8``
   * - ``krylov_atol`` / ``krylov_rtol``
     - Your ``atol`` and ``rtol``.
   * - ``krylov_max_iters``
     - One and a half times the number of unknowns cubie solves for in
       each linear system, rounded up. That number is the number of
       states, or with a FIRK method, the number of states times the
       number of stages.
   * - ``linear_correction_type``
     - - ``"lu"`` for DIRK, FIRK and Rosenbrock-W methods other than
         ``radau_iia_9``, and for every DAE system.
       - ``"bicgstab"`` for ``radau_iia_9`` on an ODE system.
       - ``"minimal_residual"`` for backward Euler and Crank--Nicolson
         on an ODE system.
   * - ``inexact_newton``
     - ``True`` for most DIRK and FIRK methods. ``False`` for the DIRK
       and FIRK methods marked as using exact Newton in
       :doc:`/user_guide/algorithms_and_tuning/choosing_algorithms`,
       for every other algorithm, and whenever you choose a linear
       solver other than the method's default.
   * - ``prefactored``
     - ``True``
   * - ``preconditioner_type``
     - ``"jacobi"``
   * - ``preconditioner_order``
     - ``0`` for ``"jacobi"``, and ``2`` for ``"neumann"``

Base classes
------------

* :doc:`BaseStepConfig <algorithms/base_step_config>` – the settings that explicit and implicit steps share.
* :doc:`BaseAlgorithmStep <algorithms/base_algorithm_step>` – the base class of every step, which compiles the step in the chosen precision.
* :doc:`StepCache <algorithms/step_cache>` – the compiled step and its Newton solver, with the step's order and sizes.

Explicit steps
--------------

* :doc:`ExplicitStepConfig <algorithms/explicit_step_config>` – the settings of an explicit step.
* :doc:`ExplicitEulerStep <algorithms/explicit_euler_step>` – the forward Euler step.
* :doc:`ERKStep <algorithms/generic_erk_step>` – the explicit Runge–Kutta step, for any ERK tableau.
* :doc:`ERK tableau registry <algorithms/generic_erk_tableaus>` – the named ERK methods, with references.

Implicit steps
--------------

* :doc:`ImplicitStepConfig <algorithms/implicit_step_config>` – the settings of an implicit step, including its Newton and linear solver settings.
* :doc:`BackwardsEulerStep <algorithms/backwards_euler_step>` – the backward Euler step.
* :doc:`BackwardsEulerPCStep <algorithms/backwards_euler_pc_step>` – the backward Euler step with a forward Euler predictor.
* :doc:`CrankNicolsonStep <algorithms/crank_nicolson_step>` – the Crank–Nicolson step.
* :doc:`DIRKStep <algorithms/generic_dirk_step>` – the diagonally implicit Runge–Kutta step, for any DIRK tableau.
* :doc:`DIRK tableau registry <algorithms/generic_dirk_tableaus>` – the named DIRK methods, with references.
* :doc:`FIRKStep <algorithms/generic_firk_step>` – the fully implicit Runge–Kutta step, for any FIRK tableau, including Gauss–Legendre and Radau IIA.
* :doc:`FIRK tableau registry <algorithms/generic_firk_tableaus>` – the named FIRK methods, with references.
* :doc:`GenericRosenbrockWStep <algorithms/generic_rosenbrock_step>` – the Rosenbrock-W step, for any Rosenbrock-W tableau.
* :doc:`Rosenbrock tableau registry <algorithms/generic_rosenbrock_tableaus>` – the named Rosenbrock-W methods, with references.

Factory function
----------------

* :doc:`get_algorithm_step <algorithms/get_algorithm_step>` – builds the step for an algorithm name or tableau.

Dependencies
------------

Cubie builds the Newton and linear solvers of the implicit steps from
:mod:`cubie.integrators.matrix_free_solvers`. Each step is a
:class:`~cubie.CUDAFactory.CUDAFactory`, so cubie compiles it when it is
first needed and caches the result. Cubie compiles each step with the
system's right-hand side, Jacobian products and observables, in the
precision you choose for the solve.
