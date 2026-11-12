Adding Algorithms
=================

CuBIE's algorithm system is designed to make adding new integration
methods straightforward, from simple tableau entries to entirely new
algorithm families.

Adding an ERK Tableau
---------------------

The simplest case.  Add a new entry to ``ERK_TABLEAU_REGISTRY`` in
``src/cubie/integrators/algorithms/generic_erk_tableaus.py``:

.. code-block:: python

   ERK_TABLEAU_REGISTRY["my_method_43"] = ERKTableau(
       a=np.array([...]),       # Butcher A matrix (lower-triangular)
       b=np.array([...]),       # weights
       b_hat=np.array([...]),   # embedded weights (or None if non-adaptive)
       c=np.array([...]),       # nodes
       order=4,
       name="my_method_43",
   )

The method is immediately available as
``Solver(system, algorithm="my_method_43")``.

Adding a DIRK, FIRK, or Rosenbrock Tableau
-------------------------------------------

The process is analogous.  Each family has its own tableau registry and
dataclass:

- ``DIRK_TABLEAU_REGISTRY`` in ``generic_dirk_tableaus.py``
- ``FIRK_TABLEAU_REGISTRY`` in ``generic_firk_tableaus.py``
- ``ROSENBROCK_TABLEAUS`` in ``generic_rosenbrockw_tableaus.py``

FIRK tableaus must also supply a transformation matrix ``T`` and its
inverse for the stage-coupled solver.

Adding a New Algorithm Family
-----------------------------

For an entirely new algorithm type:

1. **Subclass** ``ODEExplicitStep`` or ``ODEImplicitStep`` (in
   ``src/cubie/integrators/algorithms/``).

2. **Implement** ``build_step()``, which returns a CUDA device function
   that performs one integration step.

3. **Register buffers** via the buffer registry for any working arrays
   the step function needs (see :doc:`buffer_registry`).

4. **Register** the new class in ``_ALGORITHM_REGISTRY`` in
   ``src/cubie/integrators/algorithms/__init__.py``.

5. **Declare defaults** by setting ``StepControlDefaults`` on the class
   to specify preferred controller settings (tolerances, step-size
   bounds, controller type).

Implicit Helpers
^^^^^^^^^^^^^^^^

Implicit algorithms typically need Jacobian--vector products and solver
infrastructure.  Use ``build_implicit_helpers()`` to request these from
the ODE system's code generator:

.. code-block:: python

   helpers = self.system.get_solver_helper("linear_operator_cached")

The available helpers are documented in :doc:`/theory/jacobians`.
