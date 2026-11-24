GenericRosenbrockWStep
=====================

.. currentmodule:: cubie.integrators.algorithms

:class:`GenericRosenbrockWStep` provides a linearly implicit Rosenbrock-W
integrator that requires only one Jacobian factorisation per step. The factory
consumes
:class:`~cubie.integrators.algorithms.generic_rosenbrockw_tableaus.RosenbrockTableau`
instances and couples user-supplied device callbacks with matrix-free linear
solves from :mod:`cubie.integrators.matrix_free_solvers`.

Defaults
--------

``algorithm="rosenbrock"`` integrates with the ``ros3p`` tableau
(three stages, order 3, with an embedded error estimate) under
Gustafsson predictive control (step-size growth clamped to
0.2–8.0×). Rosenbrock-W methods are linearly implicit —
each stage performs one linear solve rather than a Newton
iteration — so of the solver settings in
:ref:`algorithm-defaults` only the Krylov and preconditioner
defaults apply; the ``newton_*`` parameters do not. Named schemes
are listed on the
:doc:`Rosenbrock tableau registry <generic_rosenbrock_tableaus>`
page.

.. autoclass:: GenericRosenbrockWStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_rosenbrock_w.RosenbrockWStepConfig
    :members:
