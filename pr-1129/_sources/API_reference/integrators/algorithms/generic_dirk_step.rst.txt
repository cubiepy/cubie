DIRKStep
========

.. currentmodule:: cubie.integrators.algorithms

:class:`DIRKStep` provides a diagonally implicit Runge--Kutta integrator that
solves stage systems with the cached Newton--Krylov helpers supplied by
:mod:`cubie.integrators.matrix_free_solvers`. The factory consumes
:class:`~cubie.integrators.algorithms.generic_dirk.DIRKTableau` instances,
exposing L-stable SDIRK and ESDIRK schemes for stiff problems while preserving
adaptive error control through embedded weights.

Defaults
--------

``algorithm="dirk"`` uses the ``l_stable_dirk_3`` method, which is
L-stable and stiffly accurate, with three stages and order 3. That
method has no error estimate, so cubie steps at a fixed size with it.
With a DIRK method that has one, such as ``kvaerno3``, cubie adapts the
step with the ``pi`` controller by default. Cubie solves each stage with Newton's method. Cubie uses this
class for every method in the
:doc:`DIRK tableau registry <generic_dirk_tableaus>`. You'll find the
default settings of each method in :ref:`algorithm-defaults`.

.. autoclass:: DIRKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_dirk.DIRKStepConfig
    :members:
