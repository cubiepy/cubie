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

``algorithm="dirk"`` integrates with the ``l_stable_dirk_3`` tableau
(L-stable, stiffly accurate, three stages, order 3). That tableau has
no embedded error estimate, so the default is fixed-step control;
choosing a DIRK tableau that provides an estimate enables the
family's Gustafsson predictive defaults (step-size growth clamped
to 0.2–8.0×). Each stage runs one Newton–Krylov
solve with the shared defaults listed in
:ref:`algorithm-defaults`. The implicit-midpoint, SDIRK, and
Hairer–Wanner L-stable schemes on the
:doc:`DIRK tableau registry <generic_dirk_tableaus>` page resolve
to this class.

.. autoclass:: DIRKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_dirk.DIRKStepConfig
    :members:
