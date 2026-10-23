FIRKStep
========

.. currentmodule:: cubie.integrators.algorithms

:class:`FIRKStep` provides a fully implicit Runge--Kutta integrator that solves
coupled stage systems with the cached Newton--Krylov helpers supplied by
:mod:`cubie.integrators.matrix_free_solvers`. The factory consumes
:class:`~cubie.integrators.algorithms.generic_firk_tableaus.FIRKTableau`
instances, exposing high-order Gauss--Legendre and Radau IIA schemes for stiff
problems while preserving adaptive error control through embedded weights.

Defaults
--------

``algorithm="firk"`` integrates with the ``firk_gauss_legendre_2``
tableau (two-stage Gauss–Legendre, order 4), solving all stages
together as one coupled Newton–Krylov system with the shared
defaults listed in :ref:`algorithm-defaults`. The default tableau
has no embedded error estimate, so it runs under fixed-step
control; tableaus that provide one — ``radau`` (Radau IIA-5), for
example — enable the family's Gustafsson predictive defaults
(step-size growth clamped to 0.2–8.0×). Named schemes
are listed on the
:doc:`FIRK tableau registry <generic_firk_tableaus>` page.

.. autoclass:: FIRKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_firk.FIRKStepConfig
    :members:
