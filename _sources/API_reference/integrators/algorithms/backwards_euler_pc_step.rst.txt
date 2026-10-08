BackwardsEulerPCStep
====================

.. currentmodule:: cubie.integrators

Defaults
--------

``algorithm="backwards_euler_pc"`` inherits every default from
:doc:`BackwardsEulerStep <backwards_euler_step>` — order 1,
fixed-step control, and the shared Newton–Krylov solver settings
(:ref:`algorithm-defaults`) — adding only an explicit forward-Euler
predictor evaluated before the implicit corrector runs.

.. autoclass:: BackwardsEulerPCStep
    :members:
    :show-inheritance:
