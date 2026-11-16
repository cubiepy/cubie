BackwardsEulerStep
==================

.. currentmodule:: cubie.integrators

Defaults
--------

``algorithm="backwards_euler"`` performs a single-stage, order-1
backward-Euler update under fixed-step control; the method carries
no embedded error estimate, so adaptive control is unavailable. Each
step solves the implicit stage equation with the shared
Newton–Krylov defaults listed in :ref:`algorithm-defaults`.

.. autoclass:: BackwardsEulerStep
    :members:
    :show-inheritance:
