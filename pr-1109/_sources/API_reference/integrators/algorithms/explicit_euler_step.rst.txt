ExplicitEulerStep
=================

.. currentmodule:: cubie.integrators

Defaults
--------

``algorithm="euler"`` performs a single-stage, order-1 forward-Euler
update under fixed-step control. Forward Euler carries no embedded
error estimate, so adaptive control is unavailable, and as an
explicit method it needs no nonlinear or linear solver. See
:ref:`algorithm-defaults` for the defaults shared across algorithms.

.. autoclass:: ExplicitEulerStep
    :members:
    :show-inheritance:
