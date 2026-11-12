CrankNicolsonStep
=================

.. currentmodule:: cubie.integrators

Defaults
--------

``algorithm="crank_nicolson"`` performs a single-stage, order-2
update and runs a backward-Euler companion solve each step; the
difference between the two supplies an embedded error estimate, so
the method defaults to adaptive PID control (``kp=0.7``,
``ki=-0.4``, step-size growth clamped to 0.5–2.0×). Both implicit
solves use the shared Newton–Krylov defaults listed in
:ref:`algorithm-defaults`.

.. autoclass:: CrankNicolsonStep
    :members:
    :show-inheritance:
