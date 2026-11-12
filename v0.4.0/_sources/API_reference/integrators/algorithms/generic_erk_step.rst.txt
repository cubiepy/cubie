ERKStep
=======

.. currentmodule:: cubie.integrators.algorithms

The :class:`ERKStep` factory wraps a configurable explicit Runge--Kutta
integrator. It accepts any :class:`~cubie.integrators.algorithms.generic_erk.ERKTableau`
and ships with PI step-control defaults tuned for the embedded Dormand--Prince
pair. The factory performs staged right-hand-side evaluations on the GPU and
supports optional driver and observable callbacks.

Defaults
--------

``algorithm="erk"`` integrates with the ``dormand-prince-54``
tableau — explicit Dormand–Prince 5(4), seven stages, with an
embedded fourth-order error estimate — under adaptive PI control
(``kp=0.7``, ``ki=-0.4``, ``safety=0.9``, step-size growth clamped
to 0.2–10.0×). ``dopri54``, ``rk45``, and ``ode45`` are aliases for
the same scheme; the other named schemes in the
:doc:`ERK tableau registry <generic_erk_tableaus>` resolve to this
class too, falling back to fixed-step control when a tableau has no
error estimate (:ref:`algorithm-defaults`). Explicit methods need no
nonlinear or linear solver.

.. autoclass:: ERKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_erk.ERKStepConfig
    :members:
