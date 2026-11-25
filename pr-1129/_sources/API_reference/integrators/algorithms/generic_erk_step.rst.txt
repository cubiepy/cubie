ERKStep
=======

.. currentmodule:: cubie.integrators.algorithms

The :class:`ERKStep` factory wraps a configurable explicit Runge--Kutta
integrator. It accepts any :class:`~cubie.integrators.algorithms.generic_erk.ERKTableau`
and ships with adaptive integral step-control defaults. The factory performs
staged right-hand-side evaluations on the GPU and supports optional driver
and observable callbacks.

Defaults
--------

``algorithm="erk"`` uses the ``tsit5`` method, Tsitouras's explicit
5(4) pair with seven stages. With ``tsit5``, cubie adapts the step
with the ``i`` controller and ``integral_gain=1.2`` by default. Cubie
uses this class for every method in the
:doc:`ERK tableau registry <generic_erk_tableaus>`, and steps at a
fixed size with the methods that have no error estimate. You'll
find the default settings of each method in :ref:`algorithm-defaults`.

.. autoclass:: ERKStep
    :members:
    :show-inheritance:

.. autoclass:: cubie.integrators.algorithms.generic_erk.ERKStepConfig
    :members:
