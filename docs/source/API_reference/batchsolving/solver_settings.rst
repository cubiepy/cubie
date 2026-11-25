Solver settings
===============

.. currentmodule:: cubie.batchsolving.solver_settings

A :class:`~cubie.batchsolving.solver.Solver` records the settings you
give it in a :class:`SolverSettings`, which you can read as
``solver.given``, and the settings it solves with in an
:class:`EffectiveSettings`, which you can read as ``solver.effective``.

.. autoclass:: SolverSettings
    :members:
    :show-inheritance:

.. autoclass:: EffectiveSettings
    :members:
    :show-inheritance:

.. autofunction:: owned
