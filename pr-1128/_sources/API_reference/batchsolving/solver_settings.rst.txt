Solver settings
===============

.. currentmodule:: cubie.batchsolving.solver_settings

A :class:`~cubie.batchsolving.solver.Solver` records the settings you
gave it in a :class:`SolverSettings`, which you can read as
``solver.given``. It also records the full set of settings, including
the ones it worked out from yours, in an :class:`EffectiveSettings`,
which you can read as ``solver.effective``.

.. autoclass:: SolverSettings
    :members:
    :show-inheritance:

.. autoclass:: EffectiveSettings
    :members:
    :show-inheritance:

.. autofunction:: owned
