Calibration
===========

.. currentmodule:: cubie.batchsolving.calibration

:meth:`Solver.calibrate <cubie.batchsolving.solver.Solver.calibrate>`
calls :func:`run_calibration`. The ``*_specs`` functions list the
candidates cubie races at each stage.

.. autofunction:: run_calibration

.. autoclass:: CalibrationResult
    :members:
    :show-inheritance:

.. autoclass:: CandidateResult
    :members:
    :show-inheritance:

.. autoclass:: CandidateSpec
    :members:
    :show-inheritance:

.. autofunction:: preconditioner_specs

.. autofunction:: linear_solver_specs

.. autofunction:: error_option_specs

.. autofunction:: order_specs

.. autofunction:: erk_specs

.. autodata:: CALIBRATION_FAMILIES

.. autodata:: FAMILY_REPRESENTATIVES

.. autodata:: FAMILY_ORDERS

.. autodata:: PRECONDITIONERS
