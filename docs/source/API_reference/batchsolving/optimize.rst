Optimisation
============

.. currentmodule:: cubie.batchsolving.optimize

:meth:`Solver.optimize <cubie.batchsolving.solver.Solver.optimize>`
calls :func:`run_optimization`. The launch functions choose the block
sizes and residencies cubie times.

.. autofunction:: run_optimization

.. autoclass:: OptimizeResult
    :members:
    :show-inheritance:

.. autoclass:: LaunchResult
    :members:
    :show-inheritance:

.. autofunction:: launch_candidates

.. autofunction:: launch_blocksizes

.. autofunction:: default_launch

.. autofunction:: most_resident_runs

.. autofunction:: resident_blocks_within_l2

.. autofunction:: apply_launch

.. autodata:: LOCAL_LAUNCH_BLOCKSIZES

.. autodata:: SHARED_LAUNCH_BLOCKSIZES

.. autodata:: RESIDENT_FOOTPRINT_L2_FRACTION

.. autodata:: RESIDENCY_CUT_MIN_FRAME_BYTES

.. autodata:: BUDGET_BLOCKSIZE

.. autodata:: LAUNCH_OCCUPANCY_TIE
