Optimisation
============

.. currentmodule:: cubie.batchsolving.optimize

:meth:`Solver.optimize <cubie.batchsolving.solver.Solver.optimize>`
calls :func:`run_optimization`. Cubie times your solver under a range
of configurations, such as where it keeps its working values in GPU
memory, which loops it unrolls, and the block sizes and residency
levels chosen by the launch functions below. It then keeps the fastest
configuration.

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
