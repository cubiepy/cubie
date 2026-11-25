Candidate comparison
====================

.. currentmodule:: cubie.batchsolving.comparison

:meth:`Solver.calibrate <cubie.batchsolving.solver.Solver.calibrate>`
and :meth:`Solver.optimize <cubie.batchsolving.solver.Solver.optimize>`
both time their candidates with a :class:`ComparisonRunner`, which
switches one solver between candidates and times each on the same batch.

.. autoclass:: ComparisonRunner
    :members:

.. autofunction:: rank_timings

.. autoclass:: Candidate
    :members:
    :show-inheritance:

.. autoclass:: CandidateTiming
    :members:
    :show-inheritance:

Batch sizing
------------

.. autofunction:: validate_sizing

.. autofunction:: tail_safe_runs

.. autofunction:: unused_wave_share

.. autofunction:: warn_low_waves

GPU warm-up
-----------

.. autofunction:: busy_launch

.. autofunction:: warm_clocks

Helpers
-------

.. autofunction:: settings_label

.. autofunction:: device_to_host

.. autofunction:: worth_parallelising

Timing protocol constants
-------------------------

.. autodata:: ROUNDS

.. autodata:: SOLVES_PER_ROUND

.. autodata:: WARM_MS

.. autodata:: BUSY_CHAIN

.. autodata:: SUCCESS_TIER_FRACTION

.. autodata:: TIMED_WAVES_FLOOR

.. autodata:: HOST_BYTES_PER_RUN

.. autodata:: WORKER_STARTUP_SECONDS
