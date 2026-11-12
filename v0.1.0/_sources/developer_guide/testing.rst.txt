Testing Guide
=============

CuBIE uses ``pytest`` with several custom markers and conventions.

Running Tests
-------------

.. code-block:: bash

   # Full suite (requires CUDA GPU)
   python -m pytest

   # CPU-only tests (CUDASIM mode)
   export NUMBA_ENABLE_CUDASIM=1
   python -m pytest -m "not nocudasim and not cupy"

   # Specific module
   python -m pytest tests/integrators/

   # Specific test pattern
   python -m pytest -k test_solver

Test Markers
------------

``nocudasim``
   Test requires a real GPU; skip in CUDASIM mode.

``cupy``
   Test requires CuPy; skip if not installed.

``slow``
   Long-running test; useful with ``-m "not slow"`` for quick
   iterations.

``sim_only``
   Test only runs in CUDASIM mode.

``specific_algos``
   Test targets a specific algorithm and may be skipped in broad
   sweeps.

CUDASIM Mode
------------

Setting ``NUMBA_ENABLE_CUDASIM=1`` before import causes Numba to emulate
CUDA on the CPU.  CI pipelines without GPUs use this mode.  CUDASIM is
single-threaded and much slower than GPU execution, so keep CUDASIM-
compatible tests lightweight.

Instrumented Test Copies
------------------------

``tests/integrators/algorithms/instrumented/`` contains copies of
algorithm device functions with added logging.  These are used to verify
internal step behaviour (stage values, error estimates, etc.).

.. important::

   When you change a source algorithm, replicate the change in its
   instrumented copy.  The instrumented tests will fail if the two
   diverge.

Fixture Patterns
----------------

``tests/conftest.py`` provides shared fixtures for ODE systems, solver
instances, and common parameter sets.  Prefer using these fixtures over
constructing test objects manually.

Key fixtures include pre-built ODE systems (Lotka--Volterra, van der Pol,
linear test systems) and solver factories parameterised by algorithm.

Philosophy
----------

- **Failing tests are good.**  Tests should assert intended behaviour.
  If a test fails, fix the code, not the test (unless the test itself is
  wrong).
- **Prefer real objects over mocks.**  Use actual CuBIE objects and real
  (small) ODE systems.  Never patch ``is_device`` checks or CUDA
  availability.
- **Never mark tests ``xfail``** or use ``importorskip`` to hide
  failures.
