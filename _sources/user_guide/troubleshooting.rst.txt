Troubleshooting
===============

CUDA Not Found
--------------

**Symptom:** ``CudaSupportError`` or "CUDA toolkit not found".

**Fix:**

1. Verify an NVIDIA GPU is present: ``nvidia-smi``.
2. Reinstall with a toolkit extra so the toolkit wheels ship with the
   backend: ``pip install cubie[mlir-cuda12]`` (or
   ``cubie[mlir-cuda13]``).
3. If errors persist, try the deprecated numba-cuda backend:
   ``pip install cubie[cuda12]``.

Newton Solver Not Converging
-----------------------------

If an implicit algorithm reports many failed trajectories or slow
convergence:

1. **Tighten tolerances.** Try ``atol=1e-8, rtol=1e-6`` to give the
   solver more room.
2. **Increase the preconditioner order.** Higher Neumann-series orders
   improve Krylov convergence at some compute cost.
3. **Reduce the initial step size.** A large first step can push the
   Newton iteration far from convergence.
4. **Try a different algorithm.** Rosenbrock-W methods avoid Newton
   iteration entirely; FIRK methods are more robust than DIRK for very
   stiff problems.
5. **Check the equations.** Singular or near-singular Jacobians can
   prevent convergence.

Out of VRAM
------------

**Symptom:** ``CudaAPIError`` with an out-of-memory message.

**Fixes:**

- Reduce output: save fewer variables, use summaries instead of
  time-domain data.
- Enable automatic chunking (it should trigger automatically, but verify
  ``mem_proportion`` is not set too high).
- Reduce the batch size.

CUDASIM Mode
------------

For CPU-only debugging, set the environment variable *before* importing
CuBIE:

.. code-block:: bash

   # Bash
   export NUMBA_ENABLE_CUDASIM=1

   # PowerShell
   $env:NUMBA_ENABLE_CUDASIM = "1"

CUDASIM runs the CUDA kernels on the CPU in a single thread.  It is
orders of magnitude slower than GPU execution but does not require a GPU.
Useful for debugging logic errors and running in CI environments without
GPUs.  The simulator exists only on the deprecated numba-cuda backend
(``pip install cubie[cuda12]``); the default MLIR backend has no
simulator.

.. note::

   Never set ``NUMBA_ENABLE_CUDASIM`` inside Python code.  It must be
   set before the ``numba`` module is imported.

Common Attrs Validation Errors
------------------------------

CuBIE uses ``attrs`` classes for configuration.  Common mistakes:

- **Wrong type:** passing a Python ``float`` where a NumPy float is
  expected, or vice versa.  CuBIE's validators are tolerant of NumPy
  dtypes, but passing a string where a number is expected will raise
  ``TypeError``.
- **Negative tolerance:** ``atol`` and ``rtol`` must be non-negative;
  a negative value raises ``ValueError``.
- **Step size bounds:** ``dt_min`` must be less than ``dt_max``, and
  ``dt`` must be between them.
