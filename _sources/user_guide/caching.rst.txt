Caching and Recompilation
=========================

CuBIE uses two caching layers to avoid redundant work: a *code-generation
cache* (Python source files) and Numba's *kernel disk cache* (compiled
machine code).

The ``generated/`` Directory
----------------------------

When CuBIE first compiles a system, it writes generated Python files into
a ``generated/`` directory inside the current working directory.  These
files contain the CUDA device functions for the right-hand side,
Jacobian helpers, and other system-specific code.

To relocate every cache layer at once, set the ``CUBIE_CACHE_DIR``
environment variable to the desired directory before starting Python,
or call :func:`cubie.cache_root.set_cache_root` at runtime (which takes
precedence over the environment variable):

.. code-block:: python

   from cubie.cache_root import set_cache_root

   set_cache_root("/data/cubie_caches")

When Recompilation Happens
--------------------------

CuBIE recompiles when any of the following change:

- The ODE equations.
- Constant values (since they are baked into the compiled code).
- Floating-point precision (``float32`` vs ``float64``).
- Algorithm choice or algorithm settings.
- Output configuration (saved variables, summary metrics).

Changing *parameters* or *initial values* does **not** trigger
recompilation---those are runtime inputs.

Numba Kernel Cache
------------------

After CuBIE generates Python source, Numba JIT-compiles it into GPU
machine code.  CuBIE caches the compiled kernels to disk so that the
second run with the same configuration skips the JIT step entirely.
The kernel cache lives inside the ``generated/`` directory, under
``generated/<system_name>/CUDA_cache_<hash>/``.

Cache Settings
--------------

On the :class:`~cubie.Solver`:

``cache=True`` (default)
   Enable both code-generation and Numba caching.

``cache=False``
   Disable caching; recompile every time.

``cache="path/to/dir"``
   Store the compiled-kernel cache in a custom directory.  (The
   generated Python source still goes to ``generated/`` in the working
   directory.)

.. code-block:: python

   solver = qb.Solver(system, algorithm="dormand-prince-54",
                       cache="/tmp/cubie_cache")

Clearing Caches
---------------

Delete the ``generated/`` directory to clear both caches: it holds the
generated source files and, in the ``CUDA_cache_*`` subdirectories, the
compiled kernels.  To skip the caches for a single run, pass
``cache=False``.  If you have edited CuBIE's own code-generation
internals, clear ``generated/`` manually — the kernel cache key does
not account for package-code changes.
