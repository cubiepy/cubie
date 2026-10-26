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
- Which parameters are swept, or the value of a parameter that
  isn't (those values are compiled into the code).
- Floating-point precision (``float32`` vs ``float64``).
- Algorithm choice or algorithm settings.
- Output configuration (saved variables, summary metrics).

Changing the values of swept *parameters* or *initial values* does
**not** trigger recompilation---those are runtime inputs.

Numba Kernel Cache
------------------

After CuBIE generates Python source, Numba JIT-compiles it into GPU
machine code.  CuBIE caches the compiled kernels to disk so that the
second run with the same configuration skips the JIT step entirely.
The kernel cache lives inside the ``generated/`` directory, under
``generated/<system_name>/CUDA_cache_<hash>/``.

Cache Settings
--------------

Pass these settings to the :class:`~cubie.Solver`, to
:func:`~cubie.solve_ivp`, or to :meth:`~cubie.Solver.update`:

``cache_enabled=True`` (default)
   Load compiled kernels from the disk cache and save new ones to it.

``cache_enabled=False``
   Compile every kernel without reading or writing the disk cache.

``cache_dir="path/to/dir"``
   Store the compiled-kernel cache in this directory.  The generated
   Python source still goes to ``generated/`` in the working directory.

``max_cache_entries=N``
   Keep at most ``N`` compiled kernels per cache directory, removing
   the least recently used kernel first.  ``0`` keeps every kernel.
   The default comes from the ``CUBIE_MAX_CACHE_ENTRIES`` environment
   variable and is ``0`` when it is unset.

.. code-block:: python

   solver = qb.Solver(system, algorithm="dormand-prince-54",
                       cache_dir="/tmp/cubie_cache")

Clearing Caches
---------------

Delete the ``generated/`` directory to clear both caches: it holds the
generated source files and, in the ``CUDA_cache_*`` subdirectories, the
compiled kernels.  To skip the kernel cache for a single solver, pass
``cache_enabled=False``.  If you have edited CuBIE's own code-generation
internals, clear ``generated/`` manually — the kernel cache key does
not account for package-code changes.
