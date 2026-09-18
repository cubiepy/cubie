GPU Memory Management
=====================

CuBIE manages GPU memory (VRAM) automatically, but understanding the
available options lets you run larger batches and avoid out-of-memory
errors.

Default Behaviour
-----------------

CuBIE allocates device memory through CuPy's memory pool, which recycles
allocations across calls instead of freeing and reallocating GPU memory
each time. Each call to
:meth:`~cubie.batchsolving.solver.Solver.solve` allocates the required
device arrays, runs the kernel, copies results back, then frees the
memory (returning it to the pool for reuse).

CuPy is required to run on a real GPU (install it via the
``mlir-cuda12``/``mlir-cuda13`` extras, e.g.
``pip install cubie[mlir-cuda12]``, or directly with
``pip install cupy-cuda12x``). It is not required to run under the CUDA
simulator (``NUMBA_ENABLE_CUDASIM=1``), which never touches CuPy.

VRAM Limits
-----------

CuBIE estimates the available VRAM and sizes the batch accordingly.  You
can override the proportion of VRAM that CuBIE is allowed to use:

.. code-block:: python

   solver = qb.Solver(system, algorithm="dormand-prince-54",
                       memory_settings={"mem_proportion": 0.7})

Set a lower proportion if other processes share the GPU.

Automatic Chunking
------------------

When a batch is too large to fit in VRAM in one go, CuBIE automatically
splits it into *chunks* and processes them sequentially.  The results are
concatenated transparently---you always get a single
:class:`~cubie.batchsolving.solveresult.SolveResult`.

Chunking is triggered automatically when the estimated memory requirement
exceeds the available VRAM.

Stream Groups
-------------

For advanced use, CuBIE supports running multiple chunks concurrently on
different CUDA streams via *stream groups*.  This can hide data-transfer
latency behind compute.  Configure via
``memory_settings={"stream_group": ...}`` on the ``Solver``.
