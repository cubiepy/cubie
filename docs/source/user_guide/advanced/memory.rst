GPU Memory Management
=====================

CuBIE manages GPU memory (VRAM) automatically, but understanding the
available options lets you run larger batches and avoid out-of-memory
errors. The examples on this page use this Lotka-Volterra system:

.. code-block:: python

   from cubie import Solver, create_ODE_system

   LV = create_ODE_system(
       """
       dx = a*x - b*x*y
       dy = -c*y + d*x*y
       """,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )

Default Behaviour
-----------------

A solver keeps its device and pinned host buffers between calls to
:meth:`~cubie.batchsolving.solver.Solver.solve`. A batch of the same
size or smaller reuses them; a larger batch replaces them. Closing the
solver returns its memory to the device and the operating system.

When you pass device arrays and ``on_device=True`` in a loop, write
each batch into the same arrays with ``copy_to_device`` rather than
allocating new ones: freeing a Numba device array waits for all work
on the GPU.

VRAM Limits
-----------

CuBIE estimates the available VRAM and sizes the batch accordingly.  You
can override the proportion of VRAM that CuBIE is allowed to use:

.. code-block:: python

   solver = Solver(LV, algorithm="ode45", mem_proportion=0.7)

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

Cubie runs every solver in a stream group on one CUDA stream, so the GPU
works through their solves one after another. Cubie gives each group
its own stream, so if you want two solvers' work to be free to run on
the GPU at the same time, put them in different groups with
``stream_group``. Cubie puts each solver in the ``"solver"`` group
unless you give it a different group name.

.. code-block:: python

   fast = Solver(LV, algorithm="ode45", stream_group="fast")
   slow = Solver(LV, algorithm="ode45", stream_group="slow")
