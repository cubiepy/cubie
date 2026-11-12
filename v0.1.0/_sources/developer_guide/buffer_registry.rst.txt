Buffer Registry
===============

CuBIE centralises GPU memory management through the
:mod:`~cubie.buffer_registry` module.  Every CUDA-generating component
registers its memory requirements, and the registry computes a layout
that is allocated once at kernel launch time.

``CUDABuffer`` Descriptors
--------------------------

Each buffer is described by a ``CUDABuffer``:

``name``
   Unique identifier (scoped to its parent component).

``size``
   Number of elements.

``location``
   ``"shared"`` (on-chip, fast, limited to ~48 KB per block) or
   ``"local"`` (per-thread, in registers/L1).

``persistent``
   If ``True``, the buffer survives across steps (e.g. state arrays).
   Non-persistent buffers can be aliased.

``aliases``
   Name of another buffer that this one can share storage with, provided
   their lifetimes do not overlap.

``precision``
   Element dtype (``np.float32`` or ``np.float64``).

Registering Buffers
-------------------

Components register buffers in their ``build()`` method:

.. code-block:: python

   from cubie.buffer_registry import buffer_registry

   buffer_registry.register(
       name="stage_k",
       parent=self,
       size=self.n_states * self.n_stages,
       location="shared",
       precision=self.precision,
   )

Allocators
----------

After all components have registered, the registry computes offsets and
produces allocator callables:

``get_allocator(name, parent)``
   Returns a function that, given a base pointer and thread index,
   returns a typed array slice for the named buffer.

``get_toplevel_allocators(kernel)``
   Returns ``(shared_allocator, local_allocator)`` for the top-level
   kernel launch.

``get_child_allocators(parent, child)``
   Delegates a region of the parent's allocation to a child component.

Shared Memory Budget
--------------------

Shared memory is limited (~48 KB default, configurable up to 100 KB on
some architectures).  When the total exceeds the budget, CuBIE reduces
the block size (fewer threads per block) to fit.

Aliasing buffers with non-overlapping lifetimes is the main tool for
reducing shared memory pressure.  For example, intermediate stage vectors
that are only needed during the step computation can alias the output
writeback buffer.

Layout Queries
--------------

``buffer_registry.shared_buffer_size(parent)``
   Total shared memory bytes for a component and its children.

``buffer_registry.local_buffer_size(parent)``
   Total local memory bytes (non-persistent).

``buffer_registry.persistent_local_buffer_size(parent)``
   Total local memory bytes (persistent).

Use these to verify that your component's memory footprint is reasonable.
