Memory
======

``cubie.memory``
----------------

.. currentmodule:: cubie.memory

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   memory/default_memmgr
   memory/memory_manager
   memory/array_request
   memory/array_response
   memory/stream_groups
   memory/current_cupy_stream

The memory package coordinates GPU allocations across cubie. It exposes the
package-level :class:`~cubie.memory.mem_manager.MemoryManager` through
``default_memmgr`` so integrators can request array buffers and register CUDA
streams without rewriting the coordination code. CuPy is the single device
allocation provider on a real GPU: device arrays come from CuPy's memory
pool and host staging buffers from CuPy's pinned memory pool. Supporting
modules describe allocation requests, track chunked response metadata, and
manage stream groups.

Entry point
-----------

``default_memmgr`` creates :class:`cubie.memory.mem_manager.MemoryManager`
with stream grouping ready to configure. Typical callers obtain this
singleton, register their instance identifier, and submit
:class:`cubie.memory.array_requests.ArrayRequest` objects that describe the
arrays they need.

Core manager
------------

* :doc:`default_memmgr <memory/default_memmgr>` – default :class:`MemoryManager` instance shared across
  the package.
* :doc:`MemoryManager <memory/memory_manager>` – handles allocation requests and stream registration.

Array specifications
--------------------

* :doc:`ArrayRequest <memory/array_request>` – describes requested buffers, precision factories, and chunking.
* :doc:`ArrayResponse <memory/array_response>` – returns allocated buffers and metadata for callers.

Stream coordination
-------------------

* :doc:`StreamGroups <memory/stream_groups>` – assigns host instances to CUDA streams and manages synchronisation policies.

CuPy stream interop
--------------------

* :doc:`current_cupy_stream <memory/current_cupy_stream>` – context manager that binds a Numba stream to a CuPy stream so CuPy allocations and copies stay ordered with the Numba-launched kernel.

Dependencies
------------

The package requires :mod:`numba.cuda` for kernel launch, stream management,
and context access. CuPy is required on a real GPU — it is CuBIE's single
device memory allocator, imported at package import time through
:mod:`cubie.cuda_simsafe`. Under the CUDA simulator (which never touches
device memory) CuPy is not required and the import is skipped.
