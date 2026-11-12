Architecture Overview
=====================

This page describes CuBIE's internal design patterns.  Understanding
these is essential for contributing new algorithms, metrics, or other
components.

Component Ownership Tree
------------------------

A typical solve involves the following hierarchy::

   Solver
   └── BatchSolverKernel
       └── SingleIntegratorRun
           ├── IVPLoop
           ├── OutputFunctions
           ├── Algorithm step (e.g. ERKStep, FIRKStep)
           └── Controller (PI, PID, Gustafsson)

Each component is a :class:`~cubie.CUDAFactory.CUDAFactory` subclass that
generates CUDA device functions.

The CUDAFactory Lifecycle
-------------------------

Every CUDA-generating component follows the same pattern:

1. **Configuration.**  An ``attrs``-based config class
   (subclass of ``CUDAFactoryConfig``) holds compile-time settings.
   The config is set via ``setup_compile_settings()``.

2. **Build.**  The ``build()`` method generates and returns compiled
   device functions.  Subclasses override this method.

3. **Cache.**  The result of ``build()`` is cached.  Subsequent accesses
   via the ``device_function`` property return the cached result without
   rebuilding.

4. **Invalidation.**  When ``update_compile_settings()`` is called with
   changed values, the cache is invalidated and the next property access
   triggers a rebuild.

.. important::

   Never call ``build()`` directly.  Always access compiled functions
   through properties (e.g. ``device_function``), which handle caching.

Config Hashing and Change Detection
------------------------------------

``_CubieConfigBase`` provides ``values_hash`` (a hex digest of the frozen
config) and ``values_tuple`` for comparison.  ``update()`` returns two
sets: the names of fields that were *requested* to change and those that
*actually* changed.  Only actual changes invalidate the cache.

``MultipleInstanceCUDAFactory``
-------------------------------

Some components appear multiple times in the tree (e.g. one algorithm
step per stage in a multi-stage method).  ``MultipleInstanceCUDAFactory``
manages a collection of identically-typed factories distinguished by a
prefix string, so their buffers and settings don't collide.

Attrs Conventions
-----------------

- Float attributes are stored with an underscore prefix (e.g. ``_atol``)
  and exposed via a property that casts through ``self.precision``.
- ``__init__`` signatures use the *public* name (no underscore).
- Validators from ``cubie._utils`` accept both Python floats and NumPy
  scalar dtypes.
- Never add ``alias`` to underscored fields.
