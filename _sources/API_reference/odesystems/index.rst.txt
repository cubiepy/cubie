ODE systems
===========

``cubie.odesystems``
--------------------

.. currentmodule:: cubie.odesystems

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   create_ode_system
   symbolic_ode
   base_ode
   ode_data
   system_values
   system_sizes
   ode_cache

The :func:`create_ODE_system` helper is the main entry point. It consumes
symbolic :mod:`sympy` equations and creates :class:`SymbolicODE` instances that
inherit CUDA compilation behaviour from :class:`cubie.CUDAFactory`.
:class:`BaseODE` sets the abstract requirements, and ``SymbolicODE`` is
currently its concrete implementation. Base classes and data containers expose
the precision-aware metadata required by integrator factories.

.. toctree::
   :maxdepth: 2
   :caption: Subpackages
   :titlesonly:

   symbolic

Core API
--------

* :doc:`create_ODE_system <create_ode_system>` – builds :class:`SymbolicODE` instances from SymPy
  expressions.
* :doc:`SymbolicODE <symbolic_ode>` – concrete :class:`BaseODE` subclass for symbolic systems.
* :doc:`BaseODE <base_ode>` – abstract base exposing CUDA compilation helpers.

Data containers and caches
--------------------------

* :doc:`ODEData <ode_data>` – captures compile-time metadata such as precision and state
  sizes.
* :doc:`SystemValues <system_values>` – runtime container for system-specific constants.
* :doc:`SystemSizes <system_sizes>` – records shape information for state and observable
  vectors.
* :doc:`ODECache <ode_cache>` – caches compiled device functions and solver helpers.

Dependencies
------------

- :class:`SymbolicODE` subclasses :class:`cubie.CUDAFactory` so integrator loops
  can request compiled CUDA device functions directly.
- Precision handling relies on :mod:`cubie._utils` helpers and
  :mod:`cubie.cuda_simsafe` to provide simulator-safe coercions.
- Generated kernels are consumed by :mod:`cubie.integrators` factories during
  loop construction.
