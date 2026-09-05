Symbolic
========

``cubie.odesystems.symbolic``
-----------------------------

.. currentmodule:: cubie.odesystems.symbolic

The symbolic subpackage implements the SymPy-driven pipeline that generates CUDA
kernels for right-hand-side evaluations and Newton–Krylov helpers. It parses
symbolic systems, emits CUDA ``dxdt`` kernels, and packages metadata required by
:class:`cubie.odesystems.SymbolicODE` so integrator loops can consume compiled
functions directly.

Key helpers
-----------

* ``builders`` – utilities that assemble CUDA source strings from SymPy
  expressions.
* ``codegen`` – orchestrates SymPy code generation for device kernels.
* ``templates`` – shared Jinja templates for kernel emission.
* ``transforms`` – symbolic transformations applied before code generation.

Dependencies
------------

The code generation workflow relies on :mod:`sympy` for symbolic manipulation
and :mod:`numba.cuda` for compiling emitted kernels. Generated code is cached via
:class:`cubie.CUDAFactory` and consumed by :mod:`cubie.integrators` during loop
assembly.
