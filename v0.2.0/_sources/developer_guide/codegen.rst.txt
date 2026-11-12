Code Generation Pipeline
========================

CuBIE uses SymPy to transform symbolic ODE definitions into compiled CUDA
device functions.  This page describes the pipeline.

Pipeline Overview
-----------------

::

   String equations
       → SymPy parser
       → IndexedBases (state/param/observable symbols)
       → JVPEquations (Jacobian-vector product expressions)
       → CUDAPrinter → code strings
       → ODEFile (written to generated/ directory)
       → Numba JIT → CUDA device functions

Parser
------

The parser in ``src/cubie/odesystems/symbolic/parsing/`` tokenises the
equation strings, identifies states (variables with ``d<name>`` on the
left-hand side), parameters, constants, and observables, and produces
SymPy expressions.

IndexedBases
------------

State variables are represented as ``IndexedBase`` objects so that the
code generator can emit array-indexed CUDA code (e.g. ``x[0]``,
``x[1]``).

JVPEquations
------------

For implicit algorithms, the pipeline differentiates each RHS expression
with respect to every state variable, applies chain-rule grouping to
share subexpressions, and produces a function
:math:`(x, v) \mapsto J\,v`.  See :doc:`/theory/jacobians`.

``CUDAPrinter``
---------------

``CUDAPrinter`` is a SymPy code printer customised for Numba CUDA.  It
handles:

- Mapping SymPy functions to their NumPy/Numba equivalents.
- Prefixing NumPy names with ``np_`` to avoid Numba namespace clashes.
- Emitting scalar-typed intermediate variables.

``print_cuda_multiple`` batches multiple expressions into a single
function body with shared subexpression elimination.

Solver Helpers
--------------

``get_solver_helper()`` dispatches to the appropriate code-generation
path based on the requested helper name (``"linear_operator"``,
``"prepare_jac"``, ``"calculate_cached_jvp"``,
``"time_derivative_rhs"``).

Stage Utilities
---------------

``_stage_utils`` provides helpers for FIRK methods that need to generate
code for multiple coupled stages simultaneously, including
block-structured linear algebra and transformation matrices.

Generated File Structure
------------------------

Generated files are written to ``generated/<system_name>/`` and include:

- ``rhs.py`` --- right-hand-side function.
- ``jvp.py`` --- Jacobian--vector product.
- ``prepare_jac.py`` --- shared subexpression cache.
- ``time_derivative.py`` --- explicit time derivative (for Rosenbrock-W).

These are standard Python files that Numba compiles on import.
