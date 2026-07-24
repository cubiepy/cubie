Code Generation Pipeline
========================

CuBIE transforms symbolic ODE definitions into compiled CUDA device
functions.  SymPy is the parse layer for string and SymPy input; every
expression converts to a lightweight interned expression IR (the
``engine`` package) at the parse boundary, and every later stage —
classification, structural simplification, differentiation, CSE,
hashing, and printing — runs on the IR.  This page describes the
pipeline.

Pipeline Overview
-----------------

::

   String / SymPy / CellML equations
       → SymPy parse layer and CellML substitutions
       → engine IR (hash-consed expression nodes)
       → normalise/classify → structural simplification for DAEs
       → IndexedBases (state/param/observable symbols)
       → JVPEquations (Jacobian-vector product expressions)
       → IR printer → code strings
       → ODEFile (written to generated/ directory)
       → Numba JIT → CUDA device functions

Parser
------

The parser in ``src/cubie/odesystems/symbolic/parsing/`` tokenises the
equation strings, identifies states (variables with ``d<name>`` on the
left-hand side), parameters, constants, and observables, and converts
every expression to engine IR before returning.

Array references
----------------

The IR represents array values as ``Arr(name, index)`` nodes. The
printer maps scalar names to fixed state, parameter, driver, and output
indices.

JVPEquations
------------

For implicit algorithms, the pipeline differentiates each RHS expression
with respect to every state variable, applies chain-rule grouping to
share subexpressions, and produces a function
:math:`(x, v) \mapsto J\,v`.  See :doc:`/theory/jacobians`.

Expression engine and printer
-----------------------------

``src/cubie/odesystems/symbolic/engine/`` holds the expression IR the
compute phase runs on: nodes are interned (structurally
identical expressions are the same object), so substitution,
differentiation, and common-subexpression elimination are single passes
over a DAG.  The engine's printer renders IR as Numba-CUDA source:

- Mapping function calls to their ``math.*``/builtin equivalents.
- Wrapping numeric literals in ``precision(...)`` casts.
- Rewriting small integer powers to multiplication chains and half
  powers to ``math.sqrt``.
- Emitting ``Piecewise`` selections as nested ternaries.

``print_cuda_multiple`` renders a list of assignments as source lines;
shared-subexpression extraction happens beforehand on the IR.

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

Each system uses one generated Python module at
``generated/<system_name>/<system_name>.py``. Solver helpers are added
to that module as they are requested. Numba compiles the factories on
import.
