Jacobians and Symbolic Differentiation
======================================

Implicit algorithms and Rosenbrock-W methods need derivatives of the
right-hand-side function :math:`f(t, x)`.  This page describes how CuBIE
obtains them.

What is the Jacobian?
---------------------

The Jacobian matrix :math:`J` of an :math:`n`-state ODE system has entries:

.. math::

   J_{ij} = \frac{\partial f_i}{\partial x_j}

Element :math:`J_{ij}` tells us how the rate of change of state :math:`i`
responds to a perturbation in state :math:`j`.  Implicit solvers need the
product :math:`J\, v` (a *Jacobian--vector product*, or JVP) inside their
Krylov iterations (see :doc:`solvers`).

Some solvers compute :math:`J` numerically by finite differences.  This is
fragile for stiff systems and expensive when :math:`n` is large.  CuBIE
instead uses *symbolic differentiation*.

Symbolic Differentiation Pipeline
----------------------------------

When you create a :class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`,
CuBIE:

1. Parses the equations into SymPy expressions.
2. Differentiates each :math:`f_i` with respect to every state
   :math:`x_j` symbolically.
3. Applies algorithmic chain-rule steps to group common subexpressions,
   producing an efficient *JVP function* :math:`(x, v) \mapsto J\,v`
   rather than an explicit matrix.
4. Generates CUDA-compatible code strings and writes them to a cached
   file.

The result is an exact, matrix-free JVP that runs as compiled device code.

Auxiliary Caching
-----------------

Many subexpressions are shared between the JVP evaluation and the
right-hand-side evaluation.  CuBIE identifies these shared terms and
caches them in a *prepare* step (``"prepare_jac"`` solver helper).  The
subsequent JVP call (``"calculate_cached_jvp"``) reuses the cached
intermediates, avoiding redundant work.

The solver helpers are retrieved through
:meth:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE.get_solver_helper`:

``"linear_operator"``
   Basic :math:`J\,v` product (no caching).

``"linear_operator_cached"``
   :math:`J\,v` product that reuses prepared intermediates.

``"prepare_jac"``
   Evaluates and stores shared subexpressions.

``"calculate_cached_jvp"``
   Computes :math:`J\,v` using the prepared cache.

Time Derivative for Rosenbrock-W
--------------------------------

Rosenbrock-W methods additionally require the explicit time derivative of
:math:`f`:

.. math::

   \frac{\partial f}{\partial t}
   + \sum_i \frac{\partial f}{\partial d_i}\, \dot{d}_i

where the :math:`d_i` are time-dependent driver terms.  CuBIE generates
this as the ``"time_derivative_rhs"`` solver helper.
