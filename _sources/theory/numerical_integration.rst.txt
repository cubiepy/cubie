Numerical Integration
=====================

This page covers the mathematical foundations behind the ODE integration
algorithms available in CuBIE. For a practical guide to choosing an
algorithm, see :doc:`/user_guide/choosing_algorithms`.

Euler's Method
--------------

The simplest approach to solving an ODE numerically is Euler's method.
Given a state :math:`x_n` at time :math:`t_n`, we compute the next state by
stepping in the direction of the derivative:

.. math::

   x_{n+1} = x_n + h\, f(t_n, x_n)

where :math:`h` is the step size and :math:`f` is the right-hand-side
function.  This is a first-order method: halving :math:`h` halves the local
error.  It is simple and cheap, but its low accuracy makes it impractical
for most real problems unless a very small step size is acceptable.

Runge--Kutta Methods
--------------------

Runge--Kutta (RK) methods improve on Euler by evaluating :math:`f` at
several intermediate points within each step and combining the results.
A general :math:`s`-stage RK method computes *stages*:

.. math::

   k_i = f\!\Bigl(t_n + c_i h,\; x_n + h \sum_{j=1}^{s} a_{ij}\, k_j\Bigr),
   \qquad i = 1, \dots, s

and then updates the state:

.. math::

   x_{n+1} = x_n + h \sum_{i=1}^{s} b_i\, k_i.

The coefficients :math:`a_{ij}`, :math:`b_i`, and :math:`c_i` are collected
in a *Butcher tableau*:

.. math::

   \begin{array}{c|c}
   \mathbf{c} & A \\
   \hline
              & \mathbf{b}^T
   \end{array}

The *order* of a method describes how quickly its error shrinks as
:math:`h \to 0`.  An order-:math:`p` method has a local truncation error
proportional to :math:`h^{p+1}`.

Embedded Error Pairs
^^^^^^^^^^^^^^^^^^^^

Many practical RK methods come in *embedded pairs*: two methods of
different orders that share the same stages.  The difference between their
results provides a free error estimate without extra function evaluations.
For example, the Dormand--Prince 5(4) pair uses the same six stages to
produce both a fifth-order and a fourth-order solution; the gap between
them estimates the local error and drives adaptive step-size control.

Explicit vs Implicit Methods
-----------------------------

In an *explicit* method the matrix :math:`A` is strictly lower-triangular:
each stage depends only on previously computed stages, so they can be
evaluated sequentially in a single forward pass.

In an *implicit* method :math:`A` has entries on or above the diagonal,
meaning stages depend on each other.  This produces a coupled system of
nonlinear equations that must be solved at every step (see
:doc:`solvers`).  The extra cost buys better stability, which matters for
*stiff* problems.

Stiffness
^^^^^^^^^

A system is *stiff* when it contains dynamics on vastly different time
scales.  The Jacobian :math:`\partial f / \partial x` then has eigenvalues
with large negative real parts, and explicit methods must take
impractically small steps to remain stable.  Implicit methods avoid this
because their stability regions extend far into the left half-plane.

Algorithm Families in CuBIE
----------------------------

**ERK (Explicit Runge--Kutta)**
   Lower-triangular :math:`A`.  Fast per step, best for non-stiff or
   mildly stiff problems.  CuBIE ships Dormand--Prince 5(4), Tsitouras
   5(4), Bogacki--Shampine 3(2), and others.

**DIRK (Diagonally Implicit Runge--Kutta)**
   :math:`A` is lower-triangular with non-zero diagonal entries.  Each
   stage requires solving a single nonlinear system of size :math:`n` (the
   number of states).  Good balance between stability and cost.

**FIRK (Fully Implicit Runge--Kutta)**
   :math:`A` is a full :math:`s \times s` matrix.  All stages are coupled,
   requiring a system of size :math:`s \times n`.  CuBIE provides
   Gauss--Legendre and Radau IIA methods, which achieve the highest
   possible order for their stage count.

**Rosenbrock-W**
   Linearly implicit methods that replace Newton iteration with a single
   linear solve per stage.  They require the Jacobian :math:`J` and the
   time derivative :math:`\partial f / \partial t`, but avoid the iterative
   convergence loop of fully implicit methods.  Well suited to moderately
   stiff problems where Newton iteration cost would dominate.

Adaptive Stepping
-----------------

Adaptive methods adjust :math:`h` at every step to keep the estimated
local error within user-specified tolerances.  The error is measured as:

.. math::

   \text{err} = \left\|
       \frac{x^{(p)} - x^{(p-1)}}
            {\text{atol} + \text{rtol} \cdot |x^{(p)}|}
   \right\|

where :math:`x^{(p)}` and :math:`x^{(p-1)}` are the higher- and
lower-order solutions from the embedded pair.

A *controller* uses this error to propose the next step size.  CuBIE
supports PI, PID, and Gustafsson controllers, each offering different
trade-offs between responsiveness and smoothness of the step-size
sequence.  See :doc:`/user_guide/choosing_algorithms` for practical
guidance.
