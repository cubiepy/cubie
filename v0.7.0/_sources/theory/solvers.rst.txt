Nonlinear Solvers
=================

Implicit integration methods (DIRK, FIRK) produce a system of nonlinear
equations at each time step.  This page describes how CuBIE solves those
equations on the GPU.

Why a Solver is Needed
----------------------

An implicit Runge--Kutta stage satisfies:

.. math::

   k_i = f\!\Bigl(t_n + c_i h,\; x_n + h \sum_j a_{ij}\, k_j\Bigr)

Because :math:`k_i` appears on both sides, we cannot simply evaluate the
right-hand side.  Instead we must find the :math:`k_i` that make the
equation hold---a root-finding problem:

.. math::

   G(K) = K - F(K) = 0

where :math:`K` stacks all stage vectors and :math:`F` wraps the
right-hand-side evaluations.

Newton Iteration
----------------

CuBIE uses a simplified Newton method.  Starting from an initial guess
:math:`K^{(0)}` (typically extrapolated from the previous step), each
iteration solves:

.. math::

   M\, \Delta K = -G(K^{(m)}), \qquad K^{(m+1)} = K^{(m)} + \Delta K

where :math:`M \approx \partial G / \partial K` is formed from the
Jacobian of :math:`f` and the tableau coefficients.  In the simplified
variant the Jacobian is evaluated once at the start of the step rather
than every iteration.

Convergence is checked by monitoring
:math:`\|\Delta K\| / \|K^{(m+1)}\|`.  If the ratio stalls, the step is
rejected and retried with a smaller :math:`h`.

Krylov Linear Solver
--------------------

The linear system :math:`M \Delta K = r` is large (:math:`s \times n` for
an :math:`s`-stage method with :math:`n` states), and forming :math:`M`
explicitly would require :math:`O(n^2)` storage per thread---prohibitive
on a GPU.  CuBIE therefore uses a *matrix-free Krylov* solver that needs
only the action of :math:`M` on a vector:

.. math::

   v \mapsto M\, v = v - h\, (A \otimes I)\, (I \otimes J)\, v

The Jacobian--vector product :math:`J\, v` is computed symbolically by
CuBIE's code generator (see :doc:`jacobians`), so no matrix is ever
stored.

Preconditioning
^^^^^^^^^^^^^^^

Krylov methods converge faster when the system is *preconditioned*:
instead of solving :math:`M x = b` directly, we solve
:math:`P^{-1} M x = P^{-1} b` for a cheaply invertible :math:`P` that
approximates :math:`M`.

CuBIE uses a *Neumann-series preconditioner*:

.. math::

   P^{-1} \approx I + N + N^2 + \cdots + N^k

where :math:`N = I - M` and :math:`k` is a low-order truncation
(typically 1--3).  This requires only repeated matrix--vector products
and no factorisation, making it well suited to the GPU's throughput
model.

Return Code Encoding
--------------------

CuBIE encodes the Newton iteration count into the upper 16 bits of each
step's integer return code:

.. code-block:: python

   iterations = (code >> 16) & 0xFFFF
   status     = code & 0xFFFF   # compare with IntegratorReturnCodes

This allows post-solve analysis of solver effort without extra output
buffers.  See :doc:`/user_guide/results` for how to access iteration
counters.
