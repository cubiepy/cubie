DIRK tableau registry
=====================

.. currentmodule:: cubie.integrators.algorithms

``DIRK_TABLEAU_REGISTRY`` exposes diagonally implicit Runge--Kutta schemes as
:class:`~cubie.integrators.algorithms.generic_dirk.DIRKTableau` instances.
Aliases in the registry integrate with :func:`get_algorithm_step` so callers can
select stiffly accurate SDIRK, ESDIRK, and Lobatto families without repeating
coefficients.

.. autodata:: DIRK_TABLEAU_REGISTRY
    :annotation: Dict[str, DIRKTableau]

The :class:`DIRKStep` factory defaults to ``"sdirk_2_2"``—Alexander's
second-order, L-stable SDIRK pair—providing embedded error estimates for
adaptive controllers.

Available aliases
-----------------

.. list-table:: Named diagonally implicit Runge--Kutta tableaus
   :header-rows: 1

   * - Key
     - Description
     - Reference
   * - ``"implicit_midpoint"``
     - Single-stage implicit midpoint rule with symplectic structure.
     - [SanzSerna1988]_
   * - ``"trapezoidal_dirk"``
     - Two-stage trapezoidal (Crank--Nicolson) ESDIRK scheme.
     - [CrankNicolson1947]_
   * - ``"lobatto_iiic_3"``
     - Three-stage Lobatto IIIC method with stiff accuracy.
     - [HairerLubichWanner2006]_
   * - ``"sdirk_2_2"``
     - Alexander's L-stable SDIRK pair with embedded error weights.
     - [Alexander1977]_
   * - ``"l_stable_dirk_3"``
     - Three-stage, third-order L-stable SDIRK scheme with stiff accuracy.
     - [MOOSELStableDirk3]_
   * - ``"l_stable_sdirk_4"``
     - Hairer--Wanner five-stage, fourth-order L-stable SDIRK tableau.
     - [HairerWanner1996]_

Tableau container
-----------------

.. autoclass:: cubie.integrators.algorithms.generic_dirk_tableaus.DIRKTableau
    :members:
    :show-inheritance:

References
----------

.. [SanzSerna1988] J. M. Sanz-Serna. "Runge--Kutta schemes for Hamiltonian systems."
   *BIT Numerical Mathematics* 28(4), 1988.
.. [CrankNicolson1947] J. Crank and P. Nicolson. "A practical method for numerical
   solution of partial differential equations of the heat-conduction type."
   *Math. Proc. Camb. Phil. Soc.* 43(1), 1947.
.. [HairerLubichWanner2006] E. Hairer, C. Lubich, and G. Wanner. *Geometric Numerical
   Integration* (2nd ed.). Springer, 2006.
.. [Alexander1977] R. Alexander. "Diagonally implicit Runge--Kutta methods for stiff
   ODEs." *SIAM J. Numer. Anal.* 14(6), 1977.
.. [MOOSELStableDirk3] Idaho National Laboratory. "LStableDirk3 time integrator."
   *MOOSE Framework Documentation*. https://mooseframework.inl.gov/source/
   timeintegrators/LStableDirk3.html.
.. [HairerWanner1996] E. Hairer and G. Wanner. *Solving Ordinary Differential
   Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.). Springer,
   1996.
