FIRK tableau registry
=====================

.. currentmodule:: cubie.integrators.algorithms

``FIRK_TABLEAU_REGISTRY`` exposes fully implicit Runge--Kutta schemes as
:class:`~cubie.integrators.algorithms.generic_firk_tableaus.FIRKTableau`
instances. Aliases in the registry integrate with :func:`get_algorithm_step` so
callers can select Gauss--Legendre and Radau IIA families without repeating
coefficients.

.. autodata:: FIRK_TABLEAU_REGISTRY
    :annotation: Dict[str, FIRKTableau]

The :class:`FIRKStep` factory defaults to the two-stage fourth-order
Gauss--Legendre pair (``"firk_gauss_legendre_2"``).

Available aliases
-----------------

.. list-table:: Named fully implicit Runge--Kutta tableaus
   :header-rows: 1

   * - Key
     - Description
     - Reference
   * - ``"firk_gauss_legendre_2"``
     - Two-stage fourth-order Gauss--Legendre scheme with symplectic structure.
     - [HairerLubichWanner2006FIRK]_
   * - ``"radau_iia_5"`` and ``"radau"``
     - Three-stage fifth-order Radau IIA method with stiff accuracy.
     - [HairerWanner1996FIRK]_

Tableau container
-----------------

.. autoclass:: cubie.integrators.algorithms.generic_firk_tableaus.FIRKTableau
    :members:
    :show-inheritance:

References
----------

.. [HairerLubichWanner2006FIRK] E. Hairer, C. Lubich, and G. Wanner.
   *Geometric Numerical Integration* (2nd ed.). Springer, 2006.
.. [HairerWanner1996FIRK] E. Hairer and G. Wanner. *Solving Ordinary
   Differential Equations II: Stiff and Differential-Algebraic Problems*
   (2nd ed.). Springer, 1996.
