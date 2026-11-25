ERK tableau registry
====================

.. currentmodule:: cubie.integrators.algorithms

``ERK_TABLEAU_REGISTRY`` maps human-friendly identifiers to
:class:`~cubie.integrators.algorithms.generic_erk.ERKTableau` instances. The
registry powers :func:`get_algorithm_step` aliases and allows callers to select
well-known explicit Runge--Kutta schemes without manually specifying the
coefficients.

.. autodata:: ERK_TABLEAU_REGISTRY
    :annotation: Dict[str, ERKTableau]

The default :class:`ERKStep` configuration uses
:data:`cubie.integrators.algorithms.generic_erk.DEFAULT_ERK_TABLEAU`, the
Tsitouras 5(4) pair.

Available aliases
-----------------

.. list-table:: Named explicit Runge--Kutta tableaus
   :header-rows: 1

   * - Key
     - Description
     - Reference
   * - ``"heun-21"``
     - Heun's improved Euler method (order 2).
     - [Heun1900]_
   * - ``"ralston-33"``
     - Ralston's third-order method with minimized error constants.
     - [Ralston1962]_
   * - ``"bogacki-shampine-32"``, ``"rk23"`` and ``"ode23"``
     - Bogacki--Shampine embedded 3(2) pair with FSAL property.
     - [BogackiShampine1993]_
   * - ``"dormand-prince-54"``, ``"dopri54"``, ``"rk45"`` and ``"ode45"``
     - Dormand--Prince embedded 5(4) pair with FSAL property.
     - [DormandPrince1980]_
   * - ``"classical-rk4"`` and ``"rk4"``
     - Classical fourth-order Runge--Kutta scheme.
     - [Kutta1901]_
   * - ``"cash-karp-54"``
     - Cash--Karp embedded 5(4) pair with adaptive control weights.
     - [CashKarp1990]_
   * - ``"fehlberg-45"``
     - Runge--Kutta--Fehlberg embedded 5(4) pair.
     - [Fehlberg1969]_
   * - ``"tsit5"``
     - Tsitouras embedded 5(4) pair with FSAL property; the default.
     - [Tsitouras2011]_
   * - ``"vern7"``
     - Verner's embedded 7(6) pair.
     - [Verner2010]_
   * - ``"dormand-prince-853"`` and ``"dop853"``
     - Dormand--Prince eighth-order method with 5(3) error estimates.
     - [HairerNorsettWanner1993]_

Tableau container
-----------------

.. autoclass:: cubie.integrators.algorithms.generic_erk_tableaus.ERKTableau
    :members:
    :show-inheritance:

References
----------

.. [Heun1900] K. Heun. *Neue Methoden zur approximativen Integration der
   Differentialgleichungen einer unabhängigen Veränderlichen.* Z. Math. Phys.
   45, 1900.
.. [Ralston1962] A. Ralston. "Runge-Kutta methods with minimum error bounds." *Math.
   Comp.* 16 (1962).
.. [BogackiShampine1993] P. Bogacki and L. F. Shampine. "An efficient Runge-Kutta (4,5)
   pair." *J. Comput. Appl. Math.* 46(1), 1993.
.. [DormandPrince1980] J. R. Dormand and P. J. Prince. "A family of embedded
   Runge-Kutta formulae." *J. Comput. Appl. Math.* 6(1), 1980.
.. [Kutta1901] W. Kutta. "Beitrag zur näherungsweisen Integration totaler
   Differentialgleichungen." *Zeitschr. Math. Phys.* 46, 1901.
.. [CashKarp1990] J. R. Cash and A. H. Karp. "A variable order Runge-Kutta method for
   initial value problems with rapidly varying right-hand sides." *ACM Trans.
   Math. Softw.* 16(3), 1990.
.. [Fehlberg1969] E. Fehlberg. *Low-order classical Runge-Kutta formulas with
   stepsize control and their application to some heat transfer problems.* NASA
   Technical Report 315, 1969.
.. [Tsitouras2011] Ch. Tsitouras. "Runge--Kutta pairs of order 5(4) satisfying
   only the first column simplifying assumption." *Comput. Math. Appl.*
   62(2), 2011.
.. [Verner2010] J. H. Verner. "Numerically optimal Runge--Kutta pairs with
   interpolants." *Numer. Algorithms* 53, 2010.
.. [HairerNorsettWanner1993] E. Hairer, S. P. Nørsett and G. Wanner. *Solving
   Ordinary Differential Equations I: Nonstiff Problems.* 2nd ed., Springer,
   1993.
