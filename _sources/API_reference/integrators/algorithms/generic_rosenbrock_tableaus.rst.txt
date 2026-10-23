Rosenbrock tableau registry
==========================

.. currentmodule:: cubie.integrators.algorithms

``ROSENBROCK_TABLEAUS`` maps human-friendly identifiers to
:class:`~cubie.integrators.algorithms.generic_rosenbrockw_tableaus.RosenbrockTableau`
instances. The registry powers :func:`get_algorithm_step` aliases and allows
callers to select Rosenbrock-W schemes without manually specifying the
coefficients.

.. autodata:: ROSENBROCK_TABLEAUS
    :annotation: Dict[str, RosenbrockTableau]

The default :class:`GenericRosenbrockWStep` configuration uses the three-stage
third-order ROS3P tableau (``"ros3p"``).

Available aliases
-----------------

.. list-table:: Named Rosenbrock-W tableaus
   :header-rows: 1

   * - Key
     - Description
     - Reference
   * - ``"ros3p"``
     - Three-stage third-order ROS3P method.
     - [RangAngermann2005]_
   * - ``"rodas3p"``
     - Five-stage third-order RODAS3P Kaps--Rentrop scheme.
     - [SciMLRosenbrock]_
   * - ``"rosenbrock23"``, ``"ode23s"``, ``"rosenbrock23_sciml"``
     - Three-stage SciML Rosenbrock 2(3) method (MATLAB ``ode23s`` analogue).
     - [ShampineReichelt1997]_

Tableau container
-----------------

.. autoclass:: cubie.integrators.algorithms.generic_rosenbrockw_tableaus.RosenbrockTableau
    :members:
    :show-inheritance:

References
----------

.. [RangAngermann2005] J. Rang and L. Angermann. "New Rosenbrock--W methods of
   order 3 for partial differential-algebraic equations of index 1."
   *BIT Numerical Mathematics* 45(4), 2005.
.. [SciMLRosenbrock] SciML/OrdinaryDiffEq.jl. ``Rodas3PTableau`` in
   ``rosenbrock_tableaus.jl``.
   https://github.com/SciML/OrdinaryDiffEq.jl
.. [ShampineReichelt1997] L. F. Shampine and M. W. Reichelt. "The MATLAB ODE
   Suite." *SIAM J. Sci. Comput.* 18(1), 1997.
