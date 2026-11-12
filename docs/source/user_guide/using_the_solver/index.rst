Using the solver
================

Once you have defined an ODE system (see :doc:`/user_guide/defining_systems/systems`), CuBIE provides
two ways to solve it: the convenience function
:func:`~cubie.batchsolving.solver.solve_ivp` and the reusable
:class:`~cubie.batchsolving.solver.Solver` class.

Cubie can build the batches you want to solve over (see batching), and provides results in
a number of formats (see results). You'll usually want to modify the timing parameters to
suit your integration (see timing), and deeper configuration is achievable (see configuration).


.. toctree::
   :maxdepth: 1

   solving
   batching
   results
   timing
   configuration
