Batch solving
=============

``cubie.batchsolving``
----------------------

.. currentmodule:: cubie.batchsolving

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   solver
   solve_ivp
   solve_result
   device_solve_result
   solve_spec
   calibration
   optimize
   comparison
   batch_input_handler
   solver_settings
   resolve_defaults
   batch_solver_config
   batch_solver_kernel
   system_interface
   writeback_watcher

To solve batches of your system, build a
:class:`~cubie.batchsolving.solver.Solver` for it, then call
:meth:`~cubie.batchsolving.solver.Solver.solve` for each batch.
:func:`~cubie.batchsolving.solver.solve_ivp` builds a solver, solves
one batch and closes the solver in a single call. Both return a
:class:`~cubie.batchsolving.solveresult.SolveResult`.

.. toctree::
   :maxdepth: 2
   :caption: Subpackages
   :titlesonly:

   arrays

Core API
--------

* :doc:`Solver <solver>` – builds the GPU code for your system and
  solves batches with it.
* :doc:`solve_ivp <solve_ivp>` – solves one batch in a single call.
* :doc:`SolveResult <solve_result>` – the outputs, legends and status
  codes of a solve.
* :doc:`DeviceSolveResult <device_solve_result>` – the outputs of a
  solve left on the GPU, from ``solve(on_device=True)``.
* :doc:`SolveSpec <solve_spec>` – the step, output and tolerance
  settings of a solve.
* :doc:`Calibration <calibration>` – races solver configurations on a
  representative batch and picks the fastest (:meth:`~cubie.batchsolving.solver.Solver.calibrate`).
* :doc:`Optimisation <optimize>` – times buffer placement, loop unrolling
  and launch shape for one configuration and keeps the fastest
  (:meth:`~cubie.batchsolving.solver.Solver.optimize`).
* :doc:`Candidate comparison <comparison>` – times candidate
  configurations of one solver on one batch, for calibration and
  optimisation.

Supporting infrastructure
-------------------------

* :doc:`BatchInputHandler <batch_input_handler>` – turns your initial
  values and parameters into arrays with one column per run.
* :doc:`Solver settings <solver_settings>` – the settings you gave a
  solver, and the full set it solves with, including the ones it
  worked out from yours.
* :doc:`Default resolution <resolve_defaults>` – works out the settings
  you leave out.
* :doc:`BatchSolverConfig <batch_solver_config>` – the compile settings
  of the batch kernel.
* :doc:`BatchSolverKernel <batch_solver_kernel>` – compiles and launches
  the batch kernel.
* :doc:`SystemInterface <system_interface>` – converts between the names
  and indices of a system's states, parameters and observables.
* :doc:`Writeback watcher <writeback_watcher>` – copies the outputs of a
  chunked solve into the result arrays.

Array types
-----------

.. autodata:: ArrayTypes

Dependencies
------------

- Batch kernel compilation extends :class:`cubie.CUDAFactory` and relies on
  :mod:`cubie.integrators.algorithms` and
  :mod:`cubie.integrators.step_control` for loop callables, and
  :mod:`cubie.array_interpolator` for driver interpolation.
- Array managers size buffers with
  :mod:`cubie.outputhandling.output_sizes` and allocate memory through
  :mod:`cubie.memory` managers.
- Solver results surface summary metrics registered via
  :mod:`cubie.outputhandling.summarymetrics`.
