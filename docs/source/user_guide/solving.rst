Solving ODEs
============

Once you have defined an ODE system (see :doc:`systems`), CuBIE provides
two ways to solve it: the convenience function
:func:`~cubie.batchsolving.solver.solve_ivp` and the reusable
:class:`~cubie.batchsolving.solver.Solver` class.

Quick Start with ``solve_ivp``
------------------------------

The fastest way to get results is :func:`~cubie.solve_ivp`.  Here we
solve the Lotka--Volterra system for a single initial condition:

.. code-block:: python

   import cubie as qb
   import numpy as np

   def lotka_volterra(t, y, p):
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -p.c * y.y + p.d * y.x * y.y
       return [dx, dy]

   LV = qb.create_ODE_system(
       lotka_volterra,
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       states={"x": 0.5, "y": 0.3},
       name="LotkaVolterra",
   )

   result = qb.solve_ivp(
       LV,
       y0={"x": np.array([0.5]), "y": np.array([0.3])},
       parameters={"b": np.array([0.02]), "d": np.array([0.01])},
       method="dormand-prince-54",
       duration=100.0,
   )

``result`` is a :class:`~cubie.batchsolving.solveresult.SolveResult` that
holds the time-domain output, summaries, and metadata.  See
:doc:`results` for how to inspect it.

.. note::

   CuBIE is a *batch* solver---it is designed to solve many IVPs at once.
   A single-system call works, but the real performance gains come from
   passing arrays of initial values or parameters.  See :doc:`batching`.

Changing Initial States
-----------------------

Pass arrays for any states you want to vary across the batch:

.. code-block:: python

   result = qb.solve_ivp(
       LV,
       y0={
           "x": np.linspace(0.1, 2.0, 500),
           "y": np.array([0.3]),  # same for all runs
       },
       parameters={"b": np.array([0.02]), "d": np.array([0.01])},
       method="dormand-prince-54",
       duration=100.0,
   )

States provided as length-1 arrays are broadcast to all runs.

The ``Solver`` Class
--------------------

If you plan to solve the same system multiple times (e.g. exploring
different parameter sets interactively), create a
:class:`~cubie.Solver` once and call
:meth:`~cubie.batchsolving.solver.Solver.solve` repeatedly.  This avoids
recompiling the CUDA kernel on every call:

.. code-block:: python

   solver = qb.Solver(LV, algorithm="dormand-prince-54")

   result_a = solver.solve(
       initial_values={"x": np.array([0.5]), "y": np.array([0.3])},
       parameters={"b": np.linspace(0.01, 0.05, 1000),
                   "d": np.full(1000, 0.01)},
       duration=100.0,
   )

   result_b = solver.solve(
       initial_values={"x": np.array([1.0]), "y": np.array([0.5])},
       parameters={"b": np.linspace(0.01, 0.05, 1000),
                   "d": np.full(1000, 0.01)},
       duration=200.0,
   )

The first call compiles the kernel; subsequent calls reuse it.

Two further ``Solver`` methods are useful in interactive loops:
:meth:`~cubie.batchsolving.solver.Solver.update` reconfigures a live
solver (tolerances, algorithm, output settings) without rebuilding it
from scratch, and
:meth:`~cubie.batchsolving.solver.Solver.prepare_grid_arrays` pre-builds the
input grid once so repeated ``solve`` calls with the same batch layout
skip grid construction.

.. _parameter-inputs:

Dicts and arrays
----------------

It's probably simplest to provide a dict of parameter and initial
values, like this:

.. code-block:: python

    solver.solve(
        initial_values={"y0": np.linspace(1, 100, 1000), "y1": 0.3},
        parameters={"x0": 0.4, "x1": np.logspace(1, 3, 1000)},
        duration=1.0,
    )

Every parameter you name gets a row of the parameters array, so ``x0``
above holds 0.4 in every run. If you set default values for the states
or parameters, you can leave them out of the dicts and they will use
their default values. You can leave either dict out entirely, too. If
you didn't set defaults when you built the system, you can set them on
the system with ``system.set_default_parameters({"x0": 0.4})`` for
parameters and ``system.set_initial_values({"y1": 0.3})`` for states.
If you include a constant value for a parameter or initial value that
has a different default, the default doesn't change: Cubie presumes
that arguments to ``solve`` are intended for that run only. Pass
``fix_constant_parameters=True`` to compile every constant value in
the dict into the code instead of giving it a row; see below.
``solve_ivp``, ``Solver.compile``, ``Solver.calibrate`` and
``Solver.optimize`` treat their inputs the same way.

If you're looking for something faster (by ~tens of ms per solve) or are
generating values programmatically, you can pass arrays instead.
Initial values arrays have one row per state in your system. The
parameters array can either have one row for every parameter in your
system (a full-height parameter array), or one row per parameter that
will vary across a batch only. In
both cases, rows should be ordered by parameter names.
``solver.parameter_legend`` lists the parameter name of each row. To pass
just the varying parameters, you will need to tell the system which
parameters to expect. To do this, call ``solver.set_swept_parameters``
with a set of the parameter names you want to vary, or pass the same set
to the ``swept_parameters`` argument when building the system.

.. code-block:: python

    system = create_ODE_system(
        dxdt, parameters={"x0": 0.4, "x1": 1.0}, swept_parameters=["x1"]
    )
    solver = Solver(system)
    solver.solve(inits, np.logspace(1, 3, 1000)[np.newaxis, :])

    # or, on an existing solver
    solver.set_swept_parameters(["x1"])

``solver.prepare_grid_arrays`` can generate arrays from a dict of symbol
names, to avoid doing it on every solve. It sets the swept parameters to
the parameters in your dict and gives each of them a row, so if you give
a parameter a single value, it occupies a full row with that value in
every run. To compile a value into the code instead, set it as the
parameter's default and leave it out of the dict:

.. code-block:: python

    system.set_default_parameters({"x0": 0.5})
    inits, params = solver.prepare_grid_arrays(
        {"y0": np.linspace(1, 100, 1000)},
        {"x1": np.logspace(1, 3, 1000)},
    )
    result = solver.solve(inits, params)

Arrays of swept parameters only are intended as a convenience: some
models have hundreds of parameters, and if you're only sweeping two of
them you probably don't want to throw hundreds-high arrays around. If it
ends up being easier to pass full-height parameter arrays because of how
you generate the values, you can set ``fix_constant_parameters=True`` in
your call to ``solve``, and Cubie will scan your array for values that
don't change over the whole batch and compile them into your code. This
keeps the performance benefits when running the solve, at the expense of
a few milliseconds scanning the array. The same flag compiles in the
constant entries of a dict. ``fix_constant_parameters`` works with
dicts and full-height parameter arrays, not with arrays of swept
parameters only.

If you want to go *really* fast and you are comfortable with the CUDA
environment, you can pass arrays on the GPU (device arrays) directly to
cut out the few milliseconds of transfer time as well.

Arrays are laid out ``(n_variables, n_runs)`` and must match the system
precision. Device arrays (CuPy or Numba) are wired directly into the
kernel; pair them with ``on_device=True`` to keep an entire solve →
post-process pipeline on the GPU (see :doc:`results`).

The ``duration`` Parameter
--------------------------

``duration`` sets the total integration time.  You can also specify
``t0`` (default 0) and ``settling_time``.  When ``settling_time`` is
non-zero, the solver integrates for that period *before* it starts
recording output, useful for discarding transients.

Next Steps
----------

- :doc:`batching` --- parameter sweeps and grid types.
- :doc:`results` --- inspecting and post-processing solver output.
- :doc:`choosing_algorithms` --- which algorithm to pick.
