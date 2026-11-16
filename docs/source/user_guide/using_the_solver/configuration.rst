Solver Options
==============

Pass any option on this page as a keyword argument to
:func:`~cubie.solve_ivp`, to :class:`~cubie.Solver`, or to
:meth:`~cubie.Solver.solve` or :meth:`~cubie.Solver.update`. A solver
keeps the options you give it for later solves. If you leave an option
out, or pass ``None``, cubie uses its default, and if you misspell an
option, cubie raises ``KeyError``.

You can also collect options in a dict and pass it as
``step_control_settings``, ``algorithm_settings``, ``system_settings``,
``output_settings``, ``memory_settings`` or ``loop_settings``, and cubie
treats each key as a keyword argument.

Arguments of one solve
----------------------

:meth:`~cubie.Solver.solve` and :func:`~cubie.solve_ivp` also take
these arguments, which describe a single solve. The solver doesn't keep
them for later solves.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Effect
   * - ``initial_values`` (``y0`` in ``solve_ivp``)
     - ``None``
     - Initial values, as a dict or an array (see :doc:`inputs`).
       ``None`` starts every run from the system's default initial
       values.
   * - ``parameters``
     - ``None``
     - Parameter values, as a dict or an array (see :doc:`inputs`).
       ``None`` uses the system's default parameters for every run.
   * - ``grid_type``
     - ``"combinatorial"``
     - Pass ``"combinatorial"`` to solve every combination of the
       values in dict inputs, or ``"verbatim"`` to pair them
       run-for-run (see :doc:`inputs`).
   * - ``fix_constant_parameters``
     - ``True``
     - Compile parameters that have one value across the batch into the
       code (see :ref:`parameter-inputs`).
   * - ``duration``
     - ``1.0``
     - Length of the recorded integration (see :doc:`timing`).
   * - ``settling_time``
     - ``0.0``
     - Time to integrate before cubie starts recording (see
       :doc:`timing`).
   * - ``t0``
     - ``0.0``
     - Time at the start of the integration (see :doc:`timing`).
   * - ``nan_error_trajectories``
     - ``True``
     - Overwrite the output of every run with a nonzero status code
       with NaN (see :doc:`results`).
   * - ``on_device``
     - ``False``
     - ``solve`` only. Return the outputs as device arrays, with no
       copy to the host (see :doc:`results`).

Choosing the algorithm
----------------------

**algorithm** — the integration algorithm.
    Pass an algorithm name (see
    :doc:`/user_guide/algorithms_and_tuning/choosing_algorithms`) or a
    :class:`~cubie.integrators.algorithms.ButcherTableau`.
    ``solve_ivp`` calls this option ``method``.

    - Default: ``"tsit5"``

**tableau** — the Butcher tableau for a bare family name.
    If you select an algorithm with ``erk``, ``dirk``, ``firk`` or
    ``rosenbrock``, pass a tableau of that family here to use it in
    place of the family's default tableau.

    - Default: the family's default tableau

Controlling accuracy
--------------------

Adaptive algorithms choose their own step size to keep the estimated
error of each step below ``atol + rtol * |value|`` for every state.

**atol** — absolute error tolerance.
    This tolerance sets the error you accept when a state is near zero.
    Pass a scalar for every state, or an array with one tolerance per
    state.

    - Default: ``1e-6``

**rtol** — relative error tolerance.
    This tolerance scales with the size of the state: ``rtol=1e-6``
    asks for roughly six correct digits. Pass a scalar or an array with
    one tolerance per state.

    - Default: ``1e-6``

On DAE systems, cubie measures the error of the differential states
only.

Controlling the step size
-------------------------

:doc:`timing` explains how ``dt``, ``dt_min`` and ``dt_max`` work
together, and how cubie fills in the ones you leave out.

**dt** — the fixed step, or the first step of an adaptive algorithm.
    If you give an adaptive algorithm no ``dt``, cubie estimates a
    starting step for each run.

    - Default: ``1e-3`` for fixed-step algorithms; an estimate for each
      run for adaptive algorithms

**dt_min** / **dt_max** — bounds on the adaptive step size.
    If the error is still too large at ``dt_min``, cubie stops the run
    and sets ``STEP_TOO_SMALL`` in its status code.

    - Defaults: ``dt_min=1e-9``, ``dt_max=1.0``

**starting_step_state_fraction** / **starting_step_zero_norm** / **starting_step_target** / **starting_step_max_growth** — the constants cubie uses to estimate the starting step.
    Cubie first guesses the step that changes the state by
    ``starting_step_state_fraction`` of its norm, or the geometric mean
    of ``dt_min`` and ``dt_max`` when the state or its derivative has a
    norm of ``starting_step_zero_norm`` or less. Cubie returns at most
    ``starting_step_max_growth`` first guesses. Lower
    ``starting_step_target`` for a smaller estimate.

    - Defaults: ``starting_step_state_fraction=0.01``,
      ``starting_step_zero_norm=1e-5``, ``starting_step_target=0.01``,
      ``starting_step_max_growth=100``

Tuning the step controller
--------------------------

**step_controller** — the step-size controller.
    Pass ``"fixed"``, ``"i"``, ``"pi"``, ``"pid"`` or ``"gustafsson"``
    (see :ref:`choosing-a-controller`).

    - Default: ``"fixed"`` for algorithms without an error estimate;
      otherwise ``"i"`` for ERK, ``"pi"`` for DIRK and
      ``"gustafsson"`` for FIRK and Rosenbrock-W. Some tableaus choose
      their own.

Each algorithm family also sets its own defaults for the options below,
so treat them as expert tuning.

**min_step_shrink** / **max_step_growth** — bounds on how much one step can change the step size.

    - Defaults: ``min_step_shrink=0.3``, ``max_step_growth=2.0``

**safety** — a factor on every proposed step.
    Cubie multiplies each proposed step by this factor. Values below
    1.0 aim slightly below the step the error estimate suggests, so you
    get fewer rejected steps at a small cost in speed.

    - Default: ``0.9``

**deadband_min** / **deadband_max** — a band of step ratios that cubie ignores.
    If the proposed change in step size falls inside this band, cubie
    keeps the current step. Set both to 1.0 to turn the band off.

    - Defaults: ``deadband_min=1.0``, ``deadband_max=1.0`` (off)

**integral_gain** / **proportional_gain** / **derivative_gain** — the gains of the ``i``, ``pi`` and ``pid`` controllers.
    :ref:`choosing-a-controller` explains what each gain does. Cubie
    divides each gain by ``order + 1``. If you pass a gain without a
    ``step_controller``, cubie selects the fastest of ``i``, ``pi`` and
    ``pid`` that uses it.

    - Defaults: ``integral_gain=1.0`` for ``i`` and ``0.3`` for ``pi``
      and ``pid``; ``proportional_gain=0.4``; ``derivative_gain=0.0``

**filter_coefficients** — the exponents of the step law.
    Pass a ``(beta1, beta2, beta3)`` triple for the step law
    ``dt * error^-beta1 * error_prev^-beta2 * error_prev2^-beta3``, or
    the name of a published controller: ``"basic"``, ``"PI42"``,
    ``"PI33"``, ``"PI34"``, ``"H211PI"`` or ``"H312PID"``. Cubie
    divides the exponents by ``order + 1``. You can't combine
    ``filter_coefficients`` with the gains.

    - Default: the gains above, which equal ``"PI34"`` for ``pi`` and
      ``pid``

**newton_target_iters** — the Newton iteration count the ``gustafsson`` controller aims for.
    The ``gustafsson`` controller shrinks the step when the implicit
    solver works harder than this. It doesn't limit the number of
    Newton iterations.

    - Default: ``20``

Tuning the implicit solvers
---------------------------

Implicit algorithms (backward Euler, Crank--Nicolson, DIRK, FIRK) solve
a nonlinear system at every step with Newton's method, and solve a
linear system for each Newton update. Rosenbrock-W algorithms solve
only the linear systems, so the Newton options don't apply to them.

Newton options:

**newton_atol** / **newton_rtol** — Newton convergence tolerances.
    Cubie accepts a Newton solve when the estimated error of the update
    drops below one percent of these tolerances.

    - Default: the step controller's ``atol`` and ``rtol`` divided by
      10

**newton_max_iters** — the Newton iteration limit.
    If a step doesn't converge within this many iterations, cubie
    rejects it and, under an adaptive controller, retries it with a
    smaller step.

    - Default: ``8``

**inexact_newton** — freeze the Newton matrix at the start of each step.
    Cubie evaluates the Jacobian once per step instead of once per
    Newton iteration. Rosenbrock-W algorithms ignore this option.

    - Default: ``True`` for DIRK and FIRK, except the tableaus that the
      :doc:`/user_guide/algorithms_and_tuning/choosing_algorithms`
      tables list as using exact Newton; ``False`` otherwise

**prefactored** — keep the LU factors from the start of each step.
    With ``inexact_newton`` and ``linear_correction_type="lu"``, cubie
    factorises the frozen matrix once per step for each distinct value
    on the tableau's diagonal, and reuses the factors in every solve.
    With ``False``, cubie factorises the frozen matrix in each solve.

    - Default: ``True``

**attempt_dense_prediction** — warm-start each Newton solve.
    DIRK and FIRK algorithms start each step's Newton solves from a
    prediction made from the previous step's stages. Cubie ignores
    this option for tableaus that can't support the prediction.

    - Default: ``True``, or ``False`` for adaptive DIRK tableaus

Linear solver options:

**linear_correction_type** — the linear solver.
    ``"lu"`` factorises the matrix and solves it directly. It is
    faster for stiff problems and slower for easy ones.

    ``"minimal_residual"`` and ``"steepest_descent"`` are iterative
    solvers. ``"minimal_residual"`` is usually the fastest, but the
    least robust.

    ``"bicgstab"`` is an iterative solver that applies the Jacobian
    twice in each iteration, where the other iterative solvers apply it
    once.

    - Default: ``"lu"`` for DIRK, FIRK, Rosenbrock-W and every DAE;
      ``"minimal_residual"`` otherwise

**krylov_atol** / **krylov_rtol** — the tolerances that weight the iterative linear solver's stopping test.

    - Default: the step controller's ``atol`` and ``rtol``

**krylov_max_iters** — the linear iteration limit for each Newton update.

    - Default: ``ceil(1.5 * solver width)``

**krylov_residual_reduction** / **krylov_residual_floor** — the stopping test of the iterative linear solver.
    Cubie stops each linear solve once its weighted residual falls
    below ``floor + reduction * ||b||``, where ``||b||`` is the weighted
    right-hand side at the start of the solve. The relative term stops
    cubie solving the early Newton updates more accurately than it
    needs to.

    - Defaults: ``krylov_residual_reduction`` is the smallest ``rtol``
      of an adaptive controller, or machine epsilon for a fixed one;
      ``krylov_residual_floor`` is the square root of machine epsilon

**preconditioner_type** — the preconditioner.
    Pass ``"jacobi"``, ``"neumann"`` or ``"none"``. With no
    preconditioner, cubie is fastest on easy systems. Between
    ``"jacobi"`` and ``"neumann"``, the faster one depends on the
    system and algorithm. Cubie rejects ``"neumann"`` on DAE systems.

    - Default: ``"jacobi"``

**preconditioner_order** — the number of terms in the preconditioner's series.
    Each term costs one Jacobian product each time cubie applies the
    preconditioner, and saves linear iterations only while the series
    converges, so use a low order on a system with large off-diagonal
    Jacobian entries. ``"neumann"`` needs at least one term; on
    ``"jacobi"``, order zero is a plain diagonal solve.

    - Default: ``2`` for ``"neumann"``, ``0`` for ``"jacobi"``

Error smoothing options:

**use_smoothed_error** — smooth the error estimate.
    If the tableau supports it, cubie adds one linear solve per step
    that filters stiff components out of the error estimate. The
    larger steps that you get usually outweigh the cost of the extra
    solve. Cubie warns and leaves smoothing off for a tableau that
    can't smooth its estimate.

    - Default: ``False``, or ``True`` for Radau IIA tableaus

**error_atol** / **error_rtol** / **error_max_iters** / **error_residual_reduction** / **error_residual_floor** — the settings of the smoothing solve.
    These options set the same quantities as their ``krylov_``
    counterparts, for the linear solve that ``use_smoothed_error``
    adds.

    - Default: the value of the matching ``krylov_`` option

DAE options:

**dae_initialisation** — make a DAE's initial values consistent.
    Cubie solves for consistent algebraic states at the start of every
    run, before the first step and the ``t=0`` output, starting from
    the initial values you gave. ``"brown"`` holds the differential
    states and solves for the algebraic ones. ``"shampine"`` takes one
    backward Euler step of the starting step size, which can move every
    state by up to ``O(dt)``. ``"none"`` skips the solve. If the solve
    fails, cubie ends the run at the ``t=0`` save and sets
    ``DAE_INITIALISATION_FAILED`` in its status code. Cubie ignores
    this option on systems that aren't DAEs.

    - Default: ``"brown"``

Choosing what gets saved
------------------------

:doc:`results` explains each of these options.

**output_types** — what to record.
    Pass any combination of ``"state"``, ``"observables"``, ``"time"``,
    ``"iteration_counters"`` and the summary metric names.

    - Default: ``["state"]``

**save_variables** / **summarise_variables** — the variables to save and summarise, by name.
    You can also select variables by index with
    ``saved_state_indices``, ``saved_observable_indices``,
    ``summarised_state_indices`` and ``summarised_observable_indices``.

    - Default: every state and observable; ``summarise_variables``
      follows ``save_variables``

**save_every** / **summarise_every** / **sample_summaries_every** — how often to save, summarise and sample.
    :doc:`timing` explains each interval.

    - Defaults: without ``save_every``, cubie saves the first and last
      points; without ``summarise_every``, it summarises once at the
      end. You must give ``sample_summaries_every`` if you ask for any
      summary.

Drivers
-------

:doc:`/user_guide/defining_systems/drivers` explains each of these
options.

**drivers** — the forcing data, as a :class:`~cubie.array_interpolator.DriverSamples`.

**order** — the polynomial order of the spline cubie fits to the samples.

    - Default: ``3``

**wrap** — repeat the samples outside their time range.

    - Default: ``True``

**boundary_condition** — the end condition of the spline fit.

    - Default: ``"periodic"`` with ``wrap=True``, ``"clamped"`` with
      ``wrap=False``

Precision and compilation
-------------------------

**precision** — the floating-point type of the solve.
    Pass ``numpy.float32`` or ``numpy.float64``.
    :doc:`/user_guide/algorithms_and_tuning/speed` compares them.

    - Default: the precision you built the system with, ``float32``
      unless you set it

**operation_ordering** — the order of the operations in your system's generated code.
    Pass ``"liveness_auto"``, ``"kahn"``, ``"greedy"`` or ``"dfs"``.

    - Default: the ``CUBIE_OPERATION_ORDERING`` environment variable,
      or ``"liveness_auto"`` if you haven't set it

**lineinfo** — compile source-line data for profilers such as Nsight Compute.
    Changing it rebuilds every kernel.

    - Default: the ``CUBIE_LINEINFO`` environment variable, or off if
      you haven't set it

**nsz** / **contract** / **arcp** / **afn** / **reassoc** / **ftz** — the compiler's fast-math flags.
    ``nsz`` ignores the sign of zero, ``contract`` fuses multiplies and
    adds, ``arcp`` approximates division by a reciprocal, ``afn`` uses
    approximate ``log``, ``exp`` and ``pow``, ``reassoc`` lets the
    compiler regroup chains of additions and multiplications, and
    ``ftz`` flushes subnormal numbers to zero.

    - Defaults: ``reassoc=False``; every other flag ``True``

**lto** — optimise across device functions when cubie links the kernel.

    - Default: ``True``

**kernel_name** — the name of the compiled kernel in profiler output.

    - Default: the algorithm name followed by the system name; cubie
      appends ``_ltoon`` or ``_ltooff`` to either

Performance
-----------

:doc:`/user_guide/algorithms_and_tuning/speed` explains when these
options matter.

**auto_performance** — choose buffer locations, loop unrolling and the launch shape for your system and GPU.
    Cubie never overrides a ``*_location``, ``unroll_*`` or
    ``blocksize`` option that you set.

    - Default: ``True``

**blocksize** — the number of threads in each CUDA block.
    Pass it to ``solve`` to set it for one launch.

    - Default: chosen by cubie

**max_registers** — the register limit for each thread.
    A lower limit lets more threads run at once, but spills more values
    to memory.

    - Default: no limit

**unroll_stage** / **unroll_step_element** / **unroll_accumulator** / **unroll_solver_element** / **unroll_norms** / **unroll_other_small** / **unroll_newton_exits** / **unroll_krylov_exits** — loop unrolling for each group of loops.
    Pass ``True`` to unroll the loops fully, ``False`` to leave them to
    the compiler, ``UnrollChoice.ROLLED`` (from ``cubie.CUDAFactory``)
    to keep them rolled, or ``(True, k)`` to unroll them ``k`` times.
    The groups are the loops over tableau stages, over the elements of
    the state in each step, over the stage accumulator, over the
    elements in the solvers, over norms, over small fills, saves and
    counters, over Newton iterations, and over linear-solver
    iterations.

    - Default: chosen by cubie under ``auto_performance``; otherwise
      ``True``, and ``UnrollChoice.ROLLED`` for
      ``unroll_krylov_exits``

**\*_location** options (``state_location``, ``timestep_memory_location`` and about forty more) — where each working buffer lives.
    Pass ``"local"`` for per-thread memory or ``"shared"`` for
    per-block memory. Buffer locations don't change the results.

    - Default: chosen by cubie under ``auto_performance``; otherwise
      ``"local"``

Memory and caching
------------------

:doc:`/user_guide/advanced/memory` and
:doc:`/user_guide/advanced/caching` explain these options.

**mem_proportion** — the share of VRAM this solver may use.
    Pass a number between 0.0 and 1.0. With ``None``, the solver joins
    the automatic pool, which shares the VRAM left after every solver
    with a set proportion equally between its members.

    - Default: ``None``

**stream_group** — the name of the stream group the solver runs on.
    Solvers in the same group share one CUDA stream.

    - Default: ``"solver"``

**memory_manager** — the :class:`~cubie.memory.mem_manager.MemoryManager` that allocates the solver's memory.

    - Default: ``cubie.default_memmgr``

**cache_enabled** — read compiled kernels from the disk cache and save new ones to it.

    - Default: ``True``

**cache_dir** — the directory of the compiled-kernel cache.

    - Default: the ``CUBIE_KERNEL_CACHE_DIR`` environment variable, or
      the shared cache root if you haven't set it

**max_cache_entries** — the number of compiled kernels cubie keeps in each cache directory before it removes the least recently used one.
    ``0`` keeps every kernel.

    - Default: the ``CUBIE_MAX_CACHE_ENTRIES`` environment variable, or
      ``0`` if you haven't set it

Logging
-------

**time_logging_level** — how much timing information cubie prints.
    Pass ``"default"``, ``"verbose"`` or ``"debug"`` to print more
    detail, ``"silent"`` to record GPU timings without printing them,
    or ``None`` to turn timing off.

    - Default: ``None``
