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

:meth:`~cubie.Solver.solve` and :func:`~cubie.solve_ivp` also take the
arguments below. Give them to each solve that needs them, because the
solver doesn't keep them for later solves.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Effect
   * - ``initial_values`` (``y0`` in ``solve_ivp``)
     - ``None``
     - Initial values, as a dict or an array (see :doc:`inputs`).
       Pass ``None`` to start every run from the system's default
       initial values.
   * - ``parameters``
     - ``None``
     - Parameter values, as a dict or an array (see :doc:`inputs`).
       Pass ``None`` to use the system's default parameters for every
       run.
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

With an adaptive algorithm, cubie chooses each step size to keep the
estimated error of the step below ``atol + rtol * |value|`` for every
state.

**atol** — absolute error tolerance.
    Use it to set the error you accept when a state is near zero.
    Pass a scalar for every state, or an array with one tolerance per
    state.

    - Default: ``1e-6``

**rtol** — relative error tolerance.
    Cubie scales it by the size of the state, so ``rtol=1e-6`` gives
    you roughly six correct digits. Pass a scalar or an array with
    one tolerance per state.

    - Default: ``1e-6``

On DAE systems, cubie measures the error of the differential states
only.

Controlling the step size
-------------------------

See :doc:`timing` for how ``dt``, ``dt_min`` and ``dt_max`` work
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
      ``"gustafsson"`` for FIRK and Rosenbrock-W. Cubie uses a
      different default for some tableaus.

Cubie sets its own defaults for the options below for each algorithm
family, so treat them as expert tuning.

**min_step_shrink** / **max_step_growth** — bounds on how much cubie changes the step size from one step to the next.

    - Defaults: ``min_step_shrink=0.3``, ``max_step_growth=2.0``

**safety** — a factor on every proposed step.
    Cubie multiplies each proposed step by this factor. With a value
    below 1.0, cubie takes slightly smaller steps than its error
    estimate allows, so you get fewer rejected steps at a small cost in
    speed.

    - Default: ``0.9``

**deadband_min** / **deadband_max** — a band of step ratios that cubie ignores.
    If cubie would change the step size by a ratio inside this band, it
    keeps the current step. Set both to 1.0 to turn the band off.

    - Defaults: ``deadband_min=1.0``, ``deadband_max=1.0`` (off)

**integral_gain** / **proportional_gain** / **derivative_gain** — the gains of the ``i``, ``pi`` and ``pid`` controllers.
    See :ref:`choosing-a-controller` for what each gain does. Cubie
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
    With the ``gustafsson`` controller, cubie shrinks the step when the
    Newton solve takes more iterations than this. Cubie doesn't stop
    the Newton solve at this count.

    - Default: ``20``

Tuning the implicit solvers
---------------------------

For implicit algorithms (backward Euler, Crank--Nicolson, DIRK, FIRK),
cubie solves a nonlinear system at every step with Newton's method, and
a linear system for each Newton update. For Rosenbrock-W algorithms,
cubie solves only the linear systems, so the Newton options don't apply
to them.

Newton options:

**newton_atol** / **newton_rtol** — Newton convergence tolerances.
    Cubie accepts a Newton solve when the estimated error of the update
    drops below one percent of these tolerances.

    - Default: the step controller's ``atol`` and ``rtol`` divided by
      10

**newton_max_iters** — the Newton iteration limit.
    If the Newton solve doesn't converge within this many iterations,
    cubie rejects the step and, under an adaptive controller, retries it
    with a smaller step.

    - Default: ``8``

**inexact_newton** — freeze the Newton matrix at the start of each step.
    Cubie evaluates the Jacobian once per step instead of once per
    Newton iteration. Cubie ignores this option for Rosenbrock-W
    algorithms.

    - Default: ``True`` for DIRK and FIRK, except the tableaus marked
      as defaulting to exact Newton in
      :doc:`/user_guide/algorithms_and_tuning/choosing_algorithms`;
      ``False`` otherwise

**prefactored** — keep the LU factors from the start of each step.
    With ``inexact_newton`` and ``linear_correction_type="lu"``, cubie
    factorises the frozen matrix once per step for each distinct value
    on the tableau's diagonal, and reuses the factors in every solve.
    With ``False``, cubie factorises the frozen matrix in each solve.
    Cubie ignores this option for Rosenbrock-W algorithms. With
    ``"lu"``, it factorises their matrix once per step and reuses the
    factors in every stage.

    - Default: ``True``

**attempt_dense_prediction** — warm-start each Newton solve.
    For DIRK and FIRK algorithms, cubie starts each step's Newton
    solves from a prediction it makes from the previous step's stages.
    Cubie ignores this option for tableaus that can't support the
    prediction.

    - Default: ``True``, or ``False`` for adaptive DIRK tableaus

Linear solver options:

**linear_correction_type** — the linear solver.
    Pass ``"lu"`` to factorise the matrix and solve it directly, which
    is faster for stiff problems and slower for easy ones.

    ``"minimal_residual"``, ``"steepest_descent"`` and ``"bicgstab"``
    are iterative solvers. ``"minimal_residual"`` is usually the
    fastest, but the least robust. With ``"bicgstab"``, cubie applies
    the Jacobian twice in each iteration, where it applies it once with
    the other iterative solvers.

    - Default: ``"lu"`` for DIRK, FIRK, Rosenbrock-W and every DAE;
      ``"minimal_residual"`` otherwise

**krylov_atol** / **krylov_rtol** — the tolerances cubie uses to weight the iterative linear solver's stopping test.

    - Default: the step controller's ``atol`` and ``rtol``

**krylov_max_iters** — the linear iteration limit for each Newton update.

    - Default: ``ceil(1.5 * solver width)``

**krylov_residual_reduction** / **krylov_residual_floor** — the stopping test of the iterative linear solver.
    Cubie stops each linear solve once its weighted residual falls
    below ``floor + reduction * ||b||``, where ``||b||`` is the weighted
    right-hand side at the start of the solve. With the relative term,
    cubie doesn't solve the early Newton updates more accurately than it
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
    Cubie computes one Jacobian product per term each time it applies
    the preconditioner. You save linear iterations with extra terms
    only while the series converges, so use a low order on a system
    with large off-diagonal Jacobian entries. Give ``"neumann"`` at
    least one term. With ``"jacobi"`` at order zero, cubie does a plain
    diagonal solve.

    - Default: ``2`` for ``"neumann"``, ``0`` for ``"jacobi"``

Error smoothing options:

**use_smoothed_error** — smooth the error estimate.
    If the tableau supports it, cubie adds one linear solve per step to
    filter stiff components out of the error estimate. You usually gain
    more from the larger steps than you lose to the extra solve. Cubie warns and leaves smoothing off for a tableau that
    can't smooth its estimate.

    - Default: ``False``, or ``True`` for Radau IIA tableaus

**error_atol** / **error_rtol** / **error_max_iters** / **error_residual_reduction** / **error_residual_floor** — the settings of the smoothing solve.
    Use these options like their ``krylov_`` counterparts, for the
    linear solve that ``use_smoothed_error`` adds.

    - Default: the value of the matching ``krylov_`` option

DAE options:

**dae_initialisation** — make a DAE's initial values consistent.
    Cubie solves for consistent algebraic states at the start of every
    run, before the first step and the ``t=0`` output, starting from
    the initial values you gave. With ``"brown"``, cubie holds the
    differential states and solves for the algebraic ones. With
    ``"shampine"``, cubie takes one backward Euler step of the starting
    step size, which can move every state by up to ``O(dt)``. With
    ``"none"``, cubie skips the solve. If the solve fails, cubie ends the run at the ``t=0`` save and sets
    ``DAE_INITIALISATION_FAILED`` in its status code. Cubie ignores
    this option on systems that aren't DAEs.

    - Default: ``"brown"``

Choosing what gets saved
------------------------

See :doc:`results` for each of these options.

**output_types** — what to record.
    Pass any combination of ``"state"``, ``"observables"``, ``"time"``,
    ``"iteration_counters"`` and the summary metric names.

    - Default: ``["state"]``

**save_variables** / **summarise_variables** — the variables to save and summarise, by name.
    You can also select variables by index with
    ``saved_state_indices``, ``saved_observable_indices``,
    ``summarised_state_indices`` and ``summarised_observable_indices``.

    - Default: every state and observable; without
      ``summarise_variables``, cubie summarises the saved variables

**save_every** / **summarise_every** / **sample_summaries_every** — how often to save, summarise and sample.
    See :doc:`timing` for each interval.

    - Defaults: without ``save_every``, cubie saves the first and last
      points; without ``summarise_every``, it summarises once at the
      end. You must give ``sample_summaries_every`` if you ask for any
      summary.

Drivers
-------

See :doc:`/user_guide/defining_systems/drivers` for each of these
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

    - Default: the precision you built the system with, ``float32``
      unless you set it

**operation_ordering** — the order of the operations in your system's generated code.
    Pass ``"liveness_auto"``, ``"kahn"``, ``"greedy"`` or ``"dfs"``.

    - Default: the ``CUBIE_OPERATION_ORDERING`` environment variable,
      or ``"liveness_auto"`` if you haven't set it

**lineinfo** — compile source-line data for profilers such as Nsight Compute.
    If you change it, cubie rebuilds every kernel.

    - Default: the ``CUBIE_LINEINFO`` environment variable, or off if
      you haven't set it

**nsz** / **contract** / **arcp** / **afn** / **reassoc** / **ftz** — the compiler's fast-math flags.
    Set ``nsz`` to let the compiler ignore the sign of zero,
    ``contract`` to let it fuse multiplies and adds, ``arcp`` to let it
    replace a division with a multiplication by the reciprocal, ``afn``
    to let it use approximate ``log``, ``exp`` and ``pow``, ``reassoc``
    to let it regroup chains of additions and multiplications, and
    ``ftz`` to let it flush subnormal numbers to zero.

    - Defaults: ``reassoc=False``; every other flag ``True``

**lto** — optimise across device functions when cubie links the kernel.

    - Default: ``True``

**kernel_name** — the name of the compiled kernel in profiler output.

    - Default: the algorithm name followed by the system name; cubie
      appends ``_ltoon`` or ``_ltooff`` to either

Performance
-----------

**auto_performance** — choose buffer locations, loop unrolling and the launch shape for your system and GPU.
    Cubie never overrides a ``*_location``, ``unroll_*`` or
    ``blocksize`` option that you set.

    - Default: ``True``

**blocksize** — the number of threads in each CUDA block.
    Pass it to ``solve`` to set it for one launch.

    - Default: chosen by cubie

**max_registers** — the register limit for each thread.
    With a lower limit, you can run more threads at once, but the
    compiler spills more values to memory.

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
    per-block memory. Cubie computes the same results with either
    location.

    - Default: chosen by cubie under ``auto_performance``; otherwise
      ``"local"``

Memory and caching
------------------

See :doc:`/user_guide/advanced/memory` and
:doc:`/user_guide/advanced/caching` for these options.

**mem_proportion** — the share of VRAM this solver may use.
    Pass a number between 0.0 and 1.0. With ``None``, cubie puts the
    solver in the automatic pool, and divides the VRAM left after every
    solver with a set proportion equally between the solvers in that
    pool.

    - Default: ``None``

**stream_group** — the name of the stream group the solver runs on.
    Cubie runs every solver in a group on one CUDA stream.

    - Default: ``"solver"``

**memory_manager** — the :class:`~cubie.memory.mem_manager.MemoryManager` cubie allocates the solver's memory with.

    - Default: ``cubie.default_memmgr``

**cache_enabled** — read compiled kernels from the disk cache and save new ones to it.

    - Default: ``True``

**cache_dir** — the directory of the compiled-kernel cache.

    - Default: the ``CUBIE_KERNEL_CACHE_DIR`` environment variable, or
      the shared cache root if you haven't set it

**max_cache_entries** — the number of compiled kernels cubie keeps in each cache directory before it removes the least recently used one.
    Pass ``0`` to keep every kernel.

    - Default: the ``CUBIE_MAX_CACHE_ENTRIES`` environment variable, or
      ``0`` if you haven't set it

Logging
-------

**time_logging_level** — how much timing information cubie prints.
    Pass ``"default"``, ``"verbose"`` or ``"debug"`` to print more
    detail, ``"silent"`` to record GPU timings without printing them,
    or ``None`` to turn timing off.

    - Default: ``None``
