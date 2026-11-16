#On a deeper read of this page it seems like it is either useless or poorly impelemented - I like the idea of linking each param to its deep dive page, but is this just a duplication of api ref? where does the optional arguments page live, and why? where is the distinction between the two?

Configuration Reference
========================

You can give a solve the named arguments of :func:`~cubie.solve_ivp`,
:class:`~cubie.Solver` and :meth:`~cubie.Solver.solve`, listed in the
first section below. You can also add any of the keyword options in the
second section to any of those calls. Follow the link beside each
keyword option to read what it does.

Named arguments
---------------

``solve_ivp()``
~~~~~~~~~~~~~~~

:func:`~cubie.solve_ivp` builds a :class:`~cubie.Solver` and runs a
single batch solve.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Effect
   * - ``y0``
     - ``None``
     - Initial values, as a dict or an array (see :doc:`inputs`).
       ``None`` starts every run from the system's default initial
       values.
   * - ``parameters``
     - ``None``
     - Parameter values, as a dict or an array (see :doc:`inputs`).
       ``None`` uses the system's default parameters for every run.
   * - ``drivers``
     - ``None``
     - A :class:`~cubie.array_interpolator.DriverSamples` of forcing
       data (see :doc:`/user_guide/defining_systems/drivers`).
   * - ``method``
     - ``"tsit5"``
     - Integration algorithm name (see :doc:`/user_guide/algorithms_and_tuning/choosing_algorithms`).
   * - ``duration``
     - ``1.0``
     - Total integration time.
   * - ``settling_time``
     - ``0.0``
     - Warm-up period before outputs are recorded.
   * - ``t0``
     - ``0.0``
     - Initial integration time.
   * - ``save_variables``
     - ``None``
     - State/observable names to save; ``None`` saves all, ``[]`` saves
       none.
   * - ``summarise_variables``
     - ``None``
     - State/observable names to summarise; ``None`` mirrors
       ``save_variables``.
   * - ``grid_type``
     - ``"combinatorial"``
     - Pass ``"combinatorial"`` to solve every combination of the
       values in dict inputs, or ``"verbatim"`` to pair them
       run-for-run.
   * - ``time_logging_level``
     - ``None``
     - Timing verbosity: ``'default'``, ``'verbose'``, ``'debug'``,
       ``'silent'`` (record GPU events, print nothing), or
       ``None``/``'None'`` to disable.
   * - ``nan_error_trajectories``
     - ``True``
     - Trajectories with a nonzero status code are NaN-masked in the
       output.
   * - ``fix_constant_parameters``
     - ``True``
     - Compile parameters that have one value across the batch into the
       code (see :ref:`parameter-inputs`).

``solve_ivp`` passes every other keyword argument to
:class:`~cubie.Solver`.

``Solver.__init__()``
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Effect
   * - ``algorithm``
     - ``"tsit5"``
     - Integration algorithm name or a supplied
       :class:`~cubie.integrators.algorithms.ButcherTableau`.
   * - ``lineinfo``
     - ``None``
     - Compiles every kernel and device function with source-line
       correlation data for profilers such as Nsight Compute. ``None``
       defers to the ``CUBIE_LINEINFO`` environment variable (default
       off); an explicit argument always wins. Updating it later
       triggers a full rebuild.
   * - ``auto_performance``
     - ``True``
     - Whether to use in-built heuristics for buffer locations in
       memory, loop unrolling, block size and resident blocks per SM.
       Does not override manual ``unroll_*``, ``*_location`` or
       ``blocksize`` arguments.
   * - ``time_logging_level``
     - ``None``
     - Same options as above.

You can also collect options in a dict and pass it as
``step_control_settings``, ``algorithm_settings``, ``system_settings``,
``output_settings``, ``memory_settings`` or ``loop_settings``. Cubie
treats each key of the dict as if you had passed it as a keyword
argument.

``Solver.solve()``
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Effect
   * - ``initial_values``
     - ``None``
     - Initial values, as a dict or an array (see :doc:`inputs`).
       ``None`` starts every run from the system's default initial
       values.
   * - ``parameters``
     - ``None``
     - Parameter values, as a dict or an array (see :doc:`inputs`).
       ``None`` uses the system's default parameters for every run.
   * - ``drivers``
     - ``None``
     - A :class:`~cubie.array_interpolator.DriverSamples` to use in
       place of the solver's forcing data.
   * - ``duration``
     - ``1.0``
     - Total integration time.
   * - ``settling_time``
     - ``0.0``
     - Warm-up period before outputs are recorded.
   * - ``t0``
     - ``0.0``
     - Initial integration time.
   * - ``blocksize``
     - ``None``
     - CUDA threads per block for this launch; ``None`` uses the
       solver's ``blocksize`` setting, or lets the solver pick when
       that setting was not given.
   * - ``grid_type``
     - ``"combinatorial"``
     - Same as ``solve_ivp`` above. Cubie only uses it when you pass
       dict inputs.
   * - ``nan_error_trajectories``
     - ``True``
     - Same behaviour as above; ignored when ``on_device=True``.
   * - ``on_device``
     - ``False``
     - Return a
       :class:`~cubie.batchsolving.solveresult.DeviceSolveResult` of
       device-array handles with no device-to-host copy. See
       :doc:`/user_guide/using_the_solver/results`.
   * - ``fix_constant_parameters``
     - ``True``
     - Same as ``solve_ivp`` above.

``solve`` passes every other keyword argument to
:meth:`~cubie.Solver.update`, and the solver keeps it for later solves.

.. _kwarg-routing:

Keyword options
---------------

You can pass any option below as a keyword argument to
:func:`~cubie.solve_ivp`, :class:`~cubie.Solver`,
:meth:`~cubie.Solver.solve` or :meth:`~cubie.Solver.update`:

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Options for
     - Parameters
     - Explained in
   * - Algorithm choice
     - ``tableau``
     - :doc:`/user_guide/algorithms_and_tuning/optional_arguments`
   * - Step control
     - ``step_controller``, ``dt``, ``dt_min``, ``dt_max``,
       ``atol``, ``rtol``, ``safety``, ``min_step_shrink``,
       ``max_step_growth``, ``integral_gain``, ``proportional_gain``,
       ``derivative_gain``, ``filter_coefficients``, ``deadband_min``,
       ``deadband_max``, ``newton_target_iters``
     - :doc:`/user_guide/algorithms_and_tuning/optional_arguments`
   * - Starting step
     - ``starting_step_state_fraction``, ``starting_step_zero_norm``,
       ``starting_step_target``, ``starting_step_max_growth``
     - :doc:`/user_guide/algorithms_and_tuning/optional_arguments`
   * - Implicit solvers
     - ``newton_atol``, ``newton_rtol``, ``newton_max_iters``,
       ``krylov_atol``, ``krylov_rtol``, ``krylov_max_iters``,
       ``krylov_residual_reduction``, ``krylov_residual_floor``,
       ``linear_correction_type``, ``preconditioner_type``,
       ``preconditioner_order``, ``use_smoothed_error``,
       ``dae_initialisation``
     - :doc:`/user_guide/algorithms_and_tuning/optional_arguments`, :doc:`/user_guide/algorithms_and_tuning/choosing_algorithms`
   * - Implicit solvers
     - ``inexact_newton``, ``prefactored``
     - :class:`~cubie.integrators.algorithms.ImplicitStepConfig`
   * - Implicit solvers
     - ``attempt_dense_prediction``, ``error_atol``, ``error_rtol``,
       ``error_max_iters``, ``error_residual_floor``,
       ``error_residual_reduction``
     - —
   * - Output
     - ``output_types``, ``save_variables``, ``summarise_variables``,
       ``saved_state_indices``, ``saved_observable_indices``,
       ``summarised_state_indices``, ``summarised_observable_indices``
     - :doc:`/user_guide/using_the_solver/results`
   * - Output timing
     - ``save_every``, ``summarise_every``, ``sample_summaries_every``
     - :doc:`/user_guide/using_the_solver/timing`
   * - Drivers
     - ``drivers``, ``order``, ``wrap``, ``boundary_condition``
     - :doc:`/user_guide/defining_systems/drivers`
   * - System
     - ``precision``
     - :doc:`/user_guide/algorithms_and_tuning/speed`
   * - System
     - ``operation_ordering``
     - —
   * - Memory
     - ``mem_proportion``, ``stream_group``, ``memory_manager``
     - :doc:`/user_guide/advanced/memory`
   * - Cache
     - ``cache_enabled``, ``cache_dir``, ``max_cache_entries``
     - :doc:`/user_guide/advanced/caching`
   * - Kernel
     - ``max_registers``, ``blocksize``
     - The notes below
   * - Kernel
     - ``kernel_name``
     - —
   * - Performance
     - ``unroll_stage``, ``unroll_step_element``,
       ``unroll_accumulator``, ``unroll_solver_element``,
       ``unroll_norms``, ``unroll_other_small``,
       ``unroll_newton_exits``, ``unroll_krylov_exits``, and the
       per-buffer ``*_location`` placement options
     - :doc:`/user_guide/algorithms_and_tuning/speed`
   * - Compiler flags
     - ``nsz``, ``contract``, ``arcp``, ``afn``, ``reassoc``, ``ftz``,
       ``lto``
     - —

If you pass a keyword argument that isn't an option, cubie raises
``KeyError``. Give parameter values in the ``parameters`` argument of
``solve``.

Notes on selected parameters
----------------------------

**step_controller**
    Selects the step-size controller by name: ``"fixed"``, ``"i"``,
    ``"pi"``, ``"pid"``, or ``"gustafsson"``. When you leave it out,
    cubie uses the chosen algorithm's own default controller: fixed
    for schemes without an embedded error estimate, otherwise integral
    for explicit Runge–Kutta, PI for DIRK and Gustafsson for FIRK and
    Rosenbrock-W. See
    :doc:`/user_guide/algorithms_and_tuning/choosing_algorithms` and
    the algorithm defaults table in the API reference.

**max_registers**
    Per-thread register cap forwarded to ``cuda.jit``. Default
    ``None`` leaves register allocation to ``ptxas``; capping trades
    spill traffic for more resident warps. See :doc:`/user_guide/algorithms_and_tuning/speed`.

**blocksize**
    Threads per block for every launch; ``solve(blocksize=...)``
    overrides one launch. Unset, the solver picks a block size itself.

**mem_proportion**
    Proportion of VRAM (0.0–1.0) reserved for this solver's
    allocations. Default ``None`` places the solver in the automatic
    pool, which shares the VRAM remaining after manually-proportioned
    solvers are accounted for equally between its members. See
    :doc:`/user_guide/advanced/memory`.
