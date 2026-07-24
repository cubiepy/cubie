Configuration Reference
========================

This page is the single index of every entry-point keyword argument for a
solve: the explicit parameters of :func:`~cubie.solve_ivp` and
:class:`~cubie.Solver` (:meth:`~cubie.Solver.__init__` and
:meth:`~cubie.Solver.solve`), plus how the remaining loose keyword arguments
are routed to the six underlying settings groups. Deep-dive pages for each
group are linked at the end of each section.

Entry-point signatures
-----------------------

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
   * - ``method``
     - ``"euler"``
     - Integration algorithm name (see :doc:`choosing_algorithms`).
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
     - ``"combinatorial"`` takes every combination of the supplied
       inputs; ``"verbatim"`` pairs them run-for-run. **This default
       differs from** ``Solver.solve`` **below.**
   * - ``time_logging_level``
     - ``None``
     - Timing verbosity: ``'default'``, ``'verbose'``, ``'debug'``, or
       ``None``/``'None'`` to disable.
   * - ``nan_error_trajectories``
     - ``True``
     - Trajectories with a nonzero status code are NaN-masked in the
       output.

Any other keyword argument is forwarded to :class:`~cubie.Solver`.

``Solver.__init__()``
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Effect
   * - ``algorithm``
     - ``"euler"``
     - Integration algorithm name or a supplied
       :class:`~cubie.integrators.algorithms.ButcherTableau`.
   * - ``lineinfo``
     - ``None``
     - Compiles every kernel and device function with source-line
       correlation data for profilers such as Nsight Compute. ``None``
       defers to the ``CUBIE_LINEINFO`` environment variable (default
       off); an explicit argument always wins. Updating it later
       triggers a full rebuild.
   * - ``cache``
     - ``True``
     - Compiled-kernel disk caching; accepts ``bool``, a cache-mode
       string, or a ``Path``. See :doc:`caching`.
   * - ``time_logging_level``
     - ``None``
     - Same options as above.

``step_control_settings``, ``algorithm_settings``, ``output_settings``,
``memory_settings``, and ``loop_settings`` are explicit dictionaries that
override the per-group defaults; any of their keys may also be supplied as
loose keyword arguments (see "Kwarg routing" below).

``Solver.solve()``
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Effect
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
     - ``256``
     - CUDA threads per block for the kernel launch.
   * - ``grid_type``
     - ``"verbatim"``
     - **Differs from** ``solve_ivp``'s default of
       ``"combinatorial"`` above — only relevant when dict inputs
       trigger grid construction.
   * - ``on_device``
     - ``False``
     - Return a
       :class:`~cubie.batchsolving.solveresult.DeviceSolveResult` of
       device-array handles with no device-to-host copy. See
       :doc:`results`.
   * - ``nan_error_trajectories``
     - ``True``
     - Same behaviour as above; ignored when ``on_device=True``.

Any other keyword argument is forwarded to :meth:`~cubie.Solver.update`,
routing through the same six settings groups described next.

Kwarg routing
-------------

Beyond the explicit parameters above, ``Solver`` accepts loose
keyword arguments and sorts them into settings groups. Any parameter
below may be passed directly to :func:`~cubie.solve_ivp`,
:class:`~cubie.Solver`, or :meth:`~cubie.Solver.update`:

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Group
     - Parameters
     - Deep-dive page
   * - Step control
     - ``step_controller``, ``dt``, ``dt_min``, ``dt_max``,
       ``atol``, ``rtol``, ``safety``, ``min_gain``, ``max_gain``,
       ``kp``, ``ki``, ``kd``, ``deadband_min``, ``deadband_max``,
       ``newton_target_iters``
     - :doc:`optional_arguments`
   * - Algorithm
     - ``newton_atol``, ``newton_rtol``, ``newton_max_iters``,
       ``krylov_atol``, ``krylov_rtol``, ``krylov_max_iters``,
       ``krylov_residual_reduction``, ``krylov_residual_floor``,
       ``linear_correction_type``, ``preconditioner_type``,
       ``preconditioner_order``
     - :doc:`optional_arguments`, :doc:`choosing_algorithms`
   * - Output
     - ``output_types``, ``saved_state_indices``,
       ``saved_observable_indices``, ``summarised_state_indices``,
       ``summarised_observable_indices``
     - :doc:`results`
   * - Timing / loop
     - ``save_every``, ``summarise_every``,
       ``sample_summaries_every``, ``save_last``, plus the
       per-buffer ``*_location`` placement overrides
     - :doc:`timing`, :doc:`speed`
   * - Memory
     - ``mem_proportion``, ``stream_group``
     - :doc:`memory`
   * - Cache
     - ``cache`` (``True``/``False``, a cache-mode string, or a
       ``Path``)
     - :doc:`caching`
   * - Kernel
     - ``max_registers``
     - :doc:`speed`

A keyword argument that matches no group raises ``KeyError`` at
construction rather than being silently ignored, and the legacy
timing spellings (``dt_save``, ``dt_summarise``,
``dt_update_summaries``) raise ``KeyError`` with a rename hint.

Notes on selected parameters
----------------------------

**step_controller**
    Selects the step-size controller by name: ``"fixed"``, ``"i"``,
    ``"pi"``, ``"pid"``, or ``"gustafsson"``. When omitted, the
    chosen algorithm's own default controller is used — fixed for
    schemes without an embedded error estimate, otherwise PI for
    explicit Runge–Kutta and Gustafsson for the implicit families.
    See :doc:`choosing_algorithms` and the algorithm defaults table
    in the API reference.

**max_registers**
    Per-thread register cap forwarded to ``cuda.jit``. Default
    ``None`` leaves register allocation to ``ptxas``; capping trades
    spill traffic for more resident warps. See :doc:`speed`.

**mem_proportion**
    Proportion of VRAM (0.0–1.0) reserved for this solver's
    allocations. Default ``None`` places the solver in the automatic
    pool, which shares the VRAM remaining after manually-proportioned
    solvers are accounted for equally between its members. See
    :doc:`memory`.
