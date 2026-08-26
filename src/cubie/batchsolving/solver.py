"""High level batch-solver interface.

This module exposes the user-facing :class:`Solver` class and a convenience
wrapper :func:`solve_ivp` for solving batches of initial value problems on the
GPU.

Published Classes
-----------------
:class:`Solver`
    User-facing class for configuring and executing batch ODE solves.

Module-Level Functions
----------------------
:func:`solve_ivp`
    Convenience wrapper that creates a :class:`Solver` and executes a single
    batch solve in one call.

Notes
-----
When GPU memory is insufficient for the full batch, arrays are automatically
chunked along the run axis. Chunking is transparent to the user and requires
no configuration.

See Also
--------
:class:`~cubie.batchsolving.solveresult.SolveResult`
    Result container returned by :meth:`Solver.solve`.
:class:`~cubie.batchsolving.BatchSolverKernel.BatchSolverKernel`
    Kernel factory used internally by the solver.
:class:`~cubie.batchsolving.BatchInputHandler.BatchInputHandler`
    Grid builder used for dict-based inputs.
"""

from pathlib import Path
from functools import partial
from weakref import finalize
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from numpy import asarray, ndarray

from cubie.outputhandling.output_config import OutputCompileFlags
from cubie._utils import PrecisionDType
from cubie.result_codes import decode_status_codes
from cubie.batchsolving.BatchSolverConfig import ActiveOutputs
from cubie.batchsolving.BatchInputHandler import BatchInputHandler
from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.batchsolving.calibration import (
    CalibrationResult,
    run_calibration,
)
from cubie.batchsolving.solveresult import (
    DeviceSolveResult,
    SolveResult,
    SolveSpec,
)
from cubie.batchsolving.SystemInterface import SystemInterface
from cubie.memory.mem_manager import ALL_MEMORY_MANAGER_PARAMETERS
from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.ODEData import ALL_ODE_PARAMETERS
from cubie.odesystems.symbolic import create_ODE_system
from cubie.array_interpolator import ArrayInterpolator
from cubie.integrators.algorithms.base_algorithm_step import (
    ALL_ALGORITHM_STEP_PARAMETERS,
)
from cubie.batchsolving.location_tuning import (
    TuneResult,
    tune_locations as _tune_locations,
)
from cubie.integrators.memory_heuristics import auto_memory_locations
from cubie.integrators.loops.ode_loop import (
    ALL_LOOP_SETTINGS,
)
from cubie.integrators.step_control.base_step_controller import (
    ALL_STEP_CONTROLLER_PARAMETERS,
)
from cubie._utils import merge_kwargs_into_settings
from cubie.outputhandling.output_functions import (
    ALL_OUTPUT_FUNCTION_PARAMETERS,
)
from cubie.time_logger import default_timelogger
from cubie.batchsolving.BatchSolverConfig import ALL_KERNEL_PARAMETERS
from cubie.cubie_cache import ALL_CACHE_PARAMETERS

# Register module-level events
default_timelogger.register_event(
    "solve_ivp", "runtime", "Wall-clock time for solve_ivp()"
)


def _close_abandoned_kernel(kernel: BatchSolverKernel) -> None:
    """Best-effort close for a collected solver's kernel."""
    try:
        kernel.close()
    except Exception:  # pragma: no cover - context already gone
        pass


def _finalize_solver(kernel: BatchSolverKernel) -> None:
    """Record an abandoned solver's teardown; the GC finalizer target.

    GC can fire inside any allocation — including while the memory
    manager iterates its registry — so the kernel is closed at the
    manager's next entry point, not here. See
    :meth:`MemoryManager.defer_teardown`.
    """
    kernel.memory_manager.defer_teardown(
        partial(_close_abandoned_kernel, kernel)
    )


default_timelogger.register_event(
    "solver_solve", "runtime", "Wall-clock time for Solver.solve()"
)


RENAMED_TIMING_KWARGS = {
    "dt_save": "save_every",
    "dt_summarise": "summarise_every",
    "dt_update_summaries": "sample_summaries_every",
}
"""Legacy timing keyword spellings mapped to their current names."""


def _system_from_equations(
    dxdt: Union[str, Callable, Iterable[str]],
    y0: Optional[Union[ndarray, Dict[str, object]]],
    parameters: Optional[Union[ndarray, Dict[str, object]]],
    drivers: Optional[Dict[str, object]],
    precision: Optional[PrecisionDType] = None,
) -> BaseODE:
    """Build a :class:`SymbolicODE` from equations passed to solve_ivp.

    Parameters
    ----------
    dxdt
        Equations as a callable, an equation string, or an iterable of
        equation strings.
    y0
        Initial-value input. A dict supplies state names and default
        initial values; an array defers state naming to inference.
    parameters
        Parameter input. A dict supplies parameter names and default
        values (the first value of each entry). Arrays are rejected
        because they carry no names to declare.
    drivers
        Driver configuration forwarded to system creation.
    precision
        Optional precision override for the created system.

    Returns
    -------
    BaseODE
        System constructed from the supplied equations.

    Raises
    ------
    TypeError
        If ``parameters`` is a non-dict sequence (array, list, or
        tuple), which carries no names to declare parameters.
    """
    if parameters is not None and not isinstance(parameters, dict):
        raise TypeError(
            "When equations are supplied directly to solve_ivp, "
            "parameters must be a dict mapping names to values so the "
            "system's parameters can be declared."
        )
    states = None
    if isinstance(y0, dict):
        states = {
            name: float(asarray(values).flat[0])
            for name, values in y0.items()
        }
    parameter_defaults = None
    if isinstance(parameters, dict):
        parameter_defaults = {
            name: float(asarray(values).flat[0])
            for name, values in parameters.items()
        }
    create_kwargs = {}
    if precision is not None:
        create_kwargs["precision"] = precision
    return create_ODE_system(
        dxdt=dxdt,
        states=states,
        parameters=parameter_defaults,
        drivers=drivers,
        **create_kwargs,
    )


def _check_renamed_kwargs(keys: Iterable[str]) -> None:
    """Raise ``KeyError`` for legacy keyword spellings with rename hints.

    Parameters
    ----------
    keys
        Keyword-argument names supplied by the caller.

    Raises
    ------
    KeyError
        If any key is a legacy spelling listed in
        :data:`RENAMED_TIMING_KWARGS`.
    """
    renamed = [key for key in keys if key in RENAMED_TIMING_KWARGS]
    if renamed:
        hints = ", ".join(
            f"'{key}' is now '{RENAMED_TIMING_KWARGS[key]}'"
            for key in renamed
        )
        raise KeyError(f"Renamed keyword argument(s): {hints}.")


_OUTPUT_SELECTION_KEYS = (
    "save_variables",
    "summarise_variables",
    "saved_state_indices",
    "saved_observable_indices",
    "summarised_state_indices",
    "summarised_observable_indices",
)
"""Settings recording which variables the user asked to output."""


def solve_ivp(
    system: Union[BaseODE, str, Callable, Iterable[str]],
    y0: Union[ndarray, Dict[str, ndarray]],
    parameters: Optional[Union[ndarray, Dict[str, ndarray]]] = None,
    drivers: Optional[Dict[str, object]] = None,
    method: str = "euler",
    duration: float = 1.0,
    settling_time: float = 0.0,
    t0: float = 0.0,
    save_variables: Optional[List[str]] = None,
    summarise_variables: Optional[List[str]] = None,
    grid_type: str = "combinatorial",
    time_logging_level: Optional[str] = None,
    nan_error_trajectories: bool = True,
    **kwargs: Any,
) -> SolveResult:
    """Solve a batch initial value problem.

    Parameters
    ----------
    system
        System model defining the differential equations. Accepts a
        prebuilt :class:`~cubie.odesystems.baseODE.BaseODE`, or raw
        equations as a Python callable, an equation string, or an
        iterable of equation strings. Raw equations are converted with
        :func:`~cubie.odesystems.symbolic.symbolicODE.create_ODE_system`,
        taking state names and defaults from a ``y0`` dict and
        parameter names and defaults from a ``parameters`` dict; for
        repeated solves of the same system, build it once with
        ``create_ODE_system`` and reuse a :class:`Solver` instead.
    y0
        Initial state values for each run as arrays or dictionaries mapping
        labels to arrays.
    parameters
        Parameter values for each run as arrays or dictionaries mapping labels
        to arrays.
    drivers
        Driver configuration to interpolate during integration.
    method
        Integration algorithm to use. Default is ``"euler"``.
    duration
        Total integration time. Default is ``1.0``.
    settling_time
        Warm-up period prior to storing outputs. Default is ``0.0``.
    t0
        Initial integration time supplied to the solver. Default is ``0.0``.
    save_variables : list of str, optional
        Variable names (states or observables) to save in time-domain output.
        ``None`` (default) saves all states and observables. An empty list
        ``[]`` explicitly saves no variables. When both ``save_variables`` and
        index parameters (``saved_state_indices``,
        ``saved_observable_indices``) are provided, their union is used. For
        less overhead, you can provide indices directly, which don't require
        the solver to look up variable names.
    summarise_variables : list of str, optional
        Variable names (states or observables) to include in summary
        calculations. ``None`` (default) summarises the same variables that
        are saved. An empty list ``[]`` explicitly summarises no variables.
        When both ``summarise_variables`` and index parameters are provided,
        their union is used.
    grid_type
        ``"verbatim"`` pairs each input vector while ``"combinatorial"``
        produces every combination of provided values.
    time_logging_level : str or None, default='default'
        Time logging verbosity level. Options are 'default', 'verbose',
        'debug', None, or 'None' to disable timing.
    nan_error_trajectories : bool, default=True
        When ``True`` (default), trajectories with nonzero solver status
        codes are automatically set to NaN, protecting users from analyzing
        invalid data. When ``False``, all trajectories are returned with
        original values.
    **kwargs
        Additional keyword arguments passed to :class:`Solver`.

    Returns
    -------
    SolveResult
        Result owning the solve's host output buffers. ``as_numpy``,
        ``as_numpy_per_summary``, and ``as_pandas`` build RAM
        representations on demand; disk-backed results release their
        spill files on ``close()`` or context exit.
    """
    if not isinstance(system, BaseODE):
        system = _system_from_equations(
            system,
            y0,
            parameters,
            drivers,
            precision=kwargs.pop("precision", None),
        )

    # Collect required explicit parameters from kwargs
    loop_settings = kwargs.pop("loop_settings", None)

    if save_variables is not None:
        kwargs.setdefault("save_variables", save_variables)
    if summarise_variables is not None:
        kwargs.setdefault("summarise_variables", summarise_variables)

    # Solve-time options go to solve(); the rest configure the Solver.
    solve_option_keys = ("blocksize",)
    solve_options = {
        key: kwargs.pop(key) for key in solve_option_keys if key in kwargs
    }

    solver = Solver(
        system,
        algorithm=method,
        loop_settings=loop_settings,
        time_logging_level=time_logging_level,
        **kwargs,
    )

    # Start wall-clock timing
    default_timelogger.start_event("solve_ivp")

    try:
        results = solver.solve(
            y0,
            parameters,
            drivers=drivers,
            duration=duration,
            settling_time=settling_time,
            t0=t0,
            grid_type=grid_type,
            nan_error_trajectories=nan_error_trajectories,
            **solve_options,
        )
        default_timelogger.stop_event("solve_ivp")
        default_timelogger.print_summary()
    finally:
        solver.close()

    return results


class Solver:
    """User-facing interface for solving batches of ODE systems.

    Parameters
    ----------
    system
        System model containing the ODEs to integrate.
    algorithm
        Integration algorithm to use. Defaults to ``"euler"``.
    lineinfo
        Compile all kernels and device functions with source-line
        correlation data for profilers such as Nsight Compute. ``None``
        defers to the ``CUBIE_LINEINFO`` environment variable (default
        off). Changing it later via :meth:`update` triggers a rebuild.
    step_control_settings
        Explicit controller configuration that overrides solver defaults.
    algorithm_settings
        Explicit algorithm configuration overriding solver defaults.
    system_settings
        ODE compile settings keyed by :class:`cubie.odesystems.ODEData`
        compile parameters (for example ``operation_ordering``); each
        key may also be supplied as a loose keyword.
    output_settings
        Explicit output configuration overriding solver defaults. Individual
        selectors such as ``save_variables`` or index-based parameters may also
        be supplied as keyword arguments.
    memory_settings
        Memory configuration; each key may also be a keyword argument.
        ``host_spill_threshold`` is the size in bytes above which host
        result arrays are disk-backed instead of held in RAM; by
        default only arrays larger than 80% of total system RAM spill
        — everything smaller is pageable RAM the operating system
        manages. Lower it to keep RAM free for other work, or raise it
        to keep even larger results in RAM. ``spill_directory`` is an
        existing directory for spill files (default: the system temp
        directory); point it at a fast disk for large spilled runs.
        An idle solver's completed device buffers are freed when
        another solver faces a genuine VRAM shortage; the evicted
        solver reallocates on its next solve.
    loop_settings
        Explicit loop configuration overriding solver defaults. Keys such as
        ``save_every`` and ``summarise_every`` may also be supplied as loose
        keyword arguments.
    time_logging_level : str or None, default='default'
        Time logging verbosity level. Options are 'default', 'verbose',
        'debug', None, or 'None' to disable timing.
    auto_memory : bool or str, default=True
        ``True`` applies the measured shared-memory placements of
        :mod:`cubie.integrators.memory_heuristics` at construction;
        ``False`` keeps every unspecified buffer local; ``"tune"``
        measures per-buffer placements on the first :meth:`solve`
        (see :meth:`tune_locations`). Explicit ``*_location``
        arguments always take precedence.
    **kwargs
        Additional keyword arguments forwarded to internal components. See
        "Optional Arguments" in the docs for the possibilities.

    Notes
    -----
    Instances coordinate batch grid construction, kernel configuration, and
    driver interpolation so that :meth:`solve` orchestrates a complete GPU
    integration run.

    When specifying variables:

    - ``None`` means "use all" (default behavior for both states and
      observables)
    - ``[]`` (empty list) means "explicitly no variables"
    - When both labels and indices are provided, their union is used
    """

    def __init__(
        self,
        system: BaseODE,
        algorithm: str = "euler",
        lineinfo: Optional[bool] = None,
        step_control_settings: Optional[Dict[str, object]] = None,
        algorithm_settings: Optional[Dict[str, object]] = None,
        system_settings: Optional[Dict[str, object]] = None,
        output_settings: Optional[Dict[str, object]] = None,
        memory_settings: Optional[Dict[str, object]] = None,
        loop_settings: Optional[Dict[str, object]] = None,
        time_logging_level: Optional[str] = None,
        cache: Union[bool, str, Path] = True,
        auto_memory: Union[bool, str] = True,
        **kwargs: Any,
    ) -> None:
        if auto_memory not in (True, False, "tune"):
            raise ValueError(
                "auto_memory must be True, False, or 'tune'; got "
                f"{auto_memory!r}"
            )
        self._auto_memory = auto_memory
        self._locations_tuned = False
        self._tuned_placement: Dict[str, str] = {}
        self._update_record: List[Dict[str, Any]] = []
        self._construction_record = {
            "system": system,
            "kwargs": dict(
                algorithm=algorithm,
                lineinfo=lineinfo,
                step_control_settings=dict(step_control_settings or {}),
                algorithm_settings=dict(algorithm_settings or {}),
                system_settings=dict(system_settings or {}),
                output_settings=dict(output_settings or {}),
                memory_settings=dict(memory_settings or {}),
                loop_settings=dict(loop_settings or {}),
                time_logging_level=time_logging_level,
                cache=cache,
                **dict(kwargs),
            ),
        }
        if output_settings is None:
            output_settings = {}
        if memory_settings is None:
            memory_settings = {}
        if step_control_settings is None:
            step_control_settings = {}
        if algorithm_settings is None:
            algorithm_settings = {}
        if system_settings is None:
            system_settings = {}
        if loop_settings is None:
            loop_settings = {}

        _check_renamed_kwargs(kwargs)

        # Set global time logging level
        default_timelogger.set_verbosity(time_logging_level)

        super().__init__()
        system_settings, system_recognized = merge_kwargs_into_settings(
            kwargs=kwargs,
            valid_keys=ALL_ODE_PARAMETERS,
            user_settings=system_settings,
        )
        if system_settings:
            system.update(system_settings)
        precision = system.precision
        kwargs["precision"] = precision
        interface = SystemInterface(system)
        self.system_interface = interface

        recognized_kwargs: set[str] = set()

        output_settings, output_recognized = merge_kwargs_into_settings(
            kwargs=kwargs,
            valid_keys=ALL_OUTPUT_FUNCTION_PARAMETERS,
            user_settings=output_settings,
        )
        # Label kwargs are converted to index settings, not forwarded.
        output_settings, label_recognized = merge_kwargs_into_settings(
            kwargs=kwargs,
            valid_keys=("save_variables", "summarise_variables"),
            user_settings=output_settings,
        )
        output_recognized |= label_recognized
        # Record the selection before conversion for re-resolution.
        self._output_selection_intent = {
            key: output_settings[key]
            for key in _OUTPUT_SELECTION_KEYS
            if output_settings.get(key) is not None
        }
        self._resolved_output_layout = None
        self.convert_output_labels(output_settings)
        self._resolved_output_layout = self._output_layout()

        memory_settings, memory_recognized = merge_kwargs_into_settings(
            kwargs=kwargs,
            valid_keys=ALL_MEMORY_MANAGER_PARAMETERS,
            user_settings=memory_settings,
        )

        step_settings, step_recognized = merge_kwargs_into_settings(
            kwargs=kwargs,
            valid_keys=ALL_STEP_CONTROLLER_PARAMETERS,
            user_settings=step_control_settings,
        )
        algorithm_settings, algorithm_recognized = merge_kwargs_into_settings(
            kwargs=kwargs,
            valid_keys=ALL_ALGORITHM_STEP_PARAMETERS,
            user_settings=algorithm_settings,
        )
        algorithm_settings["algorithm"] = algorithm
        loop_settings, loop_recognized = merge_kwargs_into_settings(
            kwargs=kwargs,
            valid_keys=ALL_LOOP_SETTINGS,
            user_settings=loop_settings,
        )
        # Merge cache settings from kwargs
        cache_settings, cache_recognized = merge_kwargs_into_settings(
            kwargs=kwargs,
            valid_keys=ALL_CACHE_PARAMETERS,
            user_settings={},
        )
        kernel_settings, kernel_recognized = merge_kwargs_into_settings(
            kwargs=kwargs,
            valid_keys=ALL_KERNEL_PARAMETERS,
            user_settings={},
        )
        recognized_kwargs = (
            step_recognized
            | algorithm_recognized
            | system_recognized
            | output_recognized
            | memory_recognized
            | loop_recognized
            | cache_recognized
            | kernel_recognized
        )

        self.kernel = BatchSolverKernel(
            system,
            loop_settings=loop_settings,
            lineinfo=lineinfo,
            step_control_settings=step_settings,
            algorithm_settings=algorithm_settings,
            output_settings=output_settings,
            memory_settings=memory_settings,
            cache=cache,
            cache_settings=cache_settings,
            kernel_settings=kernel_settings,
        )
        self._finalizer = finalize(self, _finalize_solver, self.kernel)
        # Grids assemble into buffers per the kernel's spill settings.
        self.input_handler = BatchInputHandler(
            interface,
            memory_manager=self.kernel.memory_manager,
            host_spill_threshold=self.kernel.host_spill_threshold,
            spill_directory=self.kernel.spill_directory,
        )
        self._solve_info_cache = None
        self._solve_info_key = None

        if set(kwargs) - recognized_kwargs:
            raise KeyError(
                "Unrecognized keyword arguments: "
                f"{set(kwargs) - recognized_kwargs}"
            )

        if auto_memory is True:
            user_location_keys = {
                key
                for source in (
                    kwargs,
                    algorithm_settings,
                    loop_settings,
                    step_control_settings,
                )
                for key in source
                if key.endswith("_location")
            }
            placements = auto_memory_locations(
                self.kernel.single_integrator,
                user_location_keys,
            )
            if placements:
                self.kernel.update(placements)

    def _apply_placement(self, placement: Dict[str, str]) -> None:
        """Apply tuned buffer locations without recording them as updates."""
        self.kernel.update(dict(placement))
        self._tuned_placement = dict(placement)
        self._locations_tuned = True

    @property
    def tuned_placement(self) -> Dict[str, str]:
        """Buffer locations applied by the last :meth:`tune_locations`."""
        return dict(self._tuned_placement)

    def tune_locations(
        self,
        initial_values: Union[ndarray, Dict[str, Union[float, ndarray]]],
        parameters: Union[ndarray, Dict[str, Union[float, ndarray]]],
        drivers: Optional[Dict[str, Any]] = None,
        duration: float = 1.0,
        settling_time: float = 0.0,
        t0: float = 0.0,
        blocksize: int = 256,
        grid_type: str = "verbatim",
        workers: int = 4,
        rounds: int = 6,
        force: bool = False,
    ) -> TuneResult:
        """Measure per-buffer shared placements and apply the fastest."""
        if self.kernel.system_config_stale:
            self.kernel.resync_system()
            self._refresh_output_selection()
        # The timing solves must not re-enter tuning.
        self._locations_tuned = True
        try:
            return _tune_locations(
                self,
                initial_values,
                parameters,
                drivers=drivers,
                duration=duration,
                settling_time=settling_time,
                t0=t0,
                blocksize=blocksize,
                grid_type=grid_type,
                workers=workers,
                rounds=rounds,
                force=force,
            )
        except BaseException:
            self._locations_tuned = False
            raise

    def close(self, shutdown_timeout: Optional[float] = None) -> None:
        """Release GPU resources after pending transfers finish.

        Parameters
        ----------
        shutdown_timeout
            Maximum seconds to wait. None waits until transfers finish.
        """
        kernel = getattr(self, "kernel", None)
        if kernel is not None:
            kernel.close(shutdown_timeout=shutdown_timeout)
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer.detach()

    def __enter__(self) -> "Solver":
        """Return self so the solver can be used as a context manager."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Release GPU resources on exit from a ``with`` block."""
        self.close()

    def _output_layout(self) -> tuple:
        """Return the system's current state/observable name layout."""
        return (
            tuple(self.system_interface.states.names),
            tuple(self.system_interface.observables.names),
        )

    def _refresh_output_selection(self) -> None:
        """Re-resolve the output selection after a layout change."""
        layout = self._output_layout()
        if layout == self._resolved_output_layout:
            return
        settings = dict(self._output_selection_intent)
        self.convert_output_labels(settings)
        self.kernel.update(settings, silent=True)
        self._resolved_output_layout = layout

    def convert_output_labels(
        self,
        output_settings: Dict[str, Any],
    ) -> None:
        """Convert variable labels to indices.

        Parameters
        ----------
        output_settings
            Output configuration kwargs. Entries used are ``save_variables``,
            ``summarise_variables``, ``saved_state_indices``,
            ``saved_observable_indices``, ``summarised_state_indices``,
            and ``summarised_observable_indices``.

        Raises
        ------
        ValueError
            If variable labels are not recognized by the system.
        """
        self.system_interface.merge_variable_labels_and_idxs(output_settings)

    @property
    def driver_interpolator(self) -> ArrayInterpolator:
        """The kernel-owned driver interpolator."""
        return self.kernel.driver_interpolator

    def _configure_drivers(self, drivers: Dict[str, Any]) -> None:
        """Update the kernel-owned driver interpolator as one unit.

        Parameters
        ----------
        drivers
            Driver samples plus interpolation settings, as accepted by
            :meth:`ArrayInterpolator.update_from_dict`.
        """
        self.kernel.configure_drivers(drivers)

    def solve(
        self,
        initial_values: Union[ndarray, Dict[str, Union[float, ndarray]]],
        parameters: Union[ndarray, Dict[str, Union[float, ndarray]]],
        drivers: Optional[Dict[str, Any]] = None,
        duration: float = 1.0,
        settling_time: float = 0.0,
        t0: float = 0.0,
        blocksize: int = 256,
        grid_type: str = "verbatim",
        nan_error_trajectories: bool = True,
        on_device: bool = False,
        **kwargs: Any,
    ) -> Union[SolveResult, DeviceSolveResult]:
        """Solve a batch initial value problem.

        Parameters
        ----------
        initial_values
            Initial state values for each integration run. Accepts
            dictionaries mapping state names to values for grid
            construction, or pre-built arrays in (n_states, n_runs)
            format for fast-path execution. Device arrays (CuPy or
            Numba) are used in place with no host-to-device transfer;
            they must already match the system precision.
        parameters
            Parameter values for each run. Accepts dictionaries
            mapping parameter names to values, or pre-built arrays
            in (n_params, n_runs) format. Device arrays are accepted
            as for ``initial_values``.
        drivers
            Driver samples or configuration matching
            :class:`cubie.array_interpolator.ArrayInterpolator`.
        duration
            Total integration time. Default is ``1.0``.
        settling_time
            Warm-up period before recording outputs. Default ``0.0``.
        t0
            Initial integration time. Default ``0.0``.
        blocksize
            CUDA block size used for kernel launch. Default ``256``.
        grid_type
            Strategy for constructing the integration grid from inputs.
            Only used when dict inputs trigger grid construction.
        nan_error_trajectories
            When ``True`` (default), trajectories with nonzero status codes
            are automatically set to NaN, making failed runs easy to identify
            and exclude from analysis. When ``False``, all trajectories are
            returned unchanged. Ignored when ``on_device`` is ``True``.
        on_device
            When ``True``, skip the device-to-host copy of the output
            arrays and return a :class:`DeviceSolveResult` holding the
            solver's device output buffers plus the CUDA stream the
            solve ran on; see Notes. Default ``False``.
        **kwargs
            Additional options forwarded to :meth:`update`. See "Optional
            Arguments" in the docs for possibilities.

        Returns
        -------
        SolveResult or DeviceSolveResult
            ``SolveResult`` owning the solve's host output buffers —
            nothing is copied. Keep it alive while its data is needed:
            once it is garbage collected the solver reuses the buffers
            on its next run. ``as_numpy``, ``as_numpy_per_summary``,
            and ``as_pandas`` build RAM representations on demand;
            disk-backed results release their spill files on
            ``close()`` or context exit. ``DeviceSolveResult`` when
            ``on_device`` is ``True``.

        Notes
        -----
        Input type detection determines the processing path:

        - Dictionary inputs trigger grid construction via
          :class:`BatchInputHandler`
        - Pre-built numpy arrays with correct shapes skip grid
          construction for improved performance
        - Device arrays are used in place: no grid construction and
          no host-to-device transfer
        - :attr:`device_initial_values`/:attr:`device_parameters`
          re-run the previous solve's inputs with nothing uploaded

        When GPU memory is insufficient for the full batch, arrays are
        automatically chunked along the run axis.

        ``on_device=True`` returns without synchronizing: buffer
        contents are valid once the returned stream is synchronized,
        and the next ``solve()`` on this solver overwrites them. A
        chunked run raises ``ValueError``.
        """
        if kwargs:
            self.update(kwargs)

        if self.kernel.system_config_stale:
            # Replay a direct system mutation through the update chain.
            self.kernel.resync_system()
            self._refresh_output_selection()

        if self._auto_memory == "tune" and not self._locations_tuned:
            self.tune_locations(
                initial_values,
                parameters,
                drivers=drivers,
                duration=duration,
                settling_time=settling_time,
                t0=t0,
                blocksize=blocksize,
                grid_type=grid_type,
            )

        # Start wall-clock timing for solve
        default_timelogger.start_event("solver_solve")

        inits, params = self.input_handler(
            states=initial_values, params=parameters, kind=grid_type
        )

        if drivers is not None:
            self._configure_drivers(drivers)

        self.kernel.run(
            inits=inits,
            params=params,
            driver_coefficients=self.driver_interpolator.coefficients,
            duration=duration,
            warmup=settling_time,
            t0=t0,
            blocksize=blocksize,
            transfer_outputs=not on_device,
        )

        if not on_device:
            # Synchronize stream, wait until arrays written in
            # "chunked" mode. Device results return unsynchronized:
            # the caller orders further work on the returned stream.
            self.kernel.synchronize()
            self.kernel.wait_for_writeback()

        # Stop wall-clock timing for solve
        default_timelogger.stop_event("solver_solve")
        default_timelogger.print_summary()

        if on_device:
            return DeviceSolveResult.from_solver(self)

        return SolveResult.from_solver(
            self,
            nan_error_trajectories=nan_error_trajectories,
        )

    def compile(
        self,
        initial_values: Union[ndarray, Dict[str, Union[float, ndarray]]],
        parameters: Union[ndarray, Dict[str, Union[float, ndarray]]],
        drivers: Optional[Dict[str, Any]] = None,
        duration: float = 1.0,
        settling_time: float = 0.0,
        t0: float = 0.0,
        grid_type: str = "verbatim",
        **kwargs: Any,
    ) -> None:
        """Compile the batch kernel for these inputs without solving."""
        if kwargs:
            self.update(kwargs)

        if self.kernel.system_config_stale:
            # Replay a direct system mutation through the update chain.
            self.kernel.resync_system()
            self._refresh_output_selection()

        inits, params = self.input_handler(
            states=initial_values, params=parameters, kind=grid_type
        )

        if drivers is not None:
            self._configure_drivers(drivers)

        self.kernel.compile(
            inits=inits,
            params=params,
            driver_coefficients=self.driver_interpolator.coefficients,
            duration=duration,
            warmup=settling_time,
            t0=t0,
        )

    def build_grid(
        self,
        initial_values: Union[
            ndarray, Dict[str, Union[float, ndarray]]
        ] = None,
        parameters: Union[
            None, ndarray, Dict[str, Union[float, ndarray]]
        ] = None,
        grid_type: str = "verbatim",
    ) -> Tuple[ndarray, ndarray]:
        """Build parameter and state grids for external use.

        Parameters
        ----------
        initial_values
            Initial state values as dictionaries mapping state names
            to value sequences, or arrays in (n_states, n_runs) format.
        parameters
            Parameter values as dictionaries mapping parameter names
            to value sequences, or arrays in (n_params, n_runs) format.
        grid_type
            Strategy for constructing the grid. ``"combinatorial"``
            produces all combinations while ``"verbatim"`` preserves
            column-wise pairings. Default is ``"verbatim"``.

        Returns
        -------
        Tuple[ndarray, ndarray]
            Tuple of (initial_values, parameters) arrays in
            (n_vars, n_runs) format with system precision dtype.
            These arrays can be passed directly to :meth:`solve`
            for fast-path execution.

        Examples
        --------
        >>> inits, params = solver.build_grid(
        ...     {"x": [1, 2, 3]}, {"p": [0.1, 0.2]}, grid_type="combinatorial"
        ... )
        >>> result = solver.solve(inits, params)  # Uses fast path
        """
        return self.input_handler(
            states=initial_values, params=parameters, kind=grid_type
        )

    def calibrate(
        self,
        initial_values: Union[ndarray, Dict[str, Any]],
        parameters: Union[ndarray, Dict[str, Any]],
        drivers: Optional[Dict[str, Any]] = None,
        duration: float = 1.0,
        settling_time: float = 0.0,
        t0: float = 0.0,
        grid_type: str = "verbatim",
        apply: bool = True,
        verbose: bool = True,
    ) -> CalibrationResult:
        """Race solver configurations and pick the fastest.

        Compare a range of integration algorithms and settings
        (preconditioners, solver types, smoothed error and
        prediction), returning a winner and a ranked list based on
        solve time and solver-failure count. Can run for up to an
        hour on large systems, but takes care of a lot of
        trial/error. Candidates inherit this solver's tolerances
        and output configuration.

        Parameters
        ----------
        initial_values
            A typical initial-values grid that you'll solve over;
            aim for at least 32768 combined initial-value/parameter
            sets to test the solver at full capacity. Accepts
            dictionaries mapping state names to values for grid
            construction, or pre-built arrays in (n_states, n_runs)
            format.
        parameters
            Parameter values for each run. Accepts dictionaries
            mapping parameter names to values, or pre-built arrays
            in (n_params, n_runs) format.
        drivers
            Driver samples or configuration matching
            :class:`cubie.array_interpolator.ArrayInterpolator`.
        duration
            Total integration time. Default is ``1.0``.
        settling_time
            Warm-up period before recording outputs. Default ``0.0``.
        t0
            Initial integration time. Default ``0.0``.
        grid_type
            Strategy for constructing the integration grid from
            inputs. Only used when dict inputs trigger grid
            construction.
        apply
            Apply the winner's configuration to this solver when
            ``True`` (default). Pass ``False`` to only report.
        verbose
            Print per-candidate progress lines. Default ``True``.

        Returns
        -------
        CalibrationResult
            Winner, ranking, and every candidate measurement. A
            candidate that fails to build or integrate is reported
            as dropped with its error message.

        Raises
        ------
        ValueError
            If the system declares drivers but none are supplied.
        """
        return run_calibration(
            self,
            initial_values,
            parameters,
            drivers=drivers,
            duration=duration,
            settling_time=settling_time,
            t0=t0,
            grid_type=grid_type,
            apply=apply,
            verbose=verbose,
        )

    def update(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> Set[str]:
        """Update solver, integrator, and system settings.

        Parameters
        ----------
        updates_dict
            Mapping of attribute names to new values.
        silent
            If ``True`` unknown keys are ignored instead of raising
            ``KeyError``.
        **kwargs
            Additional updates supplied as keyword arguments.

        Returns
        -------
        Set[str]
            Set of keys that were successfully updated.

        Raises
        ------
        KeyError
            If ``silent`` is ``False`` and unknown settings are supplied.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        _check_renamed_kwargs(updates_dict)
        self._update_record.append(dict(updates_dict))

        # Keep the recorded selection current for re-resolution.
        for key in _OUTPUT_SELECTION_KEYS:
            if updates_dict.get(key) is not None:
                self._output_selection_intent[key] = updates_dict[key]

        # Only convert output labels if variable-related keys are present
        variable_keys = {"save_variables", "summarise_variables"}
        if any(key in updates_dict for key in variable_keys):
            self.convert_output_labels(updates_dict)
            self._resolved_output_layout = self._output_layout()

        all_unrecognized = set(updates_dict.keys())
        all_unrecognized -= self.update_memory_settings(
            updates_dict, silent=True
        )
        all_unrecognized -= self.system_interface.update(
            updates_dict, silent=True
        )
        all_unrecognized -= self.kernel.update(updates_dict, silent=True)

        # Re-resolve the output selection if the layout changed.
        self._refresh_output_selection()

        recognised = set(updates_dict.keys()) - all_unrecognized
        if recognised:
            self._solve_info_key = None

        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        return recognised

    def update_memory_settings(
        self,
        updates_dict: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> Set[str]:
        """Update memory manager parameters.

        Parameters
        ----------
        updates_dict
            Mapping of memory manager settings to update.
        silent
            If ``True`` unknown keys are ignored instead of raising
            ``KeyError``.
        **kwargs
            Additional updates supplied as keyword arguments.

        Returns
        -------
        Set[str]
            Set of keys that were successfully updated.

        Raises
        ------
        KeyError
            If ``silent`` is ``False`` and unknown settings are supplied.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()
        all_unrecognized = set(updates_dict.keys())
        recognised = set()

        if "mem_proportion" in updates_dict:
            if updates_dict["mem_proportion"] is None:
                self.memory_manager.set_auto_limit_mode(self.kernel)
            else:
                self.memory_manager.set_manual_proportion(
                    self.kernel, updates_dict["mem_proportion"]
                )
            recognised.add("mem_proportion")

        recognised = set(recognised)
        all_unrecognized -= set(recognised)
        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        return recognised

    def get_state_indices(
        self, state_labels: Optional[List[str]] = None
    ) -> ndarray:
        """Return indices for the specified state variables.

        Parameters
        ----------
        state_labels
            Labels of states to query. ``None`` returns indices for all states.

        Returns
        -------
        ndarray
            Integer indices corresponding to the requested states.
        """
        return self.system_interface.state_indices(state_labels)

    def get_observable_indices(
        self, observable_labels: Optional[List[str]] = None
    ) -> ndarray:
        """Return indices for the specified observables.

        Parameters
        ----------
        observable_labels
            Labels of observables to query. ``None`` returns indices for all
            observables.

        Returns
        -------
        ndarray
            Integer indices corresponding to the requested observables.
        """
        return self.system_interface.observable_indices(observable_labels)

    @property
    def precision(self) -> PrecisionDType:
        """Expose the kernel precision."""
        return self.kernel.precision

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Expose output compile flags from the kernel."""
        return self.kernel.compile_flags

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Expose active outputs from the kernel."""
        return self.kernel.active_outputs

    @property
    def system_sizes(self):
        """Expose cached system size metadata."""
        return self.kernel.system_sizes

    @property
    def output_array_heights(self):
        """Expose output array heights from the kernel."""
        return self.kernel.output_array_heights

    @property
    def num_runs(self):
        """Expose the number of runs in the last solve."""
        return self.kernel.num_runs

    @property
    def output_length(self):
        """Expose the flattened output length."""
        return self.kernel.output_length

    @property
    def summaries_length(self):
        """Expose the flattened summary length."""
        return self.kernel.summaries_length

    @property
    def summary_legend_per_variable(self) -> dict[int, str]:
        """Expose summary legends keyed by variable index."""
        return self.kernel.summary_legend_per_variable

    @property
    def summary_unit_modifications(self) -> dict[int, str]:
        """Expose summary unit modifications keyed by variable index."""
        return self.kernel.summary_unit_modifications

    @property
    def saved_state_indices(self):
        """Expose saved state indices."""
        return self.kernel.saved_state_indices

    @property
    def saved_states(self):
        """List saved state labels."""
        return self.system_interface.state_labels(self.saved_state_indices)

    @property
    def saved_observable_indices(self):
        """Expose saved observable indices."""
        return self.kernel.saved_observable_indices

    @property
    def saved_observables(self):
        """List saved observable labels."""
        return self.system_interface.observable_labels(
            self.saved_observable_indices
        )

    @property
    def summarised_state_indices(self):
        """Expose summarised state indices."""
        return self.kernel.summarised_state_indices

    @property
    def summarised_states(self):
        """List summarised state labels."""
        return self.system_interface.state_labels(
            self.summarised_state_indices
        )

    @property
    def summarised_observable_indices(self):
        """Expose summarised observable indices."""
        return self.kernel.summarised_observable_indices

    @property
    def summarised_observables(self):
        """List summarised observable labels."""
        return self.system_interface.observable_labels(
            self.summarised_observable_indices
        )

    @property
    def state(self):
        """Expose latest state outputs."""
        return self.kernel.state

    @property
    def observables(self):
        """Expose latest observable outputs."""
        return self.kernel.observables

    @property
    def state_summaries(self):
        """Expose state summary outputs."""
        return self.kernel.state_summaries

    @property
    def observable_summaries(self):
        """Expose observable summary outputs."""
        return self.kernel.observable_summaries

    @property
    def iteration_counters(self):
        """Expose iteration counters at each save point."""
        return self.kernel.iteration_counters

    @property
    def status_codes(self):
        """Expose integration status codes."""
        return self.kernel.status_codes

    @property
    def device_state(self):
        """Expose the device buffer of state outputs."""
        return self.kernel.device_state

    @property
    def device_observables(self):
        """Expose the device buffer of observable outputs."""
        return self.kernel.device_observables

    @property
    def device_state_summaries(self):
        """Expose the device buffer of state summaries."""
        return self.kernel.device_state_summaries

    @property
    def device_observable_summaries(self):
        """Expose the device buffer of observable summaries."""
        return self.kernel.device_observable_summaries

    @property
    def device_status_codes(self):
        """Expose the device buffer of status codes."""
        return self.kernel.device_status_codes

    @property
    def device_iteration_counters(self):
        """Expose the device buffer of iteration counters."""
        return self.kernel.device_iteration_counters

    @property
    def status_messages(self):
        """Decode nonzero run status codes into named result flags.

        Returns
        -------
        dict[int, list[str]]
            Mapping from run index to the ``CUBIE_RESULT_CODES`` member
            names set in that run's status word; successful runs are
            omitted.
        """
        return decode_status_codes(self.status_codes)

    @property
    def parameters(self):
        """Expose parameter array used in the last run."""
        return self.kernel.parameters

    @property
    def initial_values(self):
        """Expose initial values array used in the last run."""
        return self.kernel.initial_values

    @property
    def device_initial_values(self):
        """Device initial values of the last run; raise if chunked."""
        return self.kernel.device_initial_values

    @property
    def device_parameters(self):
        """Device parameters of the last run; raise if chunked."""
        return self.kernel.device_parameters

    @property
    def driver_coefficients(self):
        """Expose driver interpolation coefficients."""
        return self.kernel.driver_coefficients

    @property
    def save_time(self) -> bool:
        """Return whether time points are saved."""
        return self.kernel.save_time

    @property
    def save_counters(self) -> bool:
        """Return whether iteration counters are saved."""
        return self.kernel.save_counters

    @property
    def output_types(self) -> List[str]:
        """List active output types."""
        return self.kernel.output_types

    @property
    def input_variables(self) -> List[str]:
        """List all input variable labels."""
        return self.system_interface.all_input_labels

    @property
    def output_variables(self) -> List[str]:
        """List all output variable labels."""
        return self.system_interface.all_output_labels

    @property
    def chunks(self):
        """Return the number of chunks used in the last run."""
        return self.kernel.chunks

    @property
    def memory_manager(self):
        """Return the associated memory manager instance."""
        return self.kernel.memory_manager

    @property
    def stream_group(self):
        """Return the CUDA stream group assigned to this solver."""
        return self.kernel.stream_group

    @property
    def stream(self):
        """Return the CUDA stream used by this solver."""
        return self.kernel.stream

    @property
    def mem_proportion(self):
        """Return the proportion of global memory allocated."""
        return self.kernel.mem_proportion

    @property
    def system(self) -> "BaseODE":
        """Return the underlying ODE system instance."""
        return self.kernel.system

    # Pass-through properties for solve_info components
    @property
    def dt(self) -> Optional[float]:
        """Return the fixed-step size or ``None`` for adaptive controllers."""
        return self.kernel.dt

    @property
    def dt_min(self) -> Optional[float]:
        """Return the minimum step size for adaptive controllers."""
        return self.kernel.dt_min

    @property
    def dt_max(self) -> Optional[float]:
        """Return the maximum step size for adaptive controllers."""
        return self.kernel.dt_max

    @property
    def save_every(self) -> Optional[float]:
        """Return the interval between saved time-domain outputs."""
        return self.kernel.save_every

    @property
    def summarise_every(self) -> Optional[float]:
        """Return the interval between summary computations."""
        return self.kernel.summarise_every

    @property
    def sample_summaries_every(self) -> Optional[float]:
        """Return the interval between summary metric samples."""
        return self.kernel.sample_summaries_every

    @property
    def duration(self):
        """Return the requested integration duration."""
        return self.kernel.duration

    @property
    def warmup(self):
        """Return the warm-up period length."""
        return self.kernel.warmup

    @property
    def t0(self) -> float:
        """Return the starting integration time."""

        return self.kernel.t0

    @property
    def atol(self) -> Optional[float]:
        """Return the absolute tolerance for adaptive controllers."""
        return self.kernel.atol

    @property
    def rtol(self) -> Optional[float]:
        """Return the relative tolerance for adaptive controllers."""
        return self.kernel.rtol

    @property
    def algorithm(self):
        """Return the configured algorithm name."""
        return self.kernel.algorithm

    @property
    def cache_enabled(self) -> bool:
        """Whether file-based caching is enabled."""
        return self.kernel.cache_policy.cache_enabled

    @property
    def cache_mode(self) -> str:
        """Current caching mode ('hash' or 'flush_on_change')."""
        return self.kernel.cache_policy.cache_mode

    @property
    def cache_dir(self) -> Optional[Path]:
        """Custom cache directory, or None for default location."""
        return self.kernel.cache_policy.cache_dir

    def set_cache_dir(self, path: Union[str, Path]) -> None:
        """Set a custom cache directory for compiled kernels.

        Parameters
        ----------
        path
            New cache directory path. Can be absolute or relative.

        Notes
        -----
        Invalidates the current cache, causing a rebuild on next access.
        """
        self.kernel.set_cache_dir(path)

    def set_verbosity(self, verbosity: Optional[str]) -> None:
        """Set the time logging verbosity level.

        Parameters
        ----------
        verbosity : str or None
            New verbosity level. Options are 'default', 'verbose',
            'debug', None, or 'None'.

        Notes
        -----
        Updates the global time logger verbosity. This affects all
        timing events across the entire CuBIE package.
        """
        default_timelogger.set_verbosity(verbosity)

    @property
    def solve_info(self) -> SolveSpec:
        """SolveSpec for the current settings, cached until they change."""
        key = (self.duration, self.warmup, self.t0)
        if self._solve_info_key == key:
            return self._solve_info_cache
        spec = SolveSpec(
            dt=self.dt,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            save_every=self.save_every,
            summarise_every=self.summarise_every,
            sample_summaries_every=self.sample_summaries_every,
            duration=self.duration,
            warmup=self.warmup,
            t0=self.t0,
            atol=self.atol,
            rtol=self.rtol,
            algorithm=self.algorithm,
            saved_states=self.saved_states,
            saved_observables=self.saved_observables,
            summarised_states=self.summarised_states,
            summarised_observables=self.summarised_observables,
            output_types=self.output_types,
            precision=self.precision,
        )
        self._solve_info_cache = spec
        self._solve_info_key = key
        return spec
