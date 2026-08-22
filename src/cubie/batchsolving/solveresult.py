"""Batch solver run specifications and result containers.

This module exposes :class:`SolveSpec` to describe solver configuration
and :class:`SolveResult` to own the output arrays, legends, and metadata
of one completed batch integration.

Published Classes
-----------------
:class:`SolveSpec`
    Frozen attrs dataclass describing the configuration used for a solver run.

:class:`SolveResult`
    Owns the solve's host output buffers and derives every user-facing
    representation from them lazily.

:class:`DeviceSolveResult`
    Holds device-array handles to the solve's output buffers plus the
    CUDA stream the solve ran on; nothing is copied to the host.

Ownership
---------
A :class:`SolveResult` takes ownership of the host buffers the solve
wrote — nothing is copied. Keep the result object alive for as long as
its data is needed: once it is garbage collected, the solver reuses
the buffers on its next run. Arrays extracted from a result are only
valid while the result lives. Disk-backed (spilled) results release
their files on :meth:`SolveResult.close`, on context-manager exit, or
at garbage collection.

See Also
--------
:class:`~cubie.batchsolving.solver.Solver`
    User-facing solver whose :meth:`~Solver.solve` returns a
    :class:`SolveResult`.
:func:`~cubie.batchsolving.solver.solve_ivp`
    Convenience wrapper returning a :class:`SolveResult`.
"""

from typing import Dict, Optional, TYPE_CHECKING, Union, List, Any, Tuple
from warnings import warn

if TYPE_CHECKING:
    from cubie.batchsolving.solver import Solver
    import pandas as pd

from attrs import (
    cmp_using as attrs_cmp_using,
    define,
    Factory as attrsFactory,
    field,
)
from attrs.validators import (
    instance_of as attrsval_instance_of,
    optional as attrsval_optional,
    or_ as attrsval_or,
)
from numpy import (
    amax as np_amax,
    array as np_array,
    array_equal as np_array_equal,
    concatenate as np_concatenate,
    nan as np_nan,
    ndarray,
    squeeze as np_squeeze,
    where as np_where,
)
from numpy.typing import NDArray
from cubie.batchsolving.BatchSolverConfig import ActiveOutputs
from cubie.batchsolving import ArrayTypes
from cubie.cuda_simsafe import DeviceNDArrayBase, Stream
from cubie.result_codes import decode_status_codes
from cubie._utils import (
    slice_variable_dimension,
    opt_gttype_validator,
    opt_getype_validator,
    getype_validator,
    gttype_validator,
    PrecisionDType,
)


#: Correction types ordered from least to most robust.
_CORRECTION_ROBUSTNESS = (
    "steepest_descent",
    "minimal_residual",
    "bicgstab",
    "lu",
)


def _failure_remedies(diagnostics: Dict[str, Any]) -> List[str]:
    """Return the settings changes that make failed runs less likely."""
    remedies = []
    if diagnostics.get("inexact_newton"):
        remedies.append("inexact_newton=False")
    if diagnostics.get("smoothed_error_capable") and not diagnostics.get(
        "use_smoothed_error"
    ):
        remedies.append("use_smoothed_error=True")
    correction = diagnostics.get("linear_correction_type")
    if correction in _CORRECTION_ROBUSTNESS:
        tougher = _CORRECTION_ROBUSTNESS[
            _CORRECTION_ROBUSTNESS.index(correction) + 1:
        ]
        if tougher:
            options = " or ".join(repr(name) for name in tougher)
            remedies.append(f"linear_correction_type={options}")
    return remedies


def _failed_run_message(
    status_codes: NDArray,
    diagnostics: Dict[str, Any],
    masked: bool,
) -> str:
    """Describe failed runs, the solver settings, and what to change."""
    failures = decode_status_codes(status_codes)
    counts = {}
    for flags in failures.values():
        for flag in flags:
            counts[flag] = counts.get(flag, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    flag_text = ", ".join(f"{name} ({count})" for name, count in ranked)
    outcome = " and were set to NaN" if masked else ""
    lines = [
        f"{len(failures)} of {len(status_codes)} runs failed{outcome}.",
    ]
    if flag_text:
        lines.append(f"Result codes: {flag_text}.")
    if not masked:
        lines.append(
            "Check status_codes for the failing runs, or pass "
            "nan_error_trajectories=True to mask them with NaN."
        )
    if diagnostics:
        settings = ", ".join(
            f"{key}={value!r}" for key, value in diagnostics.items()
        )
        lines.append(f"Solver settings: {settings}.")
    remedies = _failure_remedies(diagnostics)
    if remedies:
        lines.append(f"Try: {'; '.join(remedies)}.")
    return " ".join(lines)


def _format_time_domain_label(label: str, unit: str) -> str:
    """Format a time-domain legend label with unit if not dimensionless.

    Parameters
    ----------
    label
        Variable label.
    unit
        Unit string for the variable.

    Returns
    -------
    str
        Formatted label. If unit is "dimensionless", returns just the label.
        Otherwise returns "label [unit]".
    """
    if unit != "dimensionless":
        return f"{label} [{unit}]"
    return label


def _release_spill_arrays(memory_manager, arrays) -> None:
    """Release each spill mapping once."""
    cleanups = set()
    for array in arrays:
        owner = array
        while isinstance(owner, ndarray):
            cleanup = getattr(owner, "_cubie_spill_cleanup", None)
            if cleanup is not None and id(cleanup) not in cleanups:
                cleanups.add(id(cleanup))
                memory_manager.release_host_array(owner)
            owner = getattr(owner, "base", None)


@define
class SolveSpec:
    """Describe the configuration of a solver run.

    Attributes
    ----------
    dt
        Fixed step size, or ``None`` for adaptive controllers.
    dt_min
        Minimum time step size.
    dt_max
        Maximum time step size.
    save_every
        Interval at which state values are stored.
    summarise_every
        Interval for computing summary outputs.
    sample_summaries_every
        Interval for sampling summary metric updates.
    atol
        Absolute error tolerance when configured.
    rtol
        Relative error tolerance when configured.
    duration
        Total integration time.
    warmup
        Initial warm-up period prior to recording outputs.
    t0
        Initial integration time supplied to the solver.
    algorithm
        Name of the integration algorithm.
    saved_states
        Labels of states saved verbatim or ``None`` when disabled.
    saved_observables
        Labels of observables saved verbatim or ``None`` when disabled.
    summarised_states
        Labels of states with summaries computed or ``None`` when disabled.
    summarised_observables
        Labels of observables with summaries computed or ``None`` when
        disabled.
    output_types
        Types of output arrays generated during the run or ``None``.
    precision
        Floating-point precision factory used for host conversions.
    """

    dt: Optional[float] = field(validator=opt_gttype_validator(float, 0.0))
    dt_min: float = field(validator=gttype_validator(float, 0.0))
    dt_max: float = field(validator=gttype_validator(float, 0.0))
    save_every: Optional[float] = field(
        validator=opt_gttype_validator(float, 0.0)
    )
    summarise_every: Optional[float] = field(
        validator=opt_getype_validator(float, 0.0)
    )
    sample_summaries_every: Optional[float] = field(
        validator=opt_getype_validator(float, 0.0)
    )
    # ndarray first: a failing float validator reprs the whole array.
    atol: Optional[float] = field(
        validator=attrsval_or(
            attrsval_instance_of(ndarray), opt_gttype_validator(float, 0.0)
        ),
    )
    rtol: Optional[float] = field(
        validator=attrsval_or(
            attrsval_instance_of(ndarray), opt_gttype_validator(float, 0.0)
        ),
    )
    duration: float = field(validator=gttype_validator(float, 0.0))
    warmup: float = field(validator=getype_validator(float, 0.0))
    t0: float = field(validator=getype_validator(float, float("-inf")))
    algorithm: str = field(validator=attrsval_instance_of(str))
    saved_states: Optional[List[str]] = field()
    saved_observables: Optional[List[str]] = field()
    summarised_states: Optional[List[str]] = field()
    summarised_observables: Optional[List[str]] = field()
    output_types: Optional[List[str]] = field()
    precision: PrecisionDType = field()


@define
class SolveResult:
    """Own the host outputs of one solve and derive views on demand.

    The result takes the solve's host buffers wholesale — no copies.
    ``state`` (with its time column when time is saved),
    ``observables``, the summary buffers, ``status_codes``, and
    ``iteration_counters`` are the arrays the kernel wrote. Combined
    representations (:attr:`time_domain_array`,
    :attr:`summaries_array`) and the RAM materialisations
    (:attr:`as_numpy`, :attr:`as_pandas`,
    :attr:`as_numpy_per_summary`) are built lazily on first access.

    Keep the result object alive while its data is needed: once it is
    garbage collected the solver reuses the buffers on its next run,
    so arrays extracted from a dead result are not safe to read.
    Disk-backed (spilled) buffers release their files on
    :meth:`close`, context exit, or collection.

    Parameters
    ----------
    state
        State output buffer, including the trailing time column when
        time saving is enabled.
    observables
        Observable output buffer.
    state_summaries
        State summary buffer.
    observable_summaries
        Observable summary buffer.
    status_codes
        Per-run status codes, shape ``(n_runs,)``, dtype int32.
    iteration_counters
        Per-save iteration counters when requested.
    time_domain_legend
        Mapping from time-domain indices to labels.
    summaries_legend
        Mapping from summary indices to labels.
    solve_settings
        Solver run configuration snapshot.
    stream
        The kernel's memory-manager stream the solve ran on. Work
        queued on this stream executes in order after the solve's
        kernel launches and transfers.
    singlevar_summary_legend
        Mapping from summary offsets to legend labels.
    active_outputs
        :class:`ActiveOutputs` flags describing enabled arrays.
    stride_order
        Order of axes in the host arrays.
    save_time
        Whether the state buffer carries a trailing time column.
    memory_manager
        Manager used to release disk-backed buffers.
    """

    _state: Optional[NDArray] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(ndarray)),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    _observables: Optional[NDArray] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(ndarray)),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    _state_summaries: Optional[NDArray] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(ndarray)),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    _observable_summaries: Optional[NDArray] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(ndarray)),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    status_codes: Optional[NDArray] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(ndarray)),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    _iteration_counters: Optional[NDArray] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(ndarray)),
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    time_domain_legend: Optional[dict[int, str]] = field(
        default=attrsFactory(dict),
        validator=attrsval_optional(attrsval_instance_of(dict)),
    )
    summaries_legend: Optional[dict[int, str]] = field(
        default=attrsFactory(dict),
        validator=attrsval_optional(attrsval_instance_of(dict)),
    )
    solve_settings: Optional[SolveSpec] = field(
        default=None,
        validator=attrsval_optional(attrsval_instance_of(SolveSpec)),
    )
    # A backend Stream on real hardware, the simulator's stream object
    # under CUDASIM, or 0 for the legacy default stream — an isinstance
    # validator cannot cover all three, so the field is unvalidated.
    stream: Optional[Union[Stream, int]] = field(default=None, eq=False)
    _singlevar_summary_legend: Optional[dict[int, str]] = field(
        default=attrsFactory(dict),
        validator=attrsval_optional(attrsval_instance_of(dict)),
    )
    _active_outputs: Optional[ActiveOutputs] = field(
        default=attrsFactory(ActiveOutputs)
    )
    _stride_order: Union[tuple[str, ...], list[str]] = field(
        default=("time", "variable", "run")
    )
    _save_time: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )
    _memory_manager: Optional[Any] = field(
        default=None, repr=False, eq=False
    )
    _closed: bool = field(default=False, init=False, repr=False, eq=False)
    _time_domain_cache: Optional[NDArray] = field(
        default=None, init=False, repr=False, eq=False
    )
    _summaries_cache: Optional[NDArray] = field(
        default=None, init=False, repr=False, eq=False
    )

    @classmethod
    def from_solver(
        cls,
        solver: "Solver",
        nan_error_trajectories: bool = True,
    ) -> "SolveResult":
        """Create a :class:`SolveResult` owning the solver's buffers.

        The solver's host output buffers are handed to the result
        without copying. The solver reuses them on its next run only
        after the result has been garbage collected; while the result
        lives, the next run allocates fresh backing.

        Parameters
        ----------
        solver
            Object providing access to output arrays and metadata.
        nan_error_trajectories
            When ``True`` (default), trajectories with nonzero status
            codes are overwritten with NaN in place, making failed
            runs easy to identify and exclude from analysis. Failed
            runs raise a warning whether or not they are masked.

        Returns
        -------
        SolveResult
            Result owning the solve's host buffers.
        """
        outputs = solver.kernel.output_arrays
        # Buffers loaned to an already-collected result come back to
        # their slots here, so a second from_solver call after the
        # first result died hands over the same solve's data.
        outputs.reclaim_or_release_loan()
        result = cls(
            state=outputs.state,
            observables=outputs.observables,
            state_summaries=outputs.state_summaries,
            observable_summaries=outputs.observable_summaries,
            status_codes=outputs.status_codes,
            iteration_counters=outputs.iteration_counters,
            time_domain_legend=cls.time_domain_legend_from_solver(solver),
            summaries_legend=cls.summary_legend_from_solver(solver),
            solve_settings=solver.solve_info,
            stream=solver.kernel.stream,
            singlevar_summary_legend=solver.summary_legend_per_variable,
            active_outputs=solver.active_outputs,
            stride_order=outputs.host.state.stride_order,
            save_time=bool(solver.save_time),
            memory_manager=solver.kernel.memory_manager,
        )
        outputs.loan_host_arrays(result)
        error_runs = result._error_run_indices()
        if error_runs.size > 0:
            if nan_error_trajectories:
                result._mask_error_runs(error_runs)
            warn(
                _failed_run_message(
                    result.status_codes,
                    solver.kernel.single_integrator.solver_diagnostics,
                    masked=nan_error_trajectories,
                )
            )
        return result

    def _error_run_indices(self) -> NDArray:
        """Return the indices of runs with nonzero status codes."""
        codes = self.status_codes
        if codes is None or codes.size == 0:
            return np_array([], dtype=int)
        return np_where(codes != 0)[0]

    def _mask_error_runs(self, error_runs: NDArray) -> None:
        """NaN the given runs, keeping the state buffer's time column."""
        run_index = self._stride_order.index("run")
        targets = []
        if self._active_outputs.state:
            targets.append(self._state_less_time)
        if self._active_outputs.observables:
            targets.append(self._observables)
        if self._active_outputs.state_summaries:
            targets.append(self._state_summaries)
        if self._active_outputs.observable_summaries:
            targets.append(self._observable_summaries)
        for array in targets:
            if array is None or array.size == 0:
                continue
            if run_index == 0:
                array[error_runs, :, :] = np_nan
            elif run_index == 1:
                array[:, error_runs, :] = np_nan
            else:
                array[:, :, error_runs] = np_nan

    @property
    def _state_less_time(self) -> Optional[NDArray]:
        """State buffer view without the trailing time column."""
        if self._state is None:
            return None
        if not self._save_time:
            return self._state
        var_index = self._stride_order.index("variable")
        state_slice = slice_variable_dimension(
            slice(None, -1), var_index, self._state.ndim
        )
        return self._state[state_slice]

    @property
    def state(self) -> Optional[NDArray]:
        """State output buffer, with its time column when time is saved."""
        if self._active_outputs.state or self._save_time:
            return self._state
        return None

    @property
    def observables(self) -> Optional[NDArray]:
        """Observable output buffer, or ``None`` when not saved."""
        if self._active_outputs.observables:
            return self._observables
        return None

    @property
    def state_summaries(self) -> Optional[NDArray]:
        """State summary buffer, or ``None`` when not summarised."""
        if self._active_outputs.state_summaries:
            return self._state_summaries
        return None

    @property
    def observable_summaries(self) -> Optional[NDArray]:
        """Observable summary buffer, or ``None`` when not summarised."""
        if self._active_outputs.observable_summaries:
            return self._observable_summaries
        return None

    @property
    def iteration_counters(self) -> Optional[NDArray]:
        """Iteration counters, or ``None`` when not requested."""
        if self._active_outputs.iteration_counters:
            return self._iteration_counters
        return None

    @property
    def time(self) -> Optional[NDArray]:
        """Time samples cleaved from the state buffer, or ``None``."""
        if not self._save_time or self._state is None:
            return None
        var_index = self._stride_order.index("variable")
        time_slice = slice_variable_dimension(
            slice(-1, None, None), var_index, self._state.ndim
        )
        return np_squeeze(self._state[time_slice], axis=var_index)

    @property
    def time_domain_array(self) -> NDArray:
        """Combined time-domain outputs (states then observables).

        A single active source is returned as a view of the owned
        buffer — no copy, no RAM beyond what the solve already used.
        Two active sources concatenate into RAM on first access.
        """
        if self._time_domain_cache is None:
            self._time_domain_cache = self._combine_active(
                (
                    self._state_less_time
                    if self._active_outputs.state
                    else None
                ),
                (
                    self._observables
                    if self._active_outputs.observables
                    else None
                ),
            )
        return self._time_domain_cache

    @property
    def summaries_array(self) -> NDArray:
        """Combined summary outputs (states then observables)."""
        if self._summaries_cache is None:
            self._summaries_cache = self._combine_active(
                (
                    self._state_summaries
                    if self._active_outputs.state_summaries
                    else None
                ),
                (
                    self._observable_summaries
                    if self._active_outputs.observable_summaries
                    else None
                ),
            )
        return self._summaries_cache

    @staticmethod
    def _combine_active(
        first: Optional[NDArray], second: Optional[NDArray]
    ) -> NDArray:
        """Combine up to two active buffers along the variable axis."""
        active = [
            array
            for array in (first, second)
            if array is not None and array.size > 0
        ]
        if not active:
            return np_array([])
        if len(active) == 1:
            return active[0]
        return np_concatenate(active, axis=1)

    def close(self) -> None:
        """Release spill files owned by this result and drop its data."""
        if self._closed:
            return
        arrays = (
            self._state,
            self._observables,
            self._state_summaries,
            self._observable_summaries,
            self.status_codes,
            self._iteration_counters,
            self._time_domain_cache,
            self._summaries_cache,
        )
        if self._memory_manager is not None:
            _release_spill_arrays(self._memory_manager, arrays)
        self._state = None
        self._observables = None
        self._state_summaries = None
        self._observable_summaries = None
        self.status_codes = None
        self._iteration_counters = None
        self._time_domain_cache = None
        self._summaries_cache = None
        self._closed = True

    def __enter__(self) -> "SolveResult":
        """Return this result."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Release spill files on context exit."""
        self.close()

    @property
    def as_pandas(self) -> dict[str, "pd.DataFrame"]:
        """Convert the results to pandas DataFrames.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Dictionary containing ``time_domain`` and ``summaries`` DataFrames.

        Raises
        ------
        ImportError
            Raised when pandas is not available.

        Notes
        -----
        Pandas is an optional dependency that is imported lazily.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required to convert SolveResult to DataFrames. "
                "Pandas is an optional dependency- it's only used here "
                "to make analysis of data easier. Install Pandas to "
                "use this feature."
            )

        run_index = self._stride_order.index("run")
        ndim = len(self._stride_order)
        time_dfs = []
        summaries_dfs = []
        any_summaries = (
            self.active_outputs.state_summaries
            or self.active_outputs.observable_summaries
        )

        n_runs = self.time_domain_array.shape[run_index] if ndim == 3 else 1
        time_headings = list(self.time_domain_legend.values())
        summary_headings = list(self.summaries_legend.values())

        # Resolve time index once (use first run's time for multi-run)
        time_index = None
        if self.time is not None and self.time.size > 0:
            # time is (n_saves, n_runs); every run shares the save
            # schedule, so the first run's samples index the frame.
            time_index = np_array(self.time[:, 0], copy=True, subok=False)

        for run in range(n_runs):
            run_slice = slice_variable_dimension(
                slice(run, run + 1, None), run_index, ndim
            )

            singlerun_array = np_array(
                np_squeeze(
                    self.time_domain_array[run_slice], axis=run_index
                ),
                copy=True,
                subok=False,
            )
            df = pd.DataFrame(singlerun_array, columns=time_headings)

            # Create MultiIndex columns with run number as first level
            df.columns = pd.MultiIndex.from_product(
                [[f"run_{run}"], df.columns]
            )
            time_dfs.append(df)

            if any_summaries:
                singlerun_array = np_array(
                    np_squeeze(
                        self.summaries_array[run_slice], axis=run_index
                    ),
                    copy=True,
                    subok=False,
                )
                df = pd.DataFrame(singlerun_array, columns=summary_headings)
                summaries_dfs.append(df)
                df.columns = pd.MultiIndex.from_product(
                    [[f"run_{run}"], df.columns]
                )
            else:
                summaries_dfs.append(pd.DataFrame())

        time_domain_df = pd.concat(time_dfs, axis=1)
        summaries_df = pd.concat(summaries_dfs, axis=1)

        # Set time index after concat to avoid reindexing errors from
        # float32 duplicate values during alignment
        if time_index is not None:
            time_domain_df.index = time_index

        return {"time_domain": time_domain_df, "summaries": summaries_df}

    @property
    def as_numpy(self) -> dict[str, Optional[NDArray]]:
        """
        Return the results as in-RAM copies of NumPy arrays.

        Returns
        -------
        dict[str, Optional[NDArray]]
            Dictionary containing copies of time, time_domain_array,
            summaries_array, time_domain_legend, summaries_legend, and
            iteration_counters.
        """
        counters = self.iteration_counters
        return {
            "time": (
                np_array(self.time, copy=True, subok=False)
                if self.time is not None
                else None
            ),
            "time_domain_array": np_array(
                self.time_domain_array, copy=True, subok=False
            ),
            "summaries_array": np_array(
                self.summaries_array, copy=True, subok=False
            ),
            "time_domain_legend": self.time_domain_legend.copy(),
            "summaries_legend": self.summaries_legend.copy(),
            "iteration_counters": (
                np_array(counters, copy=True, subok=False)
                if counters is not None
                else None
            ),
        }

    @property
    def as_numpy_per_summary(self) -> dict[str, Optional[NDArray]]:
        """
        Return the results as separate NumPy arrays per summary type.

        Returns
        -------
        dict[str, Optional[NDArray]]
            Dictionary containing time, time_domain_array, time_domain_legend,
            iteration counters, and individual summary arrays.
        """
        counters = self.iteration_counters
        arrays = {
            "time": (
                np_array(self.time, copy=True, subok=False)
                if self.time is not None
                else None
            ),
            "time_domain_array": np_array(
                self.time_domain_array, copy=True, subok=False
            ),
            "time_domain_legend": self.time_domain_legend.copy(),
            "iteration_counters": (
                np_array(counters, copy=True, subok=False)
                if counters is not None
                else None
            ),
        }
        arrays.update(**self.per_summary_arrays)

        return arrays

    @property
    def per_summary_arrays(self) -> dict[str, NDArray]:
        """
        Split summaries_array into separate arrays keyed by summary type.

        Returns
        -------
        dict[str, NDArray]
            Dictionary where each key is a summary type and the value is the
            corresponding NumPy array. The dictionary also includes a key
            'summary_legend' mapping to the variable legend.
        """
        if (
            self._active_outputs.state_summaries is False
            and self._active_outputs.observable_summaries is False
        ):
            return {}

        variable_index = self._stride_order.index("variable")

        # Split summaries_array by type
        variable_legend = self.time_domain_legend
        singlevar_legend = self._singlevar_summary_legend
        indices_per_var = np_amax([k for k in singlevar_legend.keys()]) + 1
        per_summary_arrays = {}

        for offset, label in singlevar_legend.items():
            summ_slice = slice(offset, None, indices_per_var)
            summ_slice = slice_variable_dimension(
                summ_slice, variable_index, len(self._stride_order)
            )
            per_summary_arrays[label] = np_array(
                self.summaries_array[summ_slice], copy=True, subok=False
            )
        per_summary_arrays["summary_legend"] = variable_legend

        return per_summary_arrays

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Return the active output flags."""
        return self._active_outputs

    @property
    def status_messages(self) -> dict[int, List[str]]:
        """Decode nonzero run status codes into named result flags.

        Returns
        -------
        dict[int, list[str]]
            Mapping from run index to the list of
            :class:`~cubie.result_codes.CUBIE_RESULT_CODES` member names set
            in that run's status word. Runs that completed successfully
            (status ``0``) are omitted, so an empty mapping means every run
            succeeded.
        """
        return decode_status_codes(self.status_codes)

    @staticmethod
    def cleave_time(
        state: ArrayTypes,
        time_saved: bool = False,
        stride_order: Optional[Tuple[str, ...]] = None,
    ) -> tuple[Optional[NDArray], NDArray]:
        """Remove time from the state array when present.

        Parameters
        ----------
        state
            State array potentially containing a time column.
        time_saved
            Flag indicating if time is saved in the state array.
        stride_order
            Optional order of dimensions in the array. Defaults to
            ``["time", "variable", "run"]`` when ``None``.

        Returns
        -------
        tuple[Optional[NDArray], NDArray]
            Pair containing the time array (or ``None``) and the state array
            with time removed.
        """
        if stride_order is None:
            stride_order = ["time", "variable", "run"]
        if time_saved:
            var_index = stride_order.index("variable")
            ndim = len(state.shape)

            time_slice = slice_variable_dimension(
                slice(-1, None, None), var_index, ndim
            )
            state_slice = slice_variable_dimension(
                slice(None, -1), var_index, ndim
            )

            time = np_squeeze(state[time_slice], axis=var_index)
            state_less_time = state[state_slice]
            return time, state_less_time
        else:
            return None, state

    @staticmethod
    def summary_legend_from_solver(solver: "Solver") -> dict[int, str]:
        """Generate a summary legend from the solver instance.

        Parameters
        ----------
        solver
            Solver instance providing saved states, observables, and summary
            legends.

        Returns
        -------
        dict[int, str]
            Dictionary mapping summary array indices to labels with units.
        """
        singlevar_legend = solver.summary_legend_per_variable
        unit_modifications = solver.summary_unit_modifications
        state_labels = solver.summarised_states
        obs_labels = solver.summarised_observables
        summaries_legend = {}

        state_units = {}
        obs_units = {}

        if hasattr(solver.system, "state_units"):
            state_units = solver.system.state_units
        if hasattr(solver.system, "observable_units"):
            obs_units = solver.system.observable_units

        # state summaries_array
        for i, label in enumerate(state_labels):
            unit = state_units.get(label, "dimensionless")
            for j, (key, summary_type) in enumerate(singlevar_legend.items()):
                index = i * len(singlevar_legend) + j
                unit_mod = unit_modifications.get(j, "[unit]")

                # Apply unit modification and format legend
                if unit != "dimensionless":
                    # Replace 'unit' placeholder (not '[unit]') to preserve
                    # brackets
                    modified_unit = unit_mod.replace("unit", unit)
                    summaries_legend[index] = (
                        f"{label} {modified_unit} {summary_type}"
                    )
                else:
                    summaries_legend[index] = f"{label} {summary_type}"

        # observable summaries_array
        len_state_legend = len(state_labels) * len(singlevar_legend)
        for i, label in enumerate(obs_labels):
            unit = obs_units.get(label, "dimensionless")
            for j, (key, summary_type) in enumerate(singlevar_legend.items()):
                index = len_state_legend + i * len(singlevar_legend) + j
                unit_mod = unit_modifications.get(j, "[unit]")

                # Apply unit modification and format legend
                if unit != "dimensionless":
                    # Replace 'unit' placeholder (not '[unit]') to preserve
                    # brackets
                    modified_unit = unit_mod.replace("unit", unit)
                    summaries_legend[index] = (
                        f"{label} {modified_unit} {summary_type}"
                    )
                else:
                    summaries_legend[index] = f"{label} {summary_type}"

        return summaries_legend

    @staticmethod
    def time_domain_legend_from_solver(solver: "Solver") -> dict[int, str]:
        """Generate a time-domain legend from the solver instance.

        Parameters
        ----------
        solver
            Solver instance providing saved states and observables.

        Returns
        -------
        dict[int, str]
            Dictionary mapping time-domain indices to labels with units.
        """
        time_domain_legend = {}
        state_labels = solver.saved_states
        obs_labels = solver.saved_observables

        state_units = {}
        obs_units = {}

        if hasattr(solver.system, "state_units"):
            state_units = solver.system.state_units
        if hasattr(solver.system, "observable_units"):
            obs_units = solver.system.observable_units

        offset = 0

        for i, label in enumerate(state_labels):
            unit = state_units.get(label, "dimensionless")
            time_domain_legend[i] = _format_time_domain_label(label, unit)

        offset = len(state_labels)
        for i, label in enumerate(obs_labels):
            unit = obs_units.get(label, "dimensionless")
            time_domain_legend[offset + i] = _format_time_domain_label(
                label, unit
            )
        return time_domain_legend


@define(eq=False)
class DeviceSolveResult:
    """Device-array handles to one solve's output buffers.

    Returned by :meth:`Solver.solve` when ``on_device=True``. Nothing
    is copied to the host: the fields are the solver's device output
    buffers plus the kernel's memory-manager stream — the stream
    every launch and transfer for that solver runs on. The solve does
    not synchronize before returning — buffer contents are valid once
    :attr:`stream` has been synchronized, and work queued on that
    stream executes in order after the solve.

    This class holds handles only; it performs no stream or memory
    operations. Callers queue follow-up device work on
    :attr:`stream`, and read on the host by synchronizing that stream
    first (or run a normal host solve instead).

    The handles are views into the solver's working buffers: the next
    ``solve()`` on the same solver overwrites their contents, and a
    reallocation or memory-pressure eviction detaches them from the
    solver. Copy anything that must outlive the next solve.

    Attributes
    ----------
    status_codes
        Device buffer of per-run integration status codes.
    stream
        The kernel's memory-manager stream the solve ran on.
    """

    _state: Optional[DeviceNDArrayBase] = field(
        default=None,
        validator=attrsval_optional(
            attrsval_instance_of(DeviceNDArrayBase)
        ),
    )
    _observables: Optional[DeviceNDArrayBase] = field(
        default=None,
        validator=attrsval_optional(
            attrsval_instance_of(DeviceNDArrayBase)
        ),
    )
    _state_summaries: Optional[DeviceNDArrayBase] = field(
        default=None,
        validator=attrsval_optional(
            attrsval_instance_of(DeviceNDArrayBase)
        ),
    )
    _observable_summaries: Optional[DeviceNDArrayBase] = field(
        default=None,
        validator=attrsval_optional(
            attrsval_instance_of(DeviceNDArrayBase)
        ),
    )
    _iteration_counters: Optional[DeviceNDArrayBase] = field(
        default=None,
        validator=attrsval_optional(
            attrsval_instance_of(DeviceNDArrayBase)
        ),
    )
    status_codes: Optional[DeviceNDArrayBase] = field(
        default=None,
        validator=attrsval_optional(
            attrsval_instance_of(DeviceNDArrayBase)
        ),
    )
    # See SolveResult.stream for why this field is unvalidated.
    stream: Optional[Union[Stream, int]] = field(default=None)
    _active_outputs: Optional[ActiveOutputs] = field(
        default=attrsFactory(ActiveOutputs)
    )
    _stride_order: Union[tuple[str, ...], list[str]] = field(
        default=("time", "variable", "run")
    )
    _save_time: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )

    @classmethod
    def from_solver(cls, solver: "Solver") -> "DeviceSolveResult":
        """Create a :class:`DeviceSolveResult` from the solver's buffers.

        Parameters
        ----------
        solver
            Solver whose kernel ran the solve.

        Returns
        -------
        DeviceSolveResult
            Handles to the solver's device output buffers and the
            stream the solve ran on.
        """
        kernel = solver.kernel
        return cls(
            state=kernel.device_state,
            observables=kernel.device_observables,
            state_summaries=kernel.device_state_summaries,
            observable_summaries=kernel.device_observable_summaries,
            iteration_counters=kernel.device_iteration_counters,
            status_codes=kernel.device_status_codes,
            stream=kernel.stream,
            active_outputs=solver.active_outputs,
            stride_order=kernel.output_arrays.host.state.stride_order,
            save_time=bool(solver.save_time),
        )

    @property
    def state(self) -> Optional[DeviceNDArrayBase]:
        """Device state buffer, with its time column when time is saved."""
        if self._active_outputs.state or self._save_time:
            return self._state
        return None

    @property
    def observables(self) -> Optional[DeviceNDArrayBase]:
        """Device observables buffer, or ``None`` when not saved."""
        if self._active_outputs.observables:
            return self._observables
        return None

    @property
    def state_summaries(self) -> Optional[DeviceNDArrayBase]:
        """Device state summary buffer, or ``None`` when not summarised."""
        if self._active_outputs.state_summaries:
            return self._state_summaries
        return None

    @property
    def observable_summaries(self) -> Optional[DeviceNDArrayBase]:
        """Device observable summary buffer, or ``None`` when not
        summarised."""
        if self._active_outputs.observable_summaries:
            return self._observable_summaries
        return None

    @property
    def iteration_counters(self) -> Optional[DeviceNDArrayBase]:
        """Device iteration counters, or ``None`` when not requested."""
        if self._active_outputs.iteration_counters:
            return self._iteration_counters
        return None

    @property
    def active_outputs(self) -> ActiveOutputs:
        """Return the active output flags."""
        return self._active_outputs

    @property
    def stride_order(self) -> Union[tuple[str, ...], list[str]]:
        """Order of axes in the device buffers."""
        return self._stride_order

    @property
    def save_time(self) -> bool:
        """Whether the state buffer carries a trailing time column."""
        return self._save_time
