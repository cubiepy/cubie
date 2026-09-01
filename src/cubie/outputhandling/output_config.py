"""Output configuration for solver output selection and validation.

Published Classes
-----------------
:class:`OutputCompileFlags`
    Boolean compile-time controls for CUDA output features.

    >>> flags = OutputCompileFlags(save_state=True, summarise=True)
    >>> flags.save_state
    True

:class:`OutputConfig`
    Validated configuration for solver outputs and summaries.

    >>> from numpy import float32, array, int_
    >>> config = OutputConfig(
    ...     max_states=3, max_observables=2,
    ...     output_types=["state"], precision=float32,
    ... )
    >>> config.save_state
    True

Module-Level Functions
----------------------
:func:`_indices_validator`
    Validate index arrays for bounds and duplication (internal).

See Also
--------
:class:`~cubie.outputhandling.output_functions.OutputFunctions`
    Factory that compiles CUDA callables from this configuration.
:class:`~cubie.CUDAFactory.CUDAFactoryConfig`
    Parent class providing precision and hashing.
"""

from typing import Any, List, Tuple, Union, Optional, Sequence
from warnings import warn

from attrs import (
    cmp_using as attrs_cmp_using,
    Factory as attrsFactory,
    field,
    frozen,
)
from attrs.validators import instance_of as attrsval_instance_of
from numpy import (
    any as np_any,
    array as np_array,
    array_equal as np_array_equal,
    asarray as np_asarray,
    int32 as np_int32,
    ndarray,
    unique as np_unique,
)
from numpy.typing import NDArray

from cubie._utils import (
    opt_gttype_validator,
    PrecisionDType,
)
from cubie.CUDAFactory import CUDAFactoryConfig, _CubieConfigBase
from cubie.outputhandling.summarymetrics import summary_metrics


def _indices_validator(
    array: Optional[NDArray[np_int32]], max_index: int
) -> None:
    """Validate index arrays and enforce bounds.

    Parameters
    ----------
    array
        Array of indices to validate.
    max_index
        Maximum allowed index value (exclusive).

    Raises
    ------
    TypeError
        Raised when *array* is not an integer numpy array.
    ValueError
        Raised when indices are out of bounds or duplicated.
    """
    if array is not None:
        if not isinstance(array, ndarray) or array.dtype != np_int32:
            raise TypeError("Index array must be a numpy array of integers.")

        if np_any((array < 0) | (array >= max_index)):
            raise ValueError(f"Indices must be in the range [0, {max_index})")

        unique_array, duplicate_count = np_unique(array, return_counts=True)
        duplicates = unique_array[duplicate_count > 1]
        if len(duplicates) > 0:
            raise ValueError(f"Duplicate indices found: {duplicates.tolist()}")


def _output_types_converter(value: Any) -> Tuple[str, ...]:
    """Normalise output type specifications to a tuple of strings."""
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise TypeError(
        f"Output types must be a list or tuple of strings, "
        f"or a single string. Got {type(value)}"
    )


def _index_array_converter(value: Any) -> NDArray[np_int32]:
    """Copy index specifications into an owned read-only int array."""
    if value is None:
        value = []
    array = np_array(value, dtype=np_int32)
    array.setflags(write=False)
    return array


def _parse_output_types(
    output_types: Tuple[str, ...],
) -> tuple[dict[str, bool], Tuple[str, ...]]:
    """Parse output type names into save flags and summary types.

    Parameters
    ----------
    output_types
        Output type names to parse.

    Returns
    -------
    tuple[dict[str, bool], tuple[str, ...]]
        Flags for the four direct output paths, and the summary metric
        identifiers in declaration order. Unknown entries trigger
        warnings but do not raise.
    """
    direct_types = ("state", "observables", "time", "iteration_counters")
    flags = {name: name in output_types for name in direct_types}
    summary_types = []
    for output_type in output_types:
        if any(
            output_type.startswith(name)
            for name in summary_metrics.implemented_metrics
        ):
            summary_types.append(output_type)
        elif output_type not in direct_types:
            warn(
                f"Summary type '{output_type}' is not implemented. "
                f"Ignoring."
            )
    return flags, tuple(summary_types)


@frozen
class OutputCompileFlags(_CubieConfigBase):
    """Boolean compile-time controls for CUDA output features.

    Attributes
    ----------
    save_state
        Whether to save state variables. Defaults to ``False``.
    save_observables
        Whether to save observable variables. Defaults to ``False``.
    summarise
        Whether to compute any summary statistics. Defaults to ``False``.
    summarise_observables
        Whether to compute summaries for observables. Defaults to ``False``.
    summarise_state
        Whether to compute summaries for state variables. Defaults to
        ``False``.
    save_counters
        Whether to save iteration counters. Defaults to ``False``.
    """

    save_state: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )
    save_observables: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )
    summarise: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )
    summarise_observables: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )
    summarise_state: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )
    save_counters: bool = field(
        default=False, validator=attrsval_instance_of(bool)
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()


@frozen
class OutputConfig(CUDAFactoryConfig):
    """Validated configuration for solver outputs and summaries.

    Parameters
    ----------
    max_states
        Maximum number of state variables.
    max_observables
        Maximum number of observable variables.
    saved_state_indices
        Indices of state variables to save. Defaults to an empty collection
        that resolves to all states.
    saved_observable_indices
        Indices of observable variables to save. Defaults to an empty
        collection that resolves to all observables.
    summarised_state_indices
        Indices of state variables to summarise. Defaults to
        *saved_state_indices*.
    summarised_observable_indices
        Indices of observable variables to summarise. Defaults to
        *saved_observable_indices*.
    output_types
        Requested output type names, including summary metric identifiers.
    sample_summaries_every
        Time interval between summary metric samples. Used by derivative
        metrics to scale finite differences. Defaults to 0.01 seconds.
    precision
        Numerical precision for output calculations. Defaults to np.float32.

    Notes
    -----
    Private attributes store numpy arrays so that properties can manage
    circular dependencies between index validation and flag updates. The
    post-initialisation hook applies default indices, validates bounds, and
    ensures at least one output path is active.
    """

    # System dimensions, used to validate indices
    _max_states: int = field(validator=attrsval_instance_of(int))
    _max_observables: int = field(validator=attrsval_instance_of(int))

    _saved_state_indices: Optional[
        Union[List[int], NDArray[np_int32]]
    ] = field(
        default=attrsFactory(list),
        converter=_index_array_converter,
        eq=attrs_cmp_using(eq=np_array_equal),
    )
    _saved_observable_indices: Optional[
        Union[List[int], NDArray[np_int32]]
    ] = (
        field(
            default=attrsFactory(list),
            converter=_index_array_converter,
            eq=attrs_cmp_using(eq=np_array_equal),
        )
    )
    _summarised_state_indices: Optional[
        Union[List[int], NDArray[np_int32]]
    ] = (
        field(
            default=attrsFactory(list),
            converter=_index_array_converter,
            eq=attrs_cmp_using(eq=np_array_equal),
        )
    )
    _summarised_observable_indices: Optional[
        Union[List[int], NDArray[np_int32]]
    ] = field(
        default=attrsFactory(list),
        converter=_index_array_converter,
        eq=attrs_cmp_using(eq=np_array_equal),
    )

    _output_types: Tuple[str, ...] = field(
        default=attrsFactory(tuple),
        converter=_output_types_converter,
    )
    _save_state: bool = field(
        default=False, init=False, repr=False, eq=False
    )
    _save_observables: bool = field(
        default=False, init=False, repr=False, eq=False
    )
    _save_time: bool = field(
        default=False, init=False, repr=False, eq=False
    )
    _save_counters: bool = field(
        default=False, init=False, repr=False, eq=False
    )
    _summary_types: Tuple[str, ...] = field(
        default=attrsFactory(tuple), init=False, repr=False, eq=False
    )
    _sample_summaries_every: Optional[float] = field(
        default=None, validator=opt_gttype_validator(float, 0.0)
    )

    def __attrs_post_init__(self) -> None:
        """Derive output flags and validate the snapshot.

        Every snapshot — construction or update-derived replacement —
        recomputes the derived flags from ``output_types``, validates
        index bounds, and confirms at least one output path is active,
        so derived state can never go stale.
        """
        super().__attrs_post_init__()
        flags, summary_types = _parse_output_types(self._output_types)
        # Frozen attrs require object.__setattr__ to install derived
        # init=False fields on the new snapshot.
        object.__setattr__(self, "_save_state", flags["state"])
        object.__setattr__(
            self, "_save_observables", flags["observables"]
        )
        object.__setattr__(self, "_save_time", flags["time"])
        object.__setattr__(
            self, "_save_counters", flags["iteration_counters"]
        )
        object.__setattr__(self, "_summary_types", summary_types)
        self._validate_index_arrays()
        self._check_for_no_outputs()

    def _validate_index_arrays(self) -> None:
        """Validate all index arrays for bounds and duplication.

        Notes
        -----
        Called post-initialisation so that ``None`` values can be replaced by
        defaults before validation.
        """
        index_arrays = [
            self._saved_state_indices,
            self._saved_observable_indices,
            self._summarised_state_indices,
            self._summarised_observable_indices,
        ]
        maxima = [
            self._max_states,
            self._max_observables,
            self._max_states,
            self._max_observables,
        ]
        for i, array in enumerate(index_arrays):
            _indices_validator(array, maxima[i])

    def _check_for_no_outputs(self) -> None:
        """Ensure that at least one output type is requested.

        Raises
        ------
        ValueError
            Raised when no output types are enabled.
        """
        any_output = (
            self._save_state
            or self._save_observables
            or self._save_time
            or self._save_counters
            or self.save_summaries
        )
        if not any_output:
            raise ValueError(
                "At least one output type must be enabled (state, "
                "observables, time, iteration_counters, summaries)"
            )

    def trimmed_index_updates(
        self, max_states: int, max_observables: int
    ) -> dict:
        """Return stored index arrays trimmed to new maxima.

        Selections that already fit are not returned.
        """
        stored = {
            "saved_state_indices": (
                self._saved_state_indices, max_states,
            ),
            "summarised_state_indices": (
                self._summarised_state_indices, max_states,
            ),
            "saved_observable_indices": (
                self._saved_observable_indices, max_observables,
            ),
            "summarised_observable_indices": (
                self._summarised_observable_indices, max_observables,
            ),
        }
        updates = {}
        for key, (array, bound) in stored.items():
            if array is not None and array.size and (
                int(array.max()) >= bound
            ):
                updates[key] = array[array < bound]
        return updates

    @property
    def max_states(self) -> int:
        """Maximum number of states."""
        return self._max_states

    @property
    def max_observables(self) -> int:
        """Maximum number of observables."""
        return self._max_observables

    @property
    def save_state(self) -> bool:
        """Whether state saving is enabled with valid indices."""
        return self._save_state and (len(self._saved_state_indices) > 0)

    @property
    def save_observables(self) -> bool:
        """Whether observable saving is enabled with valid indices."""
        return self._save_observables and (
            len(self._saved_observable_indices) > 0
        )

    @property
    def save_time(self) -> bool:
        """Whether solver time samples should be saved."""
        return self._save_time

    @property
    def save_counters(self) -> bool:
        """Whether iteration counters should be saved."""
        return self._save_counters

    @property
    def save_summaries(self) -> bool:
        """Whether any summary metric is configured."""
        return len(self._summary_types) > 0

    @property
    def summarise_state(self) -> bool:
        """Whether state variables contribute to summaries."""
        return self.save_summaries and self.n_summarised_states > 0

    @property
    def summarise_observables(self) -> bool:
        """Whether observable variables contribute to summaries."""
        return self.save_summaries and self.n_summarised_observables > 0

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """
        Get compile flags for this configuration.

        Returns
        -------
        OutputCompileFlags
            Flags indicating which output features should be compiled.
        """
        return OutputCompileFlags(
            save_state=self.save_state,
            save_observables=self.save_observables,
            summarise=self.save_summaries,
            summarise_observables=self.summarise_observables,
            summarise_state=self.summarise_state,
            save_counters=self.save_counters,
        )

    @property
    def saved_state_indices(self) -> NDArray[np_int32]:
        """State indices to save, or an empty array when disabled."""
        if not self._save_state:
            return np_asarray([], dtype=np_int32)
        return self._saved_state_indices

    @property
    def saved_observable_indices(self) -> NDArray[np_int32]:
        """Observable indices to save, or an empty array when disabled."""
        if not self._save_observables:
            return np_asarray([], dtype=np_int32)
        return self._saved_observable_indices

    @property
    def summarised_state_indices(self) -> NDArray[np_int32]:
        """State indices for summaries, or an empty array when disabled."""
        if not self.save_summaries:
            return np_asarray([], dtype=np_int32)
        return self._summarised_state_indices

    @property
    def summarised_observable_indices(self) -> NDArray[np_int32]:
        """Observable indices for summaries, or an empty array when disabled.
        """
        if not self.save_summaries:
            return np_asarray([], dtype=np_int32)
        return self._summarised_observable_indices

    @property
    def n_saved_states(self) -> int:
        """
        Get number of states that will be saved.

        Returns
        -------
        int
            Number of state variables to save in time-domain output.

        Notes
        -----
        Returns the length of saved_state_indices when save_state is True,
        otherwise 0.
        """
        return len(self._saved_state_indices) if self._save_state else 0

    @property
    def n_saved_observables(self) -> int:
        """
        Get number of observables that will be saved.

        Returns
        -------
        int
            Number of observable variables to save in time-domain output.
        """
        return (
            len(self._saved_observable_indices)
            if self._save_observables
            else 0
        )

    @property
    def n_summarised_states(self) -> int:
        """
        Get number of states that will be summarised.

        Returns
        -------
        int
            Number of state variables for summary calculations.

        Notes
        -----
        Returns the length of summarised_state_indices when save_summaries
        is active, otherwise 0.
        """
        return (
            len(self._summarised_state_indices) if self.save_summaries else 0
        )

    @property
    def n_summarised_observables(self) -> int:
        """
        Get number of observables that will be summarised.

        Returns
        -------
        int
            Number of observable variables for summary calculations.
        """
        return (
            len(self._summarised_observable_indices)
            if self.save_summaries
            else 0
        )

    @property
    def summary_types(self) -> Tuple[str, ...]:
        """Configured summary metric identifiers."""
        return self._summary_types

    @property
    def summary_legend_per_variable(self) -> dict[int, str]:
        """Map per-variable summary indices to metric names.

        Returns
        -------
        dict[int, str]
            Dictionary that assigns each per-variable summary slot to its
            metric identifier.
        """
        if not self._summary_types:
            return {}
        legend_tuple = summary_metrics.legend(self._summary_types)
        legend_dict = dict(zip(range(len(legend_tuple)), legend_tuple))
        return legend_dict

    @property
    def summary_unit_modifications(self) -> dict[int, str]:
        """Map per-variable summary indices to unit modifications.

        Returns
        -------
        dict[int, str]
            Dictionary that assigns each per-variable summary slot to its
            unit modification string.
        """
        if not self._summary_types:
            return {}
        unit_mod_tuple = summary_metrics.unit_modifications(
            self._summary_types
        )
        unit_mod_dict = dict(zip(range(len(unit_mod_tuple)), unit_mod_tuple))
        return unit_mod_dict

    @property
    def sample_summaries_every(self) -> Optional[float]:
        """Time interval between summary metric samples."""
        return self._sample_summaries_every

    @property
    def summaries_buffer_height_per_var(self) -> int:
        """
        Calculate buffer size per variable for summary calculations.

        Returns
        -------
        int
            Buffer height required per variable for summary metrics.
        """
        if not self.summary_types:
            return 0
        # Convert summary_types set to list for summarymetrics
        summary_list = list(self._summary_types)
        total_buffer_size = summary_metrics.summaries_buffer_height(
            summary_list
        )
        return total_buffer_size

    @property
    def summaries_output_height_per_var(self) -> int:
        """
        Calculate output array height per variable for summaries.

        Returns
        -------
        int
            Output height required per variable for summary results.
        """
        if not self._summary_types:
            return 0
        # Convert summary_types tuple to list for summarymetrics
        summary_list = list(self._summary_types)
        total_output_size = summary_metrics.summaries_output_height(
            summary_list
        )
        return total_output_size

    @property
    def state_summaries_buffer_height(self) -> int:
        """
        Calculate total buffer height for state summaries.

        Returns
        -------
        int
            Total buffer height for all state summary calculations.
        """
        return self.summaries_buffer_height_per_var * self.n_summarised_states

    @property
    def observable_summaries_buffer_height(self) -> int:
        """
        Calculate total buffer height for observable summaries.

        Returns
        -------
        int
            Total buffer height for all observable summary calculations.
        """
        return (
            self.summaries_buffer_height_per_var
            * self.n_summarised_observables
        )

    @property
    def total_summary_buffer_size(self) -> int:
        """
        Calculate total size of all summary buffers.

        Returns
        -------
        int
            Combined size of state and observable summary buffers.
        """
        return (
            self.state_summaries_buffer_height
            + self.observable_summaries_buffer_height
        )

    @property
    def state_summaries_output_height(self) -> int:
        """
        Calculate total output height for state summaries.

        Returns
        -------
        int
            Total output height for all state summary results.
        """
        return self.summaries_output_height_per_var * self.n_summarised_states

    @property
    def observable_summaries_output_height(self) -> int:
        """
        Calculate total output height for observable summaries.

        Returns
        -------
        int
            Total output height for all observable summary results.
        """
        return (
            self.summaries_output_height_per_var
            * self.n_summarised_observables
        )

    @property
    def buffer_sizes_dict(self) -> dict[str, int]:
        """Returns a dict of buffer sizes to update other objects' settings"""
        return {
            "n_saved_states": self.n_saved_states,
            "n_saved_observables": self.n_saved_observables,
            "n_summarised_states": self.n_summarised_states,
            "n_summarised_observables": self.n_summarised_observables,
            "state_summaries_buffer_height": (
                self.state_summaries_buffer_height
            ),
            "observable_summaries_buffer_height": (
                self.observable_summaries_buffer_height
            ),
            "total_summary_buffer_size": self.total_summary_buffer_size,
            "state_summaries_output_height": (
                self.state_summaries_output_height
            ),
            "observable_summaries_output_height": (
                self.observable_summaries_output_height
            ),
            "compile_flags": self.compile_flags,
        }

    @property
    def output_types(self) -> Tuple[str, ...]:
        """Configured output type names in declaration order."""
        return self._output_types

    @classmethod
    def from_loop_settings(
        cls,
        output_types: List[str],
        precision: PrecisionDType,
        saved_state_indices: Union[
            Sequence[int], NDArray[np_int32], None
        ] = None,
        saved_observable_indices: Union[
            Sequence[int], NDArray[np_int32], None
        ] = None,
        summarised_state_indices: Union[
            Sequence[int], NDArray[np_int32], None
        ] = None,
        summarised_observable_indices: Union[
            Sequence[int], NDArray[np_int32], None
        ] = None,
        max_states: int = 0,
        max_observables: int = 0,
        sample_summaries_every: Optional[float] = 0.01,
    ) -> "OutputConfig":
        """
        Create configuration from integrator-compatible specifications.

        Parameters
        ----------
        output_types
            Output types including ``"state"``, ``"observables"``, ``"time"``,
            and summary metric names such as ``"max"`` or ``"rms"``.
        saved_state_indices
            Indices of states to save in time-domain output.
        saved_observable_indices
            Indices of observables to save in time-domain output.
        summarised_state_indices
            Indices of states for summary calculations. Defaults to
            *saved_state_indices*.
        summarised_observable_indices
            Indices of observables for summary calculations. Defaults to
            *saved_observable_indices*.
        max_states
            Total number of state variables in the system.
        max_observables
            Total number of observable variables in the system.
        sample_summaries_every
            Time interval between summary metric samples. Used by derivative
            metrics to scale finite differences. Defaults to ``0.01``.
        precision
            Numerical precision for output calculations.

        Returns
        -------
        OutputConfig
            Configured output configuration object.

        Notes
        -----
        This class method provides a convenient interface for creating
        OutputConfig objects from the parameter format used by integrator
        classes. It handles None values appropriately by converting them
        to empty arrays.
        """
        # Set boolean compile flags for output types
        output_types = list(output_types)

        # OutputConfig doesn't play as nicely with Nones as the rest of python
        # does
        if saved_state_indices is None:
            saved_state_indices = np_asarray([], dtype=np_int32)
        if saved_observable_indices is None:
            saved_observable_indices = np_asarray([], dtype=np_int32)
        if summarised_state_indices is None:
            summarised_state_indices = np_asarray([], dtype=np_int32)
        if summarised_observable_indices is None:
            summarised_observable_indices = np_asarray([], dtype=np_int32)

        return cls(
            max_states=max_states,
            max_observables=max_observables,
            saved_state_indices=saved_state_indices,
            saved_observable_indices=saved_observable_indices,
            summarised_state_indices=summarised_state_indices,
            summarised_observable_indices=summarised_observable_indices,
            output_types=output_types,
            sample_summaries_every=sample_summaries_every,
            precision=precision,
        )
