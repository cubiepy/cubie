"""Factories that compile and cache CUDA output management routines.

Published Classes
-----------------
:class:`OutputFunctionCache`
    Cache container for compiled output functions.

:class:`OutputFunctions`
    Factory that compiles and caches save-state, update-summary, and
    save-summary CUDA device functions.

    >>> from numpy import float32
    >>> of = OutputFunctions(
    ...     max_states=3, max_observables=2, precision=float32,
    ... )
    >>> of.n_saved_states
    3

Module-Level Constants
----------------------
:data:`ALL_OUTPUT_FUNCTION_PARAMETERS`
    Keyword arguments accepted by :class:`OutputFunctions`.

See Also
--------
:class:`~cubie.outputhandling.output_config.OutputConfig`
    Validated configuration consumed by this factory.
:class:`~cubie.CUDAFactory.CUDAFactory`
    Base factory providing compilation and cache management.
:mod:`~cubie.outputhandling.save_state`
    State-saving device function factory.
:mod:`~cubie.outputhandling.save_summaries`
    Summary-saving device function factory.
:mod:`~cubie.outputhandling.update_summaries`
    Summary-update device function factory.
"""

from typing import Callable, Sequence, Union, Optional

from attrs import define, field, validators
from numpy import int_
from numpy.typing import ArrayLike, NDArray

from cubie.CUDAFactory import CUDAFactory, CUDADispatcherCache
from cubie.outputhandling.output_config import OutputCompileFlags, OutputConfig
from cubie.outputhandling.output_sizes import OutputArrayHeights
from cubie.outputhandling.save_state import save_state_factory
from cubie.outputhandling.save_summaries import save_summary_factory
from cubie.outputhandling.summarymetrics import summary_metrics
from cubie.outputhandling.update_summaries import update_summary_factory
from cubie._utils import PrecisionDType


ALL_OUTPUT_FUNCTION_PARAMETERS = {
    "output_types",
    "saved_state_indices",
    "saved_observable_indices",
    "summarised_state_indices",
    "summarised_observable_indices",
    "sample_summaries_every",
    "precision",
}
"""Keyword arguments accepted by :class:`OutputFunctions`.

These parameters can be passed to :class:`OutputFunctions` or to
:meth:`OutputFunctions.update`. Parent components use this set to
filter ``**kwargs`` before forwarding them.

.. list-table:: Parameter Summary
   :header-rows: 1

   * - Parameter
     - Accepted By
     - Description
   * - ``output_types``
     - :class:`~cubie.outputhandling.output_config.OutputConfig`
     - List of output type names (``"state"``, ``"observables"``,
       ``"time"``, metric names).
   * - ``saved_state_indices``
     - :class:`~cubie.outputhandling.output_config.OutputConfig`
     - Indices of state variables to save.
   * - ``saved_observable_indices``
     - :class:`~cubie.outputhandling.output_config.OutputConfig`
     - Indices of observable variables to save.
   * - ``summarised_state_indices``
     - :class:`~cubie.outputhandling.output_config.OutputConfig`
     - Indices of state variables for summary calculations.
   * - ``summarised_observable_indices``
     - :class:`~cubie.outputhandling.output_config.OutputConfig`
     - Indices of observable variables for summary calculations.
   * - ``sample_summaries_every``
     - :class:`~cubie.outputhandling.output_config.OutputConfig`
     - Time interval between summary metric samples.
   * - ``precision``
     - :class:`~cubie.outputhandling.output_config.OutputConfig`
     - Floating-point dtype for output calculations.
"""


@define
class OutputFunctionCache(CUDADispatcherCache):
    """Cache container for compiled output functions.

    Attributes
    ----------
    save_state_function
        Compiled CUDA function for saving state values.
    update_summaries_function
        Compiled CUDA function for updating summary metrics.
    save_summaries_function
        Compiled CUDA function for saving summary results.
    """

    save_state_function: Callable = field(
        validator=validators.instance_of(Callable)
    )
    update_summaries_function: Callable = field(
        validator=validators.instance_of(Callable)
    )
    save_summaries_function: Callable = field(
        validator=validators.instance_of(Callable)
    )


class OutputFunctions(CUDAFactory):
    """Factory that compiles and caches output management functions.

    Parameters
    ----------
    max_states
        Maximum number of state variables in the system.
    max_observables
        Maximum number of observable variables in the system.
    output_types
        Types of output to generate. Defaults to ["state"].
    saved_state_indices
        Indices of state variables to save in time-domain output.
    saved_observable_indices
        Indices of observable variables to save in time-domain output.
    summarised_state_indices
        Indices of state variables to include in summary calculations.
    summarised_observable_indices
        Indices of observable variables to include in summary calculations.
    sample_summaries_every
        Time interval between summary metric samples. Used by derivative
        metrics to scale finite differences. Defaults to None.
    precision
        Numerical precision for output calculations. Defaults to np.float32.

    Notes
    -----
    The constructor converts the provided options into an
    :class:`~cubie.outputhandling.output_config.OutputConfig` instance and
    installs it as the compile settings. CUDA callables are only built once
    per configuration and cached by :class:`cubie.CUDAFactory`.
    """

    def __init__(
        self,
        max_states: int,
        max_observables: int,
        precision: PrecisionDType,
        output_types: list[str] = None,
        saved_state_indices: Union[Sequence[int], ArrayLike] = None,
        saved_observable_indices: Union[Sequence[int], ArrayLike] = None,
        summarised_state_indices: Union[Sequence[int], ArrayLike] = None,
        summarised_observable_indices: Union[Sequence[int], ArrayLike] = None,
        sample_summaries_every: Optional[float] = None,
    ):
        super().__init__()

        if output_types is None:
            output_types = ["state"]

        # Create and setup output configuration as compile settings
        config = OutputConfig.from_loop_settings(
            output_types=output_types,
            max_states=max_states,
            max_observables=max_observables,
            saved_state_indices=saved_state_indices,
            saved_observable_indices=saved_observable_indices,
            summarised_state_indices=summarised_state_indices,
            summarised_observable_indices=summarised_observable_indices,
            sample_summaries_every=sample_summaries_every,
            precision=precision,
        )
        self.setup_compile_settings(config)

    def update(
        self,
        updates_dict: Union[dict[str, object], None] = None,
        silent: bool = False,
        **kwargs: object,
    ) -> set[str]:
        """Update compile settings through the factory interface.

        Parameters
        ----------
        updates_dict
            Dictionary of parameter updates to apply.
        silent
            When ``True``, suppress warnings about unrecognised parameters.
        **kwargs
            Additional parameter updates to apply.

        Returns
        -------
        set[str]
            Recognised parameter names that were successfully updated.

        Raises
        ------
        KeyError
            If unrecognised parameters are provided and ``silent`` is ``False``.

        Notes
        -----
        Use this method for coordinated configuration updates alongside other
        components by passing ``silent=True`` so unrelated keys fall through
        without raising.
        """
        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()
        unrecognised = set(updates_dict.keys())

        # Trim stored indices to shrinking maxima; explicit indices
        # in the same update win.
        config = self.compile_settings
        new_max_states = updates_dict.get(
            "max_states", config.max_states
        )
        new_max_observables = updates_dict.get(
            "max_observables", config.max_observables
        )
        if (
            new_max_states != config.max_states
            or new_max_observables != config.max_observables
        ):
            trimmed = config.trimmed_index_updates(
                new_max_states, new_max_observables
            )
            for key, indices in trimmed.items():
                updates_dict.setdefault(key, indices)

        recognised_params = set()
        recognised_params |= self.update_compile_settings(
            updates_dict, silent=True
        )
        unrecognised -= recognised_params

        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return set(recognised_params)

    def build(self) -> OutputFunctionCache:
        """Compile output functions and calculate memory requirements.

        Returns
        -------
        OutputFunctionCache
            Container with compiled functions that target the current
            configuration.

        Notes
        -----
        This method is invoked lazily by :class:`cubie.CUDAFactory` the first
        time a compiled function is requested. The resulting cache is reused
        until configuration settings change.
        """
        config = self.compile_settings

        summary_metrics.update(
            sample_summaries_every=config.sample_summaries_every,
            precision=config.precision,
            lineinfo=config.lineinfo,
        )

        # Build functions using output sizes objects
        save_state_func = save_state_factory(
            config.saved_state_indices,
            config.saved_observable_indices,
            config.save_state,
            config.save_observables,
            config.save_time,
            config.save_counters,
            lineinfo=config.lineinfo,
        )

        update_summary_metrics_func = update_summary_factory(
            config.summaries_buffer_height_per_var,
            config.summarised_state_indices,
            config.summarised_observable_indices,
            config.summary_types,
            lineinfo=config.lineinfo,
        )

        save_summary_metrics_func = save_summary_factory(
            config.summaries_buffer_height_per_var,
            config.summarised_state_indices,
            config.summarised_observable_indices,
            config.summary_types,
            lineinfo=config.lineinfo,
        )

        return OutputFunctionCache(
            save_state_function=save_state_func,
            update_summaries_function=update_summary_metrics_func,
            save_summaries_function=save_summary_metrics_func,
        )

    @property
    def save_state_func(self) -> Callable:
        """Compiled state saving function."""
        return self.get_cached_output("save_state_function")

    @property
    def update_summaries_func(self) -> Callable:
        """Compiled summary update function."""
        return self.get_cached_output("update_summaries_function")

    @property
    def output_types(self) -> set[str]:
        """Configured output types."""
        return self.compile_settings.output_types

    @property
    def save_summary_metrics_func(self) -> Callable:
        """Compiled summary saving function."""
        return self.get_cached_output("save_summaries_function")

    @property
    def compile_flags(self) -> OutputCompileFlags:
        """Compile flags for the active configuration."""
        return self.compile_settings.compile_flags

    @property
    def save_time(self) -> bool:
        """Whether time samples are saved alongside states."""
        return self.compile_settings.save_time

    @property
    def save_counters(self) -> bool:
        """Whether iteration counters are saved at each save point."""
        return self.compile_settings.save_counters

    @property
    def has_time_domain_outputs(self) -> bool:
        """Whether any time-domain output is enabled."""
        config = self.compile_settings
        save_time = config.save_time
        save_state = config.save_state
        save_observables = config.save_observables
        return save_time or save_state or save_observables

    @property
    def has_summary_outputs(self) -> bool:
        """Whether any summary output is enabled."""
        config = self.compile_settings
        return config.summarise_state or config.summarise_observables

    @property
    def saved_state_indices(self) -> NDArray[int_]:
        """Indices of states saved in time-domain output."""
        return self.compile_settings.saved_state_indices

    @property
    def saved_observable_indices(self) -> NDArray[int_]:
        """Indices of observables saved in time-domain output."""
        return self.compile_settings.saved_observable_indices

    @property
    def summarised_state_indices(self) -> NDArray[int_]:
        """Indices of states tracked by summary metrics."""
        return self.compile_settings.summarised_state_indices

    @property
    def summarised_observable_indices(self) -> NDArray[int_]:
        """Indices of observables tracked by summary metrics."""
        return self.compile_settings.summarised_observable_indices

    @property
    def n_saved_states(self) -> int:
        """Number of state variables saved in time-domain output."""
        return self.compile_settings.n_saved_states

    @property
    def n_saved_observables(self) -> int:
        """Number of observable variables saved in time-domain output."""
        return self.compile_settings.n_saved_observables

    @property
    def state_summaries_output_height(self) -> int:
        """Height of the state summary output array."""
        return self.compile_settings.state_summaries_output_height

    @property
    def observable_summaries_output_height(self) -> int:
        """Height of the observable summary output array."""
        return self.compile_settings.observable_summaries_output_height

    @property
    def summaries_buffer_height_per_var(self) -> int:
        """Height of the summary buffer required per variable."""
        return self.compile_settings.summaries_buffer_height_per_var

    @property
    def state_summaries_buffer_height(self) -> int:
        """Total height of the buffer for state summary calculations."""
        return self.compile_settings.state_summaries_buffer_height

    @property
    def observable_summaries_buffer_height(self) -> int:
        """Total height of the buffer for observable summary calculations."""
        return self.compile_settings.observable_summaries_buffer_height

    @property
    def summaries_output_height_per_var(self) -> int:
        """Height of the summary output array per variable."""
        return self.compile_settings.summaries_output_height_per_var

    @property
    def output_array_heights(self) -> OutputArrayHeights:
        """Output array height helper built from the active configuration."""
        return OutputArrayHeights.from_output_fns(self)

    @property
    def summary_legend_per_variable(self) -> dict[str, int]:
        """Mapping of summary metric names to their per-variable heights."""
        return self.compile_settings.summary_legend_per_variable

    @property
    def summary_unit_modifications(self) -> dict[int, str]:
        """Mapping of summary indices to unit modification strings."""
        return self.compile_settings.summary_unit_modifications

    @property
    def buffer_sizes_dict(self) -> dict[str, int]:
        """Dictionary of buffer sizes for each output type."""
        return self.compile_settings.buffer_sizes_dict
