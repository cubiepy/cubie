"""Convenience interface for accessing system values.

This module provides :class:`SystemInterface`, which wraps
:class:`cubie.odesystems.SystemValues` instances for parameters, states, and
observables. It exposes helper methods for converting between user-facing
labels or indices and internal representations.

Published Classes
-----------------
:class:`SystemInterface`
    Wrapper providing label-to-index resolution and value updates for
    parameters, states, and observables.

    >>> from cubie.batchsolving.SystemInterface import SystemInterface
    >>> interface = SystemInterface(system)
    >>> interface.state_indices(["x", "y"])
    array([0, 1], dtype=int32)

Notes
-----
The interface allows updating default state or parameter values without
navigating the full system hierarchy, providing a simplified entry point for
common operations.

See Also
--------
:class:`~cubie.odesystems.SystemValues.SystemValues`
    Underlying keyed parameter container.
:class:`~cubie.odesystems.baseODE.BaseODE`
    ODE system base class from which interfaces are created.
:class:`~cubie.batchsolving.solver.Solver`
    Primary consumer of this interface.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from numpy import (
    ndarray,
    int32 as np_int32,
    arange as np_arange,
    union1d as np_union1d,
    array as np_array,
)
from cubie.odesystems.baseODE import BaseODE
from cubie.odesystems.SystemValues import SystemValues


class SystemInterface:
    """Live view onto a system's value containers.

    Parameters
    ----------
    system
        The ODE system whose value containers this interface reads.

    Notes
    -----
    The variable resolution methods interpret input values as:

    - ``None`` means "use all" (default behavior)
    - ``[]`` or empty array means "explicitly no variables"
    - When both labels and indices are provided, their union is used
    """

    def __init__(self, system: BaseODE):
        self._system = system

    @property
    def parameters(self) -> SystemValues:
        """Parameter values, read live from the system."""
        return self._system.parameters

    @property
    def states(self) -> SystemValues:
        """Initial state values, read live from the system."""
        return self._system.initial_values

    @property
    def observables(self) -> SystemValues:
        """Observable definitions, read live from the system."""
        return self._system.observables

    def update(
        self,
        updates: Optional[Dict[str, Any]] = None,
        silent: bool = False,
        **kwargs,
    ) -> Optional[Set[str]]:
        """Update default parameter or state values.

        Parameters
        ----------
        updates
            Mapping of label to new value. If ``None``, only keyword arguments
            are used for updates.
        silent
            If ``True``, suppresses ``KeyError`` for unrecognized update keys.
        **kwargs
            Additional keyword arguments merged with ``updates``. Each
            key-value pair represents a label-value mapping for updating system
            values.

        Returns
        -------
        set of str or None
            Set of recognized update keys that were successfully applied.
            Returns None if no updates were provided.

        Raises
        ------
        KeyError
            If ``silent`` is False and unrecognized update keys are provided.

        Notes
        -----
        The method attempts to update both parameters and states. Updates are
        applied to whichever :class:`SystemValues` object recognizes each key.
        """
        if updates is None:
            updates = {}
        if kwargs:
            updates.update(kwargs)
        if not updates:
            return

        all_unrecognized = set(updates.keys())
        for values_object in (self.parameters, self.states):
            recognized = values_object.update_from_dict(updates, silent=True)
            all_unrecognized -= recognized

        if all_unrecognized:
            if not silent:
                unrecognized_list = sorted(all_unrecognized)
                raise KeyError(
                    "The following updates were not recognized by the system. "
                    "Was this a typo?: "
                    f"{unrecognized_list}"
                )

        recognized = set(updates.keys()) - all_unrecognized
        return recognized

    def state_indices(
        self,
        keys_or_indices: Optional[
            Union[List[Union[str, int]], str, int]
        ] = None,
        silent: bool = False,
    ) -> ndarray:
        """Convert state labels or indices to a numeric array.

        Parameters
        ----------
        keys_or_indices
            State names, indices, or a mix of both. ``None`` returns all state
            indices.
        silent
            If ``True``, suppresses warnings for unrecognized keys or indices.

        Returns
        -------
        ndarray
            Array of integer indices corresponding to the provided identifiers.
        """
        if keys_or_indices is None:
            keys_or_indices = self.states.names
        return self.states.get_indices(keys_or_indices, silent=silent)

    def observable_indices(
        self,
        keys_or_indices: Optional[Union[List[Union[str, int]], str, int]] = None,
        silent: bool = False,
    ) -> ndarray:
        """Convert observable labels or indices to a numeric array.

        Parameters
        ----------
        keys_or_indices
            Observable names, indices, or a mix of both. ``None`` returns all
            observable indices.
        silent
            If ``True``, suppresses warnings for unrecognized keys or indices.
        Returns
        -------
        ndarray
            Array of integer indices corresponding to the provided identifiers.

        """
        if keys_or_indices is None:
            keys_or_indices = self.observables.names
        return self.observables.get_indices(keys_or_indices, silent=silent)

    def parameter_indices(
        self,
        keys_or_indices: Union[List[Union[str, int]], str, int],
        silent: bool = False,
    ) -> ndarray:
        """Convert parameter labels or indices to a numeric array.

        Parameters
        ----------
        keys_or_indices
            Parameter names, indices, or a mix of both.
        silent
            If ``True``, suppresses warnings for unrecognized keys or indices.

        Returns
        -------
        ndarray
            Array of integer indices corresponding to the provided identifiers.
        """
        return self.parameters.get_indices(keys_or_indices, silent=silent)

    def get_labels(
        self, values_object: SystemValues, indices: ndarray
    ) -> List[str]:
        """Return labels corresponding to the provided indices.

        Parameters
        ----------
        values_object
            The SystemValues object to retrieve labels from.
        indices
            A 1D array of integer indices.

        Returns
        -------
        list of str
            List of labels corresponding to the provided indices.
        """
        return values_object.get_labels(indices)

    def state_labels(self, indices: Optional[ndarray] = None) -> List[str]:
        """Return state labels corresponding to the provided indices.

        Parameters
        ----------
        indices
            A 1D array of state indices.
            If ``None``, return all state labels.

        Returns
        -------
        list of str
            List of state labels corresponding to the provided indices.
        """
        if indices is None:
            return self.states.names
        return self.get_labels(self.states, indices)

    def observable_labels(
        self, indices: Optional[ndarray] = None
    ) -> List[str]:
        """Return observable labels corresponding to the provided indices.

        Parameters
        ----------
        indices
            A 1D array of observable indices.
            If ``None``, return all observable labels.

        Returns
        -------
        list of str
            List of observable labels corresponding to the provided indices.
        """
        if indices is None:
            return self.observables.names
        return self.get_labels(self.observables, indices)

    def parameter_labels(self, indices: Optional[ndarray] = None) -> List[str]:
        """Return parameter labels corresponding to the provided indices.

        Parameters
        ----------
        indices
            A 1D array of parameter indices.
            If ``None``, return all parameter labels.

        Returns
        -------
        list of str
            List of parameter labels corresponding to the provided indices.
        """
        if indices is None:
            return self.parameters.names
        return self.get_labels(self.parameters, indices)

    def resolve_variable_labels(
        self,
        labels: Optional[List[str]],
        silent: bool = False,
    ) -> Tuple[Optional[ndarray], Optional[ndarray]]:
        """Resolve variable labels to separate state and observable indices.

        Parameters
        ----------
        labels
            Variable names that may be states or observables. If ``None``,
            returns ``(None, None)`` to signal "use defaults". If empty
            list, returns empty arrays for both.
        silent
            If ``True``, suppresses errors for unrecognized labels.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            Tuple of (state_indices, observable_indices). Returns
            ``(None, None)`` when labels is None.

        Raises
        ------
        ValueError
            If any label is not found in states or observables and
            silent is False.
        """
        if labels is None:
            return (None, None)

        if len(labels) == 0:
            return (
                np_array([], dtype=np_int32),
                np_array([], dtype=np_int32),
            )

        # Resolve each label to state or observable index
        state_idxs = self.state_indices(labels, silent=True)
        obs_idxs = self.observable_indices(labels, silent=True)

        # Track which labels were resolved
        state_names_set = set(self.states.names)
        obs_names_set = set(self.observables.names)
        resolved_labels = state_names_set.union(obs_names_set)
        unresolved = [lbl for lbl in labels if lbl not in resolved_labels]

        # Validate all labels found (unless silent)
        if not silent and unresolved:
            raise ValueError(
                f"Variables not found in states or observables: "
                f"{unresolved}. "
                f"Available states: {self.states.names}. "
                f"Available observables: {self.observables.names}."
            )

        return (
            state_idxs.astype(np_int32),
            obs_idxs.astype(np_int32),
        )

    def merge_variable_inputs(
        self,
        var_labels: Optional[List[str]],
        state_indices: Optional[Union[List[int], ndarray]],
        observable_indices: Optional[Union[List[int], ndarray]],
    ) -> Tuple[ndarray, ndarray]:
        """Merge label-based selections with index-based selections.

        Parameters
        ----------
        var_labels
            Variable names to resolve. None means "not provided".
        state_indices
            Direct state index selection. None means "not provided".
        observable_indices
            Direct observable index selection. None means "not provided".

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Final (state_indices, observable_indices) arrays.

        Notes
        -----
        When all three inputs are None, returns full range arrays
        (all states, all observables). When any input is explicitly
        empty ([] or empty array), that emptiness is preserved.
        Union of resolved labels and provided indices is returned.
        """
        # Resolve var_labels using resolve_variable_labels()
        resolved_state, resolved_obs = self.resolve_variable_labels(var_labels)

        # Check if all three inputs are None (use defaults)
        all_none = (
            var_labels is None
            and state_indices is None
            and observable_indices is None
        )
        if all_none:
            return (
                np_arange(self.states.n, dtype=np_int32),
                np_arange(self.observables.n, dtype=np_int32),
            )

        arrays_to_merge = {
            "state_from_labels": resolved_state,
            "obs_from_labels": resolved_obs,
            "state_from_indices": state_indices,
            "obs_from_indices": observable_indices,
        }

        for key, input in arrays_to_merge.items():
            if input is None:
                arrays_to_merge[key] = np_array([], dtype=np_int32)

        # Compute union of resolved label indices with provided indices
        final_state = np_union1d(
            arrays_to_merge["state_from_labels"],
            arrays_to_merge["state_from_indices"],
        )
        final_obs = np_union1d(
            arrays_to_merge["obs_from_labels"],
            arrays_to_merge["obs_from_indices"],
        )

        return final_state, final_obs

    def merge_variable_labels_and_idxs(
        self, output_settings: Dict[str, Any]
    ) -> None:
        """Convert variable label settings to index arrays in-place.

        Parameters
        ----------
        output_settings
            Settings dict containing ``save_variables``,
            ``summarise_variables``, and their index counterparts.
            Modified in-place.

        Raises
        ------
        ValueError
            If any variable labels are not found in states or observables.

        Notes
        -----
        Pops ``save_variables`` and ``summarise_variables`` from the dict
        and replaces index parameters with final resolved arrays. For
        summarised indices, defaults to saved indices when both labels
        and indices are None.
        """
        # Extract save_variables and related indices
        save_vars = output_settings.pop("save_variables", None)
        saved_state_idxs = output_settings.get("saved_state_indices", None)
        saved_obs_idxs = output_settings.get("saved_observable_indices", None)

        # Call merge_variable_inputs for save variables
        final_saved_state, final_saved_obs = self.merge_variable_inputs(
            save_vars,
            saved_state_idxs,
            saved_obs_idxs,
        )

        # Extract summarise_variables and related indices
        summarise_vars = output_settings.pop("summarise_variables", None)
        summ_state_idxs = output_settings.get("summarised_state_indices", None)
        summ_obs_idxs = output_settings.get(
            "summarised_observable_indices", None
        )

        # Handle "summarised defaults to saved" when all summarise inputs None
        all_summarise_none = (
            summarise_vars is None
            and summ_state_idxs is None
            and summ_obs_idxs is None
        )
        if all_summarise_none:
            final_summ_state = final_saved_state.copy()
            final_summ_obs = final_saved_obs.copy()
        else:
            final_summ_state, final_summ_obs = self.merge_variable_inputs(
                summarise_vars,
                summ_state_idxs,
                summ_obs_idxs,
            )

        # Update dict with final index arrays
        output_settings["saved_state_indices"] = final_saved_state
        output_settings["saved_observable_indices"] = final_saved_obs
        output_settings["summarised_state_indices"] = final_summ_state
        output_settings["summarised_observable_indices"] = final_summ_obs

    @property
    def all_input_labels(self) -> List[str]:
        """List all input labels (states followed by parameters)."""
        return self.state_labels() + self.parameter_labels()

    @property
    def all_output_labels(self) -> List[str]:
        """List all output labels (states followed by observables)."""
        return self.state_labels() + self.observable_labels()
