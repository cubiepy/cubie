"""Containers for the numerical values used to parameterise ODE systems.

Published Classes
-----------------
:class:`SystemValues`
    Keyed parameter container pairing a name-value dictionary with a
    packed NumPy array.

    >>> from numpy import float64
    >>> sv = SystemValues({"x": 1.0, "y": 2.0}, float64)
    >>> sv["x"]
    1.0
    >>> sv.names
    ['x', 'y']

See Also
--------
:class:`~cubie.odesystems.ODEData.ODEData`
    Data container that stores ``SystemValues`` instances for each
    component category (states, parameters, constants, observables).
:class:`~cubie.odesystems.baseODE.BaseODE`
    Abstract ODE factory exposing ``SystemValues`` via properties.
"""

from collections.abc import Mapping, Sequence, Sized
from types import MappingProxyType
from typing import Any, Union

from numpy import (
    arange as np_arange,
    array as np_array,
    asarray as np_asarray,
    floating as np_floating,
    int32 as np_int32,
    integer as np_integer,
    issubdtype as np_issubdtype,
    ndarray,
)
from sympy import Symbol

from cubie._utils import PrecisionDType


class SystemValues:
    """Manage keyed parameter values and their packed array representation.

    Parameters
    ----------
    values_dict
        Dictionary defining parameter values. Lists of parameter names are
        expanded to a dictionary with ``0.0`` defaults.
    precision
        Precision factory applied when creating the packed values array.
    defaults
        Optional dictionary supplying baseline parameter values.
    name
        Display label used in ``repr`` output.
    **kwargs
        Individual parameter overrides applied after ``defaults`` and
        ``values_dict``.

    Notes
    -----
    Dictionary-style and array-style indexing both operate on the packed
    ``values_array`` and backing ``values_dict``.
    """

    values_array: Union[ndarray, None]
    indices_dict: Union[dict[str, int], None]
    keys_by_index: Union[dict[int, str], None]
    values_dict: Mapping[str, float]
    _values_backing: dict[str, float]
    precision: PrecisionDType
    n: int
    name: Union[str, None]

    # Class-level defaults: freeze() seals an instance in place.
    _snapshot_frozen = False
    _values_writable = True

    def __init__(
        self,
        values_dict: Union[Mapping[str, float], Sequence[str], None],
        precision: PrecisionDType,
        defaults: Union[Mapping[str, float], Sequence[str], None] = None,
        name: Union[str, None] = None,
        **kwargs: float,
    ) -> None:
        """Initialise the packed values dictionary and array.

        Parameters
        ----------
        values_dict
            Full parameter dictionary or iterable of parameter names. Names
            are expanded to ``0.0`` defaults before precision coercion.
        precision
            Precision used when materialising ``values_array``.
        defaults
            Baseline parameter dictionary applied before ``values_dict``.
        name
            Friendly identifier displayed by ``repr``.
        **kwargs
            Parameter overrides applied after ``values_dict``.

        Notes
        -----
        Keyword arguments replace identically named entries in
        ``values_dict`` and ``defaults``.
        """

        if np_issubdtype(precision, np_integer) or np_issubdtype(
            precision, np_floating
        ):
            self.precision = precision
        else:
            raise TypeError(
                f"precision must be a numpy dtype, you provided a "
                f"{type(precision)}"
            )

        self.values_array = None
        self.indices_dict = None
        self.keys_by_index = None
        # The public mapping is a live read-only view; sanctioned
        # update paths write through the private backing dictionary.
        self._values_backing = {}
        self.values_dict = MappingProxyType(self._values_backing)

        if values_dict is None:
            values_dict = {}
        if defaults is None:
            defaults = {}

        if isinstance(values_dict, (list, tuple)):
            values_dict = {k: 0.0 for k in values_dict}

        if isinstance(defaults, (list, tuple)):
            defaults = {k: 0.0 for k in defaults}

        defaults = self._convert_symbol_keys(defaults)
        values_dict = self._convert_symbol_keys(values_dict)

        # Set default values, then overwrite with values provided in values
        # dict, then any single-parameter keyword arguments.
        combined_updates = {**defaults, **values_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        self._values_backing.update(combined_updates)

        # Initialize values_array and indices_dict
        self.update_param_array_and_indices()

        self.n = len(self.values_array)
        self.name = name

    def __repr__(self) -> str:
        """Return a readable summary of the stored parameter values."""

        if self.name is None:
            name = "System Values"
        else:
            name = self.name
        if all(val == 0.0 for val in self.values_dict.values()):
            return f"{name}: variables ({list(self.values_dict.keys())})"
        return f"{name}: ({self.values_dict})"

    def __eq__(self, other: Any) -> bool:
        """Compare by stored names, values, and precision.

        Value-exact comparison lets configuration snapshots detect
        whether a replacement container actually changed anything.
        """
        if not isinstance(other, SystemValues):
            return NotImplemented
        return (
            self.precision == other.precision
            and self.values_dict == other.values_dict
        )

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    # Mutable + value-equal cannot satisfy the hash contract.
    __hash__ = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject attribute rebinding on a frozen snapshot member."""
        if self._snapshot_frozen:
            raise AttributeError(
                "This SystemValues instance is held by a settings "
                "snapshot and is sealed. Derive a copy() and pass "
                "it through the owning factory's update path."
            )
        super().__setattr__(name, value)

    def freeze(self, values_writable: bool = False) -> "SystemValues":
        """Seal this container as a settings-snapshot member.

        Parameters
        ----------
        values_writable
            When ``True``, stored values remain updatable through
            :meth:`update_from_dict` and the paths built on it while
            the structure (names, precision, packed layout) is
            sealed. When ``False``, values seal too: the packed
            array becomes read-only and every sanctioned value
            update path raises. ``values_dict`` is a read-only view
            at every tier, frozen or not.

        Returns
        -------
        SystemValues
            This instance, sealed in place.

        Notes
        -----
        Configuration snapshots apply this through their field
        converters, so every container a snapshot holds is sealed at
        the write boundary. Values stay writable for containers
        whose stored values are runtime data (parameters, states,
        observables); the constants container seals fully because
        constant values are compile-critical. :meth:`copy` and
        :meth:`with_precision` return unfrozen copies for the
        copy-and-replace update path.
        """
        if self._snapshot_frozen:
            if self._values_writable != values_writable:
                raise ValueError(
                    "This SystemValues instance is already frozen "
                    "with a different value-mutability tier."
                )
            return self
        if not values_writable:
            self.values_array.setflags(write=False)
        object.__setattr__(self, "_values_writable", values_writable)
        object.__setattr__(self, "_snapshot_frozen", True)
        return self

    def _cubie_canonical_(self) -> tuple:
        """Return the canonical structural identity of this container.

        Only compile-critical structure enters the identity: the
        ordered names (sizes and indices are baked into generated
        code) and the precision. Stored values are runtime data —
        constant values, the one value set that is compile-critical,
        are folded into the owning system's ``config_hash``
        separately.
        """
        from numpy import dtype as np_dtype

        return (
            "SystemValues",
            tuple(self.values_dict.keys()),
            np_dtype(self.precision).name,
        )

    def copy(self) -> "SystemValues":
        """Return an independent, unfrozen copy of this container.

        Configuration snapshots seal the instances they hold (see
        :meth:`freeze`): derive a copy, modify the copy, and pass it
        back through the owning factory's update path.
        """
        return SystemValues(
            dict(self.values_dict), self.precision, name=self.name
        )

    def with_precision(self, precision: PrecisionDType) -> "SystemValues":
        """Return a copy of this container at a new precision.

        Parameters
        ----------
        precision
            Precision applied to the copy's packed array.
        """
        return SystemValues(
            dict(self.values_dict), precision, name=self.name
        )

    def _convert_symbol_keys(self, input_dict: Any) -> Any:
        """Return a dictionary whose keys are converted to strings.

        Parameters
        ----------
        input_dict
            Dictionary potentially keyed by :class:`sympy.Symbol` objects.

        Returns
        -------
        Any
            Dictionary with symbol keys converted to strings, or the original
            ``input_dict`` if it is not a dictionary.
        """
        if not isinstance(input_dict, dict):
            return input_dict
        converted: dict[str, float] = {}
        for key, value in input_dict.items():
            if isinstance(key, Symbol):
                converted[str(key)] = value
            elif isinstance(key, str):
                converted[key] = value
        return converted

    def update_param_array_and_indices(self) -> None:
        """Populate ``values_array`` and index mappings from ``values_dict``.

        Notes
        -----
        The mapping order follows dictionary insertion so lookup indices stay
        aligned with ``values_array``.
        """
        keys = list(self.values_dict.keys())
        self.values_array = np_array(
            [self.values_dict[k] for k in keys], dtype=self.precision
        )
        self.indices_dict = {k: i for i, k in enumerate(keys)}
        self.keys_by_index = {i: k for i, k in enumerate(keys)}

    def get_index_of_key(self, parameter_key: str, silent: bool = False) -> int:
        """Return the array index associated with a parameter name.

        Parameters
        ----------
        parameter_key
            Parameter name to locate within ``indices_dict``.
        silent
            When ``True`` missing keys do not trigger an exception.

        Returns
        -------
        int
            Index of ``parameter_key`` within ``values_array``.

        Raises
        ------
        KeyError
            Raised when ``parameter_key`` is not present and ``silent`` is
            ``False``.
        TypeError
            Raised when ``parameter_key`` is not a string.
        """
        if isinstance(parameter_key, str):
            if parameter_key in self.indices_dict:
                return self.indices_dict[parameter_key]
            else:
                if not silent:
                    raise KeyError(
                        f"'{parameter_key}' not found in this SystemValues"
                        f" object. Double check that you're looking in the"
                        f" right place (i.e. states, or parameters, or "
                        f"constants)",
                    )
        else:
            raise TypeError(
                f"parameter_key must be a string, "
                f"you submitted a {type(parameter_key)}."
            )

    def get_indices(
        self,
        keys_or_indices: Union[str, int, slice, list[Union[str, int]], ndarray],
        silent: bool = False,
    ) -> ndarray:
        """Convert parameter identifiers into packed array indices.

        Parameters
        ----------
        keys_or_indices
            Parameter descriptors supplied as names, indices, slices, or
            sequences of either.
        silent
            When ``True`` missing keys do not trigger an exception.

        Returns
        -------
        numpy.ndarray
            Array of ``np_int32`` indices targeting ``values_array``.

        Raises
        ------
        KeyError
            Raised when a requested name is missing and ``silent`` is
            ``False``.
        IndexError
            Raised when an index is outside the valid range.
        TypeError
            Raised when the provided descriptors are of unsupported or mixed
            types.
        """
        if isinstance(keys_or_indices, list):
            if all(isinstance(item, str) for item in keys_or_indices):
                # A list of strings
                index_list = [
                    self.get_index_of_key(state, silent)
                    for state in keys_or_indices
                ]
                # Filter out None values when silent=True
                if silent:
                    index_list = [idx for idx in index_list if idx is not None]
                indices = np_asarray(index_list, dtype=np_int32)
            elif all(isinstance(item, int) for item in keys_or_indices):
                # A list of ints
                indices = np_asarray(keys_or_indices, dtype=np_int32)
            else:
                # List contains mixed types or unsupported types
                non_str_int_types = [
                    type(item)
                    for item in keys_or_indices
                    if not isinstance(item, (str, int))
                ]
                if non_str_int_types:
                    raise TypeError(
                        f"When specifying a variable to save or modify, "
                        f"you can provide strings that match the labels,"
                        f" or integers that match the indices - you "
                        f"provided a list containing"
                        f" {non_str_int_types[0]}",
                    )
                else:
                    raise TypeError(
                        "When specifying a variable to save or modify, "
                        "you can provide a list of strings or a list of "
                        "integers, but not a mixed list of both"
                    )

        elif isinstance(keys_or_indices, str):
            # A single string
            indices = np_asarray(
                [self.get_index_of_key(keys_or_indices)], dtype=np_int32
            )
        elif isinstance(keys_or_indices, int):
            # A single int
            indices = np_asarray([keys_or_indices], dtype=np_int32)

        elif isinstance(keys_or_indices, slice):
            # A slice object
            indices = np_arange(len(self.values_array))[
                keys_or_indices
            ].astype(np_int32)

        elif isinstance(keys_or_indices, ndarray):
            indices = keys_or_indices.astype(np_int32)

        else:
            raise TypeError(
                f"When specifying a variable to save or modify, you can"
                f" provide strings that match the labels,"
                f" or integers that match the indices - you provided a "
                f"{type(keys_or_indices)}"
            )

        if any(
            index < 0 or index >= len(self.values_array) for index in indices
        ):
            raise IndexError(
                f"One or more indices are out of bounds. Valid indices are"
                f" from 0 to {len(self.values_array) - 1}."
            )

        return indices

    def get_values(
        self, keys_or_indices: Union[str, int, list[Union[str, int]], ndarray]
    ) -> ndarray:
        """Return parameter values selected by name or index.

        Parameters
        ----------
        keys_or_indices
            Parameter descriptors accepted by :meth:`get_indices`.

        Returns
        -------
        numpy.ndarray
            Precision-coerced view of the requested parameter values.

        Raises
        ------
        KeyError
            Raised when a requested name is missing.
        IndexError
            Raised when an index is outside the valid range.
        TypeError
            Raised when the descriptors are of unsupported types.
        """
        indices = self.get_indices(keys_or_indices)
        if len(indices) == 1:
            return np_asarray(
                self.values_array[indices[0]], dtype=self.precision
            )
        return np_asarray(
            [self.values_array[index] for index in indices],
            dtype=self.precision,
        )

    def set_values(
        self,
        keys: Union[str, int, slice, list[Union[str, int]], ndarray],
        values: Union[float, Sequence[float], ndarray],
    ) -> None:
        """Assign new values to the selected parameters.

        Parameters
        ----------
        keys
            Parameter descriptors accepted by :meth:`get_indices`.
        values
            Replacement values aligned with ``keys``.

        Raises
        ------
        ValueError
            Raised when the number of values does not match ``keys``.
        """
        indices = self.get_indices(keys)

        # Checks for mismatches between lengths of indices and values
        if len(indices) == 1:
            if isinstance(values, Sized):
                # Check for one key, multiple values
                if len(values) != 1:
                    raise ValueError(
                        "The number of indices does not match the number "
                        "of values provided. "
                    )
                else:
                    updates = {self.keys_by_index[indices[0]]: values[0]}
            else:
                updates = {self.keys_by_index[indices[0]]: values}

        elif not isinstance(values, Sized):
            # Check for two keys, one value
            raise ValueError(
                "The number of indices does not match the number of values"
                " provided. "
            )

        elif len(indices) != len(values):
            raise ValueError(
                "The number of indices does not match the number of values"
                " provided. "
            )
        else:
            updates = {
                self.keys_by_index[index]: value
                for index, value in zip(indices, values)
            }
        self.update_from_dict(updates)

    def update_from_dict(
        self,
        values_dict: Union[Mapping[str, float], None],
        silent: bool = False,
        **kwargs: float,
    ) -> set[str]:
        """Update stored parameter values from dictionaries.

        Parameters
        ----------
        values_dict
            Dictionary of key-value pairs to apply.
        silent
            When ``True`` missing keys do not trigger an exception.
        **kwargs
            Additional key-value pairs to apply after ``values_dict``.

        Returns
        -------
        set[str]
            Keys successfully updated in the stored values.

        Raises
        ------
        KeyError
            Raised when a key is missing and ``silent`` is ``False``.
        TypeError
            Raised when a value cannot be cast to ``precision``.
        ValueError
            Raised when this instance is a fully sealed snapshot
            member (constants held by a settings snapshot).
        """
        if self._snapshot_frozen and not self._values_writable:
            raise ValueError(
                "These values are sealed by a settings snapshot. "
                "Update them through the owning system (for "
                "constants: set_constants() or update()), which "
                "derives a replacement snapshot and rebuilds."
            )
        if values_dict is None:
            values_dict = {}
        if kwargs:
            values_dict.update(kwargs)
        if values_dict == {}:
            return set()

        # Update the dictionary
        unrecognised = [
            k for k in values_dict.keys() if k not in self.indices_dict
        ]
        recognised = {
            k: v for k, v in values_dict.items() if k in self.indices_dict
        }
        if unrecognised:
            if not silent:
                raise KeyError(
                    f"Parameter key(s) {unrecognised} not found in this "
                    f"SystemValues object. Double check that "
                    f"you're looking in the right place (i.e. states"
                    f", or parameters, or constants)",
                )
        if any(
            not isinstance(value, (int, float, np_integer,
                                    np_floating))
            for value in recognised.values()
        ):
            raise TypeError(
                f"One or more values in the provided dictionary cannot be "
                f"cast to the specified precision {self.precision}. "
                f"Please ensure all values are compatible with this "
                f"precision.",
            )
        else:
            # Update the dictionary
            self._values_backing.update(recognised)
            # Update the values_array
            for key, value in recognised.items():
                index = self.get_index_of_key(key, silent=silent)
                self.values_array[index] = value

        return set(values_dict.keys()) - set(unrecognised)

    @property
    def names(self) -> list[str]:
        """List of parameter names."""
        return list(self.values_dict.keys())

    @property
    def as_float_dict(self) -> dict[str, float]:
        """Stored values as plain Python floats keyed by name."""
        return {
            str(name): float(value)
            for name, value in self.values_dict.items()
        }

    @property
    def empty(self) -> bool:
        """Return True if this SystemValues instance has no values."""
        return self.n == 0

    def get_labels(self, indices: Union[list[int], ndarray]) -> list[str]:
        """Return parameter labels for supplied indices.

        Parameters
        ----------
        indices
            Integer indices referencing ``values_array``.

        Returns
        -------
        list[str]
            Labels corresponding to ``indices``.

        Raises
        ------
        TypeError
            Raised when ``indices`` is not a sequence or array.
        """
        if isinstance(indices, (list, ndarray)):
            return [self.keys_by_index[i] for i in indices]
        else:
            raise TypeError(
                f"indices must be a list or numpy array, you provided a "
                f"{type(indices)}."
            )

    def __getitem__(self, key: Union[str, int, slice]) -> ndarray:
        """Return parameter values using dictionary- or array-style access.

        Parameters
        ----------
        key
            Parameter descriptors accepted by :meth:`get_values`.

        Returns
        -------
        numpy.ndarray
            Precision-coerced parameter values selected by ``key``.

        Raises
        ------
        KeyError
            Raised when ``key`` is a missing name.
        IndexError
            Raised when ``key`` references an invalid index.
        TypeError
            Raised when ``key`` is of an unsupported type.
        """
        return self.get_values(key)

    def __setitem__(
        self,
        key: Union[str, int, slice],
        value: Union[float, Sequence[float], ndarray],
    ) -> None:
        """Update parameter values using dictionary- or array-style access.

        Parameters
        ----------
        key
            Parameter descriptor accepted by :meth:`set_values`.
        value
            Replacement value or sequence aligned with ``key``.

        Raises
        ------
        KeyError
            Raised when ``key`` is a missing name.
        IndexError
            Raised when ``key`` references an invalid index.
        TypeError
            Raised when ``key`` is of an unsupported type.

        Notes
        -----
        Both indexing methods update ``values_dict`` and ``values_array``.
        """
        self.set_values(key, value)

    def add_entry(self, name: str, value: float = 0.0) -> None:
        """Add a new entry to the values collection.

        Parameters
        ----------
        name
            Name of the new entry.
        value
            Initial value for the entry.

        Raises
        ------
        ValueError
            If the name already exists or this instance is a sealed
            snapshot member.
        """
        if self._snapshot_frozen:
            raise ValueError(
                "The structure of this SystemValues is sealed by a "
                "settings snapshot. Derive a copy() and pass it "
                "through the owning factory's update path."
            )
        if name in self.values_dict:
            raise ValueError(f"Entry '{name}' already exists")

        self._values_backing[name] = self.precision(value)
        self.update_param_array_and_indices()
        self.n = len(self.values_array)

    def remove_entry(self, name: str) -> float:
        """Remove an entry from the values collection.

        Parameters
        ----------
        name
            Name of the entry to remove.

        Returns
        -------
        float
            The value that was removed.

        Raises
        ------
        KeyError
            If the name does not exist.
        ValueError
            If this instance is a sealed snapshot member.
        """
        if self._snapshot_frozen:
            raise ValueError(
                "The structure of this SystemValues is sealed by a "
                "settings snapshot. Derive a copy() and pass it "
                "through the owning factory's update path."
            )
        if name not in self.values_dict:
            raise KeyError(f"Entry '{name}' not found")

        value = self._values_backing.pop(name)
        self.update_param_array_and_indices()
        self.n = len(self.values_array)
        return value
