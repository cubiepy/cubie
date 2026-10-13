"""The states, parameters and observables a system declares.

Published Classes
-----------------
:class:`Variable`
    A named, unitful quantity with no value: an observable or a driver.

:class:`State`
    A state variable and its default initial value.

:class:`Parameter`
    A parameter and its default value.

:class:`VariableSet`
    An ordered, immutable set of variables with name and index lookup.

    >>> observables = VariableSet.of(Variable, ["v", "w"])
    >>> observables.get_indices(["w"])
    array([1], dtype=int32)

:class:`ValuedSet`
    A :class:`VariableSet` of states or parameters, which adds their
    defaults.

    >>> states = VariableSet.of(State, {"x": 1.0, "y": 0.0})
    >>> states.defaults
    {'x': 1.0, 'y': 0.0}

:class:`CompiledParameter`
    A parameter as a batch compiles it: swept, or fixed at a value.

:class:`ParameterSet`
    A :class:`VariableSet` of compiled parameters, which says what a
    batch sweeps and the value it compiles in for every other parameter.

    >>> parameters = VariableSet.of(Parameter, {"a": 1.0, "b": 2.0})
    >>> batch = ParameterSet.build(parameters, ["b"], {"a": 3.0})
    >>> batch.swept, batch.fixed
    (('b',), {'a': 3.0})

See Also
--------
:class:`~cubie.odesystems.ODEData.ODEData`
    Compile settings that hold a system's variable sets.
"""

from collections.abc import Mapping
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from attrs import evolve, field, frozen
from numpy import (
    arange as np_arange,
    array as np_array,
    asarray as np_asarray,
    int32 as np_int32,
    ndarray,
)

from cubie._utils import PrecisionDType

DIMENSIONLESS = "dimensionless"


@frozen
class Variable:
    """A named quantity the equations read or produce.

    Parameters
    ----------
    name
        Symbol name, unique within its set.
    unit
        Unit string, shown in result legends.
    """

    name: str = field(converter=str)
    unit: str = field(default=DIMENSIONLESS, converter=str)


@frozen
class State(Variable):
    """A state variable.

    Parameters
    ----------
    default
        Initial value a run starts from when the caller gives none.
    """

    default: float = field(default=0.0, converter=float)


@frozen
class Parameter(Variable):
    """A parameter.

    Parameters
    ----------
    default
        Value used when a batch neither sweeps the parameter nor gives
        it a value.
    """

    default: float = field(default=0.0, converter=float)


def _optional_float(value: Optional[float]) -> Optional[float]:
    """Return ``value`` as a float, leaving ``None`` as it is."""
    return None if value is None else float(value)


@frozen
class CompiledParameter(Variable):
    """A parameter as a batch compiles it.

    We give a swept parameter no value, because each run reads it from
    the parameters array. We give every other parameter the value we
    compile into the code.

    Parameters
    ----------
    swept
        Whether runs read the parameter from the parameters array.
    value
        The value compiled in, or ``None`` for a swept parameter.

    Raises
    ------
    ValueError
        If a swept parameter has a value or a fixed one has none.
    """

    swept: bool = field(default=False, converter=bool)
    value: Optional[float] = field(default=None, converter=_optional_float)

    def __attrs_post_init__(self) -> None:
        if self.swept != (self.value is None):
            raise ValueError(
                f"Parameter '{self.name}' must be swept or have a value, "
                f"not both or neither."
            )


def _unique_entries(instance: Any, attribute: Any, entries: tuple) -> None:
    """Reject a set that names a variable twice."""
    names = [entry.name for entry in entries]
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Variables {repeated} are declared more than once.")


@frozen(hash=False)
class VariableSet:
    """An ordered, immutable set of variables.

    The order is the layout of every array that holds one value per
    variable, so a variable's index is its row in those arrays.

    Parameters
    ----------
    entries
        The variables, in array order.
    """

    entries: Tuple[Variable, ...] = field(
        converter=tuple, validator=_unique_entries
    )
    _index: Dict[str, int] = field(init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self,
            "_index",
            {entry.name: row for row, entry in enumerate(self.entries)},
        )

    @classmethod
    def of(
        cls,
        kind: Type[Variable],
        values: Union[Mapping[str, float], Iterable[str], None],
        units: Optional[Mapping[str, str]] = None,
    ) -> "VariableSet":
        """Build a set of ``kind`` records from names or defaults.

        Parameters
        ----------
        kind
            :class:`Variable`, :class:`State` or :class:`Parameter`.
        values
            Names mapped to defaults, or bare names, in array order.
            Bare names of a valued kind default to ``0.0``.
        units
            Unit strings by name; names left out are dimensionless.

        Returns
        -------
        VariableSet
            A :class:`ValuedSet` for states or parameters, and a plain
            set for valueless variables.
        """
        units = units or {}
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            values = {str(name): 0.0 for name in values}
        entries = []
        for name, default in values.items():
            unit = units.get(str(name), DIMENSIONLESS)
            if kind is Variable:
                entries.append(Variable(name, unit))
            else:
                entries.append(kind(name, unit, float(default)))
        if kind is Variable:
            return VariableSet(entries)
        return ValuedSet(entries)

    def _cubie_canonical_(self) -> tuple:
        """Return the names, which fix every array layout."""
        return ("VariableSet", self.names)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Variable]:
        return iter(self.entries)

    def __contains__(self, name: object) -> bool:
        return name in self._index

    def __getitem__(self, key: Union[str, int]) -> Variable:
        """Return the variable named ``key``, or at row ``key``."""
        if isinstance(key, str):
            return self.entries[self.index(key)]
        return self.entries[key]

    @property
    def names(self) -> Tuple[str, ...]:
        """Variable names, in array order."""
        return tuple(self._index)

    @property
    def n(self) -> int:
        """Number of variables."""
        return len(self.entries)

    @property
    def empty(self) -> bool:
        """Whether the set holds no variables."""
        return not self.entries

    @property
    def units(self) -> Dict[str, str]:
        """Unit string of each variable, by name."""
        return {entry.name: entry.unit for entry in self.entries}

    def subset(self, names: Iterable[str]) -> "VariableSet":
        """Return the named variables as a set, in this set's order.

        Parameters
        ----------
        names
            Names of variables in this set, in any order.

        Raises
        ------
        KeyError
            If a name is not in the set.
        """
        names = set(names)
        unknown = names - set(self._index)
        if unknown:
            raise KeyError(
                f"{sorted(unknown)} are not variables of this set."
            )
        return type(self)(
            [entry for entry in self.entries if entry.name in names]
        )

    def without_values(self) -> "VariableSet":
        """Return the names and units of this set, with no defaults.

        We compile a system against this view, so changing a default
        does not change the build.
        """
        return VariableSet(
            [Variable(entry.name, entry.unit) for entry in self.entries]
        )

    def index(self, name: str) -> int:
        """Return the row of the variable called ``name``.

        Raises
        ------
        KeyError
            If no variable has that name.
        """
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(
                f"'{name}' is not one of {list(self.names)}. Check that "
                f"you're looking in the right place (states, parameters "
                f"or observables)."
            ) from None

    def get_indices(
        self,
        keys_or_indices: Union[
            str, int, slice, List[Union[str, int]], ndarray
        ],
        silent: bool = False,
    ) -> ndarray:
        """Convert names, indices or a slice into an index array.

        Parameters
        ----------
        keys_or_indices
            A name, an index, a slice, or a sequence of names or of
            indices.
        silent
            Skip names that are not in the set instead of raising.

        Returns
        -------
        numpy.ndarray
            ``int32`` row indices.

        Raises
        ------
        KeyError
            If a name is not in the set and ``silent`` is ``False``.
        IndexError
            If an index is out of range.
        TypeError
            If the input mixes names and indices or has another type.
        """
        if isinstance(keys_or_indices, (list, tuple)):
            if all(isinstance(item, str) for item in keys_or_indices):
                if silent:
                    rows = [
                        self._index[name]
                        for name in keys_or_indices
                        if name in self._index
                    ]
                else:
                    rows = [self.index(name) for name in keys_or_indices]
                indices = np_asarray(rows, dtype=np_int32)
            elif all(isinstance(item, int) for item in keys_or_indices):
                indices = np_asarray(keys_or_indices, dtype=np_int32)
            else:
                kinds = sorted(
                    {type(item).__name__ for item in keys_or_indices}
                )
                raise TypeError(
                    f"Give a list of names or a list of indices, not a "
                    f"mixture of {kinds}."
                )
        elif isinstance(keys_or_indices, str):
            indices = np_asarray(
                [self.index(keys_or_indices)], dtype=np_int32
            )
        elif isinstance(keys_or_indices, int):
            indices = np_asarray([keys_or_indices], dtype=np_int32)
        elif isinstance(keys_or_indices, slice):
            indices = np_arange(self.n)[keys_or_indices].astype(np_int32)
        elif isinstance(keys_or_indices, ndarray):
            indices = keys_or_indices.astype(np_int32)
        else:
            raise TypeError(
                "Give names or indices of variables, not a "
                f"{type(keys_or_indices).__name__}."
            )
        if any(index < 0 or index >= self.n for index in indices):
            raise IndexError(
                f"Indices must lie in 0 to {self.n - 1}; got "
                f"{indices.tolist()}."
            )
        return indices

    def get_labels(self, indices: Union[List[int], ndarray]) -> List[str]:
        """Return the names at ``indices``."""
        return [self.entries[int(index)].name for index in indices]


def _valued_entries(instance: Any, attribute: Any, entries: tuple) -> None:
    """Reject entries that carry no default."""
    valueless = [
        entry.name
        for entry in entries
        if not isinstance(entry, (State, Parameter))
    ]
    if valueless:
        raise TypeError(
            f"Variables {valueless} have no default; a ValuedSet holds "
            f"states or parameters."
        )


@frozen(hash=False)
class ValuedSet(VariableSet):
    """A set of states or parameters, with their defaults.

    Parameters
    ----------
    entries
        The :class:`State` or :class:`Parameter` records, in array
        order.
    """

    entries: Tuple[Variable, ...] = field(
        converter=tuple, validator=[_unique_entries, _valued_entries]
    )

    @property
    def defaults(self) -> Dict[str, float]:
        """Default value of each variable, by name, in array order."""
        return {entry.name: entry.default for entry in self.entries}

    def defaults_array(self, precision: PrecisionDType) -> ndarray:
        """Return the defaults as an array of ``precision``."""
        return np_array(
            [entry.default for entry in self.entries], dtype=precision
        )

    def with_defaults(self, values: Mapping[str, float]) -> "ValuedSet":
        """Return a copy with new defaults for the named variables.

        Parameters
        ----------
        values
            Names mapped to new default values.

        Raises
        ------
        KeyError
            If a name is not in the set.
        """
        unknown = [name for name in values if name not in self._index]
        if unknown:
            raise KeyError(
                f"{sorted(unknown)} are not variables of this set."
            )
        entries = tuple(
            evolve(entry, default=values[entry.name])
            if entry.name in values
            else entry
            for entry in self.entries
        )
        return evolve(self, entries=entries)


@frozen(hash=False)
class ParameterSet(VariableSet):
    """The parameters as a batch compiles them.

    We hold one :class:`CompiledParameter` per parameter, in the
    system's parameter order, and fill the parameters array with one row
    per swept parameter in that same order. We keep no defaults here, so
    a default reaches the build only as the value of a fixed parameter.

    Parameters
    ----------
    entries
        One :class:`CompiledParameter` per parameter, in system order.
    """

    @classmethod
    def build(
        cls,
        parameters: ValuedSet,
        swept: Iterable[str] = (),
        values: Optional[Mapping[str, float]] = None,
    ) -> "ParameterSet":
        """Compile ``parameters`` for a batch.

        Parameters
        ----------
        parameters
            The declared parameters with their defaults.
        swept
            Names of the parameters to sweep, in any order. The rows of
            the parameters array follow the system's order.
        values
            Values to compile in for some of the other parameters.
            Every parameter neither swept nor named here compiles in at
            its default.

        Returns
        -------
        ParameterSet
            The compiled parameters.

        Raises
        ------
        KeyError
            If a name is not a parameter.
        ValueError
            If a swept name repeats or also has a value.
        """
        swept = tuple(swept)
        values = values or {}
        if len(set(swept)) != len(swept):
            raise ValueError(f"Swept parameters {list(swept)} repeat.")
        unknown = [
            name for name in (*swept, *values) if name not in parameters
        ]
        if unknown:
            raise KeyError(
                f"{sorted(unknown)} are not parameters of this system."
            )
        clashing = sorted(set(values) & set(swept))
        if clashing:
            raise ValueError(
                f"{clashing} are swept, so they have no fixed value."
            )
        swept = set(swept)
        return cls(
            [
                CompiledParameter(entry.name, entry.unit, swept=True)
                if entry.name in swept
                else CompiledParameter(
                    entry.name,
                    entry.unit,
                    value=values.get(entry.name, entry.default),
                )
                for entry in parameters
            ]
        )

    def _cubie_canonical_(self) -> tuple:
        """Return the names, the sweep and every compiled-in value."""
        return (
            "ParameterSet",
            tuple(
                (entry.name, entry.swept, entry.value)
                for entry in self.entries
            ),
        )

    @property
    def swept(self) -> Tuple[str, ...]:
        """Names of the swept parameters, in row order."""
        return tuple(entry.name for entry in self.entries if entry.swept)

    @property
    def fixed(self) -> Dict[str, float]:
        """The value compiled in for every parameter that is not swept."""
        return {
            entry.name: entry.value
            for entry in self.entries
            if not entry.swept
        }
