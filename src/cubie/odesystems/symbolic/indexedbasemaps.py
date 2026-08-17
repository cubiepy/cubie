"""Helpers that build indexed SymPy arrays for symbolic ODE metadata.

Published Classes
-----------------
:class:`IndexedBaseMap`
    Map named scalar symbols onto a fixed-size SymPy indexed base.

    >>> ibm = IndexedBaseMap("state", ["x", "y"])
    >>> ibm.length
    2

:class:`IndexedBases`
    Bundle of :class:`IndexedBaseMap` instances describing a full ODE
    system (states, parameters, constants, observables, drivers, dxdt).

    >>> ib = IndexedBases.from_user_inputs(
    ...     states=["x"], parameters=["k"], constants=["g"],
    ...     observables=["v"], drivers=["u"],
    ... )
    >>> ib.state_names
    ['x']

See Also
--------
:mod:`cubie.odesystems.symbolic.codegen`
    Code generation modules that consume ``IndexedBases`` for CUDA
    array references.
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE`
    Stores an ``IndexedBases`` instance as ``self.indices``.
"""

from typing import Any, Dict, Iterable, Optional, Union

import sympy as sp

from cubie.odesystems.symbolic.sym_utils import RESERVED_CODEGEN_PREFIX


def _reorder(values, order):
    """Permute a sequence by ``order``; pass dicts and None through."""
    if values is None or isinstance(values, dict):
        return values
    values = list(values)
    if len(values) != len(order):
        return values
    return [values[index] for index in order]


class IndexedBaseMap:
    """Map named symbols onto a SymPy indexed base, sorted by name."""

    def __init__(
        self,
        base_name: str,
        symbol_labels: Iterable[str],
        input_defaults: Optional[Iterable[Any]] = None,
        length: int = 0,
        real: bool = True,
        units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    ) -> None:
        """Initialise an indexed base with optional default values.

        Parameters
        ----------
        base_name
            Base symbol name used for the generated :class:`sympy.IndexedBase`.
        symbol_labels
            Symbol names that define the entries in the indexed base.
        input_defaults
            Optional default numeric values aligned with ``symbol_labels``.
        length
            Length override for the indexed base when ``symbol_labels`` is
            provided as an iterator.
        real
            Whether to create real-only symbols in the indexed base.
        units
            Optional units specification aligned with ``symbol_labels``.
            Can be a dictionary mapping symbol names to unit strings,
            or an iterable of unit strings. If None, defaults to
            "dimensionless" for all symbols.
        """
        labels = list(symbol_labels)
        order = sorted(range(len(labels)), key=labels.__getitem__)
        input_defaults = _reorder(input_defaults, order)
        units = _reorder(units, order)
        labels = [labels[index] for index in order]
        if length == 0:
            length = len(labels)

        self.length = length
        self.base_name = base_name
        self.real = real
        self.base = sp.IndexedBase(base_name, shape=(length,), real=real)
        self.index_map = {
            sp.Symbol(name, real=real): index
            for index, name in enumerate(labels)
        }
        self.ref_map = {
            sp.Symbol(name, real=real): self.base[index]
            for index, name in enumerate(labels)
        }
        self.symbol_map = {name: sp.Symbol(name, real=real) for name in labels}

        self._passthrough_defaults = False
        if input_defaults is None:
            defaults = [0.0] * length
            self.default_values = dict(zip(self.ref_map.keys(), defaults))
            self.defaults = {
                str(symbol): value
                for symbol, value in self.default_values.items()
            }
        elif isinstance(input_defaults, dict):
            self._passthrough_defaults = True
            self.default_values = input_defaults
            self.defaults = input_defaults
        else:
            defaults = list(input_defaults)
            if len(defaults) != length:
                raise ValueError(
                    "Input defaults must be the same length as the list of "
                    "symbols"
                )
            self.default_values = dict(zip(self.ref_map.keys(), defaults))
            self.defaults = {
                str(symbol): value
                for symbol, value in self.default_values.items()
            }

        if units is None:
            self.units = {name: "dimensionless" for name in labels}
        elif isinstance(units, dict):
            self.units = {
                name: units.get(name, "dimensionless") for name in labels
            }
        else:
            units_list = list(units)
            if len(units_list) != length:
                raise ValueError(
                    "Units must be the same length as the list of symbols"
                )
            self.units = dict(zip(labels, units_list))

    def pop(self, sym: sp.Symbol) -> None:
        """Remove a symbol from the indexed base.

        Surviving symbols are reindexed so every reference stays
        within the shrunken base's bounds.

        Parameters
        ----------
        sym
            Symbol to remove from the map.
        """
        self.ref_map.pop(sym)
        self.index_map.pop(sym)
        sym_str = str(sym)
        self.symbol_map.pop(sym_str)
        if not self._passthrough_defaults:
            self.default_values.pop(sym)
            self.defaults.pop(sym_str)
        self.units.pop(sym_str, None)
        self.base = sp.IndexedBase(
            self.base_name, shape=(len(self.ref_map),), real=self.real
        )
        self.length = len(self.ref_map)
        for index, existing in enumerate(self.ref_map):
            self.index_map[existing] = index
            self.ref_map[existing] = self.base[index]

    def push(self, sym: sp.Symbol, default_value: float = 0.0, unit: str = "dimensionless") -> None:
        """Insert a symbol at its sorted position, reindexing the rest.

        Parameters
        ----------
        sym
            Symbol to insert.
        default_value
            Default numeric value for the new entry.
        unit
            Unit string for the new entry.
        """
        sym_str = str(sym)
        symbols = dict(self.symbol_map)
        symbols[sym_str] = sym
        names = sorted(symbols)

        self.length = len(names)
        self.base = sp.IndexedBase(
            self.base_name, shape=(self.length,), real=self.real
        )
        self.symbol_map = {name: symbols[name] for name in names}
        self.index_map = {
            symbols[name]: index for index, name in enumerate(names)
        }
        self.ref_map = {
            symbols[name]: self.base[index]
            for index, name in enumerate(names)
        }
        if not self._passthrough_defaults:
            self.default_values[sym] = default_value
            self.defaults[sym_str] = default_value
            self.default_values = {
                symbols[name]: self.default_values[symbols[name]]
                for name in names
            }
            self.defaults = {name: self.defaults[name] for name in names}
        self.units[sym_str] = unit
        self.units = {
            name: self.units.get(name, "dimensionless") for name in names
        }

    def update_values(
        self,
        updates_dict: Optional[Dict[Union[str, sp.Symbol], float]] = None,
        **kwargs: float,
    ) -> None:
        """Update the stored default values.

        Parameters
        ----------
        updates_dict
            Mapping of symbol names or symbols to replacement values.
        **kwargs
            Additional symbol updates provided as keyword arguments. Entries
            take precedence over those in ``updates_dict``.

        Notes
        -----
        Silently ignores keys that are not found in the indexed base map.
        """
        if self._passthrough_defaults:
            return

        if updates_dict is None:
            updates_dict = {}
        updates_dict = updates_dict.copy()
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return

        if any(isinstance(key, sp.Symbol) for key in updates_dict.keys()):
            symbol_update_dict = {
                key: value
                for key, value in updates_dict.items()
                if key in self.ref_map
            }
        else:
            symbol_update_dict = {
                self.symbol_map[key]: value
                for key, value in updates_dict.items()
                if key in self.symbol_map
            }

        for sym, val in symbol_update_dict.items():
            self.default_values[sym] = val
            self.defaults[str(sym)] = val
        return

    def set_passthrough_defaults(self, defaults: dict[str, Any]) -> None:
        """Replace defaults with a direct dictionary mapping.

        Parameters
        ----------
        defaults
            Dictionary used directly as both ``default_values`` and
            ``defaults``.
        """

        self._passthrough_defaults = True
        self.default_values = defaults
        self.defaults = defaults


class IndexedBases:
    """Bundle of indexed bases describing a symbolic ODE definition."""

    def __init__(
        self,
        states: IndexedBaseMap,
        parameters: IndexedBaseMap,
        constants: IndexedBaseMap,
        observables: IndexedBaseMap,
        drivers: IndexedBaseMap,
        dxdt: IndexedBaseMap,
    ) -> None:
        """Initialise the combined index maps.

        Parameters
        ----------
        states
            Indexed base describing the system state vector.
        parameters
            Indexed base describing tunable model parameters.
        constants
            Indexed base describing compile-time constants.
        observables
            Indexed base describing recorded observables.
        drivers
            Indexed base describing driver signals.
        dxdt
            Indexed base describing the ``dx/dt`` outputs.
        """
        self.states = states
        self.parameters = parameters
        self.constants = constants
        self.observables = observables
        self.drivers = drivers
        self.dxdt = dxdt
        self.all_indices = {
            **self.states.ref_map,
            **self.parameters.ref_map,
            **self.observables.ref_map,
            **self.drivers.ref_map,
            **self.dxdt.ref_map,
        }

    @classmethod
    def from_user_inputs(
        cls,
        states: Union[dict[str, float], Iterable[str]],
        parameters: Union[dict, Iterable[str]],
        constants: Union[dict, Iterable[str]],
        observables: Iterable[str],
        drivers: Iterable[str],
        real: bool = True,
        state_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
        parameter_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
        constant_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
        observable_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
        driver_units: Optional[Union[Dict[str, str], Iterable[str]]] = None,
    ) -> "IndexedBases":
        """Construct indexed bases from user-provided metadata.

        Parameters
        ----------
        states
            Either a mapping of state names to default values or an iterable
            of state names.
        parameters
            Either a mapping of parameter names to default values or an
            iterable of parameter names.
        constants
            Either a mapping of constant names to default values or an
            iterable of constant names.
        observables
            Iterable of observable names.
        drivers
            Iterable of driver names.
        real
            Whether to constrain the generated symbols to real values.
        state_units
            Optional units for states. Defaults to "dimensionless".
        parameter_units
            Optional units for parameters. Defaults to "dimensionless".
        constant_units
            Optional units for constants. Defaults to "dimensionless".
        observable_units
            Optional units for observables. Defaults to "dimensionless".
        driver_units
            Optional units for drivers. Defaults to "dimensionless".

        Returns
        -------
        IndexedBases
            Combined bundle of indexed bases for the symbolic ODE system.
        """
        for group in (states, parameters, constants, observables,
                      drivers):
            for name in group:
                if str(name).startswith(RESERVED_CODEGEN_PREFIX):
                    raise ValueError(
                        f"Name '{name}' is reserved: user symbols "
                        "cannot start with "
                        f"'{RESERVED_CODEGEN_PREFIX}'."
                    )

        if isinstance(states, dict):
            state_names = list(states.keys())
            state_defaults = [states[name] for name in state_names]
        else:
            state_names = list(states)
            state_defaults = None

        if isinstance(parameters, dict):
            param_names = list(parameters.keys())
            param_defaults = [parameters[name] for name in param_names]
        else:
            param_names = list(parameters)
            param_defaults = None

        if isinstance(constants, dict):
            const_names = list(constants.keys())
            const_defaults = [constants[name] for name in const_names]
        else:
            const_names = list(constants)
            const_defaults = None

        states_ = IndexedBaseMap(
            "state", state_names, input_defaults=state_defaults, real=real,
            units=state_units
        )
        parameters_ = IndexedBaseMap(
            "parameters", param_names, input_defaults=param_defaults, real=real,
            units=parameter_units
        )
        constants_ = IndexedBaseMap(
            "constants", const_names, input_defaults=const_defaults, real=real,
            units=constant_units
        )
        observables_ = IndexedBaseMap("observables", observables, real=real,
                                      units=observable_units)
        drivers_ = IndexedBaseMap("drivers", drivers, real=real,
                                 units=driver_units)
        dxdt_ = IndexedBaseMap(
            "out", [f"d{s}" for s in state_names], real=real
        )
        return cls(
            states_, parameters_, constants_, observables_, drivers_, dxdt_
        )

    def update_constants(
        self, updates_dict: Optional[Dict[str, float]] = None, **kwargs: float
    ) -> None:
        """Update the constant defaults while preserving other entries.

        Parameters
        ----------
        updates_dict
            Mapping of constant names to replacement values.
        **kwargs
            Additional constant updates provided as keyword arguments.

        Notes
        -----
        Silently ignores keys that are not found in the constants symbol table.
        """
        self.constants.update_values(updates_dict, **kwargs)

    @property
    def state_names(self) -> list[str]:
        """List of state symbol names."""
        return list(self.states.symbol_map.keys())

    @property
    def state_values(self) -> Dict[sp.Symbol, float]:
        """Mapping of state symbols to default values."""
        return self.states.default_values

    @property
    def parameter_names(self) -> list[str]:
        """List of parameter symbol names."""
        return list(self.parameters.symbol_map.keys())

    @property
    def parameter_symbols(self) -> list[sp.Symbol]:
        """List of parameter symbols."""
        return list(self.parameters.ref_map.keys())

    @property
    def parameter_values(self) -> Dict[sp.Symbol, float]:
        """Mapping of parameter symbols to default values."""
        return self.parameters.default_values

    @property
    def constant_names(self) -> list[str]:
        """List of constant symbol names."""
        return list(self.constants.symbol_map.keys())

    @property
    def constant_values(self) -> Dict[sp.Symbol, float]:
        """Mapping of constant symbols to default values."""
        return self.constants.default_values

    @property
    def observable_names(self) -> list[str]:
        """List of observable symbol names."""
        return list(self.observables.symbol_map.keys())

    @property
    def observable_symbols(self) -> list[sp.Symbol]:
        """List of observable symbols."""
        return list(self.observables.ref_map.keys())

    @property
    def driver_names(self) -> list[str]:
        """List of driver symbol names."""
        return list(self.drivers.symbol_map.keys())

    @property
    def dxdt_names(self) -> Iterable[str]:
        """List of ``dx/dt`` output symbol names."""
        return list(self.dxdt.symbol_map.keys())

    @property
    def all_arrayrefs(self) -> dict[str, sp.Symbol]:
        """Dictionary of all indexed base references keyed by symbol."""
        return {
            **self.states.ref_map,
            **self.parameters.ref_map,
            **self.observables.ref_map,
            **self.drivers.ref_map,
            **self.dxdt.ref_map,
        }

    @property
    def all_symbols(self) -> dict[str, sp.Symbol]:
        """Dictionary of all scalar symbols keyed by name."""
        return {
            **self.states.symbol_map,
            **self.parameters.symbol_map,
            **self.constants.symbol_map,
            **self.observables.symbol_map,
            **self.drivers.symbol_map,
            **self.dxdt.symbol_map,
        }

    def __getitem__(self, item: sp.Symbol) -> sp.Symbol:
        """Return the indexed reference associated with ``item``."""
        return self.all_indices[item]

    def _refresh_all_indices(self) -> None:
        """Rebuild the combined index map after structural changes."""
        self.all_indices = {
            **self.states.ref_map,
            **self.parameters.ref_map,
            **self.observables.ref_map,
            **self.drivers.ref_map,
            **self.dxdt.ref_map,
        }

    def constant_to_parameter(self, name: str) -> None:
        """Convert a constant to a parameter.

        Parameters
        ----------
        name
            Name of the constant to convert.

        Raises
        ------
        KeyError
            If the name is not found in constants.
        """
        if name not in self.constants.symbol_map:
            raise KeyError(
                f"'{name}' is not a constant. Available constants: "
                f"{list(self.constants.symbol_map.keys())}"
            )

        sym = self.constants.symbol_map[name]
        value = self.constants.defaults.get(name, 0.0)
        unit = self.constants.units.get(name, "dimensionless")

        self.constants.pop(sym)
        self.parameters.push(sym, default_value=value, unit=unit)
        self._refresh_all_indices()

    def parameter_to_constant(self, name: str) -> None:
        """Convert a parameter to a constant.

        Parameters
        ----------
        name
            Name of the parameter to convert.

        Raises
        ------
        KeyError
            If the name is not found in parameters.
        """
        if name not in self.parameters.symbol_map:
            raise KeyError(
                f"'{name}' is not a parameter. Available parameters: "
                f"{list(self.parameters.symbol_map.keys())}"
            )

        sym = self.parameters.symbol_map[name]
        value = self.parameters.defaults.get(name, 0.0)
        unit = self.parameters.units.get(name, "dimensionless")

        self.parameters.pop(sym)
        self.constants.push(sym, default_value=value, unit=unit)
        self._refresh_all_indices()
