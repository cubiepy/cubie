"""Parameter-symbolic checkpoint and the binding specialisation pass."""

from typing import Any, Callable, Dict, Iterable, List, Optional

import attrs
import sympy as sp

from cubie.odesystems.ODEData import ParameterBinding
from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.parsing.assemble import assemble_simplified
from cubie.odesystems.symbolic.parsing.normalise import (
    NormalisedSystem,
    normalise_input,
)
from cubie.odesystems.symbolic.sym_utils import RESERVED_CODEGEN_PREFIX


def _literal_rules(
    fixed_values: Dict[str, float],
) -> Dict[ir.Expr, ir.Expr]:
    """Return the IR substitution map for ``fixed_values``."""

    return {
        ir.sym(str(name)): ir.num(float(value))
        for name, value in fixed_values.items()
    }


@attrs.define
class ParsedSystem:
    """Parameter-symbolic checkpoint of a parsed system.

    Parameters
    ----------
    normalised
        The :class:`~.normalise.NormalisedSystem`, parameters symbolic.
    states
        Declared plus inferred states mapped to initial values.
    observables
        Declared observable names.
    parameters
        Parameter names mapped to default values.
    driver_names, driver_dict
        Driver labels and the optional driver settings dictionary.
    known_symbol_map
        Name-to-SymPy-symbol map for the immutable inputs.
    user_functions, user_function_derivatives
        Equation callables and their analytic derivative helpers.
    state_priority, irreducible, simplify_options
        Options forwarded to
        :func:`~..structural.simplify.structural_simplify`.
    state_units, parameter_units, observable_units, driver_units
        Unit annotations forwarded to the assembler.
    """

    normalised: Any
    states: Dict[str, float]
    observables: List[str]
    parameters: Dict[str, float]
    driver_names: List[str]
    driver_dict: Optional[Dict[str, Any]]
    known_symbol_map: Dict[str, Any]
    user_functions: Optional[Dict[str, Callable]]
    user_function_derivatives: Optional[Dict[str, Callable]]
    state_priority: Optional[Dict[str, float]] = None
    irreducible: Optional[Iterable[str]] = None
    simplify_options: Optional[Dict[str, Any]] = None
    state_units: Any = None
    parameter_units: Any = None
    observable_units: Any = None
    driver_units: Any = None

    def __attrs_post_init__(self):
        for name in self.parameters:
            if str(name).startswith(RESERVED_CODEGEN_PREFIX):
                raise ValueError(
                    f"Name '{name}' is reserved: user symbols cannot "
                    f"start with '{RESERVED_CODEGEN_PREFIX}'."
                )

    @classmethod
    def from_parsed_equations(
        cls,
        equations,
        index_map,
        user_functions: Optional[Dict[str, Callable]] = None,
        user_function_derivatives: Optional[Dict[str, Callable]] = None,
    ) -> "ParsedSystem":
        """Build a checkpoint from pre-parsed equation products.

        Parameters
        ----------
        equations
            The system's :class:`~.parser.ParsedEquations`.
        index_map
            The system's :class:`~..indexedbasemaps.IndexedBases`.
        user_functions, user_function_derivatives
            Equation callables and their analytic derivative helpers.
        """

        states = {
            str(name): float(value)
            for name, value in index_map.state_values.items()
        }
        parameters = {
            str(name): float(value)
            for name, value in index_map.parameter_values.items()
        }
        observables = list(index_map.observable_names)
        driver_defaults = {
            str(name): value
            for name, value in index_map.drivers.default_values.items()
        }
        driver_names = list(driver_defaults)
        known_symbol_map = {
            name: sp.Symbol(name, real=True)
            for name in list(parameters) + driver_names
        }
        unknown_names = set(states) | set(observables)
        normalised = normalise_input(
            list(equations.ordered),
            unknown_names,
            known_symbol_map,
            user_functions,
            user_function_derivatives,
            False,
            set(states),
        )
        normalised.derivative_names.update(equations.derivative_names)
        return cls(
            normalised=normalised,
            states=states,
            observables=observables,
            parameters=parameters,
            driver_names=driver_names,
            driver_dict=driver_defaults or None,
            known_symbol_map=known_symbol_map,
            user_functions=user_functions,
            user_function_derivatives=user_function_derivatives,
            state_units=index_map.states.units or None,
            parameter_units=index_map.parameters.units or None,
            observable_units=index_map.observables.units or None,
            driver_units=index_map.drivers.units or None,
        )

    def default_binding(self) -> ParameterBinding:
        """Return the binding that fixes every parameter at its default."""

        return ParameterBinding(fixed=self.parameters)

    def specialise(
        self,
        binding: Optional[ParameterBinding] = None,
        state_values: Optional[Dict[str, float]] = None,
    ):
        """Assemble the system for one parameter binding.

        Parameters
        ----------
        binding
            Swept names and fixed values; ``None`` fixes every
            parameter at its default.
        state_values
            Overrides for declared-state initial values.

        Returns
        -------
        tuple
            ``(index_map, all_symbols, funcs, parsed_equations,
            fn_hash)``; ``index_map.parameters`` holds the swept
            parameters and the derived mass matrix rides on
            ``parsed_equations.mass_matrix``.
        """

        if binding is None:
            binding = self.default_binding()
        if set(binding.names) != set(self.parameters):
            raise KeyError(
                f"Binding names {list(binding.names)} do not match the "
                f"system's parameters {sorted(self.parameters)}."
            )

        states = dict(self.states)
        if state_values is not None:
            states.update(
                {
                    name: value
                    for name, value in state_values.items()
                    if name in states
                }
            )

        rules = _literal_rules(binding.fixed_values)
        source = self.normalised
        folded_equations = [
            eq.xreplace(rules) for eq in source.equations
        ]
        folded = NormalisedSystem(
            folded_equations,
            source.registry.copy(),
            dict(source.funcs),
            set(source.unknown_names),
            list(source.aux_names),
            list(source.new_params),
            list(source.inferred_states),
            dict(source.rename),
            derivative_names=source.derivative_names,
        )

        swept = {name: self.parameters[name] for name in binding.swept}
        (
            index_map,
            all_symbols,
            funcs,
            parsed_equations,
            fn_hash,
        ) = assemble_simplified(
            folded,
            states,
            list(self.observables),
            swept,
            list(self.driver_names),
            self.driver_dict,
            dict(self.known_symbol_map),
            self.user_functions,
            self.user_function_derivatives,
            state_priority=self.state_priority,
            irreducible=self.irreducible,
            state_units=self.state_units,
            parameter_units=self.parameter_units,
            observable_units=self.observable_units,
            driver_units=self.driver_units,
            simplify_options=self.simplify_options,
        )
        # Inlined non-device callables keep their entries.
        funcs = {**(self.user_functions or {}), **(funcs or {})}
        return index_map, all_symbols, funcs, parsed_equations, fn_hash
