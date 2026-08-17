"""Constants-symbolic checkpoint and the constant-specialisation pass."""

from typing import Any, Callable, Dict, Iterable, List, Optional

import attrs
import sympy as sp

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.parsing.assemble import assemble_simplified
from cubie.odesystems.symbolic.parsing.normalise import (
    NormalisedSystem,
    normalise_input,
)


def _literal_rules(
    constant_values: Dict[str, float],
) -> Dict[ir.Expr, ir.Expr]:
    """Return the IR substitution map for ``constant_values``."""

    return {
        ir.sym(str(name)): ir.num(float(value))
        for name, value in constant_values.items()
    }


@attrs.define
class ParsedSystem:
    """Constants-symbolic checkpoint of a parsed system.

    Parameters
    ----------
    normalised
        The parsed system with constants symbolic
        (:class:`~.normalise.NormalisedSystem`).
    states
        Declared plus inferred states mapped to initial values.
    observables
        Declared observable names.
    parameters
        Parameter names mapped to default values.
    constants
        Constant names mapped to their declared default values.
    driver_names, driver_dict
        Driver labels and the optional driver settings dictionary.
    known_symbol_map
        Name-to-SymPy-symbol map for parameters, constants, and
        drivers.
    user_functions, user_function_derivatives
        Callables referenced by the equations and their analytic
        derivative helpers.
    state_priority, irreducible, simplify_options
        Structural-path options forwarded to
        :func:`~..structural.simplify.structural_simplify`.
    state_units, parameter_units, constant_units, observable_units,
    driver_units
        Unit annotations forwarded to the assembler.
    """

    normalised: Any
    states: Dict[str, float]
    observables: List[str]
    parameters: Dict[str, float]
    constants: Dict[str, float]
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
    constant_units: Any = None
    observable_units: Any = None
    driver_units: Any = None

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
            Callables referenced by the equations and their analytic
            derivative helpers.
        """

        states = {
            str(name): float(value)
            for name, value in index_map.state_values.items()
        }
        parameters = {
            str(name): float(value)
            for name, value in index_map.parameter_values.items()
        }
        constants = {
            str(name): float(value)
            for name, value in index_map.constant_values.items()
        }
        observables = list(index_map.observable_names)
        driver_defaults = {
            str(name): value
            for name, value in index_map.drivers.default_values.items()
        }
        driver_names = list(driver_defaults)
        known_symbol_map = {
            name: sp.Symbol(name, real=True)
            for name in (
                list(parameters) + list(constants) + driver_names
            )
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
            constants=constants,
            driver_names=driver_names,
            driver_dict=driver_defaults or None,
            known_symbol_map=known_symbol_map,
            user_functions=user_functions,
            user_function_derivatives=user_function_derivatives,
            state_units=index_map.states.units or None,
            parameter_units=index_map.parameters.units or None,
            constant_units=index_map.constants.units or None,
            observable_units=index_map.observables.units or None,
            driver_units=index_map.drivers.units or None,
        )

    def specialise(
        self,
        constant_values: Optional[Dict[str, float]] = None,
        state_values: Optional[Dict[str, float]] = None,
    ):
        """Assemble the system for one set of constant values.

        Parameters
        ----------
        constant_values
            Constant values to fold as literals; defaults to the
            declared defaults.
        state_values
            Overrides for declared-state initial values.

        Returns
        -------
        tuple
            ``(index_map, all_symbols, funcs, parsed_equations,
            fn_hash, simplified)``.
        """

        values = dict(self.constants)
        if constant_values is not None:
            unknown = set(constant_values) - set(values)
            if unknown:
                raise KeyError(
                    f"Unknown constants in specialisation: "
                    f"{sorted(unknown)}"
                )
            values.update(constant_values)

        states = dict(self.states)
        if state_values is not None:
            states.update(
                {
                    name: value
                    for name, value in state_values.items()
                    if name in states
                }
            )

        rules = _literal_rules(values)
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

        return assemble_simplified(
            folded,
            states,
            list(self.observables),
            dict(self.parameters),
            values,
            list(self.driver_names),
            self.driver_dict,
            dict(self.known_symbol_map),
            self.user_functions,
            self.user_function_derivatives,
            state_priority=self.state_priority,
            irreducible=self.irreducible,
            state_units=self.state_units,
            parameter_units=self.parameter_units,
            constant_units=self.constant_units,
            observable_units=self.observable_units,
            driver_units=self.driver_units,
            simplify_options=self.simplify_options,
        )
