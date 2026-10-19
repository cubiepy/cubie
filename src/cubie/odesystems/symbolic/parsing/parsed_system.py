"""Parsed systems with every parameter left as a symbol."""

from typing import Any, Callable, Dict, Iterable, List, Optional

import attrs

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.parsing.assemble import (
    SpecialisedSystem,
    assemble_simplified,
)
from cubie.odesystems.symbolic.parsing.normalise import NormalisedSystem
from cubie.odesystems.symbolic.sym_utils import (
    RESERVED_CODEGEN_PREFIX,
    hash_system_definition,
)
from cubie.odesystems.variables import (
    Parameter,
    ParameterSet,
    ValuedSet,
    VariableSet,
)


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
    """A parsed system with every parameter left as a symbol.

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
    driver_names
        Driver labels.
    known_symbol_map
        Name-to-SymPy-symbol map for the immutable inputs.
    user_functions, user_function_derivatives
        Equation callables and their analytic derivative helpers.
    state_priority, irreducible, simplify_options
        Options forwarded to
        :func:`~..structural.simplify.structural_simplify`.
    state_units, parameter_units, observable_units, driver_units
        Unit strings keyed by name.
    """

    normalised: Any
    states: Dict[str, float]
    observables: List[str]
    parameters: Dict[str, float]
    driver_names: List[str]
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
        names = (
            *self.states,
            *self.observables,
            *self.parameters,
            *self.driver_names,
        )
        for name in names:
            if str(name).startswith(RESERVED_CODEGEN_PREFIX):
                raise ValueError(
                    f"Name '{name}' is reserved: user symbols cannot "
                    f"start with '{RESERVED_CODEGEN_PREFIX}'."
                )

    @property
    def definition_hash(self) -> str:
        """Return a hash of the equations and variable names.

        We hash the equations with every parameter left as a symbol and
        leave out which parameters are swept and their values, so we
        can name an unnamed system the same way whichever parameters
        it sweeps.
        """
        return hash_system_definition(
            self.normalised.equations,
            observable_labels=self.observables,
            state_labels=sorted(self.states),
            parameter_labels=sorted(self.parameters),
            driver_labels=self.driver_names,
            derivative_names=self.normalised.derivative_names,
            function_aliases=self.normalised.rename,
        )

    @property
    def declared_parameters(self) -> ValuedSet:
        """Every parameter with its default and unit, in sorted order."""
        units = self.parameter_units or {}
        return VariableSet.of(
            Parameter, dict(sorted(self.parameters.items())), units
        )

    def specialise(
        self,
        swept: Iterable[str] = (),
        values: Optional[Dict[str, float]] = None,
    ) -> SpecialisedSystem:
        """Assemble the system with ``swept`` as the array's rows.

        Parameters
        ----------
        swept
            Names read from the parameters array, in any order. We
            order the rows as the system orders its parameters, and
            substitute every other parameter as a number.
        values
            Values for some of the other parameters. The rest take
            their parsed defaults.

        Returns
        -------
        SpecialisedSystem
            The layout, equations, callables, source hash and mass
            matrix.

        Raises
        ------
        KeyError
            If a name is not a parameter.
        ValueError
            If a swept name repeats or also has a value.
        """

        return self.assemble(
            ParameterSet.build(self.declared_parameters, swept, values)
        )

    def assemble(self, parameters: ParameterSet) -> SpecialisedSystem:
        """Return the system assembled with ``parameters`` compiled in.

        Parameters
        ----------
        parameters
            Every parameter of this system, swept or fixed at a value.

        Returns
        -------
        SpecialisedSystem
            The layout, equations, callables and source hash.
        """
        rules = _literal_rules(parameters.fixed)
        source = self.normalised
        folded_equations = [
            (ir.xreplace(lhs, rules), ir.xreplace(rhs, rules))
            for lhs, rhs in source.equations
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
            dict(self.states),
            list(self.observables),
            parameters,
            list(self.driver_names),
            dict(self.known_symbol_map),
            self.user_functions,
            self.user_function_derivatives,
            state_priority=self.state_priority,
            irreducible=self.irreducible,
            state_units=self.state_units,
            observable_units=self.observable_units,
            driver_units=self.driver_units,
            simplify_options=self.simplify_options,
        )
