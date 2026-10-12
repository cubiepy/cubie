"""Parsed systems with every parameter left as a symbol."""

from typing import Any, Callable, Dict, Iterable, List, Optional

import attrs

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.parsing.assemble import assemble_simplified
from cubie.odesystems.symbolic.parsing.normalise import NormalisedSystem
from cubie.odesystems.symbolic.sym_utils import (
    RESERVED_CODEGEN_PREFIX,
    hash_system_definition,
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

    def specialise(
        self,
        swept: Iterable[str] = (),
        values: Optional[Dict[str, float]] = None,
        state_values: Optional[Dict[str, float]] = None,
    ):
        """Assemble the system with ``swept`` as the array's rows.

        Parameters
        ----------
        swept
            Names read from the parameters array, in order. Every
            other parameter is substituted as a number.
        values
            Values for the parameters. Defaults to the parsed values.
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

        swept = tuple(swept)
        values = {**self.parameters, **(values or {})}
        unknown = (set(swept) | set(values)) - set(self.parameters)
        if unknown:
            raise KeyError(
                f"{sorted(unknown)} are not parameters of this system."
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

        rules = _literal_rules(
            {
                name: value
                for name, value in values.items()
                if name not in swept
            }
        )
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

        swept_values = {name: values[name] for name in swept}
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
            swept_values,
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
