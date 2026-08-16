"""Constants-symbolic system checkpoints and constant specialisation.

The parser saves the system definition with constants still symbolic
and specialises it for one set of constant values by substituting
literal numbers into the engine IR at the head of the codegen
pipeline. Interned-IR constructor folding prunes algebra that the
literals collapse (zero terms, constant-condition Piecewise
branches), and the structural path re-runs classification,
structural simplification, and tearing on the folded equations, so
system structure always follows the current constant values.

Published Classes
-----------------
:class:`NormalisedSystemDefinition`
    Checkpoint of a normalised (string/SymPy/CellML) system. Each
    specialisation re-classifies the folded equations and re-runs
    the matching assembly backend, so a constant change can move a
    row between differential and algebraic form.

:class:`AssembledSystemDefinition`
    Checkpoint of an already assembled explicit system (callable
    input, or direct :class:`SymbolicODE` construction). Each
    specialisation folds the saved equation pairs and re-hashes;
    the array layout is fixed.

Published Functions
-------------------
:func:`fold_constant_values`
    Substitute constant values as IR literals into equation pairs.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn

import attrs

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.sym_utils import hash_system_definition


def _literal_rules(
    constant_values: Dict[str, float],
) -> Dict[ir.Expr, ir.Expr]:
    """Return the IR substitution map for ``constant_values``."""

    return {
        ir.sym(str(name)): ir.num(float(value))
        for name, value in constant_values.items()
    }


def fold_constant_values(
    pairs: Iterable[Tuple[ir.Expr, ir.Expr]],
    constant_values: Dict[str, float],
) -> List[Tuple[ir.Expr, ir.Expr]]:
    """Substitute constant values as literals into IR pairs.

    Parameters
    ----------
    pairs
        ``(lhs, rhs)`` engine IR pairs.
    constant_values
        Mapping of constant names to their current values.

    Returns
    -------
    list[tuple[Expr, Expr]]
        Folded pairs. Constructor folding runs during substitution,
        so dead algebra and constant-condition branches are already
        pruned in the result.
    """

    rules = _literal_rules(constant_values)
    if not rules:
        return list(pairs)
    memo: Dict = {}
    return [
        (
            ir.xreplace(lhs, rules, memo),
            ir.xreplace(rhs, rules, memo),
        )
        for lhs, rhs in pairs
    ]


@attrs.define
class NormalisedSystemDefinition:
    """Constants-symbolic checkpoint of a normalised system.

    Holds everything :func:`~.parser.parse_input` derived before
    constants were substituted, so a constant change can re-run
    classification, structural simplification, and assembly from
    the original equations.

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
        Unit annotations forwarded to the assemblers.
    force_simplify
        ``True`` when the user requested structural simplification
        regardless of classification.
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
    force_simplify: bool = False

    def specialise(
        self,
        constant_values: Optional[Dict[str, float]] = None,
        state_values: Optional[Dict[str, float]] = None,
        warn_on_structural: bool = True,
    ):
        """Assemble the system for one set of constant values.

        Parameters
        ----------
        constant_values
            Constant values to fold as literals. Defaults to the
            declared defaults. Keys must match the definition's
            constant names.
        state_values
            Overrides for declared-state initial values, e.g. the
            current values held by a live system.
        warn_on_structural
            Emit the DAE-detected warning when structural
            simplification is auto-enabled by classification.

        Returns
        -------
        tuple
            ``(index_map, all_symbols, funcs, parsed_equations,
            fn_hash, simplified)`` — the parser product tuple.
        """

        from cubie.odesystems.symbolic.parsing.assemble import (
            assemble_explicit,
            assemble_simplified,
        )
        from cubie.odesystems.symbolic.parsing.normalise import (
            NormalisedSystem,
            classify_system,
        )
        from cubie.odesystems.symbolic.parsing.parser import (
            EquationWarning,
        )

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

        shape = classify_system(folded, states.keys(), self.observables)
        use_structural = self.force_simplify or shape == "dae"
        if use_structural and not self.force_simplify and (
            warn_on_structural
        ):
            warn(
                "DAE constructs detected (implicit equations, "
                "higher-order or in-expression derivatives, or "
                "unknowns without derivative equations); structural "
                "simplification enabled.",
                EquationWarning,
            )

        if not use_structural:
            return assemble_explicit(
                folded,
                states,
                list(self.observables),
                dict(self.parameters),
                values,
                list(self.driver_names),
                self.driver_dict,
                self.user_functions,
                self.user_function_derivatives,
                state_units=self.state_units,
                parameter_units=self.parameter_units,
                constant_units=self.constant_units,
                observable_units=self.observable_units,
                driver_units=self.driver_units,
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


@attrs.define
class AssembledSystemDefinition:
    """Constants-symbolic checkpoint of an assembled system.

    Used for callable input and for systems constructed directly
    from equations: the array layout is fixed, and specialisation
    only folds constant values into the saved equation pairs and
    re-hashes the source identity.

    Parameters
    ----------
    equations
        ``(lhs, rhs)`` engine IR pairs with constants symbolic.
    derivative_names
        User-function name to derivative-placeholder print name.
    function_aliases
        Accepted IR call names and their generated-source names.
    """

    equations: Tuple[Tuple[ir.Expr, ir.Expr], ...]
    derivative_names: Dict[str, str] = attrs.field(factory=dict)
    function_aliases: Dict[str, str] = attrs.field(factory=dict)

    def specialise(
        self,
        constant_values: Dict[str, float],
        index_map,
    ):
        """Fold constant values and repackage the equations.

        Parameters
        ----------
        constant_values
            Constant values to fold as literals.
        index_map
            The live :class:`~..indexedbasemaps.IndexedBases` of the
            system; category membership is owned by the caller and
            passes through unchanged.

        Returns
        -------
        tuple
            ``(parsed_equations, fn_hash)`` for the folded system.
        """

        from cubie.odesystems.symbolic.parsing.parser import (
            ParsedEquations,
        )

        folded = fold_constant_values(self.equations, constant_values)
        parsed = ParsedEquations.from_equations(
            folded,
            index_map,
            derivative_names=self.derivative_names,
            function_aliases=self.function_aliases,
        )
        fn_hash = hash_system_definition(
            parsed,
            index_map.constants.default_values,
            state_labels=index_map.state_names,
            dxdt_labels=index_map.dxdt_names,
            parameter_labels=index_map.parameter_names,
            driver_labels=index_map.driver_names,
            observable_labels=index_map.observables.ref_map.keys(),
            derivative_names=parsed.derivative_names,
            function_aliases=parsed.function_aliases,
        )
        return parsed, fn_hash
