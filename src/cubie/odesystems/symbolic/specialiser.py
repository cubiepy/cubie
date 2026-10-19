"""Assemble a parsed system for the parameters a batch compiles.

Published Classes
-----------------
:class:`SpecialiserConfig`
    Compile settings of a :class:`SystemSpecialiser`: the parameters as
    the batch compiles them.

:class:`SystemSpecialiser`
    Child factory of a
    :class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE` whose
    build assembles the parsed system for its compiled parameters.

    >>> from cubie.odesystems.symbolic.parsing import parse_system
    >>> from cubie.odesystems.variables import ParameterSet
    >>> parsed = parse_system(
    ...     dxdt=["dx = -k * x + c"], states={"x": 1.0},
    ...     parameters={"k": 0.5, "c": 1.0},
    ... )
    >>> specialiser = SystemSpecialiser(
    ...     parsed, ParameterSet.build(parsed.declared_parameters, ["k"])
    ... )
    >>> specialiser.layout.swept_names
    ('k',)
    >>> specialiser.fixed_parameter_values
    {'c': 1.0}

See Also
--------
:class:`~cubie.odesystems.symbolic.parsing.parsed_system.ParsedSystem`
    The system with every parameter left as a symbol.
:class:`~cubie.odesystems.symbolic.parsing.assemble.SpecialisedSystem`
    The cache this factory builds.
"""

from typing import Any, Callable, Dict, Tuple

from attrs import field, frozen
from attrs.validators import instance_of as attrsval_instance_of

from cubie.CUDAFactory import CUDAFactory, _CubieConfigBase
from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.symbolic.parsing.assemble import SpecialisedSystem
from cubie.odesystems.symbolic.parsing.parsed_system import ParsedSystem
from cubie.odesystems.variables import ParameterSet


@frozen
class SpecialiserConfig(_CubieConfigBase):
    """Compile settings of a :class:`SystemSpecialiser`.

    Parameters
    ----------
    compiled_parameters
        Every parameter, swept or fixed at the value to compile in.
    """

    compiled_parameters: ParameterSet = field(
        validator=attrsval_instance_of(ParameterSet)
    )


class SystemSpecialiser(CUDAFactory):
    """Assemble a parsed system for the parameters a batch compiles.

    Parameters
    ----------
    parsed_system
        The system with every parameter left as a symbol.
    compiled_parameters
        Every parameter, swept or fixed at the value to compile in.
    """

    def __init__(
        self,
        parsed_system: ParsedSystem,
        compiled_parameters: ParameterSet,
    ) -> None:
        super().__init__()
        self._parsed_system = parsed_system
        self.setup_compile_settings(
            SpecialiserConfig(compiled_parameters=compiled_parameters)
        )

    def build(self) -> SpecialisedSystem:
        """Fold the fixed values in and structurally simplify the system.

        Returns
        -------
        SpecialisedSystem
            The layout, equations, callables and source hash.
        """
        return self._parsed_system.assemble(
            self.compile_settings.compiled_parameters
        )

    @property
    def compiled_parameters(self) -> ParameterSet:
        """Every parameter as the batch compiles it."""
        return self.compile_settings.compiled_parameters

    @property
    def swept_parameters(self) -> Tuple[str, ...]:
        """Names of the parameters array's rows, in order."""
        return self.compiled_parameters.swept

    @property
    def fixed_parameter_values(self) -> Dict[str, float]:
        """The value compiled in for every parameter that is not swept."""
        return self.compiled_parameters.fixed

    @property
    def layout(self) -> SystemLayout:
        """The variables in each device buffer, in buffer order."""
        return self.get_cached_output("layout")

    @property
    def equations(self) -> Any:
        """The equations with every fixed value folded in."""
        return self.get_cached_output("equations")

    @property
    def functions(self) -> Dict[str, Callable]:
        """Every callable the generated code may call, by name."""
        return self.get_cached_output("functions")

    @property
    def fn_hash(self) -> str:
        """Hash of every value the generated source embeds."""
        return self.get_cached_output("fn_hash")

    @property
    def mass(self) -> Any:
        """The mass matrix, or ``None`` for an ODE."""
        return self.get_cached_output("mass")
