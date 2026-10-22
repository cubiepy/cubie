from typing import Any, Callable, Dict, Tuple

from attrs import field, frozen
from attrs.validators import instance_of as attrsval_instance_of

from cubie.CUDAFactory import CUDAFactory, _CubieConfigBase
from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.ODEData import SystemSizes
from cubie.odesystems.symbolic.parsing.assemble import SpecialisedSystem
from cubie.odesystems.symbolic.parsing.parsed_system import ParsedSystem
from cubie.odesystems.variables import ParameterSet


@frozen
class SpecialiserConfig(_CubieConfigBase):
    compiled_parameters: ParameterSet = field(
        validator=attrsval_instance_of(ParameterSet)
    )


class SystemSpecialiser(CUDAFactory):
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
        return self._parsed_system.assemble(
            self.compile_settings.compiled_parameters
        )

    @property
    def compiled_parameters(self) -> ParameterSet:
        return self.compile_settings.compiled_parameters

    @property
    def swept_parameters(self) -> Tuple[str, ...]:
        return self.compiled_parameters.swept

    @property
    def fixed_parameter_values(self) -> Dict[str, float]:
        return self.compiled_parameters.fixed

    @property
    def layout(self) -> SystemLayout:
        return self.get_cached_output("layout")

    @property
    def equations(self) -> Any:
        return self.get_cached_output("equations")

    @property
    def functions(self) -> Dict[str, Callable]:
        return self.get_cached_output("functions")

    @property
    def fn_hash(self) -> str:
        return self.get_cached_output("fn_hash")

    @property
    def sizes(self) -> SystemSizes:
        return self.get_cached_output("sizes")

    @property
    def mass(self) -> Any:
        return self.get_cached_output("mass")
