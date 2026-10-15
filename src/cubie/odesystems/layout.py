"""The buffer layout a system compiles against.

Published Classes
-----------------
:class:`SystemLayout`
    The variables held in each device buffer, in buffer order, and the
    parameters as the batch compiles them.

    >>> layout = SystemLayout(states=("x", "y"), parameters=("k",))
    >>> layout.dxdt_names
    ('dx', 'dy')
    >>> layout.swept_names
    ('k',)

See Also
--------
:class:`~cubie.odesystems.ODEData.ODEData`
    Compile settings that hold a system's layout.
:func:`~cubie.odesystems.symbolic.engine.adapter.buffer_maps`
    Builds, once for each layout, the positions and array references
    we read in generated code.
"""

from typing import Iterable, Tuple, Union

from attrs import field, frozen
from attrs.validators import (
    deep_iterable as attrsval_deep_iterable,
    instance_of as attrsval_instance_of,
)

from cubie.odesystems.variables import (
    CompiledParameter,
    ParameterSet,
    Variable,
    VariableSet,
)


def _variables(values: Union[VariableSet, Iterable[str]]) -> VariableSet:
    """Return ``values`` as a set of valueless variables.

    We keep only names and units, because the layout fixes where each
    variable sits and never what value it holds.
    """
    if isinstance(values, VariableSet):
        return values.without_values()
    return VariableSet.of(Variable, list(values))


def _parameters(
    values: Union[ParameterSet, Iterable[str]],
) -> ParameterSet:
    """Return ``values`` as compiled parameters, sweeping bare names."""
    if isinstance(values, ParameterSet):
        return values
    return ParameterSet(
        [CompiledParameter(str(name), swept=True) for name in values]
    )


@frozen(hash=False)
class SystemLayout:
    """The variables in each device buffer, in buffer order.

    We build a layout each time we specialise a system, because
    structural simplification decides which states survive. We generate
    code that reads each variable from its buffer at its position here,
    and we fill the parameters array with one row per swept parameter,
    in the order of :attr:`parameters`.

    Parameters
    ----------
    states
        The states, in the order of the state buffer.
    parameters
        Every parameter as the batch compiles it, in the system's
        parameter order. We take bare names as swept parameters.
    observables
        The observables, in the order of the observables buffer.
    drivers
        The drivers, in the order of the drivers buffer.
    driver_derivatives
        The driver index and order of each driver derivative the
        equations read. We place them after the drivers in the drivers
        buffer, in this order.
    """

    states: VariableSet = field(default=(), converter=_variables)
    parameters: ParameterSet = field(default=(), converter=_parameters)
    observables: VariableSet = field(default=(), converter=_variables)
    drivers: VariableSet = field(default=(), converter=_variables)
    driver_derivatives: Tuple[Tuple[int, int], ...] = field(
        default=(),
        converter=tuple,
        validator=attrsval_deep_iterable(
            attrsval_deep_iterable(
                attrsval_instance_of(int), attrsval_instance_of(tuple)
            ),
            attrsval_instance_of(tuple),
        ),
    )
    _key: tuple = field(init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        # We hash a layout on every solve, so we build its key once.
        object.__setattr__(
            self,
            "_key",
            (
                "SystemLayout",
                self.states.names,
                self.parameters._cubie_canonical_(),
                self.observables.names,
                self.drivers.names,
                self.driver_derivatives,
            ),
        )

    def _cubie_canonical_(self) -> tuple:
        """Return the buffer names, compiled parameters and slots."""
        return self._key

    def __hash__(self) -> int:
        return hash(self._key)

    @property
    def state_names(self) -> Tuple[str, ...]:
        """State names, in the order of the state buffer."""
        return self.states.names

    @property
    def dxdt_names(self) -> Tuple[str, ...]:
        """Names of the state derivatives, in the order of the output."""
        return tuple(f"d{name}" for name in self.states.names)

    @property
    def swept_names(self) -> Tuple[str, ...]:
        """Swept parameter names, in the order of the parameters array."""
        return self.parameters.swept

    @property
    def observable_names(self) -> Tuple[str, ...]:
        """Observable names, in the order of the observables buffer."""
        return self.observables.names

    @property
    def driver_names(self) -> Tuple[str, ...]:
        """Driver names, in the order of the drivers buffer."""
        return self.drivers.names

    @property
    def num_drivers(self) -> int:
        """Length of the drivers buffer, which holds both drivers and
        their derivatives."""
        return self.drivers.n + len(self.driver_derivatives)
