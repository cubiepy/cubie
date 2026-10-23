"""The buffer layout a system compiles against.

Published Classes
-----------------
:class:`SystemLayout`
    The variables held in each device buffer, in buffer order, and the
    swept and fixed parameters.

    >>> states = VariableSet.of(Variable, ["x", "y"])
    >>> declared = VariableSet.of(Parameter, {"k": 1.0})
    >>> layout = SystemLayout(
    ...     states=states, parameters=ParameterSet.build(declared, ["k"])
    ... )
    >>> layout.dxdt_names
    ('dx', 'dy')
    >>> layout.swept_names
    ('k',)

See Also
--------
:mod:`cubie.odesystems.variables`
    The variable sets a layout holds.
"""

from typing import Tuple

from attrs import field, frozen
from attrs.validators import (
    deep_iterable as attrsval_deep_iterable,
    instance_of as attrsval_instance_of,
)

from cubie.odesystems.variables import ParameterSet, VariableSet


@frozen(cache_hash=True)
class SystemLayout:
    """The variables in each device buffer, in buffer order.

    Hold the position of each variable in each device buffer of one
    specialised system. Pass a parameters array with one row per swept
    parameter, in the order of :attr:`parameters`.

    Parameters
    ----------
    states
        The states, in the order of the state buffer.
    parameters
        Every parameter, swept or fixed, in the system's parameter
        order.
    observables
        The observables, in the order of the observables buffer.
    drivers
        The drivers, in the order of the drivers buffer.
    driver_derivatives
        The driver index and order of each driver derivative the
        equations read, in drivers-buffer order. The layout places
        them after the drivers.
    """

    states: VariableSet = field(
        default=VariableSet(()), validator=attrsval_instance_of(VariableSet)
    )
    parameters: ParameterSet = field(
        default=ParameterSet(()),
        validator=attrsval_instance_of(ParameterSet),
    )
    observables: VariableSet = field(
        default=VariableSet(()), validator=attrsval_instance_of(VariableSet)
    )
    drivers: VariableSet = field(
        default=VariableSet(()), validator=attrsval_instance_of(VariableSet)
    )
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
        """Length of the drivers buffer: drivers plus derivatives."""
        return self.drivers.n + len(self.driver_derivatives)
