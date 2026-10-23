"""Tests for cubie.odesystems.layout."""

from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.variables import (
    Parameter,
    ParameterSet,
    Variable,
    VariableSet,
)


def test_dxdt_names_follow_the_states():
    """We place each state's derivative at the state's position."""
    layout = SystemLayout(states=VariableSet.of(Variable, ["v", "x"]))
    assert layout.dxdt_names == ("dv", "dx")


def test_layouts_with_the_same_names_compare_equal():
    """Layouts built from the same names compare equal and hash alike."""
    declared = VariableSet.of(Parameter, {"k": 1.0})
    first = SystemLayout(
        states=VariableSet.of(Variable, ["x"]),
        parameters=ParameterSet.build(declared, ["k"]),
    )
    second = SystemLayout(
        states=VariableSet.of(Variable, ("x",)),
        parameters=ParameterSet.build(declared, ("k",)),
    )
    assert first == second
    assert hash(first) == hash(second)


def test_layout_records_the_compiled_parameters():
    """The rows of the parameters array are the swept parameters."""
    declared = VariableSet.of(Parameter, {"a": 1.0, "b": 2.0})
    layout = SystemLayout(
        states=VariableSet.of(Variable, ["x"]),
        parameters=ParameterSet.build(declared, ["b"]),
    )
    assert layout.swept_names == ("b",)
    assert layout.parameters.fixed == {"a": 1.0}


def test_num_drivers_counts_the_derivatives():
    """The drivers buffer holds the drivers and their derivatives."""
    layout = SystemLayout(
        drivers=VariableSet.of(Variable, ["u", "w"]),
        driver_derivatives=((0, 1), (1, 2)),
    )
    assert layout.num_drivers == 4
    assert layout.driver_names == ("u", "w")
