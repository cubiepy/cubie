"""Tests for cubie.odesystems.layout."""

from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.variables import Parameter, ParameterSet, VariableSet


def test_dxdt_names_follow_the_states():
    """Each state's derivative takes the state's position."""
    layout = SystemLayout(states=("v", "x"))
    assert layout.dxdt_names == ("dv", "dx")


def test_layouts_with_the_same_names_compare_equal():
    """Layouts built from the same names compare equal and hash alike."""
    first = SystemLayout(states=["x"], parameters=["k"])
    second = SystemLayout(states=("x",), parameters=("k",))
    assert first == second
    assert hash(first) == hash(second)


def test_bare_parameter_names_are_swept():
    """Bare names build a layout that sweeps every one of them."""
    layout = SystemLayout(states=("x",), parameters=("k", "c"))
    assert layout.swept_names == ("k", "c")
    assert layout.parameters.fixed == {}


def test_num_drivers_counts_the_derivatives():
    """``num_drivers`` counts the drivers and their derivatives."""
    layout = SystemLayout(
        drivers=("u", "w"), driver_derivatives=((0, 1), (1, 2))
    )
    assert layout.num_drivers == 4
    assert layout.driver_names == ("u", "w")


def test_layout_records_the_compiled_parameters():
    """The swept parameters are the rows of the parameters array."""
    declared = VariableSet.of(Parameter, {"a": 1.0, "b": 2.0})
    layout = SystemLayout(
        states=("x",), parameters=ParameterSet.build(declared, ["b"])
    )
    assert layout.swept_names == ("b",)
    assert layout.parameters.fixed == {"a": 1.0}
