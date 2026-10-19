"""Tests for cubie.odesystems.layout."""

from cubie.odesystems.symbolic.engine.expr import arr, sym
from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.variables import Parameter, ParameterSet, VariableSet
from cubie.odesystems.symbolic.engine.adapter import buffer_maps


def test_dxdt_names_follow_the_states():
    """We place each state's derivative at the state's position."""
    layout = SystemLayout(states=("v", "x"))
    assert layout.dxdt_names == ("dv", "dx")
    assert buffer_maps(layout).dxdt_order == {sym("dv"): 0, sym("dx"): 1}


def test_arrayrefs_place_each_name_in_its_buffer():
    """We read every name from its buffer at its position."""
    layout = SystemLayout(
        states=("x",),
        parameters=("k", "c"),
        observables=("o",),
        drivers=("u",),
    )
    refs = buffer_maps(layout).arrayrefs
    assert refs["x"] is arr("state", 0)
    assert refs["k"] is arr("parameters", 0)
    assert refs["c"] is arr("parameters", 1)
    assert refs["o"] is arr("observables", 0)
    assert refs["u"] is arr("drivers", 0)
    assert refs["dx"] is arr("out", 0)


def test_orders_map_symbols_to_positions():
    """The order maps hold each symbol's buffer position."""
    layout = SystemLayout(states=("a", "b"), drivers=("u", "w"))
    assert buffer_maps(layout).state_order == {sym("a"): 0, sym("b"): 1}
    assert buffer_maps(layout).driver_order == {sym("u"): 0, sym("w"): 1}
    assert buffer_maps(layout).observable_symbols == ()


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
    """The drivers buffer holds the drivers and their derivatives."""
    layout = SystemLayout(
        drivers=("u", "w"), driver_derivatives=((0, 1), (1, 2))
    )
    assert layout.num_drivers == 4
    assert layout.driver_names == ("u", "w")


def test_buffer_maps_are_built_once_per_layout():
    """Every read of a layout's maps returns the same maps."""
    layout = SystemLayout(states=("x",), parameters=("k",))
    assert buffer_maps(layout) is buffer_maps(layout)


def test_layout_records_the_compiled_parameters():
    """The rows of the parameters array are the swept parameters."""
    declared = VariableSet.of(Parameter, {"a": 1.0, "b": 2.0})
    layout = SystemLayout(
        states=("x",), parameters=ParameterSet.build(declared, ["b"])
    )
    assert layout.swept_names == ("b",)
    assert layout.parameters.fixed == {"a": 1.0}
