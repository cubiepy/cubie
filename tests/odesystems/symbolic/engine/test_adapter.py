"""Tests for cubie.odesystems.symbolic.engine.adapter."""

from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.symbolic.engine.adapter import buffer_maps
from cubie.odesystems.symbolic.engine.expr import arr, sym


def test_dxdt_order_matches_the_states():
    """Each derivative takes its state's position in the output."""
    layout = SystemLayout(states=("v", "x"))
    assert buffer_maps(layout).dxdt_order == {sym("dv"): 0, sym("dx"): 1}


def test_arrayrefs_place_each_name_in_its_buffer():
    """Every name reads from its buffer at its position."""
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
    maps = buffer_maps(SystemLayout(states=("a", "b"), drivers=("u", "w")))
    assert maps.state_order == {sym("a"): 0, sym("b"): 1}
    assert maps.driver_order == {sym("u"): 0, sym("w"): 1}
    assert maps.observable_symbols == ()


def test_buffer_maps_are_built_once_per_layout():
    """Equal layouts return the same maps."""
    first = SystemLayout(states=("x",), parameters=("k",))
    second = SystemLayout(states=["x"], parameters=["k"])
    assert buffer_maps(first) is buffer_maps(second)
