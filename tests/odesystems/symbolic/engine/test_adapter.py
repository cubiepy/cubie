"""Tests for cubie.odesystems.symbolic.engine.adapter."""

from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.variables import (
    Parameter,
    ParameterSet,
    Variable,
    VariableSet,
)
from cubie.odesystems.symbolic.engine.adapter import buffer_maps
from cubie.odesystems.symbolic.engine.expr import arr, sym


def test_dxdt_order_matches_the_states():
    """``buffer_maps`` places each derivative at its state's position
    in the output.
    """
    layout = SystemLayout(states=VariableSet.of(Variable, ("v", "x")))
    assert buffer_maps(layout).dxdt_order == {sym("dv"): 0, sym("dx"): 1}


def test_arrayrefs_place_each_name_in_its_buffer():
    """``buffer_maps`` maps each name to its element in its own buffer."""
    layout = SystemLayout(
        states=VariableSet.of(Variable, ("x",)),
        parameters=ParameterSet.build(
            VariableSet.of(Parameter, ("k", "c")), ("k", "c")
        ),
        observables=VariableSet.of(Variable, ("o",)),
        drivers=VariableSet.of(Variable, ("u",)),
    )
    refs = buffer_maps(layout).arrayrefs
    assert refs["x"] is arr("state", 0)
    assert refs["k"] is arr("parameters", 0)
    assert refs["c"] is arr("parameters", 1)
    assert refs["o"] is arr("observables", 0)
    assert refs["u"] is arr("drivers", 0)
    assert refs["dx"] is arr("out", 0)


def test_orders_map_symbols_to_positions():
    """``buffer_maps`` maps each symbol to its buffer position."""
    maps = buffer_maps(
        SystemLayout(
            states=VariableSet.of(Variable, ("a", "b")),
            drivers=VariableSet.of(Variable, ("u", "w")),
        )
    )
    assert maps.state_order == {sym("a"): 0, sym("b"): 1}
    assert maps.driver_order == {sym("u"): 0, sym("w"): 1}
    assert maps.observable_symbols == ()
