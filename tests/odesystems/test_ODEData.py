"""Tests for cubie.odesystems.ODEData."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.odesystems.ODEData import (
    ODEData,
    OPERATION_ORDERINGS,
    SystemSizes,
)
from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.variables import (
    Parameter,
    ParameterSet,
    Variable,
    VariableSet,
)


DECLARED = VariableSet.of(Parameter, {"a": 0.5, "b": 0.3, "g": 9.81})


# ── SystemSizes ───────────────────────────────────────────────── #

def test_system_sizes_construction():
    """All fields stored correctly on frozen attrs class."""
    sizes = SystemSizes(
        states=3,
        observables=2,
        swept_parameters=4,
        drivers=1,
        driver_derivatives=0,
    )
    assert sizes.states == 3
    assert sizes.observables == 2
    assert sizes.swept_parameters == 4
    assert sizes.drivers == 1
    assert sizes.driver_derivatives == 0


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("states", 1.5),
        ("observables", "x"),
        ("swept_parameters", None),
        ("drivers", 2.0),
        ("driver_derivatives", 1.5),
    ],
    ids=[
        "states",
        "observables",
        "swept_parameters",
        "drivers",
        "driver_derivatives",
    ],
)
def test_system_sizes_validates_int(field, bad_value):
    """Each field rejects non-int values."""
    kwargs = dict(
        states=1,
        observables=1,
        swept_parameters=1,
        drivers=1,
        driver_derivatives=0,
    )
    kwargs[field] = bad_value
    with pytest.raises(TypeError):
        SystemSizes(**kwargs)


def test_system_sizes_of_layout_counts_every_buffer():
    """Count a layout's states, observables, swept rows, drivers and
    driver derivatives."""
    layout = SystemLayout(
        states=VariableSet.of(Variable, ["x", "y"]),
        parameters=ParameterSet.build(DECLARED, ["b", "a"]),
        observables=VariableSet.of(Variable, ["v", "w", "z"]),
        drivers=VariableSet.of(Variable, ["u"]),
        driver_derivatives=((0, 1),),
    )
    sizes = SystemSizes.of_layout(layout)
    assert sizes.states == 2
    assert sizes.observables == 3
    assert sizes.swept_parameters == 2
    assert sizes.drivers == 2
    assert sizes.driver_derivatives == 1


# ── ODEData ───────────────────────────────────────────────────── #

@pytest.fixture
def odedata():
    """Return float32 compile settings with Kahn ordering."""
    return ODEData(precision=np.float32, operation_ordering="kahn")


def test_operation_ordering_public_values_are_exact():
    """The compile setting exposes only the approved policy names."""
    assert OPERATION_ORDERINGS == (
        "kahn",
        "greedy",
        "dfs",
        "liveness_auto",
    )


@pytest.mark.parametrize(
    "operation_ordering",
    ["greedy", "dfs", "liveness_auto"],
)
def test_odedata_operation_ordering_participates_in_identity(
    odedata, operation_ordering
):
    """Every opt-in ordering is validated and compile-critical."""
    alternative = odedata.update(operation_ordering=operation_ordering)[0]
    assert alternative.operation_ordering == operation_ordering
    assert alternative.values_hash != odedata.values_hash


@pytest.mark.parametrize("operation_ordering", ["liveness", "bogus"])
def test_odedata_rejects_invalid_operation_ordering(
    odedata, operation_ordering
):
    """Only the four supported ordering policies are accepted."""
    with pytest.raises(ValueError, match="operation_ordering"):
        odedata.update(operation_ordering=operation_ordering)


def test_odedata_precision_participates_in_identity(odedata):
    """A precision change moves values_hash."""
    assert (
        odedata.update(precision=np.float64)[0].values_hash
        != odedata.values_hash
    )
