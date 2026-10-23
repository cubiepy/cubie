
import pytest

from cubie.odesystems.symbolic.parsing import parse_system


@pytest.fixture(scope="module")
def torn_parsed_system():
    return parse_system(
        "dx = -k*x + y\n0 = y + sin(y) - a*x",
        states={"x": 1.0, "y": 0.5},
        parameters={"k": 1.0, "a": 2.0},
    )


def test_specialised_systems_compare_by_value(torn_parsed_system):
    first = torn_parsed_system.specialise(("k",))
    second = torn_parsed_system.specialise(("k",))
    assert first.mass is not None
    assert first == second
    assert hash(first) == hash(second)
    assert first != torn_parsed_system.specialise(("a",))
