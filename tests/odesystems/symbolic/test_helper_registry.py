"""Tests for cubie.odesystems.symbolic.helper_registry."""

from cubie.odesystems.symbolic.helper_registry import system_jacobian


def test_jacobian_is_reused_until_the_system_is_specialised_again(
    system_restored,
):
    """Return the same Jacobian until the swept parameters change."""
    system = system_restored
    first = system_jacobian(system)
    assert system_jacobian(system) is first
    system.set_swept_parameters(system.parameters.names[:1])
    assert system_jacobian(system) is not first
