"""End-to-end guards against user names aliasing generated bindings.

Every user constant is emitted into generated factories through a
single prefixed local (``CONSTANT_ALIAS_PREFIX``), so a model constant
named after any factory-scope binding (solver scalings, loop bounds,
tableau metadata, generated body locals, even ``precision`` itself)
can neither replace that binding nor be replaced by it. These tests
solve the ``hostile_names`` system, whose constants are named after
each class of internal binding, and compare against the identically
parameterised ``safe_names_system``.
"""

import numpy as np
import pytest


from cubie import create_ODE_system, solve_ivp
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_dxdt_fac_code,
)
from cubie.odesystems.symbolic.sym_utils import (
    CONSTANT_ALIAS_PREFIX,
    RESERVED_CODEGEN_PREFIX,
)
from tests.system_fixtures import HOSTILE_NAME_CONSTANTS


def _solve(system, method):
    result = solve_ivp(
        system,
        y0={"x": 2.0},
        method=method,
        duration=0.2,
        dt=0.01,
        save_every=0.05,
    )
    assert not np.any(result.status_codes)
    return result.time_domain_array


# ``euler`` covers the explicit dxdt factory; ``backwards_euler``
# compiles the single-stage residual/operator/Neumann templates whose
# factory scope binds beta, gamma, order, n, beta_inv, and
# h_eff_factor; ``firk`` compiles the flattened n_stage_* templates
# that additionally bind total_n, stage_width, and the tableau
# metadata symbols c_0/a_0_0 — every one shadowed by a same-named
# model constant.
@pytest.mark.parametrize(
    "method", ["euler", "backwards_euler", "firk"]
)
def test_hostile_names_match_safe_reference(
    hostile_names_system, safe_names_system, method
):
    """Solves are unaffected by hostile constant names."""
    np.testing.assert_allclose(
        _solve(hostile_names_system, method),
        _solve(safe_names_system, method),
        rtol=1e-6,
    )


def test_hostile_constants_emit_only_prefixed_loads(
    hostile_names_system,
):
    """Generated source loads every hostile constant prefixed."""
    code = generate_dxdt_fac_code(
        hostile_names_system.equations, hostile_names_system.indices
    )
    for name in HOSTILE_NAME_CONSTANTS:
        load = (
            f"{CONSTANT_ALIAS_PREFIX}{name} = "
            f"precision(constants['{name}'])"
        )
        assert load in code
        # No unprefixed binding of the user name anywhere in the
        # factory body.
        assert f"\n    {name} = " not in code


def test_reserved_prefix_names_are_rejected(precision):
    """User symbols may not enter the generated-code namespace."""
    name = f"{RESERVED_CODEGEN_PREFIX}k"
    with pytest.raises(ValueError, match="reserved"):
        create_ODE_system(
            f"dx = -{name}*x",
            states={"x": 2.0},
            constants={name: 1.0},
            precision=precision,
            name="reserved_prefix_rejected",
        )
