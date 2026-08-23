"""Guards against user names aliasing generated bindings.

Solves the ``hostile_names`` system, whose constants are named after
each class of internal binding, against the identically
parameterised ``safe_names_system``.
"""

import numpy as np
import pytest


from cubie import solve_ivp
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_dxdt_fac_code,
)
from cubie.odesystems.symbolic.sym_utils import (
    RESERVED_CODEGEN_PREFIX,
)
from tests.system_fixtures import HOSTILE_NAME_CONSTANTS


from tests.system_fixtures import _create_with_folded

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


# euler: dxdt factory; backwards_euler: single-stage templates;
# firk: n_stage_* templates. Every factory binding is shadowed by
# a same-named model constant.
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


def test_hostile_constants_never_named_in_source(
    hostile_names_system,
):
    """Generated source contains no binding for any constant name."""
    code = generate_dxdt_fac_code(
        hostile_names_system.equations, hostile_names_system.indices
    )
    for name in HOSTILE_NAME_CONSTANTS:
        # Values are folded as literals: no load, no assignment, and
        # no reference to the constant's name anywhere in the factory.
        assert f"constants['{name}']" not in code
        assert f"\n    {name} = " not in code
        assert f"\n        {name} = " not in code


def test_reserved_prefix_names_are_rejected(precision):
    """User symbols may not enter the generated-code namespace."""
    name = f"{RESERVED_CODEGEN_PREFIX}k"
    with pytest.raises(ValueError, match="reserved"):
        _create_with_folded(
            f"dx = -{name}*x",
            states={"x": 2.0},
            constants={name: 1.0},
            precision=precision,
            name="reserved_prefix_rejected",
        )
