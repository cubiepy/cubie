"""Guards against user names entering the generated-code namespace."""

import pytest

from cubie import create_ODE_system
from cubie.odesystems.symbolic.sym_utils import (
    RESERVED_CODEGEN_PREFIX,
)


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
