
import pytest

from cubie.odesystems.symbolic.parsing import parse_system
from cubie.odesystems.symbolic.sym_utils import RESERVED_CODEGEN_PREFIX


RESERVED = f"{RESERVED_CODEGEN_PREFIX}v"


@pytest.mark.parametrize(
    "declaration",
    [
        {"states": {"x": 1.0, RESERVED: 0.0}},
        {"states": {"x": 1.0}, "observables": [RESERVED]},
        {"states": {"x": 1.0}, "drivers": [RESERVED]},
    ],
    ids=["state", "observable", "driver"],
)
def test_reserved_prefix_names_are_rejected(declaration):
    with pytest.raises(ValueError, match="reserved"):
        parse_system("dx = -x", **declaration)
