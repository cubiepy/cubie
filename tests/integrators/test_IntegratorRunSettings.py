"""Tests for cubie.integrators.IntegratorRunSettings."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings


# ── Construction (items 1-5) ─────────────────────────────── #

def test_construction_with_explicit_values():
    """IntegratorRunSettings stores explicit algorithm and controller."""
    # Inline construction permitted: __init__ test
    settings = IntegratorRunSettings(
        precision=np.float32,
        algorithm="rk4",
        step_controller="pid",
    )
    assert settings.algorithm == "rk4"
    assert settings.step_controller == "pid"
    assert settings.precision == np.float32


# ── Validators (items 6-7) ───────────────────────────────── #

def test_algorithm_rejects_non_string():
    """Validator rejects non-string algorithm."""
    with pytest.raises(TypeError):
        IntegratorRunSettings(
            precision=np.float32, algorithm=42
        )


def test_step_controller_rejects_non_string():
    """Validator rejects non-string step_controller."""
    with pytest.raises(TypeError):
        IntegratorRunSettings(
            precision=np.float32, step_controller=123
        )
