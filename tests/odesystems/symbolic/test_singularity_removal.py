"""Tests for removable-singularity handling in CellML loading.

Covers the opt-in ``fix_singularities`` path of
:func:`~cubie.odesystems.symbolic.parsing.cellml.load_cellml_model`:
membrane-voltage detection, the Piecewise bridge inserted into the
generated equations, the warned no-op when no voltage is found, and
the cache-key dependence on the option. Shared CellML fixtures live in
the root ``tests/conftest.py``.
"""

import logging

import numpy as np
import pytest

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.expr import _children
from cubie.odesystems.symbolic.parsing.cellml import (
    _find_membrane_voltage,
    load_cellml_model,
)
from cubie.odesystems.symbolic.parsing.cellml_cache import CellMLCache

CELLML_LOGGER = "cubie.odesystems.symbolic.parsing.cellml"


def _has_piecewise(node):
    """Return whether an IR expression contains a Piecewise node."""
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ir.Piecewise):
            return True
        stack.extend(_children(current))
    return False


def _codegen_piecewise_count(ode):
    """Count generated equation RHSs that contain a Piecewise node."""
    equations = ode.equations
    groups = (
        list(equations.state_derivatives)
        + list(equations.observables)
        + list(equations.auxiliaries)
    )
    return sum(_has_piecewise(rhs) for _, rhs in groups)


# --- membrane-voltage detection (read-only helper) -----------------------


def test_find_membrane_voltage_identifies_real_state(beeler_reuter_raw):
    """The membrane voltage of a real cardiac model is detected."""
    assert str(_find_membrane_voltage(beeler_reuter_raw)) == "membrane$V"


def test_find_membrane_voltage_returns_none_without_match(basic_ode_raw):
    """A model with no membrane-voltage state returns None."""
    assert _find_membrane_voltage(basic_ode_raw) is None


# --- the fix inserts a Piecewise into codegen (known singularity) ---------


def test_fix_default_inserts_piecewise(ghk_singularity_model):
    """The default fix bridges the single GHK singularity."""
    assert _codegen_piecewise_count(ghk_singularity_model) == 1


@pytest.mark.parametrize(
    # Unique set: the fix-singularities toggle is the value under
    # test; every other configuration wants it on.
    "solver_settings_override",
    [{"fix_singularities": False}],
    indirect=True,
    ids=[""],
)
def test_fix_disabled_leaves_singularity(ghk_singularity_model):
    """With the fix disabled, the singular division is left intact."""
    assert _codegen_piecewise_count(ghk_singularity_model) == 0


# --- auto-detect, warning, and error behaviour (direct loads) ------------


def test_autodetect_logs_info_without_warning(
    cellml_fixtures_dir, caplog, recwarn, isolated_cache_root
):
    """Auto-detect names the voltage via INFO, applies the fix, no warn.

    Use a fresh cache root so the CellML parse cache misses and the
    parse-time INFO log is actually emitted.
    """
    path = str(cellml_fixtures_dir / "ghk_singularity.cellml")
    with caplog.at_level(logging.INFO, logger=CELLML_LOGGER):
        ode = load_cellml_model(
            path, name="ghk_autodetect", fix_singularities=True
        )
    assert _codegen_piecewise_count(ode) == 1
    assert any("membrane$V" in record.message for record in caplog.records)
    assert not any(
        issubclass(w.category, UserWarning) for w in recwarn.list
    )


def test_autodetect_missing_voltage_warns_and_skips(
    cellml_fixtures_dir, isolated_cache_root
):
    """No detectable voltage warns and loads the model unchanged.

    Use a fresh cache root so the parse-time UserWarning is not
    skipped by a CellML parse-cache hit.
    """
    path = str(cellml_fixtures_dir / "basic_ode.cellml")
    with pytest.warns(UserWarning, match="membrane voltage"):
        ode = load_cellml_model(
            path, name="basic_no_voltage", fix_singularities=True
        )
    assert _codegen_piecewise_count(ode) == 0


def test_explicit_voltage_not_found_raises(cellml_fixtures_dir):
    """An unknown explicit voltage name raises ValueError."""
    path = str(cellml_fixtures_dir / "ghk_singularity.cellml")
    with pytest.raises(ValueError):
        load_cellml_model(
            path,
            name="ghk_bad_voltage",
            fix_singularities=True,
            voltage_variable="does$not_exist",
        )


def test_fix_singularities_changes_cache_key(cellml_fixtures_dir):
    """Toggling fix_singularities yields a distinct cache key."""
    path = str(cellml_fixtures_dir / "ghk_singularity.cellml")
    cache = CellMLCache("ghk_singularity", path)
    key_off = cache.compute_cache_key(
        None, None, np.float32, "ghk_singularity",
        fix_singularities=False,
    )
    key_on = cache.compute_cache_key(
        None, None, np.float32, "ghk_singularity",
        fix_singularities=True, voltage_variable="membrane$V",
    )
    assert key_off != key_on
