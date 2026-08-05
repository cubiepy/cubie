"""Tests for removable-singularity handling in BigModel loading.

Covers the opt-in ``fix_singularities`` path of
:func:`~cubie.odesystems.symbolic.parsing.bigmodel.load_bigmodel_file`:
boundary-potential detection, the Piecewise bridge inserted into the
generated equations, the warned no-op when no voltage is found, and
the cache-key dependence on the option. Shared BigModel fixtures live in
the root ``tests/conftest.py``.
"""

import logging

import numpy as np
import pytest

from cubie.odesystems.symbolic.engine import expr as ir
from cubie.odesystems.symbolic.engine.expr import _children
from cubie.odesystems.symbolic.parsing.bigmodel import (
    _find_outerboundary_potential,
    load_bigmodel_file,
)
from cubie.odesystems.symbolic.parsing.bigmodel_cache import BigModelCache
from tests._utils import TWO_DRIVER_SYSTEM

BIGMODEL_LOGGER = "cubie.odesystems.symbolic.parsing.bigmodel"


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


# --- boundary-potential detection (read-only helper) -----------------------


def test_find_boundary_potential_identifies_real_state(BR_raw):
    """The boundary potential of a real model is detected."""
    assert str(_find_outerboundary_potential(BR_raw)) == "outerboundary$V"


def test_find_outerboundary_potential_returns_none_without_match(basic_ode_raw):
    """A model with no boundary-potential state returns None."""
    assert _find_outerboundary_potential(basic_ode_raw) is None


# --- the fix inserts a Piecewise into codegen (known singularity) ---------


def test_fix_default_inserts_piecewise(removable_singularity_model):
    """The default fix bridges the single removable singularity."""
    assert _codegen_piecewise_count(removable_singularity_model) == 1


@pytest.mark.parametrize(
    # Shares the two-driver chain, which disables the fix.
    "solver_settings_override",
    [TWO_DRIVER_SYSTEM],
    indirect=True,
    ids=[""],
)
def test_fix_disabled_leaves_singularity(removable_singularity_model):
    """With the fix disabled, the singular division is left intact."""
    assert _codegen_piecewise_count(removable_singularity_model) == 0


# --- auto-detect, warning, and error behaviour (direct loads) ------------


def test_autodetect_logs_info_without_warning(
    bigmodel_fixtures_dir, caplog, recwarn, isolated_cache_root
):
    """Auto-detect names the voltage via INFO, applies the fix, no warn.

    Use a fresh cache root so the BigModel parse cache misses and the
    parse-time INFO log is actually emitted.
    """
    path = str(bigmodel_fixtures_dir / "removable_singularity.bigmodel")
    with caplog.at_level(logging.INFO, logger=BIGMODEL_LOGGER):
        ode = load_bigmodel_file(
            path, name="removable_autodetect", fix_singularities=True
        )
    assert _codegen_piecewise_count(ode) == 1
    assert any("outerboundary$V" in record.message for record in caplog.records)
    assert not any(
        issubclass(w.category, UserWarning) for w in recwarn.list
    )


def test_autodetect_missing_voltage_warns_and_skips(
    bigmodel_fixtures_dir, isolated_cache_root
):
    """No detectable voltage warns and loads the model unchanged.

    Use a fresh cache root so the parse-time UserWarning is not
    skipped by a BigModel parse-cache hit.
    """
    path = str(bigmodel_fixtures_dir / "basic_ode.bigmodel")
    with pytest.warns(UserWarning, match="boundary potential"):
        ode = load_bigmodel_file(
            path, name="basic_no_voltage", fix_singularities=True
        )
    assert _codegen_piecewise_count(ode) == 0


def test_explicit_voltage_not_found_raises(bigmodel_fixtures_dir):
    """An unknown explicit voltage name raises ValueError."""
    path = str(bigmodel_fixtures_dir / "removable_singularity.bigmodel")
    with pytest.raises(ValueError):
        load_bigmodel_file(
            path,
            name="removable_bad_voltage",
            fix_singularities=True,
            voltage_variable="does$not_exist",
        )


def test_fix_singularities_changes_cache_key(bigmodel_fixtures_dir):
    """Toggling fix_singularities yields a distinct cache key."""
    path = str(bigmodel_fixtures_dir / "removable_singularity.bigmodel")
    cache = BigModelCache("removable_singularity", path)
    key_off = cache.compute_cache_key(
        None, None, np.float32, "removable_singularity",
        fix_singularities=False,
    )
    key_on = cache.compute_cache_key(
        None, None, np.float32, "removable_singularity",
        fix_singularities=True, voltage_variable="outerboundary$V",
    )
    assert key_off != key_on
