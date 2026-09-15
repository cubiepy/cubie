"""Tests for cubie.cache_root — the shared disk-cache root."""

from pathlib import Path

from cubie import cache_root
from cubie.cubie_cache import CUBIECacheLocator
from cubie.odesystems.symbolic.odefile import ODEFile
from cubie.odesystems.symbolic.parsing.cellml_cache import CellMLCache


def test_default_root_is_cwd_generated(monkeypatch):
    """Without an override the root is <cwd>/generated at call time."""
    previous = cache_root.get_cache_root_override()
    monkeypatch.delenv("CUBIE_CACHE_DIR", raising=False)
    cache_root.set_cache_root(None)
    try:
        assert cache_root.get_cache_root() == Path.cwd() / "generated"
    finally:
        cache_root.set_cache_root(previous)


def test_env_var_sets_root_and_override_wins(tmp_path, monkeypatch):
    """CUBIE_CACHE_DIR redirects the root; set_cache_root beats it."""
    previous = cache_root.get_cache_root_override()
    env_root = tmp_path / "env_root"
    monkeypatch.setenv("CUBIE_CACHE_DIR", str(env_root))
    cache_root.set_cache_root(None)
    try:
        assert cache_root.get_cache_root() == env_root
        override_root = tmp_path / "override_root"
        cache_root.set_cache_root(override_root)
        assert cache_root.get_cache_root() == override_root
    finally:
        cache_root.set_cache_root(previous)


def test_blank_env_var_is_ignored(monkeypatch):
    """An empty CUBIE_CACHE_DIR falls back to <cwd>/generated."""
    previous = cache_root.get_cache_root_override()
    monkeypatch.setenv("CUBIE_CACHE_DIR", "   ")
    cache_root.set_cache_root(None)
    try:
        assert cache_root.get_cache_root() == Path.cwd() / "generated"
    finally:
        cache_root.set_cache_root(previous)


def test_set_cache_root_overrides_and_clears(tmp_path, monkeypatch):
    """set_cache_root installs an override; None restores the default."""
    previous = cache_root.get_cache_root_override()
    monkeypatch.delenv("CUBIE_CACHE_DIR", raising=False)
    try:
        cache_root.set_cache_root(tmp_path)
        assert cache_root.get_cache_root() == tmp_path
        cache_root.set_cache_root(str(tmp_path / "sub"))
        assert cache_root.get_cache_root() == tmp_path / "sub"
        cache_root.set_cache_root(None)
        assert cache_root.get_cache_root() == Path.cwd() / "generated"
    finally:
        cache_root.set_cache_root(previous)


def test_all_cache_layers_share_the_root(
    isolated_cache_root, cellml_fixtures_dir
):
    """Codegen, CellML parse, and kernel caches resolve one root."""
    root = isolated_cache_root

    ode_file = ODEFile("shared_root_system", fn_hash=1234)
    assert ode_file.file_path == (
        root / "shared_root_system" / "shared_root_system_1234.py"
    )

    cellml_cache = CellMLCache(
        model_name="shared_root_model",
        cellml_path=str(cellml_fixtures_dir / "basic_ode.cellml"),
    )
    assert cellml_cache.cache_dir == root / "shared_root_model"

    locator = CUBIECacheLocator(
        system_name="shared_root_system",
        system_hash="abcdef1234567890",
        compile_settings_hash="fedcba0987654321",
    )
    assert Path(locator.get_cache_path()) == (
        root / "shared_root_system" / "CUDA_cache_abcdef12"
    )


def test_custom_kernel_cache_dir_wins_over_root(
    isolated_cache_root, tmp_path
):
    """An explicit kernel cache_dir still overrides the shared root."""
    custom = tmp_path / "elsewhere"
    locator = CUBIECacheLocator(
        system_name="shared_root_system",
        system_hash="abcdef1234567890",
        compile_settings_hash="fedcba0987654321",
        custom_cache_dir=custom,
    )
    assert Path(locator.get_cache_path()) == custom / "CUDA_cache_abcdef12"
