"""Tests for the CUBIE_* environment variable registry."""

import pytest
from numpy import float32

from cubie._env import (
    env_bool,
    lineinfo_default,
    max_cache_entries_default,
)
from cubie.CUDAFactory import CUDAFactoryConfig
from cubie.cuda_simsafe import JITFlags, compile_kwargs, get_jit_kwargs


class TestEnvBool:
    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "On"])
    def test_truthy(self, monkeypatch, raw):
        monkeypatch.setenv("CUBIE_TEST_FLAG", raw)
        assert env_bool("CUBIE_TEST_FLAG") is True

    @pytest.mark.parametrize("raw", ["0", "false", "No", "off", ""])
    def test_falsy(self, monkeypatch, raw):
        monkeypatch.setenv("CUBIE_TEST_FLAG", raw)
        assert env_bool("CUBIE_TEST_FLAG") is False

    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("CUBIE_TEST_FLAG", raising=False)
        assert env_bool("CUBIE_TEST_FLAG") is False
        assert env_bool("CUBIE_TEST_FLAG", default=True) is True

    def test_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("CUBIE_TEST_FLAG", "maybe")
        with pytest.raises(ValueError, match="CUBIE_TEST_FLAG"):
            env_bool("CUBIE_TEST_FLAG")


class TestLineinfoDefault:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("CUBIE_LINEINFO", raising=False)
        assert lineinfo_default() is False

    def test_env_enables(self, monkeypatch):
        monkeypatch.setenv("CUBIE_LINEINFO", "1")
        assert lineinfo_default() is True


class TestMaxCacheEntriesDefault:
    def test_default_disables_eviction(self, monkeypatch):
        monkeypatch.delenv("CUBIE_MAX_CACHE_ENTRIES", raising=False)
        assert max_cache_entries_default() == 0

    def test_env_sets_limit(self, monkeypatch):
        monkeypatch.setenv("CUBIE_MAX_CACHE_ENTRIES", "7")
        assert max_cache_entries_default() == 7

    @pytest.mark.parametrize("raw", ["-1", "invalid"])
    def test_invalid_raises(self, monkeypatch, raw):
        monkeypatch.setenv("CUBIE_MAX_CACHE_ENTRIES", raw)
        with pytest.raises(ValueError, match="CUBIE_MAX_CACHE_ENTRIES"):
            max_cache_entries_default()


class TestLineinfoPrecedence:
    def test_config_default_follows_env(self, monkeypatch):
        monkeypatch.setenv("CUBIE_LINEINFO", "1")
        assert CUDAFactoryConfig(precision=float32).lineinfo is True
        monkeypatch.delenv("CUBIE_LINEINFO")
        assert CUDAFactoryConfig(precision=float32).lineinfo is False

    def test_explicit_arg_beats_env(self, monkeypatch):
        monkeypatch.setenv("CUBIE_LINEINFO", "1")
        cfg = CUDAFactoryConfig(
            precision=float32, jit_flags=JITFlags(lineinfo=False)
        )
        assert cfg.lineinfo is False

    def test_lineinfo_changes_values_hash(self):
        cfg = CUDAFactoryConfig(
            precision=float32, jit_flags=JITFlags(lineinfo=False)
        )
        before = cfg.values_hash
        replacement, recognized, changed = cfg.update({"lineinfo": True})
        assert recognized == {"lineinfo"}
        assert changed == {"lineinfo"}
        assert replacement.values_hash != before
        # The original snapshot is untouched.
        assert cfg.values_hash == before


@pytest.mark.nocudasim
class TestJitKwargs:
    def test_compile_kwargs_immutable(self):
        with pytest.raises(TypeError):
            compile_kwargs["lineinfo"] = True

    def test_explicit_value_wins(self, monkeypatch):
        monkeypatch.setenv("CUBIE_LINEINFO", "1")
        assert get_jit_kwargs(False)["lineinfo"] is False

    def test_none_defers_to_env(self, monkeypatch):
        monkeypatch.setenv("CUBIE_LINEINFO", "1")
        assert get_jit_kwargs()["lineinfo"] is True
        monkeypatch.delenv("CUBIE_LINEINFO")
        assert get_jit_kwargs()["lineinfo"] is False

    def test_base_defaults_preserved(self):
        kwargs = get_jit_kwargs(True)
        assert kwargs["fastmath"] == compile_kwargs["fastmath"]
        assert kwargs["lto"] == compile_kwargs["lto"]
