"""Tests for the kernel cache settings and CUBIECache configuration."""

import os
from pathlib import Path

import pytest
from numpy import float32

from cubie._env import kernel_cache_dir_default
from cubie.batchsolving.BatchSolverConfig import (
    BatchSolverConfig,
    CacheSettings,
)
from cubie.cubie_cache import CUBIECache

DEFAULT_CUBIE_CACHE_CONFIG_HASH = (
    "def456789012345678901234567890123456789012345678901234567890abcd"
)


@pytest.fixture(scope="function")
def cache_config(request):
    """Build CacheSettings with optional overrides."""
    params = getattr(request, "param", {})
    return CacheSettings(
        cache_enabled=params.get("enabled", True),
        cache_mode=params.get("mode", "hash"),
        max_cache_entries=params.get("max_entries", 0),
        cache_dir=params.get("cache_dir", None),
    )


@pytest.fixture(scope="function")
def cubie_cache(request, tmp_path, precision):
    """Fixture to create CUBIECache with optional overrides."""
    params = getattr(request, "param", {})
    system_name = "test_system"
    system_hash = "abc123"
    return CUBIECache(
        system_name=system_name,
        system_hash=system_hash,
        config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        max_entries=params.get("max_entries", None),
        mode=params.get("mode", "hash"),
        custom_cache_dir=params.get("custom_cache_dir", None),
    )


class TestCacheSettingDefaults:
    """Tests for the cache field defaults."""

    def test_cache_setting_defaults(self, isolated_cache_root):
        """A bare config caches in hash mode at the shared root."""
        config = BatchSolverConfig(precision=float32).cache
        assert config.cache_enabled is True
        assert config.cache_mode == "hash"
        assert config.max_cache_entries == 0
        assert config.cache_dir is None

    def test_cache_settings_stay_out_of_the_hash(self, tmp_path):
        """Cache settings never enter values_hash."""
        base = BatchSolverConfig(precision=float32)
        other = BatchSolverConfig(
            precision=float32,
            cache=CacheSettings(
                cache_enabled=False,
                cache_mode="flush_on_change",
                max_cache_entries=3,
                cache_dir=tmp_path,
            ),
        )
        assert base.values_hash == other.values_hash

    def test_loose_keys_update_the_nested_settings(self, tmp_path):
        """Loose cache_* keys evolve the nested CacheSettings."""
        base = BatchSolverConfig(precision=float32)
        replacement, recognized, changed = base.update(
            cache_mode="flush_on_change", cache_dir=tmp_path
        )
        assert {"cache_mode", "cache_dir"} <= recognized
        assert {"cache_mode", "cache_dir"} <= changed
        assert replacement.cache.cache_mode == "flush_on_change"
        assert replacement.cache.cache_dir == tmp_path

    def test_shorthand_with_loose_keys(self, tmp_path):
        """A cache shorthand is the base for loose keys in one update."""
        base = BatchSolverConfig(precision=float32)
        replacement, _, _ = base.update(
            cache="flush_on_change", cache_dir=tmp_path
        )
        assert replacement.cache.cache_mode == "flush_on_change"
        assert replacement.cache.cache_dir == tmp_path


class TestCacheModeValidation:
    """Tests for cache_mode validation."""

    @pytest.mark.parametrize(
        "cache_config,mode",
        [
            ({"mode": "hash"}, "hash"),
            ({"mode": "flush_on_change"}, "flush_on_change"),
        ],
        indirect=["cache_config"],
    )
    def test_cache_mode_valid(self, cache_config, mode):
        """Verify supported modes are accepted."""
        assert cache_config.cache_mode == mode

    def test_cache_mode_validation(self, precision):
        """Verify mode only accepts 'hash' or 'flush_on_change'."""
        with pytest.raises(ValueError):
            CacheSettings(cache_mode="invalid_mode")


class TestMaxCacheEntriesValidation:
    """Tests for max_cache_entries validation."""

    @pytest.mark.parametrize(
        "cache_config,expected",
        [
            ({"max_entries": 0}, 0),
            ({"max_entries": 100}, 100),
        ],
        indirect=["cache_config"],
    )
    def test_max_cache_entries_valid(self, cache_config, expected):
        """Verify max_entries accepts zero and positive values."""
        assert cache_config.max_cache_entries == expected

    def test_max_cache_entries_validation(self, precision):
        """Verify max_entries rejects negative values."""
        with pytest.raises(ValueError):
            CacheSettings(max_cache_entries=-1)


class TestCacheDirConversion:
    """Tests for cache_dir conversion."""

    @pytest.mark.parametrize(
        "cache_config,expected",
        [
            ({"cache_dir": None}, None),
            ({"cache_dir": Path("/tmp/cache")}, Path("/tmp/cache")),
            ({"cache_dir": "/tmp/cache"}, Path("/tmp/cache")),
        ],
        indirect=["cache_config"],
    )
    def test_cache_dir_conversion(self, cache_config, expected):
        """Verify cache_dir accepts optional Path or str inputs."""
        if expected is None:
            assert cache_config.cache_dir is None
        else:
            assert cache_config.cache_dir == expected
            assert isinstance(cache_config.cache_dir, Path)


class TestCUBIECacheMaxEntries:
    """Tests for CUBIECache max_entries parameter."""

    @pytest.mark.parametrize(
        "cubie_cache,max_entries",
        [
            ({"max_entries": 5}, 5),
        ],
        indirect=["cubie_cache"],
    )
    def test_cubie_cache_max_entries_stored(self, cubie_cache, max_entries):
        """Verify max_entries override is retained."""
        assert cubie_cache._max_entries == max_entries

    def test_cubie_cache_max_entries_default(
        self, isolated_cache_root, cubie_cache
    ):
        """Verify max_entries defaults to 0, disabling eviction."""
        assert cubie_cache._max_entries == 0


class TestEnforceCacheLimitNoEviction:
    """Tests for enforce_cache_limit when under limit."""

    def test_enforce_cache_limit_no_eviction_under_limit(
        self, tmp_path, precision
    ):
        """Verify no eviction when file count < max_entries."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create 3 .nbi files (under limit of 10)
        for i in range(3):
            (cache_dir / f"cache_{i}.nbi").write_text("test")
            (cache_dir / f"cache_{i}.0.nbc").write_text("test")

        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            max_entries=10,
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.enforce_cache_limit()

        # All files should remain
        assert len(list(cache_dir.glob("*.nbi"))) == 3
        assert len(list(cache_dir.glob("*.nbc"))) == 3


class TestEnforceCacheLimitEviction:
    """Tests for enforce_cache_limit eviction behavior."""

    def test_enforce_cache_limit_evicts_oldest(self, tmp_path, precision):
        """Verify oldest files evicted when limit exceeded."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create 5 .nbi files with different mtimes
        for i in range(5):
            nbi_file = cache_dir / f"cache_{i}.nbi"
            nbc_file = cache_dir / f"cache_{i}.0.nbc"
            nbi_file.write_text("test")
            nbc_file.write_text("test")
            # Set mtime in order (oldest first)
            mtime = 1000000 + i * 100
            os.utime(nbi_file, (mtime, mtime))
            os.utime(nbc_file, (mtime, mtime))

        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            max_entries=3,
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.enforce_cache_limit()

        # Should have evicted 3 files (5 - 3 + 1 = 3) to make room
        remaining_nbi = list(cache_dir.glob("*.nbi"))
        assert len(remaining_nbi) == 2

        # The oldest files (cache_0, cache_1, cache_2) should be gone
        remaining_names = [f.stem for f in remaining_nbi]
        assert "cache_0" not in remaining_names
        assert "cache_1" not in remaining_names
        assert "cache_2" not in remaining_names
        # Newest files should remain
        assert "cache_3" in remaining_names
        assert "cache_4" in remaining_names


class TestEnforceCacheLimitDisabled:
    """Tests for enforce_cache_limit when disabled."""

    def test_enforce_cache_limit_zero_disables(self, tmp_path, precision):
        """Verify max_entries=0 disables eviction."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create 10 .nbi files
        for i in range(10):
            (cache_dir / f"cache_{i}.nbi").write_text("test")
            (cache_dir / f"cache_{i}.0.nbc").write_text("test")

        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            max_entries=0,
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.enforce_cache_limit()

        # All files should remain (eviction disabled)
        assert len(list(cache_dir.glob("*.nbi"))) == 10
        assert len(list(cache_dir.glob("*.nbc"))) == 10


class TestEnforceCacheLimitPairs:
    """Tests for enforce_cache_limit .nbi/.nbc pairing."""

    def test_enforce_cache_limit_pairs_nbi_nbc(self, tmp_path, precision):
        """Verify .nbi and .nbc files evicted together."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create 3 .nbi files with multiple .nbc files each
        for i in range(3):
            nbi_file = cache_dir / f"cache_{i}.nbi"
            nbi_file.write_text("test")
            # Each .nbi has multiple .nbc files
            (cache_dir / f"cache_{i}.0.nbc").write_text("test")
            (cache_dir / f"cache_{i}.1.nbc").write_text("test")
            # Set mtime in order
            mtime = 1000000 + i * 100
            os.utime(nbi_file, (mtime, mtime))

        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            max_entries=2,
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.enforce_cache_limit()

        # Should have evicted 2 entries (3 - 2 + 1 = 2)
        remaining_nbi = list(cache_dir.glob("*.nbi"))
        assert len(remaining_nbi) == 1

        # The .nbc files for evicted entries should also be gone
        remaining_nbc = list(cache_dir.glob("*.nbc"))
        assert len(remaining_nbc) == 2  # Only cache_2's .nbc files remain

        # Verify cache_0 and cache_1 .nbc files are gone
        remaining_nbc_names = [f.name for f in remaining_nbc]
        assert "cache_0.0.nbc" not in remaining_nbc_names
        assert "cache_0.1.nbc" not in remaining_nbc_names
        assert "cache_1.0.nbc" not in remaining_nbc_names
        assert "cache_1.1.nbc" not in remaining_nbc_names


class TestCUBIECacheModeStored:
    """Tests for CUBIECache mode parameter."""

    def test_cubie_cache_mode_stored(self, tmp_path, precision):
        """Verify mode is stored on CUBIECache instance."""
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
            mode="flush_on_change",
        )
        assert cache._mode == "flush_on_change"

    def test_cubie_cache_mode_default(self, tmp_path, precision):
        """Verify mode defaults to 'hash'."""
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )
        assert cache._mode == "hash"


class TestFlushCacheRemovesFiles:
    """Tests for flush_cache file removal."""

    def test_flush_cache_removes_files(self, tmp_path, precision):
        """Verify flush_cache removes all cache files."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Create some cache files
        (cache_dir / "cache_0.nbi").write_text("test")
        (cache_dir / "cache_0.0.nbc").write_text("test")
        (cache_dir / "cache_1.nbi").write_text("test")
        (cache_dir / "cache_1.0.nbc").write_text("test")

        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )
        # Override cache path to use tmp_path
        cache._cache_path = str(cache_dir)

        cache.flush_cache()

        # Directory should exist but be empty
        assert cache_dir.exists()
        assert len(list(cache_dir.glob("*"))) == 0


class TestFlushCacheRecreatesDirectory:
    """Tests for flush_cache directory recreation."""

    def test_flush_cache_recreates_directory(self, tmp_path, precision):
        """Verify flush_cache creates empty directory after removal."""
        cache_dir = tmp_path / "test_system" / "cache"
        cache_dir.mkdir(parents=True)

        # Add a file
        (cache_dir / "test.nbi").write_text("test")
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )
        cache._cache_path = str(cache_dir)

        cache.flush_cache()

        # Directory should be recreated and empty
        assert cache_dir.exists()
        assert cache_dir.is_dir()
        assert len(list(cache_dir.iterdir())) == 0

    def test_flush_cache_handles_missing_directory(self, tmp_path, precision):
        """Verify flush_cache handles non-existent directory."""
        cache_dir = tmp_path / "nonexistent" / "cache"
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )
        cache._cache_path = str(cache_dir)

        # Should not raise an error
        cache.flush_cache()

        # Directory should now exist
        assert cache_dir.exists()


class TestCustomCacheDir:
    """Tests for custom_cache_dir parameter."""

    def test_custom_cache_dir_used(self, tmp_path, precision):
        """Verify custom_cache_dir overrides default path."""
        custom_dir = tmp_path / "my_custom_cache"

        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            custom_cache_dir=custom_dir,
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )

        assert Path(cache._cache_path) == custom_dir / "CUDA_cache_abc123"

    def test_custom_cache_dir_none_uses_default(
        self, precision, isolated_cache_root
    ):
        """Verify None cache_dir uses the shared cache root path."""
        cache = CUBIECache(
            system_name="test_system",
            system_hash="abc123",
            custom_cache_dir=None,
            config_hash=DEFAULT_CUBIE_CACHE_CONFIG_HASH,
        )

        # Should use default path based on the shared cache root
        assert "test_system" in cache._cache_path
        assert cache._cache_path.endswith("CUDA_cache_abc123")


class TestParseCacheParam:
    """Tests for BatchSolverKernel._parse_cache_param."""

    def test_parse_cache_param_true(self, simple_system):
        """Verify cache=True enables hash-mode caching."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache=True,
        )

        assert kernel.compile_settings.cache.cache_enabled is True
        assert kernel.compile_settings.cache.cache_mode == "hash"
        # Without a cache argument the directory is the environment
        # default: CUBIE_KERNEL_CACHE_DIR when set, None otherwise.
        expected_dir = kernel_cache_dir_default()
        if expected_dir is None:
            assert kernel.compile_settings.cache.cache_dir is None
        else:
            assert kernel.compile_settings.cache.cache_dir == Path(
                expected_dir
            )

    def test_parse_cache_param_false(self, simple_system):
        """Verify cache=False disables caching."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache=False,
        )

        assert kernel.compile_settings.cache.cache_enabled is False

    def test_parse_cache_param_flush_on_change(self, simple_system):
        """Verify cache='flush_on_change' sets flush mode."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache="flush_on_change",
        )

        assert kernel.compile_settings.cache.cache_enabled is True
        assert kernel.compile_settings.cache.cache_mode == "flush_on_change"

    def test_parse_cache_param_path(self, simple_system, tmp_path):
        """Verify cache=Path sets custom cache_dir."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        custom_path = tmp_path / "custom_cache"
        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache=custom_path,
        )

        assert kernel.compile_settings.cache.cache_enabled is True
        assert kernel.compile_settings.cache.cache_mode == "hash"
        assert kernel.compile_settings.cache.cache_dir == custom_path

    def test_parse_cache_param_string_path(self, simple_system, tmp_path):
        """Verify cache=string path sets custom cache_dir."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        custom_path = str(tmp_path / "custom_cache")
        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache=custom_path,
        )

        assert kernel.compile_settings.cache.cache_enabled is True
        assert kernel.compile_settings.cache.cache_dir == Path(custom_path)


class TestKernelCacheSettings:
    """Tests for the cache fields on the kernel's compile settings."""

    def test_kernel_settings_override_the_cache_shorthand(
        self, simple_system, tmp_path
    ):
        """Explicit cache_* keys win over the cache shorthand."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache="flush_on_change",
            kernel_settings={"cache_mode": "hash", "cache_dir": tmp_path},
        )

        assert kernel.compile_settings.cache.cache_enabled is True
        assert kernel.compile_settings.cache.cache_mode == "hash"
        assert kernel.compile_settings.cache.cache_dir == tmp_path


class TestSetCacheDir:
    """Tests for BatchSolverKernel.set_cache_dir method."""

    def test_set_cache_dir_accepts_string(self, simple_system, tmp_path):
        """Verify set_cache_dir accepts string path."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache=True,
        )
        new_path_str = str(tmp_path / "string_cache_dir")

        kernel.set_cache_dir(new_path_str)

        assert kernel.compile_settings.cache.cache_dir == Path(new_path_str)
        assert isinstance(kernel.compile_settings.cache.cache_dir, Path)

    def test_set_cache_dir_accepts_path(self, simple_system, tmp_path):
        """Verify set_cache_dir accepts Path object."""
        from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

        kernel = BatchSolverKernel(
            simple_system,
            algorithm_settings={"algorithm": "euler"},
            cache=True,
        )
        new_path = tmp_path / "path_cache_dir"

        kernel.set_cache_dir(new_path)

        assert kernel.compile_settings.cache.cache_dir == new_path
        assert isinstance(kernel.compile_settings.cache.cache_dir, Path)


@pytest.fixture(scope="session")
def simple_system():
    """Create a minimal ODE system for cache parameter testing."""
    from cubie.odesystems.symbolic.symbolicODE import create_ODE_system

    equations = ["dx0 = -x0"]
    states = {"x0": 1.0}

    return create_ODE_system(
        dxdt=equations,
        states=states,
        name="simple_cache_test_system",
    )


class TestSolverCacheParam:
    """Tests for Solver cache parameter pass-through."""

    def test_solver_cache_param_passed_to_kernel(
        self, simple_system, tmp_path
    ):
        """Verify Solver passes cache param to kernel."""
        from cubie.batchsolving.solver import Solver

        custom_path = tmp_path / "solver_cache"
        solver = Solver(simple_system, cache=custom_path)

        assert solver.kernel.compile_settings.cache.cache_enabled is True
        assert solver.kernel.compile_settings.cache.cache_dir == custom_path

    def test_solver_cache_true_default(self, simple_system):
        """Verify cache=True is the default."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system)

        assert solver.kernel.compile_settings.cache.cache_enabled is True
        assert solver.kernel.compile_settings.cache.cache_mode == "hash"

    def test_solver_cache_false(self, simple_system):
        """Verify cache=False disables caching."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=False)

        assert solver.kernel.compile_settings.cache.cache_enabled is False

    def test_solver_cache_flush_on_change(self, simple_system):
        """Verify cache='flush_on_change' sets mode."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache="flush_on_change")

        cache = solver.kernel.compile_settings.cache
        assert cache.cache_enabled is True
        assert cache.cache_mode == "flush_on_change"


class TestSolverCacheProperties:
    """Tests for Solver cache-related properties."""

    def test_solver_cache_enabled_property(self, simple_system):
        """Verify cache_enabled returns kernel value."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        assert solver.cache_enabled is True

        solver_disabled = Solver(simple_system, cache=False)
        assert solver_disabled.cache_enabled is False

    def test_solver_cache_mode_property(self, simple_system):
        """Verify cache_mode returns kernel value."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        assert solver.cache_mode == "hash"

        solver_flush = Solver(simple_system, cache="flush_on_change")
        assert solver_flush.cache_mode == "flush_on_change"

    def test_solver_cache_dir_property(self, simple_system, tmp_path):
        """Verify cache_dir returns kernel value."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        # Without a cache argument the directory is the environment
        # default: CUBIE_KERNEL_CACHE_DIR when set, None otherwise.
        expected_dir = kernel_cache_dir_default()
        if expected_dir is None:
            assert solver.cache_dir is None
        else:
            assert solver.cache_dir == Path(expected_dir)

        custom_path = tmp_path / "cache_dir_test"
        solver_custom = Solver(simple_system, cache=custom_path)
        assert solver_custom.cache_dir == custom_path


class TestSolverSetCacheDir:
    """Tests for Solver.set_cache_dir method."""

    def test_solver_set_cache_dir_delegates(self, simple_system, tmp_path):
        """Verify set_cache_dir calls kernel method."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        new_path = tmp_path / "new_cache"

        solver.set_cache_dir(new_path)

        assert solver.cache_dir == new_path
        assert solver.kernel.compile_settings.cache.cache_dir == new_path

    def test_solver_set_cache_dir_string(self, simple_system, tmp_path):
        """Verify set_cache_dir accepts string."""
        from cubie.batchsolving.solver import Solver
        from pathlib import Path

        solver = Solver(simple_system, cache=True)
        new_path_str = str(tmp_path / "string_cache")

        solver.set_cache_dir(new_path_str)

        assert solver.cache_dir == Path(new_path_str)

    def test_solver_set_cache_dir_path(self, simple_system, tmp_path):
        """Verify set_cache_dir accepts Path object."""
        from cubie.batchsolving.solver import Solver

        solver = Solver(simple_system, cache=True)
        new_path = tmp_path / "path_cache"

        solver.set_cache_dir(new_path)

        assert solver.cache_dir == new_path
