"""Tests for cubie_cache module."""

from hashlib import sha256
from pathlib import Path

import pytest
from numpy import array, float32

from attrs import define, field

from cubie.cuda_backend import IS_MLIR
from cubie.cubie_cache import (
    CUBIECacheLocator,
    CUBIECacheImpl,
    CUBIECache,
    CubieCacheHandler,
    CachePolicy,
    ALL_CACHE_PARAMETERS,
    _CacheFileLock,
    _abi_fingerprint_entries,
    toolchain_fingerprint,
)
from cubie._utils import package_source_hash


# --- Test fixtures (attrs classes for testing) ---


@define
class MockCompileSettings:
    """Simple attrs class for testing hash_compile_settings."""

    precision: type = float32
    value: int = 42


@define
class MockSettingsWithCallable:
    """Attrs class with eq=False field for testing."""

    precision: type = float32
    callback: object = field(default=None, eq=False)


@define
class MockNestedSettings:
    """Attrs class with nested attrs for testing."""

    precision: type = float32
    nested: MockCompileSettings = field(factory=MockCompileSettings)


@define
class MockSettingsWithArray:
    """Attrs class with numpy array for testing."""

    precision: type = float32
    data: object = field(factory=lambda: array([1.0, 2.0, 3.0]))


# --- CUBIECacheLocator tests ---


def test_cache_locator_get_cache_path():
    """Verify cache path includes system_hash subdirectory."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    path = locator.get_cache_path()
    assert "test_system" in path
    assert "abc123" in path  # system_hash in path
    assert path.endswith("CUDA_cache_abc123")


def test_cache_locator_get_source_stamp():
    """Verify source stamp combines system_hash and environment hash."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    assert (
        locator.get_source_stamp() == f"abc123-{toolchain_fingerprint()}"
    )


def test_toolchain_fingerprint_is_stable_hex_digest():
    """Verify the fingerprint is a deterministic sha256 digest."""
    digest = toolchain_fingerprint()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert toolchain_fingerprint() == digest


def test_fingerprint_covers_declared_abi_inputs_only():
    """The fingerprint holds only declared ABI/toolchain inputs.

    Every declared input (schema, Python ABI tag, backend id, backend
    package versions) is present; unrelated installed packages, paths,
    and host identity are absent.
    """
    from cubie.cuda_backend import CUDA_BACKEND

    entries = _abi_fingerprint_entries()
    keys = [entry.split("=")[0] for entry in entries]
    assert keys[0] == "schema"
    assert keys[1] == "python-abi"
    assert entries[2] == f"backend={CUDA_BACKEND}"
    # Backend serialization owners only — nothing else from the env.
    allowed = {
        "schema",
        "python-abi",
        "backend",
        "numba-cuda",
        "numba",
        "llvmlite",
        "cubie-numba-cuda-mlir",
        "numba-cuda-mlir",
    }
    for entry in entries:
        assert entry.split("==")[0].split("=")[0] in allowed
        # No absolute paths or host identity can hide in an entry.
        assert "\\" not in entry
        assert "/" not in entry
    # An unrelated-but-installed package must not participate.
    assert not any(entry.startswith("numpy==") for entry in entries)
    assert not any(entry.startswith("attrs==") for entry in entries)


def test_fingerprint_moves_with_declared_inputs():
    """Changing any declared entry changes the digest; reordering or
    dropping an unrelated candidate does not exist as an input."""
    entries = _abi_fingerprint_entries()
    baseline = sha256("\n".join(entries).encode("utf-8")).hexdigest()
    assert toolchain_fingerprint() == baseline

    changed = list(entries)
    changed[0] = "schema=cubie-cache-v999"
    moved = sha256("\n".join(changed).encode("utf-8")).hexdigest()
    assert moved != baseline


def test_cache_file_lock_cleans_up_after_timeout(tmp_path):
    """An unsuccessful entry closes and clears its file handle."""
    path = tmp_path / "cache.lock"
    waiting_lock = _CacheFileLock(path, timeout=0.0)

    with _CacheFileLock(path):
        with pytest.raises(TimeoutError):
            waiting_lock.__enter__()

    assert waiting_lock._handle is None


def test_cache_locator_get_disambiguator():
    """Verify disambiguator returns truncated settings hash."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456789012345678",
    )
    disambiguator = locator.get_disambiguator()
    assert len(disambiguator) == 16
    assert disambiguator == "def4567890123456"


def test_cache_locator_from_function_raises():
    """Verify from_function raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        CUBIECacheLocator.from_function(None, None)


def test_cache_locator_path_includes_system_hash():
    """Verify cache path follows generated/<system_name>/CUDA_cache_<hash[
    :8]>."""
    locator = CUBIECacheLocator(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    path = locator.get_cache_path()
    assert "test_system" in path
    assert "abc123" in path
    assert path.endswith("CUDA_cache_abc123")


# --- ALL_CACHE_PARAMETERS and CacheConfig tests ---


def test_all_cache_parameters_contains_expected_keys():
    """Verify ALL_CACHE_PARAMETERS contains expected cache parameter names."""
    expected_keys = {
        "cache_enabled",
        "cache_mode",
        "max_cache_entries",
        "cache_dir",
    }
    assert ALL_CACHE_PARAMETERS == expected_keys


def test_cache_policy_has_cache_enabled_field():
    """Verify CachePolicy has cache_enabled field (not enabled)."""
    policy = CachePolicy()
    assert hasattr(policy, "cache_enabled")
    assert not hasattr(policy, "enabled")
    # Caching is on by default for every bare policy.
    assert policy.cache_enabled is True

    policy_disabled = CachePolicy(cache_enabled=False)
    assert policy_disabled.cache_enabled is False


def test_policy_replacement_never_flushes_previous_cache(tmp_path):
    """A policy change drops the configured cache without flushing.

    A cache configured under the previous policy may point at a
    directory the new policy does not own (for example the CI
    kernel-cache artifact); switching to flush_on_change and
    invalidating must not delete it.
    """
    old_dir = tmp_path / "artifact"
    handler = CubieCacheHandler(
        CachePolicy(cache_enabled=True, cache_dir=old_dir),
        system_name="flush_scope_test",
    )
    cache = handler.configured_cache("aaaa1111", "bbbb2222")
    cache.cache_path.mkdir(parents=True, exist_ok=True)
    marker = cache.cache_path / "entry.nbi"
    marker.write_bytes(b"payload")

    changed = handler.update_policy(
        CachePolicy(
            cache_enabled=True,
            cache_mode="flush_on_change",
            cache_dir=tmp_path / "elsewhere",
        )
    )
    assert changed
    assert handler.cache is None
    handler.invalidate()
    assert marker.exists()


def test_cache_policy_holds_no_identity():
    """CachePolicy is pure policy: no system name or hash fields."""
    policy = CachePolicy()
    assert not hasattr(policy, "system_name")
    assert not hasattr(policy, "system_hash")
    handler = CubieCacheHandler(policy, system_name="my_system")
    assert handler.system_name == "my_system"


# --- CUBIECacheImpl tests ---


def test_cache_impl_locator_property():
    """Verify locator property returns CUBIECacheLocator."""
    impl = CUBIECacheImpl(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    assert isinstance(impl.locator, CUBIECacheLocator)


def test_cache_impl_filename_base():
    """Verify filename_base format."""
    impl = CUBIECacheImpl(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456789012345678",
    )
    assert impl.filename_base.startswith("test_system-")
    assert "def4567890123456" in impl.filename_base


def test_cache_impl_check_cachable():
    """Verify check_cachable accepts what the backend can serialize.

    numba-cuda kernels are always cachable; the MLIR compile-result
    scheme inspects targetoptions and refuses results that link
    external files.
    """
    impl = CUBIECacheImpl(
        system_name="test_system",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    if IS_MLIR:

        class LinkFreeResult:
            metadata = {"targetoptions": {"link": []}}

        class LinkedResult:
            metadata = {"targetoptions": {"link": ["kernels.cu"]}}

        assert impl.check_cachable(LinkFreeResult()) is True
        with pytest.raises(RuntimeError):
            impl.check_cachable(LinkedResult())
    else:
        assert impl.check_cachable(None) is True


# --- CUBIECache tests ---


def test_cubie_cache_init():
    """Verify CUBIECache initializes with system info."""
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456789012345678901234567890123456789012345678901234567890abcd",
    )
    assert cache._system_name == "test_system"
    assert cache._system_hash == "abc123"
    assert cache._name == "CUBIECache(test_system)"


def test_cubie_cache_index_key():
    """The index key includes CuBIE and launch settings."""
    config_hash = (
        "def456789012345678901234567890123456789012345678901234567890abcd"
    )
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash=config_hash,
    )

    # Create a mock codegen object
    class MockCodegen:
        def magic_tuple(self):
            return ("magic", "tuple")

    sig = ("float32", "float32")
    codegen = MockCodegen()

    key = cache._index_key(sig, codegen)

    # Key is (sig, magic_tuple, system_hash, config_hash, package_hash)
    assert len(key) == 5
    assert key[0] == sig
    assert key[1] == ("magic", "tuple")
    assert key[2] == "abc123"
    assert key[3] == config_hash
    assert key[4] == package_source_hash()

    cache._launch_config_key = (1, 2, 3)
    launch_key = cache._index_key(sig, codegen)
    assert launch_key[:-1] == key
    assert launch_key[-1] == ("launch_config", (1, 2, 3))


def test_cubie_cache_path(isolated_cache_root):
    """Verify cache_path includes system_hash subdirectory."""
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456789012345678901234567890123456"
        "789012345678901234567890abcd",
    )
    path_str = str(cache.cache_path)
    assert "test_system" in path_str
    assert "abc123" in path_str  # system_hash in path
    assert "CUDA_cache" in path_str


def test_kernel_cache_dir_env_relocates(tmp_path, monkeypatch):
    """An explicit cache directory overrides the environment."""
    env_dir = tmp_path / "artifact"
    monkeypatch.setenv("CUBIE_KERNEL_CACHE_DIR", str(env_dir))
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
    )
    assert str(cache.cache_path).startswith(str(env_dir))
    explicit_dir = tmp_path / "explicit"
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
        custom_cache_dir=explicit_dir,
    )
    assert str(cache.cache_path).startswith(str(explicit_dir))


def test_max_cache_entries_env_default(monkeypatch):
    """An explicit cache limit overrides the environment."""
    monkeypatch.setenv("CUBIE_MAX_CACHE_ENTRIES", "0")
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
    )
    assert cache._max_entries == 0
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456",
        max_entries=3,
    )
    assert cache._max_entries == 3


# --- BatchSolverKernel integration tests ---


def test_batch_solver_kernel_builds_without_error(solverkernel):
    """Verify BatchSolverKernel builds successfully in all modes.

    This smoke test confirms the kernel compilation path executes
    without raising exceptions, regardless of CUDA/CUDASIM mode.
    """
    # Build the kernel - this exercises the full compilation path
    kernel = solverkernel.kernel

    # Verify kernel was created
    assert kernel is not None


# --- CUDASIM Mode Compatibility Tests ---


def test_cache_locator_instantiation_works():
    """Verify CUBIECacheLocator can be instantiated regardless of mode."""
    locator = CUBIECacheLocator(
        system_name="cudasim_test",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    # Path operations should work
    assert locator.get_cache_path() is not None
    assert (
        locator.get_source_stamp() == f"abc123-{toolchain_fingerprint()}"
    )
    assert locator.get_disambiguator() == "def456"


def test_cache_impl_instantiation_works():
    """Verify CUBIECacheImpl can be instantiated regardless of mode."""
    impl = CUBIECacheImpl(
        system_name="cudasim_test",
        system_hash="abc123",
        compile_settings_hash="def456",
    )
    # Properties should be accessible
    assert impl.locator is not None
    assert impl.filename_base is not None


# --- CubieCacheHandler tests ---


def _handler(cache_enabled, system_name, system_hash):
    """Build a handler with a fresh policy; hashes stay call-time."""
    del system_hash
    return CubieCacheHandler(
        CachePolicy(cache_enabled=cache_enabled),
        system_name=system_name,
    )


def test_cache_handler_init_with_disabled_cache():
    """Verify CubieCacheHandler initializes with None cache when disabled."""
    handler = _handler(False, "test_system", "abc123")
    assert handler.cache is None
    assert handler.policy.cache_enabled is False


def test_cache_handler_init_with_enabled_cache():
    """An enabled handler creates its cache only for a build."""
    handler = _handler(True, "test_system", "abc123")
    assert handler.cache is None
    assert handler.policy.cache_enabled is True


def test_cache_handler_configured_cache_returns_none_when_disabled():
    """Verify configured_cache() returns None when cache disabled."""
    handler = _handler(False, "test_system", "abc123")
    result = handler.configured_cache("abc123", "compile_settings_hash_123")
    assert result is None


def test_cache_handler_configured_cache_sets_hashes():
    """Each build receives a new cache with fixed hashes."""
    handler = _handler(True, "test_system", "abc123")
    compile_hash = (
        "def456789012345678901234567890123456789012345678901234567890abcd"
    )
    first = handler.configured_cache("abc123", compile_hash)
    second = handler.configured_cache("abc123", "another_hash")
    assert first is not second
    assert first._compile_settings_hash == compile_hash
    assert first._system_hash == "abc123"
    assert second._compile_settings_hash == "another_hash"


def test_cache_handler_flush_handles_none_cache():
    """Verify flush() does not error when cache is None."""
    handler = _handler(False, "test_system", "abc123")
    # This should not raise
    handler.flush()


def test_cache_handler_policy_replacement_enables_builds():
    """Installing an enabling policy affects the next build."""
    handler = _handler(False, "test_system", "abc123")
    assert handler.configured_cache("abc123", "compile_hash") is None

    changed = handler.update_policy_params({"cache_enabled": True})
    assert changed is True
    cache = handler.configured_cache("abc123", "compile_hash")
    assert isinstance(cache, CUBIECache)


# --- BatchSolverKernel cache integration tests ---


def test_batch_solver_kernel_extracts_system_hash_from_symbolic_ode(
    solverkernel, system
):
    """Verify BatchSolverKernel extracts fn_hash from SymbolicODE."""
    # SymbolicODE should have fn_hash attribute
    assert hasattr(system, "fn_hash")

    # The handler receives identity hashes at build time; the kernel
    # passes system.fn_hash when configuring the dispatcher cache.
    cache = solverkernel.cache_handler.configured_cache(
        system.fn_hash, "d" * 64
    )
    assert cache is None or system.fn_hash[:8] in str(cache.cache_path)


def test_batch_solver_kernel_uses_name_from_system(solverkernel, system):
    """Verify BatchSolverKernel uses system.name for cache directory."""
    # System should have a name
    assert hasattr(system, "name")

    # If system.name is set, it should be used; otherwise hash prefix
    handler = solverkernel.cache_handler
    if system.name and system.name != system.fn_hash:
        assert handler.system_name == system.name
    else:
        assert handler.system_name.startswith("unnamed_")


def test_batch_solver_kernel_handles_none_cache_settings(
    system,
    step_controller_settings,
    algorithm_settings,
    output_settings,
    memory_settings,
    loop_settings,
):
    """Verify BatchSolverKernel works when cache_settings=None."""
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

    # This should not raise - cache_settings defaults to None
    kernel = BatchSolverKernel(
        system,
        step_control_settings=step_controller_settings,
        algorithm_settings=algorithm_settings,
        output_settings=output_settings,
        memory_settings=memory_settings,
        loop_settings=loop_settings,
        cache=False,
        cache_settings=None,
    )
    assert kernel.cache_handler is not None
    assert kernel.cache_handler.cache is None


def test_batch_solver_kernel_update_forwards_cache_params(
    isolated_cache_root,
    solverkernel_mutable,
):
    """Verify update(cache_mode='flush_on_change') is recognized.

    Runs under an isolated cache root: switching a shared-system
    kernel to flush_on_change while its handler points at the CI
    artifact directory would flush the precompiled artifact.
    """
    # Initial mode should be 'hash' (default)
    initial_mode = solverkernel_mutable.cache_handler.policy.cache_mode
    assert initial_mode == "hash"

    # Update cache mode
    recognized = solverkernel_mutable.update(cache_mode="flush_on_change")

    # Verify cache_mode is recognized and updated
    assert "cache_mode" in recognized
    assert (
        solverkernel_mutable.cache_handler.policy.cache_mode
        == "flush_on_change"
    )


# --- Integration tests for complete cache flow ---


def test_cache_handler_uses_symbolic_ode_fn_hash(system):
    """Verify configured caches key on the call-time system hash."""
    handler = _handler(True, system.name, system.fn_hash)

    cache = handler.configured_cache(system.fn_hash, "compile_hash")
    path_str = str(cache.cache_path)
    assert system.fn_hash[:8] in path_str


def test_solver_cache_configuration_flow(system):
    """Verify Solver correctly configures cache handler."""
    from cubie import Solver

    solver = Solver(
        system,
        algorithm="euler",
        cache=True,
        cache_mode="hash",
        max_cache_entries=5,
    )

    # Verify cache handler is configured
    assert solver.kernel.cache_handler is not None
    assert solver.kernel.cache_handler.policy.cache_mode == "hash"
    assert solver.kernel.cache_handler.policy.max_cache_entries == 5


def test_solver_kernel_update_cache_mode(
    isolated_cache_root, solverkernel_mutable
):
    """Verify BatchSolverKernel.update forwards cache parameters.

    Runs under an isolated cache root so the flush-on-change switch
    can never target the CI artifact directory.
    """
    # Update cache mode
    recognized = solverkernel_mutable.update(cache_mode="flush_on_change")

    assert "cache_mode" in recognized
    assert (
        solverkernel_mutable.cache_handler.policy.cache_mode
        == "flush_on_change"
    )


def test_cache_policy_change_leaves_identity_unchanged(
    isolated_cache_root, solverkernel_mutable, tmp_path
):
    """Cache-policy updates never touch configuration identity.

    Every cache parameter changes together, yet the kernel's and the
    system's config_hash and the object build cache stay untouched,
    and the replacement policy is rebound onto this kernel's own
    helper getter — never written onto the shared system. Runs under
    an isolated cache root so the flush-on-change switch can never
    target the CI artifact directory.
    """
    kernel = solverkernel_mutable
    hash_before = kernel.config_hash
    system_hash_before = kernel.system.config_hash
    cache_valid_before = kernel._cache_valid

    recognized = kernel.update(
        cache_mode="flush_on_change",
        max_cache_entries=7,
        cache_dir=tmp_path,
    )
    assert {"cache_mode", "max_cache_entries", "cache_dir"} <= recognized

    assert kernel.config_hash == hash_before
    assert kernel.system.config_hash == system_hash_before
    assert kernel._cache_valid == cache_valid_before

    policy = kernel.cache_handler.policy
    assert policy.cache_mode == "flush_on_change"
    assert policy.max_cache_entries == 7
    assert policy.cache_dir == tmp_path
    # The replacement policy is bound into this kernel's own helper
    # getter as request context, not stored on the shared system.
    assert kernel._solver_helper_fn.keywords["cache_policy"] is policy


# --- CUBIECache.enforce_cache_limit eviction-failure tests ---


def test_enforce_cache_limit_warns_and_continues_on_unlink_failure(
    tmp_path,
):
    """Verify enforce_cache_limit warns when an .nbi entry cannot be
    removed and silently continues past a matching .nbc removal
    failure, while still evicting entries that can be removed.

    A directory cannot be removed via Path.unlink(), so entries are
    represented as directories to force real OSError conditions
    without mocking.
    """
    cache = CUBIECache(
        system_name="evict_test",
        system_hash="evicthash1",
        config_hash="a" * 64,
        max_entries=1,
        custom_cache_dir=tmp_path,
    )
    cache_path = Path(cache.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    # "blocked" entries are directories: unlink() raises OSError for both
    # the .nbi and its matching .nbc file.
    blocked_nbi = cache_path / "blocked.nbi"
    blocked_nbi.mkdir()
    blocked_nbc = cache_path / "blocked.0.nbc"
    blocked_nbc.mkdir()

    # "removable" entries are real files: unlink() succeeds normally.
    removable_nbi = cache_path / "removable.nbi"
    removable_nbi.write_bytes(b"index")
    removable_nbc = cache_path / "removable.0.nbc"
    removable_nbc.write_bytes(b"data")

    with pytest.warns(RuntimeWarning, match="Failed to remove cache file"):
        cache.enforce_cache_limit()

    # The removable entry was evicted; the blocked one remains because
    # its unlink() failed (and the failure was reported via warn()).
    assert not removable_nbi.exists()
    assert not removable_nbc.exists()
    assert blocked_nbi.exists()
    assert blocked_nbc.exists()


def test_flush_cache_handles_rmtree_failure_silently(tmp_path):
    """Verify flush_cache does not raise when rmtree fails to remove
    the cache directory, e.g. because a file inside it is still open.

    Uses a real open file handle to force a genuine OSError from
    shutil.rmtree rather than mocking.
    """
    cache = CUBIECache(
        system_name="flush_test",
        system_hash="flushhash1",
        config_hash="b" * 64,
        custom_cache_dir=tmp_path,
    )
    cache_path = Path(cache.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    locked_file = cache_path / "locked.nbi"
    locked_file.write_bytes(b"data")

    handle = open(locked_file, "rb")
    try:
        # Should not raise even though rmtree cannot remove the open file.
        cache.flush_cache()
    finally:
        handle.close()

    # flush_cache recreates the directory regardless of rmtree's outcome.
    assert cache_path.exists()


# --- CubieCacheHandler additional coverage ---


def test_cache_handler_disable_stops_configured_caches():
    """A disabling policy replacement stops new configured caches."""
    handler = _handler(True, "disable_test", "h2")
    assert handler.configured_cache("h2", "compile_hash") is not None

    handler.update_policy_params({"cache_enabled": False})
    assert handler.configured_cache("h2", "compile_hash") is None


def test_cache_handler_configured_cache_keys_on_call_time_hash():
    """Each configured cache keys on the hash supplied at build time."""
    handler = _handler(True, "hash_update_test", "old_hash")

    first = handler.configured_cache("first_hash", "c" * 64)
    second = handler.configured_cache("second_hash", "c" * 64)
    assert first._system_hash == "first_hash"
    assert second._system_hash == "second_hash"


def test_cache_handler_cache_enabled_property():
    """Verify the cache_enabled property reflects the config."""
    handler_on = _handler(True, "enabled_prop", "h5")
    assert handler_on.cache_enabled is True

    handler_off = _handler(False, "disabled_prop", "h6")
    assert handler_off.cache_enabled is False
