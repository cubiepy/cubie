"""File-based caching infrastructure for CuBIE compiled kernels.

Provides cache classes that persist compiled CUDA kernels to disk,
enabling faster startup on subsequent runs with identical settings.
Cache files are stored in
``<cache root>/<system_name>/CUDA_cache_{system_hash[:8]}`` under the
shared cache root (:mod:`cubie.cache_root`) or in
``custom_dir/CUDA_cache_{system_hash[:8]}`` if a custom directory is
provided as an argument to "cache".

Notes
-----
Built on numba-cuda-mlir's internal ``MLIRCacheImpl``/``MLIRCache``,
which serialize cubin/PTX compile results.
"""

import os
from contextlib import AbstractContextManager
from functools import cache
from hashlib import sha256
from importlib.metadata import (
    PackageNotFoundError,
    version as dist_version,
)
from pathlib import Path
from shutil import rmtree
from sys import implementation, version_info
from time import monotonic, sleep
from typing import Optional
from warnings import warn

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from cubie._env import (
    active_block_schedule,
    kernel_cache_dir_default,
    max_cache_entries_default,
)
from numba_cuda_mlir.caching import (
    MLIRCacheImpl as CacheImpl,
    MLIRCache as CUDACache,
)
from numba_cuda_mlir.numba_cuda.core.caching import (
    _CacheLocator,
    IndexDataCacheFile,
)
from cubie.cache_root import get_cache_root
from cubie.time_logger import default_timelogger
from cubie._utils import package_source_hash

# Register compile timing event with custom messages
default_timelogger.register_event(
    "compile_cuda_kernel",
    "compile",
    "CUDA kernel compilation time",
    start_message="Compiling CUDA kernel...",
    stop_message=" Compilation complete in {duration:.3f}s",
)


# Retry bounds for the cache lock, in seconds.
_LOCK_RETRY_MIN = 0.0005
_LOCK_RETRY_MAX = 0.02

# Attempts at an index write that Windows transiently denies.
_IO_RETRY_ATTEMPTS = 60


def _retry_transient_io(operation):
    """Run ``operation``, retrying while Windows denies the file rename."""
    delay = _LOCK_RETRY_MIN
    for attempt in range(_IO_RETRY_ATTEMPTS):
        try:
            return operation()
        except PermissionError:
            if attempt == _IO_RETRY_ATTEMPTS - 1:
                raise
            sleep(delay)
            delay = min(delay * 2.0, _LOCK_RETRY_MAX)


class _CacheFileLock(AbstractContextManager):
    """Cross-process lock for one cache index."""

    def __init__(self, path: Path, timeout: float = 120.0) -> None:
        self._path = path
        self._timeout = timeout
        self._handle = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a+b")
        try:
            self._handle.seek(0, os.SEEK_END)
            if self._handle.tell() == 0:
                self._handle.write(b"\0")
                self._handle.flush()

            deadline = monotonic() + self._timeout
            delay = _LOCK_RETRY_MIN
            while True:
                try:
                    self._handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(
                            self._handle.fileno(), msvcrt.LK_NBLCK, 1
                        )
                    else:
                        fcntl.flock(
                            self._handle.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    return self
                except OSError:
                    if monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for cache lock {self._path}."
                        ) from None
                    sleep(delay)
                    delay = min(delay * 2.0, _LOCK_RETRY_MAX)
        except BaseException:
            try:
                self._handle.close()
            finally:
                self._handle = None
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is None:
            return False
        try:
            self._handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                self._handle.close()
            finally:
                self._handle = None
        return False


CACHE_SCHEMA_VERSION = "cubie-cache-v1"
"""Serialized-artifact schema tag folded into the ABI fingerprint."""

_BACKEND_ABI_DISTRIBUTIONS = (
    ("cubie-numba-cuda-mlir", "numba-cuda-mlir"),
)
"""Distributions whose versions define the artifact ABI.

Each tuple lists alternatives for one component; the first installed
one supplies the version, and none installed raises.
"""


def _abi_fingerprint_entries() -> list:
    """List the minimal ABI/toolchain inputs for artifact compatibility.

    Contains only inputs that can change the stored artifact's ABI or
    code-generation compatibility: the cache schema version, the
    Python implementation ABI tag, the active typed-IR block-schedule
    policy, and the backend/compiler package versions that own the
    serialization format. Workspace
    paths, host identity, unrelated installed packages, and arbitrary
    environment state are deliberately absent.
    Target code-generation capability (compute capability and toolkit
    magic) is carried per-overload by the backend's
    ``codegen.magic_tuple()`` inside each index key.
    """
    abi_tag = implementation.cache_tag
    if not abi_tag:
        abi_tag = (
            f"{implementation.name}-{version_info.major}."
            f"{version_info.minor}"
        )
    entries = [
        f"schema={CACHE_SCHEMA_VERSION}",
        f"python-abi={abi_tag}",
        f"block-schedule={active_block_schedule()}",
    ]
    for alternatives in _BACKEND_ABI_DISTRIBUTIONS:
        for dist_name in alternatives:
            try:
                entries.append(
                    f"{dist_name}=={dist_version(dist_name)}"
                )
                break
            except PackageNotFoundError:
                continue
        else:
            raise RuntimeError(
                f"No installed distribution among {alternatives} "
                "provides the numba-cuda-mlir ABI; the "
                "cache fingerprint cannot be constructed."
            )
    return entries


@cache
def toolchain_fingerprint() -> str:
    """Hash the minimal ABI/toolchain compatibility inputs."""
    joined = "\n".join(_abi_fingerprint_entries())
    return sha256(joined.encode("utf-8")).hexdigest()


class CUBIECacheLocator(_CacheLocator):
    """Locate cache files in CuBIE's generated directory structure.

    Directs cache files to ``generated/<system_name>/CUDA_cache/`` instead
    of the default ``__pycache__`` location used by numba.

    Parameters
    ----------
    system_name
        Name of the ODE system for directory organization.
    system_hash
        Hash representing the ODE system definition for freshness.
    compile_settings_hash
        Hash of compile settings for disambiguation.
    custom_cache_dir
        Optional custom cache directory. Overrides default location.
    """

    def __init__(
        self,
        system_name: str,
        system_hash: str,
        compile_settings_hash: str,
        custom_cache_dir: Optional[Path] = None,
    ) -> None:
        self._system_name = system_name
        self._system_hash = system_hash
        self._compile_settings_hash = compile_settings_hash

        if custom_cache_dir is None:
            cache_root = get_cache_root() / system_name
        else:
            cache_root = Path(custom_cache_dir)

        self._cache_root_dir = cache_root
        hash_dir = f"CUDA_cache_{system_hash[:8]}"
        self._cache_path = cache_root / hash_dir

    def get_cache_path(self) -> str:
        """Return the directory where cache files are stored.

        Returns
        -------
        str
            Absolute path to the cache directory.
        """
        return str(self._cache_path)

    def get_source_stamp(self) -> str:
        """Return a stamp representing source freshness.

        Returns
        -------
        str
            The system hash combined with the environment hash, so a
            change to any installed package invalidates the cache.
        """
        return f"{self._system_hash}-{toolchain_fingerprint()}"

    def get_disambiguator(self) -> str:
        """Return a string to disambiguate similar functions.

        Returns
        -------
        str
            First 16 characters of compile_settings_hash.
        """
        return self._compile_settings_hash[:16]

    @classmethod
    def from_function(cls, py_func, py_file):
        """Not used - CuBIE creates locators directly.

        Raises
        ------
        NotImplementedError
            This locator does not use the from_function pattern.
        """
        raise NotImplementedError(
            "CUBIECacheLocator requires explicit system info"
        )


class CUBIECacheImpl(CacheImpl):
    """Serialization logic for CuBIE compiled kernels.

    Delegates serialization to the backend's compile-result methods
    while using the CuBIE-specific cache locator for file paths.

    Parameters
    ----------
    system_name
        Name of the ODE system.
    system_hash
        Hash representing the ODE system definition.
    compile_settings_hash
        Hash of compile settings for cache key.
    custom_cache_dir
        Optional custom cache directory.
    """

    # Override locator classes to use only CuBIE locator
    _locator_classes = []

    def __init__(
        self,
        system_name: str,
        system_hash: str,
        compile_settings_hash: str,
        custom_cache_dir: Optional[Path] = None,
    ) -> None:
        # Create CUBIECacheLocator directly
        self._locator = CUBIECacheLocator(
            system_name,
            system_hash,
            compile_settings_hash,
            custom_cache_dir=custom_cache_dir,
        )
        disambiguator = self._locator.get_disambiguator()
        self._filename_base = f"{system_name}-{disambiguator}"

    @property
    def locator(self) -> CUBIECacheLocator:
        """Return the cache locator instance."""
        return self._locator

    @property
    def filename_base(self) -> str:
        """Return base filename for cache files."""
        system_name = self._locator._system_name
        disambiguator = self._locator.get_disambiguator()
        self._filename_base = f"{system_name}-{disambiguator}"
        return self._filename_base


class CUBIECache(CUDACache):
    """File-based cache for CuBIE compiled kernels.

    Coordinates loading and saving of cached kernels, incorporating
    ODE system hash and compile settings hash into cache keys.

    Parameters
    ----------
    system_name
        Name of the ODE system.
    system_hash
        Hash representing the ODE system definition.
    config_hash
        Pre-computed hash of the compile settings.
    max_entries
        Maximum number of cache entries before LRU eviction.
        Set to 0 to disable eviction. ``None`` reads the
        ``CUBIE_MAX_CACHE_ENTRIES`` environment default.
    mode
        Caching mode: 'hash' for content-addressed caching,
        'flush_on_change' to clear cache when settings change.
    custom_cache_dir
        Optional custom cache directory. Overrides default location.

    Notes
    -----
    Unlike the base Cache class, this does not use py_func for
    initialization. Instead, system info is passed directly.
    """

    _impl_class = CUBIECacheImpl

    def __init__(
        self,
        system_name: str,
        system_hash: str,
        config_hash: str,
        max_entries: Optional[int] = None,
        mode: str = "hash",
        custom_cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize CUBIECache with system and compile info.

        Note: Does not call inherited init using __super__(), absorbs the
        responsibilities directly due to  a different set of config parameters.
        """

        self._system_name = system_name
        self._system_hash = system_hash

        self._compile_settings_hash = config_hash

        self._name = f"CUBIECache({system_name})"
        if max_entries is None:
            max_entries = max_cache_entries_default()
        if custom_cache_dir is None:
            custom_cache_dir = kernel_cache_dir_default()
        self._max_entries = max_entries
        self._mode = mode

        self._impl = CUBIECacheImpl(
            system_name,
            system_hash,
            self._compile_settings_hash,
            custom_cache_dir=custom_cache_dir,
        )
        self._cache_path = self._impl.locator.get_cache_path()

        self._cache_file = IndexDataCacheFile(
            cache_path=self._cache_path,
            filename_base=self._impl.filename_base,
            source_stamp=self._impl.locator.get_source_stamp(),
        )
        cache_path = Path(self._cache_path)
        self._write_lock_path = cache_path.with_name(
            f"{cache_path.name}.lock"
        )
        self.enable()

    def _index_key(self, sig, codegen):
        """Return the cache key, including the cubie source hash."""
        return (
            sig,
            codegen.magic_tuple(),
            self._system_hash,
            self._compile_settings_hash,
            package_source_hash(),
        )

    def holds_kernel(self) -> bool:
        """Whether the index holds this system, config and source hash."""
        identity = (
            self._system_hash,
            self._compile_settings_hash,
            package_source_hash(),
        )
        with _CacheFileLock(self._write_lock_path):
            overloads = self._cache_file._load_index()
        return any(tuple(key[2:5]) == identity for key in overloads)

    def load_overload(self, sig, target_context):
        """Load cached kernel, starting compile timer on cache miss.

        Parameters
        ----------
        sig
            Function signature.
        target_context
            CUDA target context for kernel reconstruction.

        Returns
        -------
        _Kernel or None
            Reconstructed CUDA kernel if cache hit, None if miss.
        """
        with _CacheFileLock(self._write_lock_path):
            result = super().load_overload(sig, target_context)

        if result is not None:
            # Cache hit - notify via TimeLogger
            default_timelogger.print_message(
                f"Matching compiled function found at: "
                f"{self._cache_path}. Skipping compile!"
            )
        else:
            # Cache miss - start compile timing via TimeLogger
            default_timelogger.print_message(
                "No cached file found. Beginning compilation... "
                "This can take several minutes for larger (n>30) systems."
            )
            default_timelogger.start_event("compile_cuda_kernel")

        return result

    def enforce_cache_limit(self) -> None:
        """Evict oldest cache entries if count exceeds max_entries.

        Uses filesystem mtime for LRU ordering. Evicts .nbi/.nbc
        file pairs together.
        """
        if self._max_entries == 0:
            return  # Eviction disabled

        cache_path = Path(self._cache_path)
        if not cache_path.exists():
            return

        # Find all .nbi files (index files)
        nbi_files = list(cache_path.glob("*.nbi"))
        if len(nbi_files) <= self._max_entries:
            return

        # Sort by mtime (oldest first)
        nbi_files.sort(key=lambda f: f.stat().st_mtime)

        files_to_remove = len(nbi_files) - self._max_entries + 1
        for nbi_file in nbi_files[:files_to_remove]:
            base = nbi_file.stem
            # Remove .nbi file
            try:
                nbi_file.unlink()
            except OSError as e:
                # Log warning but continue - file may be locked or deleted
                warn(
                    f"Failed to remove cache file {nbi_file}: {e}",
                    RuntimeWarning,
                )
            # Remove associated .nbc files (may be multiple)
            for nbc_file in cache_path.glob(f"{base}.*.nbc"):
                try:
                    nbc_file.unlink()
                except OSError:
                    # Silently continue - .nbc cleanup is best-effort
                    pass

    def save_overload(self, sig, data):
        """Save kernel to cache, stopping compile timer.

        Parameters
        ----------
        sig
            Function signature.
        data
            Kernel data to cache.
        """
        # Stop compile timing - TimeLogger handles the message
        default_timelogger.stop_event("compile_cuda_kernel")

        # Retry the index write the backend's guard would swallow.
        with _CacheFileLock(self._write_lock_path):
            self.enforce_cache_limit()
            try:
                _retry_transient_io(lambda: self._save_overload(sig, data))
            except PermissionError as exc:
                warn(
                    f"Compiled kernel not written to {self._cache_path}: "
                    f"{exc}",
                    RuntimeWarning,
                )

    def flush_cache(self) -> None:
        """Delete all cache files in the cache directory.

        Removes all .nbi and .nbc files, then recreates an empty
        cache directory.
        """
        cache_path = Path(self._cache_path)
        with _CacheFileLock(self._write_lock_path):
            if cache_path.exists():
                try:
                    rmtree(cache_path)
                except OSError:
                    pass
            cache_path.mkdir(parents=True, exist_ok=True)

    @property
    def cache_path(self) -> Path:
        """Return the cache directory path."""
        return Path(self._cache_path)


