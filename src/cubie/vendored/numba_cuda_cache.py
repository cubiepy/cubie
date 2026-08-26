"""Vendored Cache class from numba-cuda for CUDASIM compatibility.

Copyright (c) Anaconda, Inc. and NVIDIA CORPORATION; BSD 2-Clause.

Vendored from NVIDIA/numba-cuda on 2026-07-03
Source: numba_cuda/numba/cuda/core/caching.py (_Cache, Cache) and
numba_cuda/numba/cuda/dispatcher.py (CUDACache, including the
launch-config API added in PR #804).

The Cache class is vendored because CUDACache from numba.cuda.dispatcher
is not available in CUDASIM mode. The supporting classes (_CacheLocator,
CacheImpl, IndexDataCacheFile) import successfully in CUDASIM.
"""

import contextlib
import errno
import hashlib
import os
from abc import ABCMeta, abstractmethod

from numba.cuda.core.caching import IndexDataCacheFile
from numba.cuda.serialize import dumps
from numba.cuda import utils


class _Cache(metaclass=ABCMeta):
    """Abstract base class for caching compiled functions."""

    @property
    @abstractmethod
    def cache_path(self):
        """The base filesystem path of this cache."""

    @abstractmethod
    def load_overload(self, sig, target_context):
        """Load an overload for the given signature."""

    @abstractmethod
    def save_overload(self, sig, data):
        """Save the overload for the given signature."""

    @abstractmethod
    def enable(self):
        """Enable the cache."""

    @abstractmethod
    def disable(self):
        """Disable the cache."""

    @abstractmethod
    def flush(self):
        """Flush the cache."""


class Cache(_Cache):
    """A per-function compilation cache.

    The cache saves data in separate data files and maintains
    information in an index file.

    Note:
    This contains the driver logic only. The core logic is provided
    by a subclass of CacheImpl specified as _impl_class in the subclass.
    """

    # Must be overridden by subclass
    _impl_class = None

    def __init__(self, py_func):
        self._name = repr(py_func)
        self._py_func = py_func
        self._impl = self._impl_class(py_func)
        self._cache_path = self._impl.locator.get_cache_path()
        source_stamp = self._impl.locator.get_source_stamp()
        filename_base = self._impl.filename_base
        self._cache_file = IndexDataCacheFile(
            cache_path=self._cache_path,
            filename_base=filename_base,
            source_stamp=source_stamp,
        )
        self.enable()

    def __repr__(self):
        return "<%s py_func=%r>" % (self.__class__.__name__, self._name)

    @property
    def cache_path(self):
        return self._cache_path

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def flush(self):
        self._cache_file.flush()

    def load_overload(self, sig, target_context):
        """Load and recreate the cached object for the given signature."""
        target_context.refresh()
        with self._guard_against_spurious_io_errors():
            return self._load_overload(sig, target_context)

    def _load_overload(self, sig, target_context):
        if not self._enabled:
            return
        key = self._index_key(sig, target_context.codegen())
        data = self._cache_file.load(key)
        if data is not None:
            data = self._impl.rebuild(target_context, data)
        return data

    def save_overload(self, sig, data):
        """Save the data for the given signature in the cache."""
        with self._guard_against_spurious_io_errors():
            self._save_overload(sig, data)

    def _save_overload(self, sig, data):
        if not self._enabled:
            return
        if not self._impl.check_cachable(data):
            return
        self._impl.locator.ensure_cache_path()
        key = self._index_key(sig, data.codegen)
        data = self._impl.reduce(data)
        self._cache_file.save(key, data)

    @contextlib.contextmanager
    def _guard_against_spurious_io_errors(self):
        if os.name == "nt":
            try:
                yield
            except OSError as e:
                if e.errno != errno.EACCES:
                    raise
        else:
            yield

    def _index_key(self, sig, codegen):
        """Compute index key for the given signature and codegen."""
        codebytes = self._py_func.__code__.co_code
        if self._py_func.__closure__ is not None:
            cvars = tuple([x.cell_contents for x in self._py_func.__closure__])
            cvarbytes = dumps(cvars)
        else:
            cvarbytes = b""

        hasher = lambda x: hashlib.sha256(x).hexdigest()
        return (
            sig,
            codegen.magic_tuple(),
            (
                hasher(codebytes),
                hasher(cvarbytes),
            ),
        )


class CUDACache(Cache):
    """
    Implements a cache that saves and loads CUDA kernels and compile results.
    """

    def __init__(self, py_func):
        self._launch_config_key = None
        self._launch_config_sensitive_flag = None
        super().__init__(py_func)
        marker_name = f"{self._impl.filename_base}.lcs"
        self._launch_config_marker_path = os.path.join(
            self._cache_path, marker_name
        )

    def _index_key(self, sig, codegen):
        key = super()._index_key(sig, codegen)
        if self._launch_config_key is None:
            return key
        return key + (("launch_config", self._launch_config_key),)

    def set_launch_config_key(self, key):
        self._launch_config_key = key

    def is_launch_config_sensitive(self):
        if self._launch_config_sensitive_flag is None:
            self._launch_config_sensitive_flag = os.path.exists(
                self._launch_config_marker_path
            )
        return self._launch_config_sensitive_flag

    def mark_launch_config_sensitive(self):
        if self._launch_config_sensitive_flag is True:
            return True
        try:
            self._impl.locator.ensure_cache_path()
            with open(self._launch_config_marker_path, "a"):
                pass
        except OSError:
            self._launch_config_sensitive_flag = False
            return False
        self._launch_config_sensitive_flag = True
        return True

    def flush(self):
        super().flush()
        try:
            os.unlink(self._launch_config_marker_path)
        except FileNotFoundError:
            pass
        self._launch_config_sensitive_flag = None

    def load_overload(self, sig, target_context):
        # Loading an overload refreshes the context to ensure it is initialized.
        with utils.numba_target_override():
            return super().load_overload(sig, target_context)
