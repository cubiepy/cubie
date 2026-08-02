"""Numba CPU cache locator keyed on file content, not mtime or path."""
import hashlib
import os
from pathlib import Path

from numba.core import config
from numba.core.caching import UserProvidedCacheLocator

_ROOT = Path(__file__).resolve().parent.parent


class PortableCacheLocator(UserProvidedCacheLocator):
    """Cache under NUMBA_CACHE_DIR, keyed by repo path and content."""

    def __init__(self, py_func, py_file):
        super().__init__(py_func, py_file)
        source = Path(py_file).resolve()
        try:
            relative = source.relative_to(_ROOT)
        except ValueError:
            relative = Path(source.name)
        self._cache_path = os.path.join(
            config.CACHE_DIR, *relative.with_suffix("").parts
        )
        # Line endings differ between checkouts; the content does not.
        data = source.read_bytes().replace(b"\r\n", b"\n")
        self._stamp = hashlib.sha256(data).hexdigest()[:16]

    def get_source_stamp(self):
        return self._stamp
