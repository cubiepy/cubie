"""Single source of truth for cubie's on-disk cache root directory.

Three disk cache layers persist artefacts between sessions: generated
CUDA source modules (:class:`~cubie.odesystems.symbolic.odefile.ODEFile`),
pickled BigModel parse results
(:class:`~cubie.odesystems.symbolic.parsing.bigmodel_cache.BigModelCache`),
and compiled kernels (:mod:`cubie.cubie_cache`). All three resolve
their base directory through :func:`get_cache_root`, so redirecting
the root (for example in tests) relocates every layer together.

The root resolves, in order of precedence: the process-wide override
installed through :func:`set_cache_root`, the ``CUBIE_CACHE_DIR``
environment variable, then ``<current working directory>/generated``.

Published Functions
-------------------
:func:`get_cache_root`
    Return the active cache root directory.
:func:`set_cache_root`
    Override the cache root process-wide, or restore the default.
:func:`get_cache_root_override`
    Return the active process-wide override, if any.
"""

from os import getcwd
from pathlib import Path
from typing import Optional, Union

from cubie._env import cache_dir_default

_cache_root_override: Optional[Path] = None


def get_cache_root() -> Path:
    """Return the root directory for cubie's on-disk caches.

    Returns
    -------
    Path
        The override set through :func:`set_cache_root` when one is
        active, otherwise the ``CUBIE_CACHE_DIR`` environment variable
        when set, otherwise ``<current working directory>/generated``.
        Environment and working directory are read at call time.
    """
    if _cache_root_override is not None:
        return _cache_root_override
    env_dir = cache_dir_default()
    if env_dir is not None:
        return Path(env_dir)
    return Path(getcwd()) / "generated"


def get_cache_root_override() -> Optional[Path]:
    """Return the process-wide cache root override.

    Returns
    -------
    Optional[Path]
        The override installed through :func:`set_cache_root`, or
        ``None`` when no override is active. Save the returned value
        before installing a temporary override and pass it back to
        :func:`set_cache_root` to restore the prior state.
    """
    return _cache_root_override


def set_cache_root(path: Optional[Union[str, Path]]) -> None:
    """Override the cache root for every disk cache layer.

    Parameters
    ----------
    path
        New root directory for generated source, BigModel parse, and
        compiled-kernel caches. Pass ``None`` to restore the default
        ``<cwd>/generated`` behaviour.
    """
    global _cache_root_override
    _cache_root_override = None if path is None else Path(path)
