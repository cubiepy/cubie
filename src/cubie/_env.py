"""Central registry for ``CUBIE_*`` environment variable overrides.

Environment variables provide process-wide defaults for behaviour that can
also be controlled per-solver through arguments. Explicit arguments always
take precedence over environment values; environment values take precedence
over built-in defaults.

Values are read lazily at the point of use (factory construction or module
import), so setting a variable after the relevant object has been created
has no effect. This is deliberate: compiled device functions are cached
against their compile settings, and re-reading the environment mid-session
would bypass that bookkeeping.

Published Functions
-------------------
:func:`env_bool`
    Parse a boolean-valued environment variable.
:func:`lineinfo_default`
    Default for the ``lineinfo`` compile setting (``CUBIE_LINEINFO``).
:func:`cache_dir_default`
    Default for the on-disk cache root (``CUBIE_CACHE_DIR``).
:func:`kernel_cache_dir_default`
    Default directory for compiled-kernel caches
    (``CUBIE_KERNEL_CACHE_DIR``).
:func:`max_cache_entries_default`
    Default kernel-cache LRU limit (``CUBIE_MAX_CACHE_ENTRIES``).
:func:`cuda_backend_requested`
    Explicitly requested CUDA backend (``CUBIE_CUDA_BACKEND``).

Recognised Variables
--------------------
``CUBIE_LINEINFO``
    Compile all device functions and kernels with source-line correlation
    data (``-lineinfo``) for profilers such as Nsight Compute. Truthy
    values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive). Default
    off.
``CUBIE_CACHE_DIR``
    Root directory for all on-disk caches (generated source, CellML
    parse results, compiled kernels). Overridden by an explicit
    :func:`cubie.cache_root.set_cache_root` call; defaults to
    ``<current working directory>/generated`` when unset.
``CUBIE_KERNEL_CACHE_DIR``
    Directory for compiled-kernel caches only, leaving generated source
    and parse caches at the shared root. Overridden by an explicit
    ``cache_dir`` argument. CI sets this to relocate kernel caches into
    a portable artifact.
``CUBIE_MAX_CACHE_ENTRIES``
    Per-system LRU limit for compiled-kernel cache entries; zero
    disables eviction. Overridden by an explicit ``max_cache_entries``
    argument. Default 0.
``CUBIE_CUDA_BACKEND``
    Explicit CUDA backend selection, ``numba-cuda`` or ``mlir``.
    Read by :mod:`cubie.cuda_backend` at import. When unset, the
    installed backend is used; when both backends are installed,
    the MLIR backend is auto-selected.
``CUBIE_OPERATION_ORDERING``
    Default codegen operation-ordering policy (``liveness_auto``
    when unset). Read once at ``import cubie``; explicit
    ``operation_ordering`` arguments always win.
``CUBIE_BLOCK_SCHEDULE``
    Ordering policy for cubie's typed-IR block scheduler on the MLIR
    backend (``anchor_dfs`` default; ``source``/``off`` skip
    registration). Read once at ``import cubie``; the active policy
    folds into the compiled-kernel cache fingerprint.
``CUBIE_BLOCK_SCHEDULE_DUMP`` / ``CUBIE_BLOCK_SCHEDULE_ORDER``
    Scheduler diagnostics: gzip graph-dump path and JSON
    order-injection path (see
    :mod:`cubie.backend._typed_block_scheduler`).
"""

import os
from typing import Optional

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value returned when the variable is unset.

    Returns
    -------
    bool
        Parsed value.

    Raises
    ------
    ValueError
        If the variable is set to an unrecognised value.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"Environment variable {name}={raw!r} is not a recognised boolean; "
        f"use one of {sorted(_TRUTHY | _FALSY - {''})}."
    )


def lineinfo_default() -> bool:
    """Return the default ``lineinfo`` compile setting.

    Reads ``CUBIE_LINEINFO`` from the environment; explicit
    ``Solver(lineinfo=...)`` arguments override this value.
    """
    return env_bool("CUBIE_LINEINFO", False)


def cache_dir_default() -> Optional[str]:
    """Return the environment-supplied on-disk cache root, if any.

    Reads ``CUBIE_CACHE_DIR`` from the environment; an explicit
    :func:`cubie.cache_root.set_cache_root` call overrides this value.
    Empty and whitespace-only values are treated as unset.

    Returns
    -------
    Optional[str]
        The configured directory, or ``None`` when the variable is
        unset or blank.
    """
    raw = os.environ.get("CUBIE_CACHE_DIR")
    if raw is None or not raw.strip():
        return None
    return raw


def kernel_cache_dir_default() -> Optional[str]:
    """Return ``CUBIE_KERNEL_CACHE_DIR``, or ``None`` when unset."""
    raw = os.environ.get("CUBIE_KERNEL_CACHE_DIR")
    if raw is None or not raw.strip():
        return None
    return raw


def max_cache_entries_default() -> int:
    """Return the non-negative kernel-cache limit; zero disables eviction."""
    raw = os.environ.get("CUBIE_MAX_CACHE_ENTRIES")
    if raw is None or not raw.strip():
        return 0
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"CUBIE_MAX_CACHE_ENTRIES={raw!r} must be a non-negative integer."
        ) from None
    if value < 0:
        raise ValueError(
            f"CUBIE_MAX_CACHE_ENTRIES={raw!r} must be non-negative."
        )
    return value


def operation_ordering_default() -> str:
    """Return ``CUBIE_OPERATION_ORDERING``, ``liveness_auto`` unset."""
    raw = os.environ.get("CUBIE_OPERATION_ORDERING")
    if raw is None or not raw.strip():
        return "liveness_auto"
    return raw.strip().lower()


_active_block_schedule = "source"


def block_schedule_default() -> str:
    """Return ``CUBIE_BLOCK_SCHEDULE`` (default ``anchor_dfs``;
    ``off``/``none`` normalise to ``source``)."""
    raw = os.environ.get("CUBIE_BLOCK_SCHEDULE")
    if raw is None or not raw.strip():
        return "anchor_dfs"
    value = raw.strip().lower()
    if value in ("off", "none", "0"):
        return "source"
    return value


def set_active_block_schedule(policy: str) -> None:
    """Record the registered scheduler policy for the cache
    fingerprint."""
    global _active_block_schedule
    _active_block_schedule = policy


def active_block_schedule() -> str:
    """Return the active scheduler policy (``source`` = none)."""
    return _active_block_schedule


def cuda_backend_requested() -> Optional[str]:
    """Return the explicitly requested CUDA backend, if any.

    Reads ``CUBIE_CUDA_BACKEND`` from the environment; empty and
    whitespace-only values are treated as unset.
    :mod:`cubie.cuda_backend` resolves the active backend from this
    value and the installed packages.

    Returns
    -------
    Optional[str]
        ``"numba-cuda"`` or ``"mlir"``, or ``None`` when unset.

    Raises
    ------
    ValueError
        If the variable is set to an unrecognised value.
    """
    raw = os.environ.get("CUBIE_CUDA_BACKEND")
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value not in ("numba-cuda", "mlir"):
        raise ValueError(
            f"CUBIE_CUDA_BACKEND={raw!r} is not recognised; valid "
            "values are 'numba-cuda' and 'mlir'."
        )
    return value
