"""Keyword arguments for ``cuda.jit``.

Published Objects
-----------------
:data:`JIT_FLAG_DEFAULTS`
    Default jit flags except ``lineinfo``.
:data:`compile_kwargs`
    Kwargs for import-time ``@cuda.jit`` decorators.
:func:`get_jit_kwargs`
    Render a ``JITFlags`` to ``cuda.jit`` kwargs.
"""

from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

from cubie._env import lineinfo_default


# MLIR jit options carried by every compile.
_BACKEND_JIT_OPTIONS: Mapping[str, Any] = MappingProxyType(
    {"experimental_ast_transforms": True}
)

JIT_FLAG_DEFAULTS: Mapping[str, bool] = MappingProxyType(
    {
        "nsz": True,
        "contract": True,
        "arcp": True,
        "afn": True,
        "ftz": True,
        "lto": True,
    }
)
"""Default ``cuda.jit`` flags; ``lineinfo`` follows ``lineinfo_default``."""


def _render_jit_kwargs(lineinfo: bool) -> dict[str, Any]:
    """Return the default jit kwargs with ``lineinfo`` set."""
    return {
        "fastmath": {
            name for name, on in JIT_FLAG_DEFAULTS.items()
            if on and name != "lto"
        },
        "lineinfo": lineinfo,
        "lto": JIT_FLAG_DEFAULTS["lto"],
        **_BACKEND_JIT_OPTIONS,
    }


# Defaults for import-time device functions; factory builds use get_jit_kwargs.
compile_kwargs: Mapping[str, Any] = MappingProxyType(
    _render_jit_kwargs(lineinfo_default())
)


def get_jit_kwargs(
    jit_flags: Optional[Union[Any, bool]] = None,
) -> dict[str, Any]:
    """Return per-build ``cuda.jit`` keyword arguments.

    Parameters
    ----------
    jit_flags
        A ``JITFlags`` (any object with ``fastmath``, ``lineinfo`` and
        ``lto``), a bool as ``lineinfo`` over the defaults, or ``None``
        for the defaults.

    Returns
    -------
    dict
        ``fastmath``, ``lineinfo``, ``lto`` and
        ``experimental_ast_transforms`` keyword arguments.
    """
    if jit_flags is None:
        return _render_jit_kwargs(lineinfo_default())
    if isinstance(jit_flags, bool):
        return _render_jit_kwargs(jit_flags)
    return {
        "fastmath": jit_flags.fastmath,
        "lineinfo": jit_flags.lineinfo,
        "lto": jit_flags.lto,
        **_BACKEND_JIT_OPTIONS,
    }


__all__ = ["JIT_FLAG_DEFAULTS", "compile_kwargs", "get_jit_kwargs"]
