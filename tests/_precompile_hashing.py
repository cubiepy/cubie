"""Process-stable identity hashing for factory-less test kernels.

Production dispatchers keep the ``CUBIECache`` their owning factory
attaches, keyed by the production system and configuration identity.
Kernels defined inline in test files have no owning factory, so the
precompile plugin keys those — and only those — by function identity:
code object, closure values, defaults, and target options. The helpers
here strip process- and checkout-specific detail (file paths, line
offsets, serialization order) so a kernel compiled on a CPU runner
produces the same key when the GPU runner looks it up.

Kept out of :mod:`cubie.cubie_cache`: production caches are keyed by
system and compile-settings hashes and never need function identity.
"""
from hashlib import sha256
from types import CodeType, FunctionType
from typing import Optional

from numba_cuda_mlir.numba_cuda.serialize import dumps as cache_dumps


def _canonical_const(value):
    """Return a value-only stand-in for a code constant.

    The compiler folds ``in {...}`` and ``in (...)`` membership tests
    into frozenset constants whose iteration order varies with each
    process's hash seed, so frozensets become marker tuples sorted by
    member repr. Nested code objects become their fingerprints.
    """
    if isinstance(value, CodeType):
        return ("__code__", _code_fingerprint(value))
    if isinstance(value, frozenset):
        members = sorted(
            (_canonical_const(member) for member in value), key=repr
        )
        return ("__frozenset__",) + tuple(members)
    if isinstance(value, tuple):
        return tuple(_canonical_const(member) for member in value)
    return value


def _code_fingerprint(code: CodeType) -> str:
    """Hash a code object by value without marshal or pickle.

    ``marshal.dumps`` writes a ``FLAG_REF`` bit per object based on
    its runtime refcount at dump time, so two dumps of the same code
    object can differ depending on what else references its interned
    name strings (deterministic only on Python >= 3.12, where
    identifier strings are immortal). Hashing the repr of a tuple of
    the code object's value components has no such context
    sensitivity. Source locations (filename, line numbers) are
    excluded so checkouts at different paths agree.
    """
    structure = (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        tuple(_canonical_const(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_name,
        getattr(code, "co_exceptiontable", b""),
    )
    return sha256(repr(structure).encode("utf-8")).hexdigest()


def _stable_value_key(value, active: set[int]):
    """Return a process-stable key for a captured value."""
    py_func = getattr(value, "py_func", None)
    if py_func is not None:
        return (
            "dispatcher",
            _function_key(py_func, active),
            _stable_value_key(value.targetoptions, active),
        )
    if isinstance(value, FunctionType):
        # The serialized fallback pickles plain functions by value,
        # embedding co_filename, which differs between the CPU
        # population checkout and the GPU consumer checkout.
        return ("function", _function_key(value, active))
    if isinstance(value, CodeType):
        return ("code", _code_fingerprint(value))
    if value.__class__.__name__ == "FastMathOptions":
        return ("fastmath", tuple(sorted(value.flags)))
    if isinstance(value, tuple):
        items = tuple(_stable_value_key(item, active) for item in value)
        return ("tuple", items)
    if isinstance(value, list):
        items = tuple(_stable_value_key(item, active) for item in value)
        return ("list", items)
    if isinstance(value, dict):
        items = (
            (_stable_value_key(key, active), _stable_value_key(item, active))
            for key, item in value.items()
        )
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(value, (set, frozenset)):
        items = (_stable_value_key(item, active) for item in value)
        return ("set", tuple(sorted(items, key=repr)))
    return ("serialized", sha256(cache_dumps(value)).hexdigest())


def _stable_value_hash(value) -> str:
    """Hash a value without process-specific serialization order."""
    key = _stable_value_key(value, set())
    return sha256(cache_dumps(key)).hexdigest()


def _function_key(py_func, active: Optional[set[int]] = None):
    """Identify function code and compile-time captures."""
    if active is None:
        active = set()
    identity = id(py_func)
    if identity in active:
        return ("recursive", py_func.__module__, py_func.__qualname__)
    active.add(identity)
    try:
        closure = py_func.__closure__ or ()
        closure_key = tuple(
            _stable_value_key(cell.cell_contents, active) for cell in closure
        )
        defaults_key = _stable_value_key(
            (py_func.__defaults__, py_func.__kwdefaults__), active
        )
        return (
            py_func.__module__,
            py_func.__qualname__,
            sha256(cache_dumps(closure_key)).hexdigest(),
            _code_fingerprint(py_func.__code__),
            sha256(cache_dumps(defaults_key)).hexdigest(),
        )
    finally:
        active.remove(identity)
