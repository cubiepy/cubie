"""Versioned typed canonical serialization for cache identities.

Every semantic identity CuBIE persists or compares across processes —
compile-settings hashes, generated-source identities, bound-helper
identities, and kernel-cache key components — is derived from the byte
encoding defined here. The encoding is total over the value domain
admitted by compile-setting converters and validators, and it raises
:class:`TypeError` for anything else: there is deliberately no fallback
representation, so an unsupported value fails at the point a setting is
written rather than silently entering a cache key through ``str()``.

Encoding rules
--------------
Each value is encoded as a single tag byte followed by a fixed- or
length-prefixed payload. Container and type boundaries are explicit, so
``("ab", "c")`` and ``("a", "bc")`` produce different bytes, as do ``1``,
``1.0``, ``True``, and ``numpy.int64(1)``. Floats are encoded by IEEE-754 bit
pattern with signed zero preserved; every NaN is canonicalized to the single
quiet-NaN pattern. Arrays include dtype, shape, and C-order little-endian
element bytes. Tuple subclasses (namedtuples included) encode as plain tuples —
element values only, no class identity. Attrs instances encode their class
identity and their ``eq``-participating fields in declared order. Value objects
outside these built-in forms participate through the :data:`CANONICAL_PROTOCOL`
method, which must return a value that is itself canonically encodable.

Digests are prefixed with :data:`SCHEMA_VERSION` so a change to this
representation intentionally invalidates previously stored artifacts.
"""

import struct
from enum import Enum
from hashlib import sha256
from typing import Any

from attrs import fields as attrs_fields, has as attrs_has
from numpy import (
    ascontiguousarray,
    dtype as np_dtype,
    generic as np_generic,
    ndarray,
)

SCHEMA_VERSION = b"cubie-canonical-v1"
"""Serializer schema tag folded into every digest."""

CANONICAL_PROTOCOL = "_cubie_canonical_"
"""Method name a value object implements to join the canonical domain.

The method takes no arguments and returns a canonically encodable
value (typically a tuple of primitives) that captures the object's
semantic identity.
"""

_TAG_NONE = b"\x00"
_TAG_FALSE = b"\x01"
_TAG_TRUE = b"\x02"
_TAG_INT = b"\x03"
_TAG_FLOAT = b"\x04"
_TAG_BYTES = b"\x05"
_TAG_STR = b"\x06"
_TAG_NP_SCALAR = b"\x07"
_TAG_NDARRAY = b"\x08"
_TAG_TUPLE = b"\x09"
_TAG_ATTRS = b"\x0a"
_TAG_ENUM = b"\x0b"
_TAG_DTYPE = b"\x0c"
_TAG_SCALAR_TYPE = b"\x0d"
_TAG_PROTOCOL = b"\x0e"

_CANONICAL_NAN = struct.pack(">Q", 0x7FF8000000000000)


def _length_prefixed(payload: bytes) -> bytes:
    """Return an 8-byte big-endian length prefix followed by payload."""
    return struct.pack(">Q", len(payload)) + payload


def _encode_float_bits(value: float) -> bytes:
    """Encode a float by IEEE-754 bits with a canonical NaN pattern."""
    if value != value:
        return _CANONICAL_NAN
    return struct.pack(">d", value)


def _class_identity(cls: type) -> bytes:
    """Return the stable module-qualified name of a class."""
    name = f"{cls.__module__}.{cls.__qualname__}"
    return _length_prefixed(name.encode("utf-8"))


def _encode_ndarray(value: ndarray) -> bytes:
    """Encode dtype name, shape, and C-order little-endian bytes."""
    dtype = value.dtype
    little = dtype.newbyteorder("<")
    contiguous = ascontiguousarray(value).astype(little, copy=False)
    parts = [
        _length_prefixed(dtype.name.encode("utf-8")),
        struct.pack(">Q", value.ndim),
    ]
    for dim in value.shape:
        parts.append(struct.pack(">Q", dim))
    parts.append(_length_prefixed(contiguous.tobytes(order="C")))
    return b"".join(parts)


def canonical_bytes(value: Any) -> bytes:
    """Encode ``value`` into its canonical byte representation.

    Parameters
    ----------
    value
        A value from the canonical domain: ``None``, ``bool``, ``int``,
        ``float``, ``bytes``, ``str``, NumPy scalars and arrays, NumPy
        dtypes and scalar types, tuples of canonical values, attrs
        instances, :class:`~enum.Enum` members, or objects implementing
        the :data:`CANONICAL_PROTOCOL` method.

    Returns
    -------
    bytes
        Tagged, length-prefixed canonical encoding of ``value``.

    Raises
    ------
    TypeError
        If ``value`` (or any nested value) is outside the canonical
        domain. Converters and validators on compile settings are
        responsible for preventing such values from being stored.
    """
    if value is None:
        return _TAG_NONE
    if value is True:
        return _TAG_TRUE
    if value is False:
        return _TAG_FALSE
    value_type = type(value)
    if value_type is int:
        length = (value.bit_length() + 8) // 8 or 1
        payload = value.to_bytes(length, "big", signed=True)
        return _TAG_INT + _length_prefixed(payload)
    if value_type is float:
        return _TAG_FLOAT + _encode_float_bits(value)
    if value_type is bytes:
        return _TAG_BYTES + _length_prefixed(value)
    if value_type is str:
        return _TAG_STR + _length_prefixed(value.encode("utf-8"))
    if isinstance(value, np_generic):
        dtype = value.dtype
        little = dtype.newbyteorder("<")
        payload = (
            _length_prefixed(dtype.name.encode("utf-8"))
            + _length_prefixed(value.astype(little).tobytes())
        )
        return _TAG_NP_SCALAR + payload
    if isinstance(value, ndarray):
        return _TAG_NDARRAY + _encode_ndarray(value)
    if value_type is tuple or (
        isinstance(value, tuple) and hasattr(value, "_fields")
    ):
        parts = [struct.pack(">Q", len(value))]
        for item in value:
            parts.append(canonical_bytes(item))
        return _TAG_TUPLE + b"".join(parts)
    if isinstance(value, Enum):
        return (
            _TAG_ENUM
            + _class_identity(value_type)
            + _length_prefixed(value.name.encode("utf-8"))
        )
    if isinstance(value, np_dtype):
        return _TAG_DTYPE + _length_prefixed(value.name.encode("utf-8"))
    if isinstance(value, type) and issubclass(value, np_generic):
        name = np_dtype(value).name
        return _TAG_SCALAR_TYPE + _length_prefixed(name.encode("utf-8"))
    protocol = getattr(value, CANONICAL_PROTOCOL, None)
    if protocol is not None and callable(protocol):
        return (
            _TAG_PROTOCOL
            + _class_identity(value_type)
            + canonical_bytes(protocol())
        )
    if attrs_has(value_type):
        parts = [_class_identity(value_type)]
        encoded_fields = []
        for fld in attrs_fields(value_type):
            if fld.eq is False:
                continue
            encoded_fields.append(
                _length_prefixed(fld.name.encode("utf-8"))
                + canonical_bytes(getattr(value, fld.name))
            )
        parts.append(struct.pack(">Q", len(encoded_fields)))
        parts.extend(encoded_fields)
        return _TAG_ATTRS + b"".join(parts)
    raise TypeError(
        f"{value_type.__module__}.{value_type.__qualname__} is outside "
        "the canonical serialization domain. Compile-setting converters "
        "must normalize values to supported types, or the class must "
        f"implement {CANONICAL_PROTOCOL}()."
    )


def canonical_digest(value: Any) -> str:
    """Return the schema-versioned SHA-256 hex digest of ``value``.

    Parameters
    ----------
    value
        A value from the canonical domain (see :func:`canonical_bytes`).

    Returns
    -------
    str
        64-character SHA-256 hex digest over the schema tag and the
        canonical encoding of ``value``.
    """
    digest = sha256(SCHEMA_VERSION)
    digest.update(canonical_bytes(value))
    return digest.hexdigest()
