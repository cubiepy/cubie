"""Tests for the canonical serializer in cubie._serialize."""

import subprocess
import sys
from enum import Enum

import attrs
import numpy as np
import pytest

from cubie import _serialize
from cubie._serialize import (
    SCHEMA_VERSION,
    canonical_bytes,
    canonical_digest,
)


class _Colour(Enum):
    RED = 1
    BLUE = 2


@attrs.frozen
class _Snapshot:
    x: int
    fn: object = attrs.field(default=None, eq=False)


class _Protocol:
    def _cubie_canonical_(self):
        return ("protocol-value", 3)


# ── type and container discrimination ─────────────────────── #


def test_type_boundaries_distinguished():
    """Same-looking values of different types encode differently."""
    assert canonical_bytes(1) != canonical_bytes(1.0)
    assert canonical_bytes(1) != canonical_bytes(True)
    assert canonical_bytes(1) != canonical_bytes(np.int64(1))
    assert canonical_bytes(np.float32(1)) != canonical_bytes(
        np.float64(1)
    )
    assert canonical_bytes("1") != canonical_bytes(1)
    assert canonical_bytes(b"1") != canonical_bytes("1")


def test_container_boundaries_distinguished():
    """Adjacent-element ambiguity cannot produce equal encodings."""
    assert canonical_bytes(("ab", "c")) != canonical_bytes(("a", "bc"))
    assert canonical_bytes((1, (2, 3))) != canonical_bytes(((1, 2), 3))
    assert canonical_bytes(()) != canonical_bytes((None,))


def test_array_dtype_and_shape_distinguished():
    """Arrays encode dtype, shape, and element bytes."""
    flat = np.arange(6, dtype=np.float32)
    assert canonical_bytes(flat.reshape(2, 3)) != canonical_bytes(
        flat.reshape(3, 2)
    )
    assert canonical_bytes(flat) != canonical_bytes(
        flat.astype(np.float64)
    )
    fortran = np.asfortranarray(flat.reshape(2, 3))
    assert canonical_bytes(fortran) == canonical_bytes(
        np.ascontiguousarray(fortran)
    )


def test_signed_zero_and_nan_policy():
    """Signed zero is preserved; every NaN maps to one pattern."""
    assert canonical_bytes(0.0) != canonical_bytes(-0.0)
    quiet = float("nan")
    via_numpy = float(np.float64("nan"))
    assert canonical_bytes(quiet) == canonical_bytes(via_numpy)


def test_numpy_scalar_types_encode_by_dtype_name():
    """Precision classes (np.float32 vs np.float64) are distinct."""
    assert canonical_bytes(np.float32) != canonical_bytes(np.float64)
    assert canonical_digest(np.float32) == canonical_digest(np.float32)


def test_enum_members_distinguished():
    """Enum members encode by class identity and member name."""
    assert canonical_bytes(_Colour.RED) != canonical_bytes(_Colour.BLUE)


# ── attrs and protocol handling ───────────────────────────── #


def test_attrs_eq_false_fields_excluded():
    """eq=False fields never reach the encoding."""
    assert canonical_bytes(_Snapshot(1, fn=print)) == canonical_bytes(
        _Snapshot(1, fn=len)
    )
    assert canonical_bytes(_Snapshot(1)) != canonical_bytes(_Snapshot(2))


def test_protocol_objects_encode_via_canonical_method():
    """Objects join the domain through _cubie_canonical_()."""
    digest = canonical_digest(_Protocol())
    assert len(digest) == 64
    assert digest == canonical_digest(_Protocol())


# ── rejection: no fallback for unknown values ─────────────── #


@pytest.mark.parametrize(
    "value",
    [
        {"a": 1},
        [1, 2],
        {1, 2},
        print,
        object(),
    ],
    ids=["dict", "list", "set", "callable", "object"],
)
def test_unsupported_values_rejected(value):
    """There is deliberately no fallback encoding."""
    with pytest.raises(TypeError, match="canonical serialization"):
        canonical_bytes(value)


def test_nested_unsupported_values_rejected():
    """Rejection applies through container nesting."""
    with pytest.raises(TypeError):
        canonical_bytes((1, {"a": 1}))


# ── schema versioning and stability ───────────────────────── #


def test_digest_carries_schema_version():
    """Digests are prefixed with the serializer schema tag."""
    from hashlib import sha256

    payload = canonical_bytes((1, "x"))
    expected = sha256(SCHEMA_VERSION + payload).hexdigest()
    assert canonical_digest((1, "x")) == expected


# Load _serialize directly from file to avoid the full package import.
_SUBPROCESS_SNIPPET = """
import importlib.util
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("_serialize", sys.argv[1])
serialize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serialize)
value = (
    "stability-fixture",
    1,
    2.5,
    -0.0,
    np.float32(1.25),
    np.arange(4, dtype=np.float64),
    ("nested", (None, True)),
)
print(serialize.canonical_digest(value))
"""


def _digests_in_subprocesses(cwds):
    """Run the fixture digest in one fresh interpreter per cwd.

    The interpreters run concurrently, so the test costs one
    interpreter startup rather than one per root.
    """
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _SUBPROCESS_SNIPPET, _serialize.__file__],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
        )
        for cwd in cwds
    ]
    digests = []
    for process in processes:
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise AssertionError(
                "digest subprocess exited {0}: {1}".format(
                    process.returncode, stderr
                )
            )
        digests.append(stdout.strip())
    return digests


def test_digest_stable_across_processes_and_roots(tmp_path):
    """Digests agree across processes and working directories.

    Two fresh interpreters started from different absolute working
    directories produce the identical digest for the same value —
    no path, hash-seed, or process state enters the encoding.
    """
    root_a = tmp_path / "checkout_a"
    root_b = tmp_path / "deeper" / "checkout_b"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)
    digest_a, digest_b = _digests_in_subprocesses((root_a, root_b))
    assert digest_a == digest_b
    assert len(digest_a) == 64
