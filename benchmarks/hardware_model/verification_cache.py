"""Reuse identical successful proofs within an explicitly sealed epoch."""

import copy
import functools
import hashlib
import inspect
import io
import json
import pickle
import sys
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from pathlib import Path, PureWindowsPath


CURRENT = ContextVar("hardware_model_verification_cache", default=None)


def file_digest(path):
    """Hash the complete current bytes of one consumed file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_paths(value):
    """Admit primitive proof inputs and find their absolute path strings."""
    found, containers, strings = set(), set(), set()
    pending = [value]
    while pending:
        current = pending.pop()
        kind = type(current)
        if current is None or kind in (bool, int, float, Fraction):
            continue
        if kind is str:
            if current in strings:
                continue
            strings.add(current)
            possible = current.startswith(("/", "\\")) or (
                len(current) >= 3 and current[1] == ":"
                and current[2] in "/\\"
            )
            if possible and (Path(current).is_absolute()
                             or PureWindowsPath(current).is_absolute()):
                found.add(current)
            continue
        if kind not in (list, tuple, dict):
            raise TypeError(f"Unsupported proof input type: {kind.__name__}")
        identity = id(current)
        if identity in containers:
            continue
        containers.add(identity)
        if kind is dict:
            pending.extend(current.keys())
            pending.extend(current.values())
        else:
            pending.extend(current)
    return found


class PrimitivePickler(pickle.Pickler):
    """Serialize builtin values without invoking arbitrary object reducers."""

    def reducer_override(self, value):
        """Permit only the standard exact Fraction reduction."""
        if type(value) is Fraction:
            return Fraction, (value.numerator, value.denominator)
        if value is Fraction:
            return NotImplemented
        raise TypeError(f"Unsupported proof object: {type(value).__name__}")


def input_digest(value):
    """Hash complete type-preserving inputs using a restricted C serializer."""
    stream = io.BytesIO()
    PrimitivePickler(stream, protocol=5).dump(value)
    return hashlib.sha256(stream.getbuffer()).hexdigest()


class ProofCache:
    """Hold copied validation receipts for exact proof inputs only."""

    def __init__(self, manifest_path, expected_sha256):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest_sha256 = expected_sha256
        if file_digest(self.manifest_path) != expected_sha256:
            raise ValueError("Verification epoch manifest bytes differ")
        manifest = json.loads(self.manifest_path.read_text())
        self.files = {}
        for relative, digest in manifest["files"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Sealed epoch file escapes its source root")
            self.files[str(path)] = digest
        if not self.files:
            raise ValueError("Verification epoch has no sealed files")
        self.runtime = dict(
            python=sys.version,
            implementation=sys.implementation.name,
            cache_tag=sys.implementation.cache_tag,
            serialization="pickle protocol 5 over admitted primitive inputs",
        )
        self.entries = {}
        self.path_inventories = {}
        self.counts = Counter()
        self.proofs = {}
        self.external_files = {}
        self.guard()

    def guard(self):
        """Rehash the manifest and every sealed file before proof reuse."""
        if file_digest(self.manifest_path) != self.manifest_sha256:
            raise ValueError("Verification epoch manifest changed")
        for path, expected in self.files.items():
            if file_digest(path) != expected:
                raise ValueError(f"Sealed verification source changed: {path}")
        for name, module in tuple(sys.modules.items()):
            if not (name == "cubie" or name.startswith("cubie.")
                    or name == "benchmarks"
                    or name.startswith("benchmarks.")):
                continue
            filename = getattr(module, "__file__", None)
            if filename is None:
                continue
            path = Path(filename).resolve()
            if path.suffix == ".py" and str(path) not in self.files:
                raise ValueError(f"Model imports cross source epochs: {path}")

    def dependencies(self, paths):
        """Read all external file references without assuming key spellings."""
        records = {}
        for spelling in sorted(paths):
            path = Path(spelling).resolve()
            key = str(path)
            if key in self.files:
                continue
            if path.is_file():
                records[key] = file_digest(path)
            else:
                records[key] = "directory" if path.is_dir() else "missing"
        self.external_files.update(records)
        return records

    def verify(self, function, source_path, import_sha256, args, kwargs):
        """Run or reuse a successful, unchanged-input verification call."""
        self.guard()
        if self.files.get(source_path) != import_sha256:
            raise ValueError("Verifier import bytes differ from sealed epoch")
        inputs = (args, kwargs)
        arguments_sha256 = input_digest(inputs)
        if arguments_sha256 not in self.path_inventories:
            self.path_inventories[arguments_sha256] = input_paths(inputs)
        paths = self.path_inventories[arguments_sha256]
        dependencies = self.dependencies(paths)
        identity = dict(
            function=function.__module__ + "." + function.__qualname__,
            verifier_source_sha256=import_sha256,
            epoch_manifest_sha256=self.manifest_sha256,
            arguments_sha256=arguments_sha256,
            external_files=dependencies, runtime=self.runtime,
        )
        key = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if key in self.entries:
            self.counts[identity["function"] + ":hit"] += 1
            return copy.deepcopy(self.entries[key])
        self.counts[identity["function"] + ":miss"] += 1
        try:
            result = function(*args, **kwargs)
            self.guard()
            if input_digest(inputs) != arguments_sha256:
                raise ValueError("Verification mutated its proof inputs")
            if self.dependencies(paths) != dependencies:
                raise ValueError("External proof files changed during replay")
            input_paths(result)
        except BaseException:
            self.counts[identity["function"] + ":failed"] += 1
            raise
        self.entries[key] = copy.deepcopy(result)
        self.proofs[key] = identity
        return result

    def receipt(self):
        """Describe proof identities and reuse without changing model output."""
        return dict(
            kind="exact_process_local_verification_cache",
            manifest_path=str(self.manifest_path),
            manifest_sha256=self.manifest_sha256,
            sealed_files=len(self.files), runtime=self.runtime,
            counts=dict(sorted(self.counts.items())),
            exact_argument_path_inventories=len(self.path_inventories),
            successful_proofs=self.proofs,
            external_files=self.external_files,
            allocations_cached=False, schedules_cached=False,
            scope="Pure successful policy graph and policy plan proofs only",
        )


def cached_verifier(function):
    """Make successful proof reuse conditional on an explicit epoch context."""
    source_path = str(Path(inspect.getsourcefile(function)).resolve())
    import_sha256 = file_digest(source_path)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        cache = CURRENT.get()
        if cache is None:
            return function(*args, **kwargs)
        return cache.verify(function, source_path, import_sha256, args, kwargs)

    return wrapped


@contextmanager
def proof_epoch(manifest_path, expected_sha256):
    """Enable verification reuse within one checked immutable source epoch."""
    if CURRENT.get() is not None:
        raise ValueError("Nested verification epochs are not supported")
    cache = ProofCache(manifest_path, expected_sha256)
    token = CURRENT.set(cache)
    try:
        yield cache
    finally:
        CURRENT.reset(token)
        cache.guard()


@contextmanager
def uncached_proofs():
    """Replay ordinary verification for an exact comparison with a cache hit."""
    token = CURRENT.set(None)
    try:
        yield
    finally:
        CURRENT.reset(token)
