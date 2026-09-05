"""Emit and inspect source-to-SASS operation translation experiments.

``emit`` and ``analyze`` use only the Python standard library. ``compile``
starts an explicit worker importing the installed CUDA backend. The worker
compiles kernels but never allocates device arrays or launches a kernel.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
SCHEMA = 1
PROFILES = {
    "default": {},
    "no_contract": {"contract": False},
    "no_arcp": {"arcp": False},
    "no_afn": {"afn": False},
    "strict": {
        "contract": False,
        "arcp": False,
        "afn": False,
        "ftz": False,
        "nsz": False,
    },
}
CATEGORIES = {
    "add": ("x = x + y",),
    "multiply": ("x = x * y",),
    "fma": ("x = x * y + z",),
    "fma_multi_use": (
        "product = x * y",
        "x = product + z",
        "w = w + product",
    ),
    "divide": ("x = x / y",),
    "sqrt": ("x = float32(math.sqrt(x))",),
    "exp": ("x = float32(math.exp(x))",),
    "log": ("x = float32(math.log(x))",),
    "select": ("x = selp(x > z, x + y, x + w)",),
    "index_i32": ("i = int32(i * stride + offset)",),
    "index_u32": ("i = uint32(i * stride + offset)",),
    "index_i64": ("i = int64(i * stride + offset)",),
}
CONDITIONS = {
    "fma": "Multiply result has one add consumer; contraction is allowed.",
    "fma_multi_use": (
        "Product feeds two additions and both recurrences are observable. "
        "The compiler can retain multiplication, duplicate it, or contract."
    ),
    "select": (
        "Eager selp with one comparison and two additions per fragment; "
        "both arms depend on the changing value. Not an isolated select."
    ),
    "divide": "Runtime numerator and denominator; arcp affects lowering.",
    "sqrt": "Runtime input without range proof; exceptional paths unknown.",
    "exp": "Runtime input without range proof; afn affects lowering.",
    "log": "Runtime input without range proof; afn affects lowering.",
    "index_i32": "Signed 32-bit affine recurrence; explicit narrowing.",
    "index_u32": "Unsigned 32-bit affine recurrence; explicit narrowing.",
    "index_i64": "Signed 64-bit integer affine recurrence, not FP64.",
}
SECTION = re.compile(r"^\s*//-+\s*\.text\.([^\s]+)")
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;")
LABEL = re.compile(r"^\s*(\.L[\w.$]+):")
TARGET = re.compile(r"(\.L[\w.$]+)")
CONTROL = {"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "BSSY", "BSYNC"}
MEMORY = {
    "LD",
    "LDG",
    "LDL",
    "LDS",
    "LDC",
    "ULDC",
    "LDGSTS",
    "LDSM",
    "ST",
    "STG",
    "STL",
    "STS",
    "STSM",
    "ATOM",
    "ATOMG",
    "ATOMS",
    "RED",
    "SUATOM",
    "SULD",
    "SUST",
    "SURED",
}


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compilation_identity(manifest: dict) -> dict:
    """Select stable compiler, source, and device provenance fields."""
    return {
        key: manifest[key]
        for key in (
            "python",
            "platform",
            "backend",
            "versions",
            "git_head",
            "git_status",
            "source_sha256",
            "device_name",
            "compute_capability",
            "device_attributes",
            "nvdisasm",
            "translation_provenance",
        )
    } | {
        "devices": [
            {
                key: device[key]
                for key in ("uuid", "name", "pci.bus_id", "driver_version")
            }
            for device in manifest["clocks"]["devices"]
        ]
    }


def _tokens(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True)
    return {
        "command": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def source_counts(source: str) -> dict:
    """Count source syntax without equating it to native instructions."""
    counts = Counter()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp):
            counts[type(node.op).__name__] += 1
        elif isinstance(node, ast.Compare):
            counts.update(type(operation).__name__ for operation in node.ops)
        elif isinstance(node, ast.Call):
            counts["call:" + ast.unparse(node.func)] += 1
        elif isinstance(node, ast.Subscript):
            counts[
                "array_store"
                if isinstance(node.ctx, ast.Store)
                else "array_load"
            ] += 1
    return dict(sorted(counts.items()))


def make_source(category: str, count: int) -> str:
    """Return a float32 or integer-index compilation fragment kernel."""
    lines = [
        "import math",
        "from numpy import int64, uint32",
        "from cubie.cuda_simsafe import (",
        "    cuda, float32, int32, selp,",
        ")",
        "",
        "",
        "def probe(output, x, y, z, w, i, stride, offset):",
    ]
    for _ in range(count):
        lines.extend("    " + line for line in CATEGORIES[category])
    result = "i" if category.startswith("index_") else "x"
    if category == "fma_multi_use":
        result = "x + w"
    lines.append("    output[cuda.grid(1)] = " + result)
    source = "\n".join(lines) + "\n"
    compile(source, "<operation translation source>", "exec")
    return source


def emit(output: Path, categories: list, counts: list, profiles: list):
    """Write inspectable source without importing CUDA or executing it."""
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for category in categories:
        for profile in profiles:
            for count in counts:
                name = f"{category}_{profile}_n{count}"
                directory = output / name
                directory.mkdir()
                path = directory / "kernel.py"
                source = make_source(category, count)
                path.write_text(source, encoding="utf-8")
                case = {
                    "case": name,
                    "category": category,
                    "profile": profile,
                    "count": count,
                    "source": str(path.resolve()),
                    "source_sha256": _hash(path),
                    "flag_overrides": PROFILES[profile],
                    "fragment": list(CATEGORIES[category]),
                    "fragment_source_counts": source_counts(
                        "\n".join(CATEGORIES[category])
                    ),
                    "whole_source_counts": source_counts(source),
                    "dag_condition": CONDITIONS.get(
                        category, "Runtime dependent recurrence."
                    ),
                    "status": "source_only",
                }
                _write(directory / "case.json", case)
                cases.append(case)
    manifest = {
        "schema": SCHEMA,
        "mode": "source_only",
        "python": sys.version,
        "platform": platform.platform(),
        "generator": str(Path(__file__)),
        "generator_sha256": _hash(Path(__file__)),
        "git_head": _command(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "cases": cases,
        "semantics": {
            "inputs": "Signature values are runtime kernel parameters.",
            "precision": "float32 arithmetic; index cases use integer types.",
            "execution": "Sources are not numerical benchmark workloads.",
            "count": "Number of source DAG fragments, not instruction count.",
            "timing": "No kernel execution or timing is implemented.",
        },
    }
    _write(output / "manifest.json", manifest)
    return manifest


def inspect_sass(text: str, entry: str) -> dict:
    """Describe all native sections without assuming a dynamic math path."""
    sections = {}
    current = None
    for number, line in enumerate(text.splitlines(), 1):
        match = SECTION.match(line)
        if match:
            current = {"instructions": [], "labels": {}}
            sections[match.group(1)] = current
            continue
        if current is None:
            continue
        match = LABEL.match(line)
        if match:
            current["labels"][match.group(1)] = len(current["instructions"])
        match = INSTRUCTION.match(line)
        if match:
            address, instruction = match.groups()
            predicate = ""
            if instruction.startswith("@"):
                predicate, instruction = instruction.split(None, 1)
            full = instruction.split()[0]
            current["instructions"].append(
                {
                    "address": int(address, 16),
                    "opcode": full.split(".")[0],
                    "full_opcode": full,
                    "predicate": predicate,
                    "text": instruction,
                    "sass_line": number,
                }
            )
    if entry not in sections or not sections[entry]["instructions"]:
        raise ValueError("Compiled entry missing from disassembly")
    summaries = {}
    for name, section in sections.items():
        instructions = section["instructions"]
        if not instructions:
            continue
        controls = []
        for instruction in instructions:
            if instruction["opcode"] not in CONTROL:
                continue
            target = TARGET.search(instruction["text"])
            index = section["labels"].get(target.group(1)) if target else None
            address = (
                instructions[index]["address"]
                if (index is not None and index < len(instructions))
                else None
            )
            controls.append(dict(instruction, resolved_target_address=address))
        footer = []
        for index, instruction in enumerate(instructions):
            if instruction["opcode"] != "EXIT" or instruction["predicate"]:
                continue
            prefix = instructions[: index + 1]
            tail = instructions[index + 1 :]
            if any(item["opcode"] in CONTROL for item in prefix):
                break
            resolved = {item["address"]: item for item in controls}
            if all(
                item["opcode"] == "NOP"
                or (
                    item["opcode"] == "BRA"
                    and not item["predicate"]
                    and resolved[item["address"]]["resolved_target_address"]
                    == item["address"]
                )
                for item in tail
            ):
                footer = tail
            break
        footer_addresses = {item["address"] for item in footer}
        candidate = [
            item
            for item in instructions
            if item["address"] not in footer_addresses
        ]
        summaries[name] = {
            "instruction_count": len(instructions),
            "encoded_instruction_bytes": len(instructions) * 16,
            "address_start": instructions[0]["address"],
            "address_end_exclusive": instructions[-1]["address"] + 16,
            "opcounts": dict(Counter(i["full_opcode"] for i in instructions)),
            "nonpadding_opcounts": dict(
                Counter(
                    i["full_opcode"]
                    for i in instructions
                    if i["opcode"] != "NOP"
                )
            ),
            "memory_opcounts": dict(
                Counter(
                    i["full_opcode"]
                    for i in instructions
                    if i["opcode"] in MEMORY
                )
            ),
            "constant_operand_instructions": [
                i for i in instructions if "c[" in i["text"]
            ],
            "predicated_instructions": [
                i for i in instructions if i["predicate"]
            ],
            "control_flow": controls,
            "candidate_control_flow": [
                item
                for item in controls
                if item["address"] not in footer_addresses
            ],
            "proven_unreachable_footer": footer,
            "footer_proof": (
                "Unconditional EXIT after a prefix without control "
                "instructions; tail contains only self-BRA and NOP."
                if footer
                else None
            ),
            "candidate_opcounts": dict(
                Counter(
                    item["full_opcode"]
                    for item in candidate
                    if item["opcode"] != "NOP"
                )
            ),
            "fp64_instructions": [
                i
                for i in instructions
                if i["opcode"] in {"DADD", "DMUL", "DFMA", "DSETP", "DMNMX"}
            ],
        }
    return {
        "entry": entry,
        "sections": summaries,
        "instruction_byte_basis": {
            "bytes": 16,
            "scope": "SM89 SASS encoding",
            "source": "https://docs.nvidia.com/cuda/cuda-binary-utilities/",
        },
        "dynamic_path": (
            "unknown beyond conservative terminal-footer reachability; "
            "static controls and call sections retained"
        ),
        "scope": "Entry includes prologue, address arithmetic and output.",
    }


def _dataset_identity(directory):
    issues = []
    receipt = {}
    try:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        emitted_path = directory / "source_manifest.json"
        emitted = json.loads(emitted_path.read_text())
        provenance = manifest["translation_provenance"]
        checks = {
            "source_mode": emitted.get("mode") == "source_only",
            "compile_mode": manifest["arguments"].get("mode")
            == "compile_only",
            "source_manifest": (
                _hash(emitted_path) == provenance["source_manifest_sha256"]
            ),
            "worker": (
                _hash(directory / "compile_worker.py")
                == provenance["worker_sha256"]
            ),
            "embedded_source_manifest": (
                emitted == manifest["arguments"]["source_manifest"]
            ),
            "actual_source_identity": bool(
                provenance.get("actual_cubie_source_hash")
                and provenance.get("actual_cubie_root")
            ),
            "active_compiler_identity": all(
                key in provenance.get("compiler_identity", {})
                for key in (
                    "toolchain_fingerprint",
                    "active_block_schedule",
                    "requested_block_schedule",
                    "operation_ordering_default",
                    "jit_flags",
                    "compiler_environment",
                    "active_backend",
                )
            ),
            "imported_source_identities": all(
                provenance.get("imported_sources", {}).get(key, {}).get("path")
                and provenance.get("imported_sources", {})
                .get(key, {})
                .get("sha256")
                for key in (
                    "cubie",
                    "cuda_simsafe",
                    "cuda_backend",
                    "environment_defaults",
                    "cache_fingerprint",
                    "hardware_probe",
                    "translation_parser",
                    "compile_specialization",
                    "jit_lowering",
                    "active_cuda_api",
                )
            ),
            "numba_version": bool(manifest["versions"].get("numba")),
        }
        if not all(checks.values()):
            issues.append("Compiled/emitted manifest provenance mismatch.")
        cases = {case["case"]: case for case in emitted["cases"]}
        if len(cases) != len(emitted["cases"]):
            issues.append("Emitted manifest has duplicate case names.")
        identity = compilation_identity(manifest)
        receipt = {
            "manifest_sha256": _hash(manifest_path),
            "compilation_identity_sha256": _identity_hash(identity),
            "compilation_identity": identity,
            "checks": checks,
            "cases": cases,
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append(f"Dataset identity unavailable: {error}")
    return receipt, issues


def _case_identity(directory, row, dataset):
    issues = []
    checks = {}
    try:
        emitted = dataset["cases"][row["case"]]
        checks["emitted_case"] = all(
            row.get(key) == value
            for key, value in emitted.items()
            if key != "status"
        )
        checks["compiled_manifest"] = (
            row["compiled_manifest_sha256"] == dataset["manifest_sha256"]
        )
        checks["compilation_identity"] = (
            row["compilation_identity_sha256"]
            == dataset["compilation_identity_sha256"]
        )
        folder = directory / row["case"]
        if folder.resolve().parent != directory.resolve():
            raise ValueError("Case artifact folder escapes dataset")
        checks["result_record"] = (
            json.loads((folder / "result.json").read_text()) == row
        )
        for name in ("source", "cubin", "sass", "ptx"):
            suffix = "py" if name == "source" else name
            checks[name] = (
                _hash(folder / f"kernel.{suffix}")
                == row["artifact_sha256"][name]
            )
        checks["emitted_source"] = (
            row["artifact_sha256"]["source"] == emitted["source_sha256"]
        )
        checks["resource_cubin"] = (
            row["resources"]["cubin_sha256"] == row["artifact_sha256"]["cubin"]
        )
        checks["sass_analysis"] = (
            inspect_sass(
                (folder / "kernel.sass").read_text(), row["analysis"]["entry"]
            )
            == row["analysis"]
        )
        if not all(checks.values()):
            issues.append("Raw artifact or source/compiler identity mismatch.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append(f"Case identity unavailable: {error}")
    return {"case": row["case"], "checks": checks}, issues


def analyze(directory: Path) -> dict:
    """Report exact static increments, refusing unsupported attribution."""
    rows = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text().splitlines()
    ]
    groups = {}
    for row in rows:
        groups.setdefault((row["category"], row["profile"]), []).append(row)
    dataset, identity_issues = _dataset_identity(directory)
    report = {
        "schema": SCHEMA,
        "source": str(directory),
        "groups": [],
        "identity": {
            key: value for key, value in dataset.items() if key != "cases"
        },
    }
    for (category, profile), group in groups.items():
        group.sort(key=lambda item: item["count"])
        issues, differences = list(identity_issues), []
        receipts = []
        valid = [row for row in group if row["status"] == "compiled"]
        expected_cases = Counter(
            case["case"]
            for case in dataset.get("cases", {}).values()
            if (case["category"], case["profile"]) == (category, profile)
        )
        if Counter(row["case"] for row in group) != expected_cases:
            issues.append("Emitted case membership differs from result group.")
        if len(valid) != len(group):
            issues.append("One or more compilation cases failed.")
        positive_counts = {row["count"] for row in valid if row["count"] > 0}
        if len(positive_counts) < 3:
            issues.append("Three distinct positive fragment counts required.")
        if len({row["count"] for row in valid}) != len(valid):
            issues.append("Duplicate fragment counts cannot be differenced.")
        if any(row["count"] <= 0 for row in valid):
            issues.append(
                "Zero-count baseline is diagnostic, not attribution."
            )
        resources = set()
        effective_flags = set()
        for row in valid:
            receipt, case_issues = _case_identity(directory, row, dataset)
            receipts.append(receipt)
            issues.extend(case_issues)
            resource = row["resources"]
            if "jit_kwargs" not in resource:
                issues.append("Effective compiler flags unavailable.")
            else:
                effective_flags.add(_identity_hash(resource["jit_kwargs"]))
            resources.add(
                tuple(
                    resource[k]
                    for k in (
                        "registers",
                        "local_bytes_per_thread",
                        "static_shared_bytes",
                    )
                )
            )
            analysis = row["analysis"]
            if len(analysis["sections"]) != 1:
                issues.append("Multiple code sections need call attribution.")
            section = analysis["sections"][analysis["entry"]]
            controls = section.get(
                "candidate_control_flow", section["control_flow"]
            )
            if controls or section["predicated_instructions"]:
                issues.append(
                    "Control or predicates leave dynamic path unknown."
                )
            if section["fp64_instructions"]:
                issues.append("Unexpected FP64 arithmetic in compiled entry.")
        if len(resources) != 1:
            issues.append("Register/local/shared resources change with count.")
        if len(effective_flags) != 1:
            issues.append("Effective compiler flags differ between counts.")
        for lower, upper in zip(valid, valid[1:]):
            first = lower["analysis"]["sections"][lower["analysis"]["entry"]]
            second = upper["analysis"]["sections"][upper["analysis"]["entry"]]
            span = upper["count"] - lower["count"]
            if span <= 0:
                differences.append(
                    {
                        "lower_count": lower["count"],
                        "upper_count": upper["count"],
                        "opcode_increment_per_fragment": None,
                        "reason": "Nonpositive count span; division undefined.",
                    }
                )
                continue
            first_counts = first.get(
                "candidate_opcounts", first["nonpadding_opcounts"]
            )
            second_counts = second.get(
                "candidate_opcounts", second["nonpadding_opcounts"]
            )
            delta = {
                opcode: (
                    second_counts.get(opcode, 0) - first_counts.get(opcode, 0)
                )
                / span
                for opcode in sorted(set(first_counts) | set(second_counts))
            }
            delta = {key: value for key, value in delta.items() if value}
            differences.append(
                {
                    "lower_count": lower["count"],
                    "upper_count": upper["count"],
                    "opcode_increment_per_fragment": delta,
                    "nonpadding_byte_increment_per_fragment": sum(
                        delta.values()
                    )
                    * 16,
                }
            )
            if first["memory_opcounts"] != second["memory_opcounts"]:
                issues.append(
                    "Memory instructions vary; no baseline subtraction."
                )
            if any(
                value < 0 or value != int(value) for value in delta.values()
            ):
                issues.append("Opcode increments are negative or nonintegral.")
        if differences and any(
            item["opcode_increment_per_fragment"]
            != differences[0]["opcode_increment_per_fragment"]
            for item in differences[1:]
        ):
            issues.append("Opcode increments depend on fragment count.")
        if differences and not differences[0]["opcode_increment_per_fragment"]:
            issues.append("No growing native stream; folding/CSE is possible.")
        report["groups"].append(
            {
                "category": category,
                "profile": profile,
                "cases": [row["case"] for row in group],
                "status": "context_dependent"
                if issues
                else "static_candidate",
                "issues": sorted(set(issues)),
                "differences": differences,
                "case_identity_receipts": receipts,
                "interpretation": (
                    "Exact static differences exclude only NOP padding and "
                    "a proven unreachable terminal footer. "
                    "A candidate is conditional on this source DAG, flags and "
                    "toolchain; it is not a dynamic latency or universal weight."
                ),
            }
        )
    _write(directory / "translation.json", report)
    return report


WORKER = '''"""Explicit compile-only worker; never launches kernels."""
import hashlib
import importlib.util
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, RESEARCH_REPO)
import numpy as np
from benchmarks.hardware_model import hardware_probes as hardware
from benchmarks.hardware_model.operation_translation import (
    inspect_sass, compilation_identity, _identity_hash)
import cubie
import cubie.cuda_simsafe as imported_cuda_simsafe
import cubie.cuda_backend as imported_cuda_backend
import cubie._env as imported_environment
import cubie.cubie_cache as imported_cache
from cubie._utils import package_source_hash
from cubie.cuda_simsafe import JITFlags

def imported_source(value):
    path = Path(inspect.getfile(value)).resolve()
    return {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def effective_compiler_identity():
    flags = JITFlags()
    return {
        "active_backend": imported_cuda_backend.CUDA_BACKEND,
        "cuda_simulation": bool(imported_cuda_simsafe.CUDA_SIMULATION),
        "toolchain_fingerprint": imported_cache.toolchain_fingerprint(),
        "active_block_schedule": imported_environment.active_block_schedule(),
        "requested_block_schedule": imported_environment.block_schedule_default(),
        "operation_ordering_default": inspect.signature(cubie.create_ODE_system)
            .parameters["operation_ordering"].default,
        "jit_flags": {field.name: getattr(flags, field.name)
                      for field in flags.__attrs_attrs__},
        "compiler_environment": {
            key: value for key, value in sorted(os.environ.items())
            if (key.startswith(("CUBIE_", "NUMBA_", "CUDA_", "CUPY_"))
                or key in {"CUDAHOME", "PL_SMOKE", "NVDISASM"})
            and key not in {"CUBIE_CACHE_DIR", "CUBIE_KERNEL_CACHE_DIR",
                            "CUBIE_MAX_CACHE_ENTRIES", "CUBIE_LIVENESS_LOG"}
        },
    }

source_root, destination = map(Path, sys.argv[1:3])
source_manifest_data = (source_root / "manifest.json").read_bytes()
manifest = json.loads(source_manifest_data)
(destination / "source_manifest.json").write_bytes(source_manifest_data)
if hardware.CUDA_SIMULATION:
    raise RuntimeError("Operation translation requires the real CUDA backend")
nvdisasm = hardware._tool("nvdisasm", sys.argv[3] or None)
device = hardware.cuda.get_current_device()
if tuple(device.compute_capability) != (8, 9):
    raise RuntimeError("This encoding analysis is restricted to SM89")
compiled_manifest = hardware._manifest(
    SimpleNamespace(mode="compile_only", source_manifest=manifest), nvdisasm)
for package in ("numba", "llvmlite", "cubie-numba-cuda-mlir",
                "numba-cuda-mlir", "numba-cuda", "cuda-bindings"):
    try:
        compiled_manifest["versions"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        compiled_manifest["versions"][package] = None
compiled_manifest["translation_provenance"] = {
    "source_manifest_sha256": hashlib.sha256(source_manifest_data).hexdigest(),
    "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "translator_sha256": hashlib.sha256((Path(RESEARCH_REPO) /
        "benchmarks/hardware_model/operation_translation.py").read_bytes()
        ).hexdigest(),
    "actual_cubie_source_hash": package_source_hash(),
    "actual_cubie_root": str(Path(cubie.__file__).resolve().parent),
    "compiler_identity": effective_compiler_identity(),
    "imported_sources": {
        "cubie": imported_source(cubie),
        "cuda_simsafe": imported_source(imported_cuda_simsafe),
        "cuda_backend": imported_source(imported_cuda_backend),
        "environment_defaults": imported_source(imported_environment),
        "cache_fingerprint": imported_source(imported_cache),
        "hardware_probe": imported_source(hardware),
        "translation_parser": imported_source(inspect_sass),
        "compile_specialization": imported_source(
            hardware.compile_kernel_specialization),
        "jit_lowering": imported_source(hardware.get_jit_kwargs),
        "active_cuda_api": imported_source(hardware.cuda),
    },
}
identity_hash = _identity_hash(compilation_identity(compiled_manifest))
hardware._write_json(destination / "manifest.json", compiled_manifest)
compiled_manifest_hash = hashlib.sha256(
    (destination / "manifest.json").read_bytes()).hexdigest()
failures = 0
for case in manifest["cases"]:
    folder = destination / case["case"]
    folder.mkdir()
    row = dict(case)
    row["compiled_manifest_sha256"] = compiled_manifest_hash
    row["compilation_identity_sha256"] = identity_hash
    try:
        original = Path(case["source"])
        data = original.read_bytes()
        if hashlib.sha256(data).hexdigest() != case["source_sha256"]:
            raise ValueError("Source hash differs from emitted manifest")
        source = folder / "kernel.py"
        source.write_bytes(data)
        module_name = "translation_" + case["case"]
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        flags = JITFlags(lineinfo=False, **case["flag_overrides"])
        kwargs = hardware.get_jit_kwargs(flags)
        if effective_compiler_identity() != compiled_manifest[
                "translation_provenance"]["compiler_identity"]:
            raise ValueError("Active compiler/default identity changed within cohort")
        kernel = hardware.cuda.jit(**kwargs)(module.probe)
        index_type = {"index_u32": np.uint32, "index_i64": np.int64}.get(
            case["category"], np.int32)
        output_type = index_type if case["category"].startswith("index_") \\
            else np.float32
        signature = (np.empty(1, dtype=output_type),
                     np.float32(1.25), np.float32(0.75), np.float32(0.125),
                     np.float32(0.25), index_type(1), index_type(3),
                     index_type(7))
        start = time.perf_counter()
        hardware.compile_kernel_specialization(kernel, signature)
        elapsed = time.perf_counter() - start
        (compiled,) = kernel.overloads.values()
        cubin, entry = hardware._compiled_cubin(compiled)
        (folder / "kernel.cubin").write_bytes(cubin)
        ptx = kernel.inspect_asm()
        if isinstance(ptx, dict):
            ptx = "\\n".join(ptx.values())
        (folder / "kernel.ptx").write_text(ptx, encoding="utf-8")
        command = hardware._command([nvdisasm, "-c", folder / "kernel.cubin"])
        hardware._write_json(folder / "disassembly_command.json", command)
        if command["returncode"]:
            raise RuntimeError(command["stderr"])
        (folder / "kernel.sass").write_text(command["stdout"], encoding="utf-8")
        compiled._ensure_kernel_attrs()
        row["resources"] = {
            "registers": int(next(iter(kernel.get_regs_per_thread().values()))),
            "local_bytes_per_thread": int(next(iter(
                kernel.get_local_mem_per_thread().values()))),
            "static_shared_bytes": int(next(iter(
                kernel.get_shared_mem_per_block().values()))),
            "compile_seconds": elapsed, "jit_kwargs": kwargs,
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        }
        row["analysis"] = inspect_sass(command["stdout"], entry)
        row["artifact_sha256"] = {
            name: hashlib.sha256((folder / ("kernel." + suffix))
                                 .read_bytes()).hexdigest()
            for name, suffix in (("source", "py"), ("cubin", "cubin"),
                                 ("sass", "sass"), ("ptx", "ptx"))
        }
        row["status"] = "compiled"
        row["compiled_source"] = str(source)
    except Exception as error:
        failures += 1
        row["status"] = "error"
        row["error"] = repr(error)
    hardware._write_json(folder / "result.json", row)
    hardware._append_json(destination / "results.jsonl", row)
    print(json.dumps({"case": case["case"], "status": row["status"]}),
          flush=True)
raise SystemExit(bool(failures))
'''


def compile_dataset(source: Path, output: Path, nvdisasm: str) -> int:
    """Explicitly run the CUDA compile worker without launching kernels."""
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("mode") != "source_only"
    ):
        raise ValueError("Expected an emitted source-only manifest")
    output.mkdir(parents=True, exist_ok=False)
    worker = output / "compile_worker.py"
    text = WORKER.replace("RESEARCH_REPO", repr(str(REPO)))
    compile(text, str(worker), "exec")
    worker.write_text(text, encoding="utf-8")
    arguments = [
        sys.executable,
        str(worker.resolve()),
        str(source.resolve()),
        str(output.resolve()),
        nvdisasm or "",
    ]
    with (output / "compiler.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            arguments, cwd=REPO, stdout=log, stderr=subprocess.STDOUT
        )
    _write(
        output / "compile_command.json",
        {
            "command": arguments,
            "returncode": result.returncode,
            "worker_sha256": _hash(worker),
            "launches": 0,
        },
    )
    if (output / "results.jsonl").exists():
        analyze(output)
    return result.returncode


def main() -> int:
    """Run source emission, explicit compilation, or CPU-only analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    emitter = commands.add_parser("emit", help="CPU-only source construction")
    emitter.add_argument("--output", type=Path, required=True)
    emitter.add_argument(
        "--categories", type=_tokens, default=list(CATEGORIES)
    )
    emitter.add_argument(
        "--counts", type=_tokens, default=["1", "2", "4", "8"]
    )
    emitter.add_argument("--profiles", type=_tokens, default=["default"])
    compiler = commands.add_parser(
        "compile", help="Explicit CUDA compile only"
    )
    compiler.add_argument("--source", type=Path, required=True)
    compiler.add_argument("--output", type=Path, required=True)
    compiler.add_argument("--nvdisasm")
    inspector = commands.add_parser("analyze", help="CPU-only saved analysis")
    inspector.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "emit":
        unknown = set(args.categories) - CATEGORIES.keys()
        unknown.update(set(args.profiles) - PROFILES.keys())
        counts = [int(value) for value in args.counts]
        if unknown or not counts or min(counts) < 0:
            parser.error(
                f"Unknown category/profile or invalid counts: {unknown}"
            )
        if any(
            len(values) != len(set(values))
            for values in (args.categories, args.profiles, counts)
        ):
            parser.error("Duplicate categories, profiles or counts")
        manifest = emit(args.output, args.categories, counts, args.profiles)
        print(
            json.dumps(
                {
                    "cases": len(manifest["cases"]),
                    "manifest": str(args.output / "manifest.json"),
                }
            )
        )
        return 0
    if args.command == "compile":
        return compile_dataset(args.source, args.output, args.nvdisasm)
    report = analyze(args.directory)
    print(json.dumps({"groups": len(report["groups"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
