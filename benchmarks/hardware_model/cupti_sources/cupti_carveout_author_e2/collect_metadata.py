"""Wrap exact E5 execution with activity-only CUPTI kernel metadata."""

import argparse
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import traceback
import uuid

import numpy as np

ROOT = Path(__file__).parent
BUILD_ROOT = ROOT.parent / "cupti_carveout_author_e1"
COLLECTOR_DLL = BUILD_ROOT / "collector.dll"
RAW = ROOT.parent.parent
E5 = ROOT.parent / "controlled_carveout_author_e5/controlled_carveout.py"
E5_SHA = "083f5326dc2136bc490f6ad73427ef359c02a7b74eaf31bda368cf827e81e4b5"
PREPARED = (
    ROOT.parent / "native_policy_profile_author_e5/prepared/prepared.json"
)
PREPARED_SHA = (
    "0ce3125d1fa9960d172518c7b5255bebcb57e7a3b3aa8a1f20000e32705a2196"
)
WRAPPER = (
    ROOT.parent
    / "native_policy_profile_author_e5/source/native_policy_profile.py"
)
WRAPPER_SHA = (
    "f749a68eb5a13ba3ee431cb517625e1c0cdad4c2b4678d4700d092ebe88fed87"
)
CUDA = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
CUPTI = CUDA / "extras/CUPTI/lib64/cupti64_2026.2.1.dll"


def asset(path):
    path = Path(path)
    return dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def check(record):
    path = Path(record["path"])
    if asset(path)["sha256"] != record["sha256"]:
        raise ValueError(f"Changed bound artifact: {path}")
    return path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def loaded_path(handle):
    """Resolve the actual loaded module through the Windows API."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GetModuleFileNameW
    function.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    function.restype = ctypes.c_uint32
    buffer = ctypes.create_unicode_buffer(32768)
    length = function(handle, buffer, len(buffer))
    if not length or length >= len(buffer):
        raise OSError(ctypes.get_last_error(), "GetModuleFileNameW failed")
    return Path(buffer.value)


def validate_activity(path, solver_receipt, percent):
    """Require complete metadata and exact E5 outputs without using timing."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    headers = [row for row in rows if row["type"] == "header"]
    versions = [row for row in rows if row["type"] == "runtime_version"]
    summaries = [row for row in rows if row["type"] == "summary"]
    if len(headers) != 1 or len(versions) != 1 or len(summaries) != 1:
        raise ValueError("Incomplete collector lifecycle")
    header, summary = headers[0], summaries[0]
    if (
        header["kernel_record_version"] != 12
        or header["header_cupti_api_version"] != versions[0]["value"]
    ):
        raise ValueError("CUPTI header/runtime record version mismatch")
    if (
        summary["errors"]
        or summary["dropped_records"]
        or summary["outstanding_buffers"]
        or summary["active_callbacks"]
    ):
        raise ValueError(
            "Collector errors, loss, or incomplete buffer ownership"
        )
    if (
        not summary["safe_to_close"]
        or summary["requested_buffers"] != summary["completed_buffers"]
    ):
        raise ValueError("Collector did not finish buffer ownership")
    for row in rows:
        if row["type"] == "api" and not row["accepted"]:
            raise ValueError("CUPTI API failure retained")
        if row["type"] == "error" or (
            row["type"] == "dropped" and row["count"]
        ):
            raise ValueError("Collector loss/error record retained")
    disassembly_path = Path(solver_receipt["output"]) / "disassembly.json"
    disassembly = json.loads(disassembly_path.read_text())["stdout"]
    names = re.findall(r"^\s*//-+\s*\.text\.(\S+)", disassembly, re.MULTILINE)
    if len(names) != 1:
        raise ValueError("Unexpected exact native entry inventory")
    kernels = [row for row in rows if row["type"] == "kernel"]
    matching = [row for row in kernels if row["name"] == names[0]]
    if len(matching) != 2 or summary["kernel_records"] != len(kernels):
        raise ValueError("Expected two exact E5 kernel records")
    if len({row["correlation_id"] for row in matching}) != 2:
        raise ValueError("Kernel correlation identities are not distinct")
    resources = solver_receipt["diagnostic_gate"]["resources"]
    local_comparisons = []
    for row in matching:
        if row["kind"] != header["enabled_kind"]:
            raise ValueError("Unexpected enabled activity kind")
        if row["grid"] != [2048, 1, 1] or row["block"] != [1, 128, 1]:
            raise ValueError("Activity launch geometry differs from E5")
        actual = (
            row["registers_per_thread"],
            row["static_shared_bytes"],
            row["dynamic_shared_bytes"],
        )
        expected = (
            resources["regs"],
            resources["shared"],
            4,
        )
        if actual != expected:
            raise ValueError(
                f"Activity resource metadata differs: {actual} != {expected}"
            )
        if (
            row["carveout_requested"] != 1
            or row["requested_percent"] != percent
        ):
            raise ValueError(
                "Activity requested preference differs from actual E5 setter"
            )
        local_comparisons.append(
            dict(
                correlation_id=row["correlation_id"],
                actual_function_local_bytes=resources["local"],
                activity_local_bytes_per_thread=row["local_bytes_per_thread"],
                activity_local_bytes_total=row["local_bytes_total"],
                equal=row["local_bytes_per_thread"] == resources["local"],
                interpretation="Cross-provider comparison. No zero sentinel, inferred correction or learned whitelist. A disagreement remains unresolved.",
            )
        )
    with np.load(
        check(solver_receipt["reference"]), allow_pickle=False
    ) as saved:
        reference = {name: saved[name] for name in saved.files}
    if (
        solver_receipt["status"]
        != "ACTUAL_FUNCTION_FUNCTIONAL_REPRODUCTION_PASS"
        or solver_receipt["launches"] != 2
    ):
        raise ValueError("Exact E5 run did not finish both launches")
    if [phase["phase"] for phase in solver_receipt["phases"]] != [
        "warmup",
        "capture",
    ]:
        raise ValueError("E5 phase identities differ")
    arrays = []
    for phase in solver_receipt["phases"]:
        path = check(phase["arrays"])
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(reference):
                raise ValueError("Output array inventory differs")
            for name, expected in reference.items():
                actual = saved[name]
                if (
                    actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or actual.tobytes() != expected.tobytes()
                ):
                    raise ValueError("Exact own-candidate output differs")
        if not all(phase["checks"].values()):
            raise ValueError("E5 phase guard failed")
        if (
            phase["actual_attributes"]["CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"]
            != resources["local"]
        ):
            raise ValueError("Actual function local-memory identity differs")
        arrays.append(asset(path))
    return dict(
        header=header,
        runtime_version=versions[0],
        summary=summary,
        matching_kernels=matching,
        total_kernel_records=len(kernels),
        output_arrays=arrays,
        local_memory_comparisons=local_comparisons,
        resource_consistency_passed=all(
            row["equal"] for row in local_comparisons
        ),
        partition_qualification="shared_memory_executed_bytes copies CUpti_ActivityKernel12.sharedMemoryExecuted, the driver-selected size. It is observed independently of NCU, not inferred from the requested percent. CacheConfig is not authoritative when carveout_requested is set.",
    )


def run(percent, output_root):
    """Collect metadata in one fresh process without changing E5."""
    if percent not in (0, 8, 16, 32, 64, 100):
        raise ValueError("Unsupported preference")
    output = Path(output_root) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + f"_pid{os.getpid()}_"
        + uuid.uuid4().hex
    )
    output.mkdir(parents=True, exist_ok=False)
    activity = output / "activity.jsonl"
    receipt = dict(
        status="STARTED",
        source=asset(__file__),
        percent=percent,
        purpose="Independent activity metadata; timestamps are not timing estimates",
    )
    try:
        build_path = BUILD_ROOT / "build_receipt.json"
        build = json.loads(build_path.read_text())
        for record in (
            build["artifacts"] + build["headers"] + build["providers"]
        ):
            check(record)
        check(build["source"])
        for path, expected in (
            (E5, E5_SHA),
            (PREPARED, PREPARED_SHA),
            (WRAPPER, WRAPPER_SHA),
        ):
            check(dict(path=str(path), sha256=expected))
        receipt["inputs"] = [
            asset(path)
            for path in (
                E5,
                PREPARED,
                WRAPPER,
                build_path,
                COLLECTOR_DLL,
                CUPTI,
            )
        ]
        with ExitStack() as stack:
            for directory in (CUPTI.parent, CUDA / "bin", BUILD_ROOT):
                stack.enter_context(os.add_dll_directory(str(directory)))
            vendor = ctypes.WinDLL(str(CUPTI))
            collector = ctypes.CDLL(str(COLLECTOR_DLL))
            paths = [
                loaded_path(vendor._handle),
                loaded_path(collector._handle),
            ]
            if [os.path.normcase(str(path.resolve())) for path in paths] != [
                os.path.normcase(str(path.resolve()))
                for path in (CUPTI, COLLECTOR_DLL)
            ]:
                raise ValueError(
                    "Loaded DLL ownership differs from bound provider"
                )
            receipt["loaded_modules"] = [asset(path) for path in paths]
            collector.collector_start.argtypes = [ctypes.c_wchar_p]
            collector.collector_start.restype = ctypes.c_int
            collector.collector_stop.argtypes = []
            collector.collector_stop.restype = ctypes.c_int
            try:
                receipt["start_result"] = collector.collector_start(
                    str(activity)
                )
                if receipt["start_result"] != 0:
                    raise ValueError("CUPTI collector start failed")
                spec = importlib.util.spec_from_file_location(
                    "unchanged_e5", E5
                )
                harness = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(harness)
                receipt["solver"] = harness.run(
                    str(PREPARED),
                    str(WRAPPER),
                    percent,
                    str(output / "solver"),
                )
            finally:
                receipt["stop_result"] = collector.collector_stop()
            if receipt["stop_result"] != 0:
                raise ValueError("CUPTI collector stop failed")
            receipt["activity"] = asset(activity)
            receipt["validation"] = validate_activity(
                activity, receipt["solver"], percent
            )
        receipt["resource_consistency_passed"] = receipt["validation"][
            "resource_consistency_passed"
        ]
        receipt["status"] = (
            "ACTIVITY_METADATA_AND_EXACT_E5_OUTPUT_PASS"
            if receipt["resource_consistency_passed"]
            else "COMPLETE_WITH_LOCAL_METADATA_DISAGREEMENT"
        )
    except Exception:
        receipt["status"] = "FAILED_RETAINED"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        if activity.exists():
            receipt["activity"] = asset(activity)
        write(output / "receipt.json", receipt)
    return dict(output=str(output), **receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--percent", type=int, required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.percent, arguments.output), indent=2))
