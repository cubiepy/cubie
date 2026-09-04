"""Audit saved placement profiles with a separate cohort identity schema.

Only saved report import and saved cubin disassembly run externally.
No CuBIE, CUDA or profiling-execution API is imported. An explicit prior
independent source-review receipt can bind a historical profile script
whose current bytes have changed. This does not replace raw-data gates.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import solver_profile_analysis as core


SCRIPT = Path(__file__).resolve()
IMPORTED_SHA256 = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
IMPORTED_CORE_SHA256 = core.digest(core.SCRIPT)
ROLES = ("baseline", "duplicate", "shared")
GEOMETRY_FIELDS = (
    "blocksize",
    "dynshared",
    "bytes_per_run",
    "blocks_per_sm",
    "resident_threads",
    "waves",
)
COMPILE_FIELDS = (
    "role",
    "source_hash",
    "compiler_identity",
    "config_hash",
    "cubin_sha256",
    "entry_name",
    "requested_blocksize",
    "registers_per_thread",
    "local_bytes_per_thread",
    "shared_bytes_per_run",
    "multiprocessor_count",
    "compute_capability",
)


def read_json(path):
    """Read a retained JSON receipt."""
    return json.loads(Path(path).read_text())


def require_fields(left, right, fields, label):
    """Require exact presence and equality of every specified field."""
    differences = {
        key: [left.get(key), right.get(key)]
        for key in fields
        if key not in left or key not in right or left[key] != right[key]
    }
    if differences:
        raise ValueError(f"{label}: {differences}")


def contained(directory, value):
    """Admit only existing artifacts in the specified evidence directory."""
    path = Path(value).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"Missing or out-of-directory artifact: {path}")
    return path


def arrays(directory, path, expected):
    """Verify saved array membership, bytes, type and numerical status."""
    path = contained(directory, path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("Saved array membership differs")
        actual = {
            name: core.array_identity(archive[name]) for name in archive.files
        }
        if actual != expected:
            raise ValueError(f"Saved array identities differ: {path}")
        if (
            "state" in archive.files
            and not np.isfinite(archive["state"]).all()
        ):
            raise ValueError("Saved state is nonfinite")
        if "status_codes" in archive.files and np.any(archive["status_codes"]):
            raise ValueError("Saved status contains failures")
    return dict(path=str(path), file_sha256=core.digest(path), arrays=actual)


def artifacts(directory, compiled):
    """Check exact saved cubin/PTX and decompressed SASS identities."""
    result = {}
    for name in ("cubin", "ptx", "sass"):
        path = contained(directory, compiled["artifacts"][name])
        data = path.read_bytes()
        decoded = gzip.decompress(data) if name == "sass" else data
        result[name] = dict(
            path=str(path),
            file_sha256=hashlib.sha256(data).hexdigest(),
            content_sha256=hashlib.sha256(decoded).hexdigest(),
            content_bytes=len(decoded),
        )
    if result["cubin"]["content_sha256"] != compiled["cubin_sha256"]:
        raise ValueError("Compiled cubin differs from saved bytes")
    return result


def geometry(compiled, manifest):
    """Require requested, production-limited and actual pinned geometry."""
    blocksize = manifest["requested_blocksize"]
    actual = compiled["actual_pinned_geometry"]
    if compiled["blocked_reasons"] or (
        compiled["requested_blocksize"] != blocksize
        or actual["blocksize"] != blocksize
        or compiled["production_limited_geometry"]["blocksize"] != blocksize
        or actual["blocks_per_sm"] <= 0
        or actual["waves"] < 2
    ):
        raise ValueError("Ineligible compiled placement geometry")
    require_fields(
        compiled["requested_geometry"],
        actual,
        GEOMETRY_FIELDS,
        "Requested/actual geometry differs",
    )
    n_runs = manifest["protocol"]["n_runs"]
    minimum = 2 * compiled["multiprocessor_count"] * actual["resident_threads"]
    if n_runs < minimum or n_runs % blocksize:
        raise ValueError("Grid does not meet exact two-wave geometry")
    if actual["resident_threads"] != blocksize * actual["blocks_per_sm"]:
        raise ValueError("Resident thread count differs from geometry")
    waves = n_runs / (
        compiled["multiprocessor_count"] * actual["resident_threads"]
    )
    if actual["waves"] != waves:
        raise ValueError("Recorded occupancy waves differ from exact grid")


def validate_cohort(directory, role):
    """Revalidate a complete mirrored placement cohort and all raw gates."""
    result = read_json(directory / "result.json")
    manifest = result["manifest"]
    if result["status"] != "complete" or (
        manifest["kind"] != "matched_placement_cohort"
        or core.json_digest(manifest) != result["manifest_sha256"]
        or read_json(directory / "manifest.json")
        != {"manifest": manifest, "manifest_sha256": result["manifest_sha256"]}
    ):
        raise ValueError("Original cohort/manifest is not complete")
    protocol = manifest["protocol"]
    if protocol["samples_per_role_per_block"] < 6 or (
        protocol["minimum_kernel_ms"] < 20
        or protocol["minimum_occupancy_waves"] < 2
    ):
        raise ValueError("Original cohort lacks required measurement gates")
    for name in ROLES:
        cleanup = result["worker_cleanup"][name]
        if (
            cleanup["errors"]
            or cleanup["forced_shutdown"]
            or cleanup["returncode"]
        ):
            raise ValueError("Original worker failed cleanup")
        compiled = result["compiles"][name]
        geometry(compiled, manifest)
        require_fields(
            manifest["source_and_compiler"],
            compiled,
            ("source_hash", "compiler_identity"),
            "Compiler drift",
        )
    if (
        result["compiles"]["baseline"]["cubin_sha256"]
        != (result["compiles"]["duplicate"]["cubin_sha256"])
    ):
        raise ValueError("Original duplicate baseline binary differs")
    selected = [
        row
        for row in result["attempts"]
        if row["attempt"] == result["accepted_attempt"]
    ]
    if (
        len(selected) != 1
        or selected[0]["status"] != "complete"
        or (selected[0]["duration"] != result["duration"])
    ):
        raise ValueError("Original accepted attempt differs")
    blocks = selected[0]["blocks"]
    if sorted(row["block"] for row in blocks) != list(
        range(protocol["paired_blocks"])
    ):
        raise ValueError("Original paired block membership differs")
    path = contained(directory, result["timing_records"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_key = {row["key"]: row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Original raw timing keys are not unique")
    accepted = [
        row for row in rows if row["attempt"] == result["accepted_attempt"]
    ]
    for row in accepted:
        name = row["role"]
        if name not in ROLES or (
            row["duration"] != result["duration"]
            or row["n_runs"] != protocol["n_runs"]
            or row["blocksize"] != manifest["requested_blocksize"]
            or row["chunks"] != 1
            or row["status_hist"]["failed"]
            or row["finite_state"] is not True
            or row["manifest_sha256"] != result["manifest_sha256"]
            or row["source_hash"]
            != manifest["source_and_compiler"]["source_hash"]
            or row["cubin_sha256"] != result["compiles"][name]["cubin_sha256"]
            or not np.isfinite(row["kernel_ms"])
            or row["kernel_ms"] <= 0
        ):
            raise ValueError("Original accepted raw timing is ineligible")
        if row["phase"] in ("pilot", "measurement") and (
            row["kernel_ms"] < protocol["minimum_kernel_ms"]
        ):
            raise ValueError("Original accepted timing sample is too short")
    if Counter(
        row["role"] for row in accepted if row["phase"] == "pilot"
    ) != Counter(ROLES):
        raise ValueError("Original accepted pilots differ")
    keys = []
    warm_receipts = []
    warm = None
    for block in blocks:
        measurements = [by_key[key] for key in block["measurement_keys"]]
        expected = [
            (index, name)
            for index in range(protocol["samples_per_role_per_block"])
            for name in (
                ("baseline", "shared", "duplicate")
                if index % 2 == 0
                else ("duplicate", "shared", "baseline")
            )
        ]
        if [(row["sample"], row["role"]) for row in measurements] != expected:
            raise ValueError("Original mirrored sample order differs")
        if any(
            row["phase"] != "measurement"
            or row["paired_block"] != block["block"]
            for row in measurements
        ):
            raise ValueError("Original measurement phase/block differs")
        keys.extend(block["measurement_keys"])
        baseline = block["warm"]["baseline"]
        for name in ROLES:
            row = block["warm"][name]
            if (
                by_key.get(row["key"]) != row
                or row["arrays"] != baseline["arrays"]
            ):
                raise ValueError(
                    "Original warm identity/role equality differs"
                )
            if not all(
                block["numerical_checks"][name][key] is True
                for key in ("shape_match", "finite", "exact", "status_exact")
            ):
                raise ValueError("Original numerical comparison failed")
            warm_receipts.append(
                arrays(directory, row["snapshot"], row["arrays"])
            )
            if block["block"] == 0 and name == role:
                warm = row
    if Counter(keys) != Counter(
        row["key"] for row in accepted if row["phase"] == "measurement"
    ):
        raise ValueError(
            "Original raw/accepted measurement membership differs"
        )
    construction = result["construction"][role]
    evidence = dict(
        directory=str(directory),
        role=role,
        result_sha256=core.digest(directory / "result.json"),
        manifest_sha256=result["manifest_sha256"],
        manifest_file_sha256=core.digest(directory / "manifest.json"),
        timings_sha256=core.digest(path),
        accepted_attempt=result["accepted_attempt"],
        duration=result["duration"],
        measurement_keys=keys,
        original_warm_key=warm["key"],
        warm_arrays=warm_receipts,
        inputs=arrays(
            directory, directory / f"{role}-inputs.npz", construction["inputs"]
        ),
        artifacts=artifacts(directory, result["compiles"][role]),
    )
    return result, warm, evidence, [by_key[key] for key in keys]


def validate_profile(profile):
    """Bind the actual single placement solve to the original raw cohort."""
    directory = profile / "benchmark"
    result = read_json(directory / "result.json")
    role = result["role"]
    if role not in ("baseline", "shared") or (
        result["status"] != "complete"
        or not result["solver_closed"]
        or result["requested_solve_count"] != 1
        or result["completed_solve_count"] != 1
        or not result["gpu_execution"]
        or not result["kernel_compilation"]
    ):
        raise ValueError("Expected exactly one completed placement solve")
    for name in (
        "environment_differences",
        "construction_differences",
        "compile_differences",
        "geometry_differences",
    ):
        if result[name]:
            raise ValueError(f"Profile identity mismatch: {name}")
    cohort_dir = Path(result["cohort_evidence"]["directory"]).resolve()
    cohort, warm, evidence, timings = validate_cohort(cohort_dir, role)
    if evidence != result["cohort_evidence"]:
        raise ValueError("Retained original cohort evidence changed")
    current = result["construction"]
    original = cohort["construction"][role]
    require_fields(
        original,
        current,
        (
            "role",
            "location",
            "buffers",
            "resolved_buffers",
            "config_hash",
            "inputs",
            "workload",
            "compilation_check",
        ),
        "Construction drift",
    )
    if current["compilation_check"]["native_overloads"] != 0:
        raise ValueError("Profile host preparation compiled unexpectedly")
    require_fields(
        original["generated_source"],
        current["generated_source"],
        ("sha256", "fn_hash"),
        "Generated source mismatch",
    )
    source = contained(profile, current["generated_source"]["path"])
    if core.digest(source) != current["generated_source"]["sha256"]:
        raise ValueError("Retained generated source bytes changed")
    arrays(directory, directory / f"{role}-inputs.npz", current["inputs"])
    compiled = result["compile"]
    require_fields(
        cohort["compiles"][role], compiled, COMPILE_FIELDS, "Native drift"
    )
    geometry(compiled, cohort["manifest"])
    require_fields(
        cohort["compiles"][role]["actual_pinned_geometry"],
        compiled["actual_pinned_geometry"],
        GEOMETRY_FIELDS,
        "Geometry drift",
    )
    actual_artifacts = artifacts(directory, compiled)
    if actual_artifacts != result["artifacts"]:
        raise ValueError("Retained profile artifacts changed")
    for name in ("cubin", "ptx", "sass"):
        if (
            actual_artifacts[name]["content_sha256"]
            != evidence["artifacts"][name]["content_sha256"]
        ):
            raise ValueError("Profile artifact content differs from cohort")
    sample = result["sample"]
    if (
        sample["role"] != role
        or sample["arrays"] != warm["arrays"]
        or (
            sample["duration"] != cohort["duration"]
            or sample["n_runs"] != cohort["manifest"]["protocol"]["n_runs"]
            or sample["blocksize"] != cohort["manifest"]["requested_blocksize"]
            or sample["chunks"] != 1
            or sample["status_hist"]["failed"]
            or sample["finite_state"] is not True
            or result["array_identity_matches"] is not True
        )
    ):
        raise ValueError("Actual profile solve differs from accepted warm")
    if not all(
        result["numerical_checks"][key] is True
        for key in ("shape_match", "finite", "exact", "status_exact")
    ):
        raise ValueError("Actual profile numerical comparison failed")
    saved = arrays(directory, sample["snapshot"], sample["arrays"])
    return (
        result,
        cohort,
        dict(
            cohort=evidence,
            profile_arrays=saved,
            original_uninstrumented_timings=timings,
        ),
    )


def validate_launch(profile, metrics, result, cohort, review_path):
    """Bind the actual command and one launch to reviewed source/geometry."""
    request = read_json(profile / "request.json")
    command = read_json(profile / "command.json")
    script = SCRIPT.with_name("placement_profile.py")
    if (
        request["action"] != "profile"
        or request["target"] != "script"
        or (
            Path(request["script"]).resolve() != script
            or request["launch_skip"] != 0
            or request["launch_count"] != 1
            or request["sha256"].lower() != command["source_sha256"].lower()
            or request["sha256"].lower() != result["profile_source_sha256"]
        )
    ):
        raise ValueError("Expected one reviewed placement-profile launch")
    source_review = None
    if core.digest(script) != result["profile_source_sha256"]:
        if review_path is None:
            raise ValueError(
                "Historical script requires its independent source review"
            )
        review = read_json(review_path)
        if (
            review["status"] != "PASS"
            or Path(review["source"]).resolve() != script
            or (review["source_sha256"] != result["profile_source_sha256"])
        ):
            raise ValueError(
                "Independent historical source review does not match"
            )
        source_review = dict(
            path=str(review_path.resolve()), sha256=core.digest(review_path)
        )
    arguments = [
        "--cohort-dir",
        result["cohort_evidence"]["directory"],
        "--role",
        result["role"],
        "--execute",
    ]
    if request["arguments"] != arguments or request["output_flag"] != "--out":
        raise ValueError(
            "Requested script arguments differ from completed solve"
        )
    expected = [
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--kernel-name-base",
        "function",
        "--kernel-name",
        request["kernel_filter"],
        "--launch-skip",
        "0",
        "--launch-count",
        "1",
    ]
    for section in request["sections"]:
        expected += ["--section", section]
    if request["metrics"]:
        expected += ["--metrics", ",".join(request["metrics"])]
    expected += [
        "--csv",
        "--log-file",
        str(profile / "diagnostic.csv"),
        "--export",
        str(profile / "profile"),
        sys.executable,
        request["script"],
    ]
    expected += arguments + ["--out", str(profile / "benchmark")]
    if command["arguments"] != expected or (
        command["working_directory"] != request["_runtime_tree"]
        or command["pythonpath"] != request["_pythonpath"]
    ):
        raise ValueError("Actual NCU/script command or import roots differ")
    compiled = result["compile"]
    actual = compiled["actual_pinned_geometry"]
    n_runs = cohort["manifest"]["protocol"]["n_runs"]
    fixed = {
        "launch__block_dim_x": ("block", 1),
        "launch__block_dim_y": ("block", actual["blocksize"]),
        "launch__block_dim_z": ("block", 1),
        "launch__block_size": ("", actual["blocksize"]),
        "launch__grid_dim_x": ("", n_runs // actual["blocksize"]),
        "launch__grid_dim_y": ("", 1),
        "launch__grid_dim_z": ("", 1),
        "launch__thread_count": ("thread", n_runs),
        "launch__registers_per_thread": (
            "register/thread",
            compiled["registers_per_thread"],
        ),
        "launch__shared_mem_per_block_dynamic": (
            "byte/block",
            actual["dynshared"],
        ),
        "launch__sm_count": ("SM", compiled["multiprocessor_count"]),
    }
    for name, (unit, value) in fixed.items():
        scale = 1
        if name == "launch__shared_mem_per_block_dynamic":
            unit = metrics[name]["unit"]
            scales = {"byte/block": 1, "Kbyte/block": 1000}
            if unit not in scales:
                raise ValueError("Unknown dynamic shared memory unit")
            scale = scales[unit]
        if core.quantity(metrics, name, unit) * scale != value:
            raise ValueError(f"Profile launch differs at {name}")
    if metrics["CC"]["raw"] != ".".join(
        map(str, compiled["compute_capability"])
    ):
        raise ValueError("Profile architecture differs")
    if (
        min(
            core.quantity(metrics, f"launch__occupancy_limit_{name}", "block")
            for name in ("blocks", "registers", "shared_mem", "warps")
        )
        != actual["blocks_per_sm"]
    ):
        raise ValueError("Profile occupancy differs from compiled residency")
    return dict(
        request_sha256=core.digest(profile / "request.json"),
        command_sha256=core.digest(profile / "command.json"),
        profile_report_sha256=core.digest(profile / "profile.ncu-rep"),
        profile_source_sha256=result["profile_source_sha256"],
        historical_source_review=source_review,
        exact_geometry_fields=list(fixed),
        dynamic_shared_byte_conversion=dict(
            raw=metrics["launch__shared_mem_per_block_dynamic"],
            exact_bytes=actual["dynshared"],
            interpretation="Decimal Kbyte/block equals 1000 byte/block",
        ),
        exact_waves_from_compiled_geometry=actual["waves"],
        rounded_waves_metric=metrics["launch__waves_per_multiprocessor"],
    )


def analyze(args):
    """Analyze one saved placement profile without compiling or launching."""
    profile = args.profile_dir.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result, cohort, evidence = validate_profile(profile)
    metrics = core.read_metrics(profile / "metrics.csv")
    launch = validate_launch(
        profile, metrics, result, cohort, args.source_review
    )
    exports = core.validate_exports(profile, args.source_counters, output)
    kernel_name, rows = core.source_rows(args.source_counters)
    compiled = result["compile"]
    native, symbols = core.disassemble(
        Path(compiled["artifacts"]["cubin"]),
        args.nvdisasm,
        output,
        compiled["entry_name"],
    )
    base = core.match_native(rows, native, symbols)
    analysis = core.aggregate(rows, metrics)
    if core.digest(SCRIPT) != IMPORTED_SHA256 or (
        core.digest(core.SCRIPT) != IMPORTED_CORE_SHA256
    ):
        raise ValueError("Analysis source changed while auditing")
    analysis.update(
        schema_version=1,
        kind="saved_placement_profile_analysis",
        status="ok",
        tool_sha256=core.digest(SCRIPT),
        core_tool_sha256=core.digest(core.SCRIPT),
        profile_directory=str(profile),
        metrics_sha256=core.digest(profile / "metrics.csv"),
        source_counters_sha256=core.digest(args.source_counters),
        benchmark_result_sha256=core.digest(
            profile / "benchmark" / "result.json"
        ),
        role=result["role"],
        manifest=cohort["manifest"],
        duration=cohort["duration"],
        n_runs=cohort["manifest"]["protocol"]["n_runs"],
        construction=result["construction"],
        resources=compiled,
        evidence=evidence,
        launch_identity=launch,
        report_export_identity=exports,
        source_kernel_name=kernel_name,
        runtime_base_address=base,
        metrics=metrics,
        limitations=[
            "Exact source executions and periodic sampled stalls "
            "remain distinct.",
            "Executed bytes are a whole-grid union, "
            "not temporal cache working set.",
            "Profile durations cannot replace original "
            "ordinary timing samples.",
            "Actual shared configuration applies to "
            "this profiled launch only.",
            "Native buffer placement may change instruction work "
            "and register spills.",
            "Traffic counters do not isolate one buffer "
            "without source attribution.",
            "Hardware/source residuals are unresolved; "
            "no fitted correction is applied.",
        ],
    )
    with (output / "per_pc.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    core.write_json(output / "analysis.json", analysis)
    print(
        json.dumps(
            dict(
                status="ok",
                output=str(output),
                footprint=analysis["footprint"],
                exact=analysis["exact_totals"],
            )
        )
    )


def main():
    """Import saved reports and disassemble saved binaries on the CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--source-counters", type=Path, required=True)
    parser.add_argument("--nvdisasm", type=Path, required=True)
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
