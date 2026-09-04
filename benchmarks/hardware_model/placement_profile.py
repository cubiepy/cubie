"""Profile one exact role from a completed matched placement cohort.

Run this script with the original frozen source/harness on PYTHONPATH.
--cohort-dir selects the completed placement_probe output; --role is
baseline or shared; --out must be fresh. The default only validates raw
cohort evidence and constructs one host Solver, with zero native
overloads. --execute additionally recompiles, verifies native artifacts
and geometry, then performs exactly one state-only solve for Nsight
Compute. Its raw state/status must equal accepted warm block zero.

Profiled kernel/wall times are diagnostic records, never ordinary timing
samples or a ranking. No environment, unroll, placement, solver, duration,
run count, or geometry override is accepted by this instrument.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from cubie.cache_root import get_cache_root_override, set_cache_root

import placement_probe as matched


SCRIPT = Path(__file__).resolve()
GEOMETRY_FIELDS = (
    "blocksize",
    "dynshared",
    "bytes_per_run",
    "blocks_per_sm",
    "resident_threads",
    "waves",
)
NUMERICAL_FIELDS = ("shape_match", "finite", "exact", "status_exact")


def file_hash(path):
    """Hash the actual retained file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort_path(directory, value):
    """Require referenced artifacts to remain in the selected cohort."""
    path = Path(value).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"Missing or out-of-cohort artifact: {path}")
    return path


def artifact_receipts(directory, compiled):
    """Retain exact cubin/PTX bytes and decoded SASS content identities."""
    receipts = {}
    for name in ("cubin", "ptx", "sass"):
        path = cohort_path(directory, compiled["artifacts"][name])
        data = (
            gzip.decompress(path.read_bytes())
            if name == "sass"
            else (path.read_bytes())
        )
        receipts[name] = dict(
            path=str(path),
            file_sha256=file_hash(path),
            content_sha256=hashlib.sha256(data).hexdigest(),
            content_bytes=len(data),
        )
    if receipts["cubin"]["content_sha256"] != compiled["cubin_sha256"]:
        raise ValueError("Original cubin bytes differ from compile receipt")
    return receipts


def validate_arrays(path, expected):
    """Verify every saved array against its recorded shape/dtype/hash."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError(f"Unexpected NPZ array membership: {path}")
        actual = {
            name: matched.array_receipt(archive[name]) for name in expected
        }
        if actual != expected:
            raise ValueError(f"Saved raw arrays differ from receipt: {path}")
        if "state" in archive.files and not np.all(
            np.isfinite(archive["state"])
        ):
            raise ValueError(f"Nonfinite original warm state: {path}")
    return dict(path=str(path), file_sha256=file_hash(path), arrays=actual)


def geometry_valid(compiled, blocksize):
    """Require the exact original/pinned geometry and two full waves."""
    if compiled["blocked_reasons"]:
        return False
    requested = compiled["requested_geometry"]
    actual = compiled["actual_pinned_geometry"]
    limited = compiled["production_limited_geometry"]
    return bool(
        requested
        and actual
        and limited
        and compiled["requested_blocksize"] == blocksize
        and limited["blocksize"] == blocksize
        and actual["blocksize"] == blocksize
        and actual["blocks_per_sm"] > 0
        and actual["waves"] >= 2
        and all(actual[key] == requested[key] for key in GEOMETRY_FIELDS)
    )


def validate_timing(row, manifest, result):
    """Check cohort eligibility without deriving a performance ranking."""
    role = row["role"]
    if (
        role not in matched.ROLES
        or row["duration"] != result["duration"]
        or row["attempt"] != result["accepted_attempt"]
        or row["n_runs"] != manifest["protocol"]["n_runs"]
        or row["blocksize"] != manifest["requested_blocksize"]
        or row["chunks"] != 1
        or row["status_hist"]["failed"]
        or row["finite_state"] is not True
        or row["manifest_sha256"] != result["manifest_sha256"]
        or row["source_hash"] != manifest["source_and_compiler"]["source_hash"]
        or row["cubin_sha256"] != result["compiles"][role]["cubin_sha256"]
        or not np.isfinite(row["kernel_ms"])
        or row["kernel_ms"] <= 0
    ):
        raise ValueError(f"Ineligible original timing row: {row['key']}")
    if (
        row["phase"] in ("pilot", "measurement")
        and row["kernel_ms"] < (manifest["protocol"]["minimum_kernel_ms"])
    ):
        raise ValueError(f"Short original accepted sample: {row['key']}")


def load_cohort(directory, role):
    """Admit only a completed immutable cohort with matching raw evidence."""
    result_path = directory / "result.json"
    manifest_path = directory / "manifest.json"
    result = json.loads(result_path.read_text())
    wrapper = json.loads(manifest_path.read_text())
    manifest = result["manifest"]
    if result["status"] != "complete" or wrapper != {
        "manifest": manifest,
        "manifest_sha256": result["manifest_sha256"],
    }:
        raise ValueError("Expected a completed matched placement cohort")
    if matched.digest(manifest) != result["manifest_sha256"]:
        raise ValueError("Cohort manifest digest mismatch")
    if file_hash(matched.SCRIPT) != manifest["probe_source_sha256"]:
        raise ValueError("Imported placement_probe differs from original")
    if manifest["kind"] != "matched_placement_cohort":
        raise ValueError("Unsupported cohort kind")
    protocol = manifest["protocol"]
    if protocol["samples_per_role_per_block"] < 6 or (
        protocol["minimum_kernel_ms"] < 20
    ):
        raise ValueError("Original cohort lacks the required timing protocol")
    for original_role in matched.ROLES:
        cleanup = result["worker_cleanup"][original_role]
        if (
            cleanup["errors"]
            or cleanup["forced_shutdown"]
            or (cleanup["returncode"] != 0)
        ):
            raise ValueError("Original worker cleanup did not complete")
        compiled = result["compiles"][original_role]
        if not geometry_valid(compiled, manifest["requested_blocksize"]):
            raise ValueError("Original compiled geometry is ineligible")
        if (
            compiled["source_hash"]
            != manifest["source_and_compiler"]["source_hash"]
            or compiled["compiler_identity"]
            != manifest["source_and_compiler"]["compiler_identity"]
        ):
            raise ValueError("Original compile identity differs from cohort")
    if (
        result["compiles"]["baseline"]["cubin_sha256"]
        != result["compiles"]["duplicate"]["cubin_sha256"]
    ):
        raise ValueError("Original duplicate baseline cubin differs")
    selected = [
        attempt
        for attempt in result["attempts"]
        if attempt["attempt"] == result["accepted_attempt"]
    ]
    if (
        len(selected) != 1
        or selected[0]["status"] != "complete"
        or (selected[0]["duration"] != result["duration"])
    ):
        raise ValueError("Original accepted attempt is not complete")
    attempt = selected[0]
    if {block["block"] for block in attempt["blocks"]} != set(
        range(protocol["paired_blocks"])
    ) or len(attempt["blocks"]) != protocol["paired_blocks"]:
        raise ValueError("Original accepted paired blocks are incomplete")
    timings_path = cohort_path(directory, result["timing_records"])
    rows = [json.loads(line) for line in timings_path.read_text().splitlines()]
    by_key = {row["key"]: row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Original timing keys are not unique")
    accepted_rows = [
        row for row in rows if row["attempt"] == result["accepted_attempt"]
    ]
    for row in accepted_rows:
        validate_timing(row, manifest, result)
    pilots = [row for row in accepted_rows if row["phase"] == "pilot"]
    if Counter(row["role"] for row in pilots) != Counter(matched.ROLES):
        raise ValueError("Original accepted pilots are incomplete")
    measurement_keys = []
    warm_receipts = []
    for block in attempt["blocks"]:
        keys = block["measurement_keys"]
        measurements = [by_key[key] for key in keys]
        expected = [
            (index, original_role)
            for index in range(protocol["samples_per_role_per_block"])
            for original_role in (
                ("baseline", "shared", "duplicate")
                if index % 2 == 0
                else ("duplicate", "shared", "baseline")
            )
        ]
        if [(row["sample"], row["role"]) for row in measurements] != expected:
            raise ValueError("Original measurements violate mirrored protocol")
        if any(
            row["phase"] != "measurement"
            or row["paired_block"] != (block["block"])
            for row in measurements
        ):
            raise ValueError("Original measurement block identity mismatch")
        measurement_keys.extend(keys)
        for original_role in matched.ROLES:
            warm = block["warm"][original_role]
            if by_key.get(warm["key"]) != warm:
                raise ValueError("Original warm receipt differs from raw row")
            checks = block["numerical_checks"][original_role]
            if not all(checks[key] is True for key in NUMERICAL_FIELDS):
                raise ValueError("Original warm numerical check failed")
            path = cohort_path(directory, warm["snapshot"])
            warm_receipts.append(validate_arrays(path, warm["arrays"]))
    if Counter(measurement_keys) != Counter(
        row["key"] for row in accepted_rows if row["phase"] == "measurement"
    ):
        raise ValueError("Accepted measurement membership differs from rows")
    original = result["construction"][role]
    input_path = cohort_path(directory, directory / f"{role}-inputs.npz")
    inputs = validate_arrays(input_path, original["inputs"])
    warm = next(block for block in attempt["blocks"] if block["block"] == 0)[
        "warm"
    ][role]
    receipt = dict(
        directory=str(directory),
        role=role,
        result_sha256=file_hash(result_path),
        manifest_sha256=result["manifest_sha256"],
        manifest_file_sha256=file_hash(manifest_path),
        timings_sha256=file_hash(timings_path),
        accepted_attempt=result["accepted_attempt"],
        duration=result["duration"],
        measurement_keys=measurement_keys,
        original_warm_key=warm["key"],
        warm_arrays=warm_receipts,
        inputs=inputs,
        artifacts=artifact_receipts(directory, result["compiles"][role]),
    )
    return result, warm, receipt


def compare_fields(expected, current, fields):
    """Expose exact expected/current values for every failed identity gate."""
    return {
        key: dict(expected=expected.get(key), current=current.get(key))
        for key in fields
        if expected.get(key) != current.get(key)
    }


def reproduce_generated_source(directory, original, output):
    """Seed a private cache from the exact recorded generated source.

    Generated modules can retain helpers appended by earlier solver
    constructions. Preserve the complete recorded bytes, including such
    helpers. A content-addressed cohort sidecar is created exclusively
    from the original file only when its current bytes match the receipt.
    Existing snapshots are verified and never overwritten.
    """
    source = Path(original["path"]).resolve()
    expected = original["sha256"]
    relative = Path(source.parent.name) / source.name
    if source.suffix != ".py" or len(expected) != 64:
        raise ValueError("Invalid original generated-source receipt")
    snapshot = (
        directory / "generated_source_snapshots" / expected / relative
    )
    created = False
    if not snapshot.exists():
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(
                "Original generated-source bytes changed; no exact snapshot"
            )
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        try:
            with snapshot.open("xb") as stream:
                stream.write(data)
            created = True
        except FileExistsError:
            pass
    data = snapshot.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Generated-source snapshot differs from cohort")
    private_root = output / "generated_source_cache"
    private_source = private_root / relative
    private_source.parent.mkdir(parents=True, exist_ok=False)
    with private_source.open("xb") as stream:
        stream.write(data)
    if file_hash(private_source) != expected:
        raise ValueError("Private generated-source seed differs from cohort")
    return private_root, dict(
        original=original,
        snapshot_path=str(snapshot),
        snapshot_created=created,
        snapshot_sha256=expected,
        source_bytes=len(data),
        private_root=str(private_root),
        private_source=str(private_source),
        private_sha256=file_hash(private_source),
        policy="Exact full source bytes; existing snapshots never overwritten",
    )


def run(args):
    """Prepare or execute one exact profile, always releasing the Solver."""
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    solver = None
    previous_cache_root = get_cache_root_override()
    receipt = dict(
        schema_version=1,
        status="validating",
        role=args.role,
        profile_source_sha256=file_hash(SCRIPT),
        gpu_execution=False,
        kernel_compilation=False,
        requested_solve_count=1,
        completed_solve_count=0,
        timing_use="Profile times are diagnostic, not ordinary samples.",
    )
    try:
        cohort, warm, evidence = load_cohort(
            args.cohort_dir.resolve(), args.role
        )
        receipt["cohort_evidence"] = evidence
        manifest = cohort["manifest"]
        current = matched.manifest_for(
            manifest["case"],
            manifest["unroll_policy"],
            manifest["cohort"],
            manifest["protocol"]["paired_blocks"],
        )
        differences = compare_fields(
            manifest, current, set(manifest) | set(current)
        )
        receipt["environment_differences"] = differences
        if differences:
            raise ValueError(
                "Source/compiler/environment drift: " + json.dumps(differences)
            )
        original = cohort["construction"][args.role]
        private_cache, seed = reproduce_generated_source(
            args.cohort_dir.resolve(), original["generated_source"], output
        )
        receipt["generated_source_reproduction"] = seed
        set_cache_root(private_cache)
        job = dict(role=args.role, manifest=manifest, output=str(output))
        solver, inputs, construction = matched.construct(job)
        receipt["construction"] = construction
        differences = compare_fields(
            original,
            construction,
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
        )
        differences.update(
            {"generated_source": changed}
            if (
                changed := compare_fields(
                    original["generated_source"],
                    construction["generated_source"],
                    ("sha256", "fn_hash"),
                )
            )
            else {}
        )
        receipt["construction_differences"] = differences
        if differences:
            raise ValueError("Reconstructed placement/input identity differs")
        if construction["compilation_check"]["native_overloads"] != 0:
            raise ValueError("Host preparation unexpectedly compiled code")
        if not args.execute:
            receipt["status"] = "prepared"
            return
        receipt["kernel_compilation"] = True
        compiled = matched.compile_solver(solver, inputs, job)
        receipt["compile"] = compiled
        receipt["post_compile_generated_source_sha256"] = file_hash(
            construction["generated_source"]["path"]
        )
        if receipt["post_compile_generated_source_sha256"] != (
            original["generated_source"]["sha256"]
        ):
            raise ValueError("Generated source changed during compilation")
        differences = compare_fields(
            cohort["compiles"][args.role],
            compiled,
            (
                "source_hash",
                "compiler_identity",
                "config_hash",
                "cubin_sha256",
                "requested_blocksize",
                "registers_per_thread",
                "local_bytes_per_thread",
                "shared_bytes_per_run",
                "multiprocessor_count",
                "compute_capability",
            ),
        )
        expected_geometry = cohort["compiles"][args.role][
            "actual_pinned_geometry"
        ]
        actual_geometry = compiled["actual_pinned_geometry"] or {}
        receipt["geometry_differences"] = compare_fields(
            expected_geometry, actual_geometry, GEOMETRY_FIELDS
        )
        receipt["compile_differences"] = differences
        actual_artifacts = artifact_receipts(output, compiled)
        receipt["artifacts"] = actual_artifacts
        receipt["artifact_content_matches"] = {
            name: item["content_sha256"]
            == evidence["artifacts"][name]["content_sha256"]
            for name, item in actual_artifacts.items()
        }
        if (
            differences
            or receipt["geometry_differences"]
            or not geometry_valid(compiled, manifest["requested_blocksize"])
            or not all(receipt["artifact_content_matches"].values())
        ):
            raise ValueError(
                "Compiled native/artifact/geometry identity differs"
            )
        receipt["gpu_execution"] = True
        sample = matched.solve_sample(
            solver,
            inputs,
            job,
            dict(
                duration=cohort["duration"],
                snapshot=True,
                sample_id="profile_once",
            ),
        )
        receipt["completed_solve_count"] = 1
        receipt["sample"] = sample
        checks = matched.compare_snapshots(warm, sample)
        receipt["numerical_checks"] = checks
        receipt["array_identity_matches"] = sample["arrays"] == warm["arrays"]
        if (
            sample["chunks"] != 1
            or sample["status_hist"]["failed"]
            or not (
                sample["finite_state"]
                and receipt["array_identity_matches"]
                and all(checks.get(key) is True for key in NUMERICAL_FIELDS)
            )
        ):
            raise ValueError(
                "Profiled state/status differs from original warm solve"
            )
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        unwinding = sys.exc_info()[0] is not None
        cleanup_error = None
        if solver is not None:
            try:
                solver.close()
                receipt["solver_closed"] = True
            except Exception as error:
                cleanup_error = f"{type(error).__name__}: {error}"
                receipt["cleanup_error"] = cleanup_error
                receipt["status"] = "failed_cleanup"
        set_cache_root(previous_cache_root)
        receipt["cache_root_override_restored"] = (
            get_cache_root_override() == previous_cache_root
        )
        matched.write_json(output / "result.json", receipt)
        if cleanup_error and not unwinding:
            raise RuntimeError(cleanup_error)


def main():
    """Validate a completed placement role and optionally profile one solve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument(
        "--role", choices=("baseline", "shared"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(args)
    print(f"Placement profile receipt written: {args.out.resolve()}")


if __name__ == "__main__":
    main()
