"""Compare explicit shared carveout preferences on one unchanged cubin.

--cohort-dir is a completed placement cohort. Only its local baseline is
used. Default execution is host preparation with zero native overloads.
--execute collects two blocks of six mirrored samples per control8,
shared64 and control8_repeat slot, using the original duration and grid.
The two control slots use the same Solver, not independent compilers.
--execute --profile-arm control8|shared64 --ordinary-dir PREVIOUS performs
one state-only solve for an external profiler, matching a completed
ordinary contrast. It never supplies ordinary timing samples.

All source/input/native/geometry/state gates remain exact. The only
treatment is CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT on the
same loaded CUfunction. Requested percentages derive from the queried
maximum shared memory per SM. They are preferences, not proof of actual
ordinary-launch configuration. Separate NCU profiles must show 8192 and
65536 bytes respectively before interpreting a carveout contrast.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from cuda.bindings import driver

from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cuda_backend import IS_MLIR
from cubie.cuda_simsafe import cuda

import placement_profile as original
import placement_probe as matched


SCRIPT = Path(__file__).resolve()
ARMS = ("control8", "shared64", "control8_repeat")
TARGET_BYTES = dict(control8=8192, shared64=65536, control8_repeat=8192)
COMPILE_FIELDS = (
    "source_hash", "compiler_identity", "config_hash", "cubin_sha256",
    "entry_name", "requested_blocksize", "registers_per_thread",
    "local_bytes_per_thread", "shared_bytes_per_run",
    "multiprocessor_count", "compute_capability",
)
CONSTRUCTION_FIELDS = (
    "role", "location", "buffers", "resolved_buffers", "config_hash",
    "inputs", "workload", "compilation_check",
)


def require_equal(expected, current, fields, label, receipt):
    """Retain every exact difference before rejecting drift."""
    differences = original.compare_fields(expected, current, fields)
    receipt[label] = differences
    if differences:
        raise ValueError(f"{label}: {json.dumps(differences)}")


def read_preference(function):
    """Read the driver preference, never label it actual shared capacity."""
    attribute = driver.CUfunction_attribute
    status, value = driver.cuFuncGetAttribute(
        attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        function.handle,
    )
    if status.value != 0:
        raise RuntimeError(f"cuFuncGetAttribute failed: {status}")
    return int(value)


def get_function(solver):
    """Require the one existing native specialization and loaded function."""
    dispatcher = solver.kernel.kernel
    if len(dispatcher.overloads) != 1:
        raise ValueError("Expected one existing native specialization")
    if dispatcher.targetoptions.get("shared_memory_carveout") is not None:
        raise ValueError("Dispatcher would override the driver treatment")
    (compiled,) = dispatcher.overloads.values()
    return compiled._codelibrary.get_cufunc()


def verify_native(solver, function, compiled, receipt):
    """Check loaded handle, actual cubin bytes, resources and pinned grid."""
    current_function = get_function(solver)
    (kernel,) = solver.kernel.kernel.overloads.values()
    cubin, entry = matched._compiled_cubin(kernel)
    registers, frame = matched.pl.kernel_resources(solver)
    geometry = matched.pl.launch_geometry(
        solver, compiled["requested_blocksize"], receipt["n_runs"]
    )
    observed = dict(
        cubin_sha256=hashlib.sha256(cubin).hexdigest(), entry_name=entry,
        registers_per_thread=registers, local_bytes_per_thread=frame,
        handle_unchanged=current_function.handle == function.handle,
        function_handle=str(function.handle),
        native_overloads=len(solver.kernel.kernel.overloads),
        geometry=geometry,
    )
    if not observed["handle_unchanged"]:
        raise ValueError("Loaded CUfunction changed during treatment")
    require_equal(
        compiled, observed,
        ("cubin_sha256", "entry_name", "registers_per_thread",
         "local_bytes_per_thread"), "native_differences", observed,
    )
    require_equal(
        compiled["actual_pinned_geometry"], geometry or {},
        original.GEOMETRY_FIELDS, "geometry_differences", observed,
    )
    if geometry["blocks_per_sm"] <= 0 or geometry["waves"] < 2:
        raise ValueError("Treatment fails two-wave occupancy requirement")
    return observed


def apply_arm(solver, function, compiled, arm, receipt):
    """Set only the function preference and recheck binary/geometry."""
    maximum = receipt["maximum_shared_bytes_per_sm"]
    before = verify_native(solver, function, compiled, receipt)
    previous = read_preference(function)
    percentage = math.ceil(100 * TARGET_BYTES[arm] / maximum)
    if not 0 <= percentage <= 100:
        raise ValueError("Target shared capacity exceeds queried maximum")
    function.set_shared_memory_carveout(percentage)
    readback = read_preference(function)
    if readback != percentage:
        raise ValueError("Function carveout preference readback differs")
    result = dict(
        arm=arm, target_shared_bytes=TARGET_BYTES[arm],
        preference_percent=percentage, readback_percent=readback,
        previous_preference_percent=previous,
        actual_shared_bytes=None,
        before_native=before,
        native=verify_native(solver, function, compiled, receipt),
    )
    return result


def sample(solver, inputs, job, warm, function, compiled, arm,
           phase, block, index, receipt, output, snapshot):
    """Retain raw solve evidence and reject any failed numerical gate."""
    key = f"b{block}-{phase}-{index}-{arm}"
    treatment = apply_arm(solver, function, compiled, arm, receipt)
    row = matched.solve_sample(
        solver, inputs, job,
        dict(duration=receipt["duration"], snapshot=snapshot, sample_id=key),
    )
    row.update(
        key=key, arm=arm, phase=phase, block=block, sample=index,
        treatment=treatment, cubin_sha256=compiled["cubin_sha256"],
        profile=receipt["profile_arm"] is not None,
        manifest_sha256=receipt["manifest_sha256"],
    )
    if snapshot:
        row["numerical_checks"] = matched.compare_snapshots(warm, row)
        row["array_identity_matches"] = row["arrays"] == warm["arrays"]
    with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    receipt["completed_solve_count"] += 1
    if (
        row["chunks"] != 1 or row["status_hist"]["failed"]
        or not row["finite_state"] or not np.isfinite(row["kernel_ms"])
        or row["kernel_ms"] <= 0
    ):
        raise ValueError(f"Invalid retained solve {key}")
    if snapshot and not (
        row["array_identity_matches"]
        and all(row["numerical_checks"].get(field) is True
                for field in original.NUMERICAL_FIELDS)
    ):
        raise ValueError(f"Exact original state/status differs: {key}")
    if phase == "measurement" and row["kernel_ms"] < 20:
        raise ValueError(f"Retained sample shorter than 20ms: {key}")
    after = read_preference(function)
    if after != treatment["preference_percent"]:
        raise ValueError("Launch changed requested carveout preference")
    return row


def collect(solver, inputs, job, warm, function, compiled, receipt, output):
    """Collect one profile solve or contemporaneous repeated ordinary arms."""
    profile_arm = receipt["profile_arm"]
    if profile_arm:
        row = sample(
            solver, inputs, job, warm, function, compiled, profile_arm,
            "profile_once", 0, 0, receipt, output, True,
        )
        receipt["profile_sample"] = row
        receipt["status"] = "profile_complete_actual_carveout_unverified"
        return
    receipt["blocks"] = []
    for block in range(2):
        current = dict(block=block, warm=[], measurements=[])
        receipt["blocks"].append(current)
        for arm in ARMS:
            current["warm"].append(sample(
                solver, inputs, job, warm, function, compiled, arm,
                "warm", block, 0, receipt, output, True,
            ))
        started = time.perf_counter()
        index = 0
        while time.perf_counter() - started < matched.pl.SETTLE_S:
            for arm in ARMS:
                sample(
                    solver, inputs, job, warm, function, compiled, arm,
                    "settle", block, index, receipt, output, False,
                )
            index += 1
        for index in range(6):
            order = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                row = sample(
                    solver, inputs, job, warm, function, compiled, arm,
                    "measurement", block, index, receipt, output, False,
                )
                current["measurements"].append(row)
    receipt["status"] = "ordinary_complete_actual_carveout_unverified"


def validate_native_receipt(native, compiled, handle):
    """Check saved loaded-binary and launch identities without a device."""
    fields = (
        "cubin_sha256", "entry_name", "registers_per_thread",
        "local_bytes_per_thread",
    )
    if (
        native["native_overloads"] != 1
        or native["handle_unchanged"] is not True
        or native["function_handle"] != handle or not handle
        or native["native_differences"] != {}
        or native["geometry_differences"] != {}
        or any(native[key] != compiled[key] for key in fields)
    ):
        raise ValueError("Retained native identity differs")
    geometry = native["geometry"]
    expected = compiled["actual_pinned_geometry"]
    if (
        any(geometry[key] != expected[key]
            for key in original.GEOMETRY_FIELDS)
        or geometry["blocks_per_sm"] <= 0
        or not np.isfinite(geometry["waves"]) or geometry["waves"] < 2
    ):
        raise ValueError("Retained treatment geometry differs")


def validate_treatment_receipt(row, ordinary, handle, previous):
    """Check the requested arm, exact setter/getter and binary evidence."""
    arm = row["arm"]
    treatment = row["treatment"]
    maximum = ordinary["maximum_shared_bytes_per_sm"]
    if arm not in ARMS or maximum <= 0:
        raise ValueError("Retained arm or hardware maximum is invalid")
    percentage = math.ceil(100 * TARGET_BYTES[arm] / maximum)
    if (
        not 0 <= percentage <= 100 or treatment["arm"] != arm
        or treatment["target_shared_bytes"] != TARGET_BYTES[arm]
        or treatment["preference_percent"] != percentage
        or treatment["readback_percent"] != percentage
        or treatment["previous_preference_percent"] != previous
        or treatment["actual_shared_bytes"] is not None
    ):
        raise ValueError("Retained treatment setter/getter differs")
    for key in ("before_native", "native"):
        validate_native_receipt(treatment[key], ordinary["compile"], handle)
    return percentage


def load_ordinary(directory, warm):
    """Require the completed ordinary protocol and its retained artifacts."""
    path = directory / "result.json"
    value = json.loads(path.read_text())
    if (
        value["status"] != "ordinary_complete_actual_carveout_unverified"
        or value["profile_arm"] is not None
        or value["source_sha256"] != original.file_hash(SCRIPT)
        or value["profile_helper_sha256"] != original.file_hash(original.SCRIPT)
        or value["cleanup_errors"] or not value["preference_restored"]
        or not value["solver_closed"]
        or value["cache_root_override_restored"] is not True
    ):
        raise ValueError("Prior ordinary contrast is not complete/identical")
    rows_path = directory / "samples.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    by_key = {row["key"]: row for row in rows}
    if len(by_key) != len(rows) or len(rows) != value["completed_solve_count"]:
        raise ValueError("Prior ordinary raw row membership differs")
    measurements = []
    if [block["block"] for block in value["blocks"]] != [0, 1]:
        raise ValueError("Prior ordinary paired blocks differ")
    for block in value["blocks"]:
        if [row["arm"] for row in block["warm"]] != list(ARMS):
            raise ValueError("Prior ordinary warm membership differs")
        for row in block["warm"]:
            if (by_key.get(row["key"]) != row
                    or row["phase"] != "warm"
                    or row["block"] != block["block"]):
                raise ValueError("Prior warm receipt differs from raw row")
            snapshot = original.cohort_path(directory, row["snapshot"])
            original.validate_arrays(snapshot, row["arrays"])
            checks = matched.compare_snapshots(warm, row)
            if not all(checks.get(key) is True
                       for key in original.NUMERICAL_FIELDS):
                raise ValueError("Prior warm arrays differ from cohort")
        expected_order = [
            (index, arm) for index in range(6)
            for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS)))
        ]
        if [(row["sample"], row["arm"])
                for row in block["measurements"]] != expected_order:
            raise ValueError("Prior ordinary mirrored samples differ")
        for row in block["measurements"]:
            if (by_key.get(row["key"]) != row or row["kernel_ms"] < 20
                    or row["phase"] != "measurement"
                    or row["block"] != block["block"]):
                raise ValueError("Prior ordinary sample invalid or too short")
            measurements.append(row["key"])
    if sorted(measurements) != sorted(
        row["key"] for row in rows if row["phase"] == "measurement"
    ):
        raise ValueError("Prior ordinary measurements differ from raw rows")
    handle = value["final_native_check"]["function_handle"]
    previous = value["initial_preference_percent"]
    for row in rows:
        if (
            row["profile"] or not row["finite_state"]
            or row["status_hist"]["failed"] or row["chunks"] != 1
            or row["duration"] != value["duration"]
            or row["n_runs"] != value["n_runs"]
            or row["cubin_sha256"] != value["compile"]["cubin_sha256"]
            or row["manifest_sha256"] != value["manifest_sha256"]
            or not np.isfinite(row["kernel_ms"]) or row["kernel_ms"] <= 0
            or row["phase"] not in ("warm", "settle", "measurement")
            or row["block"] not in (0, 1)
            or row["blocksize"] != value["compile"]["requested_blocksize"]
        ):
            raise ValueError("Prior ordinary raw solve is ineligible")
        previous = validate_treatment_receipt(row, value, handle, previous)
    validate_native_receipt(value["final_native_check"], value["compile"],
                            handle)
    restored = value["preference_restore"]
    if (
        restored["previous_percent"] != previous
        or restored["requested_percent"] != value["initial_preference_percent"]
        or restored["readback_percent"] != value["initial_preference_percent"]
    ):
        raise ValueError("Prior original preference restoration differs")
    for key in ("before_native", "after_native"):
        validate_native_receipt(restored[key], value["compile"], handle)
    artifacts = original.artifact_receipts(directory, value["compile"])
    if artifacts != value["artifacts"]:
        raise ValueError("Prior ordinary native artifacts changed")
    evidence = dict(
        directory=str(directory), result_sha256=original.file_hash(path),
        samples_sha256=original.file_hash(rows_path), artifacts=artifacts,
        measurement_keys=measurements,
    )
    return value, evidence


def run(args):
    """Prepare one baseline and restore its original process-local state."""
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    solver = None
    function = None
    previous_preference = None
    previous_cache = get_cache_root_override()
    receipt = dict(
        schema_version=1, kind="same_cubin_carveout_contrast",
        status="validating", source_sha256=original.file_hash(SCRIPT),
        profile_helper_sha256=original.file_hash(original.SCRIPT),
        profile_arm=args.profile_arm, gpu_execution=False,
        kernel_compilation=False, completed_solve_count=0,
        ordinary_actual_carveout_observed=False,
        eligibility="No carveout-causal claim without matched NCU evidence",
        timing_use=("Profile times diagnostic only" if args.profile_arm
                    else "Raw preference-controlled ordinary samples"),
    )
    try:
        if not IS_MLIR:
            raise ValueError("This instrument requires the verified MLIR API")
        cohort, warm, evidence = original.load_cohort(
            args.cohort_dir.resolve(), "baseline"
        )
        receipt["cohort_evidence"] = evidence
        ordinary = None
        if args.profile_arm:
            if args.ordinary_dir is None:
                raise ValueError("Profile mode requires --ordinary-dir")
            ordinary, receipt["ordinary_evidence"] = load_ordinary(
                args.ordinary_dir.resolve(), warm
            )
            if ordinary["cohort_evidence"] != evidence:
                raise ValueError("Prior ordinary cohort identity differs")
        manifest = cohort["manifest"]
        current = matched.manifest_for(
            manifest["case"], manifest["unroll_policy"], manifest["cohort"],
            manifest["protocol"]["paired_blocks"],
        )
        require_equal(manifest, current, set(manifest) | set(current),
                      "environment_differences", receipt)
        receipt.update(
            manifest_sha256=cohort["manifest_sha256"],
            n_runs=manifest["protocol"]["n_runs"],
            duration=cohort["duration"],
            protocol=dict(blocks=2, samples_per_slot_per_block=6,
                          slots=list(ARMS), targets_bytes=TARGET_BYTES,
                          minimum_kernel_ms=20,
                          settle_s=matched.pl.SETTLE_S),
        )
        expected = cohort["construction"]["baseline"]
        private, seed = original.reproduce_generated_source(
            args.cohort_dir.resolve(), expected["generated_source"], output
        )
        receipt["generated_source_reproduction"] = seed
        set_cache_root(private)
        job = dict(role="baseline", manifest=manifest, output=str(output))
        solver, inputs, construction = matched.construct(job)
        receipt["construction"] = construction
        require_equal(expected, construction, CONSTRUCTION_FIELDS,
                      "construction_differences", receipt)
        require_equal(expected["generated_source"],
                      construction["generated_source"], ("sha256", "fn_hash"),
                      "generated_source_differences", receipt)
        if construction["compilation_check"]["native_overloads"] != 0:
            raise ValueError("Host preparation unexpectedly compiled")
        if not args.execute:
            receipt["status"] = "prepared"
            return
        receipt["kernel_compilation"] = True
        compiled = matched.compile_solver(solver, inputs, job)
        receipt["compile"] = compiled
        require_equal(cohort["compiles"]["baseline"], compiled,
                      COMPILE_FIELDS, "compile_differences", receipt)
        if ordinary is not None:
            require_equal(ordinary["compile"], compiled, COMPILE_FIELDS,
                          "ordinary_compile_differences", receipt)
        require_equal(
            cohort["compiles"]["baseline"]["actual_pinned_geometry"],
            compiled["actual_pinned_geometry"] or {},
            original.GEOMETRY_FIELDS, "geometry_differences", receipt,
        )
        if not original.geometry_valid(compiled, manifest["requested_blocksize"]):
            raise ValueError("Original pinned geometry is not valid")
        artifacts = original.artifact_receipts(output, compiled)
        receipt["artifacts"] = artifacts
        receipt["artifact_content_matches"] = {
            key: value["content_sha256"]
            == evidence["artifacts"][key]["content_sha256"]
            for key, value in artifacts.items()
        }
        if not all(receipt["artifact_content_matches"].values()):
            raise ValueError("Native artifact content differs from baseline")
        source = construction["generated_source"]
        if original.file_hash(source["path"]) != source["sha256"]:
            raise ValueError("Generated source changed during compilation")
        function = get_function(solver)
        previous_preference = read_preference(function)
        receipt["initial_preference_percent"] = previous_preference
        device = cuda.get_current_device()
        receipt["maximum_shared_bytes_per_sm"] = int(
            device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        )
        receipt["gpu_execution"] = True
        collect(solver, inputs, job, warm, function, compiled, receipt, output)
        receipt["final_native_check"] = verify_native(
            solver, function, compiled, receipt
        )
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        unwinding = sys.exc_info()[0] is not None
        errors = []
        if function is not None and previous_preference is not None:
            restoration = dict(requested_percent=previous_preference)
            receipt["preference_restore"] = restoration
            try:
                restoration["previous_percent"] = read_preference(function)
                restoration["before_native"] = verify_native(
                    solver, function, compiled, receipt
                )
            except Exception as error:
                errors.append(f"before restore: {type(error).__name__}: {error}")
            try:
                function.set_shared_memory_carveout(previous_preference)
                restoration["readback_percent"] = read_preference(function)
                if restoration["readback_percent"] != previous_preference:
                    raise ValueError("Original preference not restored")
                receipt["preference_restored"] = True
            except Exception as error:
                errors.append(f"preference: {type(error).__name__}: {error}")
            try:
                restoration["after_native"] = verify_native(
                    solver, function, compiled, receipt
                )
            except Exception as error:
                errors.append(f"after restore: {type(error).__name__}: {error}")
        if solver is not None:
            try:
                solver.close()
                receipt["solver_closed"] = True
            except Exception as error:
                errors.append(f"solver: {type(error).__name__}: {error}")
        set_cache_root(previous_cache)
        receipt["cache_root_override_restored"] = (
            get_cache_root_override() == previous_cache
        )
        receipt["cleanup_errors"] = errors
        if errors:
            receipt["status"] = "failed_cleanup"
        matched.write_json(output / "result.json", receipt)
        if errors and not unwinding:
            raise RuntimeError(str(errors))


def main():
    """Prepare or execute the fixed same-cubin preference contrast."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile-arm", choices=("control8", "shared64"))
    parser.add_argument("--ordinary-dir", type=Path)
    args = parser.parse_args()
    run(args)
    print(f"Carveout receipt written: {args.out.resolve()}")


if __name__ == "__main__":
    main()
