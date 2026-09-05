"""Reserve unused dynamic shared bytes on one unchanged baseline cubin.

Default execution constructs one Solver without native compilation. An
explicit --execute is required for ordinary samples or one profile solve.
The reservation is the accepted stage_base shared layout at block64;
no device buffer location, code, preference, input or duration changes.
Actual carveout and occupancy require separate per-launch NCU evidence.
"""

import argparse
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from cuda.bindings import driver

from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cuda_backend import IS_MLIR
from cubie.cuda_simsafe import cuda

import carveout_probe as hints
import placement_profile as original
import placement_probe as matched
import solver_profile_analysis as analysis


SCRIPT = Path(__file__).resolve()
ARMS = ("baseline", "reserved", "baseline_repeat")
CASE = "chain32-kvaerno3-stage_base-bs64"
NO_SHARED_OPCODES = frozenset(
    "BRA BREAK BSSY BSYNC CALL CS2R DADD EXIT F2F FADD FFMA FMNMX FMUL "
    "FSEL FSETP IADD3 IMAD ISETP LDG LDL LEA LOP3 MOV MUFU NOP PLOP3 "
    "R2UR REDUX RET S2R SEL SHF STG STL UIADD3 UIMAD ULDC UMOV USHF "
    "VOTE VOTEU WARPSYNC".split()
)
STATUS_ORDINARY = "ordinary_complete_actual_carveout_unverified"


def helper_identity():
    """Bind every imported instrument and architecture-table implementation."""
    sys.path.append(matched.pl.NCU_PYTHON)
    occupancy = importlib.import_module("ncu_occupancy")
    paths = [
        SCRIPT,
        hints.SCRIPT,
        original.SCRIPT,
        matched.SCRIPT,
        analysis.SCRIPT,
        Path(occupancy.__file__),
        Path(occupancy._ncu_occupancy.__file__),
    ]
    for name in ("numba_cuda_mlir.descriptor", "numba_cuda_mlir.compiler"):
        paths.append(Path(importlib.import_module(name).__file__))
    return {str(path): original.file_hash(path) for path in paths}


def metric_integer(metrics, name, units):
    """Convert explicit decimal SI byte units without rounding."""
    row = metrics[name]
    if row["unit"] not in units or row["decimal"] is None:
        raise ValueError(f"Unsupported metric value or unit: {name}")
    number = Decimal(row["decimal"]) * units[row["unit"]]
    if number < 0 or number != int(number):
        raise ValueError(f"Nonintegral metric: {name}")
    return int(number)


def reservation_plan(cohort, evidence, path, expected_sha256):
    """Derive bytes from the accepted layout and reviewed shared profile."""
    if original.file_hash(path) != expected_sha256:
        raise ValueError("Reviewed shared-analysis bytes differ")
    saved = json.loads(path.read_text())
    profile = Path(saved["profile_directory"])
    metrics_path = profile / "metrics.csv"
    metrics = analysis.read_metrics(metrics_path)
    if (
        saved["status"] != "ok"
        or saved["role"] != "shared"
        or saved["metrics_sha256"] != original.file_hash(metrics_path)
        or saved["metrics"] != metrics
        or saved["benchmark_result_sha256"]
        != original.file_hash(profile / "benchmark" / "result.json")
        or saved["core_tool_sha256"] != original.file_hash(analysis.SCRIPT)
        or saved["manifest"] != cohort["manifest"]
        or saved["duration"] != cohort["duration"]
        or saved["evidence"]["cohort"]["result_sha256"]
        != evidence["result_sha256"]
    ):
        raise ValueError("Shared profile is not the accepted exact cohort")
    differences = original.compare_fields(
        cohort["compiles"]["shared"], saved["resources"], hints.COMPILE_FIELDS
    )
    if differences:
        raise ValueError(f"Shared profile resource identity: {differences}")
    manifest = cohort["manifest"]
    baseline = cohort["compiles"]["baseline"]
    shared = cohort["compiles"]["shared"]
    if (
        manifest["case"] != CASE
        or manifest["unroll_policy"] != "u11111111"
        or manifest["requested_blocksize"] != 64
        or baseline["compute_capability"] != [8, 9]
        or baseline["shared_bytes_per_run"] != 0
    ):
        raise ValueError(
            "This bounded control requires the Ada stage_base case"
        )
    occupancy = importlib.import_module("ncu_occupancy")
    architecture = occupancy.get_gpu_data(*baseline["compute_capability"])
    unit = architecture["shared_mem_allocation_unit_size"]
    configurations = sorted(architecture["shared_mem_size_configs"])
    byte_units = {"byte/block": 1, "Kbyte/block": 1000}
    names = {
        "dynamic": "launch__shared_mem_per_block_dynamic",
        "static": "launch__shared_mem_per_block_static",
        "driver": "launch__shared_mem_per_block_driver",
        "allocated": "launch__shared_mem_per_block_allocated",
    }
    measured = {
        key: metric_integer(metrics, name, byte_units)
        for key, name in names.items()
    }
    maximum = metric_integer(
        metrics,
        "device__attribute_max_shared_memory_per_multiprocessor",
        {"": 1},
    )
    reserved = metric_integer(
        metrics,
        "device__attribute_reserved_shared_memory_per_block",
        {"": 1},
    )
    register_blocks = metric_integer(
        metrics, "launch__occupancy_limit_registers", {"block": 1}
    )
    (layout,) = cohort["construction"]["shared"]["resolved_buffers"]
    (buffer,) = cohort["construction"]["shared"]["buffers"]
    elements = buffer["elements"]
    scalar_bytes = np.dtype(
        manifest["source_and_compiler"]["precision"]
    ).itemsize
    skew = scalar_bytes if elements and elements % 2 == 0 else 0
    per_run = elements * scalar_bytes + skew
    target = per_run * manifest["requested_blocksize"]
    rounded = ((target + reserved + unit - 1) // unit) * unit
    resident_blocks = baseline["actual_pinned_geometry"]["blocks_per_sm"]
    required = rounded * resident_blocks
    capacity = min(value for value in configurations if value >= required)
    if (
        elements != 32
        or scalar_bytes != 4
        or layout["shared_slice"] != [0, elements, None]
        or layout["allocator_use_shared"] is not True
        or shared["shared_bytes_per_run"] != per_run
        or target != shared["actual_pinned_geometry"]["dynshared"]
        or measured
        != dict(dynamic=target, static=0, driver=reserved, allocated=rounded)
        or maximum != architecture["smem_per_sm"]
        or unit <= 0
        or reserved <= 0
        or resident_blocks != register_blocks
        or resident_blocks != 4
        or shared["actual_pinned_geometry"]["blocks_per_sm"] != resident_blocks
        or capacity != 65536
    ):
        raise ValueError("Accepted physical layout/allocation facts differ")
    initial = baseline["actual_pinned_geometry"]["dynshared"]
    if initial != 4 or target <= initial:
        raise ValueError(
            "Expected dummy baseline and larger unused reservation"
        )
    calculator = occupancy.OccupancyCalculator(*baseline["compute_capability"])
    allocation_checks = []
    for dynamic in (initial, target):
        allocation = ((dynamic + reserved + unit - 1) // unit) * unit
        for shared_capacity in configurations:
            if shared_capacity < allocation:
                continue
            parameters = occupancy.OccupancyParameters(
                threads_per_block=manifest["requested_blocksize"],
                registers_per_thread=baseline["registers_per_thread"],
                shared_mem_per_block=dynamic,
                shared_mem_size=shared_capacity,
            )
            resources = calculator.get_resource_utilization(parameters)
            limiters = [
                value.name
                for value in calculator.get_occupancy_limiters(parameters)
            ]
            allocated = resources["resource_utilization"]
            if allocated["shared_memory"]["resource_per_block"] != allocation:
                raise ValueError(
                    "Nsight allocation differs from rounded bytes"
                )
            if shared_capacity >= capacity and (
                resources["allocated_blocks"] != resident_blocks
                or limiters != ["REGISTERS"]
            ):
                raise ValueError(
                    "Reservation changes register-limited residency"
                )
            allocation_checks.append(
                dict(
                    dynamic_shared_bytes=dynamic,
                    shared_capacity_bytes=shared_capacity,
                    allocated_blocks=resources["allocated_blocks"],
                    allocated_shared_bytes_per_block=allocation,
                    allocated_registers_per_block=(
                        allocated["registers"]["resource_per_block"]
                    ),
                    limiters=limiters,
                )
            )
    return dict(
        baseline_dynamic_bytes=initial,
        reserved_dynamic_bytes=target,
        added_unused_bytes=target - initial,
        elements=elements,
        scalar_bytes=scalar_bytes,
        skew_bytes_per_run=skew,
        accepted_shared_bytes_per_run=per_run,
        allocation_unit_bytes=unit,
        driver_reserved_bytes_per_block=reserved,
        rounded_shared_bytes_per_block=rounded,
        required_shared_bytes_at_original_residency=required,
        minimum_supported_shared_capacity=capacity,
        expected_control_capacity=8192,
        resident_blocks_per_sm=resident_blocks,
        architecture=architecture,
        architecture_allocation_checks=allocation_checks,
        shared_profile_metrics=measured,
        shared_analysis=dict(path=str(path), sha256=expected_sha256),
        metrics_path=str(metrics_path),
        metrics_sha256=original.file_hash(metrics_path),
        capacity_scope="minimum compatible with four blocks, not observed",
    )


def profile_metric_gate(metrics, compiled, plan, arm, n_runs):
    """Check launch metrics; callers must separately bind report identity.

    This pure constraint check does not establish the report's binary,
    source, command, input or numerical identity. Those exact joins are
    mandatory before it can qualify a reservation contrast.
    """
    requested = dynamic_bytes(plan, arm)
    blocksize = compiled["requested_blocksize"]
    unit = plan["allocation_unit_bytes"]
    driver_bytes = plan["driver_reserved_bytes_per_block"]
    allocated = ((requested + driver_bytes + unit - 1) // unit) * unit
    capacity = (
        plan["minimum_supported_shared_capacity"]
        if arm == "reserved"
        else plan["expected_control_capacity"]
    )
    byte_units = {"byte/block": 1, "Kbyte/block": 1000}
    expected = {
        "launch__shared_mem_per_block_dynamic": (byte_units, requested),
        "launch__shared_mem_per_block_static": (byte_units, 0),
        "launch__shared_mem_per_block_driver": (byte_units, driver_bytes),
        "launch__shared_mem_per_block_allocated": (byte_units, allocated),
        "launch__shared_mem_config_size": (
            {"byte": 1, "Kbyte": 1000},
            capacity,
        ),
        "launch__registers_per_thread": (
            {"register/thread": 1},
            compiled["registers_per_thread"],
        ),
        "launch__sm_count": ({"SM": 1}, compiled["multiprocessor_count"]),
    }
    for axis, grid, block in zip(
        "xyz", (n_runs // blocksize, 1, 1), (1, blocksize, 1)
    ):
        expected[f"launch__grid_dim_{axis}"] = ({"": 1}, grid)
        expected[f"launch__block_dim_{axis}"] = ({"block": 1}, block)
    values = {}
    for name, (units, required) in expected.items():
        value = metric_integer(metrics, name, units)
        values[name] = dict(
            raw=metrics[name], integer=value, expected=required
        )
        if value != required:
            raise ValueError(f"Actual launch constraint differs: {name}")
    limits = {
        name: metric_integer(
            metrics, f"launch__occupancy_limit_{name}", {"block": 1}
        )
        for name in ("blocks", "registers", "shared_mem", "warps")
    }
    if (
        limits["registers"] != plan["resident_blocks_per_sm"]
        or min(limits.values()) != limits["registers"]
        or analysis.quantity(metrics, "launch__waves_per_multiprocessor", "")
        < 2
    ):
        raise ValueError("Actual launch fails four-block/two-wave constraint")
    return dict(
        scope="launch metrics only; exact report/native/input joins required",
        values=values,
        occupancy_limits_blocks=limits,
        waves_raw=metrics["launch__waves_per_multiprocessor"],
    )


def unused_shared_proof(solver, compiled):
    """Audit the complete retained native body and zero host shared extent."""
    path = Path(compiled["artifacts"]["sass"])
    native = gzip.decompress(path.read_bytes()).decode("utf-8")
    positions = []
    counts = Counter()
    labels = {
        match[1]
        for line in native.splitlines()
        if (match := analysis.LABEL.fullmatch(line))
    }
    call_targets = set()
    for line in native.splitlines():
        match = analysis.INSTRUCTION.fullmatch(line)
        if not match:
            if re.match(r"\s*/\*[0-9a-fA-F]+\*/", line):
                raise ValueError("Unparsed native instruction")
            continue
        positions.append(int(match[1], 16))
        words = match[2].split()
        if words[0].startswith("@"):
            words = words[1:]
        opcode = words[0].split(".")[0]
        if opcode not in NO_SHARED_OPCODES:
            raise ValueError(f"Unproven native memory operation: {opcode}")
        if opcode == "CALL":
            targets = analysis.TARGET.findall(match[2])
            if (
                words[0] != "CALL.REL.NOINC"
                or len(targets) != 1
                or targets[0] not in labels
            ):
                raise ValueError("Unresolved native call in shared audit")
            call_targets.update(targets)
        counts[opcode] += 1
    closure = inspect.getclosurevars(solver.kernel.kernel.py_func).nonlocals
    extents = {
        key: closure[key] for key in ("shared_elems_per_run", "run_stride_f32")
    }
    sections = re.findall(r"^\s*\.section\s+\.text\.(\S+)", native, re.M)
    if (
        not positions
        or positions != list(range(0, len(positions) * 16, 16))
        or len(sections) != 1
        or not sections[0].startswith(compiled["entry_name"] + ",")
        or re.search(r"\bSR_[A-Z0-9_]*(?:SMEM|SHARED)", native)
        or solver.kernel.shared_memory_elements != 0
        or solver.kernel.shared_memory_bytes != 0
        or solver.kernel.shared_memory_needs_padding
        or solver.kernel.single_integrator.threads_per_step != 1
        or any(extents.values())
    ):
        raise ValueError("Complete native/host unused-shared proof failed")
    kernel_source = Path(inspect.getsourcefile(type(solver.kernel)))
    return dict(
        sass_sha256=hashlib.sha256(native.encode()).hexdigest(),
        whole_native_instruction_count=len(positions),
        all_native_base_opcodes=dict(sorted(counts.items())),
        resolved_native_call_targets=sorted(call_targets),
        batch_kernel_shared_closure=extents,
        native_shared_access_count=0,
        host_shared_elements=0,
        host_shared_bytes=0,
        host_shared_skew=False,
        kernel_source=dict(
            path=str(kernel_source), sha256=original.file_hash(kernel_source)
        ),
        scope="complete single text section including its local helpers",
    )


def dynamic_bytes(plan, arm):
    """Return a role's exact launch reservation from the physical plan."""
    if arm not in ARMS:
        raise ValueError("Unknown reservation arm")
    key = "reserved" if arm == "reserved" else "baseline"
    return plan[f"{key}_dynamic_bytes"]


def expected_compile(compiled, plan, arm):
    """Specify only the one allowed launch-geometry difference."""
    geometry = dict(compiled["actual_pinned_geometry"])
    geometry["dynshared"] = dynamic_bytes(plan, arm)
    return dict(compiled, actual_pinned_geometry=geometry)


def observe(solver, function, compiled, receipt, arm):
    """Check unchanged native identity and exact driver-queried residency."""
    expected = expected_compile(compiled, receipt["plan"], arm)
    value = hints.verify_native(solver, function, expected, receipt)
    dispatcher = solver.kernel.kernel
    (kernel,) = dispatcher.overloads.values()
    attributes = kernel._codelibrary.get_kernel_attributes()
    preference = hints.read_preference(function)
    if (
        dispatcher._launch_config_enabled
        or attributes["shared_mem_per_block"] != 0
        or preference != receipt["initial_preference_percent"]
        or solver.kernel.limit_blocksize(1, 0, 0, receipt["n_runs"])
        != (
            compiled["requested_blocksize"],
            dynamic_bytes(receipt["plan"], arm),
        )
    ):
        raise ValueError("Native specialization, static shared or pin differs")
    value.update(
        launch_config_sensitive=False,
        static_shared_bytes=0,
        preference_percent=preference,
        launch_grid=[
            receipt["n_runs"] // compiled["requested_blocksize"],
            1,
            1,
        ],
        launch_block=[1, compiled["requested_blocksize"], 1],
    )
    return value


def sample(
    solver,
    inputs,
    job,
    warm,
    function,
    receipt,
    output,
    arm,
    phase,
    block,
    index,
):
    """Retain exact identities before and after one unchanged solve."""
    compiled = receipt["compile"]
    before = observe(
        solver, function, compiled, receipt, receipt["current_arm"]
    )
    matched.pl.pin_launch(
        solver,
        compiled["requested_blocksize"],
        dynamic_bytes(receipt["plan"], arm),
    )
    receipt["current_arm"] = arm
    treatment = dict(
        arm=arm,
        dynamic_shared_bytes=dynamic_bytes(receipt["plan"], arm),
        before_arm=receipt["previous_arm"],
        before_native=before,
        native=observe(solver, function, compiled, receipt, arm),
        actual_shared_capacity=None,
    )
    receipt["previous_arm"] = arm
    key = f"b{block}-{phase}-{index}-{arm}"
    snapshot = phase in ("warm", "profile_once")
    row = matched.solve_sample(
        solver,
        inputs,
        job,
        dict(duration=receipt["duration"], snapshot=snapshot, sample_id=key),
    )
    row.update(
        key=key,
        arm=arm,
        phase=phase,
        block=block,
        sample=index,
        treatment=treatment,
        profile=receipt["profile_arm"] is not None,
        cubin_sha256=compiled["cubin_sha256"],
        manifest_sha256=receipt["manifest_sha256"],
    )
    if snapshot:
        row["numerical_checks"] = matched.compare_snapshots(warm, row)
        row["array_identity_matches"] = row["arrays"] == warm["arrays"]
    try:
        row["after_native"] = observe(solver, function, compiled, receipt, arm)
        row["input_identities"] = {
            key: matched.array_receipt(value)
            for key, value in zip(("inits", "params"), inputs)
        }
    finally:
        with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        receipt["completed_solve_count"] += 1
    validate_row(row, receipt)
    if snapshot and not (
        row["array_identity_matches"]
        and all(
            row["numerical_checks"].get(key) is True
            for key in original.NUMERICAL_FIELDS
        )
    ):
        raise ValueError(f"Exact original state/status differs: {key}")
    return row


def validate_row(row, receipt):
    """Reject short, nonfinite, failed, or identity-drifted raw solves."""
    if (
        row["chunks"] != 1
        or row["status_hist"]["failed"]
        or row["finite_state"] is not True
        or not np.isfinite(row["kernel_ms"])
        or row["kernel_ms"] <= 0
        or not np.isfinite(row["wall_ms"])
        or row["wall_ms"] <= 0
        or row["duration"] != receipt["duration"]
        or row["n_runs"] != receipt["n_runs"]
        or row["blocksize"] != receipt["compile"]["requested_blocksize"]
        or row["input_identities"] != receipt["construction"]["inputs"]
        or row["manifest_sha256"] != receipt["manifest_sha256"]
        or row["cubin_sha256"] != receipt["compile"]["cubin_sha256"]
        or row["profile"] != (receipt["profile_arm"] is not None)
        or (row["phase"] == "measurement" and row["kernel_ms"] < 20)
    ):
        raise ValueError(f"Ineligible retained solve: {row['key']}")


def collect(solver, inputs, job, warm, function, receipt, output):
    """Collect mirrored baseline/reservation/baseline slots or one profile."""

    def take(arm, phase, block, index):
        return sample(
            solver,
            inputs,
            job,
            warm,
            function,
            receipt,
            output,
            arm,
            phase,
            block,
            index,
        )

    if receipt["profile_arm"]:
        receipt["profile_sample"] = take(
            receipt["profile_arm"], "profile_once", 0, 0
        )
        receipt["status"] = "profile_complete_actual_carveout_unverified"
        return
    receipt["blocks"] = []
    for block in range(2):
        group = dict(block=block, warm=[], measurements=[])
        receipt["blocks"].append(group)
        for arm in ARMS:
            group["warm"].append(take(arm, "warm", block, 0))
        started = time.perf_counter()
        index = 0
        while time.perf_counter() - started < matched.pl.SETTLE_S:
            for arm in ARMS:
                take(arm, "settle", block, index)
            index += 1
        for index in range(6):
            order = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                group["measurements"].append(
                    take(arm, "measurement", block, index)
                )
    receipt["status"] = STATUS_ORDINARY


def validate_native(value, receipt, arm, handle):
    """Validate a saved same-native snapshot at one explicit reservation."""
    hints.validate_native_receipt(
        value,
        expected_compile(receipt["compile"], receipt["plan"], arm),
        handle,
    )
    blocksize = receipt["compile"]["requested_blocksize"]
    if (
        value["launch_config_sensitive"] is not False
        or value["static_shared_bytes"] != 0
        or value["preference_percent"] != receipt["initial_preference_percent"]
        or value["launch_grid"] != [receipt["n_runs"] // blocksize, 1, 1]
        or value["launch_block"] != [1, blocksize, 1]
    ):
        raise ValueError("Saved native launch configuration differs")


def load_ordinary(directory, warm, plan, cohort_evidence):
    """Require complete ordinary raw rows, artifacts and actual restoration."""
    path = directory / "result.json"
    value = json.loads(path.read_text())
    if (
        value["status"] != STATUS_ORDINARY
        or value["profile_arm"] is not None
        or value["helper_identity"] != helper_identity()
        or value["plan"] != plan
        or value["cohort_evidence"] != cohort_evidence
        or value["cleanup_errors"]
        or value["solver_closed"] is not True
        or value["cache_root_override_restored"] is not True
        or value["original_launch_method_restored"] is not True
    ):
        raise ValueError(
            "Ordinary reservation cohort is not complete/identical"
        )
    rows_path = directory / "samples.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    by_key = {row["key"]: row for row in rows}
    if len(by_key) != len(rows) or len(rows) != value["completed_solve_count"]:
        raise ValueError("Ordinary raw row membership differs")
    handle = value["initial_native"]["function_handle"]
    validate_native(value["initial_native"], value, "baseline", handle)
    previous = "baseline"
    for row in rows:
        validate_row(row, value)
        treatment = row["treatment"]
        if (
            row["phase"] not in ("warm", "settle", "measurement")
            or row["block"] not in (0, 1)
            or treatment["before_arm"] != previous
            or treatment["arm"] != row["arm"]
            or treatment["dynamic_shared_bytes"]
            != dynamic_bytes(plan, row["arm"])
            or treatment["actual_shared_capacity"] is not None
        ):
            raise ValueError("Ordinary treatment chain differs")
        validate_native(treatment["before_native"], value, previous, handle)
        for observed in (treatment["native"], row["after_native"]):
            validate_native(observed, value, row["arm"], handle)
        previous = row["arm"]
    expected_measurements = []
    expected_warm = []
    if [group["block"] for group in value["blocks"]] != [0, 1]:
        raise ValueError("Ordinary block membership differs")
    for group in value["blocks"]:
        block = group["block"]
        if [row["arm"] for row in group["warm"]] != list(ARMS):
            raise ValueError("Ordinary warm membership differs")
        for row in group["warm"]:
            if (
                by_key.get(row["key"]) != row
                or row["phase"] != "warm"
                or row["block"] != block
            ):
                raise ValueError("Ordinary warm/raw membership differs")
            snapshot = original.cohort_path(directory, row["snapshot"])
            original.validate_arrays(snapshot, row["arrays"])
            checks = matched.compare_snapshots(warm, row)
            if row["arrays"] != warm["arrays"] or not all(
                checks[key] is True for key in original.NUMERICAL_FIELDS
            ):
                raise ValueError("Ordinary warm arrays differ from original")
            expected_warm.append(row["key"])
        order = [
            (index, arm)
            for index in range(6)
            for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS)))
        ]
        if [
            (row["sample"], row["arm"]) for row in group["measurements"]
        ] != order:
            raise ValueError("Ordinary mirrored sample order differs")
        for row in group["measurements"]:
            if (
                by_key.get(row["key"]) != row
                or row["phase"] != "measurement"
                or row["block"] != block
            ):
                raise ValueError("Ordinary measurement/raw membership differs")
            expected_measurements.append(row["key"])
    for phase, keys in (
        ("warm", expected_warm),
        ("measurement", expected_measurements),
    ):
        if sorted(keys) != sorted(
            row["key"] for row in rows if row["phase"] == phase
        ):
            raise ValueError("Ordinary retained row set differs")
    validate_native(value["final_native"], value, previous, handle)
    validate_native(
        value["launch_restore"]["after_native"], value, "baseline", handle
    )
    if (
        value["launch_restore"]["requested_dynamic_bytes"]
        != plan["baseline_dynamic_bytes"]
    ):
        raise ValueError("Ordinary original launch reservation not restored")
    artifacts = original.artifact_receipts(directory, value["compile"])
    if artifacts != value["artifacts"]:
        raise ValueError("Ordinary native artifacts changed")
    return value, dict(
        directory=str(directory),
        result_sha256=original.file_hash(path),
        samples_sha256=original.file_hash(rows_path),
        artifacts=artifacts,
        measurement_keys=expected_measurements,
    )


def run(args):
    """Prepare one baseline, then explicitly execute an isolated control."""
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_cache = get_cache_root_override()
    solver = function = previous_launch = None
    receipt = dict(
        schema_version=1,
        kind="same_cubin_unused_shared_reservation",
        status="validating",
        helper_identity=helper_identity(),
        profile_arm=args.profile_arm,
        kernel_compilation=False,
        gpu_execution=False,
        completed_solve_count=0,
        ordinary_actual_carveout_observed=False,
        eligibility=(
            "requires matched NCU actual capacity and four-block occupancy"
        ),
        timing_use=(
            "profile diagnostics only"
            if args.profile_arm
            else "ordinary same-native reservation samples"
        ),
    )
    try:
        if not IS_MLIR:
            raise ValueError("Requires the reviewed installed MLIR launch API")
        cohort, warm, evidence = original.load_cohort(
            args.cohort_dir.resolve(), "baseline"
        )
        receipt["cohort_evidence"] = evidence
        plan = reservation_plan(
            cohort,
            evidence,
            args.shared_analysis.resolve(),
            args.shared_analysis_sha256,
        )
        receipt["plan"] = plan
        ordinary = None
        if args.profile_arm:
            if args.ordinary_dir is None:
                raise ValueError("Profile mode requires --ordinary-dir")
            ordinary, receipt["ordinary_evidence"] = load_ordinary(
                args.ordinary_dir.resolve(), warm, plan, evidence
            )
        manifest = cohort["manifest"]
        current = matched.manifest_for(
            manifest["case"],
            manifest["unroll_policy"],
            manifest["cohort"],
            manifest["protocol"]["paired_blocks"],
        )
        hints.require_equal(
            manifest,
            current,
            set(manifest) | set(current),
            "environment_differences",
            receipt,
        )
        receipt.update(
            manifest_sha256=cohort["manifest_sha256"],
            n_runs=manifest["protocol"]["n_runs"],
            duration=cohort["duration"],
            protocol=dict(
                blocks=2,
                samples_per_slot_per_block=6,
                slots=list(ARMS),
                minimum_kernel_ms=20,
                settle_s=matched.pl.SETTLE_S,
            ),
        )
        expected = cohort["construction"]["baseline"]
        private, seed = original.reproduce_generated_source(
            args.cohort_dir.resolve(), expected["generated_source"], output
        )
        receipt["generated_source_reproduction"] = seed
        set_cache_root(private)
        job = dict(role="baseline", manifest=manifest, output=str(output))
        solver, inputs, construction = matched.construct(job)
        previous_launch = solver.kernel.limit_blocksize
        receipt["construction"] = construction
        hints.require_equal(
            expected,
            construction,
            hints.CONSTRUCTION_FIELDS,
            "construction_differences",
            receipt,
        )
        hints.require_equal(
            expected["generated_source"],
            construction["generated_source"],
            ("sha256", "fn_hash"),
            "generated_source_differences",
            receipt,
        )
        receipt["unused_shared_proof"] = unused_shared_proof(
            solver, cohort["compiles"]["baseline"]
        )
        if construction["compilation_check"]["native_overloads"] != 0:
            raise ValueError("Host preparation unexpectedly compiled")
        receipt["prepared_native_overloads"] = len(
            solver.kernel.kernel.overloads
        )
        if receipt["prepared_native_overloads"] != 0:
            raise ValueError("Batch closure inspection unexpectedly compiled")
        if not args.execute:
            receipt["status"] = "prepared"
            return
        receipt["kernel_compilation"] = True
        compiled = matched.compile_solver(solver, inputs, job)
        receipt["compile"] = compiled
        hints.require_equal(
            cohort["compiles"]["baseline"],
            compiled,
            hints.COMPILE_FIELDS,
            "compile_differences",
            receipt,
        )
        if ordinary is not None:
            hints.require_equal(
                ordinary["compile"],
                compiled,
                hints.COMPILE_FIELDS,
                "ordinary_compile_differences",
                receipt,
            )
        if not original.geometry_valid(
            compiled, manifest["requested_blocksize"]
        ):
            raise ValueError("Invalid baseline pinned geometry")
        hints.require_equal(
            cohort["compiles"]["baseline"]["actual_pinned_geometry"],
            compiled["actual_pinned_geometry"],
            original.GEOMETRY_FIELDS,
            "geometry_differences",
            receipt,
        )
        artifacts = original.artifact_receipts(output, compiled)
        receipt["artifacts"] = artifacts
        if any(
            item["content_sha256"]
            != evidence["artifacts"][key]["content_sha256"]
            for key, item in artifacts.items()
        ):
            raise ValueError("Exact original cubin/PTX/SASS differs")
        receipt["fresh_unused_shared_proof"] = unused_shared_proof(
            solver, compiled
        )
        if (
            receipt["fresh_unused_shared_proof"]
            != receipt["unused_shared_proof"]
        ):
            raise ValueError("Fresh native shared-access proof differs")
        source = construction["generated_source"]
        if original.file_hash(source["path"]) != source["sha256"]:
            raise ValueError("Generated source changed during compilation")
        function = hints.get_function(solver)
        receipt["initial_preference_percent"] = hints.read_preference(function)
        device = cuda.get_current_device()
        fields = driver.CUdevice_attribute
        attribute = fields.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK
        status, reserved = driver.cuDeviceGetAttribute(attribute, device.id)
        if status.value != 0:
            raise RuntimeError(f"Driver reservation query failed: {status}")
        receipt["driver_attributes"] = dict(
            reserved_shared_bytes_per_block=int(reserved),
            maximum_shared_bytes_per_sm=int(
                device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
            ),
        )
        if (
            int(reserved) != plan["driver_reserved_bytes_per_block"]
            or receipt["driver_attributes"]["maximum_shared_bytes_per_sm"]
            != plan["architecture"]["smem_per_sm"]
        ):
            raise ValueError("Queried hardware differs from reviewed profile")
        receipt["current_arm"] = receipt["previous_arm"] = "baseline"
        receipt["initial_native"] = observe(
            solver, function, compiled, receipt, "baseline"
        )
        receipt["gpu_execution"] = True
        collect(solver, inputs, job, warm, function, receipt, output)
        receipt["final_native"] = observe(
            solver, function, compiled, receipt, receipt["current_arm"]
        )
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        unwinding = sys.exc_info()[0] is not None
        errors = []
        if solver is not None and function is not None:
            try:
                initial = receipt["plan"]["baseline_dynamic_bytes"]
                matched.pl.pin_launch(
                    solver, manifest["requested_blocksize"], initial
                )
                receipt["launch_restore"] = dict(
                    requested_dynamic_bytes=initial,
                    after_native=observe(
                        solver,
                        function,
                        receipt["compile"],
                        receipt,
                        "baseline",
                    ),
                )
            except Exception as error:
                errors.append(
                    f"launch restore: {type(error).__name__}: {error}"
                )
        if solver is not None:
            if previous_launch is not None:
                solver.kernel.limit_blocksize = previous_launch
                receipt["original_launch_method_restored"] = (
                    solver.kernel.limit_blocksize is previous_launch
                )
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
    """Prepare on CPU unless explicit execution is requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--shared-analysis", type=Path, required=True)
    parser.add_argument("--shared-analysis-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile-arm", choices=("baseline", "reserved"))
    parser.add_argument("--ordinary-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
