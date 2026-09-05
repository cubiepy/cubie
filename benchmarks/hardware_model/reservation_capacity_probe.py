"""Sweep unused shared reservations on one unchanged accepted cubin.

Default preparation constructs one Solver without native compilation.
Explicit --execute collects fixed-duration ordinary samples or one
profile solve. Requested bytes derive from supported shared capacities
and allocation granularity; actual per-launch capacity needs NCU.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

from cuda.bindings import driver

from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cuda_backend import IS_MLIR
from cubie.cuda_simsafe import cuda

import carveout_probe as hints
import placement_profile as original
import placement_probe as matched
import reservation_probe as reservation


SCRIPT = Path(__file__).resolve()
FROZEN_RESERVATION_SHA256 = (
    "e31119f8cacef5032b0ef185036169ccbc0dbc6e384ecce54b8f395117f073e6"
)
ARMS = (
    "baseline", "capacity16", "capacity32", "capacity64",
    "original64", "capacity100", "baseline_repeat",
)
PROFILE_ARMS = ARMS[:-1]
STATUS_ORDINARY = "ordinary_complete_actual_carveout_unverified"
ADA_GUIDE = (
    "https://docs.nvidia.com/cuda/ada-tuning-guide/index.html"
    "#unified-shared-memory-l1-texture-cache"
)


def helper_identity():
    """Bind this instrument and every frozen helper it calls."""
    if original.file_hash(reservation.SCRIPT) != FROZEN_RESERVATION_SHA256:
        raise ValueError("Frozen reservation helper changed")
    return dict(
        reservation.helper_identity(),
        **{str(SCRIPT): original.file_hash(SCRIPT)},
    )


def allocation(dynamic, driver_bytes, unit):
    """Round the combined dynamic and driver request to allocation units."""
    return ((dynamic + driver_bytes + unit - 1) // unit) * unit


def capacity_plan(cohort, evidence, path, expected_sha256):
    """Derive the smallest integer request across supported capacities."""
    base = reservation.reservation_plan(
        cohort, evidence, path, expected_sha256
    )
    architecture = base["architecture"]
    unit = base["allocation_unit_bytes"]
    driver_bytes = base["driver_reserved_bytes_per_block"]
    blocks = base["resident_blocks_per_sm"]
    capacities = sorted(architecture["shared_mem_size_configs"])
    documented = [value * 1024 for value in (0, 8, 16, 32, 64, 100)]
    if (capacities != documented or unit != 128 or driver_bytes != 1024
            or blocks != 4):
        raise ValueError("Queried allocation rules differ from bounded Ada")
    compiled = cohort["compiles"]["baseline"]
    occupancy = importlib.import_module("ncu_occupancy")
    calculator = occupancy.OccupancyCalculator(*compiled["compute_capability"])
    levels = {}
    for arm in ARMS:
        if arm in ("baseline", "baseline_repeat"):
            dynamic = base["baseline_dynamic_bytes"]
            capacity = base["expected_control_capacity"]
            reason = "original baseline dummy allocation"
            predecessor = None
        elif arm == "original64":
            dynamic = base["reserved_dynamic_bytes"]
            capacity = base["minimum_supported_shared_capacity"]
            reason = "original accepted 8448-byte reservation control"
            predecessor = 32768
        else:
            capacity = int(arm.removeprefix("capacity")) * 1024
            predecessor = capacities[capacities.index(capacity) - 1]
            rounded = (predecessor // (blocks * unit) + 1) * unit
            dynamic = rounded - unit - driver_bytes + 1
            reason = "smallest integer request excluding predecessor"
        allocated = allocation(dynamic, driver_bytes, unit)
        required = allocated * blocks
        minimum = min(item for item in capacities if item >= required)
        if dynamic < base["baseline_dynamic_bytes"] or minimum != capacity:
            raise ValueError("Derived request has the wrong minimum capacity")
        if arm.startswith("capacity") and not (
            allocation(dynamic - 1, driver_bytes, unit) * blocks
            <= predecessor < required
        ):
            raise ValueError("Derived request is not minimal")
        checks = []
        for shared_capacity in capacities:
            if shared_capacity < allocated:
                continue
            parameters = occupancy.OccupancyParameters(
                threads_per_block=compiled["requested_blocksize"],
                registers_per_thread=compiled["registers_per_thread"],
                shared_mem_per_block=dynamic,
                shared_mem_size=shared_capacity,
            )
            resources = calculator.get_resource_utilization(parameters)
            limiters = [item.name for item in
                        calculator.get_occupancy_limiters(parameters)]
            used = resources["resource_utilization"]
            if used["shared_memory"]["resource_per_block"] != allocated:
                raise ValueError("Queried shared allocation disagrees")
            if shared_capacity >= capacity and (
                resources["allocated_blocks"] != blocks
                or limiters != ["REGISTERS"]
            ):
                raise ValueError("Target changes register-limited residency")
            shared_bound = shared_capacity // allocated
            if shared_capacity < capacity and shared_bound >= blocks:
                raise ValueError("Predecessor still admits original residency")
            checks.append(dict(
                shared_capacity_bytes=shared_capacity,
                allocated_blocks=resources["allocated_blocks"],
                shared_block_bound_from_rounded_bytes=shared_bound,
                calculator_agrees_with_capacity_bound=(
                    resources["allocated_blocks"] <= shared_bound
                ),
                raw_shared_resource_utilization=used["shared_memory"],
                allocated_shared_bytes_per_block=allocated,
                allocated_registers_per_block=(
                    used["registers"]["resource_per_block"]
                ),
                limiters=limiters,
            ))
        levels[arm] = dict(
            dynamic_shared_bytes=dynamic,
            added_unused_bytes=dynamic - base["baseline_dynamic_bytes"],
            allocated_shared_bytes_per_block=allocated,
            required_bytes_at_original_residency=required,
            minimum_supported_shared_capacity=capacity,
            preceding_supported_capacity=predecessor,
            nominal_l1_texture_complement_bytes=128 * 1024 - capacity,
            derivation=reason,
            architecture_allocation_checks=checks,
        )
    return dict(
        base_reference_plan=base,
        baseline_dynamic_bytes=base["baseline_dynamic_bytes"],
        allocation_unit_bytes=unit,
        driver_reserved_bytes_per_block=driver_bytes,
        resident_blocks_per_sm=blocks,
        architecture=architecture,
        documented_capacities_bytes=documented,
        documented_unified_capacity_bytes=128 * 1024,
        documentation=ADA_GUIDE,
        levels=levels,
        capacity_scope="minimum compatible with B blocks; profile required",
        l1_scope="nominal complement; no usable-capacity or replacement claim",
        allocation_equation="U * ceil((dynamic + driver) / U)",
        predecessor_bound="floor(capacity / rounded allocation)",
        calculator_scope=(
            "Target capacities require exact four-block REGISTERS agreement; "
            "predecessor byte-capacity contradictions are retained explicitly"
        ),
    )


def dynamic_bytes(plan, arm):
    """Return an admitted level's exact unused dynamic byte request."""
    if arm not in ARMS:
        raise ValueError("Unknown capacity arm")
    return plan["levels"][arm]["dynamic_shared_bytes"]


def profile_metric_gate(metrics, compiled, plan, arm, n_runs):
    """Require actual capacity and launch limits after exact report joins."""
    if arm not in PROFILE_ARMS:
        raise ValueError("Unknown profile capacity arm")
    level = plan["levels"][arm]
    bridge = dict(
        plan["base_reference_plan"],
        reserved_dynamic_bytes=dynamic_bytes(plan, arm),
        minimum_supported_shared_capacity=(
            level["minimum_supported_shared_capacity"]
        ),
    )
    result = reservation.profile_metric_gate(
        metrics, compiled, bridge, "reserved", n_runs
    )
    result["capacity_arm"] = arm
    result["nominal_l1_texture_complement_bytes"] = (
        level["nominal_l1_texture_complement_bytes"]
    )
    return result


def validate_row(row, receipt):
    """Retain all original solve gates and the fixed capacity-arm domain."""
    if row["arm"] not in ARMS:
        raise ValueError("Unknown retained capacity arm")
    reservation.validate_row(row, receipt)


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


def collect(solver, inputs, job, warm, function, receipt, output):
    """Collect the fixed mirrored capacity slots or one profile."""

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


def validate_original_ordinary(value, directory, cohort, arms):
    """Bind saved work directly to the original completed placement cohort."""
    manifest = cohort["manifest"]
    if (
        value["duration"] != cohort["duration"]
        or value["n_runs"] != manifest["protocol"]["n_runs"]
        or value["manifest_sha256"] != cohort["manifest_sha256"]
    ):
        raise ValueError("Saved ordinary workload differs from original")
    expected = cohort["construction"]["baseline"]
    construction = value["construction"]
    differences = original.compare_fields(
        expected, construction, hints.CONSTRUCTION_FIELDS
    )
    if differences:
        raise ValueError(f"Original construction differs: {differences}")
    generated = construction["generated_source"]
    differences = original.compare_fields(
        expected["generated_source"], generated, ("sha256", "fn_hash")
    )
    if differences or original.file_hash(generated["path"]) != generated[
        "sha256"
    ]:
        raise ValueError("Original generated-source identity differs")
    inputs = original.validate_arrays(
        original.cohort_path(directory, directory / "baseline-inputs.npz"),
        expected["inputs"],
    )
    baseline = cohort["compiles"]["baseline"]
    compiled = value["compile"]
    differences = original.compare_fields(
        baseline, compiled, hints.COMPILE_FIELDS
    )
    if differences:
        raise ValueError(f"Original compilation differs: {differences}")
    differences = original.compare_fields(
        baseline["actual_pinned_geometry"],
        compiled["actual_pinned_geometry"],
        original.GEOMETRY_FIELDS,
    )
    if differences:
        raise ValueError(f"Original pinned geometry differs: {differences}")
    protocol = dict(
        blocks=2,
        samples_per_slot_per_block=6,
        slots=list(arms),
        minimum_kernel_ms=20,
        settle_s=matched.pl.SETTLE_S,
    )
    if value["protocol"] != protocol:
        raise ValueError("Saved ordinary protocol differs")
    return dict(
        duration=cohort["duration"],
        n_runs=manifest["protocol"]["n_runs"],
        manifest_sha256=cohort["manifest_sha256"],
        source_hash=baseline["source_hash"],
        compiler_identity=baseline["compiler_identity"],
        config_hash=baseline["config_hash"],
        cubin_sha256=baseline["cubin_sha256"],
        inputs=inputs,
        generated_source_sha256=generated["sha256"],
        protocol=protocol,
    )


def load_ordinary(
    directory, warm, plan, cohort_evidence, reference_evidence, cohort,
    reference_original_workload,
):
    """Require complete ordinary raw rows, artifacts and actual restoration."""
    path = directory / "result.json"
    value = json.loads(path.read_text())
    original_workload = validate_original_ordinary(
        value, directory, cohort, ARMS
    )
    if (
        value["kind"] != "same_cubin_unused_shared_capacity_sweep"
        or value["schema_version"] != 1
        or value["reference_ordinary_evidence"] != reference_evidence
        or value["reference_original_workload"] != reference_original_workload
        or value["status"] != STATUS_ORDINARY
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
        original_workload=original_workload,
    )


def run(args):
    """Prepare one baseline, then explicitly execute an isolated control."""
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_cache = get_cache_root_override()
    solver = function = previous_launch = None
    receipt = dict(
        schema_version=1,
        kind="same_cubin_unused_shared_capacity_sweep",
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
        plan = capacity_plan(
            cohort,
            evidence,
            args.shared_analysis.resolve(),
            args.shared_analysis_sha256,
        )
        receipt["plan"] = plan
        reference, receipt["reference_ordinary_evidence"] = (
            reservation.load_ordinary(
                args.reference_ordinary_dir.resolve(), warm,
                plan["base_reference_plan"], evidence,
            )
        )
        receipt["reference_original_workload"] = validate_original_ordinary(
            reference, args.reference_ordinary_dir.resolve(),
            cohort, reservation.ARMS,
        )
        ordinary = None
        if args.profile_arm:
            if args.ordinary_dir is None:
                raise ValueError("Profile mode requires --ordinary-dir")
            ordinary, receipt["ordinary_evidence"] = load_ordinary(
                args.ordinary_dir.resolve(), warm, plan, evidence,
                receipt["reference_ordinary_evidence"],
                cohort,
                receipt["reference_original_workload"],
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
        receipt["unused_shared_proof"] = reservation.unused_shared_proof(
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
        hints.require_equal(
            reference["compile"], compiled, hints.COMPILE_FIELDS,
            "reference_compile_differences", receipt,
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
        receipt["fresh_unused_shared_proof"] = reservation.unused_shared_proof(
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
    """Prepare on CPU or explicitly execute one accepted fixed sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--shared-analysis", type=Path, required=True)
    parser.add_argument("--shared-analysis-sha256", required=True)
    parser.add_argument("--reference-ordinary-dir", type=Path, required=True,
                        help="Accepted original 4/8448-byte ordinary cohort")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile-arm", choices=PROFILE_ARMS,
                        help="Exactly one solve; requires new ordinary sweep")
    parser.add_argument("--ordinary-dir", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
