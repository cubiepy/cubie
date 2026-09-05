"""Freeze source predictions and validate a separate native solver bank."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time
import traceback

import numpy as np

from cubie import Solver
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cuda_backend import CUDA_BACKEND
from cubie.cuda_simsafe import cuda
from cubie.result_codes import CUBIE_RESULT_CODES
from benchmarks import placement_landscape as landscape
from benchmarks.hardware_model import implicit_policy_graph as policy
from benchmarks.hardware_model import implicit_workload as workload
from benchmarks.hardware_model.workload_identity import workload_identity
from benchmarks.hardware_model.buffer_descriptors import registry_layout


def file_hash(path):
    """Hash exact artifact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    """Bind an exact finite JSON record."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def compiler_record(value):
    """Keep exact compiler options, including the installed fastmath set."""
    if isinstance(value, (set, frozenset)):
        return dict(type=type(value).__name__, members=sorted(value))
    if isinstance(value, dict):
        return {key: compiler_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compiler_record(item) for item in value]
    return value


def write(path, value):
    """Retain full-precision JSON without permissive NaN serialization."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def read(path):
    """Read plain or compressed research JSON."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def copy_record(path, destination):
    """Copy an immutable prediction artifact and record both locations."""
    path, destination = Path(path).resolve(), Path(destination).resolve()
    shutil.copyfile(path, destination)
    return dict(
        path=str(destination),
        sha256=file_hash(destination),
        original_path=str(path),
    )


def constructor(request, item, candidate, folder, instrumented=False):
    """Construct the same public source action without requesting compile."""
    set_cache_root(Path(folder) / "codegen")
    kwargs = landscape.solver_kwargs(request["system"], item["algorithm"])
    kwargs.update(request.get("solver_settings", {}))
    if item["inner"] is None:
        kwargs.pop("linear_correction_type", None)
    else:
        kwargs["linear_correction_type"] = workload.PUBLIC_LINEAR_TYPES[
            item["inner"]
        ]
    kwargs.update(candidate["locations"])
    kwargs["unroll"] = policy.policy_flags(candidate["levels"])
    outputs = kwargs.get("output_types", [])
    if outputs != ["state"]:
        raise ValueError("Holdout bank requires the predicted state-only ABI")
    if instrumented:
        kwargs["output_types"] = ["state", "iteration_counters"]
    solver = Solver(landscape.SYSTEMS[request["system"]]["build"](), **kwargs)
    constants = request.get(
        "system_constants",
        landscape.SYSTEMS[request["system"]].get("constants"),
    )
    if constants:
        solver.update(constants)
    # Bind caller/child factories through their cached public output.
    solver.kernel.single_integrator.device_function
    if solver.kernel.kernel.overloads:
        solver.close()
        raise ValueError(
            "Native specialization appeared before explicit compile"
        )
    registered = registry_layout(solver.kernel.single_integrator._algo_step)
    identity = {
        row["owner"] + ":" + row["name"]: row["declared_location"]
        for row in registered
    }
    graph = read(candidate["graph"]["path"])
    if identity != candidate["placement"]:
        solver.close()
        raise ValueError(
            "Constructed placement differs from frozen source action"
        )
    if (
        landscape.bytes_per_run(solver)
        != graph["candidate_construction"]["shared_stride_bytes"]
    ):
        solver.close()
        raise ValueError(
            "Constructed shared stride differs from frozen source"
        )
    return solver, kwargs, constants


def admitted_request_hashes(request, item):
    """Reproduce the evaluator's explicitly adjusted source-regime requests."""
    hashes = set()
    for regime in request.get("source_regimes", [request["source_regime"]]):
        effective = dict(regime)
        effective.pop("id", None)
        if item["algorithm"] in ("rk23", "rosenbrock23"):
            effective["newton_bodies"] = 0
        if item["inner"] in (None, "lu"):
            effective["krylov_bodies"] = 0
        hashes.add(digest(dict(request, source_regime=effective)))
    return hashes


def source_snapshot(output):
    """Bind production code and constructor helpers across freeze and run."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src" / "cubie").rglob("*.py"))
    paths += [
        Path(landscape.__file__),
        Path(workload.__file__),
        Path(policy.__file__),
        Path(__file__),
        Path(inspect.getfile(workload_identity)),
        Path(inspect.getfile(registry_layout)),
    ]
    records = []
    for path in sorted(set(paths)):
        relative = path.relative_to(root)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = copy_record(path, destination)
        records.append(record)
    return records


def verify_graph_source(graph):
    """Match saved production function files to the current source tree."""
    root = Path(__file__).resolve().parents[2]
    records = []
    for function in graph["provenance"]["functions"]:
        path = Path(function["source"]["source_path"])
        parts = path.parts
        if "src" not in parts:
            continue
        current = root.joinpath(*parts[parts.index("src") :])
        if file_hash(path) != file_hash(current):
            raise ValueError(
                "Production source differs from predicted graph: "
                + str(current)
            )
        records.append(dict(path=str(current), sha256=file_hash(current)))
    if not records:
        raise ValueError("No production source binding found in graph")
    return records


def carveout_percent(requested, supported):
    """Find an integer hint whose supported-size roundup is exact."""
    supported = sorted(set(supported))
    if requested not in supported or not supported or supported[-1] <= 0:
        raise ValueError("Requested shared carveout is not supported")
    matches = []
    for percent in range(101):
        target = Fraction(percent * supported[-1], 100)
        rounded = next(size for size in supported if size >= target)
        if rounded == requested:
            matches.append(percent)
    if not matches:
        raise ValueError("No integer hint represents the requested carveout")
    return matches[0]


def default_protocol(system):
    """Use the existing landscape workload and AB gate measurement design."""
    spec = landscape.SYSTEMS[system]
    return dict(
        n_runs=spec["n_runs"],
        duration=spec["duration"],
        t0=0.0,
        warmups=2,
        rounds=4,
        solves_per_block=15,
        lowest_count=5,
        idle_seconds=[1.5, 3.5],
        order_seed=0,
        diagnostic_counters=False,
    )


def validate_protocol(protocol):
    """Require a complete fixed measurement design before native work."""
    for field in (
        "n_runs",
        "warmups",
        "rounds",
        "solves_per_block",
        "lowest_count",
    ):
        if type(protocol[field]) is not int or protocol[field] < 1:
            raise ValueError("Positive protocol integer required: " + field)
    if (
        protocol["rounds"] % 2
        or protocol["lowest_count"] > protocol["solves_per_block"]
    ):
        raise ValueError("Use even rounds and an admitted block statistic")
    if (
        not math.isfinite(protocol["duration"])
        or protocol["duration"] <= 0
        or not math.isfinite(protocol["t0"])
    ):
        raise ValueError("Finite positive duration and start time required")
    low, high = protocol["idle_seconds"]
    if not 0 < low <= high or not math.isfinite(high):
        raise ValueError("Positive bounded idle interval required")
    if type(protocol["diagnostic_counters"]) is not bool:
        raise ValueError("Counter diagnostics must be an explicit Boolean")


def freeze(request_path, results, baselines, output, protocol=None):
    """Freeze every candidate and ranking before any native compilation."""
    if not results or len(results) != len(baselines):
        raise ValueError(
            "Each nonempty result cohort needs exactly one baseline"
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = read(request_path)
    protocol = (
        default_protocol(request["system"]) if protocol is None else protocol
    )
    validate_protocol(protocol)
    request_record = copy_record(request_path, output / "request.json")
    entries, assets = [], [request_record]
    previous = get_cache_root_override()
    try:
        for index, path in enumerate(results):
            result = read(path)
            if (
                result["status"] != "finite_design_evaluated"
                or result["native_compilations"] != 0
                or result["kernel_launches"] != 0
                or result["native_labels_consumed"] is not False
                or result["solver_timings_consumed"] is not False
                or result["ranking"].get("status")
                not in (
                    "conditional_finite_scenario_default",
                    "conditional_compute_data_default",
                )
            ):
                raise ValueError(
                    "A complete source-only ranked result is required"
                )
            if result["workload"] not in request["workloads"] or result[
                "request_sha256"
            ] not in admitted_request_hashes(request, result["workload"]):
                raise ValueError("Result is not bound to this source request")
            folder = output / f"workload_{index:03d}"
            folder.mkdir()
            result_record = copy_record(path, folder / "result.json")
            assets.append(result_record)
            baseline = baselines[index]
            if baseline not in result["candidates"]:
                raise ValueError(
                    "Explicit baseline must belong to the finite cohort"
                )
            candidates = {}
            for identifier, candidate in result["candidates"].items():
                graph_ref = candidate["graph"]
                if file_hash(graph_ref["path"]) != graph_ref["sha256"]:
                    raise ValueError(
                        "Candidate graph artifact identity differs"
                    )
                graph = read(graph_ref["path"])
                verify_graph_source(graph)
                expected = result["comparison_identity"]["workload"]
                if (
                    graph["candidate_construction"]["workload_identity"]
                    != expected
                ):
                    raise ValueError("Candidate semantic workload differs")
                suffix = (
                    ".json.gz"
                    if graph_ref["path"].endswith(".gz")
                    else ".json"
                )
                record = copy_record(
                    graph_ref["path"],
                    folder / (identifier + "_graph" + suffix),
                )
                assets.append(record)
                candidates[identifier] = dict(candidate, graph=record)
                carveout_percent(
                    candidate["geometry"]["carveout"],
                    request["hardware"]["supported_shared_carveouts"],
                )
            solver, kwargs, constants = constructor(
                request, result["workload"], candidates[baseline], folder
            )
            try:
                actual = workload_identity(solver.system, solver)
                if actual != result["comparison_identity"]["workload"]:
                    raise ValueError(
                        "Frozen constructor differs from prediction"
                    )
                inits, params = landscape.SYSTEMS[request["system"]]["grid"](
                    solver, protocol["n_runs"]
                )
                if any(
                    x.dtype != np.float32
                    or x.ndim != 2
                    or x.shape[1] != protocol["n_runs"]
                    for x in (inits, params)
                ):
                    raise ValueError("Actual input grids must be FP32 columns")
                if not all(np.all(np.isfinite(x)) for x in (inits, params)):
                    raise ValueError("Frozen input grids must be finite")
                grid = folder / "grid.npz"
                np.savez_compressed(
                    grid, initial_values=inits, parameters=params
                )
                grid_ref = dict(path=str(grid), sha256=file_hash(grid))
                assets.append(grid_ref)
                tolerances = {
                    key: float(kwargs[key]) for key in ("atol", "rtol")
                }
                if any(
                    not math.isfinite(x) or x < 0 for x in tolerances.values()
                ):
                    raise ValueError(
                        "Finite scalar solver tolerances required"
                    )
            finally:
                solver.close()
            entries.append(
                dict(
                    id=f"workload_{index:03d}",
                    workload=result["workload"],
                    result=result_record,
                    candidates=candidates,
                    baseline=baseline,
                    ranking=result["ranking"],
                    semantic_workload=actual,
                    constructor_constants=constants,
                    grid=grid_ref,
                    numerical_tolerances=tolerances,
                    duplicate_id=baseline + "__independent_duplicate",
                )
            )
    finally:
        set_cache_root(previous)
    sources = source_snapshot(output)
    assets.extend(sources)
    manifest = dict(
        kind="frozen_native_policy_holdout",
        schema=1,
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        request=request_record,
        protocol=protocol,
        workloads=entries,
        assets=assets,
        runner_sha256=file_hash(__file__),
        production_sources=sources,
        native_compilations=0,
        gpu_launches=0,
        interpretation="Whole-solver holdout of frozen conditional source "
        "rankings; local tolerance comparison is agreement, not global accuracy",
    )
    write(output / "manifest.json", manifest)
    receipt = dict(
        status="PREDICTION_FROZEN_BEFORE_NATIVE_COMPILATION",
        manifest_sha256=file_hash(output / "manifest.json"),
        runner_sha256=manifest["runner_sha256"],
    )
    write(output / "freeze_receipt.json", receipt)
    return receipt


def sample_arrays(output, state, status, counters=None):
    """Retain every sample through exact content-addressed array storage."""
    arrays = dict(state=state, status=status)
    if counters is not None:
        arrays["iteration_counters"] = counters
    fingerprint = hashlib.sha256()
    for key, array in arrays.items():
        fingerprint.update(
            digest(
                dict(name=key, dtype=array.dtype.str, shape=list(array.shape))
            ).encode()
        )
        fingerprint.update(array.tobytes(order="C"))
    key = fingerprint.hexdigest()
    path = Path(output) / "arrays" / (key + ".npz")
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        np.savez_compressed(path, **arrays)
    return dict(path=str(path), content_sha256=key, sha256=file_hash(path))


def numerical_check(state, status, reference, tolerances, duplicate):
    """Preserve strict status/finite checks and unchanged local tolerances."""
    flags = dict(
        fp32=state.dtype == np.float32,
        nonempty=state.size > 0 and status.size > 0,
        finite=bool(np.all(np.isfinite(state))),
        status_success=bool(np.all(status == int(CUBIE_RESULT_CODES.SUCCESS))),
        baseline_available=reference is not None,
    )
    if reference is not None:
        if duplicate:
            flags["baseline_duplicate_exact"] = (
                state.shape == reference[0].shape
                and state.tobytes() == reference[0].tobytes()
                and status.shape == reference[1].shape
                and status.tobytes() == reference[1].tobytes()
            )
        else:
            flags["local_tolerance_agreement"] = state.shape == reference[
                0
            ].shape and bool(
                np.allclose(state, reference[0], equal_nan=False, **tolerances)
            )
            flags["status_matches_baseline"] = bool(
                np.array_equal(status, reference[1])
            )
    return dict(passed=all(flags.values()), checks=flags)


def measure(solver, inits, params, protocol, candidate):
    """Measure only existing kernel CUDA events and keep raw endpoint data."""
    start = time.perf_counter()
    result = solver.solve(
        inits,
        params,
        duration=protocol["duration"],
        t0=protocol["t0"],
        blocksize=candidate["geometry"]["block_threads"],
        grid_type="verbatim",
        nan_error_trajectories=False,
    )
    wall = (time.perf_counter() - start) * 1000
    state, status = np.array(result.state[-1]), np.array(result.status_codes)
    counters = result.iteration_counters
    counters = None if counters is None else np.array(counters)
    events = [
        dict(name=event.name, milliseconds=float(event.elapsed_time_ms()))
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    ]
    event_valid = bool(events) and all(
        math.isfinite(x["milliseconds"]) and x["milliseconds"] > 0
        for x in events
    )
    geometries, problems = actual_geometry(solver, candidate)
    if not event_valid:
        problems.append("Kernel CUDA events must be positive and finite")
    if len(events) != len(geometries):
        problems.append("Kernel event and actual chunk counts differ")
    kernel_ms = (
        math.fsum(x["milliseconds"] for x in events) if event_valid else None
    )
    for event in events:
        if not math.isfinite(event["milliseconds"]):
            event["milliseconds"] = event["milliseconds"].hex()
    return (
        dict(
            kernel_ms=kernel_ms,
            wall_ms=wall,
            kernel_events=events,
            geometries=geometries,
            measurement_valid=not problems,
            problems=problems,
        ),
        state,
        status,
        counters,
    )


def actual_geometry(solver, candidate):
    """Check every actual public chunk against compiled occupancy waves."""
    requested = candidate["geometry"]
    geometries, problems = [], []
    threads = int(solver.kernel.single_integrator.threads_per_step)
    for index in range(solver.kernel.run_params.num_chunks):
        runs = int(solver.kernel.run_params[index].runs)
        geometry = landscape.launch_geometry(
            solver, requested["block_threads"], runs
        )
        geometries.append(dict(chunk=index, runs=runs, actual=geometry))
        if (
            geometry is None
            or geometry["blocksize"] != requested["block_threads"]
            or geometry["dynshared"]
            != max(
                4, landscape.bytes_per_run(solver) * requested["block_threads"]
            )
            or geometry["blocks_per_sm"] < 1
            or geometry["waves"] < 2
        ):
            problems.append(
                "Public geometry mismatch or fewer than two full waves"
            )
    if not geometries:
        problems.append("No actual launch chunks")
    if threads != 1:
        problems.append(
            "Declared holdout geometry requires one thread per trajectory"
        )
    return geometries, problems


def device_identity(hardware):
    """Record and check the target device before the validation bank."""
    device = cuda.get_current_device()
    fields = dict(
        multiprocessor_count="MULTIPROCESSOR_COUNT",
        warp_size="WARP_SIZE",
        max_threads_per_block="MAX_THREADS_PER_BLOCK",
        max_threads_per_sm="MAX_THREADS_PER_MULTIPROCESSOR",
        registers_per_sm="MAX_REGISTERS_PER_MULTIPROCESSOR",
    )
    observed = {
        key: int(getattr(device, attribute))
        for key, attribute in fields.items()
    }
    if any(observed[key] != hardware[key] for key in fields):
        raise ValueError(
            "Native device capacities differ from frozen hardware"
        )
    capability = list(device.compute_capability)
    if capability != [8, 9]:
        raise ValueError(
            "Native holdout target is the declared Ada SM89 device"
        )
    return dict(
        name=str(device.name),
        compute_capability=capability,
        capacities=observed,
    )


def native_compile(
    solver, candidate, request, protocol, inits, params, folder, disassembler
):
    """Compile through the public API and retain diagnostic native artifacts."""
    start = time.perf_counter()
    solver.compile(
        inits,
        params,
        duration=protocol["duration"],
        t0=protocol["t0"],
        grid_type="verbatim",
    )
    duration = time.perf_counter() - start
    (compiled,) = solver.kernel.kernel.overloads.values()
    cufunc = compiled._codelibrary.get_cufunc()
    requested = candidate["geometry"]
    percent = carveout_percent(
        requested["carveout"],
        request["hardware"]["supported_shared_carveouts"],
    )
    cufunc.set_shared_memory_carveout(percent)
    library = compiled._codelibrary
    cubin = (
        bytes(library.get_cubin().code)
        if hasattr(library, "get_cubin")
        else bytes(library._cubin)
    )
    cubin_path = folder / "kernel.cubin"
    cubin_path.write_bytes(cubin)
    process = subprocess.run(
        [str(disassembler), "-c", str(cubin_path)],
        capture_output=True,
        text=True,
    )
    write(
        folder / "disassembly_command.json",
        dict(
            command=process.args,
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        ),
    )
    process.check_returncode()
    with gzip.open(folder / "kernel.sass.gz", "wt") as handle:
        handle.write(process.stdout)
    geometries, problems = actual_geometry(solver, candidate)
    attrs = {key: int(value) for key, value in cufunc.attrs._asdict().items()}
    if attrs["shared"] != requested["static_shared"]:
        problems.append(
            "Compiled static shared allocation differs from source hypothesis"
        )
    record = dict(
        validation_only=True,
        compile_seconds=duration,
        cubin_sha256=file_hash(cubin_path),
        sass_sha256=file_hash(folder / "kernel.sass.gz"),
        compiler_kwargs=compiler_record(dict(solver.kernel.jit_kwargs)),
        native_attributes=attrs,
        geometries=geometries,
        carveout=dict(
            requested_bytes=requested["carveout"],
            integer_percent=percent,
            actual_partition_verified=False,
            interpretation="Driver preference with exact supported "
            "roundup; achieved partition is not inferred",
        ),
        eligible=not problems,
        problems=problems,
    )
    write(folder / "compile.json", record)
    return record


def block_orders(identifiers, rounds):
    """Rotate each forward/reverse pair to distribute position drift."""
    for index in range(rounds):
        offset = (index // 2) % len(identifiers)
        order = identifiers[offset:] + identifiers[:offset]
        yield order if index % 2 == 0 else list(reversed(order))


def summarize(records, entry, protocol, failures):
    """Compare frozen predictions with all retained native block statistics."""
    blocks = {}
    for row in records:
        if row["warmup"] or row.get("diagnostic"):
            continue
        if not row["measurement_valid"]:
            continue
        blocks.setdefault(row["entry"], {}).setdefault(
            row["round"], []
        ).append(row)
    floors, medians, eligible = {}, {}, []
    failed_entries = {failure["entry"] for failure in failures}
    for identifier, rounds in blocks.items():
        floors[identifier] = {
            str(index): statistics.fmean(
                sorted(row["kernel_ms"] for row in samples)[
                    : protocol["lowest_count"]
                ]
            )
            for index, samples in rounds.items()
        }
        medians[identifier] = statistics.median(
            row["kernel_ms"] for samples in rounds.values() for row in samples
        )
        if (
            identifier not in failed_entries
            and len(rounds) == protocol["rounds"]
            and all(
                len(samples) == protocol["solves_per_block"]
                and all(row["numerical"]["passed"] for row in samples)
                for samples in rounds.values()
            )
        ):
            eligible.append(identifier)
    paired = {}
    baseline = entry["baseline"]
    if baseline in eligible:
        for identifier, values in floors.items():
            common = sorted(set(values) & set(floors[baseline]))
            paired[identifier] = [
                values[index] / floors[baseline][index] for index in common
            ]
    measured = {
        key: statistics.median(values)
        for key, values in paired.items()
        if key in eligible and key != entry["duplicate_id"]
    }
    predicted = entry["ranking"]["default"]
    result = dict(
        median_kernel_ms=medians,
        block_mean_lowest=floors,
        paired_baseline_ratios=paired,
        median_paired_ratios=measured,
        eligible_candidates=eligible,
        frozen_ranking=entry["ranking"],
        all_records_retained=True,
        failures=failures,
        validation_status="PASS"
        if not failures and len(eligible) == (len(entry["candidates"]) + 1)
        else "FAILED_OR_INCOMPLETE",
        interpretation="Whole-solver timing versus conditional attempted-step "
        "prediction; tolerance agreement is not proof of global accuracy",
    )
    if measured and predicted in measured:
        fastest = min(measured.values())
        result.update(
            measured_fastest=[
                key for key, value in measured.items() if value == fastest
            ],
            predicted_choice_relative_regret=measured[predicted] / fastest - 1,
        )
    return result


def run(frozen, output, disassembler):
    """Run a frozen bank only; never feed native observations to prediction."""
    frozen, output = Path(frozen).resolve(), Path(output).resolve()
    manifest = read(frozen / "manifest.json")
    receipt = read(frozen / "freeze_receipt.json")
    if receipt["manifest_sha256"] != file_hash(
        frozen / "manifest.json"
    ) or manifest["runner_sha256"] != file_hash(__file__):
        raise ValueError("Prediction or runner changed after the freeze")
    for asset in manifest["assets"]:
        if file_hash(asset["path"]) != asset["sha256"]:
            raise ValueError("Frozen prediction/input artifact changed")
    for source in manifest["production_sources"]:
        if file_hash(source["original_path"]) != source["sha256"]:
            raise ValueError("Source changed after prediction/native freeze")
    if CUDA_BACKEND != "mlir":
        raise ValueError("This holdout targets the installed MLIR backend")
    if not Path(disassembler).is_file():
        raise ValueError(
            "An explicit native disassembler executable is required"
        )
    output.mkdir(parents=True, exist_ok=False)
    request, protocol = read(manifest["request"]["path"]), manifest["protocol"]
    write(
        output / "run_identity.json",
        dict(
            frozen_manifest_sha256=receipt["manifest_sha256"],
            runner_sha256=file_hash(__file__),
            backend=CUDA_BACKEND,
            backend_version=importlib.metadata.version(
                "cubie-numba-cuda-mlir"
            ),
            disassembler_sha256=file_hash(disassembler),
            device=device_identity(request["hardware"]),
            predicted_inputs_updated=False,
        ),
    )
    previous = get_cache_root_override()
    summaries = []
    try:
        for entry in manifest["workloads"]:
            folder = output / entry["id"]
            folder.mkdir()
            grid = np.load(entry["grid"]["path"])
            inits, params = grid["initial_values"], grid["parameters"]
            identifiers = [entry["baseline"]] + sorted(
                key for key in entry["candidates"] if key != entry["baseline"]
            )
            identifiers.append(entry["duplicate_id"])
            solvers, records, failures = {}, [], []
            reference = None
            randomizer = random.Random(protocol["order_seed"])
            try:
                for identifier in identifiers:
                    candidate_id = (
                        entry["baseline"]
                        if identifier == entry["duplicate_id"]
                        else identifier
                    )
                    candidate = entry["candidates"][candidate_id]
                    native_folder = folder / identifier
                    native_folder.mkdir()
                    solver = None
                    try:
                        solver, _, _ = constructor(
                            request,
                            entry["workload"],
                            candidate,
                            native_folder,
                        )
                        if (
                            workload_identity(solver.system, solver)
                            != entry["semantic_workload"]
                        ):
                            raise ValueError(
                                "Native constructor workload differs"
                            )
                        compiled = native_compile(
                            solver,
                            candidate,
                            request,
                            protocol,
                            inits,
                            params,
                            native_folder,
                            disassembler,
                        )
                        if not compiled["eligible"]:
                            raise ValueError("; ".join(compiled["problems"]))
                        solvers[identifier] = (solver, candidate)
                    except Exception:
                        failures.append(
                            dict(
                                entry=identifier,
                                phase="compile_or_geometry",
                                traceback=traceback.format_exc(),
                            )
                        )
                        if solver is not None:
                            solver.close()
                    write(folder / "failures.json", failures)
                # Warmups establish the explicit baseline reference first.
                jobs = [
                    (True, -1, identifier, protocol["warmups"])
                    for identifier in identifiers
                ]
                jobs += [
                    (False, index, identifier, protocol["solves_per_block"])
                    for index, order in enumerate(
                        block_orders(identifiers, protocol["rounds"])
                    )
                    for identifier in order
                ]
                for warmup, round_index, identifier, count in jobs:
                    if identifier not in solvers:
                        continue
                    solver, candidate = solvers[identifier]
                    for sample in range(count):
                        try:
                            timing, state, status, counters = measure(
                                solver, inits, params, protocol, candidate
                            )
                            if (
                                reference is None
                                and identifier == entry["baseline"]
                            ):
                                reference = state.copy(), status.copy()
                            numerical = numerical_check(
                                state,
                                status,
                                reference,
                                entry["numerical_tolerances"],
                                identifier == entry["duplicate_id"],
                            )
                            numerical["checks"]["counters_off"] = (
                                counters is None
                            )
                            numerical["passed"] = all(
                                numerical["checks"].values()
                            )
                            row = dict(
                                entry=identifier,
                                warmup=warmup,
                                round=round_index,
                                sample=sample,
                                **timing,
                                arrays=sample_arrays(
                                    folder, state, status, counters
                                ),
                                numerical=numerical,
                            )
                            records.append(row)
                            if not numerical["passed"]:
                                failures.append(
                                    dict(
                                        entry=identifier,
                                        phase="numerical",
                                        record=len(records) - 1,
                                        checks=numerical,
                                    )
                                )
                            if not timing["measurement_valid"]:
                                failures.append(
                                    dict(
                                        entry=identifier,
                                        phase="measurement",
                                        record=len(records) - 1,
                                        problems=timing["problems"],
                                    )
                                )
                        except Exception:
                            failures.append(
                                dict(
                                    entry=identifier,
                                    phase="solve",
                                    round=round_index,
                                    sample=sample,
                                    warmup=warmup,
                                    traceback=traceback.format_exc(),
                                )
                            )
                        write(folder / "records.json", records)
                        write(folder / "failures.json", failures)
                    time.sleep(randomizer.uniform(*protocol["idle_seconds"]))
                summary = summarize(records, entry, protocol, failures)
                write(folder / "summary.json", summary)
                summaries.append(dict(workload=entry["workload"], **summary))
            finally:
                for solver, _ in solvers.values():
                    solver.close()
        if protocol["diagnostic_counters"]:
            for entry in manifest["workloads"]:
                folder = output / entry["id"]
                with np.load(entry["grid"]["path"]) as grid:
                    inits, params = grid["initial_values"], grid["parameters"]
                diagnostics(
                    entry,
                    request,
                    protocol,
                    inits,
                    params,
                    folder,
                    disassembler,
                )
    finally:
        set_cache_root(previous)
    result = dict(
        status="PASS"
        if all(x["validation_status"] == "PASS" for x in summaries)
        else "FAILED_OR_INCOMPLETE",
        workloads=summaries,
        prediction_inputs_updated=False,
    )
    write(output / "receipt.json", result)
    return result


def diagnostics(entry, request, protocol, inits, params, folder, disassembler):
    """Capture separately compiled counters only after the timing bank."""
    root = folder / "instrumented_diagnostics"
    root.mkdir()
    records = []
    for identifier, candidate in entry["candidates"].items():
        destination = root / identifier
        destination.mkdir()
        solver = None
        try:
            solver, _, _ = constructor(
                request,
                entry["workload"],
                candidate,
                destination,
                instrumented=True,
            )
            compiled = native_compile(
                solver,
                candidate,
                request,
                protocol,
                inits,
                params,
                destination,
                disassembler,
            )
            if not compiled["eligible"]:
                raise ValueError("Instrumented geometry is not admitted")
            timing, state, status, counters = measure(
                solver, inits, params, protocol, candidate
            )
            if counters is None:
                raise ValueError("Instrumented result did not expose counters")
            records.append(
                dict(
                    entry=identifier,
                    diagnostic_only=True,
                    prediction_input=False,
                    timing=timing,
                    arrays=sample_arrays(root, state, status, counters),
                )
            )
        except Exception:
            records.append(
                dict(
                    entry=identifier,
                    diagnostic_only=True,
                    failure=traceback.format_exc(),
                )
            )
        finally:
            if solver is not None:
                solver.close()
        write(root / "records.json", records)


def main():
    """Expose separate CPU freeze and explicitly requested GPU run actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--results", type=Path, nargs="+")
    parser.add_argument("--baselines", nargs="+")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--nvdisasm", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        if not args.request or not args.results or not args.baselines:
            parser.error(
                "Freeze requires request, results and explicit baselines"
            )
        result = freeze(
            args.request,
            args.results,
            args.baselines,
            args.out,
            read(args.protocol) if args.protocol else None,
        )
    else:
        if not args.frozen or not args.nvdisasm:
            parser.error("Run requires frozen predictions and nvdisasm")
        result = run(args.frozen, args.out, args.nvdisasm)
    print(json.dumps(result))
    if result.get("status") == "FAILED_OR_INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
