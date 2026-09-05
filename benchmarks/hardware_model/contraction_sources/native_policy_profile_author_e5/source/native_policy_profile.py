"""Reproduce six frozen counter-free solver images for diagnostic profiling."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import traceback
import uuid

from cubin_local_symbol_equivalence import (
    compare_cubins,
    NAMING_SOURCE_SHA256,
)

from attrs import fields
import numpy as np

import cubie
from cubie import Solver
from cubie.cache_root import set_cache_root
from cubie.cuda_backend import CUDA_BACKEND
from cubie.cuda_simsafe import UnrollFlags, cuda
from cubie.result_codes import CUBIE_RESULT_CODES
from benchmarks import placement_landscape as landscape


GROUPS = (
    "unroll_stage",
    "unroll_step_element",
    "unroll_accumulator",
    "unroll_solver_element",
    "unroll_norms",
    "unroll_other_small",
    "unroll_newton_exits",
    "unroll_krylov_exits",
)
CASES = {
    "workload_001": ("kvaerno3", "lu"),
    "workload_003": ("radau_iia_3", "bicgstab"),
    "workload_006": ("rk23", None),
}
CANDIDATES = (
    "source_0000_b128_s102400",
    "source_0002_b128_s102400",
)
RESOURCE_METHODS = {
    "regs": "get_regs_per_thread",
    "shared": "get_shared_mem_per_block",
    "local": "get_local_mem_per_thread",
    "const": "get_const_mem_size",
    "maxthreads": "get_max_threads_per_block",
}


def file_hash(path):
    """Hash exact bytes, without source or disassembly normalization."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    """Use the frozen bank's canonical JSON fingerprint convention."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def read(path):
    """Read exact JSON assets, including compressed source graphs."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    """Write a complete strict JSON record."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def asset(path):
    """Bind an existing input file."""
    return dict(path=str(Path(path).resolve()), sha256=file_hash(path))


def checked(record):
    """Refuse changed input bytes before consuming them."""
    if file_hash(record["path"]) != record["sha256"]:
        raise ValueError("Asset hash mismatch: " + record["path"])
    return Path(record["path"])


def compiler_record(value):
    """Preserve the exact original compiler option representation."""
    if isinstance(value, UnrollFlags):
        return dict(
            type="UnrollFlags",
            fields={
                field.name: compiler_record(getattr(value, field.name))
                for field in fields(type(value))
            },
        )
    if isinstance(value, (set, frozenset)):
        return dict(type=type(value).__name__, members=sorted(value))
    if isinstance(value, dict):
        return {key: compiler_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compiler_record(item) for item in value]
    return value


def array_digest(arrays):
    """Reproduce the original bank's whole-array content fingerprint."""
    result = hashlib.sha256()
    for name, array in arrays.items():
        result.update(
            digest(
                dict(
                    name=name,
                    dtype=array.dtype.str,
                    shape=list(array.shape),
                )
            ).encode()
        )
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def reference_arrays(record):
    """Load a counter-free, finite, successful own-candidate reference."""
    with np.load(checked(record), allow_pickle=False) as data:
        if set(data.files) != {"state", "status"}:
            raise ValueError("Reference must contain state and status only")
        arrays = {name: data[name].copy() for name in ("state", "status")}
    if array_digest(arrays) != record["content_sha256"]:
        raise ValueError("Original whole-array content fingerprint differs")
    if (
        arrays["state"].dtype != np.float32
        or not arrays["state"].size
        or not arrays["status"].size
        or not np.isfinite(arrays["state"]).all()
        or not np.all(arrays["status"] == int(CUBIE_RESULT_CODES.SUCCESS))
    ):
        raise ValueError("Original arrays violate the FP32/status contract")
    return arrays


def runtime_sources(records):
    """Check loaded production and constructor sources against the bank."""
    root = Path(cubie.__file__).resolve().parents[2]
    observed = []
    for record in records:
        path = root / record["relative"]
        if file_hash(path) != record["sha256"]:
            raise ValueError("Frozen runtime source mismatch: " + str(path))
        observed.append(dict(path=str(path), sha256=file_hash(path)))
    expected_landscape = next(
        row
        for row in records
        if row["relative"] == "benchmarks/placement_landscape.py"
    )
    if file_hash(landscape.__file__) != expected_landscape["sha256"]:
        raise ValueError("Imported landscape is not the frozen constructor")
    return observed


def case_kwargs(request, workload, candidate):
    """Reproduce the saved six-case public constructor recipe."""
    levels = candidate["levels"]
    if levels not in (["full"] * 8, ["count1"] + ["full"] * 7):
        raise ValueError("Only the frozen full/stage-count-one pair is valid")
    if set(candidate["locations"].values()) != {"local"}:
        raise ValueError("This diagnostic cohort requires local placements")
    kwargs = landscape.solver_kwargs(request["system"], workload["algorithm"])
    kwargs.update(request.get("solver_settings", {}))
    inner = workload["inner"]
    if inner is None:
        kwargs.pop("linear_correction_type", None)
    elif inner in ("lu", "bicgstab"):
        kwargs["linear_correction_type"] = inner
    else:
        raise ValueError("Inner solver is outside the frozen six cases")
    kwargs.update(candidate["locations"])
    kwargs["unroll"] = UnrollFlags(
        **{
            key: (True, None) if level == "full" else (True, 1)
            for key, level in zip(GROUPS, levels)
        }
    )
    if kwargs.get("output_types") != ["state"]:
        raise ValueError("The original counter-free output ABI must match")
    constants = request.get(
        "system_constants",
        landscape.SYSTEMS[request["system"]].get(
            "constants",
        ),
    )
    return kwargs, constants


def prepare(frozen, native, disassembler, output):
    """Freeze the six existing images and references without CUDA work."""
    frozen, native, output = map(Path, (frozen, native, output))
    output.mkdir(parents=True, exist_ok=False)
    manifest_record = asset(frozen / "manifest.json")
    manifest = read(checked(manifest_record))
    freeze = read(frozen / "freeze_receipt.json")
    identity_record = asset(native / "run_identity.json")
    identity = read(checked(identity_record))
    if not (
        manifest_record["sha256"]
        == freeze["manifest_sha256"]
        == identity["frozen_manifest_sha256"]
    ):
        raise ValueError("Prediction freeze and native bank identity differ")
    if file_hash(disassembler) != identity["disassembler_sha256"]:
        raise ValueError("Disassembler differs from the original native bank")
    request = read(checked(manifest["request"]))
    if request["system"] != "lorenz":
        raise ValueError("This exact native profile cohort is Lorenz")
    if manifest["protocol"]["diagnostic_counters"]:
        raise ValueError("The original native bank must be counter-free")
    sources = []
    for row in manifest["production_sources"]:
        original = checked(row)
        relative = original.relative_to(frozen / "source_snapshot")
        if relative.parts[0] == "src" or relative.as_posix() == (
            "benchmarks/placement_landscape.py"
        ):
            sources.append(
                dict(
                    relative=relative.as_posix(),
                    sha256=row["sha256"],
                    original=asset(original),
                )
            )
    runtime = runtime_sources(sources)
    cases = {}
    for entry in manifest["workloads"]:
        if entry["id"] not in CASES:
            continue
        expected = CASES[entry["id"]]
        if (
            entry["workload"]["algorithm"],
            entry["workload"]["inner"],
        ) != expected:
            raise ValueError("Native workload label differs from exact cohort")
        records_asset = asset(native / entry["id"] / "records.json")
        records = read(checked(records_asset))
        for candidate_id in CANDIDATES:
            candidate = entry["candidates"][candidate_id]
            kwargs, constants = case_kwargs(
                request,
                entry["workload"],
                candidate,
            )
            if compiler_record(constants) != entry["constructor_constants"]:
                raise ValueError("Frozen constructor constants differ")
            folder = native / entry["id"] / candidate_id
            compile_asset = asset(folder / "compile.json")
            compiled = read(checked(compile_asset))
            disassembly = asset(folder / "disassembly_command.json")
            original_disassembly = read(checked(disassembly))
            if original_disassembly["returncode"] != 0:
                raise ValueError("Original disassembly command failed")
            cubin, sass = (
                asset(folder / "kernel.cubin"),
                asset(
                    folder / "kernel.sass.gz",
                ),
            )
            if (
                cubin["sha256"] != compiled["cubin_sha256"]
                or sass["sha256"] != compiled["sass_sha256"]
                or not compiled["eligible"]
            ):
                raise ValueError("Original compiled image failed its binding")
            rows = [row for row in records if row["entry"] == candidate_id]
            protocol = manifest["protocol"]
            expected_rows = protocol["warmups"] + (
                protocol["rounds"] * protocol["solves_per_block"]
            )
            if len(rows) != expected_rows or not all(
                row["measurement_valid"] for row in rows
            ):
                raise ValueError("Original same-candidate samples incomplete")
            if any(
                row["geometries"] != compiled["geometries"] for row in rows
            ):
                raise ValueError("Original sample geometries vary")
            if len(compiled["geometries"]) != 1 or (
                compiled["geometries"][0]["actual"]["waves"] < 2
            ):
                raise ValueError(
                    "Original capture needs one chunk and two waves"
                )
            if len({row["arrays"]["content_sha256"] for row in rows}) != 1:
                raise ValueError("Original same-candidate arrays vary")
            references = {row["arrays"]["path"]: row["arrays"] for row in rows}
            for reference in references.values():
                reference_arrays(reference)
            reference = rows[0]["arrays"]
            checked(entry["grid"])
            graph = read(checked(candidate["graph"]))
            if graph["policy"] != [
                dict(
                    group=name,
                    level=level,
                    flag=[True, None if level == "full" else 1],
                )
                for name, level in zip(
                    GROUPS,
                    candidate["levels"],
                )
            ]:
                raise ValueError("Original graph policy is not this recipe")
            key = entry["id"] + "_" + candidate_id
            cases[key] = dict(
                workload=entry["workload"],
                candidate=candidate,
                constructor_kwargs=compiler_record(kwargs),
                constructor_constants=compiler_record(constants),
                grid=entry["grid"],
                compile=compile_asset,
                cubin=cubin,
                sass=sass,
                disassembly=disassembly,
                records=records_asset,
                reference=reference,
                exact_original_samples=len(rows),
                original_cross_candidate_passed=all(
                    row["numerical"]["passed"] for row in rows
                ),
                shared_stride_bytes=graph["candidate_construction"][
                    "shared_stride_bytes"
                ],
            )
    if len(cases) != 6:
        raise ValueError("The complete six-case cohort is required")
    naming_source = asset(
        importlib.util.find_spec("numba_cuda_mlir.mlir_lowering").origin
    )
    if naming_source["sha256"] != NAMING_SOURCE_SHA256:
        raise ValueError("Installed constant-array naming source differs")
    prepared = dict(
        kind="frozen_native_policy_profile",
        cubin_equivalence=dict(
            comparator=asset(inspect.getfile(compare_cubins)),
            naming_source=naming_source,
            rule="Exact cubin or typed ELF section-bound local constant-array renumbering",
        ),
        schema=1,
        wrapper_sha256=file_hash(__file__),
        original_manifest=manifest_record,
        native_identity=identity_record,
        request=manifest["request"],
        protocol=manifest["protocol"],
        sources=sources,
        disassembler=asset(disassembler),
        cases=cases,
        capture=dict(
            kernel_filter="regex:.*Lorenz_ltoon.*",
            launch_skip=1,
            launch_count=1,
            replay_mode="application",
            warmup_solves=1,
            captured_solves=1,
            expected_chunks_per_solve=1,
        ),
        interpretation="Diagnostic same-image profiling; cross-policy "
        "numerical failures remain failures and are not prediction inputs",
    )
    write(output / "prepared.json", prepared)
    receipt = dict(
        status="CPU_PREPARED",
        prepared_sha256=file_hash(
            output / "prepared.json",
        ),
        wrapper_sha256=file_hash(__file__),
        cases=len(cases),
        runtime_sources=runtime,
        gpu_launches=0,
        native_compilations=0,
    )
    write(output / "preparation_receipt.json", receipt)
    return receipt


def load_prepared(path):
    """Verify the preparation receipt and all selected immutable assets."""
    path = Path(path)
    receipt = read(path.parent / "preparation_receipt.json")
    if file_hash(path) != receipt["prepared_sha256"]:
        raise ValueError("Prepared manifest does not match its receipt")
    prepared = read(path)
    if file_hash(__file__) != prepared["wrapper_sha256"]:
        raise ValueError("Profile wrapper differs from its prepared source")
    for key in (
        "original_manifest",
        "native_identity",
        "request",
        "disassembler",
    ):
        checked(prepared[key])
    for source in prepared["sources"]:
        checked(source["original"])
    runtime_sources(prepared["sources"])
    equivalence = prepared["cubin_equivalence"]
    checked(equivalence["comparator"])
    checked(equivalence["naming_source"])
    if (
        file_hash(inspect.getfile(compare_cubins))
        != equivalence["comparator"]["sha256"]
    ):
        raise ValueError("Loaded ELF comparator differs from preparation")
    if (
        file_hash(
            importlib.util.find_spec("numba_cuda_mlir.mlir_lowering").origin
        )
        != NAMING_SOURCE_SHA256
    ):
        raise ValueError("Loaded compiler naming source differs")
    request = read(checked(prepared["request"]))
    for case in prepared["cases"].values():
        for key in (
            "grid",
            "compile",
            "cubin",
            "sass",
            "records",
            "disassembly",
        ):
            checked(case[key])
        checked(case["candidate"]["graph"])
        reference_arrays(case["reference"])
        kwargs, constants = case_kwargs(
            request, case["workload"], case["candidate"]
        )
        if (
            compiler_record(kwargs) != case["constructor_kwargs"]
            or compiler_record(constants) != case["constructor_constants"]
        ):
            raise ValueError("Prepared public constructor recipe changed")
    return prepared


def geometry(solver, case, compiled, protocol):
    """Require original compiled geometry and at least two occupancy waves."""
    run_params = solver.kernel.run_params
    if (
        run_params.num_chunks != 1
        or int(solver.kernel.single_integrator.threads_per_step) != 1
    ):
        raise ValueError("The original one-chunk, one-thread geometry changed")
    runs = int(run_params[0].runs)
    block = case["candidate"]["geometry"]["block_threads"]
    actual = landscape.launch_geometry(solver, block, runs)
    rows = [dict(chunk=0, runs=runs, actual=actual)]
    if (
        rows != compiled["geometries"]
        or runs != protocol["n_runs"]
        or actual["blocksize"] != block
        or actual["waves"] < 2
        or actual["blocks_per_sm"] < 1
        or actual["dynshared"]
        != max(4, landscape.bytes_per_run(solver) * block)
    ):
        raise ValueError(
            "Actual geometry differs from the frozen native image"
        )
    return rows


def verify_device(identity):
    """Match installed compiler, device identity and hardware capacities."""
    if CUDA_BACKEND != identity["backend"]:
        raise ValueError("Active CUDA backend differs from the native bank")
    if (
        importlib.metadata.version("cubie-numba-cuda-mlir")
        != (identity["backend_version"])
    ):
        raise ValueError("Installed MLIR compiler version differs")
    device = cuda.get_current_device()
    fields = dict(
        multiprocessor_count="MULTIPROCESSOR_COUNT",
        warp_size="WARP_SIZE",
        max_threads_per_block="MAX_THREADS_PER_BLOCK",
        max_threads_per_sm="MAX_THREADS_PER_MULTIPROCESSOR",
        registers_per_sm="MAX_REGISTERS_PER_MULTIPROCESSOR",
    )
    actual = dict(
        name=str(device.name),
        compute_capability=list(device.compute_capability),
        capacities={
            key: int(getattr(device, field)) for key, field in fields.items()
        },
    )
    if actual != identity["device"]:
        raise ValueError("Device differs from the original native bank")
    return actual


def compile_exact(solver, case, protocol, inits, params, prepared, output):
    """Refuse image, resource, flag or geometry drift before the warmup."""
    compiled = read(checked(case["compile"]))
    solver.compile(
        inits,
        params,
        duration=protocol["duration"],
        t0=protocol["t0"],
        grid_type="verbatim",
    )
    dispatcher = solver.kernel.kernel
    ((signature, specialization),) = dispatcher.overloads.items()
    library = specialization._codelibrary
    library.get_cufunc().set_shared_memory_carveout(
        compiled["carveout"]["integer_percent"],
    )
    cubin = (
        bytes(library.get_cubin().code)
        if hasattr(library, "get_cubin")
        else bytes(library._cubin)
    )
    path = output / "kernel.cubin"
    path.write_bytes(cubin)
    process = subprocess.run(
        [str(checked(prepared["disassembler"])), "-c", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    write(
        output / "disassembly.json",
        dict(
            command=process.args,
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        ),
    )
    process.check_returncode()
    with gzip.open(checked(case["sass"]), "rt") as handle:
        original_sass = handle.read()
    original_stdout = read(checked(case["disassembly"]))["stdout"]
    observed = dict(
        cubin_sha256=file_hash(path),
        sass_text_sha256=hashlib.sha256(process.stdout.encode()).hexdigest(),
        compiler_kwargs=compiler_record(dict(solver.kernel.jit_kwargs)),
        native_attributes={
            name: int(getattr(dispatcher, method)(signature))
            for name, method in RESOURCE_METHODS.items()
        },
    )
    try:
        cubin_identity = compare_cubins(
            checked(case["cubin"]).read_bytes(),
            cubin,
            prepared["cubin_equivalence"]["naming_source"]["sha256"],
        )
    except ValueError as error:
        cubin_identity = dict(
            admitted=False,
            reason=str(error),
            raw_bytes_equal=cubin == checked(case["cubin"]).read_bytes(),
        )
    observed["cubin_identity"] = cubin_identity
    checks = dict(
        cubin=cubin_identity["admitted"],
        sass=process.stdout == original_sass == original_stdout,
        compiler_kwargs=observed["compiler_kwargs"]
        == compiled["compiler_kwargs"],
        native_attributes=observed["native_attributes"]
        == compiled["native_attributes"],
    )
    write(
        output / "native_identity_checks.json",
        dict(
            checks=checks,
            observed=observed,
            expected=compiled,
        ),
    )
    if not all(checks.values()):
        raise ValueError(
            "Native identity mismatch: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    observed["geometries"] = geometry(solver, case, compiled, protocol)
    return compiled, observed


def run(prepared_path, case_id, output):
    """Run one warmup and one diagnostic solve in a unique replay folder."""
    prepared = load_prepared(prepared_path)
    case = prepared["cases"][case_id]
    for key in ("grid", "compile", "cubin", "sass", "records"):
        checked(case[key])
    checked(case["candidate"]["graph"])
    reference = reference_arrays(case["reference"])
    output = Path(output) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + f"_pid{os.getpid()}_"
        + uuid.uuid4().hex
    )
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="STARTED",
        case=case_id,
        prepared=asset(prepared_path),
        wrapper_sha256=file_hash(__file__),
        solves=[],
        original_cross_candidate_passed=case[
            "original_cross_candidate_passed"
        ],
    )
    solver = None
    try:
        identity = read(checked(prepared["native_identity"]))
        receipt["device"] = verify_device(identity)
        request = read(checked(prepared["request"]))
        kwargs, constants = case_kwargs(
            request, case["workload"], case["candidate"]
        )
        if (
            compiler_record(kwargs) != case["constructor_kwargs"]
            or compiler_record(constants) != case["constructor_constants"]
        ):
            raise ValueError("Public constructor recipe changed")
        set_cache_root(output / "codegen")
        solver = Solver(
            landscape.SYSTEMS[request["system"]]["build"](), **kwargs
        )
        if constants:
            solver.update(constants)
        solver.kernel.single_integrator.device_function
        if solver.kernel.kernel.overloads:
            raise ValueError("Native specialization preceded explicit compile")
        if landscape.bytes_per_run(solver) != case["shared_stride_bytes"]:
            raise ValueError(
                "Source shared stride differs from saved candidate"
            )
        with np.load(checked(case["grid"]), allow_pickle=False) as data:
            inits, params = data["initial_values"], data["parameters"]
        protocol = prepared["protocol"]
        compiled, receipt["native"] = compile_exact(
            solver,
            case,
            protocol,
            inits,
            params,
            prepared,
            output,
        )
        for phase in ("warmup", "capture"):
            before = geometry(solver, case, compiled, protocol)
            result = solver.solve(
                inits,
                params,
                duration=protocol["duration"],
                t0=protocol["t0"],
                grid_type="verbatim",
                blocksize=case["candidate"]["geometry"]["block_threads"],
                nan_error_trajectories=False,
            )
            arrays = dict(
                state=np.array(result.state[-1]),
                status=np.array(result.status_codes),
            )
            array_path = output / (phase + "_arrays.npz")
            np.savez_compressed(array_path, **arrays)
            checks = {
                name: value.dtype == reference[name].dtype
                and value.shape == reference[name].shape
                and value.tobytes(order="C")
                == reference[name].tobytes(order="C")
                for name, value in arrays.items()
            }
            checks["counter_free"] = result.iteration_counters is None
            receipt["solves"].append(
                dict(
                    phase=phase,
                    before_geometry=before,
                    after_geometry=geometry(solver, case, compiled, protocol),
                    arrays=asset(array_path),
                    content_sha256=array_digest(arrays),
                    exact_own_candidate_checks=checks,
                )
            )
            write(output / "receipt.json", receipt)
            if not all(checks.values()):
                raise ValueError(
                    "Whole-array own-candidate mismatch: " + phase
                )
        receipt["status"] = (
            "EXACT_NATIVE_REPRODUCTION_PASS"
            if receipt["native"]["cubin_identity"]["raw_bytes_equal"]
            else "SECTION_BOUND_NATIVE_REPRODUCTION_PASS"
        )
    except Exception:
        receipt["status"] = "FAILED"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        write(output / "receipt.json", receipt)
        if solver is not None:
            solver.close()
    return dict(output=str(output), **receipt)


def main():
    """Separate CPU preparation from explicitly requested native execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--frozen")
    parser.add_argument("--native")
    parser.add_argument("--disassembler")
    parser.add_argument("--prepared")
    parser.add_argument("--case")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.frozen, args.native, args.disassembler, args.out)
    elif args.check:
        prepared = load_prepared(args.prepared)
        result = dict(
            status="CPU_PREPARED_INTEGRITY_PASS",
            cases=len(prepared["cases"]),
            gpu_launches=0,
        )
        write(Path(args.out), result)
    else:
        result = run(args.prepared, args.case, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
