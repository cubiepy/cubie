"""Explicit native worker for the instruction-sharing probe.

This file is copied into a fresh evidence directory by the source-only
controller. Importing or executing it is an explicit CUDA operation.
"""

from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


OUTPUT = Path(__file__).resolve().parent
REQUEST = json.loads((OUTPUT / "request.json").read_text(encoding="utf-8"))
GENERATOR = Path(REQUEST["generator_path"])
if (
    hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    != (REQUEST["generator_sha256"])
):
    raise RuntimeError("Controller bytes differ from the prepared snapshot")
if (
    hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    != (REQUEST["worker_sha256"])
):
    raise RuntimeError("Worker bytes differ from the prepared snapshot")
sys.path.insert(0, REQUEST["research_root"])

import numpy as np  # noqa: E402
from cuda.bindings import driver  # noqa: E402
import cubie  # noqa: E402
import cubie.cuda_simsafe as cuda_helpers  # noqa: E402
from cubie._utils import package_source_hash  # noqa: E402
from benchmarks import unroll_landscape  # noqa: E402
from benchmarks.hardware_model import hardware_probes as hardware  # noqa: E402
from benchmarks.hardware_model.instruction_sharing_probe import (  # noqa: E402
    MIRROR,
    MODES,
    check_native,
    digest,
    write_json,
)


def normalized(value):
    """Normalize NumPy values, sets and tuples before identity checks."""
    def canonical(item):
        if isinstance(item, (set, frozenset)):
            return sorted(
                (canonical(element) for element in item),
                key=lambda element: json.dumps(
                    element, default=hardware._json_default, sort_keys=True,
                ),
            )
        if isinstance(item, dict):
            return {key: canonical(element) for key, element in item.items()}
        if isinstance(item, (tuple, list)):
            return [canonical(element) for element in item]
        return item

    return json.loads(json.dumps(canonical(value),
                                default=hardware._json_default))


def file_record(path):
    """Describe exact bytes retained in this evidence directory."""
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def source_record(module):
    """Bind actual imported helper paths rather than inferred repo paths."""
    return file_record(inspect.getfile(module))


def provenance(nvdisasm):
    """Capture installed compiler/source/device identity and raw clocks."""
    manifest = hardware._manifest(SimpleNamespace(**REQUEST), nvdisasm)
    manifest["versions"]["numba"] = importlib.metadata.version("numba")
    identity = dict(
        generator_sha256=REQUEST["generator_sha256"],
        worker_sha256=REQUEST["worker_sha256"],
        kernel_source_sha256=REQUEST["kernel_source_sha256"],
        actual_cubie_source_hash=package_source_hash(),
        actual_cubie_root=str(Path(cubie.__file__).resolve().parent),
        compiler=unroll_landscape.compiler_identity(),
        jit_kwargs=hardware.get_jit_kwargs(),
        imported_sources=dict(
            hardware=source_record(hardware),
            unroll=source_record(unroll_landscape),
            cuda_helpers=source_record(cuda_helpers),
            compiled_cubin=source_record(hardware._compiled_cubin),
        ),
        versions=manifest["versions"],
        device_name=manifest["device_name"],
        compute_capability=manifest["compute_capability"],
        device_attributes=manifest["device_attributes"],
        nvdisasm=manifest["nvdisasm"],
    )
    manifest["compilation_identity"] = normalized(identity)
    return normalized(manifest)


def capacity_reservation(maximum, reserved, static, optin, unit):
    """Derive minimum bytes excluding two blocks at maximum SM capacity."""
    values = (maximum, reserved, static, optin, unit)
    if any(type(value) is not int for value in values):
        raise ValueError("Shared capacity inputs must be exact integers")
    if min(maximum, optin, unit) <= 0 or min(reserved, static) < 0:
        raise ValueError("Invalid shared capacity or allocation input")
    target = (maximum // (2 * unit) + 1) * unit
    dynamic = max(0, target - unit + 1 - reserved - static)
    allocation = ((reserved + static + dynamic + unit - 1) // unit) * unit
    if static + dynamic > optin or allocation > maximum:
        raise ValueError("One-block reservation exceeds a device limit")
    if 2 * allocation <= maximum:
        raise ValueError("Reservation does not exclude a second block")
    predecessor = None
    if dynamic:
        predecessor = (
            (reserved + static + dynamic - 1 + unit - 1) // unit
        ) * unit
        if 2 * predecessor > maximum:
            raise ValueError("Reservation is not the minimum byte request")
    return dict(
        maximum_shared_bytes_per_sm=maximum,
        driver_reserved_bytes_per_block=reserved,
        static_shared_bytes_per_block=static,
        optin_bytes_per_block=optin,
        allocation_unit_bytes=unit,
        target_allocation_bytes=target,
        dynamic_shared_bytes=dynamic,
        allocated_bytes_per_block=allocation,
        allocation_at_one_less_dynamic_byte=predecessor,
        two_blocks_exceed_maximum=(2 * allocation > maximum),
        maximum_resident_blocks_from_shared=maximum // allocation,
    )


def unused_shared_proof(analysis, resources):
    """Admit only known register/control/constant/global native operations."""
    allowed = set(
        "MOV S2R ISETP IMAD LOP3 I2FP BRA SEL ULDC FFMA UIADD3 CALL "
        "FADD SHF LEA IADD3 STG EXIT NOP".split()
    )
    if (resources["static_shared_bytes"] != 0 or
            set(analysis["whole_opcounts"]) - allowed):
        raise ValueError("Native shared access or unknown operation")
    return dict(
        static_shared_bytes=0,
        whole_native_opcounts=analysis["whole_opcounts"],
        semantics="Only register/control, constant loads and global stores",
        calls="CFG-proved nonreturning local exits, not external helpers",
        reserved_dynamic_bytes_accessed=0,
    )


def physical_geometry(function, resources, analysis):
    """Exclude two blocks independently of the preferred shared carveout."""
    device = hardware.cuda.get_current_device()
    context = hardware.cuda.current_context()
    if tuple(device.compute_capability) != (8, 9):
        raise ValueError("Shared allocation rule is sourced for SM89")
    warp_size = int(device.WARP_SIZE)
    block_size = REQUEST["block_size"]
    if (warp_size != 32 or block_size != 256 or
            REQUEST["active_lanes"] != block_size or
            block_size > int(device.MAX_THREADS_PER_BLOCK)):
        raise ValueError("Need one full eight-warp block")
    if type(REQUEST["waves"]) is not int or REQUEST["waves"] < 2:
        raise ValueError("Need at least two complete occupancy waves")
    # CUDA 13.3 cuda_occupancy.h:620-639, compute-major 8 case.
    unit = 128
    plan = capacity_reservation(
        int(device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
        int(device.RESERVED_SHARED_MEMORY_PER_BLOCK),
        resources["static_shared_bytes"],
        int(device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),
        unit,
    )
    plan["allocation_unit_source"] = (
        "CUDA 13.3 cuda_occupancy.h:620-639 "
        "cudaOccSMemAllocationGranularity, compute-major 8"
    )
    plan["unused_native_reservation"] = unused_shared_proof(
        analysis, resources,
    )
    attribute = driver.CUfunction_attribute
    settings = (
        (attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
         plan["dynamic_shared_bytes"]),
        (attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0),
    )
    readbacks = {}
    for key, value in settings:
        (status,) = driver.cuFuncSetAttribute(function.handle, key, value)
        if status.value:
            raise RuntimeError(f"Function attribute setter failed: {status}")
        status, observed = driver.cuFuncGetAttribute(key, function.handle)
        if status.value or int(observed) != value:
            raise RuntimeError("Function attribute readback differs")
        readbacks[key.name] = int(observed)
    native_blocks = int(context.get_active_blocks_per_multiprocessor(
        function, block_size, 0,
    ))
    resident_blocks = int(context.get_active_blocks_per_multiprocessor(
        function, block_size, plan["dynamic_shared_bytes"],
    ))
    if native_blocks < 1 or resident_blocks != 1:
        raise ValueError("Driver does not admit exactly one reserved block")
    sms = int(device.MULTIPROCESSOR_COUNT)
    if sms < 1:
        raise ValueError("Invalid queried SM count")
    grid_blocks = REQUEST["waves"] * sms
    return dict(
        block_size=block_size, active_lanes=block_size, warp_size=warp_size,
        sms=sms, grid_blocks=grid_blocks, resident_blocks_per_sm=1,
        resident_warps_per_sm=8, active_warps_per_sm_upper_bound=8,
        requested_resident_warps=8,
        native_resident_blocks_per_sm=native_blocks,
        dynamic_shared_bytes=plan["dynamic_shared_bytes"],
        carveout_preference_percent=0, actual_carveout_bytes=None,
        waves=grid_blocks / sms,
        occupancy_origin=(
            "Maximum shared capacity exclusion plus driver one-block query"
        ),
        reservation_method="minimum allocation excluding two blocks at max",
        capacity_reservation=plan, function_attribute_readbacks=readbacks,
    )


def compile_probe(nvdisasm):
    """Compile one signature and reject invalid native stream structure."""
    path = OUTPUT / "kernel.py"
    if digest(path) != REQUEST["kernel_source_sha256"]:
        raise RuntimeError("Generated kernel source changed")
    specification = importlib.util.spec_from_file_location(
        "sharing_kernel_" + REQUEST["kernel_source_sha256"][:16],
        path,
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    kernel = hardware.cuda.jit(**hardware.get_jit_kwargs())(module.probe)
    signature = (
        np.empty(1, np.float32),
        np.empty(1, np.uint32),
        np.empty(1, np.uint32),
        np.empty(1, np.uint32),
        np.uint32(REQUEST["iterations"]),
        np.uint32(0),
        np.uint32(REQUEST["smid_mask"]),
        np.float32(0.99999994),
        np.float32(1e-7),
    )
    start = time.perf_counter()
    hardware.compile_kernel_specialization(kernel, signature)
    compile_seconds = time.perf_counter() - start
    (compiled,) = kernel.overloads.values()
    cubin, entry = hardware._compiled_cubin(compiled)
    (OUTPUT / "kernel.cubin").write_bytes(cubin)
    ptx = kernel.inspect_asm()
    if isinstance(ptx, dict):
        ptx = "\n".join(ptx.values())
    (OUTPUT / "kernel.ptx").write_text(ptx, encoding="utf-8")
    command = hardware._command([nvdisasm, "-c", OUTPUT / "kernel.cubin"])
    write_json(OUTPUT / "disassembly_command.json", command)
    if command["returncode"]:
        raise RuntimeError(command["stderr"])
    (OUTPUT / "kernel.sass").write_text(command["stdout"], encoding="utf-8")
    parsed = hardware._parse_sass(command["stdout"], entry)
    analysis = check_native(*parsed, REQUEST["operations_per_body"])
    write_json(OUTPUT / "sass_analysis.json", analysis)
    compiled._ensure_kernel_attrs()
    resources = dict(
        entry=entry,
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        registers=int(next(iter(kernel.get_regs_per_thread().values()))),
        local_bytes_per_thread=int(
            next(iter(kernel.get_local_mem_per_thread().values()))
        ),
        static_shared_bytes=int(
            next(iter(kernel.get_shared_mem_per_block().values()))
        ),
        jit_kwargs=hardware.get_jit_kwargs(),
        compile_seconds=compile_seconds,
    )
    if resources["local_bytes_per_thread"]:
        raise ValueError("Native local frame confounds the controlled probe")
    function = compiled._codelibrary.get_cufunc()
    geometry = physical_geometry(function, resources, analysis)
    if (
        geometry["warp_size"] != 32
        or geometry["block_size"] != 256
        or geometry["resident_blocks_per_sm"] != 1
        or geometry["resident_warps_per_sm"] != 8
        or geometry["waves"] < 2
    ):
        raise ValueError("Need one resource-limited eight-warp block per SM")
    write_json(OUTPUT / "resources.json", normalized(resources))
    write_json(OUTPUT / "geometry.json", geometry)
    return kernel, compiled, function, resources, geometry, analysis


def native_identity(kernel, compiled, function, geometry):
    """Read existing binary/handle and driver residency without recompiling."""
    if len(kernel.overloads) != 1:
        raise ValueError("The probe gained an additional native overload")
    cubin, entry = hardware._compiled_cubin(compiled)
    resident = (
        hardware.cuda.current_context().get_active_blocks_per_multiprocessor(
            function,
            geometry["block_size"],
            geometry["dynamic_shared_bytes"],
        )
    )
    return dict(
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        entry=entry,
        function_handle=str(function.handle),
        native_overloads=1,
        resident_blocks_per_sm=int(resident),
        registers=int(next(iter(kernel.get_regs_per_thread().values()))),
        local_bytes_per_thread=int(
            next(iter(kernel.get_local_mem_per_thread().values()))
        ),
        static_shared_bytes=int(
            next(iter(kernel.get_shared_mem_per_block().values()))
        ),
        block_size=geometry["block_size"],
        dynamic_shared_bytes=geometry["dynamic_shared_bytes"],
        grid_blocks=geometry["grid_blocks"],
    )


def validate_arrays(arrays, mode, geometry, mask):
    """Require full observed SM coverage, stable SMIDs and exact selection."""
    size = geometry["grid_blocks"] * geometry["block_size"]
    for name in ("output", "entry_smid", "exit_smid", "selections"):
        dtype = np.float32 if name == "output" else np.uint32
        if arrays[name].shape != (size,) or arrays[name].dtype != dtype:
            raise ValueError(f"Unexpected raw array shape/dtype: {name}")
    if not np.all(np.isfinite(arrays["output"])):
        raise ValueError("Probe output is nonfinite")
    entry = arrays["entry_smid"]
    if not np.array_equal(entry, arrays["exit_smid"]):
        raise ValueError("Observed SMID migration invalidates the contrast")
    block_ids = entry.reshape(geometry["grid_blocks"], geometry["block_size"])
    if not np.all(block_ids == block_ids[:, :1]):
        raise ValueError("Threads in a block report different SMIDs")
    observed = np.unique(entry)
    if len(observed) != geometry["sms"]:
        raise ValueError("The launch did not cover every queried physical SM")
    expected = np.full(size, MODES[mode], dtype=np.uint32)
    if mode == "mixed":
        expected = ((entry & np.uint32(mask)) != 0).astype(np.uint32)
    if not np.array_equal(expected, arrays["selections"]):
        raise ValueError("Runtime stream selection differs from recorded SMID")
    if mode == "mixed" and len(np.unique(expected)) != 2:
        raise ValueError("Observed SMID mask does not split the active device")
    warps = arrays["selections"].reshape(-1, geometry["warp_size"])
    if not np.all(warps == warps[:, :1]):
        raise ValueError("Stream selection diverges within a warp")
    counts = Counter(map(int, block_ids[:, 0]))
    return dict(
        smids=list(map(int, observed)),
        blocks_per_smid={
            str(key): value for key, value in sorted(counts.items())
        },
        selected_warps={
            str(key): int(value)
            for key, value in Counter(map(int, warps[:, 0])).items()
        },
        full_queried_sm_coverage=True,
        observed_entry_exit_migration=False,
        limitation=(
            "Equal entry/exit SMID cannot exclude intermediate migration"
        ),
    )


def load_arrays(record):
    """Read an exact retained output snapshot with no pickle payloads."""
    if digest(record["path"]) != record["sha256"]:
        raise ValueError("Raw output snapshot hash mismatch")
    with np.load(record["path"], allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_ordinary(directory):
    """Revalidate completed mirrored raw work before admitting a profile."""
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "ordinary_complete_counters_pending":
        raise ValueError("A completed ordinary cohort is required")
    request = json.loads((directory / "request.json").read_text())
    for key in (
        "generator_sha256",
        "worker_sha256",
        "kernel_source_sha256",
        "requested_body_kib",
        "operations_per_body",
        "chains",
        "block_size",
        "active_lanes",
        "waves",
        "smid_mask",
    ):
        if request[key] != REQUEST[key]:
            raise ValueError(f"Ordinary source/work setting differs: {key}")
    for record in result["artifacts"].values():
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("Original ordinary artifact changed")
    rows = [
        json.loads(line)
        for line in (directory / "timings.jsonl").read_text().splitlines()
    ]
    if rows != result["rows"]:
        raise ValueError("Raw ordinary timing records differ from result")
    expected = [
        (factor, block, name)
        for factor in (1, 2)
        for block in range(request["blocks"])
        for _ in range(3)
        for name in MIRROR
    ]
    measurements = [row for row in rows if row["phase"] == "measurement"]
    actual = [
        (row["repeat_factor"], row["block"], row["mode"])
        for row in measurements
    ]
    if actual != expected:
        raise ValueError("Ordinary mirrored sample membership/order differs")
    references = {}
    for row in rows:
        arrays = load_arrays(row["snapshot"])
        coverage = validate_arrays(
            arrays, row["mode"], result["geometry"], request["smid_mask"]
        )
        if coverage != row["coverage"] or row["native"] != result["native"]:
            raise ValueError("Original coverage/native receipt differs")
        if row["phase"] == "measurement":
            if (
                not np.isfinite(row["elapsed_ms"])
                or row["elapsed_ms"] < request["minimum_ms"]
                or row["iterations"]
                != result["iterations"] * row["repeat_factor"]
            ):
                raise ValueError("Original timing duration/work gate failed")
            prior = references.setdefault(row["iterations"], arrays["output"])
            if not np.array_equal(prior, arrays["output"]):
                raise ValueError(
                    "Original streams do not produce equal results"
                )
    return result, references[result["iterations"]]


def main():
    """Run only the explicitly selected native phase and retain failures."""
    result = dict(status="failed", rows=[], request=REQUEST)
    try:
        if hardware.CUDA_SIMULATION or not hardware.IS_MLIR:
            raise ValueError("Probe requires the real installed MLIR backend")
        if tuple(hardware.cuda.get_current_device().compute_capability) != (
            8,
            9,
        ):
            raise ValueError("Native encoding and gate are restricted to SM89")
        ordinary = reference = None
        if REQUEST["mode"] == "profile":
            ordinary, reference = load_ordinary(REQUEST["ordinary_dir"])
        nvdisasm = hardware._tool("nvdisasm", REQUEST["nvdisasm"])
        manifest = provenance(nvdisasm)
        write_json(OUTPUT / "manifest.json", manifest)
        if (
            ordinary is not None
            and manifest["compilation_identity"]
            != (ordinary["compilation_identity"])
        ):
            raise ValueError("Profile source/compiler/device identity differs")
        kernel, compiled, function, resources, geometry, analysis = (
            compile_probe(nvdisasm)
        )
        native = native_identity(kernel, compiled, function, geometry)
        result.update(
            compilation_identity=manifest["compilation_identity"],
            geometry=geometry,
            native=native,
            analysis=analysis,
        )
        if ordinary is not None:
            for key in (
                "cubin_sha256",
                "entry",
                "native_overloads",
                "resident_blocks_per_sm",
                "registers",
                "local_bytes_per_thread",
                "static_shared_bytes",
                "block_size",
                "dynamic_shared_bytes",
                "grid_blocks",
            ):
                if native[key] != ordinary["native"][key]:
                    raise ValueError(
                        f"Profile binary/resource mismatch: {key}"
                    )
            if geometry != ordinary["geometry"]:
                raise ValueError("Profile compiled geometry differs")
        result["artifacts"] = {
            name: file_record(OUTPUT / name)
            for name in (
                "request.json",
                "benchmark_source.py",
                "worker.py",
                "kernel.py",
                "kernel.cubin",
                "kernel.ptx",
                "kernel.sass",
                "resources.json",
                "geometry.json",
                "sass_analysis.json",
                "manifest.json",
            )
        }
        if REQUEST["mode"] == "compile_only":
            result["status"] = "compiled_unlaunched"
            return
        size = geometry["grid_blocks"] * geometry["block_size"]
        buffers = [
            hardware.cupy.empty(size, dtype=dtype)
            for dtype in (
                np.float32,
                np.uint32,
                np.uint32,
                np.uint32,
            )
        ]
        multiplier, increment = np.float32(0.99999994), np.float32(1e-7)
        result["inputs"] = dict(
            multiplier_bits=int(multiplier.view(np.uint32)),
            increment_bits=int(increment.view(np.uint32)),
            chain_initialization=(
                "float32(i+1) + float32(thread&7)*float32(.001)"
            ),
            mask=REQUEST["smid_mask"],
        )
        if ordinary is not None and result["inputs"] != ordinary["inputs"]:
            raise ValueError("Profile numerical input bits differ")
        references = {}

        def take(name, iterations, phase, factor=1, block=None):
            current = native_identity(kernel, compiled, function, geometry)
            if current != native:
                raise ValueError("Same-function native identity changed")
            before = hardware._clocks()
            arguments = tuple(buffers) + (
                np.uint32(iterations),
                np.uint32(MODES[name]),
                np.uint32(REQUEST["smid_mask"]),
                multiplier,
                increment,
            )
            elapsed = hardware._timed_launch(kernel, geometry, arguments)
            after = hardware._clocks()
            arrays = dict(
                zip(
                    ("output", "entry_smid", "exit_smid", "selections"),
                    (hardware.cupy.asnumpy(item) for item in buffers),
                )
            )
            snapshot = OUTPUT / f"sample_{len(result['rows']):04d}.npz"
            np.savez(snapshot, **arrays)
            row = dict(
                phase=phase,
                repeat_factor=factor,
                block=block,
                mode=name,
                iterations=iterations,
                elapsed_ms=elapsed,
                native=current,
                snapshot=file_record(snapshot),
                clocks_before=before,
                clocks_after=after,
                total_warp_ffmas=(
                    REQUEST["operations_per_body"]
                    * iterations
                    * size
                    // geometry["warp_size"]
                ),
            )
            result["rows"].append(row)
            row["coverage"] = validate_arrays(
                arrays, name, geometry, REQUEST["smid_mask"]
            )
            prior = references.setdefault(iterations, arrays["output"].copy())
            row["exact_equal_across_modes"] = bool(
                np.array_equal(
                    prior,
                    arrays["output"],
                )
            )
            hardware._append_json(OUTPUT / "timings.jsonl", row)
            if not np.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("CUDA event time is nonfinite or nonpositive")
            if not row["exact_equal_across_modes"]:
                raise ValueError("The streams produce different FP32 results")
            if phase == "measurement" and elapsed < REQUEST["minimum_ms"]:
                raise ValueError("Ordinary sample is shorter than 20 ms")
            return elapsed, arrays

        if ordinary is not None:
            iterations = ordinary["iterations"]
            _, arrays = take(REQUEST["profile_mode"], iterations, "profile")
            if not np.array_equal(reference, arrays["output"]):
                raise ValueError(
                    "Profile output differs from original ordinary"
                )
            result.update(
                status="profile_complete_counters_pending",
                iterations=iterations,
                ordinary_result=file_record(
                    Path(REQUEST["ordinary_dir"]) / "result.json"
                ),
                timing_interpretation=(
                    "Profile event duration is not performance data"
                ),
            )
        else:
            iterations = REQUEST["iterations"]
            while True:
                timings = [
                    take(name, iterations, "calibration")[0] for name in MODES
                ]
                if min(timings) >= REQUEST["minimum_ms"]:
                    break
                iterations *= 2
                if iterations >= 2**30:
                    raise ValueError(
                        "Required repeat count exceeds controlled range"
                    )
            result["iterations"] = iterations
            for factor in (1, 2):
                for block in range(REQUEST["blocks"]):
                    for _ in range(3):
                        for name in MIRROR:
                            take(
                                name,
                                iterations * factor,
                                "measurement",
                                factor,
                                block,
                            )
            result["status"] = "ordinary_complete_counters_pending"
            result["timing_interpretation"] = (
                "Six raw samples per mode/block at N and 2N; "
                "no fitted penalties"
            )
        if native_identity(kernel, compiled, function, geometry) != native:
            raise ValueError("Native identity changed after the final launch")
        if digest(GENERATOR) != REQUEST["generator_sha256"]:
            raise ValueError("Controller source changed during the run")
        for record in result["artifacts"].values():
            if digest(record["path"]) != record["sha256"]:
                raise ValueError("Retained source/native artifact changed")
        result["actual_profiled_carveout_bytes"] = None
        result["domain_admission"] = (
            "Pending matched ICC cycle/GCC request counters, actual shared "
            "configuration and executed-PC binding for both native streams"
        )
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        write_json(OUTPUT / "result.json", normalized(result))
        print(json.dumps(dict(status=result["status"], output=str(OUTPUT))))


if __name__ == "__main__":
    main()
