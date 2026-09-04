"""Auditable GPU instruction-stream and dependency microbenchmarks.

Run from the worktree root with its ``src`` on PYTHONPATH. This tool
does not set environment variables, lock clocks, or run Nsight Compute.
Use a fresh output directory. The default instruction sweep has seven
body sizes and one residency target, rather than an exhaustive grid.

Examples
--------
python -m benchmarks.hardware_model.hardware_probes icache --output OUT
python -m benchmarks.hardware_model.hardware_probes icache --output OUT \
    --body-kib 120,124,128,132,136 --resident-warps 8,16,32
python -m benchmarks.hardware_model.hardware_probes fp32 --output OUT \
    --chains 1,8 --block-size 1024 --active-lanes 1
python -m benchmarks.hardware_model.hardware_probes memory --output OUT \
    --space shared --elements 1024 --chains 1,8 --active-lanes 1

``--compile-only`` generates source, PTX, cubin, SASS and resource
records, without launching kernels or allocating device arrays.
``--profile-once --iterations N`` issues exactly one target launch per
case, without calibration or warmup. Restrict all case lists to one
value when profiling. Choose N from a prior ordinary timing record.

Instruction-cache capacity and its sharing domain remain experimental
unknowns. The output records the actual repeated SASS address range;
it does not identify that range with a per-SM cache or fit penalties.
Dependency cycles include loop/address/conversion instructions and
scheduler contention. They are not automatically intrinsic latency.
Memory payloads are 32-bit indices in a randomized pointer ring.
"""

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from cubie.cuda_backend import CUDA_BACKEND, IS_MLIR
from cubie.cuda_simsafe import (
    CUDA_SIMULATION,
    compile_kernel_specialization,
    cuda,
    cupy,
    get_jit_kwargs,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
from lorenz_mean_runtime import _compiled_cubin  # noqa: E402


SOURCES = {
    "ada_resources": (
        "https://docs.nvidia.com/cuda/ada-tuning-guide/index.html"
    ),
    "instruction_bytes": (
        "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html"
    ),
    "cache_hierarchy_and_counters": (
        "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html"
    ),
    "cache_pressure": (
        "https://developer.nvidia.com/blog/"
        "improving-gpu-performance-by-reducing-instruction-cache-misses-2/"
    ),
    "cache_hints": (
        "https://docs.nvidia.com/cuda/parallel-thread-execution/"
        "index.html#cache-operators"
    ),
    "carveout_api": (
        "https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html"
    ),
}
INSTRUCTION_BYTES = 16
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;")
LABEL = re.compile(r"^\s*(\.L[\w.$]+):")
SECTION = re.compile(r"^\s*//-+\s*\.text\.([^\s]+)")
TARGET = re.compile(r"(\.L[\w.$]+)")
CONTROL = {"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT"}
MEMORY_OPS = {"LDG", "STG", "LDL", "STL", "LDS", "STS", "ATOM"}
CSV_FIELDS = (
    "case",
    "status",
    "probe",
    "space",
    "chains",
    "requested_body_kib",
    "hot_bytes",
    "hot_instructions",
    "hot_operations",
    "registers",
    "local_bytes_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "block_size",
    "active_lanes",
    "requested_resident_warps",
    "resident_blocks_per_sm",
    "resident_warps_per_sm",
    "grid_blocks",
    "waves",
    "iterations",
    "minimum_ms",
    "maximum_ms",
    "sample_count",
    "operations_per_second",
    "error",
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=_json_default) + "\n")
        handle.flush()


def _command(command):
    result = subprocess.run(
        [str(part) for part in command],
        capture_output=True,
        text=True,
        check=False,
    )
    return dict(
        command=[str(part) for part in command],
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _tool(name, explicit=None):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    found = shutil.which(name)
    if found:
        return found
    suffix = ".exe" if os.name == "nt" else ""
    roots = [
        Path(value)
        for key, value in os.environ.items()
        if key.startswith("CUDA_PATH")
    ]
    toolkit = (
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "NVIDIA GPU Computing Toolkit"
        / "CUDA"
    )
    if toolkit.is_dir():
        roots.extend(sorted(toolkit.iterdir(), reverse=True))
    for root in roots:
        candidate = root / "bin" / (name + suffix)
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Set --{name} or put {name} on PATH")


def _clocks():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi is not on PATH"}
    fields = (
        "index,uuid,name,pci.bus_id,driver_version,clocks.current.sm,"
        "clocks.current.memory,pstate,power.draw,temperature.gpu,"
        "utilization.gpu,utilization.memory"
    )
    result = _command(
        [
            executable,
            "--query-gpu=" + fields,
            "--format=csv,noheader,nounits",
        ]
    )
    result["timestamp_ns"] = time.time_ns()
    result["fields"] = fields.split(",")
    result["units"] = {
        "clocks.current.sm": "MHz",
        "clocks.current.memory": "MHz",
        "power.draw": "W",
        "temperature.gpu": "degrees C",
        "utilization.gpu": "percent",
        "utilization.memory": "percent",
    }
    if result["returncode"] == 0:
        result["devices"] = [
            dict(zip(result["fields"], (item.strip() for item in row)))
            for row in csv.reader(result["stdout"].splitlines())
        ]
    return result


def _attribute(device, name):
    try:
        return int(getattr(device, name))
    except (AttributeError, RuntimeError):
        return None


def _manifest(args, nvdisasm):
    device = cuda.get_current_device()
    names = (
        "MULTIPROCESSOR_COUNT",
        "WARP_SIZE",
        "MAX_THREADS_PER_BLOCK",
        "MAX_THREADS_PER_MULTIPROCESSOR",
        "MAX_BLOCKS_PER_MULTIPROCESSOR",
        "MAX_REGISTERS_PER_MULTIPROCESSOR",
        "MAX_REGISTERS_PER_BLOCK",
        "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
        "MAX_SHARED_MEMORY_PER_BLOCK",
        "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
        "RESERVED_SHARED_MEMORY_PER_BLOCK",
        "L2_CACHE_SIZE",
        "CLOCK_RATE",
        "MEMORY_CLOCK_RATE",
        "GLOBAL_MEMORY_BUS_WIDTH",
    )
    versions = {}
    for package in (
        "cubie",
        "cubie-numba-cuda-mlir",
        "numba-cuda-mlir",
        "numba-cuda",
        "numpy",
        "cuda-bindings",
        "cupy-cuda13x",
        "cupy-cuda12x",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(
        schema=1,
        timestamp_ns=time.time_ns(),
        argv=sys.argv,
        arguments=vars(args),
        python=sys.version,
        platform=platform.platform(),
        backend=CUDA_BACKEND,
        versions=versions,
        git_head=_command(["git", "-C", REPO, "rev-parse", "HEAD"]),
        git_status=_command(["git", "-C", REPO, "status", "--short"]),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device_name=device.name,
        compute_capability=device.compute_capability,
        device_attributes={name: _attribute(device, name) for name in names},
        attribute_origin="CUDA driver device attributes; null = unavailable",
        nvdisasm=_command([nvdisasm, "--version"]),
        clocks=_clocks(),
        sources=SOURCES,
        constants={
            "instruction_bytes": dict(
                value=INSTRUCTION_BYTES,
                unit="bytes/SASS instruction",
                origin=SOURCES["instruction_bytes"],
                scope="SM89 encoding",
            ),
            "pointer_payload_bytes": dict(
                value=np.dtype(np.int32).itemsize,
                origin="numpy int32 dtype",
            ),
        },
        experimental_inputs={
            "fp32_multiplier": float(np.float32(0.99999994)),
            "fp32_increment": float(np.float32(1e-7)),
            "meaning": "finite contracting recurrence, not model constants",
        },
        interpretation={
            "icache": "Measured hot-stream transition; cache domain unknown",
            "latency": (
                "Clock intervals include loop/address/scheduling costs and "
                "possible one-operation endpoint effects; no latency fitted"
            ),
            "occupancy": (
                "Driver theoretical residency; achieved residency requires "
                "Nsight Compute. Grid has >=2 full theoretical waves."
            ),
            "carveout": (
                "Preferred carveout is a hint; actual allocation is unknown "
                "until LaunchStats measures it. Dynamic reservation is exact."
            ),
        },
    )


def _parse_sass(text, entry):
    sections = {}
    name = None
    for line in text.splitlines():
        match = SECTION.match(line)
        if match:
            name = match.group(1)
            sections[name] = {"instructions": [], "labels": {}}
            continue
        if name is None:
            continue
        section = sections[name]
        match = LABEL.match(line)
        if match:
            section["labels"][match.group(1)] = len(section["instructions"])
        match = INSTRUCTION.match(line)
        if match:
            address, body = match.groups()
            predicate = ""
            if body.startswith("@"):
                predicate, body = body.split(None, 1)
            full = body.split()[0]
            section["instructions"].append(
                dict(
                    address=int(address, 16),
                    predicate=predicate,
                    opcode=full.split(".")[0],
                    full_opcode=full,
                    text=body,
                )
            )
    if entry not in sections:
        raise ValueError(
            f"Entry {entry!r} not in SASS sections {list(sections)}"
        )
    section = sections[entry]
    instructions, labels = section["instructions"], section["labels"]
    loops = []
    for end, instruction in enumerate(instructions):
        if instruction["opcode"] != "BRA":
            continue
        match = TARGET.search(instruction["text"])
        if not match or match.group(1) not in labels:
            raise ValueError("Cannot resolve direct SASS branch target")
        start = labels[match.group(1)]
        if start < end:
            body = instructions[start : end + 1]
            loops.append(
                dict(
                    start_index=start,
                    end_index=end,
                    start_address=body[0]["address"],
                    end_address_exclusive=body[-1]["address"]
                    + INSTRUCTION_BYTES,
                    bytes=body[-1]["address"]
                    + INSTRUCTION_BYTES
                    - body[0]["address"],
                    instructions=len(body),
                    opcounts=dict(
                        collections.Counter(i["opcode"] for i in body)
                    ),
                    predicate=instruction["predicate"],
                )
            )
    return instructions, loops, labels


def _check_hot_region(text, entry, expected_opcode, expected_count):
    instructions, loops, labels = _parse_sass(text, entry)
    candidates = [
        loop for loop in loops if loop["opcounts"].get(expected_opcode, 0)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {expected_opcode} repeated region; "
            f"found {len(candidates)}. Retained/nested loops invalidate it."
        )
    hot = candidates[0]
    body = instructions[hot["start_index"] : hot["end_index"] + 1]
    controls = [item for item in body if item["opcode"] in CONTROL]
    if not controls or controls[-1] is not body[-1]:
        raise ValueError("Repeated region does not end at its backedge")
    final_operation = max(
        item["address"] for item in body if item["opcode"] == expected_opcode
    )
    for branch in controls[:-1]:
        target = TARGET.search(branch["text"])
        forward_exit = (
            target and labels.get(target.group(1), -1) > hot["end_index"]
        )
        direct_exit = branch["opcode"] == "BRA" or (
            branch["opcode"] == "CALL" and ".NOINC" in branch["full_opcode"]
        )
        if not (
            direct_exit
            and forward_exit
            and branch["predicate"]
            and branch["address"] > final_operation
        ):
            raise ValueError("Repeated body has interior control flow")
    if not controls[-1]["predicate"] and len(controls) != 2:
        raise ValueError("Unconditional backedge needs a verified tail exit")
    hot["tail_controls"] = controls
    if hot["opcounts"].get(expected_opcode) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {expected_opcode} in repeated body; "
            f"compiled {hot['opcounts'].get(expected_opcode, 0)}. "
            "Source-operation normalization is prohibited for this case."
        )
    for item in body:
        if item["opcode"] == expected_opcode and item["predicate"]:
            raise ValueError("Measured operations have SASS predicates")
        if item["opcode"] in MEMORY_OPS and expected_opcode == "FFMA":
            raise ValueError("Arithmetic repeated body has memory traffic")
        if item["opcode"] == expected_opcode and expected_opcode != "FFMA":
            if ".64" in item["full_opcode"] or ".128" in item["full_opcode"]:
                raise ValueError("Pointer loads are not scalar 32-bit loads")
    hot["verified_straight_line"] = True
    hot["verified_operations"] = expected_count
    hot["opcode"] = expected_opcode
    return dict(
        entry=entry,
        hot=hot,
        loops=loops,
        total_instructions=len(instructions),
        total_opcounts=dict(
            collections.Counter(
                instruction["opcode"] for instruction in instructions
            )
        ),
    )


def _header(clocks=False):
    lines = [
        "import numpy as np",
        "from cubie.cuda_simsafe import cuda, float32, int32, unroll_if",
        "ROLLED = (True, 1)",
        "FULL = True",
        "",
    ]
    if clocks:
        lines.extend(
            [
                "clock64 = cuda.intrin.define('''",
                "func.func private @probe_clock64() -> i64",
                "    attributes {always_inline} {",
                '    %value = "llvm.inline_asm"() {',
                '        asm_string = "mov.u64 $0, %clock64;",',
                '        constraints = "=l", has_side_effects',
                "    } : () -> i64",
                "    return %value : i64",
                "}",
                "''')",
                "",
            ]
        )
    return lines


def _arithmetic_source(operations, chains, active_lanes, clocks):
    lines = _header(clocks)
    arguments = "output, cycles, iterations, multiplier, increment"
    lines.extend(
        [
            f"def probe({arguments}):",
            "    thread = cuda.grid(1)",
            f"    if cuda.threadIdx.x >= {active_lanes}:",
            "        return",
        ]
    )
    for chain in range(chains):
        lines.append(
            f"    value{chain} = float32({chain + 1}) "
            "+ float32(thread & 7) * float32(0.001)"
        )
    if clocks:
        lines.extend(
            [
                "    start = clock64()",
            ]
        )
    lines.append("    for _ in unroll_if(range(iterations), ROLLED):")
    lines.append(
        f"        for _j in unroll_if(range({operations // chains}), FULL):"
    )
    for chain in range(chains):
        lines.append(
            f"            value{chain} = value{chain} * multiplier + increment"
        )
    if clocks:
        lines.extend(
            [
                "    end = clock64()",
                "    cycles[thread] = end - start",
            ]
        )
    lines.append(
        "    output[thread] = "
        + " + ".join(f"value{chain}" for chain in range(chains))
    )
    return "\n".join(lines) + "\n"


def _memory_source(space, elements, operations, chains, active_lanes):
    lines = _header(True)
    lines.extend(
        [
            "def probe(output, cycles, links, iterations):",
            "    thread = cuda.grid(1)",
            "    lane = cuda.threadIdx.x",
        ]
    )
    if space == "local":
        lines.extend(
            [
                f"    ring = cuda.local.array({elements}, dtype=int32)",
                f"    for index in unroll_if(range({elements}), ROLLED):",
                "        ring[index] = links[index]",
            ]
        )
    elif space == "shared":
        lines.extend(
            [
                f"    ring = cuda.shared.array({elements}, dtype=int32)",
                f"    for index in range(lane, {elements}, cuda.blockDim.x):",
                "        ring[index] = links[index]",
            ]
        )
    else:
        lines.append("    ring = links")
    lines.extend(
        [
            "    cuda.syncthreads()",
            f"    if lane >= {active_lanes}:",
            "        return",
        ]
    )
    for chain in range(chains):
        lines.append(
            f"    cursor{chain} = int32((thread + {chain}) & {elements - 1})"
        )
    lines.extend(
        [
            "    start = clock64()",
            "    for _ in unroll_if(range(iterations), ROLLED):",
        ]
    )
    for operation in range(operations):
        chain = operation % chains
        lines.append(f"        cursor{chain} = ring[cursor{chain}]")
    lines.extend(
        [
            "    end = clock64()",
            "    cycles[thread] = end - start",
            "    output[thread] = "
            + " ^ ".join(f"cursor{chain}" for chain in range(chains)),
        ]
    )
    return "\n".join(lines) + "\n"


def _compile(source, case_dir, signature_args, nvdisasm, opcode, count):
    path = case_dir / "kernel.py"
    path.write_text(source, encoding="utf-8")
    module_name = (
        "hardware_probe_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    kernel = cuda.jit(**get_jit_kwargs())(module.probe)
    start = time.perf_counter()
    compile_kernel_specialization(kernel, signature_args)
    compile_seconds = time.perf_counter() - start
    (compiled,) = kernel.overloads.values()
    cubin, entry = _compiled_cubin(compiled)
    (case_dir / "kernel.cubin").write_bytes(cubin)
    assembly = kernel.inspect_asm()
    if isinstance(assembly, dict):
        assembly = "\n".join(assembly.values())
    (case_dir / "kernel.ptx").write_text(assembly, encoding="utf-8")
    command = _command([nvdisasm, "-c", case_dir / "kernel.cubin"])
    _write_json(case_dir / "disassembly_command.json", command)
    if command["returncode"]:
        raise RuntimeError(command["stderr"])
    (case_dir / "kernel.sass").write_text(command["stdout"], encoding="utf-8")
    compiled._ensure_kernel_attrs()
    resources = dict(
        registers=int(next(iter(kernel.get_regs_per_thread().values()))),
        local_bytes_per_thread=int(
            next(iter(kernel.get_local_mem_per_thread().values()))
        ),
        static_shared_bytes=int(
            next(iter(kernel.get_shared_mem_per_block().values()))
        ),
        compile_seconds=compile_seconds,
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        jit_kwargs=get_jit_kwargs(),
    )
    _write_json(case_dir / "resources.json", resources)
    analysis = _check_hot_region(command["stdout"], entry, opcode, count)
    if "start = clock64()" in source:
        instructions, _, _ = _parse_sass(command["stdout"], entry)
        reads = [item for item in instructions if "SR_CLOCKLO" in item["text"]]
        hot = analysis["hot"]
        if not (
            len(reads) == 2
            and reads[0]["address"] < hot["start_address"]
            and reads[1]["address"] >= hot["end_address_exclusive"]
        ):
            raise ValueError("Two clock64 reads do not bracket the hot region")
        analysis["clock_reads"] = reads
        analysis["clock_bounds_verified"] = True
    _write_json(case_dir / "sass_analysis.json", analysis)
    return kernel, compiled._codelibrary.get_cufunc(), resources, analysis


def _geometry(function, resources, args, requested_warps):
    device = cuda.get_current_device()
    context = cuda.current_context()
    warp_size = int(device.WARP_SIZE)
    if args.block_size % warp_size:
        raise ValueError("Block size must be a whole number of device warps")
    if args.block_size > int(device.MAX_THREADS_PER_BLOCK):
        raise ValueError("Block size exceeds the queried device limit")
    driver = cuda.cudadrv.driver.driver
    # Driver API enum values, not hardware/model constants.
    driver.cuFuncSetAttribute(function.handle, 9, args.carveout)
    maximum = int(device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    maximum -= resources["static_shared_bytes"]
    if maximum < 0:
        raise ValueError("Static shared memory exceeds the device limit")
    driver.cuFuncSetAttribute(function.handle, 8, maximum)

    def residency(dynamic):
        return int(
            context.get_active_blocks_per_multiprocessor(
                function, args.block_size, dynamic
            )
        )

    dynamic = 0
    native_blocks = residency(dynamic)
    if native_blocks <= 0:
        raise ValueError("Compiled kernel has no legal resident block")
    warps_per_block = args.block_size // warp_size
    if requested_warps:
        if requested_warps % warps_per_block:
            raise ValueError(
                "Requested warps must be divisible by warps/block"
            )
        target = requested_warps // warps_per_block
        if target > native_blocks:
            raise ValueError(
                f"Requested {target} blocks/SM exceeds driver maximum "
                f"{native_blocks} for compiled resources"
            )
        if target < native_blocks:
            lower, upper = 0, maximum
            while lower < upper:
                midpoint = (lower + upper) // 2
                if residency(midpoint) > target:
                    lower = midpoint + 1
                else:
                    upper = midpoint
            dynamic = lower
            if residency(dynamic) != target:
                raise ValueError("Shared reservation cannot reach this target")
    resident_blocks = residency(dynamic)
    sms = int(device.MULTIPROCESSOR_COUNT)
    grid_blocks = args.waves * sms * resident_blocks
    return dict(
        block_size=args.block_size,
        active_lanes=args.active_lanes,
        warp_size=warp_size,
        sms=sms,
        grid_blocks=grid_blocks,
        resident_blocks_per_sm=resident_blocks,
        resident_warps_per_sm=resident_blocks * warps_per_block,
        active_warps_per_sm_upper_bound=resident_blocks
        * math.ceil(args.active_lanes / warp_size),
        requested_resident_warps=requested_warps,
        native_resident_blocks_per_sm=native_blocks,
        dynamic_shared_bytes=dynamic,
        carveout_preference_percent=args.carveout,
        actual_carveout_bytes=None,
        waves=grid_blocks / (sms * resident_blocks),
        occupancy_origin="CUDA get_active_blocks_per_multiprocessor",
        reservation_method="minimum bytes attaining queried block residency",
    )


def _timed_launch(kernel, geometry, arguments):
    start, end = cuda.event(), cuda.event()
    start.record()
    kernel[
        geometry["grid_blocks"],
        geometry["block_size"],
        0,
        geometry["dynamic_shared_bytes"],
    ](*arguments)
    end.record()
    end.synchronize()
    return float(cuda.event_elapsed_time(start, end))


def _sample_cycles(cycles, geometry, path):
    host = (
        cupy.asnumpy(cycles)
        .reshape(geometry["grid_blocks"], geometry["block_size"])[
            :, : geometry["active_lanes"]
        ]
        .copy()
    )
    np.save(path, host)
    if not np.all(host > 0):
        raise ValueError("Clock intervals contain unwritten or zero entries")
    return dict(
        path=str(path),
        count=int(host.size),
        minimum=int(host.min()),
        maximum=int(host.max()),
        mean=float(host.mean()),
        unit="SM clock cycles per active thread over repeated region",
    )


def _measure(kernel, geometry, args, case_dir, argument_factory, cycles):
    iterations = args.iterations
    samples_path = case_dir / "samples.jsonl"

    def launch(phase, index):
        before = _clocks()
        elapsed = _timed_launch(kernel, geometry, argument_factory(iterations))
        after = _clocks()
        row = dict(
            phase=phase,
            index=index,
            iterations=iterations,
            milliseconds=elapsed,
            clocks_before=before,
            clocks_after=after,
            timestamp_ns=time.time_ns(),
        )
        if cycles is not None:
            row["cycles"] = _sample_cycles(
                cycles, geometry, case_dir / f"{phase}_{index:03d}_cycles.npy"
            )
        _append_json(samples_path, row)
        if args.idle_ms:
            time.sleep(args.idle_ms / 1000)
        return row

    if args.profile_once:
        rows = [launch("profile", 0)]
    else:
        for index in range(args.warmups):
            launch("warmup", index)
        for index in range(args.max_calibrations):
            row = launch("calibration", index)
            if row["milliseconds"] >= args.min_ms:
                break
            iterations *= 2
            if iterations > np.iinfo(np.int32).max:
                raise ValueError("Calibration exceeds the int32 repeat count")
        else:
            raise ValueError(
                "Minimum duration not attained during calibration"
            )
        rows = [launch("sample", index) for index in range(args.repeats)]
    timings = [row["milliseconds"] for row in rows]
    return dict(
        iterations=iterations,
        timings_ms=timings,
        minimum_ms=min(timings),
        maximum_ms=max(timings),
        sample_count=len(timings),
        status="ok"
        if min(timings) >= args.min_ms
        else "below_minimum_duration",
    )


def _ring(elements, seed):
    order = np.random.default_rng(seed).permutation(elements).astype(np.int32)
    links = np.empty(elements, dtype=np.int32)
    links[order] = np.roll(order, -1)
    return links


def _check_ring_output(result, geometry, elements, chains, advances, seed):
    order = np.random.default_rng(seed).permutation(elements).astype(np.int32)
    position = np.empty(elements, dtype=np.int64)
    position[order] = np.arange(elements)
    threads = (
        np.arange(geometry["grid_blocks"], dtype=np.int64)[:, None]
        * geometry["block_size"]
        + np.arange(geometry["active_lanes"], dtype=np.int64)[None, :]
    )
    expected = np.zeros(threads.shape, dtype=np.int32)
    for chain in range(chains):
        initial = (threads + chain) & (elements - 1)
        final = order[(position[initial] + advances) % elements]
        expected ^= final
    if not np.array_equal(result, expected):
        raise ValueError(
            "Pointer-ring result differs from exact cycle advance"
        )


def _case(
    args, output, nvdisasm, name, operations, chains, resident, body_kib
):
    directory = output / name
    directory.mkdir()
    row = dict(
        case=name,
        probe=args.probe,
        chains=chains,
        requested_body_kib=body_kib,
        space=getattr(args, "space", None),
    )
    try:
        memory = args.probe == "memory"
        clocks = args.probe != "icache"
        dtype = np.int32 if memory else np.float32
        host_output = np.empty(1, dtype=dtype)
        host_cycles = np.empty(1, dtype=np.uint64)
        if memory:
            links = _ring(args.elements, args.seed)
            np.save(directory / "pointer_ring.npy", links)
            source = _memory_source(
                args.space,
                args.elements,
                operations,
                chains,
                args.active_lanes,
            )
            signature_args = (
                host_output,
                host_cycles,
                links,
                np.int32(args.iterations),
            )
            opcode = {"local": "LDL", "shared": "LDS", "global": "LDG"}[
                args.space
            ]
        else:
            source = _arithmetic_source(
                operations, chains, args.active_lanes, clocks
            )
            signature_args = (
                host_output,
                host_cycles,
                np.int32(args.iterations),
                np.float32(0.99999994),
                np.float32(1e-7),
            )
            opcode = "FFMA"
        kernel, function, resources, analysis = _compile(
            source, directory, signature_args, nvdisasm, opcode, operations
        )
        row.update(resources)
        geometry = _geometry(function, resources, args, resident)
        row.update(geometry)
        row.update(
            hot_bytes=analysis["hot"]["bytes"],
            hot_instructions=analysis["hot"]["instructions"],
            hot_operations=analysis["hot"]["verified_operations"],
            sass_analysis=analysis,
        )
        if memory:
            row["window_elements"] = args.elements
            row["window_bytes"] = args.elements * np.dtype(np.int32).itemsize
            row["window_sharing"] = {
                "global": "one common ring shared by all blocks/SMs",
                "shared": "one ring per block",
                "local": "one ring per thread",
            }[args.space]
        _write_json(directory / "case.json", row)
        if args.compile_only:
            row["status"] = "compiled_unlaunched"
            return row
        count = geometry["grid_blocks"] * geometry["block_size"]
        device_output = cupy.full(count, -1, dtype=dtype)
        device_cycles = cupy.zeros(count, dtype=np.uint64)
        if memory:
            device_links = cupy.asarray(links)

            def arguments(iterations):
                return (
                    device_output,
                    device_cycles,
                    device_links,
                    np.int32(iterations),
                )
        else:

            def arguments(iterations):
                return (
                    device_output,
                    device_cycles,
                    np.int32(iterations),
                    np.float32(0.99999994),
                    np.float32(1e-7),
                )

        cuda.synchronize()
        measurement = _measure(
            kernel,
            geometry,
            args,
            directory,
            arguments,
            device_cycles if clocks else None,
        )
        row.update(measurement)
        result = (
            cupy.asnumpy(device_output)
            .reshape(geometry["grid_blocks"], geometry["block_size"])[
                :, : geometry["active_lanes"]
            ]
            .copy()
        )
        np.save(directory / "output.npy", result)
        if not memory and not np.all(np.isfinite(result)):
            raise ValueError("Arithmetic result contains non-finite values")
        if memory:
            _check_ring_output(
                result,
                geometry,
                args.elements,
                chains,
                measurement["iterations"] * operations // chains,
                args.seed,
            )
            row["output_check"] = (
                "exact pointer-cycle advance for every thread"
            )
        total = (
            geometry["grid_blocks"]
            * args.active_lanes
            * measurement["iterations"]
            * operations
        )
        warp_total = (
            geometry["grid_blocks"]
            * math.ceil(args.active_lanes / geometry["warp_size"])
            * measurement["iterations"]
            * operations
        )
        row["workload_denominators"] = dict(
            scalar_operations=total,
            warp_operation_instructions=warp_total,
            participating_threads=geometry["grid_blocks"] * args.active_lanes,
            iterations_per_thread=measurement["iterations"],
            operations_per_iteration=operations,
            bytes_requested_by_scalar_loads=(
                total * np.dtype(np.int32).itemsize if memory else 0
            ),
            requested_bytes_are_not_cache_traffic=True,
        )
        row["operations_per_second"] = [
            total / (milliseconds / 1000)
            for milliseconds in measurement["timings_ms"]
        ]
        row["operation_unit"] = (
            "verified scalar pointer loads" if memory else "FP32 FMA results"
        )
        row["normalization"] = (
            "Exact verified straight-line SASS operation count multiplied "
            "by runtime iterations and participating threads"
        )
        return row
    except Exception as error:
        row.update(
            status="error", error=str(error), traceback=traceback.format_exc()
        )
        return row
    finally:
        _write_json(directory / "result.json", row)


def _integers(value):
    values = [int(item.strip()) for item in value.split(",")]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "Expected comma-separated nonnegative integers"
        )
    return values


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subcommands = parser.add_subparsers(dest="probe", required=True)
    for name in ("icache", "fp32", "memory"):
        sub = subcommands.add_parser(
            name, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        sub.add_argument(
            "--output",
            type=Path,
            required=True,
            help="fresh directory for all raw evidence",
        )
        sub.add_argument(
            "--compile-only",
            action="store_true",
            help="compile/query resources; never launch/allocate GPU arrays",
        )
        sub.add_argument("--nvdisasm", help="CUDA nvdisasm executable")
        sub.add_argument(
            "--block-size", type=int, default=128 if name == "icache" else 1024
        )
        sub.add_argument(
            "--active-lanes",
            type=int,
            help="participating threads/block; default all threads",
        )
        sub.add_argument(
            "--resident-warps",
            type=_integers,
            default=[16] if name == "icache" else [0],
            help="resident warps/SM via shared reservation; 0 = native",
        )
        sub.add_argument(
            "--carveout",
            type=int,
            default=100,
            help="preferred shared carveout percent; hint, not measured L1",
        )
        sub.add_argument(
            "--waves",
            type=int,
            default=2,
            help="full driver-occupancy waves; must be >=2",
        )
        sub.add_argument(
            "--iterations",
            type=int,
            default=128,
            help="initial repeats, doubled to reach duration independently",
        )
        sub.add_argument(
            "--min-ms",
            type=float,
            default=20,
            help="minimum accepted timing; must be >=20 ms",
        )
        sub.add_argument("--max-calibrations", type=int, default=30)
        sub.add_argument("--repeats", type=int, default=5)
        sub.add_argument("--warmups", type=int, default=1)
        sub.add_argument("--idle-ms", type=float, default=50)
        sub.add_argument(
            "--profile-once",
            action="store_true",
            help="one launch/case, no calibration/warmup; set --iterations",
        )
        if name == "icache":
            sub.add_argument(
                "--ffmas",
                type=_integers,
                help="exact FFMA counts; overrides --body-kib for fine scans",
            )
            sub.add_argument(
                "--body-kib",
                type=_integers,
                default=[112, 120, 124, 128, 132, 136, 144],
                help="requested FFMA bytes; actual hot bytes are measured",
            )
            sub.add_argument(
                "--chains",
                type=_integers,
                default=[8],
                help="independent FP32 accumulators",
            )
        else:
            sub.add_argument(
                "--operations",
                type=int,
                default=32,
                help="explicit operations per rolled runtime iteration",
            )
            sub.add_argument(
                "--chains",
                type=_integers,
                default=[1, 8],
                help="independent chains; 1 is the dependency experiment",
            )
        if name == "memory":
            sub.add_argument(
                "--space", choices=("shared", "local", "global"), required=True
            )
            sub.add_argument(
                "--elements",
                type=int,
                default=1024,
                help="power-of-two ring size in 32-bit indices",
            )
            sub.add_argument(
                "--seed",
                type=int,
                default=0,
                help="recorded RNG seed for a permutation cycle",
            )
    return parser


def main():
    """Run explicit cases and retain failed cases without timing them."""
    parser = _parser()
    args = parser.parse_args()
    if args.active_lanes is None:
        args.active_lanes = args.block_size
    if (
        args.waves < 2
        or args.min_ms < 20
        or args.iterations < 1
        or args.iterations > np.iinfo(np.int32).max
    ):
        parser.error(
            "Require >=2 waves, >=20 ms and a positive int32 repeat count"
        )
    if not 1 <= args.active_lanes <= args.block_size:
        parser.error("active-lanes must be between 1 and block-size")
    if args.probe == "icache" and args.active_lanes != args.block_size:
        parser.error("icache requires every thread to participate")
    if (
        args.repeats < 1
        or args.warmups < 0
        or args.idle_ms < 0
        or args.max_calibrations < 1
        or not 0 <= args.carveout <= 100
    ):
        parser.error(
            "Invalid repeat, warmup, idle, calibration or carveout value"
        )
    if any(chain < 1 for chain in args.chains):
        parser.error("chains must be positive")
    if args.probe == "memory":
        if args.elements < 2 or args.elements & (args.elements - 1):
            parser.error("elements must be a power of two >=2")
        if max(args.chains) > args.elements:
            parser.error("chains cannot exceed the pointer-ring size")
    if args.probe == "icache":
        if args.ffmas is not None:
            sizes = [
                (count * INSTRUCTION_BYTES / 1024, count)
                for count in args.ffmas
            ]
        else:
            sizes = [
                (size, size * 1024 // INSTRUCTION_BYTES)
                for size in args.body_kib
            ]
    else:
        sizes = [(None, args.operations)]
    if any(
        count < 1 or count % chain
        for _, count in sizes
        for chain in args.chains
    ):
        parser.error(
            "operation counts must be positive multiples of chain count"
        )
    if CUDA_SIMULATION or not IS_MLIR:
        parser.error("These diagnostics require the real MLIR CUDA backend")
    if tuple(cuda.get_current_device().compute_capability) != (8, 9):
        parser.error("This diagnostic currently validates only SM89 SASS")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    nvdisasm = _tool("nvdisasm", args.nvdisasm)
    _write_json(output / "manifest.json", _manifest(args, nvdisasm))
    csv_path = output / "results.csv"
    failures = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for body_kib, operations in sizes:
            for chains in args.chains:
                for resident in args.resident_warps:
                    space = "_" + args.space if args.probe == "memory" else ""
                    name = (
                        f"{args.probe}{space}_ops{operations}_chains{chains}"
                        f"_warps{resident}"
                    )
                    row = _case(
                        args,
                        output,
                        nvdisasm,
                        name,
                        operations,
                        chains,
                        resident,
                        body_kib,
                    )
                    _append_json(output / "results.jsonl", row)
                    writer.writerow(row)
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in CSV_FIELDS
                                if key in row
                            },
                            default=_json_default,
                        ),
                        flush=True,
                    )
                    if row["status"] not in ("ok", "compiled_unlaunched"):
                        failures += 1
    _write_json(
        output / "completion.json",
        dict(
            timestamp_ns=time.time_ns(),
            failed_cases=failures,
            clocks=_clocks(),
            compile_only=args.compile_only,
        ),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
