"""Measure the GPU instruction-cache capacity and print it with the
compute capability to report for ``INSTRUCTION_CACHE_BYTES``."""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import cubie  # noqa: F401
from cubie.backend.utils import INSTRUCTION_CACHE_BYTES, device_hardware
from cubie.cuda_simsafe import compile_kwargs, cuda

ACCUMULATORS = 8
KNEE_RATIO = 1.25


def find_nvdisasm() -> str:
    """Return the nvdisasm executable from PATH or the CUDA toolkit."""
    found = os.environ.get("NVDISASM") or shutil.which("nvdisasm")
    if found:
        return found
    for root in (os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME")):
        if root:
            candidate = Path(root) / "bin" / "nvdisasm"
            for suffix in ("", ".exe"):
                if candidate.with_name(candidate.name + suffix).exists():
                    return str(candidate.with_name(candidate.name + suffix))
    raise FileNotFoundError(
        "nvdisasm not found; set NVDISASM or add the CUDA toolkit bin "
        "directory to PATH."
    )


def build_kernel(trips: int, kernel_dir: Path):
    """Compile a kernel whose body is ``trips`` unrolled FMA groups."""
    lines = [
        "import numpy as np",
        "from cubie.cuda_simsafe import cuda, from_dtype, unroll_if",
        "float32 = from_dtype(np.dtype('float32'))",
        f"m_trips = {trips}",
        "m_unroll = (True, None)",
        "",
        "",
        "def body(out, b, c, iters):",
        "    tid = cuda.grid(1)",
        "    f = float32(tid) * float32(1e-7)",
    ]
    for k in range(ACCUMULATORS):
        lines.append(f"    a{k} = f + float32({k})")
    lines.append("    for _ in range(iters):")
    lines.append("        for _j in unroll_if(range(m_trips), m_unroll):")
    for k in range(ACCUMULATORS):
        lines.append(f"            a{k} = a{k} * b + c")
    lines.append(
        "    out[tid] = " + " + ".join(f"a{k}" for k in range(ACCUMULATORS))
    )
    path = kernel_dir / f"body_{trips}.py"
    path.write_text("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return cuda.jit(**compile_kwargs)(module.body)


def compiled_cubin(kernel) -> bytes:
    (definition,) = kernel.overloads.values()
    library = definition._codelibrary
    if hasattr(library, "get_cubin"):
        return bytes(library.get_cubin().code)
    return bytes(library._cubin)


def sass_instruction_count(cubin: bytes, nvdisasm: str) -> int:
    """Return the SASS instruction count of a cubin."""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as tmp:
        tmp.write(cubin)
        path = tmp.name
    try:
        text = subprocess.run(
            [nvdisasm, "-c", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    finally:
        os.unlink(path)
    total = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("/*") and "*/" in stripped and ";" in stripped:
            total += 1
    return total


def measure(trips_list, warps_per_sm, reps, nvdisasm, kernel_dir):
    """Return one row per body size: SASS bytes and ns per instruction."""
    device = cuda.get_current_device()
    blocks = warps_per_sm * device.MULTIPROCESSOR_COUNT
    out = cuda.device_array(blocks * 32, dtype=np.float32)
    rows = []
    previous_sass = None
    for trips in trips_list:
        instructions = ACCUMULATORS * trips
        kernel = build_kernel(trips, kernel_dir)
        iters = max(4, int(4e6 // instructions))
        kernel[blocks, 32](
            out, np.float32(1.0000001), np.float32(1e-9), np.int32(2)
        )
        cuda.synchronize()
        sass = sass_instruction_count(compiled_cubin(kernel), nvdisasm)
        if previous_sass is not None and sass <= previous_sass:
            print(
                f"trips {trips}: SASS size did not grow ({sass}); the "
                "compiler stopped unrolling, stopping here.",
                flush=True,
            )
            break
        previous_sass = sass
        times = []
        for _ in range(reps):
            start = cuda.event()
            end = cuda.event()
            start.record()
            kernel[blocks, 32](
                out, np.float32(1.0000001), np.float32(1e-9), np.int32(iters)
            )
            end.record()
            end.synchronize()
            times.append(cuda.event_elapsed_time(start, end))
        ns_per_instruction = min(times) * 1e6 / (instructions * iters)
        row = dict(
            trips=trips,
            sass=sass,
            bytes=sass * 16,
            ns_per_instruction=ns_per_instruction,
        )
        rows.append(row)
        print(
            f"trips {trips:5d} sass {sass:6d} ({sass * 16 / 1024:7.1f} KiB) "
            f"ns/instr {ns_per_instruction:7.4f}",
            flush=True,
        )
    return rows


def knee_bytes(rows) -> int:
    """Return the largest body size still on the small-kernel plateau."""
    plateau = min(row["ns_per_instruction"] for row in rows[:3])
    last_fitting = rows[0]["bytes"]
    for row in rows:
        if row["ns_per_instruction"] > KNEE_RATIO * plateau:
            break
        last_fitting = row["bytes"]
    return int(last_fitting)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trips",
        default="32,64,128,192,256,384,512,768,1024,1536,2048,3072,4096",
        help="comma-separated unrolled group counts to time",
    )
    parser.add_argument("--warps-per-sm", type=int, default=8)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", default="icache_probe.jsonl")
    args = parser.parse_args()
    nvdisasm = find_nvdisasm()
    trips_list = [int(value) for value in args.trips.split(",")]
    with tempfile.TemporaryDirectory() as kernel_dir:
        rows = measure(
            trips_list, args.warps_per_sm, args.reps, nvdisasm,
            Path(kernel_dir),
        )
    with open(args.out, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    capacity = knee_bytes(rows)
    capability = device_hardware().compute_capability
    print(
        f"instruction cache capacity: {capacity} bytes "
        f"({capacity / 1024:.0f} KiB) for compute capability {capability}"
    )
    if capability not in INSTRUCTION_CACHE_BYTES:
        print(
            "This compute capability is unmeasured in cubie; open an "
            "issue with the value above."
        )


if __name__ == "__main__":
    main()
