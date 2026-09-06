"""Record the CUPTI shared partition of a trivial kernel per geometry."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import time

import numpy as np

import cubie  # noqa: F401
from cubie.cuda_simsafe import cuda

CUDA = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
CUPTI = CUDA / "extras/CUPTI/lib64/cupti64_2026.2.1.dll"
COLLECTOR = Path(
    "C:/local_working_projects/cubie-notes/hardware_unroll_placement/"
    "verification/cupti_carveout_author_e1/collector.dll"
)
BLOCKSIZES = (32, 64, 128, 256, 512, 1024)
DYNAMIC = (4, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
           24576, 32768, 49152, 65536, 98304)


def load_collector():
    for directory in (CUPTI.parent, CUDA / "bin", COLLECTOR.parent):
        os.add_dll_directory(str(directory))
    ctypes.WinDLL(str(CUPTI))
    collector = ctypes.CDLL(str(COLLECTOR))
    collector.collector_start.argtypes = [ctypes.c_wchar_p]
    collector.collector_start.restype = ctypes.c_int
    collector.collector_stop.argtypes = []
    collector.collector_stop.restype = ctypes.c_int
    return collector


def make_kernel(max_registers, carveout=None):
    kwargs = {}
    if max_registers:
        kwargs["max_registers"] = max_registers
    if carveout is not None:
        kwargs["shared_memory_carveout"] = carveout

    @cuda.jit(**kwargs)
    def probe(out, words):
        buffer = cuda.shared.array(0, dtype=np.float32)
        tid = cuda.grid(1)
        local = cuda.threadIdx.x
        if local < words:
            buffer[local] = np.float32(tid)
        cuda.syncthreads()
        value = np.float32(0.0)
        if words > 0:
            value = buffer[(local + 1) % words]
        out[tid] = value

    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-registers", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--carveout", type=int, default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = cuda.get_current_device()
    sms = int(device.MULTIPROCESSOR_COUNT)
    kernel = make_kernel(args.max_registers, args.carveout)
    output = cuda.device_array(1024 * sms * 4, dtype=np.float32)
    kernel[sms, 32, 0, 4](output, np.int32(0))
    cuda.synchronize()
    (defn,) = kernel.overloads.values()
    context = cuda.current_context()
    cufunc = defn._codelibrary.get_cufunc()
    regs = list(kernel.get_regs_per_thread().values())[0]
    collector = load_collector()
    activity = out / "activity.jsonl"
    if collector.collector_start(str(activity)) != 0:
        raise SystemExit("collector_start failed")
    launches = []
    try:
        for _ in range(args.repeat):
            for blocksize in BLOCKSIZES:
                for dynamic in DYNAMIC:
                    blocks = int(context.get_active_blocks_per_multiprocessor(
                        cufunc, blocksize, dynamic))
                    if blocks == 0:
                        continue
                    grid = blocks * sms * 2
                    words = min(dynamic // 4, blocksize)
                    kernel[grid, blocksize, 0, dynamic](output, np.int32(words))
                    cuda.synchronize()
                    launches.append(dict(blocksize=blocksize, dynamic=dynamic,
                                         blocks_per_sm=blocks, grid=grid))
    finally:
        cuda.synchronize()
        stopped = collector.collector_stop()
    rows = [json.loads(line) for line in activity.read_text().splitlines()]
    kernels = sorted((r for r in rows if r["type"] == "kernel"),
                     key=lambda r: r["start_ns"])
    kernels = [k for k in kernels if k["grid"] != [sms, 1, 1]]
    if len(kernels) != len(launches):
        print(f"kernel records {len(kernels)} != launches {len(launches)}")
    for row, kernel_row in zip(launches, kernels):
        row["executed"] = kernel_row["shared_memory_executed_bytes"]
        row["static_shared"] = kernel_row["static_shared_bytes"]
        row["dynamic_shared"] = kernel_row["dynamic_shared_bytes"]
        row["cupti_block"] = kernel_row["block"]
        row["carveout_requested"] = kernel_row["carveout_requested"]
        row["requested_percent"] = kernel_row["requested_percent"]
    (out / "records.json").write_text(json.dumps(dict(
        regs=int(regs), max_registers=args.max_registers, sms=sms,
        carveout=args.carveout,
        collector_stop=stopped, rows=launches), indent=1))
    print(f"regs {regs} stop {stopped} launches {len(launches)}")
    for row in launches[:len(launches) // args.repeat]:
        need = row["blocks_per_sm"] * (row["dynamic"] + 1024)
        print(f"bs{row['blocksize']:<5d} dyn {row['dynamic']:6d} blk "
              f"{row['blocks_per_sm']:2d} need {need:7d} exec "
              f"{row['executed']:7d} requested "
              f"{row['carveout_requested']} {row['requested_percent']}")


if __name__ == "__main__":
    main()
