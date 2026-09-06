"""Record the CUPTI shared partition of ordinary production launches."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import placement_landscape as pl  # noqa: E402

CUDA = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
CUPTI = CUDA / "extras/CUPTI/lib64/cupti64_2026.2.1.dll"
COLLECTOR = Path(
    "C:/local_working_projects/cubie-notes/hardware_unroll_placement/"
    "verification/cupti_carveout_author_e1/collector.dll"
)
CASES = (
    ("lorenz", "vern7"),
    ("lorenz", "kvaerno3"),
    ("lorenz", "rosenbrock23_bicgstab"),
    ("chain32", "kvaerno3"),
    ("chain32", "radau_iia_5"),
    ("chain32", "rosenbrock23_bicgstab"),
    ("lorenz96_20", "kvaerno3_bicgstab"),
)
PLACEMENTS = ("local", "one", "all")
BLOCKSIZES = (64, 128, 256)


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


def placement_kwargs(solver, mode):
    buffers = pl.candidate_buffers(solver)
    if mode == "local" or not buffers:
        return {}, []
    names = [b["name"] for b in buffers]
    if mode == "one":
        largest = max(buffers, key=lambda b: b["elements"] * b["itemsize"])
        names = [largest["name"]]
    return pl.placement_for(names), names


def run_case(system_name, algo_name, mode, blocksize, records, launches):
    spec = pl.SYSTEMS[system_name]
    system = spec["build"]()
    probe = pl.make_solver(system, system_name, algo_name)
    placement, names = placement_kwargs(probe, mode)
    if mode != "local" and not names:
        return
    solver = pl.make_solver(system, system_name, algo_name,
                            placement=placement or None)
    n_runs = spec["n_runs"]
    inits, params = spec["grid"](solver, n_runs)
    solver.compile(inits, params, duration=spec["duration"])
    geometry = pl.launch_geometry(solver, blocksize, n_runs)
    if geometry is None:
        records.append(dict(system=system_name, algo=algo_name,
                            placement=mode, blocksize=blocksize,
                            status="shared_exceeds_block_limit"))
        return
    pl.pin_launch(solver, geometry["blocksize"], geometry["dynshared"])
    regs, local_bytes = (None, None)
    kernel_ms, wall_ms, snap = pl.solve_once(
        solver, inits, params, spec["duration"] / 8,
        blocksize=geometry["blocksize"], snapshot=True,
    )
    regs, local_bytes = pl.kernel_resources(solver)
    dispatcher = solver.kernel.kernel
    static_shared = list(dispatcher.get_shared_mem_per_block().values())[0]
    row = dict(
        system=system_name, algo=algo_name, placement=mode,
        shared_buffers=names, blocksize=geometry["blocksize"],
        dynshared=geometry["dynshared"], bytes_per_run=geometry["bytes_per_run"],
        static_shared=int(static_shared), regs=regs, local_bytes=local_bytes,
        blocks_per_sm=geometry["blocks_per_sm"], waves=geometry["waves"],
        kernel_ms=kernel_ms, failed=snap["status_hist"]["failed"],
        status="ok", launch_index=len(launches),
    )
    launches.append(row)
    records.append(row)
    print(json.dumps({k: row[k] for k in (
        "system", "algo", "placement", "blocksize", "dynshared",
        "regs", "blocks_per_sm", "kernel_ms")}), flush=True)
    del solver, probe, system


def join_activity(activity_path, launches):
    rows = [json.loads(line) for line in activity_path.read_text().splitlines()]
    kernels = [r for r in rows if r["type"] == "kernel"]
    kernels.sort(key=lambda r: r["start_ns"])
    summary = [r for r in rows if r["type"] == "summary"]
    return kernels, summary[0] if summary else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--placements", default=",".join(PLACEMENTS))
    parser.add_argument("--blocksizes", default=",".join(map(str, BLOCKSIZES)))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = CASES
    if args.cases:
        cases = tuple(tuple(c.split("/")) for c in args.cases.split(","))
    placements = args.placements.split(",")
    blocksizes = [int(b) for b in args.blocksizes.split(",")]
    collector = load_collector()
    activity = out / "activity.jsonl"
    started = collector.collector_start(str(activity))
    if started != 0:
        raise SystemExit(f"collector_start failed: {started}")
    records = []
    launches = []
    t0 = time.perf_counter()
    try:
        for system_name, algo_name in cases:
            for mode in placements:
                for blocksize in blocksizes:
                    try:
                        run_case(system_name, algo_name, mode, blocksize,
                                 records, launches)
                    except Exception as error:  # retained as a row
                        records.append(dict(
                            system=system_name, algo=algo_name,
                            placement=mode, blocksize=blocksize,
                            status="error", error=repr(error)))
                        print("ERROR", system_name, algo_name, mode,
                              blocksize, repr(error), flush=True)
    finally:
        from cubie.cuda_simsafe import cuda
        cuda.synchronize()
        stopped = collector.collector_stop()
    kernels, summary = join_activity(activity, launches)
    if len(kernels) != len(launches):
        print(f"kernel records {len(kernels)} != launches {len(launches)}",
              flush=True)
    for row, kernel in zip(launches, kernels):
        row["cupti"] = dict(
            name=kernel["name"], grid=kernel["grid"], block=kernel["block"],
            registers=kernel["registers_per_thread"],
            static_shared=kernel["static_shared_bytes"],
            dynamic_shared=kernel["dynamic_shared_bytes"],
            local_bytes_per_thread=kernel["local_bytes_per_thread"],
            carveout_requested=kernel["carveout_requested"],
            requested_percent=kernel["requested_percent"],
            shared_executed=kernel["shared_memory_executed_bytes"],
            cache_config_executed=kernel["cache_config_executed"],
        )
    (out / "records.json").write_text(json.dumps(dict(
        collector_stop=stopped, summary=summary, elapsed_s=time.perf_counter() - t0,
        kernel_records=len(kernels), rows=records), indent=1, default=str))
    print("collector_stop", stopped, "kernels", len(kernels), flush=True)
    for row in launches:
        c = row.get("cupti", {})
        print(f"{row['system']:12s} {row['algo']:24s} {row['placement']:5s} "
              f"bs{row['blocksize']:<4d} dyn {row['dynshared']:6d} "
              f"regs {row['regs']:3d} blk/SM {row['blocks_per_sm']:2d} "
              f"executed {c.get('shared_executed')}", flush=True)


if __name__ == "__main__":
    main()
