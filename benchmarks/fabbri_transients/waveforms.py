"""Dense voltage after switching chosen sources to a coarse target grid."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import common


def nearest_nodes(axis, values):
    """Return the indices of ``axis`` nearest to ``values`` in log space."""
    distance = np.abs(np.log(axis)[None, :] - np.log(values)[:, None])
    return np.unique(distance.argmin(axis=1))


def node_indices(axis, low, high, count, include):
    """Return log-spaced node indices with ``include`` values swapped in."""
    wanted = np.geomspace(low, high, count)
    for value in include:
        wanted[np.abs(np.log(wanted / value)).argmin()] = value
    return nearest_nodes(axis, wanted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="steady_state_log96.npz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--save-every", type=float, default=2.0**-10)
    parser.add_argument("--sources-per-solve", type=int, default=25)
    args = parser.parse_args()

    system = common.build_system()
    data = np.load(common.RESULTS / args.input)
    side = int(np.sqrt(data["ach"].size))
    ach_axis = data["ach"][:side]
    iso_axis = data["iso"][::side]

    src_ach = node_indices(ach_axis, 10.0, 300.0, 10, [63.1])
    src_iso = node_indices(iso_axis, 0.1, 10.0, 10, [0.13])
    tgt_ach = node_indices(ach_axis, 10.0, 1000.0, 15, [300.0])
    tgt_iso = node_indices(iso_axis, 0.1, 30.0, 15, [0.13])
    sources = (src_iso[:, None] * side + src_ach[None, :]).ravel()
    targets = (tgt_iso[:, None] * side + tgt_ach[None, :]).ravel()

    solver = common.make_solver(
        system, output_types=["state", "time"],
        save_variables=[common.VOLTAGE_LABEL], save_every=args.save_every,
    )
    params = common.parameter_array(
        system, data["ach"][targets], data["iso"][targets]
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = None
    voltage = None
    failed = np.zeros((sources.size, targets.size), dtype=np.bool_)
    start = perf_counter()
    kernel_total = 0.0
    for first in range(0, sources.size, args.sources_per_solve):
        chunk = sources[first:first + args.sources_per_solve]
        inits = np.ascontiguousarray(
            np.repeat(data["final_state"][:, chunk], targets.size, axis=1),
            dtype=np.float32,
        )
        result, kernel_ms, _ = common.timed_solve(
            solver, inits, np.ascontiguousarray(np.tile(params, chunk.size)),
            args.duration,
        )
        kernel_total += kernel_ms / 1e3
        traces = np.asarray(result.time_domain_array[:, 0, :])
        if voltage is None:
            time = np.asarray(result.time[:, 0], dtype=np.float64)
            n_samples = time.size
            np.save(args.out_dir / "time.npy", time)
            voltage = np.lib.format.open_memmap(
                args.out_dir / "voltage.npy", mode="w+", dtype=np.float32,
                shape=(sources.size, targets.size, n_samples),
            )
        rows = slice(first, first + chunk.size)
        voltage[rows] = traces.T.reshape(chunk.size, targets.size, n_samples)
        failed[rows] = common.failed_runs(result).reshape(
            chunk.size, targets.size
        )
        del result, traces
        voltage.flush()
        print(
            f"sources {first + chunk.size}/{sources.size}: kernel "
            f"{kernel_total:.1f} s, wall {perf_counter() - start:.1f} s",
            flush=True,
        )
    solver.close()
    np.save(args.out_dir / "failed.npy", failed)
    np.savez(
        args.out_dir / "grid.npz",
        sources=sources, targets=targets,
        source_ach=data["ach"][sources], source_iso=data["iso"][sources],
        target_ach=data["ach"][targets], target_iso=data["iso"][targets],
        source_cl=data["steady_cl"][sources],
        target_cl=data["steady_cl"][targets],
        source_prev_peak=np.nanmax(data["peak_times"][sources], axis=1)
        - float(data["char"]),
    )
    (args.out_dir / "settings.json").write_text(json.dumps({
        "input": args.input, "duration_s": args.duration,
        "save_every_s": args.save_every, "state_key": "final_state",
        "shape": [int(sources.size), int(targets.size), int(n_samples)],
    }, indent=2))


if __name__ == "__main__":
    main()
