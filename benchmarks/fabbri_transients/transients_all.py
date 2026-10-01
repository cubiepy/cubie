"""Record the first three peak times after each source-to-target switch."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="steady_state_log96.npz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--cadence", type=float, default=2.0**-12)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    args = parser.parse_args()

    system = common.build_system()
    data = np.load(common.RESULTS / args.input)
    states = np.ascontiguousarray(data["final_state"], dtype=np.float32)
    n_grid = states.shape[1]
    targets = common.parameter_array(system, data["ach"], data["iso"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    peaks_path = args.out_dir / "peak_times.npy"
    status_path = args.out_dir / "failed.npy"
    mode = "r+" if args.start else "w+"
    peaks = np.lib.format.open_memmap(
        peaks_path, mode=mode, dtype=np.float32, shape=(n_grid, n_grid, 3)
    )
    failed = np.lib.format.open_memmap(
        status_path, mode=mode, dtype=np.bool_, shape=(n_grid, n_grid)
    )
    (args.out_dir / "settings.json").write_text(json.dumps({
        "input": args.input, "batch": args.batch,
        "duration_s": args.duration, "cadence_s": args.cadence,
    }, indent=2))

    solver = common.make_solver(
        system,
        output_types=["peaks[3]"],
        summarise_variables=[common.VOLTAGE_LABEL],
        save_variables=[],
        sample_summaries_every=args.cadence,
        summarise_every=args.duration,
        time_logging_level="default",
    )
    params = np.ascontiguousarray(np.tile(targets, args.batch))
    start = perf_counter()
    kernel_total = 0.0
    stop = n_grid if args.stop is None else args.stop
    for first in range(args.start, stop, args.batch):
        sources = np.arange(first, min(first + args.batch, n_grid))
        inits = np.ascontiguousarray(
            np.repeat(states[:, sources], n_grid, axis=1)
        )
        batch_params = params[:, : inits.shape[1]]
        result, kernel_ms, _ = common.timed_solve(
            solver, inits, batch_params, args.duration
        )
        kernel_total += kernel_ms / 1e3
        slots = np.asarray(result.summaries_array)[0].T
        bad = common.failed_runs(result)
        del result
        # Update k samples t = (k + 1) * cadence; 0 marks an empty slot.
        times = np.where(slots > 0, (slots + 1.0) * args.cadence, np.nan)
        peaks[sources] = times.reshape(sources.size, n_grid, 3)
        failed[sources] = bad.reshape(sources.size, n_grid)
        done = sources[-1] + 1
        if (first // args.batch) % 100 == 0 or done >= stop:
            peaks.flush()
            failed.flush()
            print(
                f"sources {done}/{n_grid}: kernel {kernel_total:.0f} s, "
                f"wall {perf_counter() - start:.0f} s",
                flush=True,
            )
    solver.close()
    peaks.flush()
    failed.flush()


if __name__ == "__main__":
    main()
