"""Trial the ``peaks`` summary metric against dense voltage saves.

Part 1 (values): on a ``side`` x ``side`` subset of the grid, integrate
from the initial state once with dense voltage saves at the finest
cadence and once per cadence with only ``peaks[n]`` on the voltage.
At equal cadence both runs clamp steps to the same sample grid, so the
metric's peak sample indices must equal the dense trace's local-maximum
indices exactly. At coarser cadences, the metric's peak times (sample
times from the float32-accumulated grid) are compared with the
parabola-refined dense peak times.

Part 2 (performance): on the full 256 x 256 grid, the kernel time of
each configuration over ``--perf-duration`` seconds, best of
``--repeats``, with a ``save_last``-only baseline that never clamps.

Usage::

    python benchmarks/fabbri_transients/peaks_trial.py [--side 16]
        [--duration 20] [--cadences C ...]
        [--perf-duration 2] [--repeats 3] [--perf-dense-cadence C]
"""

import argparse
import json

import numpy as np

import common


def peak_solver(system, cadence, window, n_peaks):
    """Return a solver recording only voltage peaks."""
    return common.make_solver(
        system,
        output_types=[f"peaks[{n_peaks}]"],
        summarise_variables=[common.VOLTAGE_LABEL],
        save_variables=[],
        sample_summaries_every=cadence,
        summarise_every=window,
        time_logging_level="default",
    )


def dense_solver(system, cadence):
    """Return a solver saving voltage and time every ``cadence``."""
    return common.make_solver(
        system,
        output_types=["state", "time"],
        save_variables=[common.VOLTAGE_LABEL],
        save_every=cadence,
        time_logging_level="default",
    )


def last_solver(system):
    """Return a solver saving only the final state."""
    return common.make_solver(
        system,
        output_types=["state"],
        save_last=True,
        time_logging_level="default",
    )


def metric_indices(result):
    """Return per-run sorted peak sample indices from a peaks result."""
    summaries = np.asarray(result.summaries_array)
    # (window, n_peaks, run); an empty slot holds 0.
    flat = summaries.transpose(2, 0, 1).reshape(summaries.shape[2], -1)
    return [np.sort(row[row > 0]).astype(np.int64) for row in flat]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument(
        "--cadences", type=float, nargs="+",
        default=[2.0**-14, 2.0**-13, 2.0**-12, 2.0**-11, 2.0**-10],
    )
    parser.add_argument("--perf-duration", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--perf-dense-cadence", type=float, default=2.0**-10,
        help="dense-save cadence timed on the full grid (host RAM "
        "bound: 8 bytes per run per sample)",
    )
    parser.add_argument("--skip-perf", action="store_true")
    args = parser.parse_args()

    system = common.build_system()
    pick = np.round(
        np.linspace(0, common.GRID_SIDE - 1, args.side)
    ).astype(int)
    ach_axis, iso_axis = common.axes()
    ach_grid, iso_grid = np.meshgrid(ach_axis[pick], iso_axis[pick])
    ach, iso = ach_grid.ravel(), iso_grid.ravel()
    n_runs = ach.size
    inits = common.initial_state_array(system, n_runs)
    params = common.parameter_array(system, ach, iso)
    n_peaks = int(np.ceil(args.duration / 0.3)) + 4
    report = {"side": args.side, "duration": args.duration}

    finest = min(args.cadences)
    solver = dense_solver(system, finest)
    result, _, _ = common.timed_solve(solver, inits, params, args.duration)
    dense_time = np.asarray(result.time[:, 0], dtype=np.float64)
    voltage = np.array(result.time_domain_array[:, 0, :])
    dense_failed = common.failed_runs(result)
    dense_idx, dense_refined = common.dense_peaks(dense_time, voltage)
    del result
    solver.close()
    nominal = np.arange(dense_time.size) * finest
    report["dense_grid_max_drift_s"] = float(
        np.max(np.abs(dense_time - nominal))
    )
    print(
        f"dense {finest:g} s: failed {int(dense_failed.sum())}; save "
        f"grid drift from k*s up to "
        f"{report['dense_grid_max_drift_s'] * 1e3:.3f} ms"
    )

    report["values"] = []
    for cadence in sorted(args.cadences):
        solver = peak_solver(system, cadence, args.duration, n_peaks)
        result, _, _ = common.timed_solve(
            solver, inits, params, args.duration
        )
        failed = common.failed_runs(result)
        indices = metric_indices(result)
        del result
        solver.close()
        n_samples = int(round(args.duration / cadence))
        grid = common.float32_sample_grid(
            cadence, n_samples + 1, t_end=args.duration
        )
        row = {"cadence": cadence, "failed": int(failed.sum())}
        if cadence == finest:
            # Update k lands on save row k + 1.
            same = [
                np.array_equal(m + 1, d)
                for m, d in zip(indices, dense_idx)
            ]
            row["index_match_runs"] = int(np.sum(same))
            row["grid_matches_saves"] = bool(
                np.array_equal(
                    grid[:n_samples].astype(np.float32),
                    dense_time[1 : n_samples + 1].astype(np.float32),
                )
            )
        errors = []
        count_mismatch = 0
        for m, refined in zip(indices, dense_refined):
            times = grid[m]
            if times.size != refined.size:
                count_mismatch += 1
                continue
            errors.append(times - refined)
        errors = np.concatenate(errors) if errors else np.array([0.0])
        row["count_mismatch_runs"] = count_mismatch
        row["abs_error_ms_median"] = float(np.median(np.abs(errors)) * 1e3)
        row["abs_error_ms_max"] = float(np.max(np.abs(errors)) * 1e3)
        interval_errors = []
        for m, refined in zip(indices, dense_refined):
            if m.size == refined.size and m.size > 1:
                interval_errors.append(
                    np.diff(grid[m]) - np.diff(refined)
                )
        interval_errors = np.concatenate(interval_errors)
        row["interval_error_ms_max"] = float(
            np.max(np.abs(interval_errors)) * 1e3
        )
        report["values"].append(row)
        print(json.dumps(row))

    if not args.skip_perf:
        ach_all, iso_all = common.grid_values()
        full_inits = common.initial_state_array(system, ach_all.size)
        full_params = common.parameter_array(system, ach_all, iso_all)
        configs = [("save_last only", lambda: last_solver(system))]
        configs.append(
            (f"dense V every {args.perf_dense_cadence:g}",
             lambda: dense_solver(system, args.perf_dense_cadence))
        )
        for cadence in sorted(args.cadences):
            n_perf = int(np.ceil(args.perf_duration / 0.3)) + 4
            configs.append(
                (f"peaks[{n_perf}] every {cadence:g}",
                 lambda c=cadence, n=n_perf: peak_solver(
                     system, c, args.perf_duration, n)),
            )
        report["perf"] = []
        for label, factory in configs:
            solver = factory()
            times = []
            for _ in range(args.repeats + 1):
                result, kernel_ms, wall_s = common.timed_solve(
                    solver, full_inits, full_params, args.perf_duration
                )
                times.append((kernel_ms, wall_s))
                del result
            solver.close()
            # The first solve compiles; time the rest.
            best_kernel = min(t[0] for t in times[1:])
            best_wall = min(t[1] for t in times[1:])
            row = {
                "config": label,
                "runs": int(ach_all.size),
                "kernel_ms": best_kernel,
                "wall_s": best_wall,
            }
            report["perf"].append(row)
            print(json.dumps(row))

    common.RESULTS.mkdir(exist_ok=True)
    out = common.RESULTS / "peaks_trial.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
