"""Compare cycle intervals: metric vs dense per cadence, dense vs finest."""

import argparse
import json

import numpy as np

import common

CADENCES = [2.0**-k for k in (14, 13, 12, 11, 10)]


def stats(values):
    """Return median, p99 and max of absolute values in ms."""
    values = np.abs(np.concatenate(values)) * 1e3
    return {
        "n": int(values.size),
        "median_ms": float(np.median(values)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    system = common.build_system()
    pick = np.round(
        np.linspace(0, common.GRID_SIDE - 1, args.side)
    ).astype(int)
    ach_axis, iso_axis = common.axes()
    ach_grid, iso_grid = np.meshgrid(ach_axis[pick], iso_axis[pick])
    ach, iso = ach_grid.ravel(), iso_grid.ravel()
    inits = common.initial_state_array(system, ach.size)
    params = common.parameter_array(system, ach, iso)
    n_slots = int(np.ceil(args.duration / 0.25)) + 4

    dense = {}
    metric = {}
    for cadence in CADENCES:
        solver = common.make_solver(
            system,
            output_types=["state", "time"],
            save_variables=[common.VOLTAGE_LABEL],
            save_every=cadence,
        )
        result = solver.solve(
            initial_values=inits, parameters=params,
            duration=args.duration,
        )
        time = np.asarray(result.time[:, 0], dtype=np.float64)
        voltage = np.array(result.time_domain_array[:, 0, :])
        del result
        solver.close()
        indices, refined = common.dense_peaks(time, voltage)
        # Drop the start-up maximum below 0 mV.
        dense[cadence] = [
            r[voltage[i, run] > 0] for run, (i, r) in
            enumerate(zip(indices, refined))
        ]
        solver = common.make_solver(
            system,
            output_types=[f"peaks[{n_slots}]"],
            summarise_variables=[common.VOLTAGE_LABEL],
            save_variables=[],
            sample_summaries_every=cadence,
            summarise_every=args.duration,
        )
        result = solver.solve(
            initial_values=inits, parameters=params,
            duration=args.duration,
        )
        slots = np.asarray(result.summaries_array)[0].T
        del result
        solver.close()
        metric[cadence] = [
            np.sort(row[row > 0]) * cadence + cadence for row in slots
        ]

    report = {"side": args.side, "duration": args.duration, "rows": []}
    finest = CADENCES[0]
    for cadence in CADENCES:
        same, early, late, beats_differ = [], [], [], 0
        outliers = []
        for run in range(ach.size):
            m = metric[cadence][run]
            m = m[m > 0.02]
            d = dense[cadence][run]
            if m.size == d.size and d.size > 1:
                same.append(np.diff(m) - np.diff(d))
            ref = dense[finest][run]
            if ref.size != d.size:
                beats_differ += 1
                continue
            delta = np.diff(d) - np.diff(ref)
            beat_time = ref[1:]
            worst = int(np.argmax(np.abs(delta)))
            if abs(delta[worst]) > 1e-3:
                outliers.append({
                    "ach": float(ach[run]), "iso": float(iso[run]),
                    "beat_time_s": float(beat_time[worst]),
                    "interval_s": float(np.diff(ref)[worst]),
                    "delta_ms": float(delta[worst] * 1e3),
                    "beats_over_1ms": int((np.abs(delta) > 1e-3).sum()),
                })
            early.append(delta[beat_time < args.duration / 2])
            late.append(delta[beat_time >= args.duration / 2])
        row = {
            "cadence_s": cadence,
            "metric_vs_dense_same_cadence": stats(same),
            "runs_with_different_beat_count": beats_differ,
        }
        if cadence != finest:
            row["vs_finest_first_half"] = stats(early)
            row["vs_finest_second_half"] = stats(late)
            row["runs_over_1ms"] = outliers
        report["rows"].append(row)
        print(json.dumps(row))
    out = common.RESULTS / "cadence_intervals.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
