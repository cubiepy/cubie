"""Save each run's full state at a peak, the next trough, and mid-cycle."""

import argparse

import numpy as np

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="steady_state_log96.npz")
    parser.add_argument("--window", type=float, default=8.0)
    parser.add_argument("--save-every", type=float, default=2.0**-11)
    parser.add_argument("--batch", type=int, default=384)
    args = parser.parse_args()

    system = common.build_system()
    data = np.load(common.RESULTS / args.input)
    names = list(data["state_names"])
    voltage_row = names.index(common.VOLTAGE_LABEL)
    final = np.ascontiguousarray(data["final_state"])
    n_states, n_runs = final.shape
    params = common.parameter_array(system, data["ach"], data["iso"])
    phases = ("peak", "trough", "mid")
    states = {p: final.copy() for p in phases}
    times = {p: np.full(n_runs, np.nan) for p in phases}
    found = np.zeros(n_runs, dtype=bool)

    solver = common.make_solver(
        system, output_types=["state", "time"], save_every=args.save_every
    )
    for first in range(0, n_runs, args.batch):
        runs = np.arange(first, min(first + args.batch, n_runs))
        result = solver.solve(
            initial_values=np.ascontiguousarray(final[:, runs]),
            parameters=np.ascontiguousarray(params[:, runs]),
            duration=args.window,
        )
        trace = np.asarray(result.time_domain_array)
        time = np.asarray(result.time[:, 0], dtype=np.float64)
        voltage = trace[:, voltage_row, :]
        indices, _ = common.dense_peaks(time, voltage)
        for k, run in enumerate(runs):
            # Beats overshoot 0 mV; the start-up row is not a peak.
            peaks = indices[k][voltage[indices[k], k] > 0]
            if peaks.size < 2:
                continue
            p1, p2 = peaks[0], peaks[1]
            trough = p1 + int(np.argmin(voltage[p1:p2, k]))
            mid = int(np.argmin(np.abs(time - 0.5 * (time[p1] + time[p2]))))
            for phase, row in zip(phases, (p1, trough, mid)):
                states[phase][:, run] = trace[row, :n_states, k]
                times[phase][run] = time[row]
            found[run] = True
        del result, trace
        print(f"runs {runs[-1] + 1}/{n_runs}", flush=True)
    solver.close()

    out = common.RESULTS / args.input.replace(".npz", "_phases.npz")
    np.savez_compressed(
        out,
        **{f"{p}_state": states[p] for p in phases},
        **{f"{p}_time": times[p] for p in phases},
        found=found, save_every=args.save_every,
    )
    print(f"phases found for {int(found.sum())}/{n_runs}; wrote {out}")


if __name__ == "__main__":
    main()
