"""Time a full-grid ``peaks[3]`` capture launched from the settled states."""

import argparse
import json

import numpy as np

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    system = common.build_system()
    data = np.load(common.RESULTS / "steady_state.npz")
    state = np.ascontiguousarray(data["final_state"], dtype=np.float32)
    params = common.parameter_array(system, data["ach"], data["iso"])
    configs = [("final state only", {"output_types": ["state"]})]
    for cadence in (2.0**-12, 2.0**-10):
        configs.append((
            f"peaks[3] sampled every {cadence:.6f} s",
            {
                "output_types": ["peaks[3]"],
                "summarise_variables": [common.VOLTAGE_LABEL],
                "save_variables": [],
                "sample_summaries_every": cadence,
                "summarise_every": args.duration,
            },
        ))
    rows = []
    for label, settings in configs:
        solver = common.make_solver(
            system, time_logging_level="default", **settings
        )
        kernel = []
        for _ in range(args.repeats + 1):
            result, kernel_ms, _ = common.timed_solve(
                solver, state, params, args.duration
            )
            kernel.append(kernel_ms)
            if "peaks" in label:
                slots = np.asarray(result.summaries_array)[0]
                captured = (slots > 0).sum(axis=0)
                output_bytes = slots.nbytes
            del result
        solver.close()
        row = {
            "config": label,
            "runs": int(state.shape[1]),
            "duration_s": args.duration,
            "kernel_s": min(kernel[1:]) / 1e3,
        }
        if "peaks" in label:
            row["runs_with_3_peaks"] = int((captured == 3).sum())
            row["output_bytes"] = int(output_bytes)
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = common.RESULTS / "three_beat_cost.json"
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
