"""Sample cAMP densely from the settled grid; write per-run statistics."""

import argparse

import numpy as np

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="steady_state_log96.npz")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--save-every", type=float, default=2.0**-10)
    args = parser.parse_args()

    system = common.build_system()
    data = np.load(common.RESULTS / args.input)
    names = list(data["state_names"])
    solver = common.make_solver(
        system,
        output_types=["state", "time"],
        save_variables=["cAMP_cAMP", common.VOLTAGE_LABEL],
        save_every=args.save_every,
    )
    result = solver.solve(
        initial_values=np.ascontiguousarray(data["final_state"]),
        parameters=common.parameter_array(system, data["ach"], data["iso"]),
        duration=args.duration,
    )
    legend = [label for _, label in sorted(result.time_domain_legend.items())]
    camp_row = next(i for i, s in enumerate(legend) if s.startswith("cAMP"))
    camp = np.array(result.time_domain_array[:, camp_row, :])
    failed = common.failed_runs(result)
    del result
    solver.close()
    out = common.RESULTS / args.input.replace(".npz", "_camp.npz")
    np.savez_compressed(
        out,
        ach=data["ach"], iso=data["iso"],
        median=np.median(camp, axis=0), mean=camp.mean(axis=0),
        minimum=camp.min(axis=0), maximum=camp.max(axis=0),
        snapshot=data["final_state"][names.index("cAMP_cAMP")],
        failed=failed, duration=args.duration, save_every=args.save_every,
    )
    span = camp.max(axis=0) - camp.min(axis=0)
    print(
        f"cAMP median {np.median(camp, axis=0).min():.5f}-"
        f"{np.median(camp, axis=0).max():.5f} mM; within-run range max "
        f"{span.max():.2e} mM, median {np.median(span):.2e} mM; failed "
        f"{int(failed.sum())}; wrote {out}"
    )


if __name__ == "__main__":
    main()
