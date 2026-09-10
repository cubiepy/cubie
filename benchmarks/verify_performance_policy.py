#!/usr/bin/env python
"""Check the ``auto_performance`` defaults and the ``Solver.optimize``
candidate set against the measured landscape of a device.

Arms per configuration: ``tierA`` (every default applied), ``plain``
(``auto_performance=False``), every ``optimisation_candidates(force=True)``
candidate, and extra arms outside the candidate set. Every timed arm runs
at the natural residency, the residency rule, one and two blocks under
natural, and one block per SM, at the ``launch_candidates`` block sizes.

``--score`` prints per configuration the measured best and the loss of
the plain default, the ``tierA`` default and the cell ``optimize`` would
pick, as counts within 5 %, losses and worst loss; then the candidate
gaps, the residency-rule failures and the arms whose failed-run count
or NaN pattern differs from the reference.

Usage::

    python benchmarks/verify_performance_policy.py --out records.jsonl
        [--preset quick|full] [--systems a,b] [--algos x,y]
        [--n-runs N] [--workers 4] [--icache-kib 64] [--score]

``--preset quick`` runs the large-frame configurations with the ``tierA``
and ``plain`` arms only.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import policy_landscape as pl  # noqa: E402
from policy_landscape import FULL, ROLLED, ArmSpec  # noqa: E402

FULL_SYSTEMS = (
    "chain20", "chain64", "chain32_c8", "lorenz96_10", "lorenz96_40",
    "lorenz",
)
FULL_ALGOS = (
    "kvaerno3", "kvaerno5", "radau_iia_3", "radau_iia_5",
    "kvaerno3_bicgstab", "radau_iia_5_bicgstab", "bogacki-shampine-32",
    "vern7", "tsit5", "rosenbrock23",
)
QUICK_CONFIGS = (
    ("chain64", "radau_iia_3"),
    ("chain64", "radau_iia_5"),
    ("chain64", "kvaerno3_bicgstab"),
    ("chain64", "vern7"),
    ("lorenz96_40", "radau_iia_5"),
    ("lorenz96_40", "vern7"),
    ("chain32_c8", "kvaerno3_bicgstab"),
    ("chain32_c8", "radau_iia_5"),
    ("chain20", "kvaerno5"),
    ("lorenz96_10", "kvaerno3"),
)
ALL_ROLLED = {group: ROLLED for group in pl.UNROLL_GROUPS}
WITHIN = 0.05


def extra_arms(family, solver_kind):
    """Arms outside the ``optimize`` candidate set, per family."""
    newton = "unroll_newton_exits"
    krylov = "unroll_krylov_exits"
    other = "unroll_other_small"
    arms = []
    if family == "ERK":
        arms.append(ArmSpec(
            "o1+stage_accumulator=shared",
            {other: FULL, "stage_accumulator_location": "shared"},
        ))
        arms.append(ArmSpec(
            "o1+stage_rhs=shared",
            {other: FULL, "stage_rhs_location": "shared"},
        ))
    elif family == "DIRK" and solver_kind == "lu":
        arms.append(ArmSpec("n1o0", {newton: FULL, other: ROLLED}))
        arms.append(ArmSpec("n0o0", {newton: ROLLED, other: ROLLED}))
    elif family == "DIRK":
        arms.append(ArmSpec("n1k1o1", {newton: FULL, krylov: FULL}))
        arms.append(ArmSpec("n0k1o1", {newton: ROLLED, krylov: FULL}))
        arms.append(ArmSpec(
            "n0k0o0", {newton: ROLLED, krylov: ROLLED, other: ROLLED}
        ))
        arms.append(ArmSpec(
            "n1k0o0", {newton: FULL, krylov: ROLLED, other: ROLLED}
        ))
    elif family == "FIRK" and solver_kind == "lu":
        arms.append(ArmSpec("n0o0", {newton: ROLLED, other: ROLLED}))
        arms.append(ArmSpec("n1o0", {newton: FULL, other: ROLLED}))
        arms.append(ArmSpec(
            "n0o0+stage_increment=shared",
            {newton: ROLLED, other: ROLLED,
             "stage_increment_location": "shared"},
        ))
        arms.append(ArmSpec("allrolled", dict(ALL_ROLLED)))
    elif family == "FIRK":
        arms.append(ArmSpec("n1k1", {newton: FULL, krylov: FULL}))
        arms.append(ArmSpec("n0k1", {newton: ROLLED, krylov: FULL}))
        arms.append(ArmSpec(
            "n0k0o0", {newton: ROLLED, krylov: ROLLED, other: ROLLED}
        ))
        arms.append(ArmSpec("allrolled", dict(ALL_ROLLED)))
    elif family == "ROS" and solver_kind == "bicgstab":
        arms.append(ArmSpec("k1", {krylov: FULL}))
        arms.append(ArmSpec("k0o0", {krylov: ROLLED, other: ROLLED}))
    return arms


def optimize_candidate_arms(system_name, algo_name, duration):
    """The arms ``Solver.optimize`` would compile for the tierA solver."""
    system = pl.build_system(system_name)
    probe = pl.build_solver(
        system, system_name, algo_name, ArmSpec("tierA"), duration
    )
    try:
        candidates = probe.kernel.single_integrator.optimisation_candidates(
            force=True
        )
    finally:
        probe.close()
    return [
        ArmSpec(
            "opt:" + pl.settings_label(candidate), dict(candidate),
            auto_performance=True, in_optimize=True,
        )
        for candidate in candidates
    ]


def make_arms_for(preset):
    def arms_for(system_name, algo_name):
        family, solver_kind, _ = pl.ALGOS[algo_name]
        arms = [
            ArmSpec("tierA", {}, auto_performance=True),
            ArmSpec("plain", {}, auto_performance=False),
        ]
        if preset == "quick":
            return arms
        duration = pl.duration_for(system_name, algo_name)
        arms.extend(optimize_candidate_arms(system_name, algo_name, duration))
        arms.extend(extra_arms(family, solver_kind))
        return arms

    return arms_for


def tally(name, loss, tallies):
    entry = tallies.setdefault(name, dict(within=0, losses=0, worst=0.0,
                                          where="", missing=0))
    if loss is None:
        entry["missing"] += 1
        return
    if loss <= WITHIN:
        entry["within"] += 1
    else:
        entry["losses"] += 1
    if loss > entry["worst"]:
        entry["worst"] = loss
        entry["where"] = tallies["_current"]


def score(rows):
    tallies = {}
    gaps = []
    residency = []
    outputs = []
    header = (
        f"{'config':30s} {'best':>34s} {'ms':>9s} | {'plain':>7s} "
        f"{'tierA':>7s} {'optim':>7s} | {'rule/nat':>8s} | frame regs"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        key = f"{row['system']}/{row['algo']}"
        tallies["_current"] = key
        best_ms, best_label, best_cell = pl.best_overall(row)
        if best_label is None:
            print(f"{key:30s} no timed cells")
            continue
        plain_ms, _ = pl.policy_time(row, "plain", [("natural", 64)])
        tier_ms, _ = pl.policy_time(
            row, "tierA", [("auto", None), ("rule", 64)]
        )
        nat_ms, _ = pl.policy_time(row, "tierA", [("natural", 64)])
        opt_ms, opt_label = pl.optimize_choice(row)
        if opt_ms == float("inf"):
            opt_ms = None
        losses = {
            "plain": None if plain_ms is None else plain_ms / best_ms - 1,
            "tierA": None if tier_ms is None else tier_ms / best_ms - 1,
            "optimize": None if opt_ms is None else opt_ms / best_ms - 1,
        }
        for name, loss in losses.items():
            tally(name, loss, tallies)
        arms = pl.row_arms(row)
        tier_target = arms["tierA"][1]
        rule_ratio = (
            "" if tier_ms is None or nat_ms is None
            else f"{tier_ms / nat_ms:8.3f}"
        )
        if tier_ms is not None and nat_ms is not None:
            if tier_ms > (1 + WITHIN) * nat_ms:
                residency.append((key, tier_ms / nat_ms))
        if losses["optimize"] is not None and losses["optimize"] > WITHIN:
            gaps.append((key, f"{best_label}@{best_cell}", best_ms,
                         opt_label, opt_ms))
        # Compare failed-run counts and NaN patterns only.
        reference_failed = (row["arms"][0].get("output_check") or {}).get(
            "failed"
        )
        for arm in row["arms"]:
            check = arm.get("output_check") or {}
            if arm.get("error"):
                outputs.append((key, arm["label"], "FAILED " + arm["error"]))
            elif check and (
                check.get("failed") != reference_failed
                or ("nan_match" in check and not check["nan_match"])
            ):
                outputs.append((key, arm["label"], json_line(check)))
        print(
            f"{key:30s} {best_label + '@' + best_cell:>34s} {best_ms:9.3f} | "
            f"{pl.format_loss(plain_ms, best_ms)} "
            f"{pl.format_loss(tier_ms, best_ms)} "
            f"{pl.format_loss(opt_ms, best_ms)} | {rule_ratio:>8s} | "
            f"{tier_target['frame']:5d} {tier_target['regs']:4d}"
        )
    print()
    for name in ("plain", "tierA", "optimize"):
        entry = tallies.get(name)
        if entry is None:
            continue
        print(f"{name:9s} within {int(100 * WITHIN)}%: {entry['within']:3d}"
              f"  losses: {entry['losses']:3d}  worst: "
              f"{100 * entry['worst']:6.1f}% {entry['where']}"
              + (f"  (unmeasured: {entry['missing']})"
                 if entry["missing"] else ""))
    print()
    print(f"candidate gaps (measured best beats the optimize set by > "
          f"{int(100 * WITHIN)}%): {len(gaps)}")
    for key, best, best_ms, opt_label, opt_ms in gaps:
        print(f"  {key:30s} best {best} {best_ms:.3f} ms; optimize "
              f"{opt_label} {opt_ms:.3f} ms")
    print(f"residency-rule failures (rule launch > {int(100 * WITHIN)}% "
          f"slower than the natural launch, tierA kernel): "
          f"{len(residency)}")
    for key, ratio in residency:
        print(f"  {key:30s} rule/natural {ratio:.3f}")
    print(f"arms whose failed-run count or NaN pattern differs from the "
          f"reference arm, or that failed to build: {len(outputs)}")
    for key, label, detail in outputs:
        print(f"  {key:30s} {label:32s} {detail}")
    print("largest final-state difference from the reference arm per "
          "configuration (chaotic systems diverge under any rounding "
          "change):")
    for row in rows:
        diffs = [
            (arm.get("output_check") or {}).get("max_abs_diff", 0.0)
            for arm in row["arms"]
        ]
        print(f"  {row['system']}/{row['algo']:24s} {max(diffs):.3g}")


def json_line(check):
    return " ".join(f"{k}={v}" for k, v in check.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pl.add_common_arguments(parser)
    parser.add_argument("--preset", choices=("quick", "full"),
                        default="full")
    parser.add_argument("--systems", default=None,
                        help="comma-separated system names")
    parser.add_argument("--algos", default=None,
                        help="comma-separated algorithm names")
    parser.add_argument("--n-runs", type=int, default=1 << 18)
    args = parser.parse_args()
    if args.score:
        score(pl.load_rows(args.out))
        return
    log = pl.make_logger(args.log)
    pl.check_device(log)
    pl.apply_icache_override(pl.icache_bytes_from(args))
    if args.preset == "quick":
        configs = list(QUICK_CONFIGS)
    else:
        systems = (
            FULL_SYSTEMS if args.systems is None
            else tuple(args.systems.split(","))
        )
        algos = (
            FULL_ALGOS if args.algos is None
            else tuple(args.algos.split(","))
        )
        configs = [(s, a) for s in systems for a in algos]
    if args.systems is not None or args.algos is not None:
        systems = set(args.systems.split(",")) if args.systems else None
        algos = set(args.algos.split(",")) if args.algos else None
        configs = [
            (s, a) for s, a in configs
            if (systems is None or s in systems)
            and (algos is None or a in algos)
        ]
    pl.run_jobs(
        [(s, a, args.n_runs) for s, a in configs],
        make_arms_for(args.preset), args, log,
    )


if __name__ == "__main__":
    main()
