"""Score the single-buffer placement rule against the post913 bank.

Rule: shared when the removed L1 miss sectors exceed the added shared
bank wavefronts per warp and step. Win = chosen time within 5% of the
faster placement.

```powershell
python benchmarks/hardware_model/placement_traffic_score.py --bank <post913> --traffic <dir>[,<dir>...]
```
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

TIE = 0.05


def bank_ratios(bank):
    rows = [json.loads(line) for line in
            (Path(bank) / "records.jsonl").read_text().splitlines()]
    timings = defaultdict(list)
    for row in rows:
        if row.get("task") != "solve" or row.get("warm"):
            continue
        timings[(row["system"], row["algo"], row["label"])].append(
            row["kernel_ms"])
    ratios = {}
    for (system_name, algo, label), values in timings.items():
        if label.startswith("baseline") or "+" in label or "@" in label:
            continue
        base = min(timings[(system_name, algo, "baseline")])
        ratios[(system_name, algo, label)] = min(values) / base
    return ratios


def traffic_rows(folders):
    rows = {}
    for folder in folders:
        records = Path(folder) / "records.jsonl"
        if not records.exists():
            continue
        for line in records.read_text().splitlines():
            row = json.loads(line)
            rows[(row["system"], row["algo"], row["buffer"])] = row
    return rows


def per_warp_step(arm):
    counts = arm["counts"]
    scale = arm["waves"] * arm["resident_warps"]
    return {key: value / scale for key, value in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--traffic", required=True)
    parser.add_argument("--materialization", default="promote",
                        choices=("promote", "addressable"))
    args = parser.parse_args()
    ratios = bank_ratios(args.bank)
    traffic = traffic_rows(args.traffic.split(","))
    print(f"{'config':32s} {'buffer':26s} {'ratio':>6s} {'part':>5s} "
          f"{'miss_loc':>8s} {'miss_sh':>8s} {'wavefr':>7s} {'choice':>7s} "
          f"{'loss':>6s} verdict")
    wins = losses = 0
    worst = (0.0, None)
    default_losses = 0
    default_worst = (0.0, None)
    unscored = []
    for key in sorted(ratios):
        system_name, algo, label = key
        row = traffic.get(key)
        if row is None:
            continue
        if row["status"] != "ok":
            unscored.append((key, "lowering failed"))
            continue
        arms = {(arm["space"], arm["materialization"]): arm
                for arm in row["arms"]}
        local = arms.get(("local", args.materialization))
        shared = arms.get(("shared", args.materialization))
        if local is None or shared is None or local["counts"] is None:
            unscored.append((key, "no local arm"))
            continue
        if shared["counts"] is None:
            unscored.append((key, "shared arm illegal at 64 threads"))
            continue
        loc = per_warp_step(local)
        sha = per_warp_step(shared)
        removed = loc["l1_misses"] - sha["l1_misses"]
        added = sha["shared_bank_wavefronts"]
        choice = "shared" if removed > added else "local"
        ratio = ratios[key]
        measured = "shared" if ratio < 1 else "local"
        chosen_ratio = ratio if choice == "shared" else 1.0
        fastest = min(ratio, 1.0)
        loss = chosen_ratio / fastest - 1
        verdict = "win" if loss <= TIE else "LOSS"
        wins += verdict == "win"
        losses += verdict == "LOSS"
        if loss > worst[0]:
            worst = (loss, key)
        default_loss = 1.0 / fastest - 1
        default_losses += default_loss > TIE
        if default_loss > default_worst[0]:
            default_worst = (default_loss, key)
        print(f"{system_name + '/' + algo:32s} {label:26s} {ratio:6.3f} "
              f"{shared['carveout'] // 1024:4d}K {loc['l1_misses']:8.0f} "
              f"{sha['l1_misses']:8.0f} {added:7.0f} {choice:>7s} "
              f"{100 * loss:5.1f}% {verdict} (measured {measured})")
    print(f"\nrule: wins {wins} losses {losses}; worst-case loss "
          f"{100 * worst[0]:.1f}% ({worst[1]})")
    print(f"default local: losses {default_losses}; worst-case loss "
          f"{100 * default_worst[0]:.1f}% ({default_worst[1]})")
    for key, reason in unscored:
        print(f"unscored {key}: {reason}")


if __name__ == "__main__":
    main()
