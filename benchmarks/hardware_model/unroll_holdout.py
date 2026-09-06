"""Frozen unroll predictions on unseen sizes and their native score.

`freeze` chooses loop levels for every configuration in a static-slot
record file at the source cap (no measured iteration counts) and writes
`predictions.json` with the record digest. `score` reads the timing bank
that `benchmarks/unroll_landscape.py --config ... --policies ...` wrote
for those configurations and scores the frozen choice against the
fastest eligible candidate: a candidate is eligible when every timed
solve reports no failed trajectory, matching NaN structure, and the
all-full baseline duplicate is byte-exact with the baseline. The frozen
predictions are never updated from the bank.

```powershell
python benchmarks/hardware_model/unroll_holdout.py freeze --records <static_slots>/records.jsonl --out <dir>
python benchmarks/hardware_model/unroll_holdout.py score --frozen <dir> --bank <unroll_landscape out>
```
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
for entry in (str(REPO), str(BENCH)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.hardware_model import decision_model  # noqa: E402
from benchmarks.hardware_model import static_slots  # noqa: E402
from benchmarks.hardware_model import unroll_choice_score as choice  # noqa: E402

BASELINE = "u11111111"
DUPLICATE = BASELINE + "#2"
TIE = choice.TIE


def label_for(algo, levels):
    """Eight-level bank label of a Newton/Krylov level pair."""
    newton, krylov = levels
    if not algo.endswith("_bicgstab"):
        krylov = "full"
    return "u111111" + ("1" if newton == "full" else "0") + (
        "1" if krylov == "full" else "0")


def freeze(args):
    records = Path(args.records)
    rows = static_slots.load(records)
    predictions = {}
    for key in sorted(rows):
        row = rows[key]
        if row.get("status") != "ok":
            continue
        slots = choice.region_slots(row)
        newton = slots["newton_cap"] * slots["newton_loops"]
        result = decision_model.decide(row, args.block, newton,
                                       slots["krylov_cap"])
        levels = (result["unroll_newton_exits"], result["unroll_krylov_exits"])
        candidates = sorted({label_for(row["algo"], tuple(c.split("/")))
                             for c in result["candidates"]})
        predictions["/".join(key)] = dict(
            choice=label_for(row["algo"], levels), levels=list(levels),
            candidates=candidates,
            hot_kb={c: v["hot_kb"] for c, v in result["candidates"].items()},
            resident_warps=result["resident_warps"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(
        kind="frozen_unroll_predictions", block_threads=args.block,
        iteration_regime="source_cap",
        records=dict(path=str(records.resolve()),
                     sha256=hashlib.sha256(records.read_bytes()).hexdigest()),
        predictions=predictions)
    (out / "predictions.json").write_text(json.dumps(payload, indent=1))
    for key, item in predictions.items():
        print(f"{key:36s} {item['choice']}  candidates {item['candidates']}")


def bank_rows(bank):
    timings = defaultdict(list)
    checks = defaultdict(list)
    for line in (Path(bank) / "records.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("task") != "solve":
            continue
        key = (row["system"], row["algo"], row["label"])
        if row.get("warm"):
            checks[key].append(row)
        else:
            timings[key].append(row)
    return timings, checks


def eligible(rows):
    """No failed trajectory and matching NaN structure on every solve."""
    return bool(rows) and all(
        (row.get("status_hist") or {}).get("failed", 0) == 0
        and row.get("nan_match", False) for row in rows)


def score(args):
    frozen = json.loads((Path(args.frozen) / "predictions.json").read_text())
    timings, checks = bank_rows(args.bank)
    print(f"{'config':36s} {'choice':>10s} {'fastest':>10s} {'ratios':>32s} "
          f"{'loss':>6s} verdict")
    wins = losses = 0
    worst = (0.0, None)
    default_losses = 0
    default_worst = (0.0, None)
    for name, item in sorted(frozen["predictions"].items()):
        system_name, algo = name.split("/")
        base_key = (system_name, algo, BASELINE)
        dup_key = (system_name, algo, DUPLICATE)
        if base_key not in timings:
            print(f"{name:36s} no bank rows")
            continue
        # Warm rows carry the numerical checks; timed rows carry times.
        duplicate = checks.get(dup_key, [])
        exact = bool(duplicate) and all(
            row.get("max_abs_diff") == 0 and row.get("runs_differing") == 0
            for row in duplicate)
        if not exact or not eligible(checks.get(base_key, [])):
            print(f"{name:36s} baseline ineligible")
            continue
        base = min(row["kernel_ms"] for row in timings[base_key])
        ratios = {}
        for label in item["candidates"]:
            key = (system_name, algo, label)
            if label == BASELINE:
                ratios[label] = 1.0
                continue
            if key not in timings or not eligible(checks.get(key, [])):
                continue
            ratios[label] = min(r["kernel_ms"] for r in timings[key]) / base
        if item["choice"] not in ratios:
            print(f"{name:36s} chosen {item['choice']} ineligible")
            losses += 1
            continue
        fastest = min(ratios, key=ratios.get)
        loss = ratios[item["choice"]] / ratios[fastest] - 1
        verdict = "win" if loss <= TIE else "LOSS"
        wins += verdict == "win"
        losses += verdict == "LOSS"
        if loss > worst[0]:
            worst = (loss, name)
        default_loss = 1.0 / ratios[fastest] - 1
        default_losses += default_loss > TIE
        if default_loss > default_worst[0]:
            default_worst = (default_loss, name)
        cells = " ".join(f"{k[-2:]}:{v:5.3f}" for k, v in sorted(ratios.items()))
        print(f"{name:36s} {item['choice']:>10s} {fastest:>10s} {cells:>32s} "
              f"{100 * loss:5.1f}% {verdict}")
    print(f"\nmodel: wins {wins} losses {losses}; worst-case loss "
          f"{100 * worst[0]:.1f}% ({worst[1]})")
    print(f"default {BASELINE}: losses {default_losses}; worst-case loss "
          f"{100 * default_worst[0]:.1f}% ({default_worst[1]})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezer = commands.add_parser("freeze")
    freezer.add_argument("--records", required=True)
    freezer.add_argument("--out", required=True)
    freezer.add_argument("--block", type=int, default=64)
    freezer.set_defaults(run=freeze)
    scorer = commands.add_parser("score")
    scorer.add_argument("--frozen", required=True)
    scorer.add_argument("--bank", required=True)
    scorer.set_defaults(run=score)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
