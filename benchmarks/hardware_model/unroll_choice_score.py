"""Score the delivery-curve unroll choice against the timing banks.

Hot footprint per candidate = outside slots + Newton body per static copy
(+ Krylov body per copy inside it); smallest delivery service wins, the
default inside a 2% band. Win = chosen time within 5% of the fastest.

```powershell
python benchmarks/hardware_model/unroll_choice_score.py --bank lu
python benchmarks/hardware_model/unroll_choice_score.py --bank split --regime cap
```
"""

import argparse
import json
import math
from pathlib import Path
import sys

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
for entry in (str(REPO), str(BENCH)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.hardware_model import candidate_selection as selection  # noqa: E402
from benchmarks.hardware_model import decision_model  # noqa: E402
from benchmarks.hardware_model import fetch_rule_score as fetch  # noqa: E402

NOTES = fetch.NOTES
RECORDS = NOTES / "static_slots_e1/records.jsonl"
BANKS = {
    "lu": dict(
        audit=NOTES / "recovered/post882_audit_strict.json",
        labels={("full", "count1"): "u1111111", ("count1", "count1"): "u1111110"},
        skip_bicgstab=True,
    ),
    "split": dict(
        audit=NOTES / "recovered/split_flags_audit_strict.json",
        labels={("full", "full"): "u11111111", ("full", "count1"): "u11111110",
                ("count1", "full"): "u11111101",
                ("count1", "count1"): "u11111100"},
        skip_bicgstab=False,
    ),
}
DEFAULT = ("full", "count1")
BAND = 0.02
FOOTPRINT_BAND = 0.05
TIE = 0.05


def region_slots(row):
    """Static slots outside the loops and per loop body."""
    out = dict(outside=0, newton=0, krylov=0, error_krylov=0)
    for region in row["regions"]:
        role, phase = region["role"], region["phase"]
        slots = region["slots"]
        if role == "main_newton" and phase in ("body", "exit_vote"):
            out["newton"] += slots
        elif role == "main_linear" and phase in ("body", "exit_vote"):
            out["krylov"] += slots
        elif role == "main_linear":
            out["newton"] += slots
        elif role == "error_linear" and phase in ("body", "exit_vote"):
            out["error_krylov"] += slots
        else:
            out["outside"] += slots
    newton = [loop for loop in row["loops"]
              if loop["group"] == "unroll_newton_exits"]
    krylov = [loop for loop in row["loops"]
              if loop["group"] == "unroll_krylov_exits"]
    out["newton_loops"] = len(newton)
    out["newton_cap"] = max((loop["cap"] for loop in newton), default=0)
    out["krylov_cap"] = max((loop["cap"] for loop in krylov), default=0)
    return out


def copies(level, visits, cap):
    if level == "count1" or cap == 0:
        return 1
    return min(cap, max(1, math.ceil(visits)))


def hot_kb(slots, newton_level, krylov_level, newton_visits, krylov_visits):
    """Distinct instruction bytes a step touches under one policy."""
    per_loop = newton_visits / max(1, slots["newton_loops"])
    c_n = copies(newton_level, per_loop, slots["newton_cap"])
    c_k = copies(krylov_level, krylov_visits, slots["krylov_cap"])
    total = (slots["outside"]
             + c_n * (slots["newton"] + c_k * slots["krylov"])
             + c_k * slots["error_krylov"])
    return total * fetch.WIDTH / 1024 / decision_model.SLOT_FACTOR


def choose(service, default, beyond):
    """Smallest service; beyond the curve the smaller footprint; ties default."""
    inside = {c: v for c, v in service.items() if v[1] <= beyond}
    if inside:
        best = min(inside.values())[0]
        if default in inside and inside[default][0] <= best * (1 + BAND):
            return default
        return min(inside, key=lambda c: inside[c][0])
    # Past the curve only the footprint ordering is known.
    smallest = min(service.values())[1]
    if service[default][1] <= smallest * (1 + FOOTPRINT_BAND):
        return default
    return min(service, key=lambda c: service[c][1])


def measured(audit, labels):
    """Fastest measured kernel time per candidate, relative to all full."""
    out = {}
    wanted = set(labels.values())
    for config in audit["configs"]:
        key = (config["system"], config["algo"])
        best = {}
        for obs in config.get("observations", []):
            if not obs.get("eligible") or obs["policy"] not in wanted:
                continue
            ratio = obs["ratio_to_full"]
            if obs["policy"] not in best or ratio < best[obs["policy"]]:
                best[obs["policy"]] = ratio
        out[key] = best
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=sorted(BANKS), default="lu")
    parser.add_argument("--regime", choices=("measured", "cap"),
                        default="measured")
    parser.add_argument("--records", default=str(RECORDS))
    args = parser.parse_args()
    bank = BANKS[args.bank]
    labels = bank["labels"]
    rows = {}
    for line in Path(args.records).read_text().splitlines():
        row = json.loads(line)
        if row.get("status") == "ok":
            rows[(row["system"], row["algo"])] = row
    counts = fetch.measured_counts()
    curve = fetch.delivery_curve()
    times = measured(json.loads(bank["audit"].read_text()), labels)
    print(f"{'config':32s} {'warps':>5s} " + " ".join(
        f"{'/'.join(c)[:12]:>13s}" for c in labels)
        + f" {'choice':>13s} {'fastest':>13s} {'loss':>6s} verdict")
    wins = losses = 0
    worst = (0.0, None)
    default_losses = 0
    default_worst = (0.0, None)
    for key in sorted(rows):
        system_name, algo = key
        if bank["skip_bicgstab"] and algo.endswith("_bicgstab"):
            continue
        if not algo.endswith("_bicgstab") and args.bank == "split":
            continue
        present = {c: label for c, label in labels.items()
                   if label in times.get(key, {})}
        if len(present) < 2:
            continue
        row = rows[key]
        slots = region_slots(row)
        count = counts.get(key)
        if args.regime == "measured":
            if count is None:
                continue
            newton_visits = count["newton"]
            krylov_visits = (count["krylov"] / count["newton"]
                             if count["newton"] else count["krylov"])
        else:
            newton_visits = slots["newton_cap"] * slots["newton_loops"]
            krylov_visits = slots["krylov_cap"]
        registers = row.get("registers") or row["peak"]["R"]
        geometry = selection.residency(decision_model.HARDWARE, registers,
                                       64, 0, 4, 8192)
        warps = geometry["resident_warps_per_sm"]
        service = {}
        for candidate in present:
            kb = hot_kb(slots, candidate[0], candidate[1], newton_visits,
                        krylov_visits)
            service[candidate] = (fetch.service(curve, kb, warps), kb)
        default = DEFAULT if DEFAULT in present else ("full", "full")
        choice = choose(service, default, fetch.beyond(curve, warps))
        ratios = {c: times[key][label] for c, label in present.items()}
        fastest = min(ratios, key=ratios.get)
        loss = ratios[choice] / ratios[fastest] - 1
        verdict = "win" if loss <= TIE else "LOSS"
        wins += verdict == "win"
        losses += verdict == "LOSS"
        if loss > worst[0]:
            worst = (loss, key)
        default_loss = ratios[default] / ratios[fastest] - 1
        default_losses += default_loss > TIE
        if default_loss > default_worst[0]:
            default_worst = (default_loss, key)
        cells = " ".join(
            f"{service[c][1]:5.0f}K {ratios[c]:6.3f}" if c in service
            else f"{'-':>13s}" for c in labels)
        print(f"{system_name + '/' + algo:32s} {warps:5d} {cells} "
              f"{'/'.join(choice)[:13]:>13s} {'/'.join(fastest)[:13]:>13s} "
              f"{100 * loss:5.1f}% {verdict}")
    print(f"\nmodel: wins {wins} losses {losses}; worst-case loss "
          f"{100 * worst[0]:.1f}% ({'/'.join(worst[1]) if worst[1] else '-'})")
    print(f"default {'/'.join(DEFAULT)}: losses {default_losses}; worst-case "
          f"loss {100 * default_worst[0]:.1f}% "
          f"({'/'.join(default_worst[1]) if default_worst[1] else '-'})")


if __name__ == "__main__":
    main()
