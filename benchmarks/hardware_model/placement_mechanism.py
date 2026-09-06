"""Join placement bank ratios to compile-row spill, frame and partition deltas."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from placement_score import UNIFIED, partition


def load(bank):
    bank = Path(bank)
    rows = [json.loads(line)
            for line in (bank / "records.jsonl").read_text().splitlines()]
    compiles = {}
    for line in (bank / "compiles.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("status") == "ok":
            compiles[(row["system"], row["algo"],
                      "+".join(row["buffers"]) or "baseline")] = row
    return rows, compiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank")
    args = parser.parse_args()
    rows, compiles = load(args.bank)
    timings = defaultdict(list)
    geometry = {}
    candidates = {}
    for row in rows:
        if row.get("task") == "features":
            candidates[(row["system"], row["algo"])] = {
                c["name"]: c for c in row["candidates"]}
        if row.get("task") != "solve":
            continue
        key = (row["system"], row["algo"], row["label"])
        if row.get("warm"):
            geometry[key] = row["geometry"]
        else:
            timings[key].append(row["kernel_ms"])
    print(f"{'config':34s} {'buffer':26s} {'ratio':>6s} {'bytes':>6s} "
          f"{'dSpillLd':>8s} {'dSpillSt':>8s} {'dLocal':>7s} {'dRegs':>5s} "
          f"{'base_ld':>8s} {'part':>5s} verdict")
    for system, algo in sorted({k[:2] for k in timings}):
        base = compiles.get((system, algo, "baseline"))
        base_ms = min(timings[(system, algo, "baseline")])
        for key in sorted(k for k in timings if k[:2] == (system, algo)):
            label = key[2]
            if label.startswith("baseline") or "@" in label or "+" in label:
                continue
            comp = compiles.get((system, algo, label))
            if comp is None or base is None:
                continue
            ratio = min(timings[key]) / base_ms
            geo = geometry.get(key) or {}
            part = partition(geo.get("blocks_per_sm", 0),
                             geo.get("dynshared", 0)) if geo else 0
            entry = candidates.get((system, algo), {}).get(label, {})
            size = entry.get("elements", 0) * entry.get("itemsize", 0)
            verdict = ("win" if ratio < 0.95 else
                       "loss" if ratio > 1.05 else "tie")
            print(f"{system + '/' + algo:34s} {label:26s} {ratio:6.3f} "
                  f"{size:6d} "
                  f"{base['spill_load_bytes'] - comp['spill_load_bytes']:8d} "
                  f"{base['spill_store_bytes'] - comp['spill_store_bytes']:8d} "
                  f"{base['local_bytes'] - comp['local_bytes']:7d} "
                  f"{base['regs'] - comp['regs']:5d} "
                  f"{base['spill_load_bytes']:8d} {part // 1024:4d}K {verdict}")


if __name__ == "__main__":
    main()
