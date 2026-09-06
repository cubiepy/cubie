"""Score placement bank rows by ratio to baseline and shared partition."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

UNIFIED = 131072
RESERVE = 1024
SUPPORTED = (8192, 16384, 32768, 65536, 102400)
# resident blocks -> [(dynamic bytes ceiling, partition)]; SHARED_PARTITION.md
TABLE = {
    1: [(6144, 8192), (12288, 16384), (24576, 32768), (49152, 65536)],
    2: [(4, 8192), (10752, 65536)],
    3: [(4, 8192), (1024, 16384), (4096, 32768), (16384, 65536)],
    4: [(4, 8192), (5376, 65536)],
    6: [(4, 16384), (1024, 32768), (8192, 65536)],
    12: [(512, 32768), (4096, 65536)],
    24: [(4, 32768), (1024, 65536)],
}


def partition(blocks, dynamic):
    keys = sorted(TABLE)
    key = min((k for k in keys if k >= blocks), default=keys[-1])
    for ceiling, executed in TABLE[key]:
        if dynamic <= ceiling:
            return executed
    need = blocks * (dynamic + RESERVE)
    for size in SUPPORTED:
        if need <= size:
            return size
    return SUPPORTED[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank")
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (Path(args.bank) / "records.jsonl").read_text().splitlines()]
    timings = defaultdict(list)
    geometry = {}
    for row in rows:
        if row.get("task") != "solve":
            continue
        key = (row["system"], row["algo"], row["label"])
        if row.get("warm"):
            geometry[key] = row["geometry"]
        else:
            timings[key].append(row["kernel_ms"])
    configs = sorted({k[:2] for k in timings})
    print(f"{'config':34s} {'buffer':28s} {'ratio':>6s} {'bs':>4s} "
          f"{'dyn':>6s} {'blk':>3s} {'part':>6s} {'L1':>6s}")
    wins = losses = ties = 0
    by_partition = defaultdict(lambda: [0, 0, 0])
    for system, algo in configs:
        base = min(timings[(system, algo, "baseline")])
        for key in sorted(k for k in timings if k[:2] == (system, algo)):
            label = key[2]
            if label.startswith("baseline"):
                continue
            ratio = min(timings[key]) / base
            geo = geometry.get(key) or {}
            blocks = geo.get("blocks_per_sm", 0)
            dyn = geo.get("dynshared", 0)
            part = partition(blocks, dyn) if geo else 0
            l1 = (UNIFIED - part) // 1024 if part else 0
            verdict = ("win" if ratio < 0.95 else
                       "loss" if ratio > 1.05 else "tie")
            wins += verdict == "win"
            losses += verdict == "loss"
            ties += verdict == "tie"
            by_partition[part][("win", "loss", "tie").index(verdict)] += 1
            print(f"{system + '/' + algo:34s} {label:28s} {ratio:6.3f} "
                  f"{geo.get('blocksize', 0):4d} {dyn:6d} {blocks:3d} "
                  f"{part // 1024:5d}K {l1:5d}K {verdict}")
    print(f"\nshared rows: wins {wins} losses {losses} ties {ties}")
    for part in sorted(by_partition):
        w, l, t = by_partition[part]
        print(f"partition {part // 1024:4d}K: wins {w} losses {l} ties {t}")


if __name__ == "__main__":
    main()
