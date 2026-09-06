"""Tabulate projected instruction slots against post882 SASS totals."""

import argparse
import json
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return {(r["system"], r["algo"], r["policy"]): r for r in rows}


def scenario(row, key="inline|rolled"):
    return row.get("scenarios", {}).get(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    args = parser.parse_args()
    rows = load(args.records)
    configs = sorted({(s, a) for s, a, _ in rows})
    print(f"{'config':38s} {'policy':9s} {'sass':>6s} {'cov':>6s} "
          f"{'cap':>6s} {'cap/sass':>8s} {'dFull':>7s} {'dProj':>7s} "
          f"{'unmapped':>8s}")
    ratios = []
    delta_ratios = []
    for system, algo in configs:
        full = rows.get((system, algo, "u1111111"))
        base = scenario(full) if full and full["status"] == "ok" else None
        for label in sorted(k[2] for k in rows if k[:2] == (system, algo)):
            row = rows[(system, algo, label)]
            sass = row["sass"]["instructions"]
            if row["status"] != "ok":
                print(f"{system + '/' + algo:38s} {label:9s} {sass:6d} "
                      f"ERROR {row['error'].strip().splitlines()[-1][:60]}")
                continue
            sc = scenario(row)
            cap = sc["cap_slots"]
            unmapped = sum(sc["cap_unmapped"].values())
            d_full = d_proj = ""
            if base and label != "u1111111":
                d_full = full["sass"]["instructions"] - sass
                d_proj = base["cap_slots"] - cap
                if d_full:
                    delta_ratios.append((system, algo, label, d_proj / d_full))
            ratios.append(cap / sass)
            print(f"{system + '/' + algo:38s} {label:9s} {sass:6d} "
                  f"{sc['covered_slots']:6d} {cap:6d} {cap / sass:8.3f} "
                  f"{d_full!s:>7} {d_proj!s:>7} {unmapped:8d}")
    if ratios:
        ratios.sort()
        mid = ratios[len(ratios) // 2]
        print(f"\ncap/sass over {len(ratios)} rows: min {ratios[0]:.3f} "
              f"median {mid:.3f} max {ratios[-1]:.3f}")
    if delta_ratios:
        values = sorted(r for *_, r in delta_ratios)
        print(f"projected/actual delta over {len(values)} rows: min "
              f"{values[0]:.3f} median {values[len(values) // 2]:.3f} max "
              f"{values[-1]:.3f}")


if __name__ == "__main__":
    main()
