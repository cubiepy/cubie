"""Pre-compile unroll choice for the Newton and Krylov iteration loops.

The choice uses source quantities only: the static region slots of one
policy graph (`static_slots.py`), the resident warps from the occupancy
equations at the allocator's register count, and the measured
instruction-delivery curve. Each candidate policy's hot footprint is the
code a step touches under that policy; the candidate with the smallest
delivery service wins, the default (Newton full, Krylov count 1) inside a
two percent band, and past the measured curve the smaller footprint.
`unroll_choice_score.py` scores the same rule against the timing banks.

```powershell
python benchmarks/hardware_model/decision_model.py chain32 radau_iia_5 --newton 2 --krylov 3
```
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
for entry in (str(REPO), str(BENCH)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.hardware_model import candidate_selection as selection  # noqa: E402
from benchmarks.hardware_model import fetch_rule_score as fetch  # noqa: E402
from benchmarks.hardware_model import static_slots  # noqa: E402
from benchmarks.hardware_model import unroll_choice_score as choice  # noqa: E402

HARDWARE = json.loads(fetch.NOTES.joinpath(
    "verification/placement_v2_author_20260905/cohort_e1/"
    "kvaerno3_bicgstab_request.json").read_text())["hardware"]
# Median projected-slot to SASS ratio over 345 calibration rows.
SLOT_FACTOR = 1.417


def decide(row, block, newton_visits, krylov_visits):
    """Choose loop levels from static slots and per-step visit counts."""
    slots = choice.region_slots(row)
    geometry = selection.residency(HARDWARE, row["registers"], block, 0, 4,
                                   8192)
    warps = geometry["resident_warps_per_sm"]
    curve = fetch.delivery_curve()
    iterative = slots["krylov_cap"] > 0
    candidates = [("full", "count1"), ("count1", "count1")]
    if iterative:
        candidates += [("full", "full"), ("count1", "full")]
    service = {}
    for candidate in candidates:
        kb = choice.hot_kb(slots, candidate[0], candidate[1], newton_visits,
                           krylov_visits)
        service[candidate] = (fetch.service(curve, kb, warps), kb)
    chosen = choice.choose(service, choice.DEFAULT,
                           fetch.beyond(curve, warps))
    return dict(
        system=row["system"], algo=row["algo"], registers=row["registers"],
        resident_warps=warps, newton_visits=newton_visits,
        krylov_visits=krylov_visits, static_slots=slots,
        candidates={"/".join(c): dict(hot_kb=round(v[1]),
                                      service_ns=round(v[0], 3))
                    for c, v in service.items()},
        unroll_newton_exits=chosen[0], unroll_krylov_exits=chosen[1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system")
    parser.add_argument("algo")
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--newton", type=float, default=None,
                        help="Newton iterations per step; default: cap")
    parser.add_argument("--krylov", type=float, default=None,
                        help="Krylov iterations per solve; default: cap")
    parser.add_argument("--records", default=str(choice.RECORDS))
    args = parser.parse_args()
    rows = static_slots.load(args.records)
    row = rows.get((args.system, args.algo))
    if row is None or row.get("status") != "ok":
        with tempfile.TemporaryDirectory() as folder:
            row = static_slots.build(args.system, args.algo, Path(folder))
    slots = choice.region_slots(row)
    newton = args.newton
    if newton is None:
        newton = slots["newton_cap"] * slots["newton_loops"]
    krylov = args.krylov if args.krylov is not None else slots["krylov_cap"]
    print(json.dumps(decide(row, args.block, newton, krylov), indent=1))


if __name__ == "__main__":
    main()
