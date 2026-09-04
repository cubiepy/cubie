# Accepted same-binary unused reservation

For chain32/Kvaerno3/LU with all eight unroll groups full, changing only
unused dynamic shared bytes increased ordinary solve time from about
21.6 to 29.0 ms. Separate matched profiles observed shared capacity
8192 versus 65536 bytes, with identical executable, exact executed work,
state/status, register allocation, local frame and four-block residency.
This establishes cache-capacity sensitivity without changing the spill
program or placing a device buffer in shared memory.

The source is frozen `reservation_probe.py` SHA
`e31119f8cacef5032b0ef185036169ccbc0dbc6e384ecce54b8f395117f073e6`.
The ordinary cohort is
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/stage_base_reservation_ordinary_e1`.
It contains two blocks of six mirrored samples per
baseline/reserved/baseline-repeat slot: 36 measurements, all at least
20 ms, plus six exact original warm NPZs. Median CUDA-event times are
21.622096, 29.008528 and 21.648864 ms respectively. Duration is 1.6,
with 262144 runs, grid `(4096,1,1)`, block `(1,64,1)` and 18.285714
full-occupancy waves. Cleanup restored the original launch and closed
the Solver without errors. Ordinary runs did not directly measure
actual shared configuration.

The two saved profiles are `profile_a_reservation_baseline_e1` and
`profile_a_reservation_reserved_e1` under the same raw root. Their
successfully reimported reports were joined to fixed NCU/Python/script
arguments, exact source snapshots, frozen runtime paths, original
cohort, raw one-solve arrays and a fresh disassembly of every source PC.
Profile timings do not enter the ordinary comparison.

| Retained quantity | Baseline | Unused reservation |
|---|---:|---:|
| Dynamic shared bytes/block | 4 | 8448 |
| Allocated bytes/block, including driver reservation | 1152 | 9472 |
| Actual shared configuration, bytes/SM | 8192 | 65536 |
| Registers/thread; local frame bytes/thread | 255; 688 | 255; 688 |
| Resident blocks/SM | 4 | 4 |
| Exact executed warp instructions | 5,107,818,264 | 5,107,818,264 |
| Local load sectors | 1,943,004,895 | 1,943,004,895 |
| Local load L1 lookup-miss sectors | 495,388,271 | 1,418,095,152 |
| Local store sectors | 711,854,055 | 711,885,488 |
| Local store L1 lookup-miss sectors | 371,961,904 | 636,017,292 |
| L2 read sectors | 497,376,257 | 1,420,324,499 |

Both cubins have SHA
`5271c0bee09b7a4db4871b78e7b2c45e9af3964169275447813d8ead648aceee`.
The complete 23936-instruction native audit finds no shared accesses.
Source thread-instruction and predicated-thread totals also match
exactly. Hardware instruction totals exceed source totals by 51 in both
profiles; that discrepancy remains unattributed. Cache counters are
observations of traffic, so identical dynamic instruction counts do not
require identical store-sector counts.

The 2.86-fold increase in local load L1 misses, unchanged local load
sector count, and increased L2 reads support pressure on the unified
L1/texture/shared pool. The [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#unified-shared-memory-l1-texture-cache)
documents the combined 128 KiB pool; its nominal complement changes
from 120 to 64 KiB for these observed shared capacities. This does not
measure usable cache capacity, associativity, replacement, or a miss
latency. ICC lookup-miss cycles fell (116,937,324 to 48,803,053), while
GCC instruction misses were 32,193 and 35,112 requests; these units are
kept distinct and do not supply a competing timing coefficient.

Exact receipts are `verification/reservation_saved_profiles_e4`, with
audit SHA `24952440a28273c84015af89b4a2ad9a45ed9ef5f6a3b1d741be15d87d5123d0`,
and independent
`verification/reservation_saved_profile_independent_final_20260905/receipt.json`.
The independent review reopened ordinary warm arrays and artifacts,
checked 36 measurements, matched both full native inventories, and
rejected 34 altered command/workload/treatment/native receipts.

The observation applies to this family, solver, policy and binary.
Actual configuration is established for the saved profile launches,
not retroactively for every ordinary launch.

## Intermediate capacities and repeated-capacity control

The completed follow-up retains the same original workload and cubin.
Its ordinary bank is `stage_base_capacity_ordinary_e1`: 140 solves,
including 84 mirrored measurements and 14 exact warm snapshots. Every
measurement reaches 20 ms, and cleanup restores the original launch.
The separate six `profile_capacity_<arm>_e1` reports passed the complete
saved-report audit in
`verification/reservation_capacity_saved_profiles_e1/receipt.json`.
They retain four resident blocks per SM, 255 registers per thread and
the 688-byte local frame. The derivation of each minimum byte request
is in RESERVATION_CAPACITY_PROTOCOL.md.

| Arm | Dynamic bytes/block | Profile shared capacity | Ordinary median ms | Local-load L1 miss sectors | L2 read sectors |
|---|---:|---:|---:|---:|---:|
| baseline | 4 | 8 KiB | 21.636929 | 493,809,615 | 497,916,637 |
| capacity16 | 1025 | 16 KiB | 22.537008 | 564,083,172 | 563,869,440 |
| capacity32 | 3073 | 32 KiB | 25.091647 | 813,224,395 | 814,261,856 |
| capacity64 | 7169 | 64 KiB | 29.077200 | 1,418,424,626 | 1,419,924,742 |
| original64 | 8448 | 64 KiB | 29.058960 | 1,418,774,661 | 1,420,293,623 |
| capacity100 | 15361 | 100 KiB | 31.867424 | 1,860,539,696 | 1,862,050,316 |
| baseline_repeat | 4 | Unprofiled repeated slot | 21.665904 | — | — |

All six profiles have exactly 5,107,818,264 source warp instructions,
163,214,649,187 thread instructions, 162,558,476,791 predicated-on
thread instructions and 1,943,004,895 local-load sectors. The hardware
warp-instruction residual remains 51 in each arm. Thus the increase in
local-load misses does not arise from additional executed local loads.
The two different reservations reaching 64 KiB produce similar ordinary
medians and profile traffic. This supports capacity, rather than the
unused reservation's byte count alone, as the relevant control variable
for this contrast. It does not identify a replacement policy or latency.

L2 miss sectors remain between 1,128,566 and 1,142,059 across the six
profiles. ICC lookup-miss cycles decrease from 116,747,890 to 29,093,092
between the endpoint arms, while GCC instruction-miss requests change
from 31,569 to 33,510. These observations retain their distinct units;
they are not converted into independent additive timing penalties.

The immutable e1 measurement source is SHA256
`5b66e9bfaf44c30db10c72ee3b539e40099cac5d5c49170c11a159316b96dddf`.
Its saved-bank reader omitted direct original-workload joins. The
independent audit above supplies those joins for the actual data,
including original duration, run count, manifest, construction, generated
source, input arrays, native artifacts and geometry. The failed reader
mutation and its correction remain separately recorded. Repaired source
SHA256 `7b692fc56d82dcfcc7ed16cb7f8d3fd73529c9dc2585347c25d333c752c0853b`
passed independent CPU review in
`verification/reservation_capacity_repaired_independent_20260905`.
This repair changes admission of saved evidence, not the original
measured data or its source snapshot.

No fitted slowdown coefficient or general buffer-placement default
follows from this curve. Its model use is to test predicted traffic
under hardware-sized cache partitions while holding the executed spill
program and numerical workload fixed.
