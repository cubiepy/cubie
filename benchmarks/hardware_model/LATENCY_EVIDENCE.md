# Direct dependent-load chain evidence

The RTX 4070 SUPER measurement establishes a **24-cycle lower-envelope
interval per dependent 32-bit shared load** for the recorded serialized
chain. This is a usable hardware-service scenario for that workload and
launch condition. It is not an isolated instruction latency, an SM
throughput measurement, or a generic LDS default for solver kernels.

Raw paths below are relative to
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
The ordinary bank is `latency_shared8_ordinary_e1`; matched reports are
`profile_latency_shared8_n_e1` and `profile_latency_shared8_2n_e1`.
The independent review is
`verification/latency_shared8_profile_independent_20260905/receipt.json`,
SHA256 `153579141f9efd4415cf3648521474920ed81549964104784a6a5e31727c196c`.
It reimports both saved reports, checks every native PC and all ordinary
arrays, and records no native compilation or kernel launch by the audit.

## Native and physical conditions

The captured controller is
`9de6a49cd124268df63b06f6e2561c650c8803a07f9a6515f56b8c787ec6b6ac`.
Both reports and the ordinary bank use cubin
`fab351b2a44aeef8906a2dac42295fcb63f0ad05c7afc0c22a2b133c1797c2fc`.
The installed MLIR backend is 0.5.1.1, with CUDA 13.3 binary tools,
driver 610.62 and SM89. The retained compiler fingerprint and component
inventory identify the compiler independently of the binary-tool version.

Each of 112 blocks has 1,024 threads and one lane executing the timed
chain. The other lanes remain at the final synchronization, so all
32 allocated warps remain resident until that lane finishes. Both
profiles confirm one possible resident block/SM across 56 SMs and two
full occupancy waves. Registers are 20/thread and local memory is zero.
Actual configured shared capacity is **16,384 bytes/SM**, accommodating
the 8,192-byte static ring plus 1,024 driver-reserved bytes/block, with
9,216 allocated bytes/block. No carveout preference is set. Register
capacity alone permits two blocks; thread and shared limits each permit
one. These are per-profile physical observations, with the ordinary
one-block bound also enforced by its 1,024-thread block and final barrier.

The ring contains 256 randomized nodes at 32-byte spacing. Each block
uses its own shared ring and a recorded start phase. A full 256-load
priming traversal completes before the starting clock. The timed body
has 257 direct LDS instructions with an explicit register-word dependency
chain, plus decrement, comparison, terminal exit test and backedge.
There is no repeated index-to-byte address arithmetic. A final-pointer
guard precedes each clock, and the starting timestamp survives all
intervening instructions. The ending guard, loop control and scheduling
remain part of the measured interval.

## Ordinary N/2N result

One calibration launch precedes 24 mirrored measurements: 12 at N and
12 at 2N, paired in two blocks of six. N is 32,769. The nonfull-ring
remainders are one and two loads respectively, so their exact final
pointers differ. All raw counts, pointers, clock differences, retained
ring bytes, start phases, native identities and cleanup records pass.
Every measurement observes all 56 SMIDs at entry and exit; those IDs
are not used to infer a physical cache map.

| Quantity | N | 2N |
|---|---:|---:|
| Dependent loads per timed lane | 8,421,633 | 16,843,266 |
| Minimum elapsed thread cycles | 202,119,214 | 404,238,406 |
| Maximum elapsed thread cycles | 206,574,272 | 404,238,406 |
| Median ordinary event time (ms) | 151.523842 | 302.926193 |
| Measurements with a slower CTA tail | 4 of 12 | 0 of 12 |

Every measurement's minimum is exactly `24 * measured_loads + 22`
cycles, and all 12 paired minimum-cycle differences per additional load
are exactly 24. For paired median-cycle differences, eight pairs are 24
and four are 23.901318841607086, 23.735895104904237,
23.967370995625195 and 23.966312590444158 cycles/load. Their median
is 24; the full distribution retains the effect of the slower N CTAs.
The 22-cycle residual is an observed intercept
for these two native counts, not a fitted or portable endpoint constant.
No cycle cost is assigned to individual administrative instructions.
Slower N CTAs remain retained rather than discarded. The shortest
measured chain is 75.700080 ms when converted using the before/after
2670 MHz SM snapshots; those snapshots are not a continuous clock trace.
Event times include both waves and are not divided by one lane's chain
length to estimate service.

The ordinary CPU receipt is
`verification/latency_shared8_ordinary_audit_20260905/receipt.json`.
It includes the complete cycle histograms, all 12 paired differences,
and raw array hashes.

## Matched source counts and shared transactions

The saved-report audits are
`verification/latency_shared8_profile_audit_n_e2/analysis.json` and
`verification/latency_shared8_profile_audit_2n_e2/analysis.json`.
Their source SHA256 is
`c8fa19ed54b2643450fd32dc9388234711e86c51d8e3d80815f1e5fcefb88d55`.
Every one of the 336 native PCs matches the cubin and its expected warp,
thread and predicated execution counts. This includes initialization,
priming, counted loads, endpoint guards and zero invalid-path/padding
work. The final BSYNC receives 33 path arrivals per CTA because warp
zero has two disjoint masks; the final CTA barrier executes with the
32 complete warps. Path arrivals are not extra resident warps.

| Whole-launch shared quantity | N | 2N |
|---|---:|---:|
| Timed LDS instructions | 943,222,896 | 1,886,445,792 |
| Priming LDS instructions | 28,672 | 28,672 |
| Total LDS instructions | 943,251,568 | 1,886,474,464 |
| Shared-load wavefront counter | 943,251,568 | 1,886,474,464 |
| Shared-load bank-conflict counter | 0 | 0 |

The hardware shared-load count equals `112 * (257 * repeats + 256)`.
The wavefront and conflict `.sum` metrics have blank unit fields in
this NCU export. Their count meanings are bound to the installed AD104
metric descriptions, retained in each audit; they are not relabeled as
bytes or cycles. Profiling event times are excluded from the ordinary
timing evidence.

The catalog may retain this as **serialized uint32 shared-chain service,
24 cycles/load at the measured lower envelope**, with the exact body,
geometry, ring, and counter qualifications above. A solver's dependency
graph or concurrent shared traffic may have different service behavior.
The current evidence does not turn 24 into an architecture-wide latency
or throughput constant and supplies no fitted solver slowdown factor.

## Global 32 KiB `.ca` chain

The separate global bank is `latency_l1_quarter_ordinary_e1`, with
`profile_latency_l1_quarter_n_e1` and
`profile_latency_l1_quarter_2n_e1`. Its independent review is
`verification/latency_l1_quarter_profile_independent_20260905/receipt.json`,
SHA256 `a1295ae612c37afa05847ab3e56a79456cc7b9de4d869f9739d77dba4fcdb12f`.
The auditor source is
`verification/latency_l1_quarter_profile_audit_20260905.py`, SHA256
`f258eb2b5bba1e36f9e2239dbfbfe6d5ec21891f467f8dcc8494a8daf815b43f`.
Its accepted outputs are
`verification/latency_l1_quarter_profile_audit_n_e2/analysis.json` and
`verification/latency_l1_quarter_profile_audit_2n_e1/analysis.json`.
The review reimports the saved reports and independently checks all
336 native PCs, N/2N work differences, 25 raw output arrays and 12 pairs.

The controller is `bd6172f8e924583fabed2d5dd621da7824fdad46aa9dc5730eb36b0f663c76f0`;
the cubin is `1e82369627d881378ccad8f1b82e7f184a9d00651322d564399d444c52be7f36`.
This ring contains 1,024 uint64 device pointers in a 32,768-byte window,
with one node per 32-byte sector. It is shared across the grid, with
recorded CTA starting phases. The active lane in each CTA completes the
full priming ring before timing. The native body has 257 dependent
`LDG.E.64.STRONG.SM`
instructions and eight administrative instructions: two scalar pointer
copies, YIELD, ULDC, decrement, comparison, terminal exit and backedge.
The copies are exact low/high transport; no index arithmetic separates
the 256 internal load-to-load edges.

Both profiles record 26 registers/thread, 32 allocated registers/thread,
zero local/static/dynamic shared storage, 1,024 driver-reserved shared
bytes/block, and **8,192 bytes/SM actual shared configuration**. The
occupancy limits are 24 blocks, two by registers, eight by shared
allocation and one by threads/warps. The final WARPSYNC and CTA barrier
retain all 32 warps; 112 blocks across 56 SMs give two occupancy waves.
The timed chain still has only one active lane/CTA.

| Quantity | N = 32,769 | 2N = 65,538 |
|---|---:|---:|
| Timed loads/lane | 8,421,633 | 16,843,266 |
| Ordinary minimum and median cycles in every measurement | 290,333,349 | 580,666,689 |
| Largest ordinary lane interval | 290,333,350 | 580,666,690 |
| Median ordinary event time (ms) | 217.787315 | 435.284988 |
| Timed global loads/launch | 943,222,896 | 1,886,445,792 |
| Priming global loads/launch | 114,688 | 114,688 |
| Initial start-offset loads/launch | 112 | 112 |
| Total global loads and L1 global-load sectors | 943,337,696 | 1,886,560,592 |
| L1 global-load lookup-miss sectors | 57,456 | 57,456 |

All 12 paired minimum and median increments are **8,860 cycles per
257-load body**, or `8860/257 = 34.474708171206224` cycles/load. Every
measurement's minimum is `8860 * repeats + 9`; the nine-cycle residual
is an observed relation for these two counts, not an assigned endpoint
constant. The shortest recorded ordinary chain passes the 20 ms gate
at 108.739082 ms using its qualified clock snapshots. The exact pointer
remainders are 257 and 514 nodes, so N and 2N outputs differ.

The hardware global-load count, per-PC global-load count and L1 sector
count agree exactly. Assigning **all** whole-launch global-load misses
to timed loads gives conservative timed lookup-hit fractions of at
least `943165440/943222896` and `1886388336/1886445792`, respectively:
99.9939085448% and 99.9969542724%. This qualifies an L1-lookup-hit-dominated
path without assuming which misses occurred during priming. L2 read
sectors are 2,046,420 and 3,776,379; L2 read misses are 29 and one sector;
DRAM read bytes are 43,648 and 3,584. These aggregate quantities include
traffic beyond the timed pointer loads and are not assigned to their
PCs or converted into a load-latency correction.

The software/per-PC total warp-instruction counts are 973,548,352 and
1,946,132,272. Hardware totals are 977,333,168 and 1,953,587,216, leaving
residuals of 3,784,816 and 7,454,944. Those residuals numerically equal
the counted priming-plus-body YIELD visits. That equality alone does
not establish the counter-semantic cause; no correction is applied.

The usable observation is this **serialized uint64 global `.ca` chain
with L1 lookup hits dominating, 8,860 cycles per recorded body**. It
includes the explicit native administration and operand form. It does
not isolate intrinsic LDG latency or supply a generic solver load
constant. Comparing it with the shared scenario also changes pointer
width, administrative work and shared configuration, so their interval
difference is not a controlled memory-space penalty.
