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

## Global 33-load administration control

The unchanged bd6172f8 controller produced
`latency_l1_quarter33_compile_e1` and
`latency_l1_quarter33_ordinary_e1`. The paired captures are
`profile_latency_l1_quarter33_n_e1` and
`profile_latency_l1_quarter33_2n_e1`. Independent review receipt
`verification/latency_l1_quarter33_profile_independent_20260905/receipt.json`
has SHA256
`987d8e580a36c6932f1e68b0f40faf9323a0a86646b9b6024e9d8a9da03e90d0`.
It reimports both saved reports, checks all 112 native PCs and validates
all 26 ordinary raw arrays. The separate profile reader is
`verification/latency33_control_adapter_v2_20260905/profile_audit.py`,
SHA256 `49eef97bc1fee05a4ed82faa18a29f7458b9b70ce1a1240b12baac9b39587885`.

The cubin is
`96a474ae3a1b07e145f2974cf6e94bfd6f8d37e430492d168636ddde82bdae4f`.
It retains the same 32 KiB ring, uint64 pointer form, 26 registers, one
active lane/CTA and final uniform CTA barrier. Both profiles report
8192 shared bytes/SM, one resident block and two waves. The body contains
33 dependent LDG and seven administrative instructions. Both global
body lengths use GPR R0 as the counter; the shorter version has a direct
conditional backedge, while the 257-load version uses a terminal CALL
and backedge. Their native administrative sequences are not identical.

Two calibration launches precede 24 measurements. The retained N is
65539 and 2N is 131078; the shorter initial calibration at 32769 remains
in the bank. Every recorded interval, including both calibrations,
equals `1230 * repeats + 23` cycles. All twelve paired minimum and median
increments therefore equal `1230/33 = 37.27272727272727` cycles/load.
The shortest accepted measurement is 30.192132 ms at its qualified clock
snapshots; ordinary event medians are 60.712320 and 121.106434 ms.

| Counter quantity | N | 2N |
|---|---:|---:|
| Timed global loads | 242,232,144 | 484,464,288 |
| Total global loads and L1 global-load sectors | 242,346,944 | 484,579,088 |
| Global-load L1 lookup-miss sectors | 57,456 | 57,456 |
| Timed lookup-hit lower bound | 99.9762806046% | 99.9881403023% |
| Aggregate L2 read sectors | 571,815 | 1,135,895 |
| Aggregate L2 read-miss sectors | 0 | 4 |
| DRAM read bytes | 0 | 5,376 |

The lower bounds again assign every whole-launch global-load miss to
the timed loads. L2 and DRAM totals are not assigned to individual PCs.
Hardware/source warp-instruction residuals are 7,455,056 and 14,795,424;
their equality to YIELD visits remains a numerical observation without
a counter-semantics correction.

Both body lengths have L1-lookup-hit-dominated timed loads, yet their
measured chain intervals per load are not invariant to body length.
Neither observation isolates intrinsic LDG latency. Subtracting a common
overhead does not isolate that latency either, because the exact native
administrative sequences and their schedules differ. Both complete
observations remain usable as explicitly qualified chain-service
scenarios; neither supplies a fitted solver penalty.

## Global `.cg` L2-hit control

The `.cg` control keeps the 257-load body's native instructions, register
dataflow and geometry. Replacing its 258 `STRONG.GPU` modifiers with
`STRONG.SM` makes its complete native instruction text identical to the
accepted `.ca` kernel: 257 timed loads and the priming load change cache
policy; the start-offset load is unchanged. The new cubin is
`0a6238443d0163b8d1fcbee42b116eed5ca1e2673d398dd2d2ba9f79880255c5`.
The controller remains bd6172f8. The ring is the same randomized 32 KiB
window of uint64 pointers, with one active lane per 1,024-thread CTA.

The [PTX cache-operator specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)
describes `.cg` as caching at L2 and below and bypassing L1. Cache
operators are performance hints. The actual counted path is therefore
qualified with additional L1TEX-source L2 read, hit and miss metrics,
whose installed AD104 descriptions and sector units are retained.

The ordinary bank is `latency_l1_quarter_cg_ordinary_e1`, with one
calibration and 24 mirrored measurements at N=8193 and 2N=16386. All
recorded endpoints, counts, clocks, SMIDs and source/native joins pass
the raw audit. The shortest accepted lane interval is 182.252080 ms.
The median of the twelve paired median-cycle increments is
259.32650582897713 cycles/load; individual pair values range from
257.9417 to 259.7010. These are measured chain increments with the native
administration retained, not a constant intrinsic load latency.

Both matched reports, `profile_latency_l1_quarter_cg_n_e1` and
`profile_latency_l1_quarter_cg_2n_e1`, retain all 336 native PCs. They
report 26 registers/thread, zero local/static/dynamic shared storage,
8192 actual shared bytes/SM, one resident block and two full waves.

| Whole-launch quantity | N | 2N |
|---|---:|---:|
| Timed global loads | 235,827,312 | 471,654,624 |
| Total global loads | 235,942,112 | 471,769,424 |
| L1 global-load sectors | 235,942,112 | 471,769,424 |
| L1 global-load lookup-miss sectors | 235,942,112 | 471,769,424 |
| L1TEX-source L2 read sectors | 235,942,112 | 471,769,424 |
| L1TEX-source L2 hit sectors | 235,942,112 | 471,769,424 |
| L1TEX-source L2 miss sectors | 0 | 0 |
| Aggregate L2 read sectors | 239,853,685 | 479,495,931 |
| Aggregate L2 read-miss sectors | 1 | 2 |
| DRAM read bytes | 0 | 27,904 |

The L1TEX-source reads equal the counted global loads, and all those
source-qualified reads hit L2. Both source conservation residuals are
zero. Together with the L1 lookup-miss counts, these observations
corroborate a bypassed-L1, L2-hit path for the counted global data.
The aggregate L2 and DRAM counters have wider scope; their remaining
traffic and misses are not attributed to individual load PCs.

Hardware warp-instruction totals are 245,164,976 and 489,250,832, with
hardware/source residuals of 1,032,304 and 1,949,920. They again equal
counted YIELD visits numerically, without an established semantic cause
or correction. Profiling event durations remain separate from the
ordinary measurements.

Grouping the ordinary lane intervals by their observed virtual SMID
reveals repeatable variation among the observed IDs. For each of 56 IDs,
the diagnostic takes each launch's median across that ID's actual CTA
samples, then forms the twelve paired N/2N increments. The per-ID medians
range from 231.1762575150753 to 286.59454355312334 cycles/load. The median
within-ID range across twelve pairs is 1.4600077365084871 cycles/load.
Each raw entry/exit ID matches, but virtual IDs do not establish physical
SM locations, GPC membership, L2 slices or their distances. No fitted
per-SMID latency table is used by the solver model.

The raw ordinary audit is
`verification/latency_l1_quarter_cg_ordinary_audit_20260905/receipt.json`.
The full saved-report audits are
`verification/latency_l1_quarter_cg_profile_audit_n_e1/analysis.json` and
`verification/latency_l1_quarter_cg_profile_audit_2n_e1/analysis.json`.
Their reader has SHA256
`f7accf1b480bc3d16ee4514de243050a41da1d2ccf28f1137ee7a160b7a8c804`.
The SMID diagnostic is
`verification/latency_cg_smid_diagnostic_v2_20260905/analysis.json`,
SHA256 `b44cb05141599267770b2683460ee037218aa1d2fe74a785a87c1f995e741d45`.
The independent audit reimports both saved reports and recomputes all
25 ordinary arrays and all 56-by-12 SMID pairs. Its receipt is
`verification/latency_cg_actual_independent_20260905/receipt.json`, SHA256
`279a7137548b0b52bf4b3331314d6c050e54f0e00c9930898e80d0f05188e039`.
The admitted workload remains a serialized, one-active-lane
uint64 chain; full-warp solver service is a separate condition.
