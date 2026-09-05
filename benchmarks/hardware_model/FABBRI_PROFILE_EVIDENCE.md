# Fabbri Radau: more executed work can accompany faster solves

The retained Fabbri/LU winner policies execute 74.712% more warp
instructions for Radau IIA 3 and 119.016% more for Radau IIA 5 than full
unrolling. Their matched ordinary block medians are lower. These cases
refute an explanation based only on reducing total instruction work.
They also cannot be ranked by instruction-cache miss counts alone:
Radau5's winner has more GCC instruction lookup misses but far fewer
sampled no-instruction stalls. The memory and instruction observations
must remain separate, with their actual units and workload differences.

## Provenance and audit

Paths are relative to
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
The four saved profiles are `profile_fabbri_radau3_full_e1`,
`profile_fabbri_radau3_winner_e1`, `profile_fabbri_radau5_full_e1`, and
`profile_fabbri_radau5_winner_e1`. Their source exports are
`source_counters.csv`; the raw Nsight reports, metrics CSVs, reference
cubins and policy-specific iteration-label arrays are retained beside
them. The original ordinary bank is `fabbri_radau_interactions_e1`.

The frozen core analyzer's saved outputs are under
`verification/fabbri_profiles_e1_analysis/<profile-name>/`. Its SHA256 is
`ecd46638fb63c83540ead9a586f856d04856500bdeb83aba11f543a4e11731b3`.
These outputs include the saved-report reimport identity check and
per-PC agreement with the disassembled reference cubin, including
predicates and operands after the documented display normalization.

`verification/fabbri_profiles_independent_20260905/audit.py` and
`receipt.json` independently verify retained file hashes, every raw
source-PC exact/sample count, opcode sums, positive address footprints,
metrics values and their CSV unit row, counter NPZ arrays, exact
instrumented/reference state/status equality, and every cited ordinary
timing against its original JSONL line. No CUDA compilation or GPU work
is performed by this audit. Exact source warp totals equal
`smsp__inst_executed.sum` in all four profiles.

The package source hash is
`4899b5cb04523177ed3cd3f1aef566591829ed026e064b951fdfbf629cfcef6a`;
toolchain fingerprint is
`68e2731a23c73a42669ec8e16a8cc22fd42f6b09432ea0a54854c2a15ea9ce24`.
All use MLIR, `anchor_dfs`, `liveness_auto`, LTO and
`nsz/contract/arcp/afn/ftz`, with line information disabled. They use
131,072 trajectories, duration 0.2, 64-thread blocks, grid `(2048,1,1)`,
block dimensions `(1,64,1)`, and 56 SM89 SMs. Each has 255 registers per
thread and four register-limited resident blocks per SM: 9.142857
occupancy waves. Dynamic shared memory is four bytes per block; the
actual profiled shared configuration is `8.192000 Kbyte` = 8,192 bytes
in every case. This is no shared-capacity contrast.

The policies are:

- R3 full: `u11111111`; R3 winner: `u11100000`.
- R5 full: `u11111111`; R5 winner: `u00100000`.

The bit order is stage, step element, accumulator, solver element,
norms, other small loops, Newton iterations, Krylov iterations. In this
harness `0` means the explicit `(True, 1)` directive, not `False` or
libnvvm choice. The R3 winner retains full stage/step-element/accumulator
expansion; the R5 winner retains only full accumulator expansion.
These are multi-group contrasts with LU as the inner solver.

## Ordinary block timings remain the timing evidence

The winner samples occur in block 0. This table uses only that block's
two explicit baseline roles and its six winner samples. The analyzer
also retains repeated full-binary samples from other blocks and nominal
count directives that produced the same cubin; those are not silently
pooled into the block-0 comparisons below.

| Case/role | Samples | Minimum ms | Median ms | Maximum ms |
|---|---:|---:|---:|---:|
| R3 full | 6 | 107.4325 | 107.58105 | 108.5315 |
| R3 duplicate full | 6 | 107.4277 | 107.6734 | 108.6275 |
| R3 winner | 6 | 83.0294 | 83.43075 | 84.1062 |
| R5 full | 6 | 219.2531 | 219.6493 | 223.9810 |
| R5 duplicate full | 6 | 220.1099 | 221.0520 | 223.2631 |
| R5 winner | 6 | 178.0639 | 190.66410 | 245.4301 |

Winner/baseline median ratios are 0.775515 and 0.774850 for R3;
0.868039 and 0.862531 for R5, against the two baseline roles separately.
These are descriptive same-block ratios, not a new paired estimator.
All samples exceed 20 ms. R5's winner has a wide range and includes a
sample slower than either baseline's maximum; its lower median is not
an all-samples win. Profile durations and the instrumented counter
solver's printed timings do not replace these ordinary samples.

## Actual work and native address footprint

| Observable | R3 full | R3 winner | R5 full | R5 winner |
|---|---:|---:|---:|---:|
| Whole instruction addresses | 77,480 | 9,512 | 169,656 | 23,112 |
| Whole encoded bytes | 1,239,680 | 152,192 | 2,714,496 | 369,792 |
| Positive-execution addresses | 20,723 | 9,480 | 87,461 | 23,078 |
| Positive-execution encoded bytes | 331,568 | 151,680 | 1,399,376 | 369,248 |
| Exact warp instructions | 4,531,790,414 | 7,917,583,872 | 4,685,892,222 | 10,262,843,392 |
| Exact thread instructions | 143,960,900,446 | 251,499,220,003 | 149,948,551,104 | 328,410,988,544 |
| Predicated-on thread instructions | 143,660,426,670 | 240,760,137,755 | 149,759,365,166 | 309,436,333,812 |
| Local frame bytes/thread | 3,888 | 3,104 | 9,392 | 8,296 |

The positive-execution footprint is the distinct-address union over the
whole grid and solve. It is not a temporal working set or an SM-local
reuse distance. Both winner unions still exceed 128 KiB. R3 full's
1,239,680-byte body includes 56,757 addresses with no exact execution;
R5 full has 82,195 such addresses. Whole-body bytes therefore cannot be
substituted for hot-loop bytes. Positive per-address warp counts range
from 4,096–358,541 and 4,096–60,506,880 for R3 full/winner, and
3–155,648 and 4,096–59,351,040 for R5. The complete frequency buckets
remain in the receipt without a fitted hot/cold threshold.

The following opcode families combine modifiers and retain exact
warp-level executions; the full modifier-specific vectors remain in
the receipt. They are counts, not weighted cycle estimates.

| Opcode family | R3 full | R3 winner | R5 full | R5 winner |
|---|---:|---:|---:|---:|
| FFMA | 1,251,081,532 | 1,249,120,896 | 1,433,938,214 | 1,451,159,552 |
| FMUL | 1,394,710,551 | 1,420,088,290 | 1,255,239,355 | 1,241,583,616 |
| FADD | 147,228,976 | 206,012,762 | 131,512,391 | 214,011,904 |
| IADD3 | 2,019,188 | 1,089,499,524 | 1,196,090 | 1,625,399,296 |
| IMAD | 607,810 | 275,676,982 | 585,728 | 468,668,416 |
| MOV | 83,793,603 | 806,152,956 | 34,665,338 | 1,381,335,040 |
| ISETP | 1,937,715 | 307,348,582 | 770,048 | 408,989,696 |
| BRA | 3,709,940 | 185,669,422 | 1,610,164 | 329,744,384 |
| LDL | 685,128,174 | 682,891,364 | 900,939,054 | 1,156,141,056 |
| STL | 453,648,040 | 445,326,424 | 522,632,524 | 585,711,616 |

The added work is concentrated in integer/addressing, moves and control
operations, with arithmetic and memory changes also present. There is
direct native loop evidence: R3 winner offsets `0x22a00–0x22af0` contain
stride-35 `IMAD`, an `LDL`, index increments/decrements, moves and a
backward conditional branch; individual instructions there execute
60,506,880 times. R5 has the corresponding observed pattern at
`0x4e880–0x4e970`, with 59,351,040 executions. These are verifiable native
addressing/control patterns; absent line information, this audit does
not assign every added integer instruction to a particular source loop.

## Iteration labels show different numerical workloads

Each NPZ has FP32 state shape `(2,35,131072)`, counters of shape
`(2,4,131072)` in int32, and exact int64 per-run totals. Every instrumented
state/status matches its own uninstrumented reference bit for bit, is
finite and has successful status. Cross-policy states and counter arrays
are **not** identical.

| Per-lane mean label | R3 full | R3 winner | R5 full | R5 winner |
|---|---:|---:|---:|---:|
| Newton iterations | 101.732177734375 | 102.2349395751953125 | 47.03772735595703125 | 46 |
| Linear-solver iterations | 101.732177734375 | 102.2349395751953125 | 84.03772735595703125 | 83 |
| Attempted steps | 86.01670074462890625 | 86.01678466796875 | 37 | 37 |
| Rejected steps | 0 | 0 | 0 | 0 |

R3 winner adds 65,898 Newton/linear iterations in aggregate (+0.4942%),
but 98,591 lanes change with differences ranging from −23 to +23.
Attempt counts differ on 233 lanes, with aggregate difference +11.
R5 winner removes 136,017 Newton/linear iterations: 130,710 lanes change,
with differences from −25 to 0. Every R5 winner lane records 46 Newton,
83 linear iterations and 37 attempted steps. These totals do not
establish identical warp loop execution; lane activity, termination
predicates and explicit loop administration remain separate quantities.

Across policies, the largest retained-state absolute differences are
`8.0108642578125e-05` for R3 and `7.62939453125e-05` for R5. There are
3,947,490 and 4,055,654 differing FP32 bit patterns respectively in the
two saved state rows. These are observations, not a newly selected
acceptance tolerance. Counter labels are post-execution validation data,
not pre-compile predictors or a substitute for exact warp instruction
work. The mean iteration differences cannot explain a 74.7% or 119.0%
instruction increase by simply asserting equal dynamic workloads.

## Memory service and instruction delivery remain distinct

| Observable and exported unit | R3 full | R3 winner | R5 full | R5 winner |
|---|---:|---:|---:|---:|
| L1 local-load sectors | 2,728,006,643 | 3,075,309,570 | 3,605,017,784 | 4,685,004,800 |
| L1 local-load lookup-miss sectors | 2,129,297,105 | 1,952,529,834 | 2,976,066,845 | 3,084,284,568 |
| L1 local-store sectors | 1,804,928,741 | 2,125,820,796 | 2,090,145,399 | 2,386,029,444 |
| L1 local-store lookup-miss sectors | 1,684,986,537 | 1,609,378,064 | 2,054,221,095 | 1,951,631,948 |
| L2 read sectors | 2,550,161,599 | 2,081,274,928 | 4,042,235,041 | 4,072,093,288 |
| L2 write sectors | 1,806,063,592 | 2,127,940,538 | 2,092,385,803 | 2,387,874,339 |
| L2 lookup-miss sectors | 2,903,493 | 3,532,706 | 584,728,413 | 169,233,008 |
| GCC instruction lookup-miss requests | 71,915,303 | 27,601,301 | 154,280,679 | 167,920,058 |
| `sm__icc_requests_lookup_miss.sum`, cycle | 153,840,128 | 68,288,245 | 284,569,274 | 220,009,523 |
| Source sampled no-instruction stalls | 2,581,087 | 372,243 | 10,176,325 | 2,451,207 |
| Source sampled no-instruction, not issued | 2,486,619 | 341,734 | 10,018,570 | 2,339,964 |
| Source sampled long-scoreboard stalls | 11,488,696 | 9,059,659 | 26,707,432 | 18,968,994 |

The local rows are the corresponding
`l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}[...].sum` metrics; L2 rows
are `lts__t_sectors_{op_read,op_write,lookup_miss}.sum`. L2 read/write
counts include the traffic presented to that level and must not be
renamed local spill traffic. L2 misses are not directly measured DRAM
byte counts. The ICC export labels its values `cycle`; they are not
silently relabeled GCC requests or multiplied by a latency.

Both smaller local frames accompany **more** L1 local-load and store
sectors. R3's executed LDL/STL instruction counts even decrease while
its sectors increase. A local frame, native memory-instruction count,
memory request and sector count therefore cannot substitute for each
other. No frame-times-batch or frame-times-residency cache-fit calculation
is justified by these observations.

For R3, GCC instruction misses fall 61.620%, sampled no-instruction
stalls fall 85.578%, and local-load misses/L2 reads decrease. These
changes are compatible with improved instruction and data service, but
the simultaneous changes do not isolate a cycle cost for either.
For R5, GCC instruction misses **increase 8.841%**, while sampled
no-instruction stalls fall 75.913% and the ICC metric falls 22.687%.
L2 lookup-miss sectors fall 71.058% despite slightly more L2 reads and
more local loads/stores. The distinct counters support improved service
or use of service, not an additive timing prediction from miss counts or a
claim that the winner's entire executed body fits in cache.

Sampled stalls are neither exact instruction executions nor elapsed
cycles. R5 full has 38,670,981 aggregate PC samples but 38,587,596 summed
source samples: a residual of 83,385 remains unresolved. The other three
sample residuals are zero. No residual is allocated to a stall category,
and no sample count is converted into cycles or used to fit a penalty.
