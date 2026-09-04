# Instruction-supply counter evidence

Recorded 2026-09-04. An authorized elevated profiling process completed
all 15 cases with exit zero. This CPU audit uses its counter export
and matches its compiled artifacts to the ordinary timing experiment.
No GPU work or source-code change was made in this audit.

## Source receipts and matching

- [Counter receipt JSON](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_elevated_counter_receipt_20260904.json): raw selected counters with units, normalizations, source row numbers, and SHA256s.
- [Raw metrics CSV](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_contrast_elevated_20260904_metrics.csv): one header, one unit row, then 15 data rows.
- [NCU report](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_contrast_elevated_20260904.ncu-rep) and [completed status](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/elevated_profile_contrast_20260904_status.json).
- [Profile artifact results](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_contrast_elevated_20260904_artifacts/results.jsonl).
- [Ordinary results](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_contrast_20260904/results.jsonl) and [ordinary receipt](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/icache_contrast_ordinary_receipt_20260904.json).
- [Executed elevated helper](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/profile_elevated_contrast_20260904.ps1).

The case order is 64/120/128/144/192 KiB requested FFMA bodies, each
at 8/16/32 theoretical resident warps. IDs zero through 14 match the
artifact result order. Each case has 128 threads/block, all active,
4,096 runtime repeats, and two waves. Block/grid sizes, registers,
shared allocation and operation counts were checked against the
matching artifacts. Cubin files and SASS files are byte-identical to
their ordinary counterparts, not merely equal in recorded metadata.

For every case, the dynamic executed-instruction counter equals
`launched_warps * (hot_instructions * 4096 + 43)`. The constant
43 is the observed residual outside the repeated stream, not a fitted
timing coefficient. This exact correspondence additionally checks
the case pairing and repeated workload.

The helper uses three kernel replay passes, no profiler clock change
and no cache flushing between passes. There is one profile per case;
ratios combine replayed counter collections and are not simultaneous
measurements. Profiled event times and the profile harness's reported
operations/second are excluded from ordinary performance samples.
The initial unprivileged `ERR_NVGPUCTRPERM` attempt remains a failed
historical dataset. The successful dataset came from an authorized
elevated process without changing performance-counter permissions.

## Geometry and actual shared configuration

`launch__shared_mem_config_size` is 102.400000 Kbyte in every case:
102,400 bytes, or 100 KiB. The export's decimal Kbyte scale is checked
against its exact dynamic allocations. Thus the shared configuration
does not change across this contrast.

| Theoretical warps/SM | Blocks/SM | Grid blocks | Dynamic bytes/block | Driver reservation | Allocated bytes/block |
|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 224 | 33,025 | 1,024 | 34,176 |
| 16 | 4 | 448 | 19,457 | 1,024 | 20,608 |
| 32 | 8 | 896 | 10,241 | 1,024 | 11,392 |

The allocated values include rounding. For example, three blocks of
34,176 bytes exceed 102,400 bytes, while two fit. The corresponding
limits are four blocks of 20,608 and eight blocks of 11,392. NCU's
shared-memory occupancy limits agree with the driver query.
All cases allocate 24 registers/thread, despite used counts of 21
or 23. The resource allocation therefore does not change at the
128-to-144 KiB transition.

At eight warps the achieved active-warp counter is 8.0027 at 128 KiB
and 7.9981 at 144 KiB. These are profiler averages, including minor
measurement variation, not exact instantaneous counts. There is no
observed loss of achieved occupancy that could explain the cliff.

## Units and normalization

Let `W = verified_hot_FFMAs * 4096 * launched_warps`. It counts warp
FFMA instructions, not scalar results; all 32 lanes are active.

- GCC miss fraction is instruction lookup misses divided by
  instruction requests. Both raw counters have unit `request`.
- GCC misses per 1,000 warp FFMAs divide by `W`, allowing comparisons
  across the changed body lengths and grid sizes.
- ICC counters are exported with unit `cycle`. Their miss/total ratio
  is retained as a ratio of those counters, not renamed a count of
  GCC requests. Pending-hit and miss-tag-miss totals are not directly
  interchangeable with GCC request totals.
- No-instruction and branch-resolving sums accumulate warp observations
  across cycles. Dividing by `W` gives warp-cycle observations per
  warp FFMA. These can exceed one; they are not percentages.
- Eligible warps per SMSP elapsed cycle are
  `smsp__warps_eligible.sum / (4 * sm__cycles_elapsed.sum)`. Four
  schedulers per SM is a queried device attribute. The denominator
  uses elapsed cycles, not active cycles.
- Achieved active warps/SM is the direct profiler metric
  `sm__warps_active.avg.per_cycle_active`; it has a different
  denominator from the eligibility normalization.

All ICC hit-plus-miss and GCC instruction hit-plus-miss identities
hold exactly. ICC pending-hit plus miss-tag-miss also equals the
reported ICC miss total in each row.

NVIDIA's [Profiling Guide, §2.3.4](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#units)
describes ICC at the TPC level and GCC as the per-GPC L1.5 cache for
instructions and constant data, with misses fetched from L2. It also
describes immediate constant operands as using a separate SMSP cache.
The measured GCC counters specifically filter instruction requests.
This distinguishes their traffic from the fixed constant addend
used by every FFMA. The metric prefix `sm__` does not establish
physical per-SM instruction-cache ownership or capacity.

## Eight-warp cliff versus the 16-warp plateau

| Body KiB | Warps | Ordinary TFMA/s | GCC instruction miss % | GCC misses / 1,000 warp FFMAs | ICC miss/total % | No-instruction / warp FFMA | Eligible warps / SMSP elapsed cycle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8 | 18.664 | 0.000475 | 0.0000076 | 0.621 | 0.571 | 1.383 |
| 64 | 16 | 18.836 | 0.000349 | 0.0000039 | 0.612 | 0.396 | 3.199 |
| 128 | 8 | 16.389 | 3.234 | 0.254 | 20.972 | 0.545 | 1.458 |
| 128 | 16 | 12.542 | 0.973 | 0.148 | 38.478 | 3.869 | 1.065 |
| 144 | 8 | 6.489 | 50.079 | 3.933 | 50.622 | 4.402 | 0.437 |
| 144 | 16 | 12.794 | 34.143 | 1.880 | 52.988 | 3.481 | 1.515 |
| 192 | 8 | 6.493 | 50.061 | 3.927 | 50.638 | 4.392 | 0.437 |
| 192 | 16 | 12.799 | 50.157 | 1.968 | 60.389 | 3.368 | 1.604 |

Ordinary TFMA/s is the maximum of five unprofiled samples. All counter
columns come from the separate elevated experiment. These columns
must not be used to synthesize a calibrated nanosecond penalty.

At eight warps, moving from 131,184 to 147,568 actual hot bytes
increases GCC instruction misses per work by approximately 15.5
times. No-instruction observations per work increase about eight
times, while eligible warps fall below one per scheduler on the
elapsed-cycle average. Ordinary throughput falls to 39.6% of its
128 KiB value. The instruction-supply counters support instruction
starvation associated with increased GCC instruction misses as the
mechanism accompanying this cliff.

The forward `CALL.REL.NOINC` exit and backedge `BRA` are present at
every footprint, including 64 KiB. Branch-resolution observations per
1,000 warp FFMAs decrease from 0.3665 to 0.3257 at the eight-warp
128-to-144 transition. Shared configuration, allocated registers,
and achieved residency remain effectively fixed. These controls do
not support a new branch form or resource-occupancy loss as the
explanation for this particular transition.

At 16 warps the same footprint increase also raises GCC instruction
misses, but no-instruction observations per work decrease and
eligibility improves. At 144 KiB, doubling from eight to 16 warps
roughly doubles work and ordinary throughput, while total GCC
instruction misses change from 133,017,664 to 127,187,996. Misses per
work therefore fall. This is evidence that residency changes the
instruction request/reuse behavior as well as the supply of eligible
warps. It cannot be explained by assigning every excess code byte
one fixed cost independent of residency.

The 64 KiB baseline has only 114 and 117 GCC instruction misses in
the whole profiled eight- and 16-warp grids. Executed warp instructions
per SM elapsed cycle are 3.938 and 3.962, approaching the four-warp
FP32 issue ceiling derived from
[128 scalar results/SM/cycle](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions)
and 32 lanes/warp. At 128 KiB, increasing residency to 16 already
increases ICC pending-miss counter units per 1,000 warp FFMAs from
16.742 to 53.775. Thus the pre-cliff regime is itself affected by
the number and relative positions of the instruction streams.

The 32-warp rows are retained in the JSON. They show achieved active
warps around 28–29 and a different mixture of GCC reuse and
no-instruction observations. Their presence further rules out using
one body-size-only timing curve as the hardware model.

## What this establishes and what remains unknown

The experiment establishes a repeatable, residency-dependent
instruction-supply transition in these exact eight-chain FFMA loops.
It supports retaining hot instruction footprint and resident
instruction-stream behavior as separate model inputs.

It does not measure a physical 128 KiB cache or locate that capacity
at the SM, TPC or GPC level. The transition is bracketed by actual hot
ranges of 131,184 and 147,568 bytes, with relative start address
`0x1b0`. Replacement, associativity, instruction-fetch grouping,
inter-warp phase behavior and interference can change the usable
working set. The documented cache hierarchy is provenance for the
counters, not a capacity measurement. Global memory traffic or L2
misses were not measured by this counter set.

A bounded follow-on can bisect 128/132/136/140/144 KiB first at eight
warps, retaining the same paired ordinary/counter procedure. The
minimum confirming residency contrast is 16 warps at the identified
transition's two neighbors. Fine `--ffmas` increments must be multiples
of eight, giving 128 bytes of additional FFMA instructions per step;
always inspect actual hot ranges. A transition point remains a
property of that controlled stream, not automatically a byte-capacity
constant. Measuring physical sharing requires distinct warmed code
regions and controlled interference with execution-location evidence.
