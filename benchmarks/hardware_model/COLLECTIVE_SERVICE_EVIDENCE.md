# Target collective instruction evidence

The SM89 probe uses retained dependent native chains, with PTX assembly
and independent disassembly review before measurement. Its two forms are
257 predicate all-votes and 257 ballot/compare pairs. Results qualify
these complete recurring motifs, including loop administration. They do
not identify standalone instruction latencies by subtraction.

Each launch has 112 blocks of 1,024 threads. The kernel's resource limits
permit one block per SM on the 56-SM RTX 4070 SUPER: two full occupancy
waves. One or 32 full warps per block execute the timed chain; other warps
wait at the final block barrier. Both endpoints of every active lane are
checked exactly. The ending clock is guarded by the computed endpoint.

Measurements alternate N and 2N launch order, with N=65,539 and six
retained pairs per population. The first warmup pair is saved separately.
Intervals use exact integer clock differences and rational medians. All
28 arrays per form, including warmups, are preserved. These are unsigned
integer predicate probes; floating-point precision does not enter their
computation.

| Native motif | One active warp, cycles/motif | 32 active warps, cycles/motif |
|---|---:|---:|
| VOTE.ALL predicate dependency | 12.978–13.149 | 16.17–16.32 |
| VOTE.ANY ballot and dependent ISETP | 17.032–17.142 | 32.167–35.670 |

The ballot population's second pair is slower than the other five:
`1201611349/33687046` cycles per motif. It remains in the evidence.
No timing residual is fitted, and no pair is discarded to improve a
service estimate. Population differences describe contention under this
specific resident block geometry, not a universal occupancy correction.

The all-vote cubin SHA256 is
`b7aa5f6c01338c0a7b6b83735e7c6a043bff2272763e6755b466e7c70f48ab29`.
The ballot cubin SHA256 is
`8a281ede2ea043441f131e9cba094798a60036062b0d8f809416fd15c91d1ab5`.
Their register counts are 14 and 16 respectively, with zero local bytes.
Archived generator versions accompany each preparation: the ballot
extension does not retroactively change the earlier source epoch.

Separate Nsight captures use N=4,099. Independent imports confirm that
each of the 257 intended vote PCs executes 459,088 times at one active
warp and 14,690,816 times at 32 active warps, with 32 active lanes per
execution. The ballot's 257 dependent ISETP PCs have matching counts.
Slow fallback paths and divergent branches have zero executions. Profiled
clocks are excluded from ordinary timing evidence.

All raw paths below are relative to
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`:

- All-vote preparation and samples: `collective_vote257_e2/ordinary_e1`.
- Ballot preparation and samples: `collective_ballot257_e1/ordinary_e1`.
- Profiles: `profile_collective_{vote,ballot}257_w{1,32}_e1`.
- Independent all-vote receipt:
  `verification/cpu_continuation_independent_20260905/collective_measurement_independent_e1/receipt.json`.
- Independent ballot receipt:
  `verification/cpu_continuation_independent_20260905/ballot_measurement_independent_e1/receipt.json`.

The nominal service catalog may select a named motif transfer while
retaining its measured spread and instruction-form qualification. A mask
operation, a different predicate operation, or a different control path
requires an explicit transfer hypothesis. These measurements contain no
solver timings, measured solver iteration counts or family winner labels.
