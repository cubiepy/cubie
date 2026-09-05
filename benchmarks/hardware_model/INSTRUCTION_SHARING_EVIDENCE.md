# Instruction-stream sharing evidence

The accepted mask-1 experiment changes the instruction stream selected
by each SM while retaining one native binary, launch geometry and total
repeated arithmetic work. Mixing two streams produces substantially
more instruction-delivery stalls than selecting either stream uniformly.
It does not identify a physical cache-sharing map from virtual SM IDs.

Raw paths below are relative to
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
The accepted ordinary bank is `instruction_sharing_ordinary_e3`.
The corresponding profiles are `profile_sharing_all_a_e3`,
`profile_sharing_all_b_e3` and `profile_sharing_mixed_e3`.
The independent ordinary audit is
`verification/instruction_sharing_e3_independent_20260905/receipt.json`.
The complete saved-report audit is
`verification/instruction_sharing_profiles_e3/receipt.json`, with status
`EXACT_SHARING_PROFILE_WORK_AND_RESIDENCY_VERIFIED`.

## Controlled native work and resources

Controller SHA256 is
`21db7e0df15667715095673da635ed616caa7f0ebd90ca9270392a525bee060b`;
worker SHA256 is
`1dde5b8ccacf65bc19076da56252097d449ea354aa6baa385be7d2bcf4f43f2d`.
Every accepted ordinary and profile arm uses cubin SHA256
`495962af77fdd626996e2dd09b6c13b60e9abd398a217398a970b009bc96aa9e`.
Each repeated region contains 5,120 FP32 FFMAs over eight independent
recurrences and its counted-loop administration. The two regions occupy
82,000 bytes each, at native byte intervals [464, 82464) and
[82480, 164480). Their fixed entry/exit work differs and remains counted.

The runtime count is N=4,096 or 2N=8,192. All-A and all-B each select
896 warps for one stream; mixed selects 448 warps for each. The native
SMID instruction reports virtual identifiers. Observed identifiers span
0 through 55; their numerical order does not prove a TPC or GPC map.
Matching entry/exit identifiers also does not prove absence of migration
between those observations.

Each block has 256 threads, 27 registers/thread and zero local bytes.
The 112-block grid gives two waves at one resident block per SM. This
one-block bound is independent of the carveout preference: 50,177 dynamic
shared bytes plus 1,024 driver bytes, rounded to the 128-byte allocation
unit, requires 51,328 bytes/block. Two blocks require 102,656 bytes,
exceeding the queried 102,400-byte maximum SM shared capacity. At one
less dynamic byte, two 51,200-byte allocations would fit. The reservation
has no native shared-memory operations.

All three profiles report 65,536 shared bytes/SM and occupancy limits of
24 blocks, eight register-limited blocks, one shared-limited block and
six warp-limited blocks. Actual profile geometry therefore agrees with
the independent maximum-capacity bound. The profile does not directly
observe every ordinary launch's chosen carveout, but the capacity bound
does not depend on that choice. The allocation repair was independently
checked in
`verification/instruction_sharing_capacity_independent_20260905`.

## Ordinary timing

There are three calibration launches and 72 measurements. Each cell is
the median of 12 ordinary CUDA-event measurements; these are kernel
times, not profile-event times. All raw outputs, source identities,
SMIDs, selections and repeat counts pass their exact checks.

| Repeat count | All A (ms) | All B (ms) | Mixed (ms) |
|---|---:|---:|---:|
| N | 38.380001 | 38.221935 | 97.396126 |
| 2N | 76.654224 | 76.208336 | 194.603394 |

The roughly proportional N/2N response retains the mixed-stream
slowdown. It does not assign a cycle cost to each instruction miss.

## Exact work and counters

Each profile captures exactly one N launch. The saved source-PC rows
verify every repeated FFMA, decrement, comparison, backedge and exit
guard, including N versus N-1 executions. Summed source counts exactly
match hardware warp instructions in every arm; aggregate periodic
sampling counts also reconcile without residual. The fixed paths are
retained, so whole-kernel counts are close but not identical.

| Quantity | All A | All B | Mixed |
|---|---:|---:|---:|
| Warp instructions | 18,808,895,616 | 18,808,890,240 | 18,808,895,168 |
| Thread instructions | 601,884,659,712 | 601,884,487,680 | 601,884,645,376 |
| Predicated-on thread instructions | 601,767,133,184 | 601,767,075,840 | 601,767,161,856 |
| ICC request cycles | 1,877,123,970 | 1,876,601,279 | 2,618,628,716 |
| ICC miss cycles | 60,910,530 | 71,600,630 | 1,434,399,402 |
| GCC instruction requests | 193,660,264 | 199,353,303 | 294,983,551 |
| GCC instruction misses | 1,696 | 9,223 | 155,765,178 |
| No-instruction stalled warps | 19,967,595,777 | 19,769,049,444 | 92,062,052,913 |
| Eligible warps | 21,424,787,469 | 21,472,762,903 | 21,906,051,708 |
| Summed SM elapsed cycles | 5,695,987,734 | 5,693,497,142 | 14,478,707,484 |
| L2 lookup-miss sectors | 0 | 0 | 2 |

ICC cycle counts and GCC request counts have different units. They
cannot be added or treated as equivalent miss events. NVIDIA describes
ICC as shared within a TPC and GCC as serving constant/instruction
caches within a GPC; that architectural description does not decode the
virtual identifiers in this experiment.
[Nsight Compute metric units](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#units).

The simultaneous increase in GCC instruction misses, ICC miss cycles,
no-instruction stalls and elapsed cycles supports instruction-delivery
pressure under mixed selection. Nearly unchanged arithmetic work and
negligible L2 lookup misses rule out those particular work changes as
the explanation. The result does not establish an isolated miss latency,
replacement policy, cache capacity or universal penalty above 128 KiB.
Distinct executed-PC unions are not temporal working sets.

## Additional selection controls and excluded epochs

The unchanged instrument's mask-2 ordinary bank is
`instruction_sharing_mask2_ordinary_e1`. Its N medians for A/B/mixed are
38.352097/38.184879/91.602322 ms; 2N medians are
76.709919/76.350910/183.028450 ms. The independent mask-aware adapters
replay both ordinary banks and preserve all three accepted mask-1
profile results:
`verification/instruction_sharing_mask_independent_20260905/receipt.json`.
The subsequent six matched profiles for masks 2 and 32 completed with
the same cubin, actual one-block/two-wave geometry and exact per-PC work.
Their complete audits are
`verification/instruction_sharing_mask2_profiles_e1/receipt.json`
(SHA256 `700d0758196a94bd5ef08f6c15d41147c8041712a1237571595bc50f52146bd4`)
and `verification/instruction_sharing_mask32_profiles_e1/receipt.json`
(SHA256 `a3d5022df23d2ac37ab99b8f4029fcc8814f9870125a8827476e0c2d94a40e96`).
Mask-32's A/B/mixed ordinary medians are
38.429567/38.149632/91.265568 ms at N and
76.485153/76.184498/182.262367 ms at 2N. Each median has 12 samples.
Mask-32 mixed selects 512 A and 384 B warps; masks 1 and 2 select
448 each. The different fixed-path counts remain explicit: mask-32
mixed has 18,808,895,232 warp instructions, 64 more than mask-1/2 mixed.
Every profile's hardware/source warp-instruction residual is zero.

| Mixed selection | Mask 1 | Mask 2 | Mask 32 |
|---|---:|---:|---:|
| GCC instruction misses | 155,765,178 | 69,717,051 | 68,102,624 |
| Minimum GCC instance instruction misses | 26,432,318 | 10,774,024 | 9,651,478 |
| Maximum GCC instance instruction misses | 34,612,551 | 15,894,210 | 15,828,658 |
| ICC miss cycles | 1,434,399,402 | 928,663,827 | 907,884,374 |
| No-instruction stalled warps | 92,062,052,913 | 77,375,571,683 | 79,506,194,248 |
| Summed SM elapsed cycles | 14,478,707,484 | 13,559,488,788 | 13,562,972,580 |

The two fresh uniform controls retain low GCC instruction misses:
mask-2 A/B are 1,696/9,223 and mask-32 A/B are 1,700/9,375. The mixed
minimum/maximum values are the retained NCU `.min` and `.max` rollups,
not recovered instance IDs or a map from virtual SM IDs to physical
GPCs. They show elevated counts even in the least-affected reported GCC
instance. Changing the mask does not establish that physical mapping.
The approximately halved GCC miss count produces a much smaller timing
change; an additive fixed-cost-per-miss timing explanation is inadequate
for this contrast without modeling overlap and delivery behavior.

The earlier e1/e2 ordinary runs are not accepted residency controls.
Their dynamic reservation was only 3,073 bytes. The first completed
profile, `profile_sharing_all_a_e2`, reports six possible blocks/SM at
65,536 shared bytes, making 112 blocks only one third of a full wave.
The frozen helper's occupancy search under preference zero did not
constrain the actual launch. Those records remain preserved; the
corrected experiment does not relabel their timings. The e1 profile
failed even earlier on fastmath-set serialization, before compilation.

The instruction-sharing result informs the model's explicit cache-domain
and instruction-stream scenarios. It supplies no fitted slowdown
coefficient and does not replace family-specific generated workloads.
