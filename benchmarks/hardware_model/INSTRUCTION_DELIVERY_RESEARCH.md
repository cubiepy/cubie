# Instruction delivery research checkpoint

This is a research design and parameter ledger, not an implemented
delivery service. `instruction_addresses.py` supplies its source-address
input; `nominal_execution.py` supplies arithmetic and memory readiness.

The source model should attach stable synthetic instruction addresses to
the allocated trace and insert fetch readiness into `nominal_execution`.
Fetch completion limits warp eligibility before dispatch; other warps,
arithmetic, memory, and outstanding fills continue concurrently. A cache
miss is never an additive whole-kernel penalty. Repeated dynamic visits
reuse static addresses; counted lanes and actual full-unroll copies retain
distinct addresses. Spill/reload forms and retained loop administration
must receive addresses too. Helper inlining is the source-supported nominal
alternative; identical-specialization sharing is a sensitivity.

The corrected local measurements show an effective transition near
128 KiB, not independently established physical cache ownership. Eight
resident-warp capacity declines from 16.359 to 11.585 scalar TFFMA/s over
131184 to 135280 hot bytes. Sixteen-warp capacity remains approximately
13 scalar TFFMA/s across 135280 to 147568 bytes even as GCC miss fractions
increase. These results identify residency-dependent delivery pressure;
miss fractions cannot be converted directly into exposed latency.
The independent six-profile receipt is in
`verification/continuation_profiles_independent_20260905/icache/receipt.json`
under the external hardware evidence root.

The independent sharing probes strengthen this distinction. Alternating
two disjoint 82000-byte regions increases GCC misses and no-instruction
observations while L2 misses remain negligible. Different virtual-SMID
masks approximately halve GCC misses with much smaller timing changes.
Virtual SM identifiers do not establish GPC/TPC adjacency or prove a
cache-sharing domain. See `INSTRUCTION_SHARING_EVIDENCE.md` for exact banks.

Primary-source constraints:

- The [Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
  describes four SM partitions with private L0 instruction caches, without
  specifying their capacity.
- [Nsight Compute's profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
  describes ICC per TPC and GCC per GPC, with misses continuing to L2.
  Counter prefixes do not establish ownership. The unified data-L1/shared
  carveout is a different capacity and must not shrink instruction caches.
- [Jia et al., 2019, sections 3.3 and 3.4](https://arxiv.org/pdf/1903.07486)
  use warmed instruction sequences and interference experiments to study
  capacities and ownership on Turing/Volta. These older devices provide
  explicit architecture-transfer alternatives, not Ada measurements.
- [Huerta et al., 2025](https://arxiv.org/html/2503.20481)
  study an instruction-fetch model with stream-buffer prefetching. Their
  selected 16-entry prefetcher minimizes prediction error and is excluded
  from this hardware-derived model. Demand fetching and ideal sequential
  streaming can instead be named sensitivity endpoints without fitting
  a buffer capacity.

## Implementable input and numerical ledger

The event stream uses the synthetic PC projection's exact graph/plan
hashes. Each warp holds a next-event position, fetched-instruction state,
and the existing register/memory readiness. Fetch completion is another
readiness event. A cache request names the synthetic instruction line and
an explicit sharing domain; equal outstanding requests may merge in the
named coalescing alternative. Completed fills install cache lines and
wake dependent warps. No fitted code-size factor enters this mechanism.

| Input | Value or finite alternative | Qualification |
|---|---|---|
| Instruction width | 16 bytes | NVIDIA Binary Utilities, SM89 |
| SM partitions | 4 | Ada hardware organization |
| L0 ownership | Per partition | Ada whitepaper |
| L0 capacity | 16 KiB nominal; 12 KiB sensitivity | Turing/Volta transfer |
| Effective large-body transition | 128 KiB | Target probe anchor; not ownership proof |
| ICC/GCC ownership | TPC/GPC | Nsight Compute documentation |
| Shared-path transaction granularity | 256 B proxy; 64 B sensitivity | Constant-path transfer, not measured instruction line |
| Shared-level complete fill latency | 92 cycles; 89-cycle sensitivity | Turing/Volta constant-path proxy |
| L2-backed complete fill latency | 215 cycles; 245-cycle sensitivity | Turing/Volta constant-path proxy |
| Fetch/decode ceiling | 1 instruction/partition/cycle hypothesis | Huerta model assumption |
| Instruction fill initiation interval | Not identified | Needs a service measurement or explicit endpoint |

The geometry and constant-path numbers above come from
[Jia et al., Table 3.1 and section 3.4](https://arxiv.org/pdf/1903.07486).
That table does not publish instruction-line sizes or instruction-fill
latencies. Using its shared constant path for instruction delivery is an
explicit path-transfer hypothesis. These are complete alternative path
latencies, not additions to one another. The paper establishes constant
and instruction interaction at the shared level on its tested devices;
it does not establish identical Ada requester pipelines.

[Huerta et al., section 5.2](https://arxiv.org/html/2503.20481)
motivate a three-entry instruction buffer from a two-stage fetch/decode
model and greedy issue observations. Their fetch policy and buffer model
are assumptions. They can be tested as one named transfer; they must not
be promoted to Ada specifications. Their fitted 16-entry prefetch capacity
does not enter the ledger.

## Finite delivery alternatives

A demand-fetch alternative issues only a required missing line. An ideal
sequential-streaming alternative overlaps future contiguous line fills
with useful execution, subject to the supplied line-service resource.
These are explicit sensitivity endpoints; neither needs a fitted
prefetch-depth parameter. Branch targets become available under the
declared source path and branch readiness, not an assumed omniscient
branch predictor. The first demand fill and subsequent line traffic are
separate from steady resident instruction delivery.

For the target, the effective 128-KiB capacity should be exercised under
both per-SM transfer and documented shared-domain hypotheses. The
per-SM-only interpretation is insufficient for the mixed-region sharing
experiment: each individual SM executes one 82000-byte region, yet mixing
regions across SMs introduces large GCC pressure. Explicit GPC sharing
is therefore the source-supported nominal domain direction. Virtual SM
IDs cannot determine which particular SMs share a physical GPC. Domain
assignments must remain stated hardware-layout alternatives.

Warp drift should arise from source regime differences and scheduled
dependencies. Identical start PCs do not imply identical later PCs.
Alternative source-cap/Newton/Krylov/FSAL paths can populate resident
warps explicitly, without measured iteration counters or fitted phase
offsets. The scheduler's interleaving then generates temporal reuse and
overlap. A reserved code span or an accessed-PC union alone cannot replace
this sequence.

## One necessary service distinction

The existing data already rejects treating every large-body GCC miss as
an identical exposed delay. It also makes unprefetched 92-cycle line
fills an implausible nominal explanation for the near-peak 64-KiB,
eight-warp control. Conversely, ideal latency hiding with unlimited fill
bandwidth can erase the measured capacity cliff. A finite fill initiation
service, distinct from latency and fetch width, is therefore the missing
physical quantity that matters most for a quantitative delivery model.

A targeted probe should preserve the corrected FFMA body, exact native
endpoints, two full occupancy waves, and a warmed L2-resident instruction
working set. Compare source-defined synchronized versus separated region
streams at several resident-warp capacities, collecting GCC instruction
requests and instruction-attributed L2 sector traffic. Transaction bytes
per independently conserved GCC miss identify granularity; sustained
sector completion per hardware cycle identifies an aggregate service
ceiling only when saturation is demonstrated. Ordinary timing remains
separate from the profiled companion. A throughput plateau without the
conserved traffic/residency evidence must not become a fitted miss cost.
