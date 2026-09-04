# Nominal physical services for typed implicit workloads

This catalog supplies useful **estimates**, with explicit physical transfer
hypotheses. Epoch 2 covers all 21 opcodes found in the six-policy selector bank
and 36 placement arms across nine family/inner-solver cases, and adds
indexed immutable LDC support for the captured-table lowering. It consumes
no solver timing. The JSON hashes those input artifacts and local evidence
receipts. Status: author estimates, awaiting independent review.

The target is RTX 4070 SUPER, AD104, SM89, float32. Published Turing and
A100 native latencies transfer across architectures; RTX 4090 memory
measurements transfer across Ada chips. Neither transfer is a claim of
an exact target-chip constant. The existing ARITHMETIC_PROXY_CATALOG
remains provenance; this catalog adds integer, predicate, memory and
control distinctions. It is not a drop-in legacy service dictionary.

## Nominal instruction services

Cycles below are core cycles. An initiation interval is aggregate SM
capacity reservation per full warp, not fractional scheduler dispatch.

| Forms | Result latency | Aggregate issue interval | Qualification |
|---|---:|---:|---|
| FADD, FMUL, FFMA | 4 | 1/4 | Native arithmetic transfer |
| IADD3, IMAD, LOP3 | 4 | 1/2 | IMAD sensitivity 5; true three-operand add |
| ISETP, FSETP, SEL, MOV | 4 | 1/2 | Measured latency; integer-like port hypothesis |
| PLOP3 | 4 | 1/2 | Low-confidence compare/logic-form transfer |
| MUFU.RCP, MUFU.SQRT | 15 | 2 | Generic MUFU transfer; sensitivity 14 |
| VOTE.ALL | 879918643/67374092 | 1/2 | Target dependent motif, including administration |
| Other VOTE, ACTIVEMASK | unresolved | 1/2 | Converged vote capacity; result transfer required |
| LDC indexed immutable | 29 | 1 per partition | Broadcast hit; scheduler-only issue ceiling |
| LDS | 24 | 1280/637 | Target shared chain; scalar issue transfer |
| LDL, L1-hit scenario | 32 | 1280/637 | Ada pointer chase, local-path transfer |
| STS, STL | no result register | 1280/637 | Read-to-store issue-capacity hypothesis |
| BRA | unresolved control delay | one dispatch slot | Separate motif evidence below |
| CAPTURED_LOOKUP | expand concrete form | expand concrete form | Abstract source operation |

The fixed latencies come from [Jia et al., Table 4.1](https://arxiv.org/pdf/1903.07486#page=40)
and [CuAsmRL, Table 1](https://arxiv.org/html/2501.08071).
A100 plain IMAD has four-cycle minimum safe dependence; Turing reports
five. A100 IMAD.WIDE also reports five and must retain its distinct form.
Generic MUFU is about 14 on Volta and 15 on Turing. Treating its RCP or
SQRT variant as one such operation excludes refinement/exceptional paths.
The PLOP3 estimate explicitly transfers a four-cycle predicate/logic form;
there is no direct PLOP3 result-latency measurement in these sources.

[CUDA Best Practices Table 5](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions)
gives SM89 FP32 128, three-operand integer add 64, integer multiply-add
64, bitwise logic 64, approximate SFU 16 and converged warp vote 64
scalar results per clock per SM. HTML column spans were checked:
**warp vote 128 applies to CC12.0/12.1, not SM89**. Two-operand integer
addition has a separate 128-result entry. FMA has 32 results per warp
under this convention, although it performs 64 scalar FLOPs.

The [Ampere PTX study](https://arxiv.org/html/2208.11174) distinguishes
independent CPI from dependent latency. Its independent two-cycle FP32
CPI is not a two-cycle result dependency. Its shared-store timer has no
store-to-dependent-read chain, so the reported 19 cycles cannot establish
store visibility or same-cell RAW readiness. The catalog does not use
that number as a standalone store latency.

## Capacity coupling and finite scheduling

The [Ada SM diagram](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
has four scheduler/dispatch partitions. Each has dedicated FP32 lanes
and lanes shared by FP32/INT32. A nominal aggregate schedule must satisfy
both `32*(FP32 + integer_like) <= 128*t` and
`32*integer_like <= 64*t`, alongside per-partition dispatch. Allowing
128 FP32 plus another independent 64 INT32 results per cycle overcounts
mixed capacity. Predicate, select and move use that shared pool only as
an explicit port-transfer hypothesis; a scheduler-only alternative is
a sensitivity scenario, not a measured issue rate.

For the finite scheduler, keep separate fields for register-result
latency, resource reservation, source-register consumption, and memory
ordering. Reserve each instruction's dispatch and applicable execution
resources together. Advance dependent instructions from result-ready
times. Independent warps and operations can overlap those intervals.
A resource-demand floor combined with a dependency-path floor using
`max` is useful diagnostically, but a full schedule must enforce both
constraints at each issue, with fixed warp/partition assignment.

Do not implement the capacity equations as one scalar opcode initiation
field: FP32 needs a route through dedicated or shared capacity, while
integer-like work consumes shared capacity. For a continuous aggregate
scheduler, reserve shared capacity for integer work and route FP32 to
either pool; fractional reservations are an approximation. Retain the
scheduler bottleneck separately. Loop regimes and residency waves must
represent identical source work across candidates.

## Memory and placement

The target shared chain supports a 24-cycle dependent interval. It
includes recurring native control, with one active lane, and transfers
to conflict-free full-warp scalar access as an estimate. Alternatives are
23 cycles from Ampere and 30.1 from the RTX 4090 pointer chase. A nominal
local L1-hit path uses 32 cycles. Sensitivities include Ampere 33 and the
target 64-bit global.ca chain composite 8860/257, whose width, address
space and administrative work differ from LDL.

[The Hopper/Ada study, Tables 3 and 5](https://arxiv.org/html/2501.12084v2)
reports RTX 4090 scalar L1/shared throughput of 63.7 bytes per SM clock,
hence 128/63.7 = 1280/637 cycles per full-warp scalar access. Shared
vector throughput is 126.5, versus scalar 63.7; a source scalar access
must not silently receive the vector rate. Shared bank capacity provides
an ideal 128-byte/clock alternative. Actual bank conflicts and local
sectors must come from address mapping. Applying scalar-read capacity
to store issue is an additional explicit hypothesis.

For local loads, Table 3 reports RTX 4090 L2 273 cycles, while section
4.1 prose reports 284.8. Preserve both as separate scenarios. DRAM is
571 cycles. These are alternative complete load paths: do not add
L1 + L2 + DRAM. Do not transplant the paper's whole-device L2 bandwidth
to AD104 by dividing by SM count or multiplying by cache-size ratio.
Chip topology and issue capacity remain unidentified by that comparison.

The independently verified same-cell store/load pair intervals are:

| Placement | Nominal B257 | Sensitivity B33 |
|---|---:|---:|
| Shared | 7465/257 cycles | 968/33 cycles |
| Local | 9012/257 cycles | 1171/33 cycles |

These are serialized **motif** intervals, including recurring native
administration. They support a nominal contracted motif service. Reserve
both memory issue demands and advance motif completion using the pair
interval; do not additionally charge its load latency or loop control.
Subtracting a load constant to invent a store latency is unsupported.
Nor does the interval alone identify a store-to-load edge for arbitrary
issue spacing or queue state. A compiler-visible matching motif can use
this approximation; an unmatched store keeps visibility and drain
separate from source-register readiness.

The local33 profile shows that L1-hit stores can still create downstream
L2 traffic. Some local33 and all local257 L2 read/write conservation
checks fail. Those counters therefore cannot determine an exact L2
write-service term or justify a fitted residual correction. The JSON
binds both independent profile receipts.

## Control, instruction delivery and decisions

Six target loop profiles give about 3.00195 stalled warp cycles per
runtime iteration in the branch-resolving counter. This is a recurring
control motif observation, including exit frequency, not an isolated BRA
latency. It must not become an additive charge on every source branch.
The exact fractions and receipt are retained in JSON.

The instruction-cache measurements show a residency-dependent transition
near an actual 131,184-byte hot body. Eight resident-capacity warps lose
throughput rapidly beyond that point; sixteen have a gentler response.
That evidence does not identify a universal 128 KiB cutoff, a physical
sharing domain, or a constant per-miss penalty. Cache footprint and
carveout still give useful risk discrimination. Do not add total fetch
cycles on top of already overlapping execution cycles.

Run the JSON's finite physically sourced latency/path alternatives
separately and report whether the preferred policy changes. Keep family
and LU/MR/BiCGSTAB workload differences in the graph, including iteration
regimes and masks. Shared placement changes capacity and register
allocation; it is not only a replacement load constant.

For non-ALL VOTE/ACTIVEMASK, the sources establish throughput but not a
qualified dependent result service. Keep a named sensitivity parameter and report the break-even
latency at which a comparison changes. A nonnegative parameter domain
is not a finite measured bound. A user-chosen diagnostic interval may
be evaluated with its endpoints plainly labeled assumptions; do not
present an unrelated opcode latency as a measured vote bound. Comparisons
whose identical control contributions cancel, or whose ranking persists
under those sensitivity assumptions, remain useful. CAPTURED_LOOKUP
instead requires source-specific expansion or proven constant folding.

The VOTE.ALL dependent chain is qualified in epoch 2. A mask-producing
ballot/compare probe has native preflight approval and awaits its separate
measurement qualification; it is not yet an ACTIVEMASK service.
Standalone store completion requires a different ordering experiment;
the existing pair evidence already supports a qualified motif estimate.
These are specific evidence gaps, not a requirement to certify every
kernel before producing any heuristic estimate.


## Epoch 2: immutable constants and qualified collective motif

The epoch-1 catalog and scheduler bytes are preserved in
`verification/nominal_service_catalog_epoch1_snapshot_20260905`, with a
SHA256 manifest. This epoch imports no fitted solver coefficients.

[Huerta et al., section 5.4, Table 2](https://arxiv.org/html/2503.20481)
separately measures register-result and source-overwrite dependencies.
Indexed 32-bit LDC uses a regular address register: both delays are 29
cycles on Ampere. That form transfers nominally to Ada; the 26-cycle
Turing broadcast result is a separately labeled sensitivity. LDC retains
its computed IMAD address and immutable address-space-4 table identity.
An execution-witness offset does not become a compile-time constant.

The same table supports LDS source hold 9 and result 24, and STS source
hold 12. Applying global-load hold 11 or global-store hold 14 to LDL/STL
is an explicit address-space transfer hypothesis. These values provide
no standalone store visibility or drain constant. Table 1 additionally
supports a two-cycle shared request interval and four-cycle subpartition
address issue with queueing; that alternative is recorded, not added to
the existing scalar bandwidth model. Its simulation-selected stream
prefetcher size is deliberately not imported.

The independently verified target VOTE.ALL bank has six exact N/2N
intervals at each of one and 32 timed full warps, with 32 allocated warps
and two occupancy waves. The one-warp median is
`879918643/67374092` cycles per dependent motif, including recurring
control. It supplies the ALL-form nominal result estimate. The 32-warp
median `1092167979/67374092` is retained as a contention observation;
using it as intrinsic latency and also charging vote-capacity contention
would count that contention twice. All twelve observations remain in
JSON. No population interpolation or administration subtraction is fit.

Only events identifying `vote_operation: all` receive that default.
The measured probe administration is not attached again per vote.
Generated solver control still has its own work; this motif transfer can
therefore conservatively overlap a small amount of that administration.
