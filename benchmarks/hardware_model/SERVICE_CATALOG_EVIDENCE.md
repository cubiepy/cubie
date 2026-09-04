# Conditional service catalog and exact coverage

`ARITHMETIC_PROXY_CATALOG.json` is an executable arithmetic-only scenario
for the frozen NativePlan stream. It keeps measured target observations,
published capacity, proxy latency and unfilled memory services separate.
No solver timing or native holdout label is an input.

The exact 48 retained plans contain ten model categories: FADD, FFMA,
FMUL, FNEG, MOV, MUFU.RCP, LDL, LDS, STL and STS. IADD3/IMAD are useful
arithmetic-probe targets but are not in this plan union. The input
inventory and every file hash are retained in
`arithmetic_proxy_estimates_e1/receipt.json`. The evaluator at
`verification/arithmetic_proxy_estimates_20260905.py` extracts only the
existing `service_estimate` function from the bound f547ee91 NativePlan
source and feeds its existing modeled register traces and geometry.
It does not rerun lowering or change a plan.

| Category | Scenario service | Qualification |
|---|---|---|
| FADD/FMUL/FFMA | 4-cycle proxy; 32/128 SM cycles per full-warp instruction | Turing latency transferred explicitly; SM89 published aggregate FP32 capacity |
| FNEG | FADD service | The model declares signed/zero-operand FADD at native_plan.py:626; no distinct native FNEG claim |
| MUFU.RCP | approximately 15-cycle proxy; 32/16 SM cycles per warp | Generic Turing MUFU latency, not a measured SM89 reciprocal value; normal single-instruction approximate path assumed |
| MOV | 4-cycle proxy; one instruction/scheduler/cycle scenario | Measured Turing MOV latency; no extra execution bottleneck assumed beyond the model scheduler; no measured MOV initiation claim |
| LDS | Unfilled | Target 24-cycle observation is a one-active-lane uint32 chain with administration, not a full-warp latency/initiation pair |
| LDL | Unfilled | The qualified global uint64 chain does not identify uint32 local-frame service |
| STL/STS | Unfilled | Same-cell memory completion and register-result latency are different quantities |

The older-architecture latency values come from the original measured
instruction table, not an expected-value README. Fixed-latency entries
were investigated by changing producer/consumer stall separation;
generic MUFU values are approximate.
[Jia et al., Table 4.1, printed page 40](https://arxiv.org/pdf/1903.07486#page=40).
NVIDIA's operation table reports results per SM clock. An FMA produces
one result in that convention; it still performs two scalar FLOPs.
[CUDA Best Practices 13.3, Table 5](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions).

All FP32 categories share one aggregate capacity in this scenario; SFU
capacity is separate. MOV shares the existing scheduler issue limit and
has no additional modeled contention. The engine's cyclic warp-to-four-
scheduler assignment is an explicit condition. These choices expose
resource-sharing hypotheses; aggregate published rates do not prove
every physical routing or operand-port detail on AD104.

The fractional 0.25-cycle FP32 reservation is a continuous aggregate-
capacity scheduling convention. The published capacity does not establish
quarter-cycle physical issue times or per-partition dispatch. The model
can therefore produce fractional readiness times without claiming an
observed subcycle hardware issue trace.

Eight plans contain only filled categories: chain16/17 local placement,
promoted materialization, both block sizes and both contraction states.
Forty plans keep missing-service status. The eight estimates are:

| Workload | Contraction hypothesis | Cycles per modeled resident wave | Cycles per resident warp |
|---|---|---:|---:|
| chain16 local promote | False | 15,905 | 1,988.125 |
| chain16 local promote | True | 10,802.5 | 1,350.3125 |
| chain17 local promote | False | 16,983.5 | 2,122.9375 |
| chain17 local promote | True | 11,421.75 | 1,427.71875 |

Each row applies to both 32- and 64-thread plans because their frozen
modeled resident-warp counts coincide. These are conditional one-step
service outputs, not observed times, measured winners or a comparison
across different systems. The hypotheses of materialization and
contraction remain visible. Actual native holdout disagreements are
evaluated independently; they cannot be repaired by fitting these
service values. Instruction fetch, cache misses, caller setup and outer
integration work remain excluded.

## Memory completion experiment and engine boundary

The existing engine updates a cell's `memory_ready` timestamp after
every memory event and includes each event's supplied latency in wave
completion (`native_plan.py:1331-1350`). Consequently its store field
means memory/ordering completion. A store has no result register whose
latency can simply be assigned the measured load interval. The present
catalog leaves that field unfilled.

A bounded instrument should use separate runtime write/read offsets
that are numerically equal at launch. Both offsets remain unknown to
the compiler. A side-effecting inline-PTX body contains actual STS/LDS
or STL/LDL pairs, with no address arithmetic between a store and its
dependent load. Each load result supplies the next store value. Native
admission must reject forwarding/elimination by the compiler and bind
both addresses to the same thread-private cell in the raw request.

For shared memory, a 1,024-thread block has one four-byte cell per lane
and a 4-KiB static window, plus independently queried driver reservation.
One full-warp and 32-full-warp populations use the same binary and final
CTA barrier. For local memory, a four-byte local cell per thread has an
actual native frame and stack-relative addresses that must be verified.
Both forms keep the two runtime offsets in separate registers even
though their supplied values agree. Local L1/L2 traffic and shared bank
conflicts must be qualified by actual counters.

Initialize the cell to a poison value different from its lane's seed.
After each odd 33- or 257-pair body, one integer increment changes the
next stored value. N and 2N therefore have distinct exact modular final
values. The increment, address setup, counter and endpoint guards stay
in the native inventory; the repeated pairs themselves have a true
store-data/load-result dependence. The first and final memory values,
clock words and operation counts are retained. No unfounded subtraction
turns the resulting composite interval into a standalone store latency.

Two explicit ordering variants discriminate mechanisms: ordinary
same-thread store/load order permits any hardware forwarding; an
inserted CTA-scope memory fence forces the additional ordering path.
Both require retained native operations and exact data. The fence is
part of the measured composite, never zero-cost. A 33/257 comparison is
an administration control only when the actual scheduling/resource and
ordering forms match; differing native forms cannot justify fitting an
intercept.

The corresponding engine change should distinguish store issue and
source-register consumption from a later same-cell RAW edge. A matched
store/load composite can provide an explicitly conditional edge service
without fabricating two independent instruction latencies or counting
the same load twice. Unrelated cells may proceed subject to their
shared issue/queue resources; a shared store with no modeled consumer
does not acquire an invented load-like delay at the step boundary.
Pending read-after-write, source-register reuse and external-consumer
completion must be explicit state, not hidden in one generic store
constant. This is a design proposal; no engine or store instrument has
been changed by the catalog work.
