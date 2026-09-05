# A precompile model for unroll and buffer choices

The first predictor should construct a small machine model from each
candidate's generated program and estimate its execution. It should return
a recommended candidate, predicted costs under named assumptions, and the
conditions that change that recommendation. A hardware limit is a useful
filter; it is not the whole heuristic. Overlapping lower bounds do not
prevent approximate predictions, provided the approximation is explicit
and tested without fitting solve-time coefficients.

This is an implementation design, not a claim that the predictor exists.
It extends [MODEL_PROTOCOL.md](MODEL_PROTOCOL.md) with a concrete estimate
mode. That mode is distinct from the certified bounds implemented by
`physical_capacity.py`: its scenario envelope is neither a confidence
interval nor a guaranteed enclosure of native behavior.

## Inputs and the first executable path

Use `predict_candidates(snapshot, candidates, hardware, scenarios)` on
JSON-compatible inputs. Construction captures the actual helper graph,
closure values, effective JIT/compiler settings, resolved buffer registry,
and requested launch geometry before native specialization. Give each
construction a fresh codegen cache; retain source-byte and function hashes,
typed array contents/hashes, imported helper paths, and zero-overload
receipts. A compiled baseline, observed registers, saved SASS, measured
iteration counts and bank winners are validation labels only.

The implementation has five composable passes:

1. **Candidate graph:** join `describe_workload`, `describe_expansion` and
   `describe_buffers`; use `describe_source_values` for complete admitted
   regions. Produce typed value/control/memory graphs, storage identities,
   recurrent regions and actual call bindings.
2. **Lowering scenarios:** rewrite admitted graph patterns into native
   instruction categories, operand classes, storage alternatives and
   address computations. Save every rewrite's source nodes and conditions.
3. **Schedule and allocation:** construct an instruction order, assign
   modeled registers or explicit memory slots, and insert modeled
   spill/reload/rematerialization work. Derive residency from hardware.
4. **Execution:** generate warp issue and memory-address streams for a
   declared iteration regime. Simulate dependency, pipeline and cache
   service events to obtain cycles and traffic, retaining component counts.
5. **Selection:** rank the same finite candidate set in each scenario;
   choose a nominal recommendation and report changes across scenarios.

Start the complete path with the actual Lorenz/RK4 and chain32/Vern7
graphs already admitted in [SOURCE_VALUE_GRAPH.md](SOURCE_VALUE_GRAPH.md).
Their complete source witnesses make the transformation and scheduling
checks executable without another source-tracing abstraction. Add actual
generated LU helpers next, then Newton/BiCGSTAB recurrence regions and
their caller live-through sets. The latter require explicit loop phis and
branch masks: the existing complete value-graph API accepts fully expanded
ERK and rejects these residual controls. The conservative buffer inventory
must not be relabeled as a complete value graph to bypass that boundary.

The result schema must distinguish:

```text
candidate, source_identity, hardware_identity, model_version
coverage: complete_regions / residual_regions / residual_reasons
scenario: lowering / allocation / scheduling / cache / iteration regime
features: source operations, native categories, code addresses, accesses
resources: modeled GPR words, allocation quanta, blocks, shared carveout
estimate: cycles by execution scope, issued work, stalls, memory traffic
bounds: separately qualified physical bounds with compatible units
decision: selected / conditional / unsupported, rank, assumption changes
validation: empty during prediction; joined independently afterward
```

An unknown region remains symbolic and is charged to its call count.
Do not silently give it zero work. The engine can still return conditional
rankings and break-even expressions for the covered differences. It must
identify when an omitted component could reverse its nominal ranking.

The first executable milestone is an ERK `NativePlan` estimator: actual
RHS arithmetic plus accumulator updates, exact byte-cell identities,
typed lowering nodes, a complete scheduled witness, modeled register
assignment, explicit spill slots and local/shared access streams. Run
both scalar-replacement and addressable-buffer plans for the same
fully expanded graph, then add count1/2/4 stage/element/accumulator
templates. Inactive Newton/Krylov groups remain labeled inactive.
This yields concrete instruction, allocation and traffic estimates even
while some cycle-service terms remain symbolic. It also gives a direct
test of whether moving an accumulator to shared saves its modeled local
traffic but raises spills or reduces modeled L1 capacity.

Use the fully expanded source graph as a semantic reference, not as
permission to constant-fold rolled code. Candidate compilation visibility
and per-iteration execution values are different facts: a runtime
coefficient test can take a known path in one logical iteration without
disappearing from its modeled native loop body. Each plan retains both
the static instruction template and its executed instance mapping.

## Candidate-specific program construction

Keep full, count 1, count 2, count 4 and `False` distinct. In the retained
harness, label `0` means `(True, 1)`; `n` means `False`, and `libnvvm`
selects all `False`. These are directive identities, not measured native
equivalence classes. The eight groups are stage, step element,
accumulator, solver element, norms, other small, Newton and Krylov.

For a fixed trip count `m` and requested factor `q`, retain a repeated
main body with `q` requested copies and a tail of `m % q` instances.
Track loop administration separately. An index is known only when its
source expansion and closed values prove it. In particular, a counted
loop does not make its changing outer induction variable constant.
Zero tableau coefficients can eliminate a guarded region only under the
typed predicate rules in [EXPANSION.md](EXPANSION.md). Do not add host
NumPy promotion or unsafe floating reassociation to gain more folds.

Early exits require the actual check location and active mask. Newton
and Krylov counts multiply executed visits, not allocated source bodies.
For DIRK, retain `N[attempt, stage, warp]` and
`K[attempt, stage, newton_iteration, warp]`. FIRK has one coupled Newton
solve; Rosenbrock has stage linear solves without Newton. Error smoothing
has its own solve instance and width. Per-run totals cannot reconstruct
these warp-indexed visits: neither their mean nor their maximum across
whole-run totals substitutes for the sequence of per-invocation masks.

Before integration-level iteration histories are available, predict a
step attempt over coherent-warp regimes from the actual configured caps.
Enumerate admissible integer `N` and `K` values, retain zero/cached/early
paths where source permits them, and add explicitly staggered mask
scenarios. Coherent equal counts across stages/inner calls are a modeling
assumption, not a solver fact. Report per-attempt costs and the resulting
decision regions; do not manufacture a typical iteration count or total
integration duration from tolerance alone.

## Native lowering and materialization alternatives

Use graph rewrites, not a weighted sum of Python operation counts.
Each rule specifies input/result dtype, use count, constant/uniform
operands, flags, instruction nodes, dependencies, operand storage classes
and validity conditions. Rewrites consume nonoverlapping graph patterns,
so a multiply contracted into an FMA is not counted again as an FMUL.
Keep addressing, branch/call, predicate, conversion, constant-load and
padding contributions distinct from arithmetic categories.

The 48-case [translation evidence](OPERATION_TRANSLATION_EVIDENCE.md)
supports conditional one-instruction increments for FP32 add, multiply,
single-use FMA and signed/unsigned 32-bit affine patterns under its exact
flags. It also demonstrates why rule context matters: a reused product
changes contraction, and an invariant divisor can share a reciprocal.
The observed counts do not certify full-kernel register demand or a
context-independent instruction multiplier. Attach their actual compiler
identity and match their operand/use conditions before applying a rule.

The nominal compiler approximation is:

- honor requested fixed-loop main/tail structure; fold only proven guards;
- inline a known helper only under an explicit supported inlining rule;
- scalar-replace private exact byte cells whose complete effects and
  aliases permit it, retaining caller live-through values;
- apply legal, explicitly implemented typed contraction/CSE patterns;
- lower remaining addressable accesses with their actual space and width.

Add separate scenarios for addressable versus promoted eligible cells,
retained versus inlined helper boundaries, and legal alternative
contraction/rematerialization choices. Opaque calls and unresolved aliases
cannot qualify for optimistic scalar replacement. Shared placement is
addressable in the nominal model; promotion requires a separate proof of
thread-private effects and a compiler-treatment hypothesis.

Counted-unroll preservation is also a hypothesis. The exact stage replays
in [STAGE_REPLAY_EVIDENCE.md](STAGE_REPLAY_EVIDENCE.md) show that distinct
Radau5 LLVM/LTO directives can converge to one final binary; Kvaerno3
does not share that result. Include late full-unroll as an alternative
for a counted recurrent loop where legal, without a family-specific
alias lookup or a claim that the backend ignores counts. Diagnostic PTX
is not the unsaved original intermediate.

Report category vectors for each concrete lowering. Their min/max over
the enumerated alternatives is a **model scenario range**, not a rigorous
native instruction bound. An unsupported lowering has no finite certified
upper bound merely because a few native examples were observed.

## Scheduling, retention and spills without a liveness multiplier

The source graph's 22-value Lorenz peak and 423-value chain32 peak are
source quantities. Even the exact source optimum is not a GPR bound.
First construct the modeled native graph, including integer/address
temporaries, predicate/uniform classes, copies, register pairs, and
operand/destination overlap rules. Storage width follows the lowered
type; no coefficient converts a source peak into registers.

Use a deterministic ready-list schedule: prioritize the longest remaining
hardware-latency path, then smaller incremental retention, then stable
source ID. Preserve value, memory-order and control dependencies. Retain
source order as a second schedule hypothesis. This choice is a compiler
approximation; its priorities are not fitted weights.

For each schedule, assign reusable physical words to nonoverlapping live
ranges. The nominal allocation uses the smallest modeled no-spill budget
when it fits the per-thread limit. Otherwise allocate at the largest legal
budget and evict the value with the farthest next use, preserving dirty
values in explicit typed spill slots. Reloads introduce instructions,
address work and dependencies. A pure expression may instead be
rematerialized when all operands are available and its modeled service
cost is lower than that reload; retain both alternatives when service is
unresolved. This is a declared allocation policy, not an inference about
the installed compiler's allocator.

Evaluate the other register budgets at hardware allocation/residency
transitions as sensitivity scenarios. Do not optimize a fictional
register cap and present it as the compiler's choice. A budget that
cannot hold the modeled operands rejects that mapping; the original
candidate may still compile by spilling or choosing another lowering.
Caller storage, named local arrays and compiler spill slots remain
separate even when they compete for the same data caches.

For this SM89 target, the installed CUDA 13.3 header gives:

```text
warps_per_block = ceil(block_threads / 32)
registers_per_warp = round_up(32 * modeled_R, 256)
warps_per_subpartition = floor(16384 / registers_per_warp)
register_block_limit = floor(4 * warps_per_subpartition / warps_per_block)
shared_per_block = round_up(static + dynamic + reserved_per_block, 128)
```

Apply the header's per-block register checks, rounded subpartition
allocation, per-thread limit and thread/block limits as well. These are
conditional resource calculations, not a prediction that `modeled_R`
equals the final native register count. At 64 threads, hypothetical
96 versus 99 registers gives register limits of 640 versus 512 resident
threads; 161 versus 167 gives 384 for both. Source:
`CUDA/v13.3/include/cuda_occupancy.h:620–639,1414–1416,1530–1585`, SHA256
`f21d4ea4057a111e64114116ada02bb03c414bd753ceb57fb756d034a0d78d28`.
The shared allocation unit here is 128 bytes, not the register unit.

Enumerate supported shared carveouts jointly with residency. The nominal
driver hypothesis chooses the smallest supported carveout preserving
the non-shared residency target; other legal carveouts remain scenarios.
Never infer actual carveout from a preference or occupancy query alone.
The recent sharing query/profile disagreement makes that distinction
observable, as recorded in [INSTRUCTION_SHARING_DESIGN.md](INSTRUCTION_SHARING_DESIGN.md).
Ada's documented 128 KiB unified data pool and supported carveouts
0/8/16/32/64/100 KiB provide capacities; the retained target's queried
L2 is 48 MiB. The nominal data-L1 model subtracts carveout from that pool;
it does not claim a guaranteed fully usable cache of that size.
[Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#unified-shared-memory-l1-texture-cache),
[target hardware receipts](HARDWARE_EVIDENCE.md).

## Memory, instruction reuse and execution cost

From each modeled instruction, emit participating-lane addresses, access
width, read/write kind and allocation identity. Coalesce by the sourced
transaction/sector rules; do not equate a four-byte payload with a whole
sector or a frame with traffic. Resolve named-buffer strides from the
actual allocator. The spill layout is its own declared native-layout
hypothesis. Memory streams include caller state, coefficients and opaque
effects, not only the buffer being moved.

Start with explicit fully associative LRU data-cache scenarios, cold at
kernel entry and persistent across repeated steps/waves. Track dirty
evictions, reads, writes and writebacks separately. LRU and effective
capacity are assumptions; conflict/indexing alternatives must use an
explicit organization, not a fitted cache fraction. Retain alternative
interleavings of resident warps and the actual batch tail. A large batch
does not multiply every thread's whole local frame into a simultaneously
active working set. Shared bank demand comes from actual concurrent
addresses and the target's sourced bank geometry.

Assign modeled native instructions addresses separately from their visit
counts. On this target instructions occupy 16 bytes. Repeated rolled
bodies reuse addresses; full copies and inlined call instances can use
different addresses. Build temporal instruction and constant-operand
streams, then reuse-distance distributions within an explicit sharing
domain. The union of executed addresses and a ranked-PC histogram do not
establish temporal residency. A 128 KiB code threshold is not a model.

Instruction-cache capacity, sharing domain, fetch transaction and miss
service require separate catalog entries. The fine eight-/sixteen-warp
sweeps and corrected two-stream experiment constrain these hypotheses;
they do not supply a universal miss penalty. Until that catalog is
qualified, return instruction-delivery break-even conditions and an
explicit fetch-hit scenario alongside the compute/data estimate. A rank
under the fetch-hit assumption is useful but is not a complete runtime
prediction for a large instruction stream. Never silently charge zero
fetch cost and omit the assumption.

The execution engine uses per-SM events, four scheduler issue slots, and
separate execution-pipeline and memory-service queues. Nominally assign
warps to schedulers cyclically and issue the oldest ready warp; record
this scheduling assumption. A dependency completes after its catalog
latency; a pipeline accepts work at its catalog initiation rate. Memory
requests wait for the modeled cache/queue path. Track actual overlap
through events instead of adding instruction, dependency and bandwidth
lower bounds. Multiple possible pipelines are alternatives for one
instruction, not simultaneous charges to all of them.

The engine must satisfy the compatible `W/4` aggregate-SM-cycle dispatch
check from [ISSUE_CAPACITY.md](ISSUE_CAPACITY.md). An estimated category
vector is not admitted as certified work by `physical_capacity.py`; use
the same algebra as a model consistency check while keeping its status.
Elapsed device cycles require explicit work assignment across SMs;
conversion to seconds additionally requires a stated clock scenario.

Every service value needs units, opcode/access width, architecture,
method, source and uncertainty status. Separate latency from throughput.
The older dependent-load intervals include address/loop/endpoint work and
cannot become intrinsic latency constants. The direct-address protocol
in [LATENCY_PROTOCOL.md](LATENCY_PROTOCOL.md) targets that gap. A published
same-architecture/different-SKU value can be an explicit proxy scenario:
for example, Luo et al. Table 3 reports RTX4090 L1/shared/L2/global
intervals of 32.0/30.1/273.0/571 cycles, with 284.8 instead of 273.0 in
the L2 prose. Preserve both source values as alternatives; do not average
the discrepancy or label them RTX4070 SUPER measurements. Width-specific
throughput and whole-device bandwidth must not be transplanted by simply
scaling SM count. [Primary paper](https://arxiv.org/html/2501.12084v2#S4.T3).

Thus the first estimate is executable with a supplied service catalog,
including clearly labeled proxies. Missing rates stay symbolic and give
rank-change conditions. The narrowly required local measurements are
the surviving instruction categories' dependent/independent service,
qualified data-cache paths, and instruction-delivery service/domain.
They are hardware measurements, not free solver-specific parameters.

## Family defaults come from distinct graphs

| Family / inner solve | Required graph and resulting decision mechanism |
|---|---|
| ERK | Actual sparse tableau, cached first RHS and accumulation dependencies. Expansion can remove guards and expose cells, while repeated RHS/accumulator live-through changes modeled retention and code reuse. |
| DIRK / LU | Separate implicit stages reuse an n-wide solve. Use actual generated sparse factor/solve operations, including factor reuse and `inexact_newton`/`prefactored`; do not substitute a dense n-cubed formula. |
| FIRK / LU | Main width is s*n, smoothing width is n. Simultaneously live factors and reconstructed stages differ from DIRK; a shared placement or unroll decision must price each actual call instance. |
| DIRK or FIRK / BiCGSTAB | Five work buffers cross actual operator/preconditioner calls. Solver-element expansion and Newton/Krylov replication change index visibility, retention and instruction reuse together. Evaluate those interacting groups jointly over symbolic counts. |
| Rosenbrock / LU or iterative | Prepare the actual Jacobian once per step and model stage linear calls and optional smoothing. There is no Newton multiplier; Krylov and factor-cache lifetime follow the captured linear solver. |

The generated-source receipts give a concrete distinction: Fabbri Radau5
main LU has width 105 and 1,227 factor elements, whereas its smoothing
and Kvaerno3 LU have width 35 and 137. The prefactored Lorenz example
reads prepared cache elements instead of that factor allocation.
These are source graph differences, not fitted family coefficients.
See [BUFFER_DESCRIPTOR_DESIGN.md](BUFFER_DESCRIPTOR_DESIGN.md) for exact
source locations, helper identities and read/write counts.

Preserve one existing descriptor limitation: `workload.py:730–736`
records the error solver's subtype but currently chooses its iteration
description using the main solver's `inner`. The new adapter must read
each actual main/error subtype, width and cap directly. Verify a
constructed mixed-subtype configuration before admitting that combination;
do not change the frozen descriptor or reuse its mislabeled field.

## Decision rules and held-out verification

For a supplied finite policy/placement set, evaluate joint candidates;
do not commit one loop group's choice before evaluating interacting
groups. Memoize identical region descriptors, retaining every directive
identity. Equality of the modeled graph is not a precompile cubin-alias
claim. A complete policy search may enumerate the five levels of each
active group; a restricted search must state its coverage and cannot
claim a global optimum. Placement candidates come from actual registry
owners and legal spaces, not a family winner list.

The nominal recommendation minimizes estimated cycles under the declared
nominal compiler, allocation, cache and iteration scenario. When the
caller supplies no iteration regime, return decision regions and a
default minimizing worst relative modeled regret over the explicitly
enumerated coherent regimes: `max_r(T[p,r] / min_q T[q,r])`. This is a
declared selection preference, not a fitted time law or probability
distribution. Report sensitivity to other compiler/cache/mask scenarios;
do not mix incompatible scenarios into one apparent physical measurement.
An incomplete service catalog yields a conditional recommendation with
its break-even requirements rather than an invented scalar penalty.

| Decision | Admissible claim |
|---|---|
| Resolved shared request exceeds a hardware launch limit | Reject that placement/geometry pair; a smaller block may remain legal. |
| A complete hypothetical allocation exceeds its register budget | Reject that allocation hypothesis, not compilation of the policy. |
| One candidate wins every enumerated model scenario | Model-stable recommendation within that coverage, not hardware-proven dominance. |
| One candidate has a smaller instruction lower bound | It needs less minimum issue work; runtime ordering does not follow. |
| Native aliases observed in validation | One measured binary identity; preserve requested policies and do not count aliases as independent wins. |

Freeze predictions before obtaining each held-out native/timing label.
Validate the chain of mechanisms: lowering categories and addressability,
registers/stack and actual geometry, exact executed warp work, local/shared
traffic and misses, then uninstrumented paired time. Numerical status and
raw state/counter checks precede timing eligibility. Keep same-epoch
repeated references, at least two actual occupancy waves, and aliases as
one physical observation. Do not infer warp work from mean counters.

Use the already qualified contrasts as falsification cases:

- **Counted N2/N4:** predict distinct requested graphs, then test whether
  the late-full alternative explains the exact Radau5 alias without
  asserting the same lowering for Kvaerno3.
- **L96 Kvaerno3 K4 versus both rolled:** their saved per-run state and
  counter arrays match, but K4 has more native work and local traffic.
  A model that changes only iteration counts fails this contrast.
- **Fabbri Radau joint rollback:** lower ordinary medians coexist with
  higher executed instruction work. An issue-count-only ranking fails;
  examine its modeled storage and temporal instruction reuse instead.
- **Same-cubin stage-base reservation:** identical native work with more
  local misses under a larger actual carveout tests cache/reuse service
  separately from instruction translation or allocation changes.

Exact qualifications and raw joins remain in
[SIZE_FAMILY_EVIDENCE.md](SIZE_FAMILY_EVIDENCE.md),
[FABBRI_PROFILE_EVIDENCE.md](FABBRI_PROFILE_EVIDENCE.md), and
[RESERVATION_EVIDENCE.md](RESERVATION_EVIDENCE.md).
Do not apportion their total slowdowns among mechanisms from aggregate
counters alone.

The next compiler-only fragments should preserve actual generated
helper bodies and caller live-through: literal versus dynamic indexing,
full/count1/2/4 plus tails, and single-use versus multi-use products under
the recorded contraction flags. Inspect promoted cells, address nodes
and native retention; do not merely fit total instruction count. Then
choose fresh dimension holdouts by evaluating generated `chainN` graphs
without compilation, where that actual factory accepts the requested
dimension. Locate adjacent sizes where modeled register allocation,
resident blocks, or a supported shared carveout changes; save predictions
for the immediately preceding and following valid dimensions. Select
arithmetic-intensity contrasts through actual generated expressions and
retain their exact DAG differences. These sizes are derived from resource
equations, not cutoffs selected from runtime data. Freeze each prediction,
candidate set and service catalog before the same-flags native labels
and fresh ordinary two-wave measurements. Existing solver banks inform
mechanism selection and are not untouched holdouts. Apply the same
sequence separately to each family and main/error solve combination.
A prediction failure changes
an identified lowering or hardware hypothesis and triggers a new holdout;
it never creates a per-family timing multiplier.
