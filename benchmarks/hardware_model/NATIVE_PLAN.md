# Conditional ERK NativePlan

`native_plan.py` implements the first ERK path in
[HARDWARE_MODEL_DESIGN.md](HARDWARE_MODEL_DESIGN.md). It produces a typed
instruction model, explicit register/spill allocation, named-buffer
accesses, two local-backing/cache scenarios and optional service-cycle
rankings. It does not compile, inspect an existing native solver, execute
a device function, or read a timing-bank winner.

Materialization is a compiler hypothesis. The user-visible candidate is
the actual accumulator placement and requested block size; promotion and
contraction alternatives are separate scenarios for that candidate.
The output never recommends a nonexistent "force scalarization" setting.

## Construction and provenance

`construct` writes its own benchmark-source snapshot, exact worker,
request, stdout/stderr and completed worker receipt in a fresh directory.
The worker uses the public cache-root API, restores it, and closes the
Solver before successful process exit. `lorenz` uses the actual Lorenz
factory. `chainN` uses `placement.build_chain(N, 3)` and the existing
chain solver configuration template; its true generated dimension is
checked independently against the graph. No system table is modified.

Local and shared graphs are separately constructed through
`stage_accumulator_location`. Shared stride is read from the actual
kernel properties, including the one-word skew for a positive even
FP32 shared-element count. The model rechecks the requested placement
against the resolved stage owner, the element count against captured
shared views, and the stride/padding relation. The allocator/kernel
source file has its own byte hash. Driver occupancy is not queried.

`estimate` reads JSON and imports only standard-library modules and NumPy.
It verifies referenced extractor/helper/caller source bytes, the retained
constructor literal and executed command, successful worker exit, the
actual source graph's zero-overload receipt, dimension and shared layout.
It binds each resulting plan to its input graph and hardware-manifest
hash. Earlier development receipts remain under their original names.

Supported complete graphs are the existing extractor's fully expanded
FP32 ERK graphs with exact cells and supported arithmetic. Unknown
control, aliases, indices, types or source regions fail admission; they
are not dropped. Newton, Krylov and rolled source templates are outside
this first graph adapter's admitted input contract. The overall model
still requires the distinct family/solver graphs described in the design.

## Explicit lowering rules

Each rule retains original node IDs, dtype and transformation provenance.
The lowered graph keeps value dependencies separately from inherited
memory-order constraints. Known helpers inline under the stated scenario;
the complete caller live-through cuts remain in the output.

| Source operation | Modeled instruction or transformation |
|---|---|
| Typed FP32 add / multiply | One FADD / FMUL category. |
| FP32 subtraction / negation | One signed-operand FADD form, labeled FSUB / FNEG in the model. These labels are categories, not additional SASS opcodes. |
| Literal precision cast | Exact finite normal/zero FP32 or bounded integer constant payload; source bits retained. |
| Same-type cast | Value alias with ordering retained. |
| Constant operand | Deduplicated typed constant, materialized by MOV on demand and rematerialized after eviction. |
| FP32 division under actual `arcp` | One reciprocal and one multiply, with no refinement or reciprocal CSE. |
| Single-use product feeding one add | Optional one-FFMA rewrite, requiring actual `contract` and no other product consumer or observable product. |
| Exact promoted cell | Read/write aliases disappear as instructions while their semantic value flow and required ordering remain. |
| Addressable FP32 cell | Explicit LDL/STL or LDS/STS, a fresh load value, exact source memory-version identity and byte offset. |

Arithmetic rules are conditional compiler approximations. The isolated
[translation evidence](OPERATION_TRANSLATION_EVIDENCE.md) does not prove
them for every lane-varying full-helper context. In particular the
reciprocal and FMA alternatives need not equal the source's separately
rounded arithmetic bit for bit. The memory/register conservation proof
checks the lowered semantics, not native exceptional-input behavior.
Subnormal/nonfinite constants and unsupported promotions fail rather
than receiving approximate folds.

Every modeled value has its own identity, storage type and one 32-bit
word in this adapter. The plan retains one supplied local stack-base
word and, when needed, a supplied shared-base word. Addressing assumes
the literal offsets fit native memory operands. Caller address setup,
outer integration control and additional caller-native values remain
explicit excluded terms; no fitted register overhead is added.

## Allocation and semantic conservation

The schedule is a stable topological order, tied by lowered source order.
There is no claim of an optimal schedule or the installed compiler's
schedule. Arithmetic not needed by observable values or retained memory
effects is eliminated by a backward dependency walk.

An unlimited-budget pass measures this lowered plan's own no-spill word
demand. The nominal bounded pass uses that demand up to the hardware
maximum of 255 words. `--register-budget` can expose other allocation
scenarios. This is not a source-peak multiplier or a native-register
lower bound.

The allocator evicts the unprotected value whose next use is farthest
away, with a stable value-ID tie. Constants are discarded and recreated
by MOV. Other values receive typed spill slots and explicit stores;
unchanged already-spilled values need no duplicate store. Reloads create
instructions. Slots and words are reused only after their values die.
Inputs are read before a dead input word can hold the destination.
Base words stay pinned. A caller entry ABI exceeding the requested
budget rejects that allocation scenario.

The final trace records every operand/result word and named/spill access.
`verify_allocation` independently replays register tags, named-cell
versions and spill-slot versions. It checks every lowered instruction
exactly once, dependency order, operand/result IDs, opcode/memory identity,
register bounds, nonoverlapping spill offsets and all observable exit
values. Entry and exit membership, location schemas, allocation-generated
MOV/LDL/STL operands, spill addresses and memory-version identities are
validated explicitly. The recorded register/spill extents must agree
with the trace. Exit values may remain in valid spill slots; they need not all
fit simultaneously in registers. It does not treat a passing tag replay
as numerical validation of reciprocal approximation or native code.

General expression rematerialization, algebraic CSE, reassociation,
uniform/predicate allocation and unknown compiler temporaries are not
assumed. Constants-only rematerialization and explicit spilling form the
implemented allocation scenario. Alternative compiler treatments remain
uncertainty, not hidden adjustments to the modeled word count.

## Hardware allocation and access streams

The SM89 resource calculation applies per-warp 256-word register
allocation, four subpartitions, 16,384 registers per subpartition,
per-block launch checks, thread/block limits and a 255-register ceiling.
Shared allocation rounds static/dynamic plus queried driver reservation
to 128 bytes. The formulas and their installed CUDA 13.3 source are
recorded in [HARDWARE_MODEL_DESIGN.md](HARDWARE_MODEL_DESIGN.md).

The nominal carveout is the smallest supported value preserving the
non-shared residency target, or the largest legal value when shared
capacity prevents that target. Every legal carveout is retained. This
is a driver hypothesis, not an observation of actual carveout. The
nominal L1 capacity subtracts it from the documented unified data pool.
Geometry infeasibility rejects that modeled placement/block pair only.
The requested block is an explicit geometry assumption. The production
launch path can reduce a shared-heavy requested block at
`BatchSolverKernel.py:928–959`; the model does not adopt that existing
32 KiB performance heuristic as a hardware law. Native validation must
observe the same actual geometry or use separately reviewed pinning.

Named local arrays use a packed, aligned per-thread model frame with
actual allocation identities; source aliases share one segment. Modeled
spill slots occupy a separate tail. The physical local frame may differ.
Same-offset FP32 accesses coalesce over a full warp in the model, with
four 32-byte sectors per instruction. Two backing hypotheses are emitted:

- `resident_slots`: later waves reuse each resident warp slot's frame;
- `trajectory_unique`: later waves receive disjoint frame segments.

Neither uses a generic local pointer as an observed global physical
address. Both retain exact symbolic segment offsets and per-event sector
descriptors. The cache starts cold, persists across two modeled waves,
uses fully associative sector LRU and write allocation, and records
read/write sectors, misses, dirty evictions and retained dirty sectors.
Warp interleaving is cyclic at each memory-event ordinal. Those are
cache/mapping assumptions, not inferred hardware behavior.

Shared accesses use the actual padded per-run stride. Bank assignments
and wavefront counts assume 32 four-byte banks and thread-private views.
There is no L2/DRAM miss or time prediction from local frame size, and no
whole-batch frame multiplication. The stream covers one step per warp;
outer-loop initialization and repeated integration-step histories remain
excluded. Shared/local alternatives can therefore be compared under an
explicit workload boundary without pretending to model the whole solve.

## Service estimates and ranks

Every emitted instruction category contributes a symbolic service term.
With no catalog, the output retains missing terms and a separate issue
capacity component. Its ordering is explicitly not a runtime
recommendation. No synthetic cycle constant enters the saved predictions.

A complete optional catalog supplies each emitted category's positive
latency, initiation interval, pipeline, SM/subpartition scope, source
provenance and assumption. The engine simulates a resident wave with
cyclic warp-to-scheduler assignment and earliest-ready issue. It checks
register readiness, WAW hazards and same-cell memory ordering; distinct
pipelines may overlap. Each scheduler issues at most once per cycle.
This is an executable conditional service estimate, not a sum of
independent lower bounds.

The supplied load/store service is a fixed path assumption. Cache-stream
miss counts are not silently converted into that latency: instruction
fetch and cache-miss service remain explicit excluded terms. A complete
kernel timing estimate requires those paths and outer work. The service
simulator's algorithm fixtures use synthetic rates only to check its
scheduling invariants; those fixtures are separate from predictions.

`rank` requires the same workload/compiler/hardware identity and service
catalog. Rankings stay separate by materialization/contraction scenario.
It normalizes resident-wave cycles by resident warps for a steady full
grid, and states the condition that excluded terms cancel. A recommendation
under that condition is not a measured winner, a global policy optimum,
or permission to ignore a large instruction stream.

## Commands and validation boundary

Use the frozen source tree on `PYTHONPATH` for construction. Every output
directory is fresh. Construction imports CuBIE host factories but makes
no native overload or device launch; estimate/rank import no CUDA modules.

```powershell
python -m benchmarks.hardware_model.native_plan construct --system chain17 --algo vern7 --placement local --output <fresh-source-dir>
python -m benchmarks.hardware_model.native_plan estimate --graph <source-dir>/graph.json --hardware-manifest <saved-hardware-manifest> --mode promote --output <fresh-plan-dir>
python -m benchmarks.hardware_model.native_plan rank --plans <plan-a>/plan.json <plan-b>/plan.json --output <fresh-ranking-dir>
```

Author validation uses actual Lorenz/RK4 and chain16/17/18/32 Vern7
graphs, including separately constructed shared placements. It checks
conservation, a forced-spill Lorenz case, optional contraction, exact
FP32 arithmetic replay for uncontracted Lorenz, and deliberate corrupted
allocation traces. Independent review is required before the predictions
are released for native holdout comparison. Neither construction nor
author validation is a GPU correctness or performance test.
