# Implicit policy source graph

`implicit_policy_graph.py` constructs an actual implicit `Solver` with one
explicit eight-group unroll policy and interprets the generated source without
requesting a native specialization. Each construction uses a fresh codegen
cache and restores the prior cache root when it closes the solver.

## Why a separate representation is required

The verified implicit graph accepts only `(True, None)` fixed loops because its
interpreter has no dynamic-index region (`implicit_source_graph.py:451-455`).
The underlying value interpreter likewise requires every array index to be a
proved integer constant (`source_value_graph.py:167-179`) and directly selects
captured arrays with such an index (`source_value_graph.py:261-270`). Feeding a
trace index into that constant environment would therefore misrepresent a
rolled or strip-mined loop as compile-time specialization.

The policy graph instead gives a dynamic induction two separate meanings:

- its source value is a typed `int32` live-in with no `constant` field;
- its declared trace value selects the exact cell and execution path used by
  the finite semantic replay.

Dynamic reads from captured coefficient tables become `CapturedIndexRead`
nodes. The selected FP32 value is a replay witness and is not a source
constant. The typed plan expands admitted four-byte NumPy tables into
conditional indexed `IMAD` and immutable `LDC` operations before allocation.
The installed backend materializes a contiguous constant copy of each captured
view; original view strides and roots remain provenance. See
[captured-table lowering](CAPTURED_LOOKUP_LOWERING.md) for the exact storage,
uniform-index, and native-form assumptions.

Local-array addresses retain row-major byte strides, a source-constant byte
displacement, and each surviving dynamic index value. Scalar promotion requires
constant addressing across the entire captured storage extent, including every
alias. One dynamic access keeps the whole local extent addressable. A dynamic
zero-fill loop therefore remains memory traffic in the nominal form; eliminating
that fill or replacing an array by a register select network needs a separately
proved compiler alternative. A late full-unroll alternative must construct its
own fully expanded source graph rather than reinterpret a rolled trace witness.

`policy_address_lowering.py` models each dynamic index term as a conditional
32-bit `IMAD(index, byte_stride, address)` followed by `LDL`/`STL` or `LDS`/`STS`
using a source-constant displacement. Memory operands consume the resulting
address value, so its dependencies and register lifetime enter allocation.
The witnessed cell offset remains available for memory-sector accounting but
is explicitly distinct from that instruction displacement. Shared allocations
remain addressable. Constant-only local allocations retain exact-cell promotion.
Dynamic negative-index normalization and runtime slice extents are unresolved
source constructs and fail explicitly rather than borrow witnessed geometry.

## Fixed-loop contract

Every intercepted production `unroll_if(range(...), flag)` records a static
template structure and an exact executed-instance sequence. These are distinct
objects:

- `full` has one statically indexed template per trip;
- count 1, 2, or 4 uses the canonical strip-mined main and constant tail from
  `expansion.py:568-590`; main indices remain dynamic whenever more than one
  main chunk executes;
- `False` has one backend-choice dynamic template with the exact trip count;
- a recurrent Newton or Krylov loop records its full-cap static structure but
  executes only the declared body count and exit-vote instance.

Neither source templates nor execution instances are asserted to be native
instruction copies. Counted main/tail structure is an explicit source-model
alternative; the backend may lower it differently.

## Verification

`verify_policy_graph` reconstructs normalized closure flags, main/tail
structures, execution instances, dynamic induction witnesses, address edges,
captured-array bytes and selections, hot-template identities, source edges,
and dependency hashes. It reparses each original `For` and `range` under the
captured owning closure, checks fixed traces in full, and derives recurrent
prefix lengths from the re-evaluated scenario and vote regime. Stable policy
loop IDs bind each execution context and dynamic induction back to its exact
role, call, entry mask, closure flag, and source range. `verify_policy_plan`
reconstructs the complete typed plan from its graph, architecture, compiler
alternative, and materialization.

Role ownership is reconstructed from source-call parentage and lexical stage
indices. Each Newton body must contain its corresponding linear invocation;
the complete role invocation multiset must match the declared scenario. Every
node, control, and helper call site inherits that independently reconstructed
owner, so relabeling several stage contexts together cannot merge two stages.

The cohort certificate replays three deterministic finite input assignments
with typed FP32, int32, uint32, and Boolean operations. It exact-compares every
boundary cell and non-cell observable by type and bits. This permits a fully
expanded graph and a dynamically indexed graph to have different source DAGs
while proving that their selected source executions produce the same outputs.
The certificate describes the selected uniform path; it is not a proof over
all convergence masks or all floating-point inputs.

Source replay follows IEEE FP32 gradual underflow: finite subnormal results
and signed zeros are admitted. Only the underflow notification is suppressed
inside a single FP32 arithmetic evaluation; divide-by-zero, invalid, and
overflow remain errors, and every FP32 arithmetic result must be finite.
[NumPy's error-state context](https://numpy.org/doc/stable/reference/generated/numpy.errstate.html)
restores the caller's error handling after each evaluation. Native typed
plans retain their separately declared FTZ compiler alternative. These source
certificates do not establish equality with native FTZ numerical execution.

Equality and inequality between a proved `None` and a nonoptional numeric
scalar fold to false and true using operand types alone. The rule is limited
to source names and constants and does not admit ordering comparisons or
read an induction witness. Captured-index dependency inventories are sorted
sets; coordinate order remains in the index template used for selection.

The retained CPU evidence constructs Lorenz/Kvaerno3 with direct LU and the
same one-body Newton regime for five policies that vary only
`unroll_accumulator`: full, count 1, count 2, count 4, and `False`. A sixth
count-1 `unroll_stage` diagnostic exercises dynamic two-dimensional tableau
reads and the extended typed lowerer. Every worker reports zero overloads,
native compilations, and kernel launches.

## Model boundary

This layer consumes no timing bank, native label, fitted coefficient, CUDA
runtime, or register observation. Address arithmetic has an explicit conditional
native form; captured-table materialization, native loop replication,
scheduling, ABI temporaries, and
cache service remain explicit compiler or hardware alternatives. Its output is
an input to candidate modeling, not a performance ranking.
# ERK frontend integration

The shared `describe_policy_source` constructor also dispatches actual
`ERKStep` objects through `erk_policy_graph.py`. ERK has distinct explicit
workload and graph kinds, an empty inner-solver scenario mapping, and an
explicit FSAL runtime state. Its counted loops, captured constant tables,
dynamic addresses, typed body and fresh allocation use the common
frontend. See `ERK_POLICY_GRAPH.md` for the source contracts and recorded
dynamic-slice proofs. The author cohort in
`verification/cpu_continuation_independent_20260905/erk_author_e3` contains
60 source-only cases and exact source snapshots; independent review is a
separate gate.


### Exact exceptional intermediate snapshots

Scalar snapshots restore the existing `float_hex` encoding, including
positive/negative infinity and signed zero, with NaN rejected. Exact
FP32 payload comparison admits infinity only for intermediates in a
graph that declares the existing IEEE intermediate domain. Boundary
inputs and observable outputs retain finite-only payload admission.
The Fabbri ANS=1 source replay exercises this distinction without changing
its declared trace values or specializing arithmetic to the replay point.

Independent receipt: `verification/cpu_continuation_independent_20260905/
scalar_snapshot_independent_e2/receipt.json` under the external hardware
model evidence root. It covers eight exact FP32 snapshot cases, strict
boundary/NaN checks and a fresh 23,529-node Fabbri ANS=1 source graph.
