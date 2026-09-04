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
constant. The typed plan retains a `CAPTURED_LOOKUP` with two unresolved native
alternatives: a constant/parameter-memory lookup or a comparison/select tree.

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

The retained CPU evidence constructs Lorenz/Kvaerno3 with direct LU and the
same one-body Newton regime for five policies that vary only
`unroll_accumulator`: full, count 1, count 2, count 4, and `False`. A sixth
count-1 `unroll_stage` diagnostic exercises dynamic two-dimensional tableau
reads and the extended typed lowerer. Every worker reports zero overloads,
native compilations, and kernel launches.

## Model boundary

This layer consumes no timing bank, native label, fitted coefficient, CUDA
runtime, or register observation. Dynamic address arithmetic, captured-table
materialization, native loop replication, scheduling, ABI temporaries, and
cache service remain explicit compiler or hardware alternatives. Its output is
an input to candidate modeling, not a performance ranking.
