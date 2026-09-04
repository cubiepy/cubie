# Typed implicit source regions

`implicit_source_graph.py` expands one actual implicit solver step into a
typed value, memory, and control graph before native specialization.  It
uses the actual generated helper registry and the role mapping from
`implicit_workload.py`.  This connects the family-specific Newton, LU,
minimal-residual, and BiCGSTAB source to a typed native-lowering model
without replacing those paths with a generic iteration multiplier.

## Admission boundary

The initial admitted regime is a uniform successful path.  Every step call
has an explicit entry mask.  Newton and Krylov bodies have explicit active
masks and a terminal inactive mask.  `evaluate_regime` first validates the
family's actual call structure, widths, caps, and common step-entry mask.
An `AllSync` node is admitted only when its participating mask equals the
active entry mask for that call.  Every runtime branch outside those votes
requires a source-line keyed choice.  The choice becomes a required
`BranchDecision` root and an ordering predecessor of operations on the
selected path.  It is recorded as a runtime-path assumption, never as a
code-generation constant.

Fixed loops are source-expanded only when their captured directive is
exactly full unroll.  A recurrent Newton or Krylov loop emits the requested
dynamic bodies plus its terminal vote.  Its `hot_template_identity` excludes
the dynamic body index while retaining fixed loop indices.  Repeating a
template in the execution trace therefore does not assert that the compiler
made another native code copy.  Partial fixed loops, nonuniform paths, and
unproved vote masks reject explicitly.

The graph retains the source spelling and callable identity for `abs`,
`math.fabs`, `sqrt`, `min`, and `max`.  In particular, Python `min` and
`max` are not identified with `fmin` and `fmax`.  The selected-path contract
requires finite FP32 inputs and intermediates.  It records FP32 source
operations and leaves native reassociation, contraction, and instruction
choice to the next lowering layer.

Integer nodes use an int32 storage-compatible source abstraction.  Casts,
selects, bitwise operations, and counter arithmetic retain their typed value
edges; arithmetic and bitwise operations conserve the low 32 bits.  The
actual compiler may carry an intermediate in a wider integer type before a
store or call boundary, so `native_integer_width` remains unresolved.  A
bare typing-context overload lookup is insufficient evidence for the
specialized kernel typemap.  The downstream lowerer must keep the width
alternatives explicit or join an observed compiler typemap.

## Machine-readable contract

The JSON record contains:

- the complete `implicit_workload` descriptor and declared regime;
- actual source-function IDs, source hashes, closures, helper calls, and
  buffer-registry allocations;
- typed SSA values and nodes with separate value and ordering edges;
- exact scalar memory cells, aliases, reads, writes, and final versions;
- runtime-region metadata for role, call, body index, phase, and entry mask;
- stable source-template identities that are separate from execution
  instances;
- required branch-control roots and declared vote results;
- per-call lexical certificates with live-ins and conservative caller
  live-through outputs; and
- operation signatures counted by input and output dtype.

Caller-boundary cells remain conservative exit observables.  Their presence
is an explicit source-model boundary and does not prove that the full native
kernel keeps every value alive.  A whole-kernel caller-use/kill analysis can
narrow that boundary only by proving the actual later uses, aliases, and
overwrites.

## CPU construction evidence

Nine actual Lorenz constructions were generated in isolated cache roots with
zero dispatcher overloads.  Each uses one Newton and one Krylov body where
the role exists.  Kvaerno3 uses an explicit first-step path choice; its MR
path also declares nonzero-denominator and nonconverged body branches.

| Family / linear solver | Nodes | Values | Branch roots | Role-node counts |
| --- | ---: | ---: | ---: | --- |
| Kvaerno3 / LU | 1,141 | 1,076 | 7 | Newton 621; linear 96 |
| Kvaerno3 / MR | 1,623 | 1,602 | 19 | Newton 624; linear 633 |
| Kvaerno3 / BiCGSTAB | 3,141 | 3,207 | 16 | Newton 624; linear 2,151 |
| Radau IIA 5 / LU | 1,565 | 2,201 | 2 | Newton 563; main 293; error 37 |
| Radau IIA 5 / MR | 2,032 | 2,817 | 10 | Newton 564; main 646; error 295 |
| Radau IIA 5 / BiCGSTAB | 4,015 | 4,960 | 7 | Newton 564; main 2,115; error 809 |
| Rosenbrock23 / LU | 695 | 584 | 0 | linear 282 |
| Rosenbrock23 / MR | 1,307 | 1,226 | 12 | linear 894 |
| Rosenbrock23 / BiCGSTAB | 2,855 | 2,918 | 6 | linear 2,442 |

The retained records are under
`hardware_unroll_placement/implicit_source_graph_cpu_e17`.  The smallest
complete example is `kvaerno3_lu/graph.json`, SHA-256
`045465fbd0e757b4001c8ce6331134ca63073502bc30301f7d8ff5277302c51c`.
The other graph hashes and full typed inventories are recorded in
`verification/implicit_source_graph_20260905/receipt_e17.json`.

The independent CPU validator rehashes all frozen source functions, checks
every producer/consumer and ordering edge, recomputes each per-call cut,
replays exact memory-version ordering, binds final cells, recomputes template
identities and typed inventories, and validates branch roots, vote masks,
role entry, and recurrence/code-copy separation.  The nine-graph receipt has
status `TYPED_IMPLICIT_SOURCE_VALIDATION_PASS`.  This construction executes
no device function and requests no native specialization.

A separate Kvaerno3/MR trace with two Newton and two Krylov bodies contains
3,813 nodes.  Of 984 distinct hot templates inside dynamic solver regions,
981 occur at more than one dynamic body index while retaining the same
identity.  The exact instances are in
`verification/implicit_source_graph_20260905/iteration2_template_reuse.json`;
the source graph is SHA-256
`0a6e37345f73a45c14c37f4e89a34c07099e6754cdbbb9bd25e27940972f42f2`.

## Reproduction

Set `PYTHONPATH` to the frozen production `src` followed by the research
worktree, and set `NUMBA_ENABLE_CUDASIM=0`.  A representative construction is:

```powershell
python -m benchmarks.hardware_model.implicit_source_graph `
  --system lorenz --algo kvaerno3 --linear-solver lu `
  --newton-bodies 1 --krylov-bodies 1 `
  --branch-choices <choices.json> --output <fresh-directory>
```

The output directory must be fresh.  The tool installs that directory as a
temporary public code-generation cache root, restores the previous root in a
`finally` block, closes the solver, and fails if any captured dispatcher has
an overload.
