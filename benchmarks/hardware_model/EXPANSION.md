# Captured source expansion

`expansion.describe_expansion(solver, unroll=...)` describes an already
constructed, uncompiled solver. It uses the actual callable graph from
`workload.py`, reads each Python source and captured closure, and applies
restricted AST transformations. `static_descriptors.syntax_counts` supplies
the same unweighted operation vocabulary. No device function is executed,
no native specialization is requested, and no coefficients are fitted.

Each actual callable instance has its own ID, source receipt, complete
supported closure values and closure hash. Array receipts include dtype,
shape, values and a byte hash. Caller arguments and roles join the workload
graph; source path plus definition line joins static descriptors. Distinct
main and smoothed-error solvers retain their separate closures and widths.

The result separates authored syntax, canonical expanded bodies before
folding, and those bodies after proven constant/guard folds. The inventory
sums each captured callable once. It is neither dynamic per-step work nor
an inlined hot instruction working set. Calls remain calls: multiplying
their execution frequency does not replicate a callee's source body.

## Directive semantics

| Directive | Canonical source representation |
|---|---|
| Full | One body per known fixed index; index-dependent constants can fold. |
| Count 1 | One rolled body for multiple trips; its index stays unknown. |
| Count 2 or 4 | Separate main chunk and constant tail, with exact quotient/remainder. |
| False | One explicitly unresolved template; its count is not a size estimate. |
| Newton/Krylov | One recurrent body template, symbolic `N[call,warp]` or `K[call,warp]`, configured cap and requested replication recorded separately. |

For counted loops, the main index is unknown when there are multiple
chunks. A single main chunk has known indices. The tail uses its exact
fixed indices. These are canonical strip-mining facts, **not a promise
about the backend's counted-unroll lowering**. Full, count 1/2/4 and False
remain distinct metadata even where their canonical body counts coincide.
Loop overhead is not priced; branch/loop lowering remains unknown.

For a 35-element loop, count 2 has a two-copy main repeated 17 times and
tail index 34. Count 4 has a four-copy main repeated eight times and tail
indices 32, 33, 34. Only the source copies enter body syntax counts;
dynamic repetitions are separate fields. Nested region counts are
inclusive and must not be summed again into the function inventory.

## Supported proofs and residual cases

The interpreter supports captured scalar/array/tuple values, literal
indices, bounded integer arithmetic, verified scalar casts, comparisons
and short-circuit boolean expressions. It never calls generated code.
Floating comparisons require the same concrete typed floating values in
normal finite range, or an exact-zero predicate. Mixed floating precision
and host-literal promotion remain unresolved with an explicit reason.
Known stage indices can expose tableau elements and resolve stage-copy
guards. Floating arithmetic is retained to avoid assuming target rounding,
contraction or reassociation. Scalar recurrences are invalidated at loop
template entry; post-loop constants require conservative merge rules.

The concrete opportunities are visible at
`generic_firk.py:836-901` (stage-index matrix/weight loads and selected-stage
copy guards) and `generic_dirk.py:867-947` (column coefficients, successor
indices and selected-stage copy guards). A closed zero coefficient does
**not** alone prove that `0 * unknown_float` can be deleted. Current
`JITFlags.fastmath` at `cuda_simsafe.py:235` enables no `nnan` or `ninf`
assumption. Such products are retained and reported by source line,
occurrence count and exact coefficient value.

Unknown calls that receive possibly mutable captures invalidate them
before folding an expression, including keyword arguments and method
receivers. Invalidations survive loop parts and loop exit. Destructuring,
deletion and named-expression targets conservatively lose constant facts.
Unsupported structured control clears the environment and retains
authored syntax. Unknown/dynamic loop bounds, unsupported expressions,
early exits and the False directive are explicit residual records.

No interprocedural side-effect or alias proof is attempted. Captures are
snapshots at extraction; externally mutated captures or hidden callback
side effects are outside this source model. Captured values are checked
for change during extraction, and an edited extractor requires a new
process. Register allocation, spilling, SASS mapping, native dead-code
elimination and hot instruction-cache footprint remain separate evidence.

## CPU validation and use

The verification matrix uses real generated Lorenz and Fabbri sources,
Radau IIA 3/5 and Kvaerno3 DIRK, and all five stage directives. Additional
Fabbri Radau5 element-count 2/4 cases check exact 35-element tails. Lorenz
Radau5 BiCG checks distinct Newton requests with caps 8 (Newton), 14
(main Krylov, width 9) and 5 (smoothed-error Krylov, width 3). Each recurrent
region retains one body, and every inspected dispatcher has zero native
overloads. Radau5 Fabbri preserves widths 105 and 35; Radau3 has width 70.

The step callable's canonical source inventory is below. These totals
only count syntax items; each category remains separate in JSON. Full
and rolled counts are different source representations, not measures of
their relative speed or dynamic operation totals.

| System / family | Full, before → after folds | Count 1, before → after folds |
|---|---:|---:|
| Lorenz / Radau3 | 368 → 123 | 212 → 97 |
| Lorenz / Radau5 | 572 → 226 | 212 → 131 |
| Lorenz / Kvaerno3 DIRK | 729 → 321 | 349 → 157 |
| Fabbri / Radau3 | 3568 → 1145 | 1876 → 831 |
| Fabbri / Radau5 | 5820 → 2240 | 1876 → 1121 |
| Fabbri / Kvaerno3 DIRK | 6873 → 3263 | 3165 → 1371 |

Fabbri Kvaerno3 full stage expansion exposes nine column coefficients at
`generic_dirk.py:879`. Three are zero, producing 105 retained zero-product
sites at line 882 across 35 elements. Count 1 leaves that stage index
unknown. This is a concrete coefficient-visibility difference, without
assuming that the backend deletes those products.

Authoritative CPU receipts are the 35 reports and `summary.json` in
`C:\local_working_projects\cubie-notes\hardware_unroll_placement\expansion_validation_v4`.
Every final descriptor, excluding extractor provenance, is also checked
equal to the corresponding corrected v3 descriptor. Earlier directories
are authoring checkpoints and do not supply final source provenance.

Run from the research worktree with its `src` and repository root on
`PYTHONPATH`, real-backend imports enabled, and a separate codegen cache:

```powershell
$env:CUBIE_CUDA_BACKEND = 'mlir'
$env:NUMBA_ENABLE_CUDASIM = '0'
$env:CUBIE_CACHE_DIR = 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\expansion_probe_codegen'
python -m benchmarks.hardware_model.expansion --system fabbri --algo radau_iia_5 --unroll '{"unroll_stage":[true,2]}' --output expansion.json
```

The JSON preserves per-category counts rather than converting them to a
single physical cost. The canonical AST representation cannot explain why
fresh native Radau5 full/Newton-count2/Newton-count4 cubins are identical;
that remains a backend-lowering observation to compare against this
separate source evidence.
