# Matched placement contrasts

`placement_probe.py` executes the three placement contrasts named in
`MODEL_PROTOCOL.md`, each under an explicit eight-group unroll policy:

| Case | Named buffer | Requested threads/block |
|---|---|---|
| `chain32-kvaerno3-stage_base-bs64` | `stage_base` | 64 |
| `chain64-radau5-delta-bs32` | `delta` | 32 |
| `chain32-vern7-stage_accumulator-bs32` | `stage_accumulator` | 32 |

These are fresh experiments. No old placement timing supplies a new
baseline. The manifest records the imported source/compiler epoch, package
versions, harness hashes, algorithm defaults, exact unroll flags, source
grid, probe hash, duration rule, requested geometry and sampling protocol.
Every worker independently reconstructs and checks that manifest.

## Isolation and placement

Three persistent subprocesses own one Solver each: local baseline,
independent local duplicate, and shared treatment. No process constructs
another Solver or inherits a populated registry. Each uses the same
`pl.make_solver`, grid builder and unroll policy. The only changed setting
is the named buffer's location. The generated helper graph is constructed
through cached properties, and actual registry entries must have positive
size and the requested location. The receipt also resolves each entry's
actual owner, shared/persistent slices, local extent and alias parent,
then checks the generated allocator's shared/persistent branch constants.
All such allocators must have zero native overloads during preparation.
For `stage_base`, a declared alias of a local accumulator, a shared request
must resolve to its own shared fallback slice. Input hashes must match
across workers; the two local config hashes must match.

Compilation retains cubins, diagnostic PTX, SASS, config/source/compiler
identities, register count, local frame bytes, shared bytes per run and
occupancy. Diagnostic PTX is the dispatcher's representation of its compile
result; the actual cubin may be linked from LTO IR. Cubin hashes establish
native identity. The local duplicate must reproduce its baseline cubin.
Byte-identical shared/local cubins are explicitly marked as aliases and
count as one physical code identity, while all control timing samples
remain retained. This fresh cohort uses unchanged linker options and
does not install a global diagnostics hook.

The worker records both requested geometry and the production limiter's
result. If the limiter changes 32/64 threads, shared storage exceeds the
device limit, residency is zero, or the grid has fewer than two full
occupancy waves, the case is marked blocked and is not silently relabeled.
An admitted solver uses the verified harness's per-instance `pin_launch`.
The resulting pinned geometry must exactly match requested block size,
dynamic shared bytes, bytes per run, resident blocks/threads and waves;
the pinned residency and two-wave gate are checked explicitly. Every
solve must use one chunk.
The driver occupancy query does not reveal actual shared carveout, so that
field is explicitly unavailable; no guessed L1 capacity is attached.

## Timings and numerical evidence

All solves are serialized across workers. A common duration begins at the
source harness default and doubles until every role's pilot and every
retained measurement is at least 20 ms. The maximum is the source harness's
recorded duration bound. Earlier short attempts stay in the output with
their own attempt identifiers; only a complete accepted attempt supplies
the final cohort. The run count is unchanged and is checked against each
compiled kernel's occupancy.

Each paired block contains all three roles. It first saves full warm state
and raw status arrays, together with shapes/dtypes/hashes. Exact state and
status comparisons, maximum absolute difference, differing values and
differing runs are retained. Nonfinite outputs, status failures, differing
warm results or multiple chunks prevent cohort acceptance. Finite state
is also checked and recorded on every pilot, settle and measurement solve;
these checks use the returned host state and do not extend kernel timing.
No numerical
tolerance is fitted or relaxed. Each role then settles under the same
recorded protocol and provides six measurements (`ROUNDS * REPEATS` from
the frozen harness). The order alternates baseline/shared/duplicate and
its reverse, keeping contemporaneous controls beside each treatment.

`timings.jsonl` preserves every pilot, warm, settling and measurement
kernel/wall timing with role, sample, paired block, attempt, duration,
run count, status, source identity, cubin identity and manifest digest.
`result.json` identifies accepted measurement keys and includes all warm
comparisons and compile receipts. Failures preserve available records and
artifacts. There is no timing alias deduplication or fitted cost/ranking
function. External GPU load and clocks are not measured by this runner.

Workers own their handles as soon as construction starts; construction
failure closes them. Shutdown closes the command stream, waits at most
10 seconds, then escalates through terminate and kill with two bounded
5-second waits. Every worker is cleaned independently before the final
receipt write, so a failed write cannot strand later workers. Cleanup
errors, return codes and forced shutdowns are retained and prevent an
otherwise successful cohort from being accepted.

## CPU preparation and later GPU execution

Use the frozen source/harness tree through external `PYTHONPATH`. The
runner does not edit environment variables or any existing harness.

```powershell
$env:PYTHONPATH='C:\local_working_projects\cubie-worktrees\hardware-epoch-ff3a567f\src;C:\local_working_projects\cubie-worktrees\hardware-epoch-ff3a567f\benchmarks;C:\local_working_projects\cubie-worktrees\hardware-unroll-placement\benchmarks'
$env:CUBIE_CUDA_BACKEND='mlir'
$env:NUMBA_ENABLE_CUDASIM='0'
& C:\local_working_projects\cubie\.venv\Scripts\python.exe benchmarks/hardware_model/placement_probe.py --case chain32-kvaerno3-stage_base-bs64 --policy u11111111 --cohort stage-base-placement-e1 --out C:\local_working_projects\cubie-notes\hardware_unroll_placement\stage_base_placement_prepare_e1
```

Without `--execute`, all three workers perform only host construction and
code generation; each reports zero native overloads. Add `--execute` for
the GPU run, using a new output directory. `--blocks` requests additional
paired blocks, each with the duplicate baseline and six samples per role.
Existing output directories are refused to preserve immutable attempts.

Source contracts:

- `placement_landscape.py:434,449`: algorithm settings and solver creation.
- `placement_landscape.py:176,185`: deterministic parameter grids.
- `placement_landscape.py:339`: named buffer setting mapping.
- `placement_landscape.py:595,705`: compiled occupancy and per-instance pin.
- `BatchSolverKernel.py:821-839`: shared bytes, limiter and launch layout.
- `workload.py:585`: actual per-factory registry entries and aliases.
- `buffer_registry.py:210`: allocator storage branch closure;
  `:334`: deterministic layouts and alias resolution.
- `numba_cuda_mlir/descriptor.py:2449` in the installed backend:
  `inspect_asm` returns PTX from the compile result or its retained LTO IR.

This runner supplies physical-model evidence; it does not supply measured
values to a pre-compile predictor.
