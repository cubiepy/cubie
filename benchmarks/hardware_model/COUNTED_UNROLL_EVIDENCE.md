# Counted Newton unroll: retained compiler evidence

CPU inspection on 2026-09-05 found the original optimized MLIR and
LTO-IR inside the fresh Lorenz bank's compilation cache. The requested
Newton counts survive into optimized MLIR. The Radau5 full/2/4 cubin
alias therefore does not establish that the frontend ignores counts.
The specific downstream optimization causing native convergence remains
unidentified by the retained artifacts.

The [extraction receipt](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/counted_unroll_cache_audit_20260904/receipt_v2.json)
binds original cache bytes, literal offsets, MLIR literal/export hashes,
compile records, cubin hashes and installed compiler file hashes. Only
`pickletools.genops` literal data was read. No pickle reconstruction,
CuBIE import, CUDA compilation, linking or GPU execution occurred.

The cached MLIR string contains LF newlines; Windows `write_text` exported
CRLF. Schema 2 records both `mlir_literal_utf8_sha256` and
`mlir_export_file_sha256`, and verifies that replacing export CRLF with LF
recovers the literal exactly in all six cases. LTO exports are byte-exact.
The original receipt is retained with its hash; its `mlir_sha256` field
referred to literal UTF-8 bytes, not the exported file bytes. No cache or
export was modified by this clarification.

## Measured native identities

These are Lorenz with BiCGSTAB, all other unroll groups full, source
`4899b5cb04523177ed3cd3f1aef566591829ed026e064b951fdfbf629cfcef6a`,
MLIR wheel 0.5.1.1, `anchor_dfs`, and the bank's unchanged fast-math/LTO
settings. Each recorded compilation has `cached=False`.

| Method | Newton directive | Compile seconds | SASS instructions | Registers | Cubin SHA prefix |
|---|---|---:|---:|---:|---|
| Kvaerno3 | full | 5.21 | 26,320 | 96 | `13c893c1fb4f` |
| Kvaerno3 | count 2 | 2.51 | 6,912 | 98 | `e00121621c77` |
| Kvaerno3 | count 4 | 3.22 | 13,480 | 96 | `d19a002f2611` |
| Radau5 | full | 20.04 | 60,920 | 167 | `d58264257477` |
| Radau5 | count 2 | 19.07 | 60,920 | 167 | `d58264257477` |
| Radau5 | count 4 | 18.41 | 60,920 | 167 | `d58264257477` |

The exact records are [compiles.jsonl](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/lorenz_split_bridge_e1/compiles.jsonl:1)
lines 1/8/9 for Kvaerno3 and 10/17/18 for Radau5. Extracted cached cubin
bytes independently match these records. Instruction totals describe
whole native kernels; they are not Newton-body copy counts.

## The directive reaches the Newton backedge

The frozen [Newton builder](/C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f/src/cubie/integrators/matrix_free_solvers/newton_krylov.py:302)
casts `max_iters` to int32, captures `unroll_newton_exits` at line 325,
and uses it on `range(max_iters)` at line 400. Both retained MLIR kernels
have a cap of eight, not a cap of two or four: Radau's constant appears
at [line 58](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/counted_unroll_cache_audit_20260904/radau5_count2.mlir:58)
and initializes the Newton iterator at lines 1664/1668; Kvaerno's
constant is at [line 59](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/counted_unroll_cache_audit_20260904/kvaerno3_count2.mlir:59).
Both Newton loops retain a warp vote before their dynamic exit.

The lowering path in the installed code is explicit:

1. CuBIE's [AST rewrite](/C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f/src/cubie/backend/_mlir_cubie_extensions.py:123)
   emits `cuda.unroll(iterable, count)` with a literal count. Full omits
   the count; False emits a plain iterable. It attaches a hint and does
   not clone the Python loop body.
2. Installed [typing](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/typing/cuda.py:1061)
   requires a positive IntegerLiteral bounded by the i32 metadata range
   and preserves it in the intrinsic signature.
3. Installed [lowering](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/lowering/cuda.py:1587)
   creates `#llvm.loop_annotation<unroll = <count = k>>`, or `full = true`.
   Iterator propagation records the loop header in
   [mlir_lowering.py:1527](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_lowering.py:1527);
   [line 3068](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_lowering.py:3068)
   attaches that annotation to the backedge's `llvm.br`.

This is confirmed in actual saved kernels, not just inferred from code:

| Method | Count annotation definition | Annotated Newton backedge |
|---|---|---|
| Kvaerno3 count 2/4 | `#llvm.loop_unroll<count = 2/4 : i64>` | `llvm.br ^bb122`, line 2977 |
| Radau5 count 2/4 | `#llvm.loop_unroll<count = 2/4 : i64>` | `llvm.br ^bb113`, line 3494 |

See the extracted [Kvaerno3 count-2 MLIR](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/counted_unroll_cache_audit_20260904/kvaerno3_count2.mlir:2977)
and [Radau5 count-2 MLIR](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/counted_unroll_cache_audit_20260904/radau5_count2.mlir:3494).
The printed MLIR attribute's `i64` is not evidence of an invalid final
LLVM metadata operand; the later translation was not inspected here.

Within each family, full versus count 2 differs in exactly two added
metadata definitions and the annotation on that one backedge. Count 2
versus count 4 differs only in the count literal. Removing those metadata
differences makes all three optimized MLIR texts byte-identical.
The saved [Radau diff](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/counted_unroll_cache_audit_20260904/radau5_full_count2.diff)
and [Kvaerno diff](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/counted_unroll_cache_audit_20260904/kvaerno3_full_count2.diff)
retain the complete changes. At this boundary there are no additional
body copies or source-arithmetic differences between these directives.

## Where the evidence ends

The installed [cache serializer](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/caching.py:53)
retains optimized MLIR, LTO-IR and cubin independently. All six cache
entries contain distinct LTO-IR byte hashes; their PTX fields are empty.
Distinct LTO-IR bytes do not prove distinct executable bodies: metadata
alone can distinguish compiler IR. No readable pre-libnvvm LLVM IR or
post-LTO PTX was retained with these six cases.

Installed [optimization](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_optimization.py:764)
saves optimized MLIR before pre-codegen patterns. Those
[patterns](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/optimization/__init__.py:234)
handle argument attributes, float storage, fast division and zero powers;
their Python implementation does not rewrite loop annotations. SM89
selects the [LLVM70 translation path](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_optimization.py:102).
The LTO path invokes that translator at line 809, then nvJitLink completes
the native binary at line 832. The wheel supplies the translator as a
DLL; this audit records its hash rather than substituting upstream code.

The established localization is therefore: the count survives the
retained optimized MLIR; native convergence occurs somewhere after that
boundary. The artifacts do not distinguish metadata translation/loss,
libnvvm unrolling, later loop transformations, or nvJitLink codegen.
They also do not establish that Radau's native Newton loop was fully
expanded merely because it equals the full-request binary.

## Discriminating compiler-only experiment

Use the six saved kernels as exact controls, with the original SM89
target, JIT flags, linking protocol and imported compiler/source hashes.
No timing or kernel launch is needed.

1. Capture architecture-natural LLVM IR and the exact IR fed to libnvvm.
   The installed `get_llvmir` starts at
   [mlir_optimization.py:485](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_optimization.py:485);
   `NUMBA_CUDA_MLIR_DUMP_NVVM` capture is implemented at lines 256/325.
   Inspect the Newton backedge, metadata width/value, loop cap and exits.
   If the directives already coincide here, the loss precedes native
   optimization. Otherwise that explanation is excluded.
2. Retain the existing LTO-IR and obtain diagnostic post-LTO PTX through
   [get_lto_ptx](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_optimization.py:666),
   keeping its separate diagnostic-link provenance. Compare loop bodies,
   backedges and convergence exits, not whole-file hashes alone. Equal
   Radau bodies here localize convergence before final SASS emission;
   distinct PTX bodies with identical native cubins localize it later.
3. If pass attribution remains unresolved, reduce the retained Newton
   region to a cap-eight runtime float32 recurrence with the same
   warp-voted early exit and a nested counted workload. Compare full,
   count 1/2/4 and False, then remove the nested workload or early exit
   one at a time. A reduction must reproduce Radau's alias while its
   Kvaerno control retains count sensitivity before assigning a cause.

This is a diagnostic protocol, not an executed experiment. Inspection
methods can invoke native compilation or linking; none were called in
this read-only audit.

For the pre-compile model, full/count 1/2/4/False remain distinct inputs.
Requested source expansion and provable folds are useful descriptors;
they are not guaranteed native-body replication counts. Newton/Krylov
dynamic work remains symbolic, and no fitted multiplier is introduced
to force these two families onto one expansion rule.
