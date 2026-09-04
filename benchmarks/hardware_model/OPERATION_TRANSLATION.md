# Source-operation translation experiment

This instrument measures the one permitted calibration: how a specified
source expression DAG lowers to native instructions on the installed
compiler and GPU. It does not fit solve times, algorithm-family weights,
register multipliers, cache penalties, or latency constants.

## Why source syntax is insufficient

The current [IR printer](/C:/local_working_projects/cubie-worktrees/hardware-unroll-placement/src/cubie/odesystems/symbolic/engine/printer.py:199)
emits add/multiply syntax and converts negative powers into division.
Its [call and Piecewise printing](/C:/local_working_projects/cubie-worktrees/hardware-unroll-placement/src/cubie/odesystems/symbolic/engine/printer.py:303)
uses math functions and eager `selp`. The
[JIT flags](/C:/local_working_projects/cubie-worktrees/hardware-unroll-placement/src/cubie/cuda_simsafe.py:181)
enable contraction, reciprocal approximation, approximate functions,
flush-to-zero and signed-zero relaxation by default. Native FFMA versus
separate multiplication/addition depends on the consumer DAG and flags;
division and transcendental lowering can include special-case paths.
Index lowering also depends on integer width and signedness.

## Stages and evidence

`operation_translation.py emit` imports only the standard library.
It emits inspectable source, per-fragment and whole-kernel AST counts,
DAG conditions, flag overrides, and source hashes. Counts default to
1/2/4/8. Categories are add, multiply, single-use FMA, multi-use FMA,
divide, sqrt, exp, log, eager select, and signed32/unsigned32/signed64
index arithmetic. The 64-bit index category contains integer arithmetic,
not FP64 calculations.

All arithmetic inputs are runtime float32 kernel parameters. Recurrences
change the next operation's input; source stores keep the result live.
The multi-use product feeds two live additions. The select fragment
contains a comparison and two eager additions, making repeated selects
non-idempotent in general. The report labels that entire DAG rather than
pretending its full cost belongs to a single select operation. Compiler
folding and CSE remain possible and must be detected in native evidence.

`compile` explicitly starts a generated worker with imports from the
installed CuBIE CUDA backend. It reuses the existing compilation,
manifest and cubin helpers. It specializes with host signature arguments,
disassembles the cubin, and records PTX, SASS, used registers, local/shared
allocation, compiler flags, versions and GPU identity. It never allocates
device arrays, launches kernels, or measures execution time. Compilation
requires the orchestrator's released GPU slot because the CUDA driver
and compiler are involved. The worker currently restricts encoding
analysis to SM89.

The signature values supply types, not compile-time constants. They
are not safe execution fixtures for repeated exp/log composition.
These fragments are compile experiments; their exceptional dynamic
math paths are not measured by this instrument.

`analyze` is CPU-only and retains full opcode suffixes, all native text
sections, addresses, constant operands, predicates, direct controls,
resolved local targets and memory instructions. It records static
library/call footprint without counting every static path as executed.
Constant loads (`LDC` and `ULDC`) are memory instructions in the
invariance check. Constant operands of arithmetic instructions remain
separately visible in the raw evidence.
Every entry contains setup, output addressing and stores; these are
explicitly included in the raw totals. Sixteen bytes per SM89 SASS
instruction comes from NVIDIA's
[Binary Utilities documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/).

## Conditional attribution

For at least three distinct positive counts, the analyzer reports exact adjacent
opcode-count differences per additional source fragment. It labels a
group `static_candidate` only if increments are identical, nonnegative
integers and nonzero, resources and memory-opcode counts stay unchanged,
and there are no predicates, nontrivial control flow or extra sections.
Padding NOPs and a proven unreachable compiler footer are excluded from
the candidate calculation but retained in the full static footprint.
The footer exception requires an unconditional `EXIT` after a prefix
without branch/call/return/reconvergence instructions; its tail may
contain only NOPs and unconditional branches to their own addresses.
This is a conservative local proof, not general CFG reconstruction.
All other controls and predicates prevent candidate attribution.
A zero-count kernel is diagnostic only;
it cannot justify baseline subtraction.

Eligibility requires matching emitted case membership, a saved exact
source manifest, the compiled manifest and worker hashes, and matching
per-case source, cubin, PTX, and SASS hashes. Saved analyses are checked
against a fresh parse of their SASS. Every row binds to one compiler,
source/helper, device/driver and disassembler identity; effective JIT
arguments must be identical across the compared counts. Case identity
receipts list each check. Missing or inconsistent provenance prevents
attribution while retaining raw differences. Duplicate counts produce
an explicit undefined-span diagnostic instead of division by zero.
The compiled identity includes the actual imported CuBIE package source
hash/root, toolchain fingerprint, active/requested scheduler, source
operation-ordering default, JIT defaults, relevant compiler environment,
Numba/backend package versions, and paths/hashes of the imported CUDA
facade, compiler helpers and parser. It does not infer import origins
from the research checkout's Git status. Effective defaults are checked
again before every compilation; per-case lowered JIT arguments remain
part of the group eligibility check.

A candidate is a conditional observation for the emitted DAG, flags,
compiler and GPU. It is not automatically a universal per-operation
coefficient. In particular, stable register counts do not prove identical
register allocation or scheduling. Changes in resources, folding,
library control flow or opcode increments produce `context_dependent`
with the raw differences retained. No regression is performed. Dynamic
paths, invocation frequency, and liveness remain separate measurements.

Profiles are `default`, `no_contract`, `no_arcp`, `no_afn`, and `strict`.
Use the single-flag contrasts on the relevant categories; the default
48-case construction contains only the default profile. Full flag
manifests must accompany any translation used by a model. LTO stays at
the installed project's default and is recorded.

## Commands

CPU-only construction from the research worktree:

```powershell
& 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.operation_translation emit --output 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\operation_translation_sources_20260904b'
```

For a targeted contraction contrast, emit another source dataset using
`--categories fma,fma_multi_use --profiles default,no_contract`.
For division use `--categories divide --profiles default,no_arcp`;
for math use `--categories sqrt,exp,log --profiles default,no_afn`.
Use fresh output directories for every emitted or compiled dataset.

Only after the compiler slot is released:

```powershell
$env:PYTHONPATH = 'C:\local_working_projects\cubie-worktrees\hardware-unroll-placement\src'
& 'C:\local_working_projects\cubie\.venv\Scripts\python.exe' -m benchmarks.hardware_model.operation_translation compile --source 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\operation_translation_sources_20260904b' --output 'C:\local_working_projects\cubie-notes\hardware_unroll_placement\operation_translation_compiled_20260904'
```

The compile worker saves its exact script, command and compiler log.
Compilation errors remain per-case records and make the process fail.
Reanalysis of saved results is `analyze --directory COMPILED_DIRECTORY`.

## Current validation boundary

The reviewed translator compiled all 48 cases from the source-only
`operation_translation_sources_20260904b` dataset successfully, with
zero kernel launches. An independent CPU audit re-disassembled every
cubin and checked source, toolchain, artifact, opcode and resource
identities. [OPERATION_TRANSLATION_EVIDENCE.md](OPERATION_TRANSLATION_EVIDENCE.md)
records the five conditional instruction-vector candidates and the
seven contexts that fail the complete-cost attribution gate. This
validates neither runtime latency nor a register predictor. The first
`operation_translation_sources_20260904` emission remains an uncompiled,
superseded draft with unexported scalar imports.
