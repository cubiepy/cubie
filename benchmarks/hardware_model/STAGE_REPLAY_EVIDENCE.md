# Exact counted-unroll replay evidence

All six corrected Lorenz Kvaerno3/Radau5 BiCGSTAB stage replays reproduce
their original LTO and cubin bytes exactly. The requested Newton counts
survive into the actual libNVVM input of these replays. Radau5's three
distinct LTO blobs produce one identical diagnostic PTX and one original
cubin; Kvaerno3 retains three different outputs at both stages. The
responsible optimization/pass is still unidentified.

The independent CPU audit is
[receipt.json](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/verification/stage_replay_independent_20260905/receipt.json),
with its runnable `audit.py` and exact text diffs alongside it. It
rechecks the original schema-2 cache extraction, six original compile
rows, all replay stage hashes, actual imported source/native inputs and
effective link options. It hashes 712 distinct file identities in the
replay records and also verifies the prepared inventory. The review
imports no CuBIE/compiler package and performs no native compilation or
kernel launch. The six
parent-run observations also report zero kernel launches; these are
compiler observations, not timing or numerical experiments.

## Provenance and reproduced outputs

Raw results are in `hardware_unroll_placement/unroll_stages_native_e2`.
The private observer SHA is
`25b994cf716c99dff176ebc4a4e524103f24357ab407f1f7c74a111746ca6627`;
its prepared manifest SHA is
`5e9e01752763d2d39e583445bb841782c04d5b9ac148bff669220247414aee4e`.
The frozen CuBIE package hash is
`4899b5cb04523177ed3cd3f1aef566591829ed026e064b951fdfbf629cfcef6a`
at epoch `ff3a567f1646a63e70e04c1ab2ea999dc5ac1df4`. Original records are
`lorenz_split_bridge_e1/compiles.jsonl` lines 1/8/9 for Kvaerno3 and
10/17/18 for Radau5. Policies are respectively `u11111111`,
`u11111121`, `u11111141`: Newton full/count-2/count-4, Krylov full.

Every run binds the actual frozen
[_pre_codegen_with_external_shmem hook](/C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f/src/cubie/backend/_mlir_compat.py:283),
installed LLVM70 translator, libNVVM, libdevice, resolved LLVM-C runtime,
and loaded nvJitLink files. Both recorded link invocations use SM89,
optimization level 3, LTO, verbose output and cache bypass; FTZ/FMA are
enabled and precise division/square-root flags are false. The diagnostic
PTX invocation additionally requests PTX output. All six runs have the
same recorded options and compiler input identities.

These sizes are **file bytes**, not native instruction footprints:

| Family / Newton directive | Reproduced LTO | Diagnostic PTX | Original/relinked cubin | Cubin SHA prefix |
|---|---:|---:|---:|---|
| Kvaerno3 full | 23,352 | 1,515,271 | 434,024 | `13c893c1fb4f` |
| Kvaerno3 2 | 23,384 | 407,857 | 121,064 | `e00121621c77` |
| Kvaerno3 4 | 23,384 | 770,668 | 226,920 | `d19a002f2611` |
| Radau5 full | 37,600 | 3,889,140 | 986,984 | `d58264257477` |
| Radau5 2 | 37,632 | 3,889,140 | 986,984 | `d58264257477` |
| Radau5 4 | 37,632 | 3,889,140 | 986,984 | `d58264257477` |

The three Radau5 diagnostic PTX files are byte-identical, with SHA
`af5268045323` as a short identifier; the receipt retains full hashes.
Each family's three LTO files are distinct. All Kvaerno3 diagnostic PTX
and cubin files are distinct.

## Where the requested counts remain visible

Count-2 and count-4 natural replay LLVM differ by exactly **one byte**
within each family: the metadata digit `2` versus `4` in
`!5 = !{!"llvm.loop.unroll.count", i32 ...}`. The locations are:

- [Kvaerno3 natural input, line 3811](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/unroll_stages_native_e2/kvaerno3_count2/nvvm_input/nvvm-53028-0.ll:3811),
  byte offset 148,449.
- [Radau5 natural input, line 5666](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/unroll_stages_native_e2/radau5_count2/nvvm_input/nvvm-79476-0.ll:5666),
  byte offset 238,243.

Against full unrolling, the count variants add the two count-metadata
definitions and change the selected backedge's metadata reference from
`!2` to `!4`. The backedge lines are 3118 for Kvaerno3 and 3610 for
Radau5. The audit removes only metadata definition lines and
`!llvm.loop` reference suffixes: all remaining LLVM text is identical
across each family's three variants. Exact unmodified diffs are retained.
The count request therefore has not disappeared before the replay's
libNVVM input. This also rules out source-level replication differences
at that observed stage: the distinction there is loop metadata.

For all six cases, the separate diagnostic `gen_llvmir=True` result
happens to equal the natural replay input byte for byte. That equality
is measured, not assumed from the inspection API.

## Interpretation boundary

The natural input is from an **exact-output replay**. The original
compilation did not save its libNVVM input, and equal outputs do not
prove equal unsaved inputs. The PTX files come from a separate
original-LTO re-link with PTX output requested; they are not recovered
intermediates of the original cubin-producing link.

On this diagnostic route, Radau5's convergence is visible by PTX output.
Its original cubins independently confirm the native alias. Opaque LTO
byte differences do not show whether they contain different executable
loop transformations or only metadata. These artifacts therefore do
not distinguish optimization during libNVVM LTO production from later
link optimization, and cannot identify the responsible pass. Kvaerno3's
different count-2/count-4 outputs are a direct counterexample to a
general claim that the backend ignores the count.

The earlier failed replay remains intact. Its Radau5 count-2 natural
input differs from the corrected input only at the shared global:
internal linkage with a zero-length initializer becomes external
linkage without that initializer. The restored hook and harness link
options yield exact original outputs without weakening either byte
gate. `radau5_count2_failed_vs_corrected_input.diff` retains this change.

For the pre-compile model, a counted directive remains a source/IR
request, not an observed number of native body copies. Source expansion
facts must retain that uncertainty and the family distinction; these
observations supply no fitted parameter, native register estimate or
instruction-count conversion weight.
