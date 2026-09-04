# Observed source-to-native instruction translation

The 2026-09-04 bank contains 48 successful compilations: twelve source
fragments, each repeated 1/2/4/8 times under the default flags. It contains
zero kernel launches. The [independent CPU audit](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_audit_20260904.json)
recomputed all source/cubin/PTX/SASS hashes, re-disassembled all 48 cubins,
and independently counted addresses, opcodes, registers and the terminal
footer. Every result matched its saved analysis, and every adjacent
opcode-count delta matched [translation.json](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/translation.json).
No compiler or GPU execution was invoked by the auditor.

## Compiler and source conditions

The [compiled manifest](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/manifest.json)
binds the bank to RTX 4070 SUPER/SM89, driver 610.62, Python 3.14.3,
Numba 0.67.0, `cubie-numba-cuda-mlir` 0.5.1.1, llvmlite 0.49.0,
NVVM/CUDA 13.3.73 and nvdisasm 13.3.73. Actual imported CuBIE source is
the frozen `hardware-epoch-ff3a567f/src/cubie`, whose complete Python-file
hash was independently recomputed as
`4899b5cb04523177ed3cd3f1aef566591829ed026e064b951fdfbf629cfcef6a`.
It uses `anchor_dfs`, `liveness_auto` and identical effective JIT arguments
in all 48 cases:

```json
{"fastmath":["nsz","afn","contract","arcp","ftz"],
 "lineinfo":false,"lto":true,"experimental_ast_transforms":true}
```

The executed translator SHA256 is
`fd0360f57a80379f3cb65a840a6a279058598c2fb613b05212d5b8b745cec6ec`;
the compiled manifest SHA256 is
`201c933bf0aec40cb23a8f747334d01187d24e059ddf0c61f7903649502ec355`.
The worker, saved emitted manifest, imported helper files and each exact
source copy were checked separately. The earlier emission generator hash
differs from the compiled translator hash; the audit does not conflate
these versions. It validates the saved emitted source against the exact
compiled source instead.

Inputs are runtime scalar kernel parameters, uniform across lanes.
Only the output address uses `cuda.grid(1)`. Floating arithmetic is FP32;
the signed64 case is integer arithmetic. The compile-only worker uses
host arrays to supply specialization types, then `compile_for`; it does
not allocate device arrays or execute the recurrences. Exceptional-input
numerics and latency are therefore unmeasured.

## Exact conditional opcode vectors

Each delta is the native opcode-count difference divided by the number
of added source fragments, at spans 1→2, 2→4 and 4→8. Setup, output
addressing and stores remain in each entry's counts. Only NOP padding and
the independently verified unreachable self-branch footer after EXIT are
excluded from the delta. All 48 entries contain one native text section,
no conditional instruction predicates or reachable branch/call, and
reported zero local bytes and zero static shared bytes. `FSEL` consumes
a predicate; it remains an executed instruction in this straight stream.

The table reports the full deltas, including residual changes defined
below. Register counts are for the complete compiled entry at 1/2/4/8,
not a per-fragment register model. A `candidate` satisfies the conservative
tool gate: stable integer opcode increments, memory counts and resources.

| Source fragment | Native delta per added fragment | Registers at 1/2/4/8 | Gate |
|---|---|---|---|
| `x = x + y` | `FADD.FTZ:1` at every span | 8/8/8/8 | candidate |
| `x = x * y` | `FMUL.FTZ:1` at every span | 8/8/8/8 | candidate |
| `x = x*y + z`, one product consumer | `FFMA.FTZ:1` at every span | 10/10/10/10 | candidate |
| Product feeds two live additions | `FFMA.FTZ:2`, plus `MOV:1` only at 1→2 | 10/12/14/14 | context dependent |
| `x = x / y`, invariant denominator | `FMUL.FTZ:1` at every span | 9/10/10/10 | context dependent |
| `x = float32(math.sqrt(x))` | `MUFU.SQRT:1` at every span | 10/10/10/13 | context dependent |
| `x = float32(math.exp(x))` | `FMUL.FTZ:1, MUFU.EX2:1` at every span | 10/10/11/11 | context dependent |
| `x = float32(math.log(x))` | `MUFU.LG2:1, FMUL.FTZ:1` at every span | 8/8/10/11 | context dependent |
| `selp(x>z, x+y, x+w)` | `FSETP.GT.FTZ.AND:1, FSEL:1, FADD.FTZ:1`, plus S residuals | 8/9/10/10 | context dependent |
| Signed32 affine index recurrence | `IMAD:1` at every span | 10/10/10/10 | candidate |
| Unsigned32 affine index recurrence | `IMAD:1` at every span | 10/10/10/10 | candidate |
| Signed64 affine index recurrence | `UIMAD:2, UIMAD.WIDE.U32:1, UIADD3:1`, plus I residual at 4→8 | 10/10/10/8 | context dependent |

The complete residual vectors are:

| Residual | 1→2 | 2→4 | 4→8 |
|---|---|---|---|
| S, eager select | `IMAD.MOV.U32:+1, MOV:-1` | `IADD3:+1/2, IMAD.IADD:-1/2, IMAD.MOV.U32:+1/2, MOV:-1/2` | `IADD3:-1/4, IMAD.IADD:+1/4, IMAD.MOV.U32:+1/4, MOV:-1/4` |
| I, signed64 | zero | zero | `IADD3:+1/4, IMAD.IADD:-1/4, IMAD.MOV.U32:-1/4, IMAD.U32:+1/4, MOV:-1/4` |

These fractional entries describe finite differences between whole
kernels; a native operation never costs a quarter of an instruction.
They must not become fitted per-operation constants.

## Why the context-dependent cases differ

For the [multi-use source](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/fma_multi_use_default_n1/kernel.py:9),
`product=x*y` feeds `x=product+z` and `w=w+product`; the output stores
`x+w`. The [one-fragment SASS](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/fma_multi_use_default_n1/kernel.sass:28)
contains two FFMAs sharing the multiplication inputs and one final FADD.
The compiler contracts each add and duplicates multiplication inside the
two FFMAs; there is no separately retained FMUL. At n=1/2/4/8 the actual
arithmetic counts are respectively FFMA=2/4/8/16 and FADD=1 throughout.
The [two-fragment SASS](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/fma_multi_use_default_n2/kernel.sass:20)
also materializes `z` in an additional MOV. MOV counts are 4/5/5/5, and
whole-entry registers change. The two-FFMA component is observed exactly,
but the full expansion cannot be modeled as a constant total cost per
multi-use source fragment. Nor may one count the source multiply once
and add a separate universal FMA allowance on top.

For signed64, the [source](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/index_i64_default_n1/kernel.py:9)
computes `int64(i*stride+offset)` from uniform runtime parameters.
The [PTX](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/index_i64_default_n1/kernel.ptx:38)
loads three 64-bit parameters and performs low-64-bit multiply/add.
The [native body](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/index_i64_default_n8/kernel.sass:20)
uses uniform registers: low-word wide multiply, two cross-word UIMADs,
and UIADD3 form four instructions per recurrence. The n=8 case changes
output-address and uniform-to-thread-register transfer instructions:
the [IADD3](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/index_i64_default_n8/kernel.sass:45)
replaces the shorter entries' IMAD.IADD, and
[final transfers](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/index_i64_default_n8/kernel.sass:61)
change MOV/IMAD variants. Those observed surrounding changes produce the
fractional residual; they are not a fractional recurrence instruction.
The fall from 10 to 8 thread registers does not measure uniform-register
use or establish a register rule for lane-varying indices loaded from
arrays. This uniform-input experiment cannot assign the same UIMAD
vector to those indices without evidence of their lowering.

Division exposes another DAG condition: the denominator is invariant.
The [n=8 native body](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/divide_default_n8/kernel.sass:18)
contains **one MUFU.RCP followed by eight FMUL.FTZ**, rather than eight
reciprocals. The increment of one FMUL therefore cannot mean division
generally equals multiplication. Sqrt emits one MUFU.SQRT per call.
Exp emits multiplication by the recorded FP32 log2(e) constant followed
by MUFU.EX2; log emits MUFU.LG2 and multiplication by FP32 ln(2).
Those constants are mathematical conversion constants, not fitted costs.
No library fallback sections occur in this default-flags bank; it says
nothing about `no_afn`, `no_arcp`, `no_contract` or strict lowering.

For eager select, [SASS](/C:/local_working_projects/cubie-notes/hardware_unroll_placement/operation_translation_compiled_20260904/select_default_n2/kernel.sass:22)
selects the addend first and adds the common `x` once. The two source
additions therefore lower to one FADD for this DAG. The per-span changes
in MOV/IMAD forms and whole-entry registers remain visible in the
residual vector; raw syntax counts would overcount its arithmetic.

## What a post-codegen model may use

The five candidate vectors are conditional instruction-count templates
for the matched typed DAG and recorded toolchain/flags. They provide the
per-instruction-type calibration permitted by the research specification;
they do not authorize a scalar timing multiplier or arbitrary family
coefficient. Every application must first account for product consumers,
constant/identity folds, repeated denominators, integer width and
uniformity. Unsupported contexts stay explicit rather than borrowing a
coefficient from a superficially similar syntax operation.

The other rows preserve exact observed arithmetic components and whole
kernel deltas. Their context-dependent classification is retained; no
universal complete-cost coefficient is accepted. Register counts,
scheduling, copy instructions, memory traffic and hardware latency remain
separate quantities. This bank validates no pre-compile register estimate.

At SM89, each encoded instruction occupies 16 bytes. Multiplying an
accepted opcode count by 16 describes that instruction component, not
the entire function footprint or its hot instruction-cache working set.
For example, add has 272/288/320/384 bytes through EXIT, but its complete
encoded sections occupy 512/512/512/640 bytes after footer and padding.
The audit retains both sizes for every case. Setup, helper inlining,
control structure, alignment and dynamic call frequency must be handled
separately when translating generated solver code.
