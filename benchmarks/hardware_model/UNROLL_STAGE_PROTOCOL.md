# Counted-unroll stage observation

`unroll_stages.py` prepares the six exact Lorenz Kvaerno3/Radau5
full/count-2/count-4 controls from the original cache audit. Preparation
and verification are CPU-only standard-library operations. The separate
`observe` invocation requires `--execute-native`; it compiles and links
one saved control, without launching a kernel. Native observation is
subject to the root agent's serialized GPU/compiler slot.

The existing [counted-unroll evidence](COUNTED_UNROLL_EVIDENCE.md)
establishes that the requested counts survive optimized MLIR. Radau5's
three distinct LTO blobs converge to one cubin; Kvaerno3's remain three
different cubins. The native optimization responsible is unresolved.

## Exact controls and boundaries

The preparer interprets `pickletools` opcodes as inert data. Serialized
globals, reducers, classes and object construction become symbolic
records, never executable Python objects. Only literal target options
and the recognized FastMathOptions flag set are accepted for replay.
The original cache, compilation row, compiler identity, MLIR literal,
LTO, cache cubin and independently saved bank cubin must all agree with
the schema-2 audit. Original CRLF MLIR exports are checked against their
own hashes and normalized only to verify the recorded LF literal
relation. New controls use the exact literal UTF-8 bytes.

The manifest also binds the executing Python, distribution versions,
compiler Python/native files and explicitly supplied toolkit libraries.
It includes a byte-identical private copy of the observer. Preparation
does not import CuBIE, Numba, CUDA bindings or MLIR and does not touch a
codegen cache. Outputs must use fresh directories. `verify` rechecks the
complete manifest on CPU; changing an original control, compiler input
or observer source invalidates it.
Every verification re-derives case membership, target options, linker
target, NRT flags and prepared artifact bytes from the hashed original
cache and extraction receipt. Editable manifest fields cannot redefine
the original settings or substitute a different prepared control.

The supported replay is specifically the installed SM89 LLVM70 LTO
path, with no external links and no NRT requirement. The original cache
does not serialize accumulated helper link objects. The observer does
not invent them: it requires the original LTO alone to reproduce the
original cubin. A mismatch remains a rejected replay with raw artifacts,
not a successful localization of an optimization.

## Installed route and observed stages

The installed [optimization path](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_optimization.py:764)
saves optimized MLIR before pre-codegen patterns. SM89 selects
`_call_llvm70_capi(module, options, gen_lto=True)` at line 809. Native
observation reparses that exact saved MLIR and uses those installed
patterns and that same callable. It retains the post-pattern MLIR and
replayed LTO bytes.

The installed [dump hook](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_optimization.py:256)
requests the actual libnvvm input from the LLVM70 C API and writes its
bytes at lines 309/325. The external environment variable must name
the fresh observation's `nvvm_input` directory. Exactly one natural
replay dump is required before optional inspections. Text dumps retain
literal annotation lines; bitcode remains explicitly undecoded.

The observer then links the **original** LTO with the saved linker
architecture and effective options from
[mlir_lowering.py:157](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_lowering.py:157).
Replayed LTO and relinked cubin are checked independently against the
original bytes. It records actual resolved libdevice/libnvvm/translator
inputs, actual loaded LLVM/nvvm/nvJitLink DLL paths and hashes, imported
compiler module paths and hashes, arguments, environment and options.
Any unbound loaded compiler input rejects the observation.

`original_outputs_reproduced=true` means both outputs match and the
loaded compiler inputs are bound. It does **not** prove byte equality
with the original unsaved libnvvm input: compilation is not injective.
The captured input is an observation of the verified replay route.
No original LLVM input is retroactively fabricated.

Optional `--diagnostic-llvm` uses
[get_llvmir](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_optimization.py:485),
whose `gen_llvmir=True` invocation is separate from natural LTO
production. Optional `--diagnostic-ptx` uses
[get_lto_ptx](/C:/local_working_projects/cubie/.venv/Lib/site-packages/numba_cuda_mlir/mlir_optimization.py:666),
which recreates the linker and requests `-ptx`. This is a diagnostic
re-link, not PTX retained from the original native compilation. Equality
or inequality in this output alone cannot identify where original
native optimization converged. Inspection failures retain the earlier
stages and reject the combined observation.

## Invocation

Run `prepare` with the schema-2 extraction receipt, installed package
root, repeated `--library` arguments for the actual toolkit libnvvm,
libdevice and nvJitLink files, and a fresh `--output` directory. The
bundled LLVM DLLs are included in the compiler package inventory.
`python <prepared>/observer.py verify --manifest <prepared>/manifest.json`
performs the CPU identity check. Each native case then uses:

```powershell
$env:NUMBA_ENABLE_CUDASIM = '0'
$env:NUMBA_CUDA_MLIR_DUMP_NVVM = '<fresh-case-output>/nvvm_input'
python '<prepared>/observer.py' observe `
  --manifest '<prepared>/manifest.json' --case radau5_count2 `
  --output '<fresh-case-output>' --execute-native `
  --diagnostic-llvm --diagnostic-ptx
```

Set environment variables externally before the private process starts.
The worker never modifies compiler installations or process environment.
Run all six cases under one unchanged inventory and retain failures as
well as matches. Do not reuse an output directory for a retry.

## Discriminating interpretation and reduction

Compare the counted backedge and its metadata across matched replay
inputs. Missing or collapsed count annotations there would localize a
loss before libnvvm on the replay route. Distinct annotations would
exclude that explanation for the replay and narrow examination to later
optimization. Whole-file LTO differences can be metadata alone; whole
cubin equality does not establish Newton-body replication. The two
families remain separate controls throughout.

If these stages cannot distinguish the responsible transformation,
the compiler-only reduction must retain the actual cap-eight int32
Newton iterator, runtime float32 state and warp-voted early exit from
[newton_krylov.py:400](/C:/local_working_projects/cubie-worktrees/hardware-epoch-ff3a567f/src/cubie/integrators/matrix_free_solvers/newton_krylov.py:400).
Start from retained regions and preserve a counted inner workload and
observable output. Admit a reduction only after it reproduces Radau's
full/2/4 alias and Kvaerno's count sensitivity. Then remove the nested
workload and vote separately, retaining full/count-1/2/4/False as distinct
requests. Widths must come from the actual helper call instances, not
an arbitrary workload constant. No reduction has been compiled here.

This tool observes compiler stages; it supplies no fitted parameter,
SASS conversion weight, native register estimate or performance model.
