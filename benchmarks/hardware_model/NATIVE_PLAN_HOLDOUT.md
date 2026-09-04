# Fresh ERK compiler holdouts

`native_plan_holdout.py` tests the compiler assumptions in
[NATIVE_PLAN.md](NATIVE_PLAN.md) on the source-selected chain16/17/18
Vern7 dimensions. Each dimension has actual local and shared accumulator
placements. No earlier native/timing label for these configurations is
an input. Preparation imports the estimator and source factories; only
the explicit `compile` command requests native specialization.

Preparation freezes 48 complete predictions before starting the source
workers: six configurations, both 32/64-thread geometries, promoted and
addressable local storage, and both optional contraction scenarios.
All use the same actual JIT flags. The latter two axes are compiler
hypotheses, not user settings. Predictions remain unmodified when native
labels arrive. Both block sizes are saved because the public kernel's
shared-memory limiter may reduce a requested 64-thread block to 32.

The existing source graphs, constructor receipts, source bytes, compiler
fingerprint and hardware query are checked before estimation. Every
worker receives an exact hashed request and fresh cache directory. Its
public kernel-cache directory is explicitly set inside the new output,
so an external kernel-cache override cannot redirect native artifacts. It
constructs the actual `build_chain(N, 3)` system and unchanged chain/Vern7
configuration with the requested accumulator placement. The complete
typed graph must equal the original graph after replacing source file
paths with their exact file-byte hashes. Definition lines, actual closure
values, calls, controls, aliases, value and order edges, registry layouts
and allocation identities all remain in that comparison. The NativePlan
construction envelope is checked separately; it contains cache-specific
paths and a different worker identity.

Each CPU worker retains the new graph, generated source, copies of
observed helper/caller/kernel files, config hash, generated function hash,
actual flags and shared stride/padding, and exact input arrays. Its
completed process receipt contains the executed command, working
directory, full stdout/stderr and return code after Solver cleanup.
The final prepared manifest binds those receipts and every prediction.
The `compile` command requires its externally pinned SHA256.

A native worker reconstructs into another fresh cache and proves the
same complete source graph, configuration, compiler and array identities
before calling the public `Solver.compile`. It records the resulting
cubin, `nvdisasm -c` command/output, complete static per-section opcode
counts, reported registers, total local-memory bytes and static shared
bytes. There is no diagnostic PTX relink, verbose-link monkeypatch,
native code change or kernel launch. Whole-kernel counts cover outer
integration/control/helper code beyond the modeled ERK step; they are
not silently equated to per-step dynamic instruction work. Reported local
bytes include named arrays and other local allocation, so they are not
labeled as compiler spill bytes.

The installed MLIR compatibility `CodeLibrary` stores the natural cubin
in `_cubin`; it does not implement `get_cubin`. Extraction joins
`CompileResult.cres.metadata['cubin']` and `['func_name']` to that same
live library's `_cubin` and `_func_name`. The observer binds the actual
result/library/dispatcher classes and exact installed `compiler.py` and
`descriptor.py` bytes, then retains the library's existing `get_cufunc`
and dispatcher resource-query methods. No replacement library is loaded
and no intermediate is relinked. The earlier extraction failure remains
in `native_plan_holdout_native_e2/chain17_shared`; its public compile
completed but supplied no admitted native comparison.

Compilation can refresh settings through `_prepare_batch`. Before a
result passes, the observer compares postcompile config/function hashes,
flags, dimensions, shared layout, compiler identity and all observed
source bytes with the pre-native record. A changed source/configuration
rejects comparison; its postcompile record and native artifacts remain
available for diagnosis.

After compilation the worker calls the production block-size limiter
with the actual padded shared stride and first allocated chunk's run
count, records requested and limited
geometry, and selects only the already-frozen matching block predictions.
It does not pin a launch or reinterpret 64-thread predictions as 32-thread
ones. Any other resulting block size fails comparison admission and
retains the diagnostic artifacts. The batch size is twice the saved
device's maximum resident-thread capacity across all SMs; a driver
occupancy query reports waves for each actual chunk at that geometry.
A small last chunk is explicitly prospective timing-ineligible without
invalidating a source-matched compile-only label. That query is
prospective: no actual runtime residency or carveout is observed by this
compile-only protocol.

Run preparation with the frozen package source first on `PYTHONPATH` and
the research repository root also present, with CUDA simulation disabled
externally. External worker scripts need the repository root to import
`benchmarks.hardware_model`. Use a fresh output directory:

```powershell
python -m benchmarks.hardware_model.native_plan_holdout prepare --graphs <native_plan_cpu_v3> --hardware <saved-device-manifest> --nvdisasm <nvdisasm.exe> --output <fresh-prepared-dir>
```

After independent source/receipt review and the root's serialized native
slot, compile one configuration into another fresh directory:

```powershell
python -m benchmarks.hardware_model.native_plan_holdout compile --prepared <prepared-dir> --manifest-sha <frozen-manifest-sha256> --case chain17_shared --output <fresh-native-dir>
```

The exact six case labels are `chain16_local`, `chain16_shared`,
`chain17_local`, `chain17_shared`, `chain18_local`, and `chain18_shared`.
The compile-only result supplies falsifying labels for lowering,
materialization, allocation and geometry hypotheses. It supplies no
numerical or performance verdict. Ordinary repeated timing and exact
work/counter validation remain separately controlled measurements under
the original two-wave and numerical-validity requirements.

When only this observer changes, `prepare --reuse-predictions <request>
--reuse-predictions-sha <sha256>` can preserve an earlier frozen prediction
set. The model, hardware and complete graph/construction identities must
still match, every prediction keeps its original file/hash, and all six
source workers run again under the new observer. This does not rerank or
alter predictions after labels.
