# Fresh shared-capacity holdouts

`native_plan_forwarding_holdout.py` prepares chain21 and chain22 with
local/shared stage-accumulator placement. It retains the reviewed public
construction, compile-only extraction and first-chunk geometry protocol
from `native_plan_holdout.py`; the original observer and its 48 plans
remain unchanged. The new worker imports this separate observer and
requires eight selected shared forecasts or four local forecasts at the
actual compiled block size. It never solves, pins a launch, changes an
installed compiler, or relinks an observed cubin.

`native_plan_forwarding_construct_e1/selection.json` in the raw bank was
saved before these source constructions. Its hardware derivation selects
the adjacent four-to-three-block shared-capacity crossing. The actual
generated strides are 756 and 796 bytes. At 32 threads, the 1024-byte
reserved allocation and 128-byte quantum give 25216 and 26496 bytes per
block. Four allocations occupy 100864 and 105984 bytes against 102400
bytes available per SM. This conditional capacity crossing is independent
of the unknown native register count within the supported range.

Preparation binds that decision and the root's original source receipt
to all four graph records, the unchanged base estimator, the forwarding
estimator and the exact hardware query. It independently recomputes the
allocation and crossing from each actual graph's construction. Complete
generated source bytes, call/value/alias data, compiler identity, actual
padding and zero native overloads are checked again in isolated workers.

Each local case has eight original-estimator forecasts: block32/64,
promote/addressable, and contraction disabled/enabled. Each shared case
has sixteen forwarding forecasts with those combinations plus early/late
final-store scheduling. All 48 files are saved and bound before native
compilation. Source-only preparation never reads heldout native labels.
Shared forecasts retain the separately reported caller initialization
phase; neither native whole-kernel resources nor all across-step retention
are claimed to equal the complete-step model.

Compilation rechecks all prepared source workers and every prediction
record before starting an isolated worker. The installed MLIR metadata
cubin/entry must equal the same live compatibility library's bytes/name.
The observer records both requested block64 and the public limiter's
actual first-chunk geometry, all chunk sizes and prospective wave counts.
It selects the forecasts at that actual block size. A small final chunk
remains explicitly timing-ineligible. Exact source, config, function,
compiler settings and cache identity are rechecked after compilation;
failed native artifacts are retained. No runtime performance is measured
by this command.

The partially completed first preparation is retained at
`native_plan_forwarding_prepare_e1`. It was stopped for the independent
forwarding verifier's source-semantic binding repair before native work.
The corrected source epoch generated all 48 forecasts in
`native_plan_forwarding_prepare_e2`. Its selection gate rejected the
different hardware-query file used by the parent source receipt, although
the parsed hardware capacities were identical. The next preparation,
`native_plan_forwarding_prepare_e3`, reuses those exact 48 forecast files
and binds the parent's original query. Every forecast's own hardware
query is separately rehashed and must yield the same hardware model.
Independent release is required before root's serial native lane runs:

```text
python -m benchmarks.hardware_model.native_plan_forwarding_holdout \
  compile --prepared <raw>/native_plan_forwarding_prepare_e3 \
  --manifest-sha <reviewed-manifest-sha> --case chain21_shared \
  --output <fresh-native>/chain21_shared
```

The parent and external worker must receive the frozen production `src`
and the research repository root on `PYTHONPATH`. All codegen and native
caches are private to the new output directory. No original cache or
prediction file is overwritten.
