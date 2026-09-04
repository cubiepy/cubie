# Shared forwarding compiler alternative

`native_plan_forwarding.py` models exact-cell shared read forwarding and
intermediate-write elimination on complete, fully expanded ERK source
graphs. It pins the original `native_plan.py` at `f547ee91` and leaves its
forecasts unchanged. The observed chain16/17/18 shared kernels motivated
this alternative; they are mechanism-selection diagnostics, not untouched
validation data. See `NATIVE_PLAN_HOLDOUT_EVIDENCE.md` for the original
source, optimized MLIR and native evidence.

Every shared access must identify one aligned FP32 cell inside the actual
captured per-run shared slice. The graph must contain every known helper
body, allocator, alias and selected control path. A read forwards only
the source value of the most recent write to that exact cell. A read
before any in-region write rejects this alternative. Opaque calls,
unresolved controls and non-exact cells also reject. The admitted graph
has no unresolved barrier or cross-thread access; private per-run slice
ownership is an explicit required construction assumption.

The transformation removes all such reads and every write except the
last write to each cell. Every accessed shared cell must appear in the
boundary-visible final-cell contract, with that last write's source value.
The lowered plan retains exactly one shared store per final cell and
checks source-node witnesses, stored-value semantics, final-memory
membership and an acyclic schedule. The original lowering independently
resolves each source exit value through constant/cast aliases; the store
must retain that semantic and its direct value-producer dependency.
The original allocator then checks
complete SSA and memory conservation, including modeled spill traffic.
Other buffers keep their selected original `promote` or `addressable`
behavior. Arithmetic, constant materialization, contraction and reciprocal
assumptions are inherited without new algebraic rewrites.

Two explicit scheduling hypotheses are emitted. `early` keeps each final
store at its last source-write position. `late` makes all retained
non-final-store nodes precede all final stores. Forwarded consumers depend
on the value producer, not on the visibility store. These hypotheses can
change retention; neither is asserted to reproduce a compiler scheduler.
The plans include actual helper caller live-through cuts. Outer kernel
values and across-step retention remain outside the complete-step graph.

Caller allocator initialization is recorded separately from the step
trace, using the actual zero flag, source identities and byte window.
Its scalar zero-store count is a source expansion count, with native
width/elimination unresolved. Its multiplicity is once per entered
integration call. The per-step final stores repeat with the step. Neither
initialization nor dynamic step multiplicity is silently added to hot-code
copies. In the inspected chain17 kernel, these phases were 153 initial
zero stores outside the integration loop and 153 value stores inside it.

The CPU receipt is
`verification/native_plan_forwarding_20260905/v5/receipt.json` in the raw
bank. It binds the source and all 20 new diagnostic plans. Lorenz/RK4 and
chain16/17/18 Verner7 use their retained actual generated graphs. For both
store schedules and local-buffer materializations, the allocated trace's
observable values and memory are bit-identical to the original lowering
with the same uncontracted FP32 arithmetic. Lorenz also exercises an
eight-word allocation budget and real modeled spill/reload paths. This
comparison validates the memory transformation under the inherited
reciprocal scenario, not GPU division accuracy or native register counts.

For uncontracted promoted chain16/17/18, the model forwards
2592/2745/2916 shared reads, eliminates 1440/1530/1620 writes and retains
144/153/162 final stores per step. Early and late schedules have equal
peak demand on these graphs. No fixed caller register overhead, fitted
liveness factor or invented store-completion latency is introduced.
Missing instruction-service terms remain symbolic, and source-step
allocation remains a compiler hypothesis rather than a native GPR bound.

Fresh source-only construction targets chain21 and chain22 with local and
shared placement. From the actual Verner stage layout, shared stride is
expected to be 756 and 796 bytes after odd-element padding. At block32,
dynamic plus the queried 1024-byte reservation, rounded to 128 bytes,
gives 25216 and 26496 bytes per block. The 102400-byte hardware maximum
permits four versus three blocks. The actual generated strides and all
candidate forecasts must be saved before any native labels. These sizes
were selected from that hardware transition, not from observed timings.

CLI example (inputs are immutable, output must be new):

```text
python -m benchmarks.hardware_model.native_plan_forwarding \
  --graph <source>/graph.json --hardware <hardware>/manifest.json \
  --mode promote --store-schedule late --contract --block 32 \
  --output <fresh>/plan.json
```
