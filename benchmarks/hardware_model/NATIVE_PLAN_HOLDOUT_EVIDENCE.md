# First ERK native holdouts

All six `native_plan_holdout_native_e3` public compilations passed their
original source/configuration/input joins, retained native artifacts and
exited cleanly without a solver launch. The 48 predictions from
`native_plan_holdout_prepare_e1` remain unchanged. The released e3
preparation manifest is `6aa73cc8…244dd8`; the estimator remains
`f547ee91…9480471`. These are compiler labels, not timing or numerical
validation.

| Case | Native GPR/thread | Total local B/thread | Actual block | Driver blocks/SM | Whole entry instructions |
|---|---:|---:|---:|---:|---:|
| chain16 local | 168 | 0 | 64 | 6 | 2,896 |
| chain17 local | 255 | 0 | 64 | 4 | 3,072 |
| chain18 local | 254 | 0 | 64 | 4 | 3,232 |
| chain16 shared | 255 | 0 | 32 | 5 | 3,192 |
| chain17 shared | 255 | 24 | 32 | 4 | 3,408 |
| chain18 shared | 255 | 40 | 32 | 4 | 3,592 |

The public limiter reduced every requested shared block from 64 to 32;
matching predictions were frozen at both sizes. All prospective chunk
geometries satisfy two full driver-queried occupancy waves. That query
does not establish actual runtime carveout or residency. Matching the
shared case's occupancy can follow from its shared-capacity constraint
even when the register estimate is wrong.

The original promoted local plans allocate 239/253/255 modeled words
and 0/0/28 modeled spill bytes for dimensions 16/17/18. Native chain18
has no local frame. Shared plans at actual block 32 predict 109/115/121
words under promoted-local storage, with shared accesses retained. The
native shared allocations are all 255 registers. These observations
falsify direct use of those conditional allocations as compiler
predictions. No conversion factor is inferred. Native total local bytes
are not relabeled as spill bytes.

The shared mismatch has a concrete memory mechanism. Native shared
loads are zero in all three complete kernels; shared stores number
288/306/324, exactly `2*n*9`. The old plans emit 2,592/2,745/2,916 shared
loads and 1,584/1,683/1,782 shared stores for one expanded step. The source
model treats every shared array access as addressable, while the native
compiler eliminates those loads and most writes.

For chain17, the exact cached optimized MLIR still contains 53 shared
load sites and six shared store sites in loops. A direct GEP/cast proof
traces 31 pointer values from `@__dynamic_shmem__0`, including its
address-space cast to a generic pointer, with no unhandled uses of those
proven pointers. Thus the shared-load elimination happens after that
cached stage. Static loop sites are not executed instruction counts,
and this does not identify the particular downstream optimization pass.

The native chain17 stores give more detail: 153 zero stores cover each
four-byte offset from 0 through 608 before the outer-loop header at
`0x1440`. Another 153 value stores cover those same offsets inside the
step, at PCs `0xa870` through `0xcc30`. The shared base is not rewritten
after `0x0510`. In `generic_erk.py`, the accumulator clear at line 468,
repeated updates at line 487 and state conversion at line 497 produce
these same fixed cells. The source shape is nine rows of 17 FP32 values.
The observed pattern supports modeling exact-cell read forwarding and
intermediate-store elimination while retaining final stores, separately
from the one-time outer initialization. It does not justify removing
all observable shared writes.

Late final-store scheduling can extend the lifetime of retained cell
values. Offline `nvdisasm --life-range-mode count` on the unchanged
chain17 shared cubin reports a 253-GPR live peak, including arithmetic
at `0x5f10`, `0x5f20`, `0x5f70`, `0x5f80` and `0xa830`. The allocation
therefore is not merely a high register number with a small live set.
It remains unproved how much of that peak belongs to accumulator
retention, caller values, arithmetic scheduling or other compiler
transformations. Early and late final-store schedules are useful
separate compiler hypotheses, not fitted overheads.

Whole-kernel operation counts also include controller/caller work beyond
the modeled ERK step. In particular the native FP64 time bookkeeping
does not change the FP32 system, input arrays or step arithmetic into an
FP64 workload. Native and modeled per-step opcode totals are not compared
as equal scopes. No service rate, store completion latency or instruction
cache penalty is inferred from these compile-only labels.

Raw joins, exact cached-MLIR export, pointer chains, native store offsets
and liveness PCs are retained under
`verification/native_plan_materialization_e3`. The six-native independent
audit is `verification/native_plan_first_native_independent_20260905`.
The next estimator version must preserve these original forecasts and
test its forwarding/storage alternatives on fresh, source-selected
dimensions before reading their native labels.
