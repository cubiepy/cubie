# Conditional typed lowering for implicit regions

`implicit_native_lowering.py` consumes a completed source-only implicit
execution region. It never imports CuBIE, CUDA, a compiler, a disassembler,
or measured iteration labels. It keeps source FP32, signed/unsigned 32-bit,
and Boolean values distinct. Boolean expression results use a separate `P`
bank; FP32, integer, mask, base-address, and explicitly materialized numeric
literal values use the `R` bank. A Boolean value reaches local spill storage
only through a recorded `SEL` to a canonical uint32 0/1 word and returns to
the predicate bank through an `LDL` plus `ISETP.NE 0` sequence. PT and !PT
represent Boolean constants and do not occupy either modeled bank.

The lowering admits one explicit compiler alternative supplied as JSON. It
records FTZ and contraction choices, 32-bit dynamic integer operations,
approximate reciprocal-multiply division and square root, numeric-literal
materialization, source-order scheduling, and predicate spill representation.
These are auditable hypotheses, not facts inferred from source operation
counts. Comparisons carry type, relation, and unordered behavior. Selects
carry typed payloads. Boolean-to-word and word-to-Boolean conversions include
their explicit typed 0/1 or zero operands. `ActiveMask`, `VOTE`, and runtime
branch decisions retain exact declared masks, results, values, and selected
paths. They use abstract operation families where Ada encoding or routing is
unproved. Built-in Python `min`/`max` become an ordered comparison and select,
retaining the first argument on a tie or unordered comparison.

Admission rederives the source value edges, same-cell version frontier,
observable final cells, complete caller cuts, required runtime-control roots,
and the one-to-one branch-decision/control records. It recomputes every hot
template digest and separates fixed source-loop indices from recurrent trace
indices. The finite-FP32 and NaN-sensitive primitive contracts must match the
source frontend exactly before the conditional min/max lowering is admitted.
The selected iteration regime is reconstructed from runtime regions,
loop-top vote/branch pairs, source workload roles, entry masks, body indices,
and recurrent controls. Each trace entry mask must equal the call's unique
runtime entry mask. The plan exposes exact source-value, contextual cell-read,
and source-node mappings; validation reconstructs the entire typed lowering
from the admitted source before allocation is accepted. Per-call fields,
warp totals, and both lane-counter rows must equal the reconstruction; they
are not accepted as free labels.

The allocation scenario is also an input. The retained SM89 example uses 64
R words per thread because a complete 1,024-thread CTA shares the documented
64K 32-bit-register SM/block capacity. It uses seven P slots because NVIDIA's
CUDA-GDB documentation exposes P0 through P6, while PT is a constant. The P
count is a conditional architectural interpretation, not a compiler promise.
The allocator follows the fixed selected source trace and evicts the value
with the farthest next use. It claims no optimum. Every source operation,
register read/write/release, constant materialization, predicate conversion,
spill/reload, stable four-byte local home, and observable exit location is
retained. A replay validator proves conservation, bank typing, exact named
cell semantics, and frame extents for this alternative.

The actual Kvaerno3/LU example is generated from
`implicit_source_graph_cpu_e17/kvaerno3_lu/graph.json`. Its caller cuts retain
live-through values, all seven runtime `BranchDecision` nodes remain required
roots, and every vote binds the participating and active-entry masks. Under
the retained default-flag alternative (FTZ and contraction enabled), its
1,141 source nodes produce 543 modeled operations and 961 allocation events.
The 64-R/7-P, promoted-local scenario reaches both supplied bank capacities
and uses two four-byte predicate spill homes. The addressable-local
alternative produces 999 operations, retains 288 named-frame bytes, reaches
32 R/7 P, and uses the same eight-byte spill extent. These are consequences
of the stated alternatives rather than measured register or spill counts.

The validation receipt also constructs all nine DIRK/FIRK/ROS cases across
LU, MR, and BiCGSTAB. It stores only the small Kvaerno3/LU plan; the remaining
rows are construction checks rather than fitted observations. The stored plan
is produced through the CLI and binds the exact graph, architecture, and
compiler-alternative paths and SHA-256 digests.

The plan separates selected dynamic trace work from static source templates.
Repeated recurrent instances may share a hot-template identity, but neither
the template nor a modeled operation establishes native code replication or
instruction bytes. The result supplies no native register count, scheduling,
uniform-register route, reconvergence instruction, memory-service time,
cache penalty, liveness multiplier, or solver runtime prediction. Memory
layout and service remain separate physical-model inputs. Native holdout
labels can test this conditional compiler alternative; they are never inputs
to it.

Architecture facts are sourced from the current
[CUDA Programming Guide compute-capability tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html),
which list 64K 32-bit registers per SM and per block for CC8.9, and the
[CUDA-GDB register documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html),
which exposes predicate registers P0 through P6. The
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
explains that predicates cannot be directly loaded or stored and gives
`selp.u32` as the predicate-to-word conversion pattern. The
[CUDA Binary Utilities instruction reference](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
lists distinct GPR and predicate locations and P2R/R2P, comparison, select,
logic, load/store, and control families. Those references establish available
hardware concepts; they do not establish this modeled trace as native output.

CPU construction example:

```text
python -m benchmarks.hardware_model.implicit_native_lowering \
  --graph <actual-graph.json> --architecture <architecture.json> \
  --compiler <compiler-alternative.json> --out <fresh-plan.json>
```

The CLI uses exclusive output creation. The verification receipt is under
`verification/implicit_native_lowering_final8_20260905/receipt.json`; its
exact source and artifact hashes are recorded there.
