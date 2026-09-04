# Typed source-value retention certificates

`source_value_graph.py` constructs a complete source graph for a captured,
fully expanded ERK step and its actual generated helper calls. It accepts
a constructed Solver before specialization. The completed cases below use
the actual Lorenz/RK4 and chain32/Vern7 code, coefficients, allocator
closures and integration-loop argument bindings. No native label, timing,
simulator trace or register measurement is an input.

This component measures a conditional source scheduling quantity. Its
frontier counts are not GPU register counts, spill estimates, hardware
lower bounds or a completed placement heuristic.

## Source contract and identities

`caller_bindings` reads the actual `step_function(...)` call in
`src/cubie/integrators/loops/ode_loop.py:873` and its allocation assignments
at lines 577–623. The captured allocator flags, local sizes and parent
byte slices determine storage identities. Loop-entry zeroing is not
mistaken for the contents at an arbitrary step invocation. Accessed
caller cells are typed symbolic live-ins; effective timestep and narrowed
time have the caller's FP32 type. The adapter checks the captured caller
precision. Unaccessed driver-coefficient extents remain unknown.

Each result of FP32 arithmetic, negation or a precision cast has a distinct
typed value ID. Scalar assignments alias their right-hand value. A read
of an exactly known same-type byte cell aliases the reaching stored value;
a write changes that cell's version. This is a **conditional scalar
replacement interpretation** of the source, not an assertion that a load,
copy or store instruction survives lowering. Captured tableau arrays keep
their dtype, shape, exact values and byte hash; captured array views retain
their common root and byte offset. The original AST is interpreted, so
turning an FP32 coefficient into an untyped Python float does not erase
its type.

The JSON separates:

- `values` and `value_edges`: actual produced/consumed scalar identities;
- `nodes`, `order_edges` and `final_cells`: source operations, same-cell
  read/write hazards and final memory versions;
- `aliases`, `allocations`, `calls` and `registry`: scalar aliases,
  resolved storage windows, actual argument bindings and registry data;
- `controls`: selected constant branches and every fully expanded loop
  with its source line, captured directive and iteration indices;
- `certificates`: complete schedules and every typed frontier, including
  the empty prefix, for the whole step and each actual helper call cut.

The old destination is not automatically a consumed value of a write.
For example, a reset `array[i] = typed_zero` has an ordering dependency on
the prior write, but no value dependency on the overwritten value.
`array[i] += coefficient * increment` consumes the reaching value through
its read and addition. This distinction is missing from the old
`spill_dag.py` pending-read model; the detailed source audit is in
`verification/source_retention_mechanism_20260905/MECHANISM_PROPOSAL.md`.

Names/indices are expanded only through the bounded integer constant
rules in `expansion.constant`. Dynamic numeric types must be supported
explicitly; mixed unknown promotions fail. Floating arithmetic is kept
as operations: no CSE, reassociation, contraction, zero-coefficient
elimination or floating constant folding is assumed. A captured loop must
request `(True, None)`. Residual loops, unknown branches/calls/indices,
overlapping differently typed cells and uninitialized private reads raise
`Unsupported` with a concrete source location. These conditions do not
produce a partial certificate presented as complete.

## Quantity and exact-search claim

For an executed dependency ideal `I`, the frontier contains each distinct
nonconstant scalar value whose producer has executed (or is a live-in)
and which has a value consumer outside `I` or is an observable exit value.
Constants remain explicit graph values but do not count as retained
mutable values. The count is taken **after** each source operation. It
does not model simultaneous machine operand/destination storage.

The whole-step exit contract preserves final contents of every accessed
caller cell and the returned status. Private scratch cells are observed
only through their actual uses. A helper certificate is a cut through its
actual lexical call interval: surrounding live-through values remain in
its entry and exit sets. Thus a no-op observables helper can have a
nonzero frontier; it preserves the surrounding live values and does not
create that demand itself.

The exact objective is the smallest possible maximum count of retained
FP32 **source values**, over schedules satisfying both edge sets. Other
types are counted separately. Dijkstra bottleneck search over dependency
ideals proves this objective only when it reaches the complete ideal.
Every successful result includes its full schedule and every frontier.
When the explicit state budget is exhausted the status is
`state_limit_no_optimum_claim`; the complete source-order witness remains
available. Neither a time limit nor a fitted liveness multiplier is used.

## Completed actual-source receipts

Raw root: `C:/local_working_projects/cubie-notes/hardware_unroll_placement`.
The completed receipts share extractor SHA256
`7209fcad89e1aa8e6e59f8cac992c5a1728456c97cdf284179c98ea3afa0d0da`.
The retained step source is the frozen measurement tree's
`generic_erk.py`, SHA256
`17818e33dbf6cba10b3c7ccf036e3b147469e8fe140ae2a2116b42ce4f402027`.
Generated helpers, allocator sources, caller source and descriptor
dependencies each have separate paths/hashes in the JSON.

| Actual source case | RHS calls | Accumulator | Nodes | Value IDs | Source-order peak FP32 values | Exact whole-step result |
|---|---:|---:|---:|---:|---:|---|
| Lorenz, RK4 | 4 | 9 FP32 elements / 36 bytes | 391 | 262 | 22 | Unresolved at 5,000 states |
| chain32, Vern7 | 10 | 288 FP32 elements / 1,152 bytes | 28,403 | 20,003 | 423 | Unresolved at 1 state |

These counts include source operations that a compiler may remove. They
are not an explanation of the observed native register count or local
traffic. The fourth Lorenz RHS call cut has a complete exact witness with
peak 13, compared with 14 in source order. The earlier RHS call cuts did
not finish their searches at 5,000 states; there is no optimality claim
for those cuts. The complete chain32/Vern7 witness records the ordered
accumulator additions at `generic_erk.py:473–500`, the actual FP32 tableau
and the generated RHS bindings for each of the ten stages.

Files under `verification/source_value_graph_20260905/`:

- `lorenz_rk4_v3.json`: complete Lorenz graph and certificates;
  SHA256 `3e3224eee65df988117292158390924d949f4238a6f0a941781bf5b4c4f5616a`.
- `chain32_vern7_v3.json`: completed chain32/Vern7 graph and certificates;
  SHA256 `2a0e0db85332320259e312b9688835d69c6461260dfdc70681c5b3ec95ee4501`.
- `validation_v3.json` and `validate_receipt.py`: independent CPU
  recomputation of all schedule/frontier rows using interval endpoints,
  exact value/order edge joins, byte-cell version/hazard checks and
  distinct negation/cast identities. The Lorenz result additionally
  replays the graph numerically and compares every output bit with a
  FP32 streaming-tableau reference using the retained generated RHS
  expressions. This checks source semantics, not native numerical
  behavior. Both constructions retained zero native overloads.

Boolean array indices are explicitly unsupported, including Python and
NumPy Boolean constants. They are not coerced to integer offsets.

The earlier `attempt*.json`, `*_v1.json` and `*_v2.json` files are
development receipts. The first
chain attempt was stopped during an unnecessarily quadratic frontier
enumeration and produced no successful graph. The completed witness uses
an incremental remaining-use calculation; its independent validator uses
the separate interval-endpoint formulation.

Commands, with the documented frozen-source `PYTHONPATH` and a fresh
cache/output path per execution:

```powershell
python -m benchmarks.hardware_model.source_value_graph --system lorenz --algo rk4 --max-states 5000 --output <fresh-json> --cache-root <fresh-cache>
python -m benchmarks.hardware_model.source_value_graph --system chain32 --algo vern7 --max-states 1 --output <fresh-json> --cache-root <fresh-cache>
python <raw-root>/verification/source_value_graph_20260905/validate_receipt.py <lorenz-json> <chain-json> --out <fresh-validation-json>
```

## Compiler obligations before hardware use

The certificate supplies auditable post-codegen values and dependencies.
Using it for native retention requires separate evidence for each of:

1. Which fixed-loop instances and constant branches actually survive
   lowering, including coefficient folding and zero products.
2. Which byte cells are promoted and whether escaping arrays, helper
   boundaries, aliasing or ABI requirements prevent that promotion.
3. Which values survive casts, algebraic rewrites, CSE, FMA contraction,
   reassociation and rematerialization, and how those rewrites change the
   value graph and allowable schedules.
4. Which native storage classes hold each surviving value, including
   uniform registers, predicates, address arithmetic, hidden temporaries,
   register-pair requirements and operand/destination overlap rules.
5. Whether the native scheduler's allowed instruction orders correspond
   to this graph; spill loads/stores introduce their own dependencies.

Until these obligations are met, even an exact source optimum is not a
native register lower bound. Hardware register capacities cannot simply
be compared with 22 or 423. The intended interface is a source graph and
its checkable certificates on which a separately evidenced lowering
transformation can operate; it is not a rescaled source peak.
