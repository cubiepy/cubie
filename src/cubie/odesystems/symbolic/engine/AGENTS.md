<!-- Parent: ../AGENTS.md -->

# engine

## Purpose
Lightweight hash-consed expression IR and the compute passes the whole symbolic
pipeline runs on: differentiation, substitution, common-subexpression elimination,
dependency ordering, pruning, structural simplification, and CUDA source emission all
operate on interned IR nodes. SymPy is a parse-boundary translation layer only
(string/AST parsing and user-supplied SymPy input); the normaliser and the CellML
loader convert every expression to IR via `from_sympy` before any downstream pass.
`to_sympy` is used only by verification tests.
Nodes pickle through their constructor functions, so unpickled expressions re-intern
(the CellML disk cache relies on this).

## Key Files
| File | Description |
|------|-------------|
| `expr.py` | IR nodes, weak interning, algebraic folding, substitution, differentiation, and operation counts. `Local` represents generated scalar temporaries. |
| `from_sympy.py` | The only SymPy-importing module: `from_sympy`/`convert_assignments` (SymPy → IR, memoised), `to_sympy` (verification utility for tests), `derivative_name_map` (recovers `fdiff` placeholder names from the parser's dynamic device-function classes). |
| `adapter.py` | `SystemIR` + `system_ir(equations, index_map)` — builds the equations, ordered symbol tables, array-reference maps, and derivative names used by generators. |
| `assignments.py` | Assignment-list transforms: `topological_sort` (policy-driven ordering — `liveness_auto` default, `kahn`, `greedy`, `dfs` — deterministic tie-breaks), `prune_unused` (drop assignments not feeding outputs), `cse_and_stack` (reference-counting CSE over the DAG plus partial Add/Mul subset matching, atomic-assignment inlining before and after extraction, and pow-family strength reduction). |
| `printer.py` | `IRPrinter` and `print_cuda`/`print_cuda_multiple`: renders IR as Numba-CUDA source — `precision(...)` literal wrapping, integer and integral-float powers up to `_POW_CHAIN_LIMIT` as multiplication chains (structural Pow rules), half powers to `math.sqrt`, guarded reciprocals, Piecewise as branchless `selp` selections (bitwise `&`/`|` predicates), `CUDA_FUNCTIONS` mapping, scalar→array symbol remapping. Constants never reach the printer as symbols — their values fold in as `Num` literals before generation. Accepts SymPy input at the boundary (auto-converts). |

## For AI Agents

### Interning is the invariant everything relies on
Live structurally identical expressions are the same Python object: equality is `is`,
hashing is `id`. The weak intern pool releases unused graphs. Constructors fold algebra on the way in (flattening, like-term and
power collection, numeric folding, zero/one identities; `call` evaluates
known math functions of numeric literals when the result is finite —
domain errors and overflows leave the node in place; `rel` folds numeric
operands to `TRUE`/`FALSE`, `bool_op` folds boolean literals, `piecewise`
drops false branches, truncates at the first true one, and merges
default-valued branches into the default). **Never
instantiate node classes directly** — always build through the constructor
functions, or interning breaks and `xreplace`/CSE silently stop matching.

### Determinism
Commutative arguments are ordered by the structural `sort_key` computed at
construction — never by hash or intern order — so generated source is byte-identical
across processes regardless of `PYTHONHASHSEED` or session history. Keep it that way:
no set iteration may influence emitted structure.

### Array references
`Arr(name, index)` with a fixed Python int index is the engine's entire "IndexedBase".
Bracket-named SymPy symbols (`sp.Symbol("jvp[0]")`) and 1-D `sp.Indexed` leaves both
convert to `Arr`. JVP outputs are detected as `Arr("jvp", i)` — not by string prefix.

### Differentiation
`diff` uses an analytic rule table (`_DERIVATIVES`); `Min`/`Max` differentiate to
Piecewise selections, `Abs` to `sign`, `sign`/`floor`/`ceiling` to zero. Unknown
applied functions differentiate to `d_<name>` placeholders (chain rule appended as a
trailing integer arg index) unless `derivative_names` overrides the target — the
adapter recovers those names from the parser's `fdiff` classes via
`derivative_name_map`. `gamma`/`loggamma` raise `DifferentiationError` (no CUDA-side
polygamma exists).

### Substitution maps compose, passes don't repeat
`xreplace` applies one node-for-node map in a single memoised pass and does not
revisit replacement images. Generators build one combined map per stage instead of
chaining `.subs` calls; when sequential semantics are genuinely needed, compose the
maps, don't re-walk the tree.

### CSE
`cse_and_stack` extracts every multiply-referenced composite node, then a partial
subset pass recovers sharing that n-ary flattening hides (`2*e*a` vs `e*a` — see
`_find_partial_subsets`). `_cse<N>` numbering continues after existing locals.
Extraction produces the assignments; `topological_sort` orders them.
Around extraction, `_inline_atomic_assignments` substitutes literal-, symbol-,
and local-valued targets into later right-hand sides (constant chains fold
through the constructors; extraction-created renames collapse), and
`_reduce_pow_families` names one primal per non-integer power family
(`x**p` alongside `x**(p±1)`, `x**(2p)`, `x**(2p±1)` — differentiation and
same-base product folding produce exactly these, so matching is exact float
equality against each derivation's roundings) and derives the rest by
multiply/divide of the primal local. Both passes only rewrite — the callers'
`prune_unused` drops assignments they leave unreferenced.

### Testing
`tests/odesystems/symbolic/engine/test_engine.py` (unit: folding, diff vs SymPy
ground truth via `to_sympy` + numeric spot checks, subs, CSE numeric equivalence,
ordering, pruning); `tests/odesystems/symbolic/test_cuda_printer.py` (printer
emission rules, including the subtracted-sum parenthesisation and division
regression guards). The generator/solver test suites exercise the engine end to end
against finite-difference references.

## Dependencies
### Internal
- None. Consumed by every module in `codegen/`, plus
  `parsing/jvp_equations.py` and `parsing/auxiliary_caching.py`.
### External
- `sympy` (only in `from_sympy.py`); `attrs` (`SystemIR`). Stdlib `fractions`.
