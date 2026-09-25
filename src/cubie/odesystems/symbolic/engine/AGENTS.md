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
| `expr.py` | IR nodes, weak interning, algebraic folding, substitution, `expand` (products over sums, integer powers over products), `rationalize` (float literals to exact rationals), differentiation, and operation counts. `Local` represents generated scalar temporaries. |
| `from_sympy.py` | The only SymPy-importing module: `from_sympy`/`convert_assignments` (SymPy → IR, memoised), `to_sympy` (verification utility for tests), `derivative_name_map` (recovers `fdiff` placeholder names from the parser's dynamic device-function classes). |
| `adapter.py` | `SystemIR` + `system_ir(equations, index_map)` — builds the equations, ordered symbol tables, array-reference maps, and derivative names used by generators. |
| `assignments.py` | Assignment-list transforms: `topological_sort` (policy-driven ordering — `liveness_auto` default, `kahn`, `greedy`, `dfs` — deterministic tie-breaks), `prune_unused` (drop assignments not feeding outputs), `cse_and_stack` (reference-counting CSE over the DAG plus partial Add/Mul subset matching, atomic-assignment inlining before and after extraction, and pow-family strength reduction). |
| `printer.py` | `IRPrinter` and `print_cuda`/`print_cuda_multiple`: renders IR as Numba-CUDA source — `precision(...)` literal wrapping, integer and integral-float powers up to `_POW_CHAIN_LIMIT` as multiplication chains (structural Pow rules), half powers to `math.sqrt`, guarded reciprocals, Piecewise as branchless `selp` selections (bitwise `&`/`|` predicates), `CUDA_FUNCTIONS` mapping, scalar→array symbol remapping. Constants never reach the printer as symbols — their values fold in as `Num` literals before generation. Accepts SymPy input at the boundary (auto-converts). |

## Interning
Structurally identical live expressions are one Python object: equality is `is`, hashing
is `id`, and a weak pool releases unused graphs. Constructors fold algebra on the way in:
flattening, like-term and power collection, numeric folding, zero/one identities; `call`
folds known math functions of numeric literals when the result is finite; `rel` folds
numeric operands to `TRUE`/`FALSE`; `bool_op` folds boolean literals; `piecewise` drops
false branches, truncates at the first true one and merges default-valued branches into
the default. Build nodes only through the constructor functions; instantiating a node
class directly breaks interning, and `xreplace`/CSE stop matching.

## Determinism
Commutative arguments order by the structural `sort_key` computed at construction, never
by hash or intern order, so generated source is byte-identical across processes. No set
iteration may influence emitted structure.

## Array references
`Arr(name, index)` with a fixed int index is the engine's `IndexedBase`. Bracket-named
SymPy symbols (`sp.Symbol("jvp[0]")`) and 1-D `sp.Indexed` leaves convert to `Arr`. JVP
outputs are `Arr("jvp", i)`.

## Differentiation
`diff` uses the rule table `_DERIVATIVES`: `Min`/`Max` → Piecewise selections, `Abs` →
`sign`, `sign`/`floor`/`ceiling` → zero. Unknown functions differentiate to
`d_<name>` placeholders (argument index appended) unless `derivative_names` overrides
them; the adapter recovers those names from the parser's `fdiff` classes via
`derivative_name_map`. `gamma`/`loggamma` raise `DifferentiationError`.

## Substitution
`xreplace` applies one node-for-node map in a single memoised pass and does not revisit
replacement images. Build one combined map per stage; compose maps when sequential
semantics are needed rather than re-walking the tree.

## CSE
`cse_and_stack` extracts every multiply-referenced composite, then a partial-subset pass
recovers sharing that n-ary flattening hides (`2*e*a` vs `e*a`, `_find_partial_subsets`).
`_cse<N>` numbering continues after existing locals; `topological_sort` orders the
result. Around extraction, `_inline_atomic_assignments` substitutes literal-, symbol- and
local-valued targets into later right-hand sides, and `_reduce_pow_families` names one
primal per non-integer power family (`x**p` with `x**(p±1)`, `x**(2p)`, `x**(2p±1)`),
deriving the rest from it. Both only rewrite; callers' `prune_unused` drops what they
leave unreferenced.

## Dependencies
### Internal
- None. Consumed by every module in `codegen/`, plus
  `parsing/jvp_equations.py` and `parsing/auxiliary_caching.py`.
### External
- `sympy` (only in `from_sympy.py`); `attrs` (`SystemIR`). Stdlib `fractions`.
