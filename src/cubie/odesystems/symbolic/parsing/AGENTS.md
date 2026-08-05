<!-- Parent: ../AGENTS.md -->

# parsing

## Purpose
Front end of the symbolic codegen pipeline. Converts every supported input form —
newline/iterable equation strings, raw SymPy equations, a Python callable, or a BigModel file —
into a frozen `ParsedEquations` container plus an `IndexedBases` symbol map and a system hash.
String and SymPy equations converge on one normalised structural representation
(`normalise.py`); the parser classifies the system and assembles it (`assemble.py`): solved
explicit systems are packaged directly, while DAE constructs (implicit equations, higher-order
or in-expression derivatives, algebraic unknowns) route through
`structural.structural_simplify` — automatically, or forced with `simplify=True`. `parse_input`
is the single entry point used by `SymbolicODE.create`; BigModel loading (`load_bigmodel_file`)
and the Jacobian-vector-product structures (`JVPEquations`, `plan_auxiliary_cache`) used later
by `codegen` also live here.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Star-imports `auxiliary_caching`, `bigmodel`, `jvp_equations`, `parser`; declares `__all__ = ["load_bigmodel_file"]` (the rest is re-exported via star imports). |
| `parser.py` | Orchestrator. `parse_input` dispatches on input type (callable → `function_parser`; symbolic → normalise/classify/assemble); `ParsedEquations` (frozen attrs) partitions equations into state-derivatives/observables/auxiliaries; `EquationWarning`; constants `PARSE_TRANSFORMS`, `KNOWN_FUNCTIONS`, `TIME_SYMBOL`, `DRIVER_SETTING_KEYS`; shared lexing/user-function machinery (`_sanitise_input_math`, `_rename_user_calls`, `_build_sympy_user_functions`, `_inline_nondevice_calls`). |
| `normalise.py` | The single symbolic front end and the SymPy→IR boundary. `normalise_input` parses string, SymPy, or pre-converted IR equations into structural `Equation` objects holding engine-IR expressions with `DerivativeRegistry` derivative symbols (`NormalisedSystem`); `classify_system` labels the result `"explicit"` or `"dae"`. Holds the state-aware LHS rules and symbol inference. SymPy appears only during string parsing, derivative-notation replacement, and non-device user-function inlining; every expression converts to IR before the normaliser returns. |
| `assemble.py` | The two backends, computing on IR pairs throughout. `assemble_explicit` packages an explicit-shaped system directly; `assemble_simplified` runs `structural_simplify` and maps the result back (declaration-order states, residuals paired by state, mass matrix rebuilt over the final order as nested float lists, eliminated-state warnings). Both inline observable definitions into consuming dynamics. |
| `bigmodel.py` | `load_bigmodel_file` — sanitises BigModel symbols, converts equations to IR, classifies values, and calls `parse_input`. |
| `bigmodel_cache.py` | `BigModelCache` — disk LRU of parse results keyed by file content, arguments, and edited values. |
| `jvp_equations.py` | `JVPEquations` (mutable attrs) — holds ordered JVP/auxiliary assignments as engine-IR pairs (JVP outputs are `Arr("jvp", i)` nodes) and derives dependency graphs, op-cost, JVP usage/closure, dependency levels, and slot limits; lazily computes/stores a `CacheSelection`; `cached_partition()` splits into cached/runtime/prepare. |
| `auxiliary_caching.py` | Greedy polynomial cache planner. `CacheGroup`/`CacheSelection` (frozen attrs) and `plan_auxiliary_cache` — grow the cached-leaf set by best marginal runtime saving (any positive marginal while the plan is below `min_ops_threshold`, then `min_ops_threshold` per extra slot), simulating each addition in one linear pass (issue #603). |
| `function_inspector.py` | AST analysis of a callable ODE. `inspect_ode_function` → `FunctionInspection`; `_OdeAstVisitor` collects state/constant accesses, assignments (incl. annotated), calls, unrolls `for` (also inside if-branches), synthesises `IfExp` from if/elif/else, rejects unsupported constructs (`while`/`with`/`try`/`match`/nested `def`/comprehensions; branch bodies raise on statements other than assignments and nested `if`/`for`); `AstToSympyConverter` maps AST nodes to SymPy — resolves user-function calls before `KNOWN_FUNCTIONS` (inlining non-device callables), inlines dxdt-named locals, and (in `strict_names` mode) raises on unknown bare names, suggesting the container access when the name is declared. Extra args used only by bare name are `scalar_params` (SciPy `args=` convention), bound to the like-named declared symbol. |
| `function_parser.py` | `parse_function_input` — bridges `FunctionInspection` to the parser's `(equation_map, funcs, new_params)` triple: builds the symbol map (container accesses search parameters → constants → drivers; undeclared attribute/string accesses infer parameters in non-strict mode with `EquationWarning`), emits auxiliary/observable/dxdt equations, inlines `dx = expr; return [dx]` aliases. `infer_function_states` derives state names from dict-return keys or synthesises them for pure positional access when `states` is omitted. |

## For AI Agents

### parse_input — the entry point
Returns `(index_map, all_symbols, funcs, parsed_equations, fn_hash, simplified)` — a 6-tuple
consumed directly by `SymbolicODE.create` and `bigmodel.load_bigmodel_file`. `simplified` is the
`SimplifiedSystem` when structural simplification ran (it carries the mass matrix for torn
systems) and `None` on the explicit fast path. `_detect_input_type` dispatches to `"string"`,
`"sympy"`, or `"function"` (the function branch imports `function_parser` lazily; callable input
is explicit-only and rejects `simplify=True`). `strict=False` is the default: undeclared RHS
symbols are inferred as parameters; `strict=True` requires every RHS symbol declared and refuses
a stateless system. An LHS assignment defines its symbol, so anonymous auxiliaries are admitted
in both modes. `normalise`/`assemble` are imported inside `parse_input` (the file's established
cycle-breaking pattern, like `function_parser`).

### One normalisation layer, two backends
`normalise_input` handles string and SymPy input with the same state-aware rules: `dX` on the
LHS is a derivative only if `X` is a declared unknown (with no declared states, non-strict `dX`
assignments infer state `X`); `d(x, t)` calls and `sympy.Derivative` (any order, nested) are the
explicit derivative notations and may appear inside expressions; a bare `dX` token on an RHS is
*not* a derivative — it binds to the `dX` assignment emitted for state `X`. Numeric-literal LHS
(`0 = g(...)`) marks an implicit equation. `classify_system` returns `"explicit"` only for fully
solved systems (each declared state exactly one first-order derivative equation, no RHS
derivatives, no repeated or implicit LHS, every declared observable assigned) — anything else
goes through structural simplification, with an `EquationWarning` when the user did not pass
`simplify=True`. States are unknowns everywhere: a declared state assigned algebraically is
*reduced* (eliminated with a warning), not an error, and there is no underived-state→observable
conversion. Observable definitions consumed by the dynamics are inlined on both backends so the
generated dxdt never reads the stale observables buffer. Symbols are created `real=True`
throughout (`TIME_SYMBOL = sp.Symbol("t", real=True)`).

### Hash stability contract
`fn_hash` is computed over the IR pairs' reprs, so identical systems hash identically
regardless of input pathway (string vs SymPy vs IR) — codegen caches key on the hash.
Guard this cross-pathway equality when touching the normaliser or the assemblers; the
IR's deterministic folding is what makes it hold.

### ParsedEquations & JVPEquations
`ParsedEquations` is frozen — build a new one via `from_equations`, don't mutate; its
`_state/_observable/_auxiliary_symbols` fields are exposed through same-named properties.
`JVPEquations` is produced by `codegen.jacobian.generate_analytical_jvp`, not here — this module
only defines the container and its derived metadata; treat its `_*` fields as `init=False` computed
state set in `__attrs_post_init__` (never set them directly).

### Driver settings
Keys in `DRIVER_SETTING_KEYS` (`time`, `driver_sample_period`, `wrap`, `order`) are configuration, not driver
symbols; they're stripped before building driver names and reattached via
`drivers.set_passthrough_defaults`.

### BigModel (optional)
`bigmodelmanip` is imported in a `try/except` and may be `None`; `load_bigmodel_file` raises
`ImportError` at call time when it's absent (never import it at top level unguarded). Numeric Dummy
atoms (e.g. `_0.5`) are converted to `sp.Float`/`sp.Integer`; algebraic equations with a numeric
RHS become constants (or parameters if named), non-numeric ones become observables/auxiliaries.
`BigModelCache` is a disk LRU (≤5 configs per model) under `<cache root>/<model>/`, keyed by
file-content SHA-256 + serialised args in `bigmodel_cache_manifest.json`; the root comes from
`cubie.cache_root.get_cache_root()` (shared with codegen and kernel caches) and entries
invalidate on any content change (whitespace included).

### User functions
Renamed with a trailing underscore during string parsing to dodge SymPy name clashes; device
functions / functions with derivative helpers are wrapped in dynamic `sp.Function` subclasses whose
`fdiff` emits derivative placeholders (`d_<name>` or the provided derivative's `__name__`).
Non-device callables are inlined when they accept SymPy args. `function_parser` intentionally does
**not** map `dx`/`dv` dxdt symbols into the symbol map, so `dx = expr; return [dx]` inlines `expr`
instead of creating a circular reference to the output.

### function_inspector — supported AST subset
A `for` loop is supported only if it can be **fully unrolled** at parse time: the iterable must be
a literal `range(...)` (integer-literal args), a literal list, or a literal tuple, and the target a
simple name. Every other `for` (over a variable, a non-`range` call, non-constant elements, or with
tuple-unpacking) raises `NotImplementedError`, as do `while`, comprehensions, generators, `with`,
`del`, `assert`, `raise`, `global`, `nonlocal`.

### auxiliary_caching
`plan_auxiliary_cache` skips `_cse`-prefixed symbols when simulating removals and rejects an
addition if any removed node still has a live dependent (`_simulate_cached_leaves` returns
`None`). Planning cost is bounded: candidates are capped at a multiple of the slot limit and
each greedy round is one simulation per candidate.

### Testing
`tests/odesystems/symbolic/` (`test_parser`, `test_bigmodel`, `test_bigmodel_cache`,
`test_function_inspector`, `test_function_parser`; `JVPEquations` via `test_jacobian`).
Pure-Python parsing — runs without a GPU; BigModel tests need optional `bigmodelmanip`. See root for
CUDASIM/real-CUDA commands.

## Dependencies
### Internal
- `cubie.odesystems.symbolic.indexedbasemaps` (`IndexedBases`); `cubie.odesystems.symbolic.sym_utils`
  (`hash_system_definition`); `cubie.odesystems.symbolic.symbolicODE` (`SymbolicODE`, lazy in
  `bigmodel.py`); `cubie.odesystems.symbolic.codegen.jacobian` (produces `JVPEquations`; imported by
  callers, not here); `cubie._utils` (`is_devfunc`, `PrecisionDType`),
  `cubie.time_logger.default_timelogger`, `cubie.gui.constants_editor` (lazy).
### External
- `sympy` (symbols, parsing, `cse`, `Function`, `Piecewise`); `attrs` (`ParsedEquations`,
  `JVPEquations`, `CacheGroup`, `CacheSelection`); `bigmodelmanip` (optional); `numpy` (precision dtype
  in the BigModel loader). Stdlib `ast`, `inspect`, `pickle`, `json`, `hashlib`, `re`, `itertools`.
