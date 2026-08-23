<!-- Parent: ../AGENTS.md -->

# parsing

## Purpose
Front end of the symbolic codegen pipeline. Converts every supported input form —
newline/iterable equation strings, raw SymPy equations, a Python callable, or a CellML file —
into a frozen `ParsedEquations` container plus an `IndexedBases` symbol map and a system hash.
All input converges on one normalised structural representation (`normalise.py`), folds its
constant values in as literals, and assembles through `structural.structural_simplify`
(`assemble.py`). `parse_input` is the single entry point used by `SymbolicODE.create`; CellML
loading (`load_cellml_model`) and the Jacobian-vector-product structures (`JVPEquations`,
`plan_auxiliary_cache`) used later by `codegen` also live here.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Star-imports `auxiliary_caching`, `cellml`, `jvp_equations`, `parse_primitives`, `parser`; declares `__all__ = ["load_cellml_model"]` (the rest is re-exported via star imports). |
| `parser.py` | Orchestrator. `parse_input` dispatches on input type (callable → `function_parser` then normalise; symbolic → normalise), builds the `ParsedSystem` checkpoint, and specialises it; `DRIVER_SETTING_KEYS`. |
| `parse_primitives.py` | Shared parse-layer primitives; the leaf module below `normalise`/`assemble`/`function_parser`. `ParsedEquations` (frozen attrs; partitions equations into state-derivatives/observables/auxiliaries and carries the derived `mass_matrix`), `EquationWarning`, `PARSE_TRANSFORMS`, `KNOWN_FUNCTIONS`, `TIME_SYMBOL`, and the lexing/user-function machinery (`_sanitise_input_math`, `_rename_user_calls`, `_build_sympy_user_functions`, `_inline_nondevice_calls`). |
| `parsed_system.py` | The constants-symbolic checkpoint and the constant-specialisation pass. `ParsedSystem.specialise` folds constant values as IR literals into the normalised equations and assembles them, so structure follows values; category moves evolve the checkpoint's `parameters`/`constants` dicts directly (see `SymbolicODE.set_categories`); `ParsedSystem.from_parsed_equations` rebuilds a checkpoint from pre-parsed products for direct `SymbolicODE` construction. |
| `normalise.py` | The single symbolic front end and the SymPy→IR boundary. `normalise_input` parses string, SymPy, or pre-converted IR equations into structural `Equation` objects holding engine-IR expressions with `DerivativeRegistry` derivative symbols (`NormalisedSystem`). Holds the state-aware LHS rules, derivative-token binding, and symbol inference. SymPy appears only during string parsing, derivative-notation replacement, and non-device user-function inlining; every expression converts to IR before the normaliser returns. |
| `assemble.py` | The single assembly backend, computing on IR pairs throughout. `assemble_simplified` runs `structural_simplify` and maps the result into parser products (name-sorted states, residuals paired by state, eliminated-state warnings), inlining observable definitions into consuming dynamics; the mass matrix is rebuilt over that order and attached as `ParsedEquations.mass_matrix`. |
| `cellml.py` | `load_cellml_model` — sanitises CellML symbols, converts equations to IR, classifies values, and calls `parse_input`. |
| `cellml_cache.py` | `CellMLCache` — disk LRU of parse results keyed by file content, arguments, and edited values. |
| `jvp_equations.py` | `JVPEquations` (mutable attrs) — holds ordered JVP/auxiliary assignments as engine-IR pairs (JVP outputs are `Arr("jvp", i)` nodes) and derives dependency graphs, device-weighted op costs (`engine.count_device_ops`), JVP usage/closure, v-dependence (`v_dependent_nodes`), and slot limits; lazily computes/stores a `CacheSelection`; `cached_partition()` splits into cached/runtime/prepare. Canonical consumer views: `jacobian_entry(i, j)` (the graph's `_cubie_codegen_j_<i>_<j>` symbol, `ZERO` when structurally zero; index passed by `generate_analytical_jvp` or derived from the reserved names), `cached_slot_order`, `cached_runtime_assignments()` (cached symbols bound to `cached_aux` slots, everything else by its graph expression), and `prepare_fill_assignments()` (prepare chain plus slot stores). |
| `auxiliary_caching.py` | Min-cut cache planner. `CacheSelection` (frozen attrs) and `plan_auxiliary_cache` — solve a maximum-weight closure (project selection) problem: removing a node earns its device-weighted cost, each cached slot charges `read_price` per operator call, and a removed node stays uncached only when every consumer is removed too; the price rises by bisection when the slot cap binds. Consumers of a cached leaf stay runtime and read the buffer slot. |
| `function_inspector.py` | AST analysis of a callable ODE. `inspect_ode_function` → `FunctionInspection`; `_OdeAstVisitor` collects state/constant accesses, assignments (incl. annotated), calls, unrolls `for` (also inside if-branches), synthesises `IfExp` from if/elif/else, rejects unsupported constructs (`while`/`with`/`try`/`match`/nested `def`/comprehensions; branch bodies raise on statements other than assignments and nested `if`/`for`); `AstToSympyConverter` maps AST nodes to SymPy — resolves user-function calls before `KNOWN_FUNCTIONS` (inlining non-device callables), inlines dxdt-named locals, and (in `strict_names` mode) raises on unknown bare names, suggesting the container access when the name is declared. Extra args used only by bare name are `scalar_params` (SciPy `args=` convention), bound to the like-named declared symbol. |
| `function_parser.py` | `parse_function_input` — bridges `FunctionInspection` to the parser's `(equation_map, funcs, new_params)` triple: builds the symbol map (container accesses search parameters → constants → drivers; undeclared attribute/string accesses infer parameters in non-strict mode with `EquationWarning`), emits auxiliary/observable/dxdt equations, inlines `dx = expr; return [dx]` aliases. `infer_function_states` derives state names from dict-return keys or synthesises them for pure positional access when `states` is omitted. |

## For AI Agents

### parse_input — the entry point
Returns `(index_map, all_symbols, funcs, parsed_equations, fn_hash,
parsed_system)` — a 6-tuple consumed directly by `SymbolicODE.create` and
`cellml.load_cellml_model`. The derived mass matrix rides on
`parsed_equations.mass_matrix` (`None` for solved systems). `parsed_system` is the
constants-symbolic checkpoint (`parsed_system.py`); assembly runs *inside* its
`specialise`, on the constant-folded equations, so structure can change with
constant values. `_detect_input_type` dispatches to `"string"`, `"sympy"`, or
`"function"`, routing the resulting IR pairs through `normalise_input` on every
pathway. `strict=False` is the default: undeclared RHS symbols are inferred as
parameters; `strict=True` requires every RHS symbol declared and refuses a
stateless system. An LHS assignment defines its symbol, so anonymous auxiliaries
are admitted in both modes.

### One normalisation layer, one assembler
`normalise_input` handles every input with the same state-aware rules: `dX` on the
LHS is a derivative only if `X` is a declared unknown (with no declared states, non-strict `dX`
assignments infer state `X`); `d(x, t)` calls and `sympy.Derivative` (any order, nested) are the
explicit derivative notations and may appear inside expressions; any other bare, unassigned
`dX` token — inside an expression LHS (`c*dx = f(...)`, an implicit equation) or on an RHS —
binds to the derivative of unknown `X`. A numeric or expression LHS marks an implicit
equation. Every system assembles through structural simplification on its constant-folded
equations, so a scaled-derivative row like `Cs*dU = g(...)` is algebraic when `Cs` is zero
and differential otherwise. States are unknowns everywhere: a declared state assigned
algebraically is *reduced* (eliminated with a warning), not an error, and there is no
underived-state→observable conversion. Observable definitions consumed by the dynamics are
inlined so the generated dxdt never reads the stale observables buffer. Symbols are created
`real=True` throughout (`TIME_SYMBOL = sp.Symbol("t", real=True)`).

### Hash stability contract
`fn_hash` is computed over the IR pairs' reprs after constant values fold in, so
identical systems with identical constant values hash identically regardless of input
pathway (string vs SymPy vs IR), and different constant values hash differently —
codegen caches key on the hash. Guard this cross-pathway equality when touching the
normaliser, the assemblers, or the specialisation pass; the IR's deterministic folding
is what makes it hold.

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

### CellML (optional)
`cellmlmanip` is imported in a `try/except` and may be `None`; `load_cellml_model` raises
`ImportError` at call time when it's absent (never import it at top level unguarded). Numeric Dummy
atoms (e.g. `_0.5`) are converted to `sp.Float`/`sp.Integer`; algebraic equations with a numeric
RHS become constants (or parameters if named), non-numeric ones become observables/auxiliaries.
`CellMLCache` is a disk LRU (≤5 configs per model) under `<cache root>/<model>/`, keyed by
file-content SHA-256 + serialised args in `cellml_cache_manifest.json`; the root comes from
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
`plan_auxiliary_cache` caches any v-independent node — consumers of a cached leaf stay in
the runtime body and read the value from the `cached_aux` buffer slot, and `_cse` locals are
first-class candidates. Selection is a min-cut over the whole graph: a removed node earns its
device-weighted cost (`engine.count_device_ops`), a cached slot charges `read_price` per
operator call, and a removed node stays uncached only when every consumer is removed.
Over-cap plans are re-solved at bisected higher prices and trimmed to `cache_slot_limit`;
`CacheSelection.read_price` records the price used, `duplicate_cost` the work computed in
both the fill and the runtime body.

### Testing
`tests/odesystems/symbolic/` (`test_parser`, `test_cellml`, `test_cellml_cache`,
`test_function_inspector`, `test_function_parser`; `JVPEquations` via `test_jacobian`).
Pure-Python parsing — runs without a GPU; CellML tests need optional `cellmlmanip`. See root for
CUDASIM/real-CUDA commands.

## Dependencies
### Internal
- `cubie.odesystems.symbolic.indexedbasemaps` (`IndexedBases`); `cubie.odesystems.symbolic.sym_utils`
  (`hash_system_definition`); `cubie.odesystems.symbolic.symbolicODE` (`SymbolicODE`, lazy in
  `cellml.py`); `cubie.odesystems.symbolic.codegen.jacobian` (produces `JVPEquations`; imported by
  callers, not here); `cubie._utils` (`is_devfunc`, `PrecisionDType`),
  `cubie.time_logger.default_timelogger`, `cubie.gui.constants_editor` (lazy).
### External
- `sympy` (symbols, parsing, `cse`, `Function`, `Piecewise`); `attrs` (`ParsedEquations`,
  `JVPEquations`, `CacheSelection`); `cellmlmanip` (optional); `numpy` (precision dtype
  in the CellML loader). Stdlib `ast`, `inspect`, `pickle`, `json`, `hashlib`, `re`, `itertools`.
