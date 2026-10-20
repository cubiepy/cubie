<!-- Parent: ../AGENTS.md -->

# parsing

## Purpose
Front end of the symbolic codegen pipeline. Converts every supported input form —
newline/iterable equation strings, raw SymPy equations, a Python callable, or a CellML file —
into a `ParsedSystem`, which `ParsedSystem.assemble` turns into a `SpecialisedSystem` for
one `ParameterSet`: the buffer layout, a frozen `ParsedEquations` container, the
callables, a system hash and the mass matrix.
All input converges on one normalised structural representation (`normalise.py`), folds its
fixed parameter values in as literals, and assembles through `structural.structural_simplify`
(`assemble.py`). `parse_system` is the single entry point used by `SymbolicODE.create`
and CellML loading. CellML
loading (`load_cellml_model`) and the Jacobian-vector-product structures (`JVPEquations`,
`plan_auxiliary_cache`) used later by `codegen` also live here.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Star-imports `auxiliary_caching`, `cellml`, `jvp_equations`, `parse_primitives`, `parser`; declares `__all__ = ["load_cellml_model"]` (the rest is re-exported via star imports). |
| `parser.py` | Orchestrator. `parse_system` dispatches on input type (callable → `function_parser` then normalise; symbolic → normalise) and returns the `ParsedSystem` checkpoint. |
| `parse_primitives.py` | Shared parse-layer primitives; the leaf module below `normalise`/`assemble`/`function_parser`. `ParsedEquations` (frozen attrs; partitions equations into state-derivatives/observables/auxiliaries and holds the derived `mass_matrix`), `EquationWarning`, `PARSE_TRANSFORMS`, `KNOWN_FUNCTIONS`, `TIME_SYMBOL`, and the lexing/user-function machinery (`_sanitise_input_math`, `_rename_user_calls`, `_build_sympy_user_functions`, `_inline_nondevice_calls`). |
| `parsed_system.py` | `ParsedSystem`, the parsed system with every parameter left as a symbol. `declared_parameters` is every parameter with its default and unit, in sorted order. `ParsedSystem.specialise(swept, values)` builds the batch's `ParameterSet` from them, substitutes the fixed values as IR literals and assembles the equations, with the swept parameters read from the parameters array, so structure follows values. It returns a `SpecialisedSystem`. State, observable, parameter and driver names under the reserved codegen prefix raise. |
| `normalise.py` | The single symbolic front end and the SymPy→IR boundary. `normalise_input` parses string, SymPy, or pre-converted IR equations into `(lhs, rhs)` engine-IR pairs with `DerivativeRegistry` derivative symbols (`NormalisedSystem`); the registry reserves every name in the input, so derivative symbols (`x_t`, `x_tt`, ...) never collide with a user symbol. Holds the state-aware LHS rules, derivative-token binding, and symbol inference. SymPy appears only during string parsing, derivative-notation replacement, and non-device user-function inlining; every expression converts to IR before the normaliser returns. |
| `assemble.py` | The single assembly backend, computing on IR pairs throughout. `assemble_simplified` runs `structural_simplify` and packages the result as a `SpecialisedSystem` (name-sorted states, observables and drivers, the batch's `ParameterSet` and the driver-derivative slots in a `SystemLayout`, residuals paired by state, eliminated-state warnings), inlining observable definitions into consuming dynamics with `fixpoint_sub`; when residuals exist the mass matrix is built over that order from the differential states and attached as `ParsedEquations.mass_matrix`. |
| `cellml.py` | `load_cellml_model` — sanitises CellML symbols, converts equations to IR, classifies values, and calls `parse_system`. |
| `cellml_cache.py` | `CellMLCache` — disk LRU of `ParsedSystem`s keyed by file content, arguments, and edited values. |
| `jvp_equations.py` | `JVPEquations` (mutable attrs) — holds ordered JVP/auxiliary assignments as engine-IR pairs (JVP outputs are `Arr("jvp", i)` nodes) and derives dependency graphs, device-weighted op costs (`engine.count_device_ops`), JVP usage/closure, v-dependence (`v_dependent_nodes`), and slot limits; lazily computes/stores a `CacheSelection`; `cached_partition()` splits into cached/runtime/prepare. Canonical consumer views: `jacobian_entry(i, j)` (the graph's `_cubie_codegen_j_<i>_<j>` symbol, `ZERO` when structurally zero; index passed by `generate_analytical_jvp` or derived from the reserved names), `cached_slot_order`, `cached_runtime_assignments()` (cached symbols bound to `cached_aux` slots, everything else by its graph expression), and `prepare_fill_assignments()` (prepare chain plus slot stores). |
| `auxiliary_caching.py` | Min-cut cache planner. `CacheSelection` (frozen attrs) and `plan_auxiliary_cache` — solve a maximum-weight closure (project selection) problem: removing a node earns its device-weighted cost, each cached slot charges `read_price` per operator call, and a removed node stays uncached only when every consumer is removed too; the price rises by bisection when the slot cap binds. Consumers of a cached leaf stay runtime and read the buffer slot. |
| `function_inspector.py` | AST analysis of a callable ODE. `inspect_ode_function` → `FunctionInspection`; `_OdeAstVisitor` collects state/parameter accesses, assignments (incl. annotated), calls, unrolls `for` (also inside if-branches), synthesises `IfExp` from if/elif/else, rejects unsupported constructs (`while`/`with`/`try`/`match`/nested `def`/comprehensions; branch bodies raise on statements other than assignments and nested `if`/`for`); `AstToSympyConverter` maps AST nodes to SymPy — resolves user-function calls before `KNOWN_FUNCTIONS` (inlining non-device callables), inlines dxdt-named locals, and (in `strict_names` mode) raises on unknown bare names, suggesting the container access when the name is declared. Extra args used only by bare name are `scalar_params` (SciPy `args=` convention), bound to the like-named declared symbol. |
| `function_parser.py` | `parse_function_input` — bridges `FunctionInspection` to the parser's `(equation_map, funcs, new_params)` triple from the declared state, parameter, driver and observable names: builds the symbol map (it reads parameters by name through the first container argument after the state and drivers through the second, and in non-strict mode it infers a parameter from an undeclared attribute or string access on the first, with an `EquationWarning`), emits auxiliary/observable/dxdt equations, inlines `dx = expr; return [dx]` aliases. `infer_function_states` derives state names from dict-return keys or synthesises them for pure positional access when `states` is omitted. |

## parse_system and assemble
`parse_system` returns the `ParsedSystem`, with every parameter left as a symbol
(`parsed_system.py`). The `SystemSpecialiser` child of `SymbolicODE` calls
`ParsedSystem.assemble(parameters)` for the batch's `ParameterSet`, which returns a
`SpecialisedSystem`: the layout, equations, callables, source hash and mass matrix.
Structural simplification runs after the values are substituted, so
structure can change with fixed values. Inferred parameters join
`parsed_system.parameters`.
`_detect_input_type` dispatches to `"string"`, `"sympy"` or `"function"`, and every
pathway goes through `normalise_input`. `strict=False` (default) infers undeclared RHS
symbols as parameters; `strict=True` requires every RHS symbol declared and refuses a
stateless system. An LHS assignment defines its symbol in both modes.

## Normalisation and assembly
`normalise_input` applies one set of state-aware rules:
- `dX` on the LHS is a derivative only if `X` is a declared unknown (with no declared
  states, non-strict `dX` assignments infer state `X`).
- `d(x, t)` and `sympy.Derivative` (any order, nested) are explicit derivatives and may
  appear inside expressions; any other unassigned `dX` token binds to the derivative of
  unknown `X`.
- A numeric or expression LHS marks an implicit equation.
- Assembly runs structural simplification after fixed values are substituted, so
  `Cs*dU = g(...)` is algebraic when `Cs` is zero and differential otherwise.
- A declared state assigned algebraically is eliminated with a warning.
- Observable definitions the dynamics use are inlined, so dxdt never reads the
  observables buffer.
- Symbols are `real=True` (`TIME_SYMBOL = sp.Symbol("t", real=True)`).

## Hash stability
`fn_hash` is computed over the IR pairs' reprs after fixed values fold in: identical
systems with identical swept names and fixed values hash identically on every input
pathway, and different fixed values hash differently. Keep that equality when changing the normaliser,
assemblers or specialisation.

## ParsedEquations and JVPEquations
`ParsedEquations` is frozen; build a new one with `from_equations`. `JVPEquations` is
produced by `codegen.jacobian.generate_analytical_jvp`; its `_*` fields are set in
`__attrs_post_init__`.

## Drivers
A driver dict gives the driver names; the system keeps no driver values. Sampled
driver data is a Solver setting (`DriverSamples`), not part of the system.
`ParsedEquations.driver_derivatives` maps each driver derivative read to `(driver, order)`;
we place them after the drivers in the drivers buffer.

## CellML
`cellmlmanip` is imported under `try/except` and may be `None`; `load_cellml_model` raises
`ImportError` at call time without it. Numeric Dummy atoms (`_0.5`) become
`sp.Float`/`sp.Integer`; algebraic equations with a numeric RHS become parameters, others
observables or auxiliaries. `CellMLCache` is a disk LRU (≤5
configs per model) under `<cache root>/<model>/`, keyed by file-content SHA-256 plus
serialised args in `cellml_cache_manifest.json`; any content change (whitespace
included) invalidates. We cache the `ParsedSystem` before specialisation, so one entry
serves every choice of swept parameters and every precision: a hit builds the system
for the requested `swept_parameters` and precision.

## User functions
String parsing renames user functions with a trailing underscore to avoid SymPy
clashes. Device functions and functions with derivative helpers are wrapped in dynamic
`sp.Function` subclasses whose `fdiff` emits `d_<name>` (or the given derivative's
`__name__`). A `user_function_derivatives` value is one callable or a list in increasing
order; `derivative_names` maps each function and helper to the next helper. Give
every helper its own `__name__`, different from every user function's name, because the
generated module looks helpers up by name.
`check_derivative_orders` raises when a function is differentiated past its last
helper. Non-device callables are inlined when they accept SymPy arguments.
`function_parser` leaves dxdt symbols out of the symbol map, so `dx = expr; return [dx]`
inlines `expr`.

## function_inspector
A `for` loop must unroll fully at parse time: the iterable a literal `range(...)` with
integer-literal arguments, or a literal list or tuple, and the target a simple name. Any
other `for`, and `while`, comprehensions, generators, `with`, `del`, `assert`, `raise`,
`global`, `nonlocal`, raise `NotImplementedError`.

## auxiliary_caching
`plan_auxiliary_cache` caches v-independent nodes: consumers of a cached leaf stay in the
runtime body and read its `cached_aux` slot, and `_cse` locals are candidates. Selection
is a min-cut over the whole graph: a removed node earns its device-weighted cost
(`engine.count_device_ops`), a cached slot costs `read_price` per operator call, and a
removed node stays uncached only when every consumer is removed. Over-cap plans are
re-solved at bisected higher prices and trimmed to `cache_slot_limit`;
`CacheSelection.read_price` records the price used and `duplicate_cost` the work done in
both the fill and the runtime body.

## Dependencies
### Internal
- `cubie.odesystems.layout` (`SystemLayout`); `cubie.odesystems.variables`
  (`VariableSet`, `State`, `Variable`); `cubie.odesystems.symbolic.sym_utils`
  (`hash_system_definition`); `cubie.odesystems.symbolic.symbolicODE` (`SymbolicODE`, lazy in
  `cellml.py`); `cubie.odesystems.symbolic.codegen.jacobian` (produces `JVPEquations`; imported by
  callers, not here); `cubie._utils` (`is_devfunc`, `PrecisionDType`),
  `cubie.time_logger.default_timelogger`, `cubie.gui.parameters_editor` (lazy).
### External
- `sympy` (symbols, parsing, `cse`, `Function`, `Piecewise`); `attrs` (`ParsedEquations`,
  `JVPEquations`, `CacheSelection`); `cellmlmanip` (optional); `numpy` (precision dtype
  in the CellML loader). Stdlib `ast`, `inspect`, `pickle`, `json`, `hashlib`, `re`, `itertools`.
