<!-- Parent: ../AGENTS.md -->

# algorithms

## Purpose
Step-function factories: each is a `CUDAFactory` subclass (`BaseAlgorithmStep`) that
JIT-compiles one device function advancing a single ODE/SDE step. Tableau methods pair
with a `*_tableaus.py` module of Butcher coefficients. Covers explicit Euler, explicit
/ diagonally-implicit / fully-implicit Runge-Kutta (ERK/DIRK/FIRK), Rosenbrock-W,
backward Euler (plain + predictor-corrector), and Crank-Nicolson. Implicit methods own
a `NewtonKrylov` or `LinearSolver` from `../matrix_free_solvers/`. `get_algorithm_step()`
resolves a name or `ButcherTableau` to the right factory.

See `CUDAFactory` (repo root) for the build/cache/`update`, buffer-registry, and
attrs-config mechanics; CUDA-authoring *optimisation* patterns are in
`../../writing_cuda_functions.md`. This file documents the algorithm specifics.

> **Numerical correctness is critical.** Every device function here is
> verified against a plain CPU reference implementation under
> `tests/integrators/cpu_reference/algorithms.py`. Any change to an
> algorithm/solver device function's numerical behaviour MUST be
> replicated in its CPU reference counterpart.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public surface: `get_algorithm_step()`, `_ALGORITHM_REGISTRY`, `_TABLEAU_REGISTRY_BY_ALGORITHM`, `resolve_alias`/`resolve_supplied_tableau`; re-exports every step class and tableau registry. |
| `base_algorithm_step.py` | Core abstractions: `ButcherTableau` (typed accessors, FSAL/error detection), `BaseStepConfig`, `StepCache`, `StepControlDefaults`, `BaseAlgorithmStep` (CUDAFactory base), `ALL_ALGORITHM_STEP_PARAMETERS`. |
| `ode_explicitstep.py` | `ExplicitStepConfig` + `ODEExplicitStep`: `build()` delegates to `build_step` (no solver); `is_implicit` → `False`. |
| `ode_implicitstep.py` | `ImplicitStepConfig` (beta, gamma, preconditioner_order) + `ODEImplicitStep`: owns a `NewtonKrylov`/`LinearSolver`, requests operator/preconditioner/residual helpers as immutable `SolverHelperRequest`s, resolves `preconditioner_type` into concrete kinds, routes solver-param updates; `is_implicit` → `True`. |
| `explicit_euler.py` | `ExplicitEulerStep`: forward Euler, order 1, fixed-step. |
| `generic_erk.py` | `ERKStep` + `ERKStepConfig`: streamed-accumulator explicit RK; FSAL caching; controller auto-selected from `tableau.has_error_estimate`. |
| `generic_erk_tableaus.py` | `ERKTableau` + ERK sets (Heun, Ralston, Bogacki-Shampine, Dormand-Prince 5(4)/8(5,3), RK4, Cash-Karp, Fehlberg, Tsit5, Vern7); `ERK_TABLEAU_REGISTRY`, `DEFAULT_ERK_TABLEAU`. |
| `generic_dirk.py` | `DIRKStep` + `DIRKStepConfig`: diagonally-implicit RK, one Newton solve per implicit stage, stage-skipping for explicit stages, FSAL caching, dense-predictor warm starts. |
| `generic_dirk_tableaus.py` | `DIRKTableau` (adds `diagonal()`, validates `c[i] == sum(a[i])`) + tableaus (implicit midpoint, trapezoidal/ESDIRK, Kvaerno 3/5, SDIRK_2_2, L-stable DIRK3 default, L-stable SDIRK4). |
| `generic_firk.py` | `FIRKStep` + `FIRKStepConfig`: fully-implicit RK; all stages as one coupled `n*stages` Newton system; dense-predictor warm starts; Kahan-summed output accumulation. |
| `generic_firk_tableaus.py` | `FIRKTableau` + Gauss-Legendre-2 (default) and Radau IIA-5; `compute_embedded_weights_radauIIA`. |
| `generic_rosenbrock_w.py` | `GenericRosenbrockWStep` + `RosenbrockWStepConfig`: linearly-implicit Rosenbrock-W using a cached Jacobian and a **linear** (not Newton) solve per stage; needs `driver_del_t` and time-derivative helpers. |
| `generic_rosenbrockw_tableaus.py` | `RosenbrockTableau` (adds `C`, `gamma`, `gamma_stages`) + ROS3P (default), RODAS3P, SciML Rosenbrock23. RODAS4P/5P and ode23s 2(3) are commented-out / non-working. |
| `backwards_euler.py` | `BackwardsEulerStep` + config: single-stage implicit, order 1, fixed-step; persistent `increment_cache` warm-starts Newton. |
| `backwards_euler_predict_correct.py` | `BackwardsEulerPCStep`: subclass adding an explicit forward-Euler predictor before the Newton corrector. |
| `crank_nicolson.py` | `CrankNicolsonStep` + config: order-2 adaptive implicit; two implicit solves per step (CN + backward Euler), the difference giving the embedded error estimate. |

## For AI Agents

### Device step contract (the caller — `IVPLoop` — must match)
- **Signature (16 positional args, identical across every algorithm):**
  `(state, proposed_state, parameters, driver_coefficients, drivers_buffer,
  proposed_drivers, observables, proposed_observables, error, dt_scalar, time_scalar,
  first_step_flag, accepted_flag, shared, persistent_local, counters)`.
- Non-adaptive (fixed-step) methods receive a **zero-length `error` slice** (no
  embedded estimate); adaptive methods get a length-`n` `error` array.
- **Returns an `int32` status code** from the `CUBIE_RESULT_CODES` vocabulary
  (`cubie/result_codes.py`, captured as device closure constants). For implicit methods it
  OR-combines the solver status from each stage's solve (`status_code |= solver_status`);
  explicit methods return `SUCCESS`. Iteration counts are written to the `counters` array
  (the last argument, forwarded to the solver).
- The commented-out `@cuda.jit` signature block atop each kernel documents the intended
  types; keep it in sync but it stays commented.

### Factory & dispatch
- Subclasses implement **`build_step(...)`** (not `build()` — the bases provide that),
  returning a `StepCache(step=..., nonlinear_solver=...)`; the compiled step is exposed
  via the `step_function` property.
- `get_algorithm_step(precision, settings, **kwargs)` requires `settings["algorithm"]`
  — a name string or a `ButcherTableau` instance. Names resolve via
  `_TABLEAU_REGISTRY_BY_ALGORITHM` (`resolve_alias`); tableau instances dispatch by
  type (`resolve_supplied_tableau`). For the **tableau methods**, the bare keys
  `"erk"`, `"dirk"`, `"firk"`, `"rosenbrock"` use the class-default tableau; the
  **non-tableau methods** `"euler"`, `"backwards_euler"`, `"backwards_euler_pc"`,
  `"crank_nicolson"` are fixed schemes with no tableau.
- `StepControlDefaults` are chosen from `tableau.has_error_estimate` (fixed when no
  embedded estimate exists; otherwise PI for explicit RK, order-dependent PI
  defaults for adaptive DIRK, and Gustafsson for FIRK/Rosenbrock-W/Crank-Nicolson).
  **Errorless tableaus must use a fixed controller** — constructors enforce
  this; never pair an adaptive controller with an errorless tableau. Controller-defaults
  dicts must never contain keys that are also algorithm parameters: on hot-swap
  the defaults merge into the shared `updates_dict`
  and would overwrite the algorithm's own setting.
- **`update` additions:** new keywords must be added to `ALL_ALGORITHM_STEP_PARAMETERS`
  (`base_algorithm_step.py`) or `update` rejects them; `ODEImplicitStep` routes solver
  params to its owned solver via `_LINEAR_SOLVER_PARAMS` / `_NEWTON_KRYLOV_PARAMS`.

### Tableaus
- **Adding a tableau:** append to the relevant `*_tableaus.py` registry; `__init__.py`'s
  registry-merge loops pick it up as a valid `algorithm` name. Coefficients are validated
  in `ButcherTableau.__attrs_post_init__` (`b` and `b_hat` must sum to 1).
- **Tableau-derived compile-time optimisations** (do not hand-roll — they come from
  `ButcherTableau` properties): `b_matches_a_row` / `b_hat_matches_a_row` replace
  streaming accumulation with a direct copy when a stage state already equals the
  solution / embedded estimate; `first_same_as_last` / `can_reuse_accepted_start` (FSAL)
  enable stage-0 RHS reuse.

### Explicit vs implicit
- **Explicit** (`ODEExplicitStep`, no solver): `ExplicitEulerStep`, `ERKStep`.
- **Implicit** (`ODEImplicitStep`, owns a solver): `BackwardsEulerStep`,
  `BackwardsEulerPCStep`, `CrankNicolsonStep`, `DIRKStep`, `FIRKStep`,
  `GenericRosenbrockWStep`. All use **Newton-Krylov except `GenericRosenbrockWStep`**,
  which is linearly-implicit and constructs a `LinearSolver` directly
  (no Newton iteration). The class attribute `is_linear` (`True` on
  `GenericRosenbrockWStep`, `False` elsewhere) exposes the distinction.

### Registered buffers
- Each step registers its working buffers (e.g. DIRK `stage_base`/`accumulator`,
  CN `cn_dxdt`). Buffers with disjoint lifetimes are aliased to share storage — e.g.
  CN's `base_state` aliases `error`, and DIRK's `stage_base` aliases `accumulator`.
  Implicit steps additionally pull **child allocators** for their owned solver via
  `get_child_allocators(self, self.solver, ...)`.
- **Rosenbrock auxiliary-cache sizing:** `register_buffers` registers
  `cached_auxiliaries` at size 0; `build_implicit_helpers()` resizes it via
  `update_buffer` from `prepare_jac`'s `HelperResult.cached_auxiliary_count`.

### Dense stage prediction (FIRK and DIRK)
Both steps own a `DenseStagePredictor` (`../stage_predictors.py`) child that
turns the previous accepted step's stage increments into the next step's
Newton starting guesses. The algorithm owns eligibility: it keeps the
persistent `previous_step_size` scalar and folds first-step, rejection, and
the tableau's per-precision `dense_prediction_ratio_*` ceiling (calibrated
by `benchmarks/dense_prediction_ratio_sweep.py`) into a flag the predictor
commits per lane via predicated `selp`. Tableau-derived behaviour lives on
the tableaus: `prediction_sample_stages` (one sample per distinct node),
`first_stage_is_explicit` (an explicit first stage is never predicted; its
`dt*f` sample still enters DIRK's history), and DIRK's
`prediction_source_stages` (a repeated stage time starts from the earlier
same-time stage's row). `predictor_function` pipes through compile settings
like `solver_function`.

### Step-size control order
`controller_order` = `min(order, embedded_order)`; tableaus with `b_hat`
declare `embedded_order` (validated together). `SingleIntegratorRunCore`
feeds it to controllers as `algorithm_order`; `order` stays classical. FIRK
smoothing swaps in `RadauIIATableau.smoothed_embedded_order` (stage count).

### Smoothed error estimate (DIRK, FIRK, Rosenbrock-W)
`use_smoothed_error` filters the embedded estimate through
`(I - smoothing_gamma * h * J)^-1`, one extra linear solve per step. The step's
`smooth_error` property = request AND `tableau.supports_smoothed_error` AND
adaptive; when False the smoothing code compiles out, and an unsupported
request warns. `smoothing_gamma` is `a[-1][-1]` on `ButcherTableau`, the
reciprocal real eigenvalue of `inv(a)` on `RadauIIATableau`. DIRK and
Rosenbrock-W reuse their own linear solver and buffers. FIRK owns a width-`n`
`error_solver` child whose shared window aliases `solver_shared`; its estimate
`sum_i d_i*k_i - gamma*h*f(y_n)` takes `d` from
`RadauIIATableau.smoothed_error_weights` and always accumulates.

### Solver helpers arrive as requests
Implicit steps derive an immutable `SolverHelperRequest`
(`cubie.odesystems.solver_helpers`) from their compile settings and call
`get_solver_helper_fn(request).device_function`. `_resolve_preconditioner`
maps `preconditioner_type` to concrete kinds through
`resolve_preconditioner_kind`/`resolve_chained_kind` (`cached=True` for
Rosenbrock-W, `n_stage=True` for FIRK); a multi-type sequence becomes one
chained-kind request. `ODEImplicitStep.update` refreshes the step settings
first, then adds the derived `solver_width` (the coupled all-stages length
for FIRK; `n` elsewhere) for the solver subtree. `ODEImplicitStep.build()` runs `build_implicit_helpers()`
**before** reading `compile_settings` — the helper refresh replaces the
snapshot.

### FSAL warp-coherence
- FSAL stage-0 RHS reuse is guarded by `all_sync(activemask(), accepted_flag != 0)` so
  the cache is reused only when every active warp lane agrees. (General warp-coherence
  guidance: `../../writing_cuda_functions.md`.)

### Testing
Tests under `tests/integrators/algorithms/`:
`test_step_algorithms.py`, `test_init.py`, `test_ode_explicitstep.py`,
`test_ode_implicitstep.py`, `test_tableau_properties.py`, `test_*_tableaus.py`,
`test_last_step_caching_integration.py`.

## Dependencies
### Internal
- `cubie.CUDAFactory` — step/config/cache base classes.
- `cubie.buffer_registry` — buffer allocation for shared/local memory.
- `cubie.integrators.matrix_free_solvers` — `NewtonKrylov`, `LinearSolver` (owned by
  implicit steps).
- `cubie.cuda_simsafe` — `all_sync`, `activemask` (FSAL warp votes).
- `cubie._utils` — `build_config`, `PrecisionDType`, validators.
- Solver-helper device functions come from the ODE system via `get_solver_helper_fn`
  (`stage_residual`, `linear_operator`, `neumann_preconditioner`, `n_stage_*`,
  `prepare_jac`, `time_derivative_rhs`, …).

### External
- `numba` (`cuda`, `int32`); `attrs`; `numpy` (coefficient math, embedded-weight solves).
