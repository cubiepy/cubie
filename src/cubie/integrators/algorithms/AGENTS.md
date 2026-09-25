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

> **Numerical correctness is critical.** Every device function here is
> verified against a plain CPU reference implementation under
> `tests/integrators/cpu_reference/algorithms.py`. Any change to an
> algorithm/solver device function's numerical behaviour MUST be
> replicated in its CPU reference counterpart.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public surface: `get_algorithm_step()`, `_ALGORITHM_REGISTRY`, `_TABLEAU_REGISTRY_BY_ALGORITHM`, `resolve_alias`/`resolve_supplied_tableau`; re-exports every step class and tableau registry. |
| `base_algorithm_step.py` | Core abstractions: `ButcherTableau` (typed accessors, FSAL/error detection, per-tableau `defaults`), `BaseStepConfig`, `StepCache`, `AlgorithmDefaults`, `BaseAlgorithmStep` (CUDAFactory base), `ALL_ALGORITHM_STEP_PARAMETERS`. |
| `ode_explicitstep.py` | `ExplicitStepConfig` + `ODEExplicitStep`: `build()` delegates to `build_step` (no solver); `is_implicit` → `False`. |
| `ode_implicitstep.py` | `ImplicitStepConfig` (operator_beta, operator_gamma, preconditioner_order, `preconditioner_type` validated against `PRECONDITIONER_ROLES`) + `ODEImplicitStep`: owns a `NewtonKrylov`/`LinearSolver`, requests operator/preconditioner/residual helpers by role and variant name through `get_solver_helper_fn`, routes solver-param updates; `is_implicit` → `True`. |
| `explicit_euler.py` | `ExplicitEulerStep`: forward Euler, order 1, fixed-step. |
| `generic_erk.py` | `ERKStep` + `ERKStepConfig`: streamed-accumulator explicit RK; FSAL caching; controller auto-selected from `tableau.has_error_estimate`. |
| `generic_erk_tableaus.py` | `ERKTableau` + ERK sets (Heun, Ralston, Bogacki-Shampine, Dormand-Prince 5(4)/8(5,3), RK4, Cash-Karp, Fehlberg, Tsit5, Vern7); `ERK_TABLEAU_REGISTRY`, `DEFAULT_ERK_TABLEAU`. |
| `generic_dirk.py` | `DIRKStep` + `DIRKStepConfig`: diagonally-implicit RK, one Newton solve per implicit stage, stage-skipping for explicit stages, FSAL caching, dense-predictor warm starts. |
| `generic_dirk_tableaus.py` | `DIRKTableau` (adds `diagonal()`, validates `c[i] == sum(a[i])`) + tableaus (implicit midpoint, trapezoidal/ESDIRK, Kvaerno 3/5, SDIRK_2_2, L-stable DIRK3 default, L-stable SDIRK4). |
| `generic_firk.py` | `FIRKStep` + `FIRKStepConfig`: fully-implicit RK; all stages as one coupled `n*stages` Newton system; dense-predictor warm starts; Kahan-summed output accumulation. |
| `generic_firk_tableaus.py` | `FIRKTableau` + `RadauIIATableau` (adds the smoothed estimate, gated on `inv(a)` having a sole real eigenvalue — odd stage counts only); Gauss-Legendre 2 (default, errorless) and 4, Radau IIA 3/5/9; `compute_embedded_weights` (moment conditions over any node set). |
| `generic_rosenbrock_w.py` | `GenericRosenbrockWStep` + `RosenbrockWStepConfig`: linearly-implicit Rosenbrock-W using a cached Jacobian and a **linear** (not Newton) solve per stage; needs `driver_derivative_fn` and time-derivative helpers. |
| `generic_rosenbrockw_tableaus.py` | `RosenbrockTableau` (adds `C`, `gamma`, `gamma_stages`) + ROS3P (default), RODAS3P, SciML Rosenbrock23. RODAS4P/5P and ode23s 2(3) are commented-out / non-working. |
| `backwards_euler.py` | `BackwardsEulerStep` + config: single-stage implicit, order 1, fixed-step; persistent `increment_cache` warm-starts Newton. |
| `backwards_euler_predict_correct.py` | `BackwardsEulerPCStep`: subclass adding an explicit forward-Euler predictor before the Newton corrector. |
| `crank_nicolson.py` | `CrankNicolsonStep` + config: order-2 adaptive implicit; two implicit solves per step (CN + backward Euler), the difference giving the embedded error estimate. |

## Device step contract (`IVPLoop` must match)
- Signature, identical for every algorithm: `(state, proposed_state, parameters,
  driver_coefficients, drivers_buffer, proposed_drivers, observables,
  proposed_observables, error, dt_scalar, time_scalar, first_step_flag, accepted_flag,
  shared, persistent_local, counters)`.
- `error` has length `n_states` only when `uses_error` (`has_error_estimate and
  is_adaptive`; `is_adaptive` comes from the controller); otherwise it is zero-length,
  the estimate compiles out and `error_weights` returns zeros. Crank–Nicolson overrides
  `uses_error` to `True`.
- Returns an `int32` status (`../AGENTS.md` lists the codes). Implicit steps OR in each
  stage solve's status; explicit steps return `SUCCESS`. The loop zeroes `counters`
  before every step and every solve adds to it: `[0]` is the step's Newton iterations,
  `[1]` its Krylov iterations over all linear solves.
- The commented-out `@cuda.jit` signature block above each kernel documents the types;
  keep it in sync and commented.

## Factory and dispatch
- Subclasses implement `build_step(...)` returning `StepCache(step_fn=...,
  nonlinear_solver_fn=...)`; `BaseAlgorithmStep.build()` fills the remaining fields from
  the same-named properties. The compiled step is the `step_fn` property.
- `get_algorithm_step(precision, settings, **kwargs)` requires `settings["algorithm"]`:
  a name (resolved through `_TABLEAU_REGISTRY_BY_ALGORITHM` by `resolve_alias`) or a
  `ButcherTableau` instance (dispatched by type in `resolve_supplied_tableau`). The bare
  family names `"erk"`, `"dirk"`, `"firk"`, `"rosenbrock"` use the class's
  `default_tableau`; `"euler"`, `"backwards_euler"`, `"backwards_euler_pc"` and
  `"crank_nicolson"` have no tableau.
- `AlgorithmDefaults` holds one flat settings dict per family (controller and solver
  keys together), adaptive or fixed by `tableau.has_error_estimate`; a tableau's own
  `defaults` overlay it in `BaseAlgorithmStep.algorithm_defaults`. Keys in
  `ALL_ALGORITHM_STEP_PARAMETERS` are step defaults (`step_default_settings`), the rest
  controller defaults (`controller_default_settings`). `family_defaults(tableau)` and
  `algorithm_facts(algorithm, tableau)` (`AlgorithmFacts`: step class, tableau,
  defaults, `has_error_estimate`, `is_implicit`, `is_linear`) give the table without a
  step instance.
- Errorless tableaus require a fixed controller; constructors enforce it.
- New `update` keywords go in `ALL_ALGORITHM_STEP_PARAMETERS` or `update` rejects them.
  `BaseAlgorithmStep._update` runs `_apply_updates` (settings and buffers), counts
  unapplied names from that set as recognised and warns unless `silent`.
  `ODEImplicitStep` extends `_apply_updates` with its solvers and predictor.

## Tableaus
- Add a tableau to the relevant `*_tableaus.py` registry; `__init__.py` merges the
  registries into valid `algorithm` names. `ButcherTableau.__attrs_post_init__` checks
  that `b` and `b_hat` sum to 1.
- Tableau properties drive compile-time shortcuts; use them rather than hand-rolling:
  `b_matches_a_row`/`b_hat_matches_a_row` copy a stage state instead of accumulating;
  `first_same_as_last`/`can_reuse_accepted_start` enable FSAL stage-0 reuse (gated on
  `all_sync(activemask(), accepted_flag != 0)`); `explicit_first_stage` and
  `DIRKTableau.last_implicit_stage` split stage 0, the Newton loop and trailing explicit
  stages.
- `algorithm_order` = `min(order, embedded_order)`; tableaus with `b_hat` declare
  `embedded_order`. Controllers receive `algorithm_order`; `order` stays classical. FIRK
  smoothing uses `RadauIIATableau.smoothed_embedded_order`.

## Explicit and implicit steps
- Explicit (`ODEExplicitStep`, no solver): `ExplicitEulerStep`, `ERKStep`.
- Implicit (`ODEImplicitStep`, owns a solver): `BackwardsEulerStep`,
  `BackwardsEulerPCStep`, `CrankNicolsonStep`, `DIRKStep`, `FIRKStep` use Newton-Krylov;
  `GenericRosenbrockWStep` is linearly implicit with a `LinearSolver` and no Newton
  iteration (`is_linear = True`).

## Registered buffers
- Steps register their working buffers (DIRK `stage_base`/`accumulator`, CN
  `cn_dxdt`); buffers with disjoint lifetimes alias (CN `base_state` → `error`, DIRK
  `stage_base` → `accumulator`). Implicit steps take child allocators for their solver
  via `get_child_allocators(self, self.solver, ...)`.
- Rosenbrock registers `cached_auxiliaries` at size 0; `build_implicit_helpers()`
  resizes it from `prepare_jac`'s `HelperResult.cached_auxiliary_count`.

## Dense stage prediction (FIRK, DIRK)
Both own a `DenseStagePredictor` (`../stage_predictors.py`) that turns the last accepted
step's stage increments into the next step's Newton starting guesses. The step keeps
the persistent `previous_step_size` and folds first-step, rejection and the tableau's
per-precision `dense_prediction_ratio_*` ceiling (from
`benchmarks/dense_prediction_ratio_sweep.py`) into a flag the predictor commits per lane
with `selp`. Tableau properties: `prediction_sample_stages` (one sample per distinct
node), `explicit_first_stage` (never predicted; its `dt*f` sample still enters DIRK's
history), DIRK's `prediction_source_stages` (a repeated stage time starts from the
earlier stage's row). `predictor_fn` arrives through compile settings;
`predictor_*_location` keys place its buffers.

## Smoothed error estimate (DIRK, FIRK, Rosenbrock-W)
- `use_smoothed_error` filters the embedded estimate through
  `(M - smoothing_gamma * h * J)^-1`, one extra linear solve per step. It is active when
  requested, supported by the tableau (`supports_smoothed_error`) and adaptive;
  otherwise it compiles out, and an unsupported request warns.
  `FIRKStep.family_defaults(tableau)` turns it on for Radau.
- `smoothing_gamma` is `a[-1][-1]` on `ButcherTableau` and the sole real eigenvalue of
  `a` on `RadauIIATableau`, computed exactly and rounded once. The tableau also derives
  `smoothed_error_weights`.
- DIRK and FIRK own an `error_solver` (width `n_states`, `AT_STATE` helpers, aliased
  into `solver_shared`) when `owns_error_solver`, registered only while smoothing is
  on. It has `instance_label="error"`, reads `error_atol`, `error_rtol`,
  `error_max_iters`, `error_residual_reduction`, `error_residual_floor` (resolved from
  the `krylov_*` settings when not given) and produces `error_linear_solver_fn`.
  Rosenbrock-W reuses its cached-Jacobian solver.
- The RHS comes from the generated `apply_mass`: DIRK and Rosenbrock-W `M @ raw_error`
  (DIRK at the final stage state, time and drivers, into `error_rhs`); FIRK
  `M @ (sum_i w_i*K_i) - gamma*h*f(y_n)` at the step-start state.

## DIRK stage data
`stage_rhs` holds `k_i = M^-1 @ f(Y_i)`: implicit stages store `stage_increment / dt`;
explicit stages evaluate the generated `evaluate_inv_mass_f` (plain `f` for identity
mass).

## Solver helpers
- Implicit steps and the initialiser take `get_solver_helper_fn` at construction and
  call `get_solver_helper_fn(role, jacobian_at=..., prefactored=..., stacked=...,
  **kwargs).device_function` with a role name (`"residual"`, `"linear_operator"`,
  `"apply_mass"`, ...) or the configured `preconditioner_type`: `jacobian_at="step"` for
  frozen-J chains, `stacked=True` for FIRK, `jacobian_at="state"` for error smoothing,
  `prefactored=True` for step-start LU factors. `preconditioner_type` is validated
  against `PRECONDITIONER_ROLES`.
- Constructors and `update` (after a recognised key) run `build_implicit_helpers()`,
  which requests the helpers, pushes them into the solver children and writes their
  device functions and an `OperationCounts` into the config; `build()` reads only the
  config. An `update` with another `tableau` or `n_states` raises;
  `SingleIntegratorRunCore` rebuilds the step instead.
- `performance_defaults` and `step_operation_count` feed
  `BatchSolverKernel.performance_defaults`; `optimisation_candidates` lists what
  `Solver.optimize` times.
- With `linear_correction_type="lu"` (`uses_direct_solver`) steps request the
  `lu_solve` role; `HelperResult.lu_nnz` sizes the solver's `lu_factor` buffer via
  `update(lu_solve_fn=..., lu_nnz=...)`.

## Simplified Newton (`inexact_newton`)
`ImplicitStepConfig.inexact_newton` (default `False`) freezes the Newton iteration
matrix at the step start; the residual stays exact. The frozen chain wires a per-step
prepare function (`(state, parameters, drivers, t, h, cached_aux) -> int32`, OR'd into
the step status) into `compile_settings.prepare_jacobian_fn`, resizes
`cached_auxiliaries` and sets `use_cached_auxiliaries=True` on the solver. LU pairings
follow `ImplicitStepConfig.prefactored` (default `True`: step-start factors per distinct
tableau diagonal; `False`: frozen entries factorised per call); FIRK + LU runs the
stacked prefactored eigenvalue block transform, with smoothing sharing its real block.
Rosenbrock-W ignores both flags.

## Dependencies
### Internal
- `cubie.CUDAFactory` — step/config/cache base classes.
- `cubie.buffer_registry` — buffer allocation for shared/local memory.
- `cubie.integrators.matrix_free_solvers` — `NewtonKrylov`, `LinearSolver` (owned by
  implicit steps).
- `cubie.cuda_simsafe` — `all_sync`, `activemask` (FSAL warp votes).
- `cubie._utils` — `build_config`, `PrecisionDType`, validators.
- Solver-helper device functions come from the ODE system via `get_solver_helper_fn`
  (role names `residual`, `linear_operator`, `neumann`/`jacobi`,
  `apply_mass`, `evaluate_inv_mass_f`, `time_derivative_rhs`, each
  crossed with a variant name).

### External
- `numba` (`cuda`, `int32`); `attrs`; `numpy` (coefficient math, embedded-weight solves).
