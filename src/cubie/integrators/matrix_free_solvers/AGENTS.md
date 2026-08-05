<!-- Parent: ../AGENTS.md -->

# matrix_free_solvers

## Purpose
CUDA device-function factories for the inner solvers of implicit methods: a
Jacobian-free preconditioned **linear** solver (steepest-descent /
minimal-residual) and a **Newton–Krylov** nonlinear solver that calls the
linear solver for each correction. The implicit algorithm steps
(`generic_dirk`, `generic_firk`, `backwards_euler`, `crank_nicolson`,
Rosenbrock-W) invoke these once per implicit stage. No Jacobian is materialised —
the caller passes device callbacks that apply the operator / preconditioner /
residual, and the solver iterates using only those plus preallocated scratch.

Both are `MultipleInstanceCUDAFactory` subclasses; the build/cache/`update`,
buffer-registry, attrs-config, and cache-invalidation mechanics are common to all
factories and live with `CUDAFactory` (repo root). This file documents only what
is specific to the solvers.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Re-exports factories/configs/caches; re-exports `CUBIE_RESULT_CODES` from `cubie.result_codes`. |
| `base_solver.py` | `MatrixFreeSolver` / `MatrixFreeSolverConfig` base — holds the norm device function and the shared `solver_width` / `max_iters` / tolerance plumbing. |
| `linear_solver_base.py` | `LinearSolverBase` / `LinearSolverBaseConfig` — shared linear-solver infrastructure: stopping settings, `norm_reference`, buffer and update plumbing. |
| `linear_solver.py` | `MRLinearSolver` — matrix-free preconditioned steepest-descent / minimal-residual linear solve (cached and non-cached variants). |
| `bicgstab_solver.py` | `BiCGSTABSolver` — matrix-free preconditioned BiCGSTAB linear solve. |
| `newton_krylov.py` | `NewtonKrylov` — NLNewton-style Newton iteration. |

## For AI Agents

**Class factories, not free functions.** The public surface is the classes
`MRLinearSolver`, `BiCGSTABSolver`, and `NewtonKrylov`; there are no
`linear_solver_factory` / `newton_krylov_solver_factory` functions. Get the
compiled callable from `.device_function`.

### Compiled device-function signatures (the caller contract)
- Linear solvers (MR/SD and BiCGSTAB share it; non-cached): `linear_solver(state,
  parameters, drivers, base_state, t, h, a_ij, rhs, x, shared, persistent_local,
  krylov_iters_out) -> int32`. The cached variant inserts `cached_aux` after
  `base_state`. `rhs` enters as the RHS and is overwritten with the residual; `x`
  enters as the initial guess and is overwritten with the solution;
  `krylov_iters_out` is a length-1 int32 array.
- `NewtonKrylov`: `newton_krylov_solver(stage_increment, parameters, drivers, t, h,
  a_ij, base_state, step_start, shared_scratch, persistent_scratch, counters) -> int32`.
  `stage_increment` is updated in place. `counters` is a length-2 int32 array:
  `[0]` = Newton iterations, `[1]` = total Krylov iterations.

### Caller-supplied callbacks (set via config/`update`)
- `operator_apply` — applies `F @ v`; sig `(state, parameters, drivers, base_state,
  t, h, a_ij, v, out)` (cached variant inserts `cached_aux` after `drivers`).
- `preconditioner` (optional; `None` → search direction is `rhs`); sig
  `(state, parameters, drivers, base_state, t, h, a_ij, rhs,
  preconditioned_vec, jvp, scratch, chain_scratch)` (cached variant inserts
  `cached_aux` after `drivers`). `chain_scratch` is written only by
  chained compositions.
- `residual_function` (Newton); sig `(stage_increment, parameters, drivers, t, h,
  a_ij, base_state, residual_out)`.
- `linear_solver_function` (Newton) — the inner linear solver's
  `device_function`. `NewtonKrylov` owns a child linear solver: its `update`
  forwards `krylov_`-prefixed params to the child and re-injects the
  recompiled device function.

### Registered buffers (length `solver_width` unless noted)
- `MRLinearSolver`: `preconditioned_vec`, `temp`, `mr_precond_scratch`,
  `mr_chain_scratch` (length `solver_width` when
  `preconditioner_is_chained`, else 0).
- `BiCGSTABSolver`: `bicg_r0_hat`, `bicg_p`, `bicg_v`, `bicg_tmp`,
  `bicg_s_hat`, `bicg_precond_scratch`, `bicg_chain_scratch` (length
  `solver_width` when `preconditioner_is_chained`, else 0).
- `NewtonKrylov`: `delta`, `residual`, `krylov_iters_local` (length 1,
  int32), and `prev_theta` (length 1, persistent — contraction
  history carried between solves).

### Status codes & convergence
- Status codes come from the package-central `CUBIE_RESULT_CODES` (`cubie/result_codes.py`,
  re-exported from this package): `SUCCESS=0`,
  `MAX_NEWTON_ITERATIONS_EXCEEDED=2`, `MAX_LINEAR_ITERATIONS_EXCEEDED=4` (captured as device
  closure constants). `newton_krylov_solver` OR-combines these into a **low-bits** status
  word — it does NOT pack the iteration count into high bits (counts go to `counters`).
  Callers OR this word into their own step status.
- Linear norms use `ScaledNorm` (`TiledScaledNorm` for coupled FIRK
  solves, whose reference tiles the single-stage base state across
  all stages). The Newton norm is a `DIRKCorrectionNorm` or
  `FIRKCorrectionNorm`, whose whole-vector function scales the update
  by `atol + rtol * max(|stage_value|, |step_start|)` (DIRK: one
  diagonal coefficient; FIRK: the full tableau row).
- **Norm tolerances are per physical state** (`n` entries, the length
  the step controller takes); stage-tiled norms read entry `i mod n`.
  `n` is a required norm-constructor argument.
- **Every linear solve (MR, SD, BiCGSTAB; Newton-owned or direct)
  stops on** `||r|| <= krylov_residual_floor +
  krylov_residual_reduction * ||b||`. `||.||` = the solver's
  `ScaledNorm` (1.0 sits at the `krylov_atol`/`krylov_rtol`
  envelope); `||b||` = the untouched RHS at solve entry. Norm
  reference: stage base state (Newton-owned) or model state
  (direct) — `norm_reference` config field, bound at compile time.
  Derived defaults: `krylov_atol`/`krylov_rtol` = the step
  controller's `atol`/`rtol`; reduction = adaptive controller min
  `rtol`, divided by 100 for linearly-implicit (`is_linear`) steps
  (machine epsilon for non-adaptive runs); floor = `sqrt(eps)`.
- **Newton convergence follows OrdinaryDiffEq's NLNewton.** Consecutive
  full steps estimate the contraction `theta` (decay-floored at
  `0.3 * prev_theta`, warm-started across solves via the persistent
  `prev_theta` buffer; a failed solve resets the stored value). The
  solve accepts when `theta / (1 - theta) * ||dz|| < 1/100`, or on the
  first iteration when `||dz|| < 1e-5`. `theta > 2` or a non-finite
  update norm exits with `NEWTON_DIVERGENCE=256`; at the
  floating-point stagnation limit (`theta ≈ 1`) the update norm alone
  decides between convergence and divergence. Commits are gated on
  linear-solver success — a failed linear solve moves nothing and
  clears the in-solve contraction history.
- There is no line search: a diverging solve exits early with a
  nonzero status and the adaptive step controller rejects the step
  and shrinks `dt`.

### Solver-specific gotchas
- **Warp-coherent loops.** Iterative loops exit on warp votes (`all_sync`/`any_sync`
  from `cuda_simsafe`) so every active lane agrees before breaking; `selp` gives
  branchless commits. Don't add un-voted data-dependent `break`/early-return — it
  breaks lane lockstep.

### Testing
Solver behaviour is exercised through the implicit algorithm steps under
`tests/integrators/algorithms/`, verified against the plain CPU reference
solvers in `tests/integrators/cpu_reference/cpu_utils.py`
(`newton_solve`, `krylov_solve`). Any change to a device function's
algorithm, signature, buffers, or status logic must be replicated in its
CPU reference counterpart. Run e.g.
`pytest tests/integrators/algorithms -k "newton or krylov or implicit"`.

## Dependencies
### Internal
- `cubie.CUDAFactory` — `MultipleInstanceCUDAFactory` + config/cache bases.
- `cubie.integrators.norms` — convergence norm device function.
- `cubie.buffer_registry` — scratch buffer allocators.
- `cubie.cuda_simsafe` — `activemask`, `all_sync`, `any_sync`, `selp`.
- `cubie._utils` — `build_config`, device/precision validators, `PrecisionDType`.
- Consumed by `cubie.integrators.algorithms.*` (implicit steps).
### External
- `numba.cuda`, `attrs`, `numpy`.
