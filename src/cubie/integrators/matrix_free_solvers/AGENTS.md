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

Both are `MultipleInstanceCUDAFactory` subclasses.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Re-exports factories/configs/caches; re-exports `CUBIE_RESULT_CODES` from `cubie.result_codes`. |
| `base_solver.py` | `MatrixFreeSolver` / `MatrixFreeSolverConfig` base — holds the norm device function and the shared `solver_width` / `max_iters` / tolerance plumbing. |
| `linear_solver_base.py` | `LinearSolverBase`/`LinearSolverBaseConfig` (shared contract: `zero_initial_guess`, `norm_reference`, buffer and update plumbing) and `IterativeLinearSolverBase`/`IterativeLinearSolverConfig` (stopping settings, operator/preconditioner callbacks, krylov tolerance surface). |
| `linear_solver.py` | `MRLinearSolver` — matrix-free preconditioned steepest-descent / minimal-residual linear solve. |
| `bicgstab_solver.py` | `BiCGSTABSolver` — matrix-free preconditioned BiCGSTAB linear solve. |
| `lu_solver.py` | `LUSolver` — direct sparse LU solve (`linear_correction_type="lu"`); wraps the generated `lu_solve` helper (codegen: `odesystems/symbolic/codegen/lu_solver.py`) in the shared linear-solver contract. |
| `newton_krylov.py` | `NewtonKrylov` — Newton iteration with a warm-started contraction test. |

## Classes
The public surface is `MRLinearSolver`, `BiCGSTABSolver`, `LUSolver` and `NewtonKrylov`;
get the compiled callable from `.device_function`.

## Device-function signatures
- Linear solvers (MR/SD, BiCGSTAB, LU): `linear_solver(state, parameters, drivers,
  base_state, cached_aux, t, h, a_ij, rhs, x, shared, persistent_local,
  krylov_iters_out) -> int32`. `cached_aux` may be zero-length. `rhs` is overwritten
  with the residual and `x` (initial guess) with the solution; `krylov_iters_out` is a
  length-1 int32 array. `LUSolver` is exact per call: `rhs` is read-only, the guess is
  ignored and the status is always `SUCCESS`.
- `NewtonKrylov`: `nonlinear_solver_fn(stage_increment, parameters, drivers,
  cached_aux, t, h, a_ij, base_state, step_start, shared_scratch, persistent_scratch,
  counters) -> int32`. `stage_increment` updates in place;
  `use_cached_auxiliaries=True` solves at `step_start`. `counters` is length-2 int32:
  `[0]` Newton iterations, `[1]` total Krylov iterations.
- Status bits (`../AGENTS.md`) are OR-combined; iteration counts never enter the status
  word. Callers OR it into their step status.

## Caller-supplied callbacks (config / `update`)
- `operator_apply_fn` applies `F @ v`: `(state, parameters, drivers, cached_aux,
  base_state, t, h, a_ij, v, out)`.
- `preconditioner` (`None` makes the search direction `rhs`): `(state, parameters,
  drivers, cached_aux, base_state, t, h, a_ij, rhs, preconditioned_vec, jvp)`.
- `residual_fn` (Newton): `(stage_increment, parameters, drivers, t, h, a_ij,
  base_state, residual_out)`.
- `krylov_linear_solver_fn` (Newton): the child linear solver's `device_function`.
  `NewtonKrylov.update` forwards `krylov_`-prefixed keys to the child and re-injects its
  recompiled function.

## Registered buffers (length `solver_width` unless noted)
- `MRLinearSolver`: `preconditioned_vec`, `temp`.
- `BiCGSTABSolver`: `bicg_r0_hat`, `bicg_p`, `bicg_v`, `bicg_tmp`, `bicg_s_hat`.
- `LUSolver`: `lu_factor` (length `lu_nnz`, location `lu_factor_location`; 0 for
  substitution-only variants).
- `NewtonKrylov`: `delta`, `residual`, `krylov_iters_local` (1, int32), `prev_theta`
  (1, persistent contraction history).

## Norms and convergence
- Linear norms are `ScaledNorm` (`TiledScaledNorm` for coupled FIRK solves, tiling the
  single-stage base state across stages). The Newton norm is `DIRKCorrectionNorm` or
  `FIRKCorrectionNorm`, scaling the update by
  `atol + rtol * max(|stage_value|, |step_start|)` (DIRK: one diagonal coefficient;
  FIRK: the full tableau row).
- Tolerances are per physical state (`n_states` entries, a required norm argument);
  stage-tiled norms read entry `i mod n_states`. Every norm floors `atol` at `1e-16` per
  entry on the host with a `UserWarning`; correction-norm `rtol` floors at 4 ULPs;
  Krylov norms keep raw `rtol`.
- Every linear solve stops on `||r|| <= krylov_residual_floor + krylov_residual_reduction
  * ||b||`, `||.||` the solver's `ScaledNorm` and `||b||` the RHS at entry (squared target
  capped at `finfo.max`). The norm reference is the stage base state (Newton-owned) or
  the model state (direct), bound at compile time via `norm_reference`. Defaults:
  `krylov_atol`/`krylov_rtol` = the controller's `atol`/`rtol`; reduction = the adaptive
  controller's minimum `rtol` (divided by 100 for `is_linear` steps; machine epsilon
  for non-adaptive runs); floor = `sqrt(eps)`.
- Newton: consecutive full steps estimate the contraction `theta` (floored at
  `0.3 * prev_theta`, warm-started from `prev_theta`, stored clamped to 1, reset by a
  failed solve). Accept on `theta / (1 - theta) * ||dz|| < 1/100`, a first-iteration
  `||dz|| < 1e-5`, or `||dz|| >= ||dz_prev||` with `||dz|| <= 1`. A non-finite norm exits
  with `NEWTON_DIVERGENCE`; otherwise an unconverged solve ends at `newton_max_iters`,
  adding `NEWTON_DIVERGENCE` if any `||dz|| > 1` update had `theta > 2`. A failed linear
  solve commits nothing and clears the contraction history.
- Unset `krylov_max_iters` resolves to `ceil(1.5 * solver_width)`.
- No line search: a diverging solve exits with a nonzero status and the adaptive
  controller rejects the step.

## CPU references
The CPU reference solvers in `tests/integrators/cpu_reference/cpu_utils.py`
(`newton_solve`, `krylov_solve`) must match every change to a device function's
algorithm, signature, buffers or status logic.

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
