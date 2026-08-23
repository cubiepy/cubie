<!-- Parent: ../AGENTS.md -->

# integrators

## Purpose
The integration layer: it contains the **complete integrator for a single thread /
system — nothing here is batched** (batching lives in `batchsolving/`).
`SingleIntegratorRunCore` is the **composition/glue layer** (a composition root): it
instantiates the algorithm step, step controller, output functions, and CUDA loop, wires
their compiled device functions and data together, and injects the composed objects into
the children that need them; `SingleIntegratorRun` adds read-only properties over it. The
directory also provides a general scaled error-norm utility (`ScaledNorm`). The
kernel-level status vocabulary is the package-central `CUBIE_RESULT_CODES`
(`cubie/result_codes.py`), re-exported here.

See `CUDAFactory` (repo root) for the build/cache/`update`, buffer-registry, and
attrs-config mechanics. Subsystems (algorithms, loops, matrix_free_solvers, step_control)
each have their own `AGENTS.md`.

## Key Files
| File | Description |
|------|-------------|
| `SingleIntegratorRun.py` | `SingleIntegratorRun(SingleIntegratorRunCore)`: read-only properties exposing compiled loop artifacts, memory sizing, controller bounds, and output metadata to `BatchSolverKernel`. No `build()` override. |
| `SingleIntegratorRunCore.py` | `SingleIntegratorRunCore(CUDAFactory)`: owns `_output_functions`, `_algo_step`, `_step_controller`, `_loop`; wires them and delegates compilation to `IVPLoop` in `build()`. Defines `SingleIntegratorRunCache` (holds `single_integrator_function`). |
| `IntegratorRunSettings.py` | `IntegratorRunSettings(CUDAFactoryConfig)`: thin compile-settings holding only `algorithm` and `step_controller` names (plus inherited `precision`) — the core's own cache key. |
| `norms.py` | CUDA factories for scaled vector norms and DIRK/FIRK Newton correction terms. |
| `stage_predictors.py` | `DenseStagePredictor(CUDAFactory)`: in-place read-ahead of a persistent stage-increment vector that warm-starts the next step's Newton solves; step-size-ratio polynomials precomputed from the tableau. FIRK and DIRK own one as a buffer-registry child. |
| `dae_initialiser.py` | `DAEInitialiser(CUDAFactory)`: one-shot consistent-initialisation solve at loop entry before the t0 save. Always constructed by the core; non-DAE systems and mode `"none"` compile a no-op with zero-size buffers. Owns a `NewtonKrylov` over a direct LU solve with a fixed cap `newton_max_iters=50`; the constructor takes the algorithm step's `settings_dict` wholesale, filters the Newton tolerances and buffer locations it uses, and ignores the rest. Modes: `"brown"` (default; solves the constraint rows via `init_residual`/`init_lu_solve`, differential values held exactly), `"shampine"` (one backward-Euler solve of the initial dt via the standard residual/`lu_solve`), `"none"`. A failed solve commits nothing and returns the solver bits with `DAE_INITIALISATION_FAILED` set; the loop ends the run at the t0 save. |
| `__init__.py` | Package API re-exports (`SingleIntegratorRun`, `IVPLoop`, algorithm/solver/controller classes, `get_algorithm_step`, `get_controller`); re-exports `CUBIE_RESULT_CODES` from `cubie.result_codes`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `algorithms/` | Step-function factories + `get_algorithm_step()` (see `algorithms/AGENTS.md`). |
| `loops/` | `IVPLoop` and `ODELoopConfig` (see `loops/AGENTS.md`). |
| `matrix_free_solvers/` | Matrix-free linear (steepest-descent / minimal-residual) and Newton-Krylov solvers (see `matrix_free_solvers/AGENTS.md`). |
| `step_control/` | Fixed/adaptive step-size controllers + `get_controller()` (see `step_control/AGENTS.md`). |

## For AI Agents

### CUBIE_RESULT_CODES — kernel status-bit meanings
The status vocabulary is the package-central `CUBIE_RESULT_CODES(IntFlag)` (defined in
`cubie/result_codes.py`, re-exported from this package and from `cubie`). Device functions
capture its values as closure constants and OR them into the returned status word:
`SUCCESS=0`, `MAX_NEWTON_ITERATIONS_EXCEEDED=2`,
`MAX_LINEAR_ITERATIONS_EXCEEDED=4`, `STEP_TOO_SMALL=8` (controllers' reject-at-min),
`DT_EFF_EFFECTIVELY_ZERO=16` and `MAX_LOOP_ITERS_EXCEEDED=32` (reserved, unemitted),
`STAGNATION=64` (loop no-progress), `BICGSTAB_BREAKDOWN=128`,
`NEWTON_DIVERGENCE=256`, `DAE_INITIALISATION_FAILED=1024` (t0
consistent-initialisation solve failed; the run ends at the t0 save
with the solver failure bits also set).
Iteration counts are returned separately via the
`counters` array, never packed into the status word. Host-side, decode via
`cubie.result_codes.decode_status_codes` (exposed as `SolveResult.status_messages` /
`Solver.status_messages`).

### Component assembly (`SingleIntegratorRunCore.__init__`)
Order matters — each component seeds the next:
1. `OutputFunctions` first (its compile flags + summary buffer heights feed `IVPLoop`).
2. `_algo_step = get_algorithm_step(precision, settings)` — supplies
   `controller_defaults.step_controller`, seeding the controller settings before user
   overrides merge in. `_apply_dae_linear_solve_defaults()` fills unset
   `preconditioner_type` (jacobi) and `linear_correction_type` (lu) on mass-matrix
   systems, and scales an unset `krylov_max_iters` to the solver width when an
   iterative correction is in effect; user-set keys are preserved across
   hot-swaps. `neumann` is rejected on mass-matrix systems.
3. `_step_controller = get_controller(precision, controller_settings)`.
4. `check_compatibility()` — if the algorithm is errorless but the controller is
   adaptive, the controller is **silently replaced with `FixedStepController`** and a
   `UserWarning` is issued (an errorless algorithm gives no error signal to adapt on).
   Happens before the loop is created.
5. `instantiate_loop()` — creates `IVPLoop` from the finalised sizes/flags/timing.
6. `get_child_allocators(self._loop, self._algo_step, name='algorithm')` (and the
   controller equivalent) — registers algo/controller buffers as children of the loop's
   group. **Must be re-run after every algo/controller swap** (in `update()`,
   `_switch_algos()`, `_switch_controllers()`, `build()`).
7. `DAEInitialiser` — always constructed from the algorithm step's `settings_dict`
   and registered as loop child `initialiser` with `aliases="algorithm_shared"`
   (re-registered wherever the algo/controller children are). `update()` receives
   the shared updates dict with the system's `mass_diagonal_flags` injected; no-op
   configurations register zero-size buffers.

### build() delegates to IVPLoop
`SingleIntegratorRunCore.build()` defines no device function of its own. It (1) updates
`_algo_step` if the system's `evaluate_f`/`evaluate_observables`/`get_solver_helper_fn`
changed; (2) re-registers child allocators; (3) calls `self._loop.update(...)` with the
latest compiled device-function references; (4) accesses `self._loop.device_function`
(triggering the loop's build if invalid); (5) returns
`SingleIntegratorRunCache(single_integrator_function=loop_fn)` — the same object as the
loop's `loop_function`.

### update() follows the system layout
`update()` passes all size parameters after a system update.

### Two-phase timing
`_process_loop_timing()` derives `save_every`, `summarise_every`,
`sample_summaries_every`, and the `save_*`/`summarise_regularly` flags from user intent.
If `summarise_every` is omitted, `is_duration_dependent=True` and
`set_summary_timing_from_duration()` must be called later (done by `BatchSolverKernel`
before each solve); this triggers a recompile on first use (warned).

### Hot-swap
A new `"algorithm"`/`"step_controller"` in `update()` routes through
`_switch_algos()`/`_switch_controllers()`, which call `buffer_registry.reset()`, rebuild
the sub-component from the old settings as a base, and propagate defaults into
`updates_dict`. Controller gains (`kp`/`ki`/`kd`) do not carry across a controller
change; the new controller uses its own gain defaults unless the update supplies them.
Never call these directly — go through `update()`. Because the swap calls
`buffer_registry.reset()`, any cached allocator references become stale.

### Testing
Top-level files are exercised via `tests/integrators/` integration tests and
`tests/batchsolving/` end-to-end tests. `ScaledNorm` is tested with the
matrix-free solvers.

## Dependencies
### Internal
- `cubie.CUDAFactory`; `cubie.buffer_registry`; `cubie._utils` (`PrecisionDType`,
  `unpack_dict_values`, `build_config`, tolerance validators, `tol_converter`);
  `cubie.cuda_simsafe`; `cubie.odesystems.ODEData` (`SystemSizes`), `baseODE`
  (TYPE_CHECKING); `cubie.outputhandling` (`OutputFunctions`, `OutputCompileFlags`); the
  four integrators subpackages.
### External
- `numba` (`cuda.jit`, `int32`, `from_dtype`); `numpy`; `attrs`.
