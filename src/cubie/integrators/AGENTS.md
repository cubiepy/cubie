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
| `SingleIntegratorRunCore.py` | `SingleIntegratorRunCore(CUDAFactory)`: owns `_output_functions`, `_algo_step`, `_step_controller`, `_dae_initialiser`, `_loop`; `update` writes user keys and resolved settings into each child in order and captures `loop_fn` on its settings. `SingleIntegratorRunCache`: `loop_fn` plus the product fields the kernel reads (`compile_flags`, `n_states`, `threads_per_step`, `is_implicit`, `algorithm_family`, `newton_solves_per_step`, `step_operation_count`). |
| `IntegratorRunSettings.py` | `IntegratorRunSettings(CUDAFactoryConfig)`: the run's given keys (`ALL_RUN_PARAMETERS`) as underscored slots, the inputs they resolve against (`algorithm_defaults`, `has_error_estimate`, `is_implicit`, `is_linear`, `has_mass`, `controller_atol/rtol`, output flags, `summary_window`), the resolving properties (`step_controller`, `controller_settings`, `step_settings`, `inner_tolerances`, `loop_timing`, `summary_window_derived`) and the captured `loop_fn`. |
| `norms.py` | CUDA factories for scaled vector norms (`ScaledNorm`, `TiledScaledNorm`, `TwoRefMaskedScaledNorm`) and DIRK/FIRK Newton correction terms; every config floors `atol` at `ATOL_FLOOR` per entry on the host with a `UserWarning`. |
| `stage_predictors.py` | `DenseStagePredictor(CUDAFactory)`: in-place read-ahead of a persistent stage-increment vector that warm-starts the next step's Newton solves; step-size-ratio polynomials precomputed from the tableau. FIRK and DIRK own one as a buffer-registry child. |
| `dae_initialiser.py` | `DAEInitialiser(CUDAFactory)`: one-shot consistent-initialisation solve at loop entry before the t0 save, a damped Newton over a direct LU. Always constructed by the core from the algorithm step's `settings_dict`; non-DAE systems and mode `"none"` compile a no-op with zero-size buffers. Modes: `"brown"` (default; corrects only the algebraic components), `"shampine"` (one backward-Euler solve of the initial dt), `"none"`. A failed solve commits nothing and returns the solver bits with `DAE_INITIALISATION_FAILED` set; the loop ends the run at the t0 save. |
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
2. `IntegratorRunSettings` takes the run-owned keys of the three settings dicts plus
   `has_mass`, `has_summary_outputs`, `has_time_domain_outputs`.
3. `_new_step` from its given keys, `algorithm` and the system's products;
   `_record_step_products` writes `algorithm_defaults`, `has_error_estimate`,
   `is_implicit`, `is_linear`.
4. `_new_controller` for the resolved `step_controller` (given, else the family default
   promoted within `i`/`pi`/`pid` to carry given gains; `fixed` with a `UserWarning`
   when the step has no error estimate) from its given keys, sizes, `algorithm_order`,
   `mass_flags` and `controller_settings` (family gains only for the family's
   controller without a filter); `_record_controller_products` writes
   `controller_atol/rtol`.
5. `_push_resolved_step_settings` writes `step_settings` (family, tableau and
   `DAE_SOLVER_DEFAULTS` values for unset keys, DAE over family on a mass system,
   `neumann` rejected; Newton-variant keys only with the family's linear solver;
   `inner_tolerances`) and `is_adaptive` into the step.
6. `DAEInitialiser` from the step's `settings_dict`, sizes, helper factory,
   `dae_initialisation` and Newton tolerances; outputs get `sample_summaries_every`.
7. `IVPLoop` from its placements and the inputs (sizes, output flags and heights,
   `n_error`, `dt`, `is_adaptive`, `loop_timing`, device functions); the step,
   controller and initialiser (`aliases="algorithm_shared"`) register under it;
   `loop_fn` is captured.

`update()` runs the system and outputs, writes the run-owned keys and inputs (a new
`algorithm` without `step_controller` resets the request to the family default; the
last of `filter_coefficients` or loose gains wins), then `_sync` repeats 3–7: a changed
`algorithm` or controller name swaps the child carrying its `settings_dict`, compile
flags and (controllers) `dt`; `precision`, `unroll`, `jit_flags`, `lineinfo` broadcast;
`summary_window` (pushed by the kernel only while `summary_window_derived`) sets the
derived schedule. `build()` returns `loop_fn`. `settings_dict` is the run's given keys
plus `child_settings` of the children; `grouped_settings()` splits it into the
constructor's groups; `copy()` rebuilds on `system.copy()`.
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
