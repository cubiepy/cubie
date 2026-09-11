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
| `SingleIntegratorRunCore.py` | `SingleIntegratorRunCore(CUDAFactory)`: owns `_output_functions`, `_algo_step`, `_step_controller`, `_loop`; wires them and delegates compilation to `IVPLoop` in `build()`. Defines `SingleIntegratorRunCache` (holds `loop_fn`). |
| `IntegratorRunSettings.py` | `IntegratorRunSettings(CUDAFactoryConfig)`: thin compile-settings holding only `algorithm` and `step_controller` names (plus inherited `precision`) — the core's own cache key. |
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
The constructor reads the system's `products` (sizes, `mass_flags`, `precision`,
`dxdt_fn`, `observables_fn`, `get_solver_helper_fn`) and builds each child with its
static settings: `OutputFunctions`; the step via `get_algorithm_step`
(`_apply_algorithm_step_defaults` and `_apply_dae_linear_solve_defaults` fill unset
keys; `neumann` is rejected on mass-matrix systems); the controller via
`get_controller` from the family defaults, the user's settings and the resolved name
(a given name, else gains promote the family default within `i`/`pi`/`pid`; an
errorless algorithm forces `fixed` with a `UserWarning`); `IVPLoop` from the
children's products; `DAEInitialiser` from the step's `settings_dict` plus
`dae_initialisation`, `mass_flags` and the helper getter. It then runs
`_distribute({})` once so every child holds the others' products.

### update() distributes products
`update()` records givenness (inner tolerances, performance keys, timing), then
`_distribute(updates)` updates the children in a fixed order, merging each child's
`products` into the dict before the next: system; output functions (twice, around
`_loop_timing`); `_switch_algos` and `_switch_controllers` (swap on a new
`algorithm`/`step_controller`, primed from the predecessor's `settings_dict`); the
controller (delivers `is_adaptive`, `dt`, `dt_min`, `dt_max`, `atol`, `rtol`); the
step, then the family, DAE, inner-tolerance and performance defaults; the controller
again with the step's `algorithm_order`; the initialiser; `_register_loop_children`;
the loop; then `update_compile_settings` on the run, whose `loop_fn` field captures
the loop's product. Unrecognised user keys raise unless `silent`.

`settings_dict` merges the children's `settings_dict`s minus the keys the core injects
(`_INJECTED_KEYS`, the loop's `dt` and schedule); timing comes from `_user_timing`, inner
tolerances only when in `_user_given_inner_tols`, performance defaults and unroll flags only
when in `_user_given_keys` (all of them when `auto_performance` is off). `grouped_settings()`
splits it by the children's `settings_keys`; `copy()` rebuilds on `system.copy()` from those
groups. A controller swap drops `CONTROLLER_GAIN_PARAMETERS`.

### build() reads the captured loop
`build()` returns `SingleIntegratorRunCache(loop_fn=compile_settings.loop_fn, ...)` with
the sizes, flags, counts and `performance_defaults` from the children's products. The
cache invalidates when `update` captures a different `loop_fn`.

### Timing
`_loop_timing(updates)` derives `save_every`, `summarise_every`,
`sample_summaries_every` and the `save_*`/`summarise_regularly` flags from
`_user_timing` and the output types. With summaries requested and no
`summarise_every`, `is_duration_dependent=True` and a `duration` key in `update`
(routed by `set_summary_timing_from_duration`, called by `BatchSolverKernel` before
each solve) sets `summarise_every=duration` and
`sample_summaries_every=duration / 100`; the last duration holds until the next.

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
