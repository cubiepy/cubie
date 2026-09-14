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
| `SingleIntegratorRunCore.py` | `SingleIntegratorRunCore(CUDAFactory)`: owns `_output_functions`, `_algo_step`, `_step_controller`, `_dae_initialiser`, `_loop`; `update` writes the flat settings it is passed into each child with the products that child consumes and recaptures `loop_fn`. Defines `SingleIntegratorRunCache` (holds `loop_fn`). |
| `IntegratorRunSettings.py` | `IntegratorRunSettings(CUDAFactoryConfig)`: `algorithm`, `step_controller`, `precision`, the driver functions and the captured `loop_fn`. |
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
The run derives nothing; it takes one flat dict of resolved settings and the system's
precision. Order:
1. `IntegratorRunSettings` (`algorithm`, `step_controller`, driver functions, compile
   flags); every child takes the same `unroll` and `jit_flags`.
2. `OutputFunctions` from the output keys and the system's sizes.
3. The step from the settings plus `_step_inputs()` (sizes, device functions,
   `is_adaptive`); an explicit step on a mass-matrix system raises.
4. The controller from the settings plus `_controller_inputs()`.
5. `DAEInitialiser` from the step's `settings_dict`, the settings and
   `_initialiser_inputs()`; no-op configurations register zero-size buffers.
6. `IVPLoop` from the loop keys plus `_loop_inputs()` (the children's products); the
   step, controller and initialiser register under it and `loop_fn` is captured.

`update()` writes the system's config fields, this run's config, then each child with
the same dict plus its inputs, and recaptures `loop_fn`; a new `algorithm` or
`step_controller` swaps that child. `build()` returns `loop_fn`. `settings_dict`
merges the children's; `copy()` rebuilds on `system.copy()` from it.

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
