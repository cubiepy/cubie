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
| `SingleIntegratorRunCore.py` | `SingleIntegratorRunCore(CUDAFactory)`: owns `_output_functions`, `_algo_step`, `_step_controller`, `_dae_initialiser`, `_starting_step`, `_loop`; `update` writes the flat settings it is passed into each child with the products that child consumes and recaptures `loop_fn`; `optimisation_candidates` is the step's. Defines `SingleIntegratorRunCache` (holds `loop_fn`). |
| `IntegratorRunSettings.py` | `IntegratorRunSettings(CUDAFactoryConfig)`: `algorithm`, `step_controller`, `precision`, the driver functions and the captured `loop_fn`. |
| `norms.py` | CUDA factories for scaled vector norms (`ScaledNorm`, `TiledScaledNorm`, `TwoRefMaskedScaledNorm`) and DIRK/FIRK Newton correction terms; every config floors `atol` at `ATOL_FLOOR` per entry on the host with a `UserWarning`. Under `low_precision_norms`, `ScaledNorm`, the all-rows `TwoRefMaskedScaledNorm` and both correction norms reduce bfloat16 pairs: `packed_squared_ratios` scales by a bit-pattern reciprocal estimate of the tolerance (`PACKED_RECIPROCAL_MAGIC`, at most 5.1% error) and accumulates in bfloat16; an odd last entry runs in float32. |
| `stage_predictors.py` | `DenseStagePredictor(CUDAFactory)`: in-place read-ahead of a persistent stage-increment vector that warm-starts the next step's Newton solves; step-size-ratio polynomials precomputed from the tableau. FIRK and DIRK own one as a buffer-registry child. Under `low_precision_predictor` the matrix product runs on bfloat16 pairs of states; an odd last state runs in run precision. |
| `dae_initialiser.py` | `DAEInitialiser(CUDAFactory)`: one-shot consistent-initialisation solve at loop entry before the t0 save, a damped Newton over a direct LU. Always constructed by the core from the algorithm step's `settings_dict`; non-DAE systems and mode `"none"` compile a no-op with zero-size buffers. Modes: `"brown"` (default; corrects only the algebraic components), `"shampine"` (one backward-Euler solve of the initial dt), `"none"`. A failed solve commits nothing and returns the solver bits with `DAE_INITIALISATION_FAILED` set; the loop ends the run at the t0 save. |
| `starting_step.py` | `StartingStepEstimator(CUDAFactory)` compiles `starting_step_fn`, Hairer's starting-step estimate, for an adaptive controller given no `dt` (`estimates_dt`); otherwise `starting_step_fn` is `None`. It takes the Solver settings named in `ALL_STARTING_STEP_PARAMETERS`. |
| `__init__.py` | Package API re-exports (`SingleIntegratorRun`, `IVPLoop`, algorithm/solver/controller classes, `get_algorithm_step`, `get_controller`); re-exports `CUBIE_RESULT_CODES` from `cubie.result_codes`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `algorithms/` | Step-function factories + `get_algorithm_step()` (see `algorithms/AGENTS.md`). |
| `loops/` | `IVPLoop` and `ODELoopConfig` (see `loops/AGENTS.md`). |
| `matrix_free_solvers/` | Matrix-free linear (steepest-descent / minimal-residual) and Newton-Krylov solvers (see `matrix_free_solvers/AGENTS.md`). |
| `step_control/` | Fixed/adaptive step-size controllers + `get_controller()` (see `step_control/AGENTS.md`). |

## CUBIE_RESULT_CODES
The kernel status word is `CUBIE_RESULT_CODES(IntFlag)` (`cubie/result_codes.py`,
re-exported here and from `cubie`); device functions capture its values as closure
constants and OR them into their status:
`SUCCESS=0`, `MAX_NEWTON_ITERATIONS_EXCEEDED=2`, `MAX_LINEAR_ITERATIONS_EXCEEDED=4`,
`STEP_TOO_SMALL=8` (controller reject at minimum step), `DT_EFF_EFFECTIVELY_ZERO=16` and
`MAX_LOOP_ITERS_EXCEEDED=32` (reserved, never set), `STAGNATION=64` (loop made no
progress), `BICGSTAB_BREAKDOWN=128`, `NEWTON_DIVERGENCE=256`,
`DAE_INITIALISATION_FAILED=1024` (the t0 consistent-initialisation solve failed; the run
ends at the t0 save with the solver bits also set). Iteration counts go in the
`counters` array, never the status word. Host-side, decode with
`cubie.result_codes.decode_status_codes` (`SolveResult.status_messages`).

## Component assembly (`SingleIntegratorRunCore`)
The core resolves nothing; it takes one flat settings dict. Each child class declares
`system_inputs(system, ...)`, what it reads from the system (sizes, precision, products,
mass flags). Construction order:
1. `IntegratorRunSettings` (`algorithm`, `step_controller`, driver functions, compile
   flags).
2. `OutputFunctions` from the output keys and its system inputs.
3. The step named by `algorithm`, from the settings, its system inputs, the driver
   evaluators and `is_adaptive`; an explicit step on a mass-matrix system raises.
4. The controller named by `step_controller`, given the step's order.
5. `DAEInitialiser` from the step's `settings_dict`. A no-op initialiser registers
   zero-size buffers.
6. `StartingStepEstimator` from the settings, its system inputs and the step's and
   controller's products.
7. `IVPLoop` from the loop keys plus `_loop_inputs()` (the children's products). We
   register the step, controller, initialiser and estimator under it and capture
   `loop_fn`.

`update()` writes this config, then each child with the same dict plus its inputs, and
recaptures `loop_fn`. A new `algorithm`, `tableau` or state count rebuilds the step; a new
`step_controller` rebuilds the controller, from `init_kwargs` plus the update. The
Solver updates the system before this chain runs. `build()` returns `loop_fn`;
`settings_dict` merges the children's.

## Dependencies
### Internal
- `cubie.CUDAFactory`; `cubie.buffer_registry`; `cubie._utils` (`PrecisionDType`,
  `unpack_dict_values`, `build_config`, tolerance validators, `tol_converter`);
  `cubie._cudasim_extensions`; `cubie.backend`; `cubie.odesystems.ODEData` (`SystemSizes`), `baseODE`
  (TYPE_CHECKING); `cubie.outputhandling` (`OutputFunctions`, `OutputCompileFlags`); the
  four integrators subpackages.
### External
- `numba` (`cuda.jit`, `int32`, `from_dtype`); `numpy`; `attrs`.
