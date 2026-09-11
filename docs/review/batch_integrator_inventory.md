# Batch solving, integrator, and developer-guide inventory

## Audit totals

| Measure | Result |
|---|---:|
| Developer-guide pages | 7 |
| API root-index pages | 1 |
| Batch-solving API pages | 17 |
| Integrator API pages | 44 |
| Total pages | 69 |
| Total RST lines read | 1,917 |
| Autodoc directives checked | 58 |
| Resolved autodoc targets | 54 |
| Broken autodoc targets | 4 |
| Unique source-docstring lines followed through resolved `:members:` targets | 3,396 |
| Authored RST em-dash or semicolon lines | 30 |
| Rendered source-docstring em-dash lines | 7 |
| Rendered source-docstring semicolon lines | 32 |
| Prospective missing-page autodoc targets checked | 4 |
| Prospective rendered em-dash lines | 2 |
| Prospective rendered semicolon lines | 5 |
| Prospective rendered avoidable-colon lines | 4 |
| Prospective rendered framing tells | 1 |
| False-drama staccato findings | 0 |

`Read` means every physical RST line was inspected. `Resolved` means the
documented Python object exists at the cited source location. Resolution does
not establish that rendered docstrings are fresh or stylistically compliant.

## Page-by-page inventory

### Developer guide

| Page | Lines | Read | Autodoc target | Freshness and completeness | Issues |
|---|---:|---|---|---|---|
| `developer_guide/index.rst` | 18 | Complete | Not applicable | Complete page tree; opening is framed around reader identity | DG-00 |
| `developer_guide/adding_algorithms.rst` | 73 | Complete | Not applicable | Stale tableau fields, false FIRK transformation requirement, incomplete family steps, removed helper API | DG-01, DG-02, DG-03, STYLE-RST-01, STYLE-FRAME-01 |
| `developer_guide/adding_metrics.rst` | 118 | Complete | Not applicable | Example imports, metadata, signatures, CUDA import, decorators, reset, and registration are stale | DG-04, DG-05, DG-06, STYLE-FRAME-01 |
| `developer_guide/architecture.rst` | 73 | Complete | Not applicable | Ownership, factory lifecycle, config update, and prefixed-instance descriptions are stale | DG-07, DG-08, DG-09, DG-10, STYLE-RST-02, STYLE-FRAME-01 |
| `developer_guide/buffer_registry.rst` | 95 | Complete | Not applicable | Registry scope, local-memory claim, registration timing, allocator signatures, budget policy, and units are stale | DG-11, DG-12, DG-13, STYLE-RST-01 |
| `developer_guide/codegen.rst` | 89 | Complete | Not applicable | Parser scope is incomplete; helper API, stage utilities, cache location, and JIT timing are stale | DG-14, DG-15, DG-16, STYLE-RST-01, STYLE-RST-02 |
| `developer_guide/testing.rst` | 72 | Complete | Not applicable | Commands are not PowerShell or current marker selections; marker, fixture, and test-rule prose is stale | DG-17, DG-18, DG-19, STYLE-RST-01 |

### API root and batch solving

| Page | Lines | Read | Autodoc target and status | Freshness and completeness | Issues |
|---|---:|---|---|---|---|
| `API_reference/index.rst` | 18 | Complete | Not applicable | Module tree is current; opening is framed | API-01, STYLE-FRAME-01 |
| `batchsolving/index.rst` | 62 | Complete | Not applicable | Omits three public exports and mislabels `solve_ivp` as single-run | API-01, API-03, STYLE-FRAME-01 |
| `batchsolving/arrays.rst` | 57 | Complete | Not applicable | Array lifetime and transfer description is stale; `ActiveOutputs` is placed under the arrays subtree | API-02, API-03, STYLE-RST-02, STYLE-FRAME-01 |
| `batchsolving/active_outputs.rst` | 8 | Complete | `cubie.batchsolving.ActiveOutputs` resolved at `src/cubie/batchsolving/BatchSolverConfig.py:35` | API target fresh; rendered prose needs direct wording | STYLE-DOC-03 |
| `batchsolving/array_container.rst` | 8 | Complete | `cubie.batchsolving.ArrayContainer` resolved at `src/cubie/batchsolving/arrays/BaseArrayManager.py:227` | Fresh | CLEAN |
| `batchsolving/base_array_manager.rst` | 8 | Complete | `cubie.batchsolving.BaseArrayManager` resolved at `src/cubie/batchsolving/arrays/BaseArrayManager.py:313` | API target fresh; rendered member prose has framing and punctuation findings | STYLE-DOC-02, STYLE-DOC-03 |
| `batchsolving/batch_solver_config.rst` | 9 | Complete | `cubie.batchsolving.BatchSolverConfig` resolved at `src/cubie/batchsolving/BatchSolverConfig.py:132` | API target fresh; rendered class prose has punctuation findings | STYLE-DOC-02 |
| `batchsolving/batch_solver_kernel.rst` | 8 | Complete | `cubie.batchsolving.BatchSolverKernel` resolved at `src/cubie/batchsolving/BatchSolverKernel.py:211` | API target fresh; rendered prose has em dashes, semicolons, colons, and framing | STYLE-DOC-01, STYLE-DOC-02, STYLE-DOC-03 |
| `batchsolving/input_array_container.rst` | 8 | Complete | `cubie.batchsolving.InputArrayContainer` resolved at `src/cubie/batchsolving/arrays/BatchInputArrays.py:60` | API target fresh; one rendered colon rewrite | STYLE-DOC-03 |
| `batchsolving/input_arrays.rst` | 8 | Complete | `cubie.batchsolving.InputArrays` resolved at `src/cubie/batchsolving/arrays/BatchInputArrays.py:128` | API target fresh; rendered punctuation findings | STYLE-DOC-02, STYLE-DOC-03 |
| `batchsolving/managed_array.rst` | 8 | Complete | `cubie.batchsolving.ManagedArray` resolved at `src/cubie/batchsolving/arrays/BaseArrayManager.py:72` | Fresh | CLEAN |
| `batchsolving/output_array_container.rst` | 8 | Complete | `cubie.batchsolving.OutputArrayContainer` resolved at `src/cubie/batchsolving/arrays/BatchOutputArrays.py:61` | API target fresh; one rendered colon rewrite | STYLE-DOC-03 |
| `batchsolving/output_arrays.rst` | 8 | Complete | `cubie.batchsolving.OutputArrays` resolved at `src/cubie/batchsolving/arrays/BatchOutputArrays.py:152` | API target fresh; rendered factory wording is framed | STYLE-DOC-03 |
| `batchsolving/solve_ivp.rst` | 6 | Complete | `cubie.batchsolving.solve_ivp` resolved at `src/cubie/batchsolving/solver.py:215` | Target fresh; rendered docstring has semicolon findings | STYLE-DOC-02 |
| `batchsolving/solve_result.rst` | 8 | Complete | `cubie.batchsolving.SolveResult` resolved at `src/cubie/batchsolving/solveresult.py:195` | Target fresh; rendered ownership prose has em-dash, semicolon, and framing findings | STYLE-DOC-01, STYLE-DOC-02, STYLE-DOC-03 |
| `batchsolving/solve_spec.rst` | 9 | Complete | `cubie.batchsolving.SolveSpec` resolved at `src/cubie/batchsolving/solveresult.py:116` | Fresh | CLEAN |
| `batchsolving/solver.rst` | 8 | Complete | `cubie.batchsolving.Solver` resolved at `src/cubie/batchsolving/solver.py:350` | Target fresh; extensive rendered punctuation findings | STYLE-DOC-01, STYLE-DOC-02, STYLE-DOC-03 |
| `batchsolving/system_interface.rst` | 8 | Complete | `cubie.batchsolving.SystemInterface` resolved at `src/cubie/batchsolving/SystemInterface.py:48` | Target fresh; rendered update description is framed | STYLE-DOC-03 |

### Integrator package and algorithms

| Page | Lines | Read | Autodoc target and status | Freshness and completeness | Issues |
|---|---:|---|---|---|---|
| `integrators/index.rst` | 53 | Complete | Not applicable | Misidentifies primary interface, references unexported settings type, and names removed result-code class | INT-01, INT-02, INT-15, STYLE-FRAME-01 |
| `integrators/integrator_return_codes.rst` | 8 | Complete | `cubie.integrators.IntegratorReturnCodes` does not resolve | Broken target; current object is `CUBIE_RESULT_CODES` | INT-02 |
| `integrators/single_integrator_run.rst` | 8 | Complete | `cubie.integrators.SingleIntegratorRun` resolved at `src/cubie/integrators/SingleIntegratorRun.py:38` | Target fresh; internal settings and core composition lack a reference page | INT-15 |
| `integrators/algorithms.rst` | 192 | Complete | Not applicable | Family classification, adaptive selection, deadbands, tolerances, cache description, and resolver types are stale | INT-05, INT-06, INT-07, INT-08, INT-09, STYLE-RST-01, STYLE-RST-02, STYLE-FRAME-01 |
| `algorithms/backwards_euler_pc_step.rst` | 17 | Complete | `cubie.integrators.algorithms.BackwardsEulerPCStep` resolved at `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py:27` | Facts current; two em-dash rewrites | STYLE-RST-01 |
| `algorithms/backwards_euler_step.rst` | 17 | Complete | `cubie.integrators.algorithms.BackwardsEulerStep` resolved at `src/cubie/integrators/algorithms/backwards_euler.py:61` | Facts current; one semicolon rewrite | STYLE-RST-01 |
| `algorithms/base_algorithm_step.rst` | 8 | Complete | `cubie.integrators.algorithms.base_algorithm_step.BaseAlgorithmStep` resolved at `src/cubie/integrators/algorithms/base_algorithm_step.py:705` | Target fresh; rendered base description is framed | STYLE-DOC-03 |
| `algorithms/base_step_config.rst` | 8 | Complete | `cubie.integrators.algorithms.base_algorithm_step.BaseStepConfig` resolved at `src/cubie/integrators/algorithms/base_algorithm_step.py:596` | Fresh | CLEAN |
| `algorithms/crank_nicolson_step.rst` | 19 | Complete | `cubie.integrators.algorithms.CrankNicolsonStep` resolved at `src/cubie/integrators/algorithms/crank_nicolson.py:71` | Facts current; one semicolon rewrite | STYLE-RST-01 |
| `algorithms/explicit_euler_step.rst` | 17 | Complete | `cubie.integrators.algorithms.ExplicitEulerStep` resolved at `src/cubie/integrators/algorithms/explicit_euler.py:47` | Fresh | CLEAN |
| `algorithms/explicit_step_config.rst` | 8 | Complete | `cubie.integrators.algorithms.ExplicitStepConfig` resolved at `src/cubie/integrators/algorithms/ode_explicitstep.py:36` | Fresh | CLEAN |
| `algorithms/generic_dirk_step.rst` | 33 | Complete | `cubie.integrators.algorithms.DIRKStep` resolved at `src/cubie/integrators/algorithms/generic_dirk.py:181`; `generic_dirk.DIRKStepConfig` resolved at line 113 | Universal adaptive claim is false; RST and rendered punctuation findings | INT-11, STYLE-RST-01, STYLE-DOC-02, STYLE-DOC-03 |
| `algorithms/generic_dirk_tableaus.rst` | 66 | Complete | `DIRK_TABLEAU_REGISTRY` resolved at `src/cubie/integrators/algorithms/generic_dirk_tableaus.py:429`; `DIRKTableau` resolved at line 56 | Alias table omits `ode23t`, `kvaerno3`, and `kvaerno5` | INT-10, STYLE-RST-01, STYLE-DOC-02 |
| `algorithms/generic_erk_step.rst` | 31 | Complete | `cubie.integrators.algorithms.ERKStep` resolved at `src/cubie/integrators/algorithms/generic_erk.py:146`; `generic_erk.ERKStepConfig` resolved at line 115 | Facts current; RST punctuation findings | STYLE-RST-01, STYLE-DOC-03 |
| `algorithms/generic_erk_tableaus.rst` | 77 | Complete | `ERK_TABLEAU_REGISTRY` resolved at `src/cubie/integrators/algorithms/generic_erk_tableaus.py:751`; `ERKTableau` resolved at line 43 | Alias table omits 11 current keys; introduction has framing language | INT-10, STYLE-FRAME-01 |
| `algorithms/generic_firk_step.rst` | 32 | Complete | `cubie.integrators.algorithms.FIRKStep` resolved at `src/cubie/integrators/algorithms/generic_firk.py:180`; `generic_firk.FIRKStepConfig` resolved at line 117 | Universal adaptive claim is false; punctuation findings | INT-11, STYLE-RST-01, STYLE-DOC-02, STYLE-DOC-03 |
| `algorithms/generic_firk_tableaus.rst` | 48 | Complete | `FIRK_TABLEAU_REGISTRY` resolved at `src/cubie/integrators/algorithms/generic_firk_tableaus.py:219`; `FIRKTableau` resolved at line 53 | Omits the order-eight four-stage Gauss-Legendre tableau | INT-10 |
| `algorithms/generic_rosenbrock_step.rst` | 33 | Complete | `cubie.integrators.algorithms.GenericRosenbrockWStep` resolved at `src/cubie/integrators/algorithms/generic_rosenbrock_w.py:155`; `RosenbrockWStepConfig` resolved at line 119 | False factorization claim; punctuation findings | INT-12, STYLE-RST-01, STYLE-DOC-03 |
| `algorithms/generic_rosenbrock_tableaus.rst` | 54 | Complete | `ROSENBROCK_TABLEAUS` resolved at `src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py:416`; `RosenbrockTableau` resolved at line 43 | Registry rows match all active keys; introduction has framing language | STYLE-FRAME-01 |
| `algorithms/get_algorithm_step.rst` | 6 | Complete | `cubie.integrators.algorithms.get_algorithm_step` resolved at `src/cubie/integrators/algorithms/__init__.py:150` | Target fresh; narrative page describes enum input incorrectly | INT-09 |
| `algorithms/implicit_step_config.rst` | 8 | Complete | `cubie.integrators.algorithms.ImplicitStepConfig` resolved at `src/cubie/integrators/algorithms/ode_implicitstep.py:91` | Target fresh; one rendered colon rewrite | STYLE-DOC-03 |
| `algorithms/step_cache.rst` | 8 | Complete | `cubie.integrators.algorithms.base_algorithm_step.StepCache` resolved at `src/cubie/integrators/algorithms/base_algorithm_step.py:685` | Target fresh; parent-page description is stale | INT-09 |

### Integrator loops, matrix-free solvers, and step control

| Page | Lines | Read | Autodoc target and status | Freshness and completeness | Issues |
|---|---:|---|---|---|---|
| `integrators/loops.rst` | 40 | Complete | Not applicable | ODELoopConfig is assigned layout responsibility that belongs to BufferRegistry | INT-14 |
| `loops/ivp_loop.rst` | 8 | Complete | `cubie.integrators.loops.IVPLoop` resolved at `src/cubie/integrators/loops/ode_loop.py:132` | Fresh | CLEAN |
| `loops/ode_loop_config.rst` | 9 | Complete | `cubie.integrators.loops.ode_loop_config.ODELoopConfig` resolved at `src/cubie/integrators/loops/ode_loop_config.py:44` | Target fresh; one rendered semicolon finding | INT-14, STYLE-DOC-02 |
| `integrators/matrix_free_solvers.rst` | 56 | Complete | Not applicable | Removed names, missing BiCGSTAB, false damping, wrong ownership, wrong CUDA import path | INT-02, INT-03, INT-04, INT-15 |
| `matrix_free_solvers/linear_solver_factory.rst` | 16 | Complete | `LinearSolver` broken; `LinearSolverConfig` broken; `LinearSolverCache` resolved at `src/cubie/integrators/matrix_free_solvers/linear_solver_base.py:152` | Two broken targets and missing current linear solvers | INT-03, INT-15, STYLE-DOC-03 |
| `matrix_free_solvers/newton_krylov_solver_factory.rst` | 16 | Complete | `NewtonKrylov` resolved at `src/cubie/integrators/matrix_free_solvers/newton_krylov.py:161`; `NewtonKrylovConfig` at line 69; `NewtonKrylovCache` at line 149 | Targets fresh; rendered punctuation findings | STYLE-DOC-02, STYLE-DOC-03 |
| `matrix_free_solvers/solver_ret_codes.rst` | 8 | Complete | `cubie.integrators.matrix_free_solvers.SolverRetCodes` does not resolve | Broken target; current object is `CUBIE_RESULT_CODES` | INT-02 |
| `integrators/step_control.rst` | 107 | Complete | Not applicable | Suggested beta table does not match current config fields or defaults | INT-13 |
| `step_control/adaptive_i_controller.rst` | 8 | Complete | `cubie.integrators.step_control.AdaptiveIController` resolved at `src/cubie/integrators/step_control/adaptive_I_controller.py:36` | Fresh | CLEAN |
| `step_control/adaptive_pi_controller.rst` | 8 | Complete | `cubie.integrators.step_control.AdaptivePIController` resolved at `src/cubie/integrators/step_control/adaptive_PI_controller.py:79` | Fresh | CLEAN |
| `step_control/adaptive_pid_controller.rst` | 8 | Complete | `cubie.integrators.step_control.AdaptivePIDController` resolved at `src/cubie/integrators/step_control/adaptive_PID_controller.py:70` | Fresh | CLEAN |
| `step_control/adaptive_step_control_config.rst` | 8 | Complete | `cubie.integrators.step_control.AdaptiveStepControlConfig` resolved at `src/cubie/integrators/step_control/adaptive_step_controller.py:85` | Target fresh; parent page defaults are stale | INT-13 |
| `step_control/base_adaptive_step_controller.rst` | 8 | Complete | `cubie.integrators.step_control.BaseAdaptiveStepController` resolved at `src/cubie/integrators/step_control/adaptive_step_controller.py:199` | Fresh | CLEAN |
| `step_control/base_step_controller_config.rst` | 9 | Complete | `cubie.integrators.step_control.BaseStepControllerConfig` resolved at `src/cubie/integrators/step_control/base_step_controller.py:166` | Target fresh; one rendered semicolon finding | STYLE-DOC-02 |
| `step_control/base_step_controller.rst` | 8 | Complete | `cubie.integrators.step_control.BaseStepController` resolved at `src/cubie/integrators/step_control/base_step_controller.py:244` | Fresh | CLEAN |
| `step_control/fixed_step_control_config.rst` | 8 | Complete | `cubie.integrators.step_control.FixedStepControlConfig` resolved at `src/cubie/integrators/step_control/fixed_step_controller.py:42` | Fresh | CLEAN |
| `step_control/fixed_step_controller.rst` | 8 | Complete | `cubie.integrators.step_control.FixedStepController` resolved at `src/cubie/integrators/step_control/fixed_step_controller.py:97` | Fresh | CLEAN |
| `step_control/get_controller.rst` | 6 | Complete | `cubie.integrators.step_control.get_controller` resolved at `src/cubie/integrators/step_control/__init__.py:66` | Fresh | CLEAN |
| `step_control/gustafsson_controller.rst` | 8 | Complete | `cubie.integrators.step_control.GustafssonController` resolved at `src/cubie/integrators/step_control/gustafsson_controller.py:84` | Fresh | CLEAN |
| `step_control/gustafsson_step_control_config.rst` | 8 | Complete | `cubie.integrators.step_control.GustafssonStepControlConfig` resolved at `src/cubie/integrators/step_control/gustafsson_controller.py:54` | Target fresh; parent page defaults are stale | INT-13 |
| `step_control/pi_step_control_config.rst` | 8 | Complete | `cubie.integrators.step_control.PIStepControlConfig` resolved at `src/cubie/integrators/step_control/adaptive_PI_controller.py:49` | Target fresh; parent page defaults are stale | INT-13 |
| `step_control/pid_step_control_config.rst` | 8 | Complete | `cubie.integrators.step_control.PIDStepControlConfig` resolved at `src/cubie/integrators/step_control/adaptive_PID_controller.py:52` | Target fresh; parent page defaults are stale | INT-13 |

## Broken autodoc target inventory

| Page and line | Broken target | Current source-backed target |
|---|---|---|
| `integrators/integrator_return_codes.rst:6` | `cubie.integrators.IntegratorReturnCodes` | `cubie.integrators.CUBIE_RESULT_CODES`, exported at `src/cubie/integrators/__init__.py:36,68-93` |
| `matrix_free_solvers/linear_solver_factory.rst:6` | `LinearSolver` | `MRLinearSolver` and `BiCGSTABSolver`, exported at `src/cubie/integrators/matrix_free_solvers/__init__.py:29-42` |
| `matrix_free_solvers/linear_solver_factory.rst:10` | `LinearSolverConfig` | `MRLinearSolverConfig` and `BiCGSTABSolverConfig`, same export block |
| `matrix_free_solvers/solver_ret_codes.rst:6` | `SolverRetCodes` | `CUBIE_RESULT_CODES`, exported at `src/cubie/integrators/matrix_free_solvers/__init__.py:29-42` |

## Source-driven completeness inventory

### Public batch-solving exports

The authoritative export list is
`src/cubie/batchsolving/__init__.py:54-74`.

| Public symbol | Coverage | Finding |
|---|---|---|
| `ActiveOutputs` | Dedicated page | Target resolves; page is placed under arrays although the class is declared with `BatchSolverConfig` |
| `ArrayContainer` | Dedicated page | Current |
| `ArrayTypes` | Mention only in arrays dependencies | No definition page or linked API target |
| `BatchInputHandler` | None | Public grid-building and direct batch-input normalization API is absent; prospective docstring findings are in STYLE-PROSPECTIVE-01 |
| `BatchSolverConfig` | Dedicated page | Current target; rendered style findings |
| `BatchSolverKernel` | Dedicated page | Current target; rendered style findings |
| `BaseArrayManager` | Dedicated page | Current target; rendered style findings |
| `DeviceSolveResult` | None | Public device-buffer lifetime and synchronization contract is absent; prospective docstring findings are in STYLE-PROSPECTIVE-01 |
| `InputArrayContainer` | Dedicated page | Current |
| `InputArrays` | Dedicated page | Current target; narrative page omits device-input passthrough and staged transfers |
| `ManagedArray` | Dedicated page | Current |
| `OutputArrayContainer` | Dedicated page | Current |
| `OutputArrays` | Dedicated page | Current target; narrative page misstates per-launch mirroring and omits loans |
| `Solver` | Dedicated page | Current target; rendered style findings |
| `SolveResult` | Dedicated page | Current target; rendered style findings |
| `SolveSpec` | Dedicated page | Current |
| `SystemInterface` | Dedicated page | Current target; rendered style finding |
| `solve_ivp` | Dedicated page | Current target; batch index mislabels it as single-run |
| `summary_metrics` | Covered in the outputhandling documentation lane | No duplicate page required here |

Missing batch capabilities that need either a page or a direct cross-reference:

- Caller-supplied device input detection and the no-copy path.
- `on_device=True`, `DeviceSolveResult`, explicit synchronization, and device
  buffer lifetime.
- Pinned staging pools and asynchronous transfer/writeback watchers.
- Output-buffer loans from `OutputArrays` to `SolveResult`.
- Spill thresholds, disk-backed arrays, and explicit `close()` lifetime.
- `BatchInputHandler` grid modes and direct device-array bypass.

These behaviors are exposed by the public classes and methods in
`src/cubie/batchsolving/solver.py:636-711`,
`src/cubie/batchsolving/arrays/BatchInputArrays.py:450-522`, and
`src/cubie/batchsolving/arrays/BatchOutputArrays.py:311-364`.

### Public integrator exports

The authoritative export list is `src/cubie/integrators/__init__.py:68-93`.

| Public symbol | Coverage | Finding |
|---|---|---|
| `SingleIntegratorRun` | Dedicated page | Current target; internal role should be stated |
| `CUBIE_RESULT_CODES` | Two pages use removed names | Replace both broken targets |
| `get_algorithm_step` | Dedicated page | Target current; parent prose incorrectly says enum or name |
| `ExplicitStepConfig` | Dedicated page | Current |
| `ImplicitStepConfig` | Dedicated page | Current target; rendered colon finding |
| `ExplicitEulerStep` | Dedicated page | Current |
| `BackwardsEulerStep` | Dedicated page | Current target; RST semicolon finding |
| `BackwardsEulerPCStep` | Dedicated page | Current target; RST em-dash findings |
| `CrankNicolsonStep` | Dedicated page | Current target; RST semicolon finding |
| `IVPLoop` | Dedicated page | Current |
| `MRLinearSolver` | Removed-name page | Replace broken `LinearSolver` target |
| `MRLinearSolverConfig` | Removed-name page | Replace broken `LinearSolverConfig` target |
| `LinearSolverCache` | Dedicated target on removed-name page | Target current |
| `BiCGSTABSolver` | None | Add target and describe breakdown/status behavior; prospective docstring findings are in STYLE-PROSPECTIVE-01 |
| `BiCGSTABSolverConfig` | None | Add target; prospective docstring findings are in STYLE-PROSPECTIVE-01 |
| `NewtonKrylov` | Dedicated page | Current target; overview falsely says damped |
| `NewtonKrylovConfig` | Dedicated page | Current target; rendered punctuation findings |
| `NewtonKrylovCache` | Dedicated page | Current |
| `AdaptiveIController` | Dedicated page | Current |
| `AdaptivePIController` | Dedicated page | Current |
| `AdaptivePIDController` | Dedicated page | Current |
| `FixedStepController` | Dedicated page | Current |
| `GustafssonController` | Dedicated page | Current |
| `get_controller` | Dedicated page | Current |

The algorithms subpackage publicly exports `algorithm_is_adaptive` at
`src/cubie/integrators/algorithms/__init__.py:31-53`; it has no API target.

Internal capabilities that materially affect documented settings but have no
focused coverage:

- `SingleIntegratorRunCore` tolerance derivation and
  `IntegratorRunSettings` composition.
- `memory_heuristics` and automatic memory-policy selection.
- Scaled, tiled, correction, and FIRK norms used by matrix-free solvers.
- Dense stage prediction and its applicability limits.
- BiCGSTAB scratch layout, residual reduction, residual floor, and breakdown
  result codes.

### Registry freshness

| Registry | Documentation status | Source |
|---|---|---|
| `ERK_TABLEAU_REGISTRY` | Eleven active aliases omitted | `src/cubie/integrators/algorithms/generic_erk_tableaus.py:751-771` |
| `DIRK_TABLEAU_REGISTRY` | `ode23t`, `kvaerno3`, and `kvaerno5` omitted | `src/cubie/integrators/algorithms/generic_dirk_tableaus.py:429-438` |
| `FIRK_TABLEAU_REGISTRY` | `firk_gauss_legendre_4` omitted | `src/cubie/integrators/algorithms/generic_firk_tableaus.py:219-224` |
| `ROSENBROCK_TABLEAUS` | All active aliases present | `src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py:416-424` |

## Language-sweep accounting

- Every authored RST em dash and semicolon is listed with an exact replacement
  in `STYLE-RST-01` of the proposal artifact.
- Every rendered autodoc em dash is listed in `STYLE-DOC-01`.
- Every rendered autodoc semicolon is listed in `STYLE-DOC-02`.
- Punctuation and framing in four proposed missing API targets are listed in
  `STYLE-PROSPECTIVE-01`.
- Non-structural colon rewrites and retained structural colon categories are
  recorded in `STYLE-RST-02` and `STYLE-DOC-03`.
- Every detected framing or intensifying tell is listed in
  `STYLE-FRAME-01` or `STYLE-DOC-03`.
- No scoped prose contains false-drama staccato matching the supplied examples.
