# Integrator source and test inventory

## Coverage receipt

The manifest came from this command.

```powershell
git ls-files -- 'src/cubie/integrators/**' 'tests/integrators/**' `
  'tests/integrated_numerical_tests/**'
```

The command returned 115 tracked entries. The manifest reconciles as
follows.

| Entry class | Files | Physical lines or bytes | Review state |
|---|---:|---:|---|
| Source text under `src/cubie/integrators/**` | 45 | 17,687 lines | READ-COMPLETE |
| Test and reference text in the two test trees | 63 | 16,869 lines | READ-COMPLETE |
| `CLAUDE.md` symbolic-link mirrors | 5 | One-line link payload each | MIRROR-RECONCILED |
| Previously audited Julia data README | 1 | 29 lines | EXCLUDED-PRIOR-AUDIT |
| Binary Julia reference data | 1 | 4,171,336 bytes | BINARY-INVENTORIED |
| Total | 115 | 34,556 newly read text lines | RECONCILED |

Every line of all 108 newly scoped text files was read. Empty tracked files
are recorded with zero physical lines and are vacuously complete. The
prerequisite instruction files `AGENTS.md` at the repository root and
`src/cubie/AGENTS.md` were also read in full. They contain 126 and 210
physical lines respectively.

Impact labels refer to entries in
`integrator_supplemental_proposals.md`. `EXISTING` means the issue is
already captured by another review artifact. `STYLE-AI-01` is detailed
below.

## Integrator source manifest

| Tracked file | Lines | Read state | Documentation impact |
|---|---:|---|---|
| `src/cubie/integrators/AGENTS.md` | 105 | READ-COMPLETE | D2-05, D2-06, EXISTING INT-15 |
| `src/cubie/integrators/IntegratorRunSettings.py` | 54 | READ-COMPLETE | NONE |
| `src/cubie/integrators/SingleIntegratorRun.py` | 471 | READ-COMPLETE | NONE |
| `src/cubie/integrators/SingleIntegratorRunCore.py` | 931 | READ-COMPLETE | EVIDENCE D2-01, D2-05 |
| `src/cubie/integrators/__init__.py` | 93 | READ-COMPLETE | EXISTING INT-15 |
| `src/cubie/integrators/algorithms/AGENTS.md` | 165 | READ-COMPLETE | D2-02, D2-03 |
| `src/cubie/integrators/algorithms/__init__.py` | 216 | READ-COMPLETE | EXISTING INT-10, INT-15 |
| `src/cubie/integrators/algorithms/backwards_euler.py` | 339 | READ-COMPLETE | NONE |
| `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py` | 216 | READ-COMPLETE | NONE |
| `src/cubie/integrators/algorithms/base_algorithm_step.py` | 912 | READ-COMPLETE | D2-01 |
| `src/cubie/integrators/algorithms/crank_nicolson.py` | 378 | READ-COMPLETE | NONE |
| `src/cubie/integrators/algorithms/explicit_euler.py` | 266 | READ-COMPLETE | NONE |
| `src/cubie/integrators/algorithms/generic_dirk.py` | 872 | READ-COMPLETE | EXISTING rendered punctuation |
| `src/cubie/integrators/algorithms/generic_dirk_tableaus.py` | 442 | READ-COMPLETE | EXISTING INT-10 and rendered punctuation |
| `src/cubie/integrators/algorithms/generic_erk.py` | 599 | READ-COMPLETE | NONE |
| `src/cubie/integrators/algorithms/generic_erk_tableaus.py` | 772 | READ-COMPLETE | EXISTING INT-10 |
| `src/cubie/integrators/algorithms/generic_firk.py` | 731 | READ-COMPLETE | EXISTING rendered punctuation |
| `src/cubie/integrators/algorithms/generic_firk_tableaus.py` | 224 | READ-COMPLETE | D2-03, EXISTING INT-10 |
| `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` | 789 | READ-COMPLETE | NONE |
| `src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py` | 442 | READ-COMPLETE | D2-04 |
| `src/cubie/integrators/algorithms/ode_explicitstep.py` | 106 | READ-COMPLETE | NONE |
| `src/cubie/integrators/algorithms/ode_implicitstep.py` | 728 | READ-COMPLETE | EVIDENCE D2-01, D2-02, EXISTING STYLE-DOC-03 |
| `src/cubie/integrators/loops/AGENTS.md` | 118 | READ-COMPLETE | D2-07, D2-11 |
| `src/cubie/integrators/loops/__init__.py` | 9 | READ-COMPLETE | NONE |
| `src/cubie/integrators/loops/ode_loop.py` | 1,121 | READ-COMPLETE | D2-09 |
| `src/cubie/integrators/loops/ode_loop_config.py` | 325 | READ-COMPLETE | EVIDENCE D2-07, EXISTING rendered punctuation |
| `src/cubie/integrators/matrix_free_solvers/AGENTS.md` | 142 | READ-COMPLETE | D2-06, D2-08, D2-11 |
| `src/cubie/integrators/matrix_free_solvers/__init__.py` | 42 | READ-COMPLETE | EVIDENCE D2-02, EXISTING INT-03, INT-15 |
| `src/cubie/integrators/matrix_free_solvers/base_solver.py` | 201 | READ-COMPLETE | NONE |
| `src/cubie/integrators/matrix_free_solvers/bicgstab_solver.py` | 730 | READ-COMPLETE | EVIDENCE D2-06, EXISTING INT-15 |
| `src/cubie/integrators/matrix_free_solvers/linear_solver.py` | 559 | READ-COMPLETE | D2-10 |
| `src/cubie/integrators/matrix_free_solvers/linear_solver_base.py` | 340 | READ-COMPLETE | NONE |
| `src/cubie/integrators/matrix_free_solvers/newton_krylov.py` | 636 | READ-COMPLETE | D2-10, EVIDENCE D2-06, EXISTING INT-04 |
| `src/cubie/integrators/memory_heuristics.py` | 327 | READ-COMPLETE | EXISTING INT-15 |
| `src/cubie/integrators/norms.py` | 481 | READ-COMPLETE | EVIDENCE D2-08, EXISTING INT-15 |
| `src/cubie/integrators/stage_predictors.py` | 546 | READ-COMPLETE | EXISTING INT-15 |
| `src/cubie/integrators/step_control/AGENTS.md` | 87 | READ-COMPLETE | D2-11 |
| `src/cubie/integrators/step_control/__init__.py` | 121 | READ-COMPLETE | NONE |
| `src/cubie/integrators/step_control/adaptive_I_controller.py` | 193 | READ-COMPLETE | NONE |
| `src/cubie/integrators/step_control/adaptive_PID_controller.py` | 275 | READ-COMPLETE | NONE |
| `src/cubie/integrators/step_control/adaptive_PI_controller.py` | 265 | READ-COMPLETE | NONE |
| `src/cubie/integrators/step_control/adaptive_step_controller.py` | 392 | READ-COMPLETE | NONE |
| `src/cubie/integrators/step_control/base_step_controller.py` | 459 | READ-COMPLETE | EXISTING rendered punctuation |
| `src/cubie/integrators/step_control/fixed_step_controller.py` | 180 | READ-COMPLETE | NONE |
| `src/cubie/integrators/step_control/gustafsson_controller.py` | 287 | READ-COMPLETE | NONE |

Source subtotal is 45 files and 17,687 physical lines.

## Integrated numerical test manifest

| Tracked file | Lines | Read state | Documentation impact |
|---|---:|---|---|
| `tests/integrated_numerical_tests/__init__.py` | 0 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/julia_reference/__init__.py` | 0 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/julia_reference/conftest.py` | 69 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/julia_reference/data/algorithms.csv` | 23 | READ-COMPLETE | EVIDENCE D2-04 |
| `tests/integrated_numerical_tests/julia_reference/data/controller_constants.csv` | 21 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/julia_reference/ne_gate.py` | 247 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/julia_reference/test_julia_reference.py` | 197 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/test_chunking.py` | 53 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/test_dt_min_hang.py` | 115 | READ-COMPLETE | EVIDENCE D2-09 |
| `tests/integrated_numerical_tests/test_numerical_equivalence.py` | 33 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/test_ode_loop.py` | 623 | READ-COMPLETE | EVIDENCE D2-03 |
| `tests/integrated_numerical_tests/test_output_functions.py` | 741 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/test_rejection_liveness.py` | 37 | READ-COMPLETE | EVIDENCE D2-09 |
| `tests/integrated_numerical_tests/test_status_staining.py` | 121 | READ-COMPLETE | EVIDENCE D2-09 |
| `tests/integrated_numerical_tests/test_step_algorithms.py` | 503 | READ-COMPLETE | NONE |
| `tests/integrated_numerical_tests/test_step_controllers.py` | 507 | READ-COMPLETE | NONE |

Integrated numerical test subtotal is 16 text files and 3,290 physical
lines.

## Integrator unit and reference test manifest

| Tracked file | Lines | Read state | Documentation impact |
|---|---:|---|---|
| `tests/integrators/__init__.py` | 0 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/__init__.py` | 0 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_base_algorithm_step.py` | 129 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_dirk_tableaus.py` | 225 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_explicit_euler.py` | 48 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_generic_dirk.py` | 201 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_generic_erk_tableaus.py` | 84 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_generic_firk.py` | 69 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_generic_rosenbrock_w.py` | 59 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_init.py` | 248 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_last_step_caching_integration.py` | 21 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_ode_explicitstep.py` | 54 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_ode_implicitstep.py` | 297 | READ-COMPLETE | EVIDENCE D2-01, D2-02 |
| `tests/integrators/algorithms/test_rosenbrock_tableaus.py` | 54 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_step_algorithms.py` | 791 | READ-COMPLETE | NONE |
| `tests/integrators/algorithms/test_tableau_properties.py` | 108 | READ-COMPLETE | NONE |
| `tests/integrators/cpu_reference/__init__.py` | 37 | READ-COMPLETE | STYLE-AI-01, D2-12 |
| `tests/integrators/cpu_reference/algorithms.py` | 2,069 | READ-COMPLETE | NONE |
| `tests/integrators/cpu_reference/cpu_ode_system.py` | 523 | READ-COMPLETE | NONE |
| `tests/integrators/cpu_reference/cpu_utils.py` | 869 | READ-COMPLETE | STYLE-AI-01, D2-12, EVIDENCE D2-11 |
| `tests/integrators/cpu_reference/loops.py` | 323 | READ-COMPLETE | EVIDENCE D2-11 |
| `tests/integrators/cpu_reference/step_controllers.py` | 219 | READ-COMPLETE | EVIDENCE D2-11 |
| `tests/integrators/cpu_reference/test_cpu_utils.py` | 246 | READ-COMPLETE | NONE |
| `tests/integrators/loops/__init__.py` | 0 | READ-COMPLETE | NONE |
| `tests/integrators/loops/test_buffer_settings.py` | 125 | READ-COMPLETE | NONE |
| `tests/integrators/loops/test_interp_vs_symbolic.py` | 56 | READ-COMPLETE | NONE |
| `tests/integrators/loops/test_ode_loop.py` | 70 | READ-COMPLETE | NONE |
| `tests/integrators/loops/test_ode_loop_config.py` | 281 | READ-COMPLETE | EVIDENCE D2-07 |
| `tests/integrators/matrix_free_solvers/__init__.py` | 0 | READ-COMPLETE | EVIDENCE D2-11 |
| `tests/integrators/matrix_free_solvers/conftest.py` | 836 | READ-COMPLETE | EVIDENCE D2-06, D2-11 |
| `tests/integrators/matrix_free_solvers/test_base_solver.py` | 167 | READ-COMPLETE | EVIDENCE D2-11 |
| `tests/integrators/matrix_free_solvers/test_bicgstab.py` | 199 | READ-COMPLETE | EVIDENCE D2-11 |
| `tests/integrators/matrix_free_solvers/test_linear_solver.py` | 807 | READ-COMPLETE | EVIDENCE D2-06, D2-11 |
| `tests/integrators/matrix_free_solvers/test_newton_krylov.py` | 494 | READ-COMPLETE | EVIDENCE D2-11 |
| `tests/integrators/step_control/test_adaptive_step_controller.py` | 484 | READ-COMPLETE | NONE |
| `tests/integrators/step_control/test_controllers.py` | 272 | READ-COMPLETE | NONE |
| `tests/integrators/step_control/test_fixed_step_controller.py` | 179 | READ-COMPLETE | NONE |
| `tests/integrators/step_control/test_gain_specs.py` | 133 | READ-COMPLETE | EVIDENCE D2-11 |
| `tests/integrators/step_control/test_gustafsson_controller.py` | 89 | READ-COMPLETE | EVIDENCE D2-11 |
| `tests/integrators/step_control/test_init.py` | 92 | READ-COMPLETE | EVIDENCE D2-11 |
| `tests/integrators/test_IntegratorRunSettings.py` | 50 | READ-COMPLETE | NONE |
| `tests/integrators/test_SingleIntegratorRun.py` | 242 | READ-COMPLETE | NONE |
| `tests/integrators/test_SingleIntegratorRunCore.py` | 1,071 | READ-COMPLETE | EVIDENCE D2-05 |
| `tests/integrators/test_init.py` | 65 | READ-COMPLETE | NONE |
| `tests/integrators/test_memory_heuristics.py` | 116 | READ-COMPLETE | EXISTING INT-15 |
| `tests/integrators/test_norms.py` | 673 | READ-COMPLETE | EVIDENCE D2-08 |
| `tests/integrators/test_stage_predictors.py` | 404 | READ-COMPLETE | EVIDENCE D2-03, EXISTING INT-15 |

Integrator unit and reference test subtotal is 47 files and 13,579
physical lines. The two test subtotals reconcile to 63 files and 16,869
physical lines.

## Symbolic-link mirrors

Git records each entry below with mode `120000` and the one-line payload
`AGENTS.md`. The filesystem resolves each link to the sibling `AGENTS.md`
already listed and read above. They were not counted as duplicate text.

| Tracked file | Payload lines | State | Target |
|---|---:|---|---|
| `src/cubie/integrators/CLAUDE.md` | 1 | MIRROR-RECONCILED | `src/cubie/integrators/AGENTS.md` |
| `src/cubie/integrators/algorithms/CLAUDE.md` | 1 | MIRROR-RECONCILED | `src/cubie/integrators/algorithms/AGENTS.md` |
| `src/cubie/integrators/loops/CLAUDE.md` | 1 | MIRROR-RECONCILED | `src/cubie/integrators/loops/AGENTS.md` |
| `src/cubie/integrators/matrix_free_solvers/CLAUDE.md` | 1 | MIRROR-RECONCILED | `src/cubie/integrators/matrix_free_solvers/AGENTS.md` |
| `src/cubie/integrators/step_control/CLAUDE.md` | 1 | MIRROR-RECONCILED | `src/cubie/integrators/step_control/AGENTS.md` |

## Exclusion and binary inventory

| Tracked file | Size | State | Reason |
|---|---:|---|---|
| `tests/integrated_numerical_tests/julia_reference/data/README.md` | 29 lines | EXCLUDED-PRIOR-AUDIT | The user-facing lane already audited every line and recorded `UFR-030`. |
| `tests/integrated_numerical_tests/julia_reference/data/julia_reference_ne.npz` | 4,171,336 bytes | BINARY-INVENTORIED | NumPy archive with no text lines. |

## AI-language receipt

The supplied false-drama and framing examples were searched across all 108
text files after the line-by-line read. No matching instance was found.
Two documentation-bearing test docstrings directly discuss AI authorship.

### STYLE-AI-01

| Location | Exact current text | Exact proposed replacement |
|---|---|---|
| `tests/integrators/cpu_reference/__init__.py:1-6` | `Reference CPU implementations used across integrator tests.` followed by `I've let genAI agents run fairly free on this module, adding many of the over-engineered and pointless checks and complicated chains that it loves to add, as all we really want in here is a reference implementation of the GPU integrator components.` | Keep only `Reference CPU implementations used across integrator tests.` |
| `tests/integrators/cpu_reference/cpu_utils.py:626-628` | `Return Horner evaluations and range flag for coefficients. Busybody AI over-checking left in place.` | `Return Horner evaluations and range flag for coefficients.` |

These replacements remove framing, editorial judgment, and AI-authorship
commentary without changing any instruction or behavioral description.

## Freshness and completeness reconciliation

The complete read produced twelve supplemental issue groups. Existing
artifacts already cover the public-page registry gaps, public status list,
missing API targets, memory heuristics, stage prediction, and public
damped-Newton claim. Those findings are marked `EXISTING` in the manifest
and are not duplicated as public-page proposals.

No tracked text file is absent from the tables. The category equation is
`108 text + 5 mirrors + 1 prior-audit exclusion + 1 binary = 115 tracked
entries`.
