# Integrator source-document supplemental proposals

## Scope

This file records stale or missing documentation found during the complete
read of `src/cubie/integrators/**`, `tests/integrators/**`, and the scoped
integrated numerical tests. Existing review artifacts cover the public
Sphinx pages. The D2 entries identify source-docstring and `AGENTS.md`
locations absent from those artifacts.

Em dashes and semicolons in `Current text` blocks are verbatim source
quotations. Proposed prose contains neither mark.

## D2-01 Remove the obsolete mass-matrix algorithm parameter

Current location

`src/cubie/integrators/algorithms/base_algorithm_step.py:170-172`

Current text

```rst
   * - ``M``
     - :class:`ImplicitStepConfig`
     - Mass matrix for residual and Jacobian actions.
```

Proposed replacement

Delete this row. The mass matrix is part of the ODE system and is rejected
when supplied as an algorithm setting.

Evidence

- `src/cubie/integrators/SingleIntegratorRunCore.py:174-184`
- `src/cubie/integrators/algorithms/base_algorithm_step.py:62-85`
- `src/cubie/integrators/algorithms/ode_implicitstep.py:94-107`
- `tests/integrators/algorithms/test_ode_implicitstep.py:132-150`

## D2-02 Replace the removed `LinearSolver` class name

Current locations

- `src/cubie/integrators/algorithms/AGENTS.md:10-12`
- `src/cubie/integrators/algorithms/AGENTS.md:30`
- `src/cubie/integrators/algorithms/AGENTS.md:94-101`
- `src/cubie/integrators/algorithms/AGENTS.md:153-157`

Current text

```text
backward Euler (plain + predictor-corrector), and Crank-Nicolson. Implicit methods own
a `NewtonKrylov` or `LinearSolver` from `../matrix_free_solvers/`. `get_algorithm_step()`
resolves a name or `ButcherTableau` to the right factory.

| `ode_implicitstep.py` | `ImplicitStepConfig` (beta, gamma, preconditioner_order) + `ODEImplicitStep`: owns a `NewtonKrylov`/`LinearSolver`, requests operator/preconditioner/residual helpers as immutable `SolverHelperRequest`s, resolves `preconditioner_type` into concrete kinds, routes solver-param updates; `is_implicit` → `True`. |

- **Implicit** (`ODEImplicitStep`, owns a solver): `BackwardsEulerStep`,
  `BackwardsEulerPCStep`, `CrankNicolsonStep`, `DIRKStep`, `FIRKStep`,
  `GenericRosenbrockWStep`. All use **Newton-Krylov except `GenericRosenbrockWStep`**,
  which is linearly-implicit and constructs a `LinearSolver` directly
  (no Newton iteration). The class attribute `is_linear` (`True` on
  `GenericRosenbrockWStep`, `False` elsewhere) exposes the distinction.

- `cubie.integrators.matrix_free_solvers` — `NewtonKrylov`, `LinearSolver` (owned by
  implicit steps).
```

Proposed replacements

```text
Nonlinear implicit methods own a `NewtonKrylov` solver. Linearly implicit
methods own an `MRLinearSolver` or `BiCGSTABSolver`.
`get_algorithm_step()` resolves a name or `ButcherTableau` to its factory.

| `ode_implicitstep.py` | `ImplicitStepConfig` with beta, gamma, and preconditioner order. `ODEImplicitStep` owns `NewtonKrylov` for nonlinear algorithms and an `MRLinearSolver` or `BiCGSTABSolver` for linearly implicit algorithms. It requests operator, preconditioner, and residual helpers as immutable `SolverHelperRequest` values. It resolves `preconditioner_type` into concrete kinds and routes solver parameter updates. `is_implicit` is `True`. |

- Implicit `ODEImplicitStep` implementations include `BackwardsEulerStep`,
  `BackwardsEulerPCStep`, `CrankNicolsonStep`, `DIRKStep`, `FIRKStep`, and
  `GenericRosenbrockWStep`. `BackwardsEulerStep`, `BackwardsEulerPCStep`,
  `CrankNicolsonStep`, `DIRKStep`, and `FIRKStep` use `NewtonKrylov`.
  `GenericRosenbrockWStep` is linearly implicit and constructs
  an `MRLinearSolver` or `BiCGSTABSolver` directly. It does not run Newton
  iteration. `is_linear` is `True` on `GenericRosenbrockWStep` and `False`
  on the other implementations.

- `cubie.integrators.matrix_free_solvers` provides `NewtonKrylov`,
  `MRLinearSolver`, and `BiCGSTABSolver` for implicit steps.
```

Evidence

- `src/cubie/integrators/algorithms/ode_implicitstep.py:255-280`
- `src/cubie/integrators/algorithms/ode_implicitstep.py:286-318`
- `src/cubie/integrators/matrix_free_solvers/__init__.py:8-42`
- `tests/integrators/algorithms/test_ode_implicitstep.py:234-280`

## D2-03 Add Gauss-Legendre-4 to both source inventories

Current locations

- `src/cubie/integrators/algorithms/AGENTS.md:37`
- `src/cubie/integrators/algorithms/generic_firk_tableaus.py:15-28`

Current text

```text
| `generic_firk_tableaus.py` | `FIRKTableau` + Gauss-Legendre-2 (default) and Radau IIA-5; `compute_embedded_weights_radauIIA`. |
```

```rst
Constants
---------
:data:`GAUSS_LEGENDRE_2_TABLEAU`
    Two-stage, fourth-order Gauss–Legendre tableau.

:data:`RADAU_IIA_5_TABLEAU`
    Three-stage, fifth-order Radau IIA tableau with second-order
    embedded error estimate.

:data:`FIRK_TABLEAU_REGISTRY`
    Name → tableau mapping for alias-based lookup.

:data:`DEFAULT_FIRK_TABLEAU`
    Default tableau (Gauss–Legendre 2-stage).
```

Proposed replacements

```text
| `generic_firk_tableaus.py` | `FIRKTableau` with Gauss-Legendre-2 (default), Gauss-Legendre-4, and Radau IIA-5. `compute_embedded_weights_radauIIA` computes Radau IIA embedded weights from moment conditions. |
```

Proposed module entry

```rst
:data:`GAUSS_LEGENDRE_4_TABLEAU`
    Four-stage, eighth-order Gauss-Legendre tableau without an
    embedded error estimate.
```

Evidence

- `src/cubie/integrators/algorithms/generic_firk_tableaus.py:157-214`
- `src/cubie/integrators/algorithms/generic_firk_tableaus.py:216-224`
- `tests/integrated_numerical_tests/test_ode_loop.py:75-93`
- `tests/integrators/test_stage_predictors.py:83-87`
- `tests/integrators/test_stage_predictors.py:267-269`

`INT-10` and `UFR-009` record the public-page omission. D2-03 records the
two source-document omissions.

## D2-04 Correct the Rosenbrock23 method order

Current location

`src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py:17-18`

Current text

```rst
:data:`ROSENBROCK_23_SCIML_TABLEAU`
    Three-stage, third-order SciML Rosenbrock-23 variant.
```

Proposed replacement

```rst
:data:`ROSENBROCK_23_SCIML_TABLEAU`
    Three-stage, second-order SciML Rosenbrock23 method with a
    third-order embedded error estimate.
```

Evidence

- `src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py:358-405`
- `tests/integrated_numerical_tests/julia_reference/data/algorithms.csv:21`
- `docs/source/API_reference/integrators/algorithms/generic_rosenbrock_tableaus.rst:33-34`

## D2-05 Correct component registration and hot-swap mechanics

Current locations

- `src/cubie/integrators/AGENTS.md:52-67`
- `src/cubie/integrators/AGENTS.md:85-90`

Current text

```text
### Component assembly (`SingleIntegratorRunCore.__init__`)
Order matters — each component seeds the next:
1. `OutputFunctions` first (its compile flags + summary buffer heights feed `IVPLoop`).
2. `_algo_step = get_algorithm_step(precision, settings)` — supplies
   `controller_defaults.step_controller`, seeding the controller settings before user
   overrides merge in.
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

### Hot-swap
A new `"algorithm"`/`"step_controller"` in `update()` routes through
`_switch_algos()`/`_switch_controllers()`, which call
`buffer_registry.reset()`, rebuild the sub-component from the old settings as a base,
and propagate defaults into `updates_dict`. Never call these directly — go through
`update()`. Because the swap calls `buffer_registry.reset()`, any cached allocator
references become stale.
```

Proposed replacement for step 6 and the hot-swap block

```text
6. `buffer_registry.register_child(self._loop, self._algo_step,
name="algorithm")` and the controller equivalent register the active
component subtrees under the loop.

`_switch_algos()` and `_switch_controllers()` call
`buffer_registry.clear_parent()` on the outgoing component. The methods
build the replacement from the old settings. `update()` registers both
active components under the loop. A swap invalidates allocator references
for the outgoing subtree and preserves the sibling subtree.
```

Evidence

- `src/cubie/integrators/SingleIntegratorRunCore.py:251-258`
- `src/cubie/integrators/SingleIntegratorRunCore.py:700-718`
- `src/cubie/integrators/SingleIntegratorRunCore.py:736-766`
- `src/cubie/integrators/SingleIntegratorRunCore.py:817-832`
- `tests/integrators/test_SingleIntegratorRunCore.py:579-622`

## D2-06 Complete the internal result-code inventories

Current locations

- `src/cubie/integrators/AGENTS.md:40-50`
- `src/cubie/integrators/matrix_free_solvers/AGENTS.md:75-81`

Current text

```text
### CUBIE_RESULT_CODES — kernel status-bit meanings
The status vocabulary is the package-central `CUBIE_RESULT_CODES(IntFlag)` (defined in
`cubie/result_codes.py`, re-exported from this package and from `cubie`). Device functions
capture its values as closure constants and OR them into the returned status word:
`SUCCESS=0`, `MAX_NEWTON_ITERATIONS_EXCEEDED=2`,
`MAX_LINEAR_ITERATIONS_EXCEEDED=4`, `STEP_TOO_SMALL=8` (controllers' reject-at-min),
`DT_EFF_EFFECTIVELY_ZERO=16` and `MAX_LOOP_ITERS_EXCEEDED=32` (reserved, unemitted),
`STAGNATION=64` (loop no-progress). Iteration counts are returned separately via the
`counters` array, never packed into the status word. Host-side, decode via
`cubie.result_codes.decode_status_codes` (exposed as `SolveResult.status_messages` /
`Solver.status_messages`).
```

```text
### Status codes & convergence
- Status codes come from the package-central `CUBIE_RESULT_CODES` (`cubie/result_codes.py`,
  re-exported from this package): `SUCCESS=0`,
  `MAX_NEWTON_ITERATIONS_EXCEEDED=2`, `MAX_LINEAR_ITERATIONS_EXCEEDED=4` (captured as device
  closure constants). `newton_krylov_solver` OR-combines these into a **low-bits** status
  word — it does NOT pack the iteration count into high bits (counts go to `counters`).
  Callers OR this word into their own step status.
```

Proposed replacement for the package inventory

```text
`SUCCESS=0`, `MAX_NEWTON_ITERATIONS_EXCEEDED=2`,
`MAX_LINEAR_ITERATIONS_EXCEEDED=4`, `STEP_TOO_SMALL=8`,
`DT_EFF_EFFECTIVELY_ZERO=16` (reserved and unemitted),
`MAX_LOOP_ITERS_EXCEEDED=32` (reserved and unemitted), `STAGNATION=64`,
`BICGSTAB_BREAKDOWN=128`, and `NEWTON_DIVERGENCE=256`.
```

Proposed replacement for the matrix-free inventory

```text
The matrix-free solvers emit `SUCCESS=0`,
`MAX_NEWTON_ITERATIONS_EXCEEDED=2`,
`MAX_LINEAR_ITERATIONS_EXCEEDED=4`,
`BICGSTAB_BREAKDOWN=128`, and `NEWTON_DIVERGENCE=256`.
```

Evidence

- `src/cubie/result_codes.py:22-58`
- `src/cubie/integrators/matrix_free_solvers/bicgstab_solver.py:659-670`
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py:498-510`
- `tests/integrators/matrix_free_solvers/test_linear_solver.py:331-366`
- `tests/integrators/matrix_free_solvers/conftest.py:73-105`

`UFR-016` records the public result vocabulary. D2-06 records the internal
inventories.

## D2-07 State the absolute summary-ratio tolerance

Current location

`src/cubie/integrators/loops/AGENTS.md:96-100`

Current text

```text
- `samples_per_summary` (on `ODELoopConfig`) raises `ValueError` if `summarise_every` is
  not within 1% of an integer multiple of `sample_summaries_every`; a ≤1% deviation is
  accepted with a `UserWarning` and `summarise_every` adjusted. Evaluated on every
  `build()`.
```

Proposed replacement

```text
`samples_per_summary` rounds the ratio
`summarise_every / sample_summaries_every` to the nearest integer. It
accepts and warns when the absolute difference from that integer is at
most `0.01`. It raises `ValueError` for a larger difference.
```

Evidence

- `src/cubie/integrators/loops/ode_loop_config.py:245-295`
- `tests/integrators/loops/test_ode_loop_config.py:182-212`

## D2-08 Remove the nonexistent correction-tolerance warning

Current location

`src/cubie/integrators/matrix_free_solvers/AGENTS.md:113`

Current text

```text
- Correction-norm `rtol` floors at 4 ULPs with a warning; Krylov norms keep raw values.
```

Proposed replacement

```text
Correction-norm `rtol` silently floors nonzero components at 4 ULPs.
Krylov norms keep their requested values.
```

Evidence

- `src/cubie/integrators/norms.py:24-33`
- `tests/integrators/test_norms.py:632-647`
- `tests/integrators/test_norms.py:650-673`

## D2-09 Correct the compiled loop termination and return description

Current locations

- `src/cubie/integrators/loops/ode_loop.py:501-505`
- `src/cubie/integrators/loops/ode_loop.py:548-551`

Current text

```text
The loop terminates when every output schedule passes its stop time, or
when the maximum number of iterations is reached.

Status code aggregating errors and iteration counts.
```

Proposed replacement

```text
The loop terminates when every output schedule passes its stop time or
an irrecoverable status is set.

Status bit field containing loop, step, controller, and solver failure
flags. Iteration counts are written to `iteration_counters_output`.
```

Evidence

- `src/cubie/integrators/loops/ode_loop.py:691-736`
- `src/cubie/integrators/loops/ode_loop.py:816-894`
- `src/cubie/result_codes.py:36-47`
- `tests/integrated_numerical_tests/test_rejection_liveness.py:19-37`
- `tests/integrated_numerical_tests/test_status_staining.py:1-121`

`MAX_LOOP_ITERS_EXCEEDED` is reserved and is not emitted by the loop.

## D2-10 Remove two nonexistent solver features

Current locations

- `src/cubie/integrators/matrix_free_solvers/linear_solver.py:47-53`
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py:20-22`

Current text

```text
Line-search strategy ('steepest_descent' or 'minimal_residual').

CUDAFactory subclass that compiles a damped Newton--Krylov solver.
```

Proposed replacements

```text
Linear correction strategy. Accepted values are
`"steepest_descent"` and `"minimal_residual"`.

CUDAFactory subclass that compiles a Newton-Krylov solver.
```

Evidence

- `src/cubie/integrators/matrix_free_solvers/linear_solver.py:232-365`
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py:293-515`
- `src/cubie/integrators/matrix_free_solvers/AGENTS.md:102-116`

`INT-04` records the stale public overview. D2-10 records the two
source-docstring locations.

## D2-11 Refresh the internal test map

Current locations

- `src/cubie/integrators/loops/AGENTS.md:105-108`
- `src/cubie/integrators/matrix_free_solvers/AGENTS.md:124-131`
- `src/cubie/integrators/step_control/AGENTS.md:77-80`

Current text

```text
### Testing
Tests in `tests/integrators/loops/`. Loop correctness is also exercised end-to-end in
`tests/batchsolving/` and against `tests/integrators/cpu_reference.py`
(`run_reference_loop()`).
```

```text
### Testing
Solver behaviour is exercised through the implicit algorithm steps under
`tests/integrators/algorithms/`, verified against the plain CPU reference
solvers in `tests/integrators/cpu_reference/cpu_utils.py`
(`newton_solve`, `krylov_solve`). Any change to a device function's
algorithm, signature, buffers, or status logic must be replicated in its
CPU reference counterpart. Run e.g.
`pytest tests/integrators/algorithms -k "newton or krylov or implicit"`.
```

```text
### Testing
Tests under `tests/integrators/step_control/` (`test_controllers.py`,
`test_adaptive_step_controller.py`, `test_fixed_step_controller.py`); CPU reference in
`tests/integrators/cpu_reference/step_controllers.py`.
```

Proposed replacement for the loop test map

```text
Tests live in `tests/integrators/loops/`. `tests/integrated_numerical_tests/`
and `tests/batchsolving/` exercise loop behavior.
`tests/integrators/cpu_reference/loops.py` defines `run_reference_loop()`.
```

Proposed replacement for the matrix-free test map

```text
Direct solver tests live in
`tests/integrators/matrix_free_solvers/`. Implicit algorithm tests live in
`tests/integrators/algorithms/`. The CPU reference implementations are in
`tests/integrators/cpu_reference/cpu_utils.py`.
```

Proposed replacement for the step-control test map

```text
Tests live in `tests/integrators/step_control/`. The CPU reference is
`tests/integrators/cpu_reference/step_controllers.py`.
```

Evidence

- `tests/integrators/cpu_reference/loops.py:47-142`
- `tests/integrators/matrix_free_solvers/conftest.py:1-836`
- `tests/integrators/matrix_free_solvers/test_base_solver.py:1-167`
- `tests/integrators/matrix_free_solvers/test_bicgstab.py:1-199`
- `tests/integrators/matrix_free_solvers/test_linear_solver.py:1-807`
- `tests/integrators/matrix_free_solvers/test_newton_krylov.py:1-494`
- `tests/integrators/step_control/test_gain_specs.py:1-133`
- `tests/integrators/step_control/test_gustafsson_controller.py:1-89`
- `tests/integrators/step_control/test_init.py:1-92`

## D2-12 Remove AI-authorship commentary from test docstrings

Current location

`tests/integrators/cpu_reference/__init__.py:1-6`

Current text

```text
Reference CPU implementations used across integrator tests.

I've let genAI agents run fairly free on this module, adding many of the
over-engineered and pointless checks and complicated chains that it loves to
add, as all we really want in here is a reference implementation of the
GPU integrator components.
```

Proposed replacement

```text
Reference CPU implementations used across integrator tests.
```

Current location

`tests/integrators/cpu_reference/cpu_utils.py:626-628`

Current text

```text
Return Horner evaluations and range flag for ``coefficients``.

Busybody AI over-checking left in place.
```

Proposed replacement

```text
Return Horner evaluations and range flag for ``coefficients``.
```

## Deduplication receipt

The following findings were not repeated as supplemental proposals.

- Public algorithm registry omissions are covered by `INT-10` and `UFR-009`.
- Public result-code omissions are covered by `UFR-016`.
- Missing public integrator API pages and memory-heuristic coverage are
  covered by `INT-15`.
- The public damped-Newton claim is covered by `INT-04`.
- Rendered source punctuation is covered by `STYLE-DOC-01` through
  `STYLE-DOC-03`.
