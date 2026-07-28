# Cubie Test Structure Guide

This document defines the canonical structure for all test files in the Cubie
test suite. It serves as both a reference for human developers and an
instruction set for AI agents performing test refactoring.

---

## Core Principles

### 1. Session-scoped fixtures, minimal setup/teardown

Cubie's architecture makes object construction expensive (each device function
call triggers a CUDA compilation/build, ~2 minutes per build in real CUDA).
Execution after build is fast. Therefore:

- **All reusable fixtures are session-scoped.**
- Session-scoped fixtures are torn down and rebuilt for each unique parameter
  set. Tests that need similar settings MUST share the same parameter set.
- The only exception is **mutable fixtures** (see §5).

### 2. Single source of truth: `solver_settings`

The `solver_settings` dictionary in `tests/conftest.py` is the central
configuration object. Every fixture in the hierarchy derives its configuration
from `solver_settings`. This means:

- Changing a parameter set tears down and rebuilds the **entire fixture tree**
  for that session.
- Tests parametrize by overriding keys in `solver_settings` via
  `solver_settings_override` (indirect parametrization).
- There is **one** override fixture: `solver_settings_override`. Tests that
  need the cross product of two parameter axes (e.g. solver type ×
  preconditioner order) build the merged dicts in a single parametrize
  list rather than stacking a second override fixture.

### 3. No mocks, no dummies

Unless a convincing argument is made that using a real object would
significantly increase test duration, all tests use full-complexity, real
library objects. Mocks and dummy objects are not permitted by default.

**The burden of proof is on the person proposing a mock.** They must
demonstrate that the real object adds meaningful build/setup time that cannot
be avoided by sharing a session-scoped fixture.

### 4. No inline object construction

Tests must not construct solver components (Solver, SingleIntegratorRun,
BatchSolverKernel, SymbolicODE systems, etc.) inline within test functions.
All object creation happens through fixtures. If a test needs an object that
doesn't exist as a fixture, add it to `conftest.py`.

### 5. Mutable fixtures are rare and deliberate

Some tests must modify fixture state. For these:

- Create a **function-scoped** copy of the session fixture, named
  `<fixture>_mutable` (e.g., `solver_mutable`, `loop_mutable`).
- The mutable fixture rebuilds the object each invocation — use sparingly.
- **Combine multiple mutation tests into single test functions** to minimise
  the number of rebuilds.
- Mutable fixtures live in `conftest.py` alongside their session-scoped
  counterparts.

---

## Fixture Architecture

### Fixture location rules

| Scope | Location |
|-------|----------|
| Used by multiple modules or part of the fixture hierarchy | `tests/conftest.py` (root) |
| Used only within one subdirectory AND not part of hierarchy | Subdirectory `conftest.py` (rare, requires justification) |
| Inside a test file | **Never.** Move to conftest. |

**Default assumption:** fixtures belong in the root `tests/conftest.py`.
Module-level conftest files and test-file-local fixtures are almost certainly
mistakes left by previous refactoring. Consolidate them upward.

#### Exception: library singletons (`summary_metrics`, `memory_manager`)

These modules carry state between sessions at the library level. Their
session-scoped fixtures from the root conftest may contain stale state from
a previous parameter set's session. For these modules:

- Define **function-scoped** fixtures in a **module-level conftest** (e.g.,
  `tests/memory/conftest.py`, `tests/outputhandling/summarymetrics/conftest.py`).
- This ensures each test gets a clean singleton instance, uncontaminated by
  prior test sessions.
- This is one of the very few legitimate uses of module-level conftest files.

#### Exception: sub-numerical unit fixtures

Three further module-level conftests are sanctioned because they pin
protocols the shared hierarchy cannot express:

- `tests/odesystems/symbolic/conftest.py` — SymPy symbols, `IndexedBases`,
  and `ParsedEquations` fixtures for parser/codegen unit tests. These run
  no device code; routing them through the `system` fixture would test the
  wrong layer.
- `tests/integrators/matrix_free_solvers/conftest.py` — bespoke
  conditioning-controlled 3-state systems with algebraic reference
  solutions (`mr_expected` via dense operator probing, `nk_expected` via
  fixed-point iteration) for testing solver device functions in isolation.
  Its `linear_solver_instance` / `newton_solver_instance` fixtures draw
  their configuration from the central `solver_settings`, so solver
  parametrization still flows through `solver_settings_override`.
- `tests/integrated_numerical_tests/julia_reference/conftest.py` — the
  golden-reference gate against vendored DifferentialEquations.jl
  sweeps. Solvers and the Lorenz system flow through the shared
  `solver_settings` hierarchy (one `solver_settings_override` param
  set per algorithm); this conftest only loads the vendored data and
  runs the per-algorithm dt/tolerance sweeps required by the
  numerical-equivalence protocol shared with GPUODEBenchmarks.

### Fixture hierarchy

The fixture tree flows from settings → objects → computed outputs:

```
solver_settings_override (parametrize this)
        │
        ▼
  solver_settings ──────────────────────────────┐
        │                                       │
        ├── precision                           │
        ├── tolerance                           │
        ├── system (SymbolicODE)                │
        │                                       │
        ├── algorithm_settings ─┐               │
        ├── loop_settings ──────┤               │
        ├── step_controller_settings ───┤       │
        ├── output_settings ────┤               │
        ├── memory_settings ────┘               │
        │                                       │
        ├── driver_settings → driver_array      │
        │                                       │
        ├── single_integrator_run ──────────────┤
        │       ├── loop (._loop)               │
        │       └── step_object (._algo_step)   │
        │                                       │
        ├── solverkernel                        │
        ├── solver                              │
        │                                       │
        ├── output_functions                    │
        ├── initial_state                       │
        │                                       │
        ├── cpu_system ─────────────────────────┤
        ├── cpu_driver_evaluator                │
        ├── cpu_step_controller                 │
        │                                       │
        ▼                                       ▼
  cpu_loop_outputs              device_loop_outputs
```

**Key rule:** Lower-level fixtures (e.g., `loop`, `step_object`) are
extracted as attributes from higher-level fixtures (e.g.,
`single_integrator_run`), not constructed independently. This avoids
duplicate builds.

### What triggers a build (~2 min each in real CUDA)

Any of the following triggers CUDA compilation:

- Calling `solver.solve()` or `solver.run()`
- Calling any `.device_function()` on a compiled object
- Calling `.update()` on a compile setting (triggers rebuild before the
  next solve)

Note: merely constructing a Cubie object (e.g., `Solver(...)`) does **not**
trigger a build. The build happens lazily on the first solve/run/device call.

In CUDASIM mode (remote CI), builds are instant. But fixture design must
optimise for real CUDA, where each unique parameter set costs ~2 min.

---

## Standard Parameter Sets

Two tiers of run configuration are defined in `tests/_utils.py`:

| Name | Definition | Use case |
|------|-----------|----------|
| **SHORT_RUN** | Default `solver_settings` (no override) — duration=0.2, dt=0.01 | Quick smoke tests, attribute checks |
| **MID_RUN** | `MID_RUN_PARAMS` — dt=0.001, finer save intervals | Numerical behaviour / drift over an integration duration |

**SHORT_RUN is the default.** Use it for everything that does not require
running a solve: property forwarding, attribute checks, construction,
configuration, update routing, compatibility warnings, etc.

**ONLY escalate to MID_RUN_PARAMS** when verifying numerical
behaviour or drift over an integration duration (e.g. comparing device
output against a CPU reference).

Tests that can share a parameter set **must** share it. Each unique parameter
set causes a full teardown/rebuild cycle of the entire fixture tree.

To combine a parameter set with algorithm or system overrides, use
`merge_dicts` or `merge_param`:

```python
from tests._utils import MID_RUN_PARAMS, merge_dicts, merge_param

@pytest.mark.parametrize(
    "solver_settings_override",
    [merge_dicts(MID_RUN_PARAMS, {"algorithm": "rk4", "step_controller": "pid"})],
    indirect=True,
)
def test_my_feature(single_integrator_run):
    ...
```

---

## Parametrization Pattern

### The canonical pattern

```python
"""Tests for cubie.<module>.<component>."""

import pytest
from numpy.testing import assert_allclose
from tests._utils import MID_RUN_PARAMS, merge_dicts


@pytest.mark.parametrize(
    "solver_settings_override",
    [merge_dicts(MID_RUN_PARAMS, {"algorithm": "rk4"})],
    indirect=True,
)
def test_step_object_exists(single_integrator_run):
    assert single_integrator_run._algo_step is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [merge_dicts(MID_RUN_PARAMS, {"algorithm": "rk4"})],
    indirect=True,
)
def test_output_shape(device_loop_outputs, solver_settings):
    # Uses the same parameter set as above — no rebuild
    assert device_loop_outputs["state"].shape[0] > 0
```

The test file name identifies the component under test. The test function
name describes the specific behaviour being verified. Test classes are
prohibited (see Anti-Patterns §7).

### Multiple parameter sets (triggers rebuild per set)

```python
@pytest.mark.parametrize(
    "solver_settings_override",
    [
        pytest.param(
            merge_dicts(MID_RUN_PARAMS, {"algorithm": "euler"}),
            id="euler",
        ),
        pytest.param(
            merge_dicts(MID_RUN_PARAMS, {"algorithm": "rk4"}),
            id="rk4",
        ),
    ],
    indirect=True,
)
def test_algorithm_convergence(device_loop_outputs, cpu_loop_outputs, tolerance):
    ...
```

### Tests that don't need solver infrastructure

Lightweight config/utility classes (e.g., `OutputConfig`, `ScaledNormConfig`)
that can be instantiated without triggering a build may be tested with
simpler fixtures. However:

- They **still parametrize through `solver_settings`** so their settings
  remain consistent with the rest of the hierarchy.
- Their fixtures still live in `tests/conftest.py`.
- Integration-level tests of these components (testing interactions with
  built objects) must use the standard fixture hierarchy.

### Any fixture in the hierarchy is available to any test file

Test files are not restricted to a single fixture. A test for
`SingleIntegratorRunCore` may use the `solver` fixture to verify
behaviour that only manifests at the solver level (e.g. warnings
emitted during `solver.solve()`). The fixture hierarchy exists to
share expensive builds, not to limit what a test can request.

### Implicit coverage

A functionality item is covered if an existing test **will fail when
that functionality is removed or broken**, even if the test does not
name the functionality directly. For example:

- `instantiate_loop` forwarding `n_states` to the loop is implicitly
  covered by any test that asserts `loop.compile_settings.n_states ==
  system.sizes.states` after construction.
- `build` assembling compiled functions is implicitly covered by
  `test_device_function_callable` — if build fails to wire functions,
  the callable will not work.

Implicit coverage counts. Do not write redundant tests just to name
every internal step, but **do** verify that the observable outcome of
each internal step is checked by at least one test.

---

## Test File Organisation

### One file per component, strictly

Each source module gets exactly one test file. If related tests are split
across multiple files, consolidate them. If a test file covers multiple
unrelated components, split it.

```
cubie/batchsolving/solver.py       → tests/batchsolving/test_solver.py
cubie/integrators/loops/ode_loop.py → tests/integrators/loops/test_ode_loop.py
cubie/memory/mem_manager.py        → tests/memory/test_memmgmt.py
```

### File structure template

```python
"""Tests for cubie.<module>.<component>."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests._utils import MID_RUN_PARAMS, merge_dicts  # as needed


# ── Tests using MID_RUN settings ────────────────────────────────────────── #

@pytest.mark.parametrize(
    "solver_settings_override",
    [MID_RUN_PARAMS],
    indirect=True,
)
def test_attribute_exists(<fixture>):
    assert <fixture>.some_attr is not None


@pytest.mark.parametrize(
    "solver_settings_override",
    [MID_RUN_PARAMS],
    indirect=True,
)
def test_computed_value(<fixture>, tolerance):
    result = <fixture>.compute()
    assert_allclose(result, expected, atol=tolerance.abs_loose)


@pytest.mark.parametrize(
    "solver_settings_override",
    [MID_RUN_PARAMS],
    indirect=True,
)
def test_numerical_output(device_loop_outputs, cpu_loop_outputs, tolerance):
    """Compare device results against CPU reference."""
    assert_allclose(
        device_loop_outputs["state"],
        cpu_loop_outputs["state"],
        atol=tolerance.abs_loose,
        rtol=tolerance.rel_loose,
    )


# ── Mutation tests (use sparingly) ──────────────────────────────────────── #

def test_multiple_mutations(<fixture>_mutable):
    """Combine mutation operations to minimise rebuilds."""
    <fixture>_mutable.setting = new_value_1
    assert <fixture>_mutable.derived_property == expected_1

    <fixture>_mutable.setting = new_value_2
    assert <fixture>_mutable.derived_property == expected_2


# ── Tests using default settings (SHORT_RUN) ────────────────────────────── #

def test_default_behaviour(<fixture>):
    """No solver_settings_override → uses SHORT_RUN defaults."""
    assert <fixture>.is_valid()
```

---

## Anti-Patterns to Eliminate

These are patterns left by previous AI agents and must be refactored:

### 1. Inline imports inside test functions
```python
# BAD — import inside function body
def test_something(fixture):
    from cubie.some_module import SomeClass
    assert isinstance(fixture, SomeClass)

# GOOD — all imports at the top of the file
from cubie.some_module import SomeClass

def test_something(fixture):
    assert isinstance(fixture, SomeClass)
```

All imports must be at the top of the file. No `from` or `import`
statements inside test functions.

### 2. Inline object construction
```python
# BAD — constructs a Solver inside the test
def test_something():
    system = build_three_state_nonlinear_system(np.float32)
    solver = Solver(system, ...)
    solver.solve()
    assert solver.result is not None

# GOOD — uses fixture hierarchy
def test_something(solver, ...):
    assert solver.result is not None
```

### 3. Fixtures in test files
```python
# BAD — fixture defined in test_solver.py
@pytest.fixture(scope="session")
def solved_solver_simple(solver, ...):
    ...

# GOOD — fixture lives in tests/conftest.py
```

### 4. Module conftest files duplicating root fixtures
```python
# BAD — tests/memory/conftest.py redefines memory-related fixtures
# that should be in root conftest

# GOOD — root conftest.py contains the fixture, module conftest is
# empty or deleted
```

### 5. Standalone test files ignoring the fixture hierarchy
```python
# BAD — test_norms.py constructs everything from scratch
def test_norm_computation():
    config = ScaledNormConfig(n=3, ...)
    norm = ScaledNorm(config)
    ...

# GOOD — uses a norm fixture derived from solver_settings
def test_norm_computation(scaled_norm, ...):
    ...
```

### 6. Independent parametrization bypassing solver_settings
```python
# BAD — parametrizes precision independently
@pytest.mark.parametrize("prec", [np.float32, np.float64])
def test_something(prec):
    ...

# GOOD — parametrizes through solver_settings_override
@pytest.mark.parametrize(
    "solver_settings_override",
    [{"precision": np.float32}, {"precision": np.float64}],
    indirect=True,
)
def test_something(precision, ...):
    ...
```

### 7. Test classes
```python
# BAD — test class grouping
class TestSolverBehaviour:
    def test_something(self, solver):
        ...
    def test_something_else(self, solver):
        ...

# GOOD — bare test functions; the file name identifies the component
def test_something(solver):
    ...

def test_something_else(solver):
    ...
```

Test classes add indirection with no benefit. The test file already
identifies the component under test; the function name describes the
specific behaviour. Flatten all test classes into module-level functions.

---

## Agent Instructions for Test Refactoring

When refactoring a test file, follow this checklist:

1. **`Read tests/conftest.py`** to understand default (non-parametrized) settings in solver_settings and see available fixtures.
2. **Read the existing file and its module conftest** (if any).
2. **Identify all locally defined fixtures.** Move them to
   `tests/conftest.py`. If they duplicate existing fixtures, delete them.
3. **Identify all inline object construction.** Replace with fixture
   requests. Add new fixtures to `tests/conftest.py` if needed.
4. **Identify all independent parametrization.** Convert to
   `solver_settings_override` indirect parametrization.
5. **Identify all mocks/dummies.** Replace with real objects from fixtures.
   Only keep a mock if you can demonstrate it saves significant build time
   AND the real object cannot be shared via session-scoped fixtures.
6. **Flatten all test classes** into bare module-level test functions.
   Apply `@pytest.mark.parametrize` directly to each function.
7. **Consolidate mutation tests.** Multiple small mutation tests become
   fewer, larger test functions using `<fixture>_mutable`.
8. **Verify the file maps 1:1 to a source module.** Merge or split as
   needed.
9. **Remove module conftest files** if all their fixtures have been moved to
   the root conftest. Keep a module conftest only if it contains fixtures
   genuinely scoped to that subdirectory alone (this is rare).
10. **Run the tests** to verify nothing is broken. .

### Skepticism protocol

Regard any of the following with extreme skepticism — they are almost
certainly mistakes from a previous AI agent:

- Custom per-module conftest fixtures
- Test-file-local fixtures
- Mock objects or dummy classes
- Tests that construct solver/system/kernel objects inline
- Standalone test files that don't participate in the fixture hierarchy
- Multiple test files for a single source module
- Function-scoped fixtures that aren't named `*_mutable`
- Test classes (should be bare functions)

**Default assumption:** these are wrong and must be brought into conformance
with this guide. The previous agent took shortcuts as its context window
filled. Do not preserve these patterns out of deference to existing code.

---

## Fixture Composition: Adding New Fixtures

When a component needs testing and no fixture exists:

1. **Check if it's already accessible** as an attribute of an existing
   fixture (e.g., `single_integrator_run._loop`).
2. If yes, add a thin fixture that extracts it:
   ```python
   @pytest.fixture(scope="session")
   def loop(single_integrator_run):
       return single_integrator_run._loop
   ```
3. If no, determine where in the hierarchy it belongs and construct it from
   `solver_settings` sub-dicts:
   ```python
   @pytest.fixture(scope="session")
   def new_component(algorithm_settings, system, ...):
       return NewComponent(system=system, **algorithm_settings)
   ```
4. Add a `*_mutable` variant only if mutation tests are required.
5. Place the fixture in `tests/conftest.py`.

---

## Quick Reference

| Rule | Summary |
|------|---------|
| Fixture scope | Session by default, function only for `*_mutable` |
| Fixture location | `tests/conftest.py` (root) unless strong justification |
| Parametrization | Always via `solver_settings_override`, indirect |
| Parameter sets | SHORT_RUN (defaults), MID_RUN_PARAMS |
| Object construction | Fixtures only, never inline |
| Mocks | Not permitted unless proven necessary |
| File mapping | 1:1 with source modules |
| Mutation tests | Combine into single functions, use `*_mutable` fixtures |
| Build cost | ~2 min per unique parameter set (real CUDA) |
| Test classes | Prohibited — use bare functions |
| Module conftest | Almost always a mistake — consolidate to root |
| Singleton modules | Exception: function-scoped fixtures in module conftest |

---

## CUDA Availability Check

Before running tests, agents must check whether a real CUDA device is
available and only enable simulation mode when necessary:

```bash
# Check CUDA — prints CUDA_AVAILABLE=1 or CUDA_AVAILABLE=0
"./.venv/Scripts/python.exe" .claude/check_cuda.py

# If CUDA_AVAILABLE=0, enable simulation mode:
$env:NUMBA_ENABLE_CUDASIM="1"    # PowerShell
export NUMBA_ENABLE_CUDASIM="1"  # Bash

# If CUDA_AVAILABLE=1, do NOT set NUMBA_ENABLE_CUDASIM.
# Running under CUDASIM when real CUDA is available skips GPU
# compilation and may mask device-specific bugs.
```

Tests marked `nocudasim` require real CUDA and must be excluded when
running in simulation mode (`-m "not nocudasim and not cupy"`).
