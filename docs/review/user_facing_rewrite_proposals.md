# User-facing documentation rewrite proposals

## Scope and method

This ledger covers 42 files and 6,973 lines. It includes the repository
README, the Sphinx narrative pages and examples assigned to this lane, the
Fleet runbook, the CUDA-writing guide, the test guide, the Julia-reference
data README, and the changelog. Every line was read. Factual claims were
checked against the current source, tests, package metadata, workflow files,
and infrastructure code. The changelog was checked only for release-history
contradictions because project policy forbids editing it.

Every line was reread in a separate semantic pass for framing, qualification,
hedging, emphasis, intensification, false drama, reader-role assignment, and
high-bar colon or semicolon use. Syntax punctuation and changelog history were
retained.

The proposals do not edit any source document. Locations refer to the current
working tree on 2026-08-06.

## Factual and completeness rewrites

### UFR-001: Replace the README implementation summary

**Location:** `readme.md:11-22`

**Proposed text:**

> CuBIE JIT-compiles CUDA kernels for batches of independent ODE and DAE
> initial-value problems. It provides `solve_ivp` and `Solver` interfaces for
> parameter and initial-condition sweeps. The default compiler backend is
> numba-cuda-mlir. The deprecated numba-cuda backend supports Python 3.10 and
> the CUDA simulator.
>
> CuBIE generates system-specific right-hand-side, Jacobian-vector product,
> residual, and preconditioner functions. Implicit algorithms use the generated
> operators in matrix-free linear and nonlinear solves.

**Reason:** The current paragraph names numba-cuda as the sole backend and
mixes a machine-specific speed claim into the product definition. Backend
resolution prefers MLIR.

**Evidence:** `src/cubie/cuda_backend.py:1-17`,
`src/cubie/cuda_backend.py:42-62`, `pyproject.toml:45-67`.

### UFR-002: Replace the README installation option count

**Location:** `readme.md:49-63`

**Proposed text:**

> Select one CUDA backend extra. `mlir-cuda12` and `mlir-cuda13` install the
> default MLIR backend with toolkit and CuPy wheels. `cuda12` and `cuda13`
> install the deprecated numba-cuda backend with toolkit and CuPy wheels. Use
> `mlir` or `cuda` when a compatible system CUDA toolkit is already installed.
> The MLIR backend requires Python 3.11 or later. Python 3.10 and the CUDA
> simulator require numba-cuda.

**Reason:** Six extras exist, not four. The bare system-toolkit extras are
missing from the page.

**Evidence:** `pyproject.toml:45-67`, `src/cubie/cuda_backend.py:45-50`.

### UFR-003: Make the README benchmark reproducible or remove it

**Location:** `readme.md:11-13`, `readme.md:89-91`, `readme.md:125`

**Proposed text:**

> The example constructs 1,048,576 combinations of initial values and
> parameters. Compilation and execution time depend on the selected backend,
> algorithm, output configuration, GPU, and cache state.

Remove the `~10000x` headline and footnote unless the repository also records
the benchmark command, CuBIE revision, backend and package versions, CPU
parallelism, warm-up policy, cache state, output equivalence, tolerances, and
raw results.

**Reason:** The reported timings cannot be reproduced from the repository and
are presented as a general property of CuBIE.

### UFR-004: Synchronize the Sphinx release metadata

**Location:** `docs/source/conf.py:9-12`

**Proposed code:**

```python
from importlib.metadata import version

project = "cubie"
copyright = "2025-2026, Chris Cameron"
author = "Chris Cameron"
release = version("cubie")
```

**Reason:** Sphinx reports 0.0.4 while package metadata and the changelog report
0.4.0.

**Evidence:** `pyproject.toml:13-21`, `CHANGELOG.md:3`.

### UFR-005: Correct the documentation landing page

**Location:** `docs/source/index.rst:4-8`,
`docs/source/user_guide/index.rst:4-6`

**Proposed text:**

> CuBIE integrates batches of initial-value problems on NVIDIA GPUs. Source
> code is at `github.com/cubiepy/cubie
> <https://github.com/cubiepy/cubie>`__. :doc:`getting_started` documents
> installation and the first solve. The :doc:`User Guide <user_guide/index>`
> documents workflows. The :doc:`API Reference <API_reference/index>`
> documents functions and classes.

Use the source-code sentence and repository URL in the user-guide index.
Retain its existing toctree.

**Reason:** Both indexes point to the former repository owner. The landing
page's Reference Manual link also points back to the user guide.

**Evidence:** `git remote -v` reports `cubiepy/cubie`; the toctree at
`docs/source/index.rst:14-19` contains the separate API reference.

### UFR-006: Correct the one-call compilation description

**Location:** `docs/source/getting_started.rst:80-84`,
`docs/source/user_guide/coming_from_scipy.rst:122-127`

**Proposed text:**

> `solve_ivp` constructs a `Solver` for each call. Reuse a `Solver` to retain
> the in-process compiled objects and managed allocations between solves. The
> disk cache reuses a compiled kernel across separate solver instances and
> Python sessions when the cache identity matches.

**Reason:** `solve_ivp` reconstructs the object graph, but a compatible disk
cache can avoid recompilation. “Builds and compiles on every call” is false.

**Evidence:** `src/cubie/batchsolving/solver.py:215-244`,
`src/cubie/cubie_cache.py:680-748`.

### UFR-007: Replace the cache page

**Location:** `docs/source/user_guide/caching.rst:4-79`

**Proposed structure and text:**

> CuBIE has three on-disk cache layers under one cache root. They store
> generated source, CellML parse results, and compiled kernels. The default
> root is `<cwd>/generated`. Set `CUBIE_CACHE_DIR` before importing CuBIE or
> call `set_cache_root` to change the shared root.
>
> `Solver(cache=True)` enables compiled-kernel persistence. `cache=False`
> disables compiled-kernel persistence for that solver. It does not disable
> generated-source or CellML caches. `cache="flush_on_change"` selects the
> flush-on-change policy. Any other string or `Path` selects a compiled-kernel
> cache directory.
>
> The loose policy keywords are `cache_enabled`, `cache_mode`,
> `max_cache_entries`, and `cache_dir`. `cache_mode` accepts `"hash"` and
> `"flush_on_change"`.
>
> Package source changes are part of the compiled-kernel index key. Compatible
> cache entries are invalidated when CuBIE package source changes.

**Reason:** The current page says there are two layers, says `cache=False`
disables code generation and forces recompilation, omits the CellML layer and
four policy keywords, and says package-code changes are not keyed. Each claim
contradicts the cache implementation.

**Evidence:** `src/cubie/cache_root.py:34-50`,
`src/cubie/cache_root.py:67-75`, `src/cubie/cubie_cache.py:135-145`,
`src/cubie/cubie_cache.py:680-719`, `src/cubie/cubie_cache.py:740-747`,
`src/cubie/cubie_cache.py:548-565`.

### UFR-008: Correct CellML version and DAE support

**Location:** `docs/source/user_guide/cellml.rst:52-61`

**Proposed text:**

> CuBIE accepts valid CellML 1.0 and 1.1 files. CellML 2.0 is not supported.
> The loader preserves differential equations and algebraic constraints.
> DAE-shaped symbolic input is routed through structural simplification before
> code generation.

**Reason:** The current caveat says only ODE-based models are supported and
hedges about CellML 2.0. The loader has an explicit 1.0/1.1 contract and the
parser routes algebraic constraints through structural simplification.

**Evidence:** `src/cubie/odesystems/symbolic/parsing/cellml.py:195-204`,
`src/cubie/odesystems/symbolic/parsing/parser.py:651-658`.

### UFR-009: Add omitted algorithms and fix adaptive DIRK claims

**Location:** `docs/source/user_guide/choosing_algorithms.rst:108-150`

**Proposed additions:**

| Name | Order | Adaptive | Notes |
|---|---:|:---:|---|
| `kvaerno3` | 3 | Yes | Four-stage ESDIRK with an embedded estimate. |
| `kvaerno5` | 5 | Yes | Seven-stage ESDIRK with an embedded estimate. |
| `firk_gauss_legendre_4` | 8 | No | Four-stage Gauss-Legendre FIRK. |

Replace “the only adaptive DIRK tableau” for `l_stable_sdirk_4` with
“Five-stage adaptive SDIRK.”

**Evidence:** `src/cubie/integrators/algorithms/generic_dirk_tableaus.py:156-193`,
`src/cubie/integrators/algorithms/generic_dirk_tableaus.py:203-285`,
`src/cubie/integrators/algorithms/generic_dirk_tableaus.py:429-438`,
`src/cubie/integrators/algorithms/generic_firk_tableaus.py:157-196`,
`src/cubie/integrators/algorithms/generic_firk_tableaus.py:219-224`.

### UFR-010: Correct controller defaults

**Location:** `docs/source/user_guide/choosing_algorithms.rst:225-233`,
`docs/source/user_guide/configuration.rst:194-201`

**Proposed text:**

> Each algorithm supplies its own controller defaults. Adaptive DIRK tableaus
> use the PI controller. Algorithms without an error estimate use fixed-step
> control. Algorithm registrations define the controller defaults for every
> family.

**Reason:** The pages state that implicit families use Gustafsson. Adaptive
DIRK explicitly defaults to PI.

**Evidence:** `src/cubie/integrators/algorithms/generic_dirk.py:81-93`.

### UFR-011: Correct MATLAB and SciPy method mappings

**Location:** `docs/source/user_guide/coming_from_matlab.rst:63-72`,
`docs/source/user_guide/coming_from_scipy.rst:82-92`

**Proposed rows:**

| Source method | CuBIE method |
|---|---|
| MATLAB `ode23s` | `"ode23s"` |
| MATLAB `ode15s` | No direct equivalent. |
| SciPy `Radau` | `"radau"` |
| SciPy `BDF`, `LSODA` | No direct equivalent. |

**Evidence:**
`src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py:416-424`,
`src/cubie/integrators/algorithms/generic_firk_tableaus.py:219-224`.

### UFR-012: Complete the configuration index

**Location:** `docs/source/user_guide/configuration.rst:4-9`,
`docs/source/user_guide/configuration.rst:63-95`,
`docs/source/user_guide/configuration.rst:175-184`

**Proposed changes:**

- Replace “six underlying settings groups” with “seven loose-keyword groups.”
- Add `auto_memory` with default `True` to `Solver.__init__`.
- Add cache policy keys `cache_enabled`, `cache_mode`,
  `max_cache_entries`, and `cache_dir` to the Cache group.
- State that only `"flush_on_change"` is a cache-mode spelling. Other strings
  select a path.
- List every `ALL_KERNEL_PARAMETERS` entry in the Kernel group rather than only
  `max_registers`.

**Reason:** `Solver` routes output, memory, step-control, algorithm, loop,
cache, and kernel keyword sets. The explicit `auto_memory` argument and most
cache policy settings are absent.

**Evidence:** `src/cubie/batchsolving/solver.py:420-433`,
`src/cubie/batchsolving/solver.py:459-513`,
`src/cubie/cubie_cache.py:135-145`,
`src/cubie/cubie_cache.py:740-747`.

### UFR-013: Correct sampled-driver timing keys

**Location:** `docs/source/user_guide/drivers.rst:120-127`

**Proposed text:**

> Supply sample timing with either a one-dimensional `"time"` array or a
> scalar `"driver_sample_period"`. `"t0"` is optional with
> `"driver_sample_period"` and defaults to zero. A `"time"` array must be
> strictly increasing and uniformly spaced.

**Reason:** `"dt"` is not a recognized driver key and uniform spacing is not
documented.

**Evidence:** `src/cubie/array_interpolator.py:151-156`,
`src/cubie/array_interpolator.py:365-425`.

### UFR-014: Correct memory lifetime and stream-group behavior

**Location:** `docs/source/user_guide/memory.rst:11-16`,
`docs/source/user_guide/memory.rst:40-54`

**Proposed text:**

> CuBIE allocates device memory through CuPy. Managed allocations remain
> registered with a live solver and are released when the owner closes or is
> collected. Host output backing becomes eligible for reuse after its prior
> result owner is collected.
>
> A solve that does not fit its memory budget is divided along the run axis.
> Its chunks are submitted in order on the solver's stream. Stream groups
> coordinate the memory shares and CUDA streams of solver instances. They do
> not run chunks from one solve concurrently.

**Reason:** The page says every solve allocates and frees its device arrays and
that stream groups run multiple chunks concurrently. The object lifecycle and
kernel loop contradict both statements.

**Evidence:** `src/cubie/memory/mem_manager.py:1128-1177`,
`src/cubie/batchsolving/arrays/BaseArrayManager.py:916-946`,
`src/cubie/batchsolving/BatchSolverKernel.py:731-803`.

### UFR-015: Correct output defaults and legacy timing names

**Location:** `docs/source/user_guide/results.rst:121-128`,
`docs/source/user_guide/results.rst:186-190`,
`docs/source/user_guide/optional_arguments.rst:222-239`

**Proposed text:**

> `save_every` controls time-domain snapshots. `summarise_every` controls
> summary windows. `sample_summaries_every` controls sampling within a summary
> window. `dt_save`, `dt_summarise`, and `dt_update_summaries` are rejected
> legacy names.
>
> `output_types` defaults to `["state"]`. State and observable selection lists
> default to all indices. An output is produced when its output type is active.
> Add `"observables"` to record observables.

**Reason:** The page advertises rejected legacy aliases and says observables
are saved by default.

**Evidence:** `src/cubie/batchsolving/solver.py:117-119`,
`src/cubie/batchsolving/solver.py:192-212`,
`src/cubie/outputhandling/output_functions.py:158-184`,
`src/cubie/outputhandling/output_config.py:120-151`,
`src/cubie/outputhandling/output_config.py:299-305`.

### UFR-016: Document the complete result status vocabulary

**Location:** `docs/source/user_guide/results.rst:205-230`

**Proposed addition:**

> `status_codes` is an `int32` bit field. `status_messages` maps each failed run
> index to the names of all set flags. The nonzero flags are
> `MAX_NEWTON_ITERATIONS_EXCEEDED`, `MAX_LINEAR_ITERATIONS_EXCEEDED`,
> `STEP_TOO_SMALL`, `DT_EFF_EFFECTIVELY_ZERO`, `MAX_LOOP_ITERS_EXCEEDED`,
> `STAGNATION`, `BICGSTAB_BREAKDOWN`, and `NEWTON_DIVERGENCE`.

**Evidence:** `src/cubie/result_codes.py:22-58`,
`src/cubie/result_codes.py:61-88`,
`src/cubie/batchsolving/solveresult.py:754-772`.

### UFR-017: Correct system terminology and callable support

**Location:** `docs/source/user_guide/systems.rst:4-16`,
`docs/source/user_guide/systems.rst:155-169`,
`docs/source/user_guide/systems.rst:187-207`

**Proposed text:**

> `create_ODE_system` returns a `SymbolicODE`. `BaseODE` is its abstract
> interface. There is no `GenericODE` class.
>
> `user_functions` and `user_function_derivatives` work with string and Python
> callable system definitions. CuBIE inlines non-device callables that evaluate
> on symbolic arguments. Device functions remain symbolic calls in generated
> code.
>
> Explicit ODE states require derivatives. Symbolic systems accept algebraic
> unknowns and implicit equations. Structural simplification produces a mass
> matrix when the reduced system requires one.

**Evidence:** `src/cubie/odesystems/__init__.py:18-32`,
`src/cubie/odesystems/symbolic/parsing/function_parser.py:38-75`,
`src/cubie/odesystems/symbolic/symbolicODE.py:102-119`,
`src/cubie/odesystems/symbolic/symbolicODE.py:160-170`.

### UFR-018: Replace internal user-function examples with public API

**Location:** `docs/source/user_guide/userfunctions.rst:13-106`,
`docs/source/user_guide/userfunctions.rst:116-150`

**Proposed changes:**

- Build examples with `create_ODE_system`, not the internal `parse_input` and
  code-generation functions.
- Import `cuda` with `from cubie.cuda_simsafe import cuda` so examples use the
  resolved backend.
- Remove five-value `parse_input` unpacking. The internal function currently
  returns six values.
- Keep the public statement that derivative helpers are needed by methods that
  generate Jacobian terms.

**Replacement example:**

```python
from cubie import create_ODE_system
from cubie.cuda_simsafe import cuda


@cuda.jit(device=True)
def myfunc(a, b):
    return a * b


@cuda.jit(device=True)
def myfunc_grad(a, b, index):
    if index == 0:
        return b
    if index == 1:
        return a
    return 0.0


system = create_ODE_system(
    ["dx = myfunc(x, y)", "dy = x"],
    states={"x": 1.0, "y": 0.0},
    user_functions={"myfunc": myfunc},
    user_function_derivatives={"myfunc": myfunc_grad},
)
```

**Evidence:** `src/cubie/odesystems/symbolic/parsing/parser.py:953-972`,
`src/cubie/odesystems/symbolic/symbolicODE.py:102-119`,
`src/cubie/cuda_simsafe.py:1-17`.

### UFR-019: Correct the duration default

**Location:** `docs/source/user_guide/timing.rst:245-258`

**Proposed text:**

> `duration` defaults to `1.0` and must be non-negative.

**Evidence:** `src/cubie/batchsolving/BatchSolverKernel.py:130`,
`src/cubie/_utils.py:442-458`.

### UFR-020: Correct the Newton-convergence troubleshooting instruction

**Location:** `docs/source/user_guide/troubleshooting.rst:21-35`

**Proposed first items:**

1. Inspect `result.status_messages` to distinguish Newton failure, linear
   failure, divergence, and step-size failure.
2. Reduce `dt` or `dt_max` when the nonlinear problem is too large for the
   current step.
3. Increase `newton_max_iters` when the status reports the iteration limit and
   additional iterations continue to reduce the update.

**Reason:** Tightening `atol` and `rtol` does not “give the solver more room.”
It requests stricter error control and also influences derived inner
tolerances.

**Evidence:** `src/cubie/result_codes.py:22-58`,
`src/cubie/integrators/matrix_free_solvers/newton_krylov.py:270-317`,
`src/cubie/integrators/matrix_free_solvers/newton_krylov.py:450-480`.

### UFR-021: Replace the performance page's universal claims

**Location:** `docs/source/user_guide/speed.rst:1-57`,
`docs/source/user_guide/speed.rst:85-115`

**Proposed text:**

> Measure performance with the production backend and target GPU. Batch size,
> system size, algorithm, precision, output cadence, cache state, and host
> transfers affect the result. No fixed GPU-to-CPU crossover applies.
>
> Reduce saved variables and save frequency when the output is not required.
> Use on-device summaries for aggregate-only output. Declare a value as a
> constant when it is unchanged across the batch. Changing a
> constant changes compiled-system identity.
>
> `TimeLogger` reports host phases and registered CUDA event durations. It does
> not report occupancy or memory usage.
>
> Working buffers default to local placement unless a component declares a
> different setting. Treat location overrides and `max_registers` as measured,
> hardware-specific tuning parameters.
>
> Reuse a `Solver` for repeated batches with compatible compile settings.

**Reason:** The current page asserts linear scaling, nearly free additional
runs, a universal memory bottleneck, an unsupported 30-second compile bound,
automatic buffer placement, and fixed precision advice. These are not API
guarantees.

**Evidence:** `src/cubie/time_logger.py:189-231`,
`src/cubie/time_logger.py:547-593`,
`src/cubie/integrators/loops/ode_loop_config.py:127-169`,
`src/cubie/batchsolving/solver.py:420-433`.

### UFR-022: Rewrite CUDA execution and memory claims

**Location:** `docs/source/theory/cuda.rst:11-16`,
`docs/source/theory/cuda.rst:26-43`,
`docs/source/theory/cuda.rst:45-87`

**Proposed text:**

> CuBIE launches independent trajectories across a two-dimensional CUDA block.
> One-lane algorithms assign one trajectory per lane. Tiled algorithms assign
> multiple lanes per trajectory. The achievable concurrency depends on the
> algorithm's thread and memory requirements and the target GPU.
>
> Constants are captured in generated code. Runtime parameters and state are
> stored in per-run working buffers. Matrix-free helpers avoid materializing a
> dense Jacobian per trajectory. Selected outputs are written to device output
> arrays. Summary accumulators use registered working buffers and write their
> requested results to output arrays.
>
> Buffer locations are compile settings with local and shared choices.
> `TimeLogger` reports timing events. Use CUDA profiling tools for occupancy and
> hardware counters.

**Reason:** The page promises near-perfect efficiency, a fixed 10x single-run
penalty, a break-even batch size, one thread for every algorithm, all
intermediate data on chip, default shared-memory packing, and TimeLogger
occupancy reporting. The implementation does not guarantee those properties.

**Evidence:** `src/cubie/batchsolving/BatchSolverKernel.py:728-729`,
`src/cubie/batchsolving/BatchSolverKernel.py:757-800`,
`src/cubie/integrators/loops/ode_loop_config.py:127-169`,
`src/cubie/time_logger.py:547-593`.

### UFR-023: Correct the Jacobian pipeline

**Location:** `docs/source/theory/jacobians.rst:22-41`

**Proposed text:**

> CuBIE computes derivative operators by differentiating its symbolic
> expression IR.
>
> CuBIE parses string and SymPy input through SymPy, then converts supported
> expressions to its hash-consed expression IR. Differentiation, common
> subexpression processing, assignment planning, and source emission operate on
> the IR. The generated helpers apply Jacobian-vector products without storing
> a dense Jacobian for each trajectory.

**Reason:** The page says differentiation and code-generation optimization run
on SymPy expressions. Those compute passes now run on the internal IR.

**Evidence:** `src/cubie/odesystems/symbolic/engine/__init__.py:1-9`,
`src/cubie/odesystems/symbolic/engine/from_sympy.py:71-88`,
`src/cubie/odesystems/symbolic/engine/expr.py:1076-1102`.

### UFR-024: Correct Euler error and Dormand-Prince stage count

**Location:** `docs/source/theory/numerical_integration.rst:19-22`,
`docs/source/theory/numerical_integration.rst:60-65`

**Proposed text:**

> Forward Euler has local truncation error proportional to `h^2` and accumulated
> global error proportional to `h`. Halving `h` approximately halves the global
> error in the asymptotic regime.
>
> An embedded pair shares stage evaluations between its main and embedded
> formulas. CuBIE's Dormand-Prince 5(4) tableau has seven stages.

**Reason:** The current page calls the halved quantity local error and says the
tableau has six stages.

**Evidence:** `docs/source/theory/numerical_integration.rst:53-55`,
`src/cubie/integrators/algorithms/generic_erk_tableaus.py:751-770` and the
seven-row `DORMAND_PRINCE_54_TABLEAU` definition in the same module.

### UFR-025: Correct nonlinear solver, preconditioner, and status descriptions

**Location:** `docs/source/theory/solvers.rst:28-46`,
`docs/source/theory/solvers.rst:65-82`,
`docs/source/theory/solvers.rst:84-97`

**Proposed text:**

> CuBIE's Newton-Krylov solver uses an update-norm contraction estimate. It
> accepts a small first update or an update whose estimated remaining error is
> below the scaled bound. Nonfinite updates, excessive contraction estimates,
> and stagnation terminate an iteration early.
>
> Implicit methods support Neumann and Jacobi preconditioners. A sequence of
> preconditioner names creates one chained generated helper.
>
> Run status is stored as `CUBIE_RESULT_CODES` bit flags. Iteration counts are
> stored in a separate `iteration_counters` output. They are not packed into
> the high bits of a return code.

**Evidence:**
`src/cubie/integrators/matrix_free_solvers/newton_krylov.py:270-317`,
`src/cubie/integrators/matrix_free_solvers/newton_krylov.py:450-480`,
`src/cubie/integrators/algorithms/ode_implicitstep.py:119-140`,
`src/cubie/integrators/algorithms/ode_implicitstep.py:540-585`,
`src/cubie/result_codes.py:22-88`.

### UFR-026: Repair the array-interpolation example

**Location:** `docs/source/examples/array_interpolation_example.py:3-7`,
`docs/source/examples/array_interpolation_example.py:110-112`

**Proposed code:**

```python
device_values[boundary_condition] = evaluate_on_device(
    interp.evaluation_function,
    interp.coefficients,
    dense_times,
)
```

Change the class reference to `cubie.array_interpolator.ArrayInterpolator`.

**Reason:** The documented module path and `device_function` attribute do not
exist. The public compiled property is `evaluation_function`.

**Evidence:** `src/cubie/array_interpolator.py:151-156`,
`src/cubie/array_interpolator.py:625-634`.

### UFR-027: Repair the controller-analysis example

**Location:** `docs/source/examples/controller_step_analysis.py:298-324`

**Proposed code:**

```python
return OutputFunctions(
    max_states=system.sizes.states,
    max_observables=system.sizes.observables,
    precision=system.precision,
    output_types=solver_settings["output_types"],
    saved_state_indices=solver_settings["saved_state_indices"],
    saved_observable_indices=solver_settings["saved_observable_indices"],
    summarised_state_indices=solver_settings["summarised_state_indices"],
    summarised_observable_indices=(
        solver_settings["summarised_observable_indices"]
    ),
)
```

**Reason:** The current positional call supplies the parameter count as the
observable count, supplies `output_types` as `precision`, shifts every later
argument, and omits precision.

**Evidence:** `src/cubie/outputhandling/output_functions.py:125-169`.

### UFR-028: Replace the stale test guide

**Location:** `tests/README.md:1-609`

**Proposed replacement:**

> # CuBIE test guide
>
> Use the shared session-scoped fixtures in `tests/conftest.py` and their
> default parameter sets unless the task grants an exception. Do not add mocks
> or patches without an explicit user exception. Do not weaken assertions.
>
> Subdirectory `conftest.py` files define isolated protocols for Julia reference
> data, matrix-free solver fixtures, symbolic parser fixtures, and memory
> singleton state. Read the nearest test instructions and existing fixture
> contract before adding a fixture.
>
> Run targeted simulator tests, then verify device behavior with the real-GPU
> suite. The repository `AGENTS.md` defines the commands and marker exclusions.

**Reason:** The current guide asserts that local fixtures and test classes do
not exist, although the current suite contains many of both. It says every
device-function access triggers a two-minute build, uses an absent
`.claude/check_cuda.py` script, omits current marker exclusions, and contains
instructions about previous AI agents rather than test behavior.

**Evidence:** a repository scan found local `@pytest.fixture` definitions in
multiple current test files and many current `class Test...` groups;
`.claude/check_cuda.py` is absent; current commands are in `AGENTS.md`.

### UFR-029: Replace the CUDA-writing stub with binding instructions

**Location:** `src/cubie/writing_cuda_functions.md:1-44`

**Proposed replacement:**

> # Writing CUDA device functions in CuBIE
>
> Import CUDA symbols from `cubie.cuda_simsafe`. Do not import a backend package
> directly. Obtain compiled functions through cached factory properties.
>
> Capture configuration-dependent booleans and scalar values in `build`. The
> compiler specializes the generated device function for those values. Use
> runtime predicates for values that vary between lanes or calls.
>
> Use `selp` for per-lane value selection when both candidate values are already
> available. Do not use it to force unconditional evaluation of expensive or
> unsafe work.
>
> Gate warp-coherent loop exits with `activemask` and the required `all_sync` or
> `any_sync` vote. Select the vote from the loop's acceptance condition and
> document that condition next to the exit.
>
> Register scratch buffers through `buffer_registry`. Keep configurable buffer
> locations in compile settings. Pass persistence through buffer registration.
> Follow `src/cubie/AGENTS.md` for factory, cache, import, precision, and buffer
> contracts.

**Reason:** The current file is a stub containing open questions, hedging, and
nonbinding framing. Those statements do not provide factual instructions.

**Evidence:** `src/cubie/integrators/algorithms/backwards_euler.py:118-134`.

### UFR-030: Normalize the Julia-reference data README

**Location:**
`tests/integrated_numerical_tests/julia_reference/data/README.md:7-25`

**Proposed text:**

> `julia_reference_ne.npz` stores the golden state arrays and the fixed and
> adaptive sweep grids. `algorithms.csv` maps CuBIE aliases to
> DifferentialEquations.jl constructors, orders, and families.
> `controller_constants.csv` stores the controller constants resolved by the
> Julia runner. The CuBIE gate applies those constants for the matched adaptive
> comparisons.

**Reason:** This preserves the content while removing dash-led and
semicolon-joined clause framing.

### UFR-031: Keep Fleet facts and split dense clauses

**Location:** `infra/fleet/README.md:8-28`, `infra/fleet/README.md:113-123`,
`infra/fleet/README.md:125-143`, `infra/fleet/README.md:145-176`,
`infra/fleet/README.md:202-245`, `infra/fleet/README.md:260-263`

The infrastructure claims checked against current code are fresh. Preserve the
facts and split each semicolon or dash-joined sentence into independent factual
sentences. Apply these exact transformations throughout the listed ranges.

- Replace `A; B` with `A. B`.
- Replace parenthetical em-dash clauses with a separate sentence.
- Convert the dashboard feature sentence at lines 113-117 into a bullet list
  with one feature per item.
- Replace “cost latency, not red legs” with “Launch failures leave the job
  queued while Fleet retries with backoff.”
- Remove measured anecdotes from parentheticals unless their command and raw
  measurement are recorded.

**Evidence:** `infra/fleet/cost_dashboard.py:66-86`,
`infra/fleet/cost_dashboard.py:639`,
`infra/fleet/cost_dashboard.py:2109`,
`infra/fleet/cost_dashboard.py:2219-2227`, `infra/fleet/main.tf:43-86`,
`.github/workflows/ci_cuda_tests.yml:416-441`,
`infra/fleet/bootstrap/cloudshell-iam.sh:22-24`.

## Framing and punctuation rewrites

### UFR-S01: Remove explicit framing and intensifiers

Apply the following sentence replacements.

| Location | Proposed text |
|---|---|
| `docs/source/getting_started.rst:4-6` | “CuBIE integrates batches formed from initial values, parameter sets, or both.” |
| `docs/source/getting_started.rst:62-64` | “`create_ODE_system` accepts equation strings and Python functions. Pass either form to `solve_ivp`.” |
| `docs/source/theory/index.rst:4-7` | “CuBIE uses numerical integration algorithms, generated derivative operators, nonlinear solvers, and CUDA execution.” |
| `docs/source/theory/cuda.rst:26-28` | “CuBIE uses Numba-compatible backends to JIT-compile Python functions into CUDA kernels.” |
| `docs/source/theory/numerical_integration.rst:77-78` | “The selected tableau determines the stability region. A-stable tableaus include the left half-plane. L-stable tableaus additionally damp modes in the stiff limit.” |
| `docs/source/theory/solvers.rst:17-19` | “The stage equation is nonlinear in :math:`k_i` and requires a root solve.” |
| `docs/source/tutorials/extracting_summaries.rst:63-68` | “A 50-unit summary interval produces one window for this run. A shorter interval produces one statistic per window. If `summarise_every` is omitted, CuBIE derives a whole-run window and warns because changing `duration` then changes compile settings.” |
| `docs/source/tutorials/extracting_summaries.rst:77-80` | “Summary arrays use `[window, summary, run]` indexing. `as_numpy_per_summary` returns one `[window, variable, run]` array per metric.” |
| `docs/source/tutorials/extracting_summaries.rst:89-92` | “Metrics that share an accumulator are computed by one generated metric function. Result keys retain the requested metric names.” |
| `docs/source/tutorials/extracting_summaries.rst:94-98` | Use the heading “Reduce saved output.” Delete “Two more levers.” |
| `docs/source/tutorials/first_sweep.rst:41-48` | “The function definition must be available through Python source inspection. A script, module, or notebook cell is supported. `exec` and `python -c` definitions are not. Equivalent equation strings avoid source inspection.” |
| `docs/source/tutorials/first_sweep.rst:68-70` | “String definitions infer state names from `dx = ...` left-hand sides. Function definitions require `states` to associate returned derivatives with state names.” |
| `docs/source/tutorials/first_sweep.rst:166-167` | Delete the paragraph. |
| `docs/source/tutorials/stiff_systems.rst:39-42` | “`method=\"radau\"` selects the fifth-order Radau IIA fully implicit method.” |
| `docs/source/tutorials/stiff_systems.rst:61-65` | “Rosenbrock-W methods perform a linear solve at each stage without Newton iteration. Crank-Nicolson and DIRK methods solve nonlinear stage equations.” |
| `docs/source/tutorials/stiff_systems.rst:152-156` | “`method="rosenbrock"` selects a linearly implicit Rosenbrock-W method. Explicit `atol` and `rtol` values record the requested error scale for failure diagnosis.” |
| `docs/source/user_guide/drivers.rst:4-8` | “Use `t` in a system definition for explicit time dependence. Use a driver for a named time-dependent input supplied as a function or sampled array.” |
| `docs/source/user_guide/choosing_algorithms.rst:17` | Replace “Recommended family” with “Algorithm family.” |
| `docs/source/user_guide/choosing_algorithms.rst:32-35` | “`erk` selects `dormand-prince-54`. `dirk` selects `l_stable_dirk_3`. `firk` selects `firk_gauss_legendre_2`. `rosenbrock` selects `ros3p`.” |
| `docs/source/user_guide/coming_from_matlab.rst:40-58` | “MATLAB `y(1)` and `y(2)` map to Python `y[0]` and `y[1]`. CuBIE accepts `y.x` and `y.v` attribute access. Declare parameters as named function arguments and in `parameters`. Return derivatives as a list or a dictionary. MATLAB `[0 20]` maps to `duration=20.0` with optional `t0`. `RelTol`, `AbsTol`, and `MaxStep` map to `rtol`, `atol`, and `dt_max`. `result` stores batch output arrays.” |
| `docs/source/user_guide/solving.rst:47-49` | “CuBIE accepts a single IVP and is designed for batches. Arrays of initial values or parameters create a batch.” |
| `docs/source/user_guide/systems.rst:176-182` | “Integration algorithms request generated linear operators, Jacobian-vector products, preconditioners, residuals, and time-derivative helpers through `SymbolicODE.get_solver_helper`.” |
| `docs/source/user_guide/optional_arguments.rst:216-217` | “`algorithm` accepts a custom `ButcherTableau` instance.” |
| `docs/source/user_guide/speed.rst:78` | “`default_timelogger` provides programmatic access to the logger.” |

The second semantic pass found the following additional constructions. Apply
these exact replacements. These findings include meaning-level framing and
intensification that a seed-phrase scan does not detect.

- `readme.md:34-40` becomes “Supply time-dependent forcing as functions or
  sampled arrays. Save selected states or observables. Compute summary metrics
  on the GPU without retaining full trajectories. CuBIE chunks batches to fit
  device and host memory. The cache reuses compatible compiled kernels.”
- `infra/fleet/README.md:26-28` becomes “The fleets define no `schedule`.
  RunsOn warm pools use on-demand capacity. This account has no on-demand
  G-instance quota.”
- `infra/fleet/README.md:93-102` becomes “The runners do not enable the
  `s3-cache` extra. It requires a `runs-on/action@v2` step in every job. Without
  that step, the sidecar intercepts the GitHub artifact service and
  `actions/upload-artifact` receives a non-JSON `CreateArtifact` response.
  Runners available to public repositories must not use the shared S3 cache
  bucket. CuBIE is public. `setup-uv` uses GitHub's cache service.”
- `docs/source/getting_started.rst:18-28` becomes “A backend extra is required.
  `mlir-cuda12` and `mlir-cuda13` install numba-cuda-mlir with toolkit wheels.
  `mlir` uses a compatible system toolkit. The deprecated numba-cuda backend is
  available through `cuda12`, `cuda13`, or `cuda`. Use numba-cuda for Python
  3.10, the CUDA simulator, or as the documented fallback after an MLIR error.”
- `docs/source/getting_started.rst:33-36` becomes “Define an ODE system, then
  solve a batch of initial-value problems.”
- `docs/source/getting_started.rst:102-105` becomes “:doc:`Coming from SciPy
  <user_guide/coming_from_scipy>` maps SciPy interfaces to CuBIE.
  :doc:`Coming from MATLAB <user_guide/coming_from_matlab>` maps MATLAB
  interfaces to CuBIE.”
- `docs/source/getting_started.rst:128-133` becomes “Install Pandas for
  DataFrame output. Install Matplotlib for driver plots and plots created from
  result arrays.”
- `docs/source/examples/array_interpolation_example.py:1-8` becomes a module
  docstring stating “Plot CUDA driver-array interpolation against SciPy
  references. The script builds harmonic sampled data, evaluates four boundary
  conditions on CUDA, and compares the results with `CubicSpline`. It requires
  SciPy, NumPy, Matplotlib, and a CUDA-capable device or the CUDA simulator.”
- `docs/source/examples/cpu_gpu_driver_evaluator_comparison.py:1-13` becomes a
  module docstring stating “Compare CPU and GPU driver evaluators on shared
  input samples. The script evaluates one sampled sequence with
  `ArrayInterpolator` and the CPU `DriverEvaluator`. Each subplot shows one
  endpoint-handling mode. It requires NumPy, Matplotlib, and a CUDA-capable
  device or the CUDA simulator.”
- `docs/source/examples/controller_step_analysis.py:1-12` becomes a module
  docstring stating “Visualise CPU reference controller step sizes for selected
  ODE systems. The module uses the integration-test configuration and plots
  accepted and rejected steps for each controller and system. The CPU reference
  determines the plotted step sizes.”
- `docs/source/examples/controller_step_analysis.py:679-683` becomes “Render
  accepted and rejected steps. Each system uses two plot rows. The upper row
  shows step-size history. The lower row shows states and drivers.”
- `docs/source/theory/cuda.rst:4-6` becomes “CuBIE executes batch ODE solves
  through CUDA on NVIDIA GPUs.”
- `docs/source/theory/jacobians.rst:4-6` becomes “Implicit algorithms and
  Rosenbrock-W methods use derivatives of the right-hand-side function
  :math:`f(t, x)`. CuBIE generates the required derivative operators from its
  symbolic expression IR.”
- `docs/source/theory/numerical_integration.rst:4-6` becomes “CuBIE provides
  explicit Runge-Kutta, diagonally implicit Runge-Kutta, fully implicit
  Runge-Kutta, and Rosenbrock-W algorithms.
  :doc:`/user_guide/choosing_algorithms` lists the registered methods.”
- `docs/source/theory/numerical_integration.rst:92-113` becomes “ERK tableaus
  are strictly lower triangular. DIRK tableaus are lower triangular with
  nonzero diagonal entries and solve one nonlinear stage system at a time.
  FIRK tableaus couple all stages in one nonlinear system. Rosenbrock-W methods
  perform a linear solve at each stage and do not run Newton iteration.”
- `docs/source/theory/numerical_integration.rst:131-135` becomes “PI, PID, and
  Gustafsson controllers update the step size from current or prior error
  information. Their registered gains and per-algorithm defaults determine the
  update.”
- `docs/source/theory/solvers.rst:4-6` becomes “DIRK and FIRK methods solve
  nonlinear stage equations at each time step. CuBIE uses generated operators
  in Newton-Krylov solves.”
- `docs/source/tutorials/extracting_summaries.rst:4-9` becomes “Summary outputs
  reduce per-run storage by computing requested metrics during integration.
  `output_types` accepts summary metric names. Omitting `"state"` prevents
  state-trajectory output.”
- `docs/source/tutorials/extracting_summaries.rst:14-15` becomes “The
  Lotka-Volterra system from :doc:`first_sweep` has oscillating state
  trajectories.”
- `docs/source/tutorials/first_sweep.rst:4-10` becomes “The Lotka-Volterra
  example sweeps two parameters and maps final state values to a heatmap.”
- `docs/source/tutorials/first_sweep.rst:15-21` becomes
  “:func:`~cubie.odesystems.symbolic.symbolicODE.create_ODE_system` accepts a
  Python function whose arguments are time, state, and named values. Return
  derivatives in state order. Access states and named values by attribute,
  name, or index.”
- `docs/source/tutorials/first_sweep.rst:84-87` becomes “With
  `grid_type="combinatorial"`, CuBIE solves every combination of the supplied
  parameter values. The 1,000-value `b_values` and `d_values` arrays produce
  1,000,000 initial-value problems.”
- `docs/source/tutorials/first_sweep.rst:103-111` becomes “Scalar initial
  values are shared across runs. Arrays sweep an initial state. `method="ode45"`
  selects the adaptive fifth-order Dormand-Prince pair. `save_every=0.5`
  records a snapshot every half time unit. CuBIE chunks batches that exceed the
  device-memory budget.”
- `docs/source/tutorials/first_sweep.rst:128` becomes “Check the per-run status
  codes before using the result.”
- `docs/source/tutorials/index.rst:4-11` becomes “1.
  :doc:`first_sweep` constructs a parameter-sweep heatmap. 2.
  :doc:`extracting_summaries` computes GPU-side summary metrics without state
  trajectories. 3. :doc:`stiff_systems` uses implicit integration and sampled
  forcing.”
- `docs/source/tutorials/stiff_systems.rst:4-12` becomes “A stiff system mixes
  fast and slow dynamics. Explicit-method step sizes remain limited by the
  fastest stable timescale after that component has decayed. Implicit methods
  have larger stability regions. Solve the Van der Pol system with an implicit
  method. Inspect status codes after a failed run. Supply sampled forcing
  through a driver.”
- `docs/source/tutorials/stiff_systems.rst:97-106` becomes “`STEP_TOO_SMALL`
  reports that the required step is below `dt_min`. Loosen `atol` and `rtol`
  when the requested error is below the arithmetic or model resolution. Lower
  `dt_min` when the dynamics require a smaller step. A vector `atol` sets a
  separate absolute tolerance for each state.”
- `docs/source/tutorials/stiff_systems.rst:163-167` becomes “Write closed-form
  forcing directly in the equations with the time symbol `t`. Equations using
  `t` do not use driver arrays or interpolation.”
- `docs/source/tutorials/stiff_systems.rst:185-196` is deleted because it
  repeats the preceding instructions as a framed recap.
- `docs/source/user_guide/batching.rst:4-5` becomes “CuBIE solves batches of
  independent initial-value problems in parallel. Parameter arrays and initial
  value arrays define the batch.”
- `docs/source/user_guide/choosing_algorithms.rst:4-7` becomes “Pass an
  algorithm name as `method` to :func:`~cubie.solve_ivp` or as `algorithm` to
  :class:`~cubie.Solver`. Names are case-insensitive.”
- `docs/source/user_guide/choosing_algorithms.rst:19-30` uses these decision
  notes. “ERK uses explicit stages. `dormand-prince-54` is the ERK family
  default.” “DIRK runs a nonlinear solve for each stage. Rosenbrock-W runs a
  linear solve for each stage.” “FIRK couples all stages. `radau_iia_5` is a
  three-stage method of order five with an embedded estimate.” “`euler` and
  `backwards_euler` are fixed-step methods.”
- `docs/source/user_guide/choosing_algorithms.rst:84-100` describes
  `dormand-prince-54` as the seven-stage embedded 5(4) ERK default, `tsit5` as
  an embedded 5(4) method, and `dormand-prince-853` as an order-eight method
  with fifth- and third-order estimators. Delete “industry standard,” “good
  default,” “often slightly more efficient,” and “useful.”
- `docs/source/user_guide/choosing_algorithms.rst:143-173` states tableau name,
  stage count, order, embedded-estimate availability, and aliases. Delete
  “excellent for stiff problems.”
- `docs/source/user_guide/choosing_algorithms.rst:211-233` becomes “`fixed`
  uses a constant step. `i` uses the current error. `pi` uses current and prior
  error. `pid` uses the change in error history. `gustafsson` uses prior
  error ratios and, for implicit methods, the Newton iteration count. Each
  algorithm supplies its registered controller defaults.”
- `docs/source/user_guide/coming_from_matlab.rst:4-7` becomes “CuBIE accepts a
  right-hand-side function, an algorithm name, and solver tolerances. MATLAB
  `ode45(f, [0 20], [1; 0], opts)` maps to CuBIE
  `solve_ivp(..., method="ode45", duration=20.0, ...)`.”
- `docs/source/user_guide/coming_from_matlab.rst:136-149` becomes “Equation
  strings preserve variable names and use Python operators. Use `**` for
  exponentiation. MATLAB's `^` operator is not accepted.”
- `docs/source/user_guide/coming_from_scipy.rst:4-8` becomes “CuBIE accepts a
  SciPy-style right-hand-side function when its body uses supported arithmetic,
  scalar calls, branches, and constant-bound loops. Parameters are declared by
  name. CuBIE returns batch arrays and uses `duration` and regular output
  intervals.”
- `docs/source/user_guide/coming_from_scipy.rst:97-102` uses the heading
  “Parameter batches” and the sentence “Arrays of parameter or initial values
  define a batch that one call integrates on the GPU.”
- `docs/source/user_guide/coming_from_scipy.rst:145-146` becomes
  “`Solver.solve` defaults to `grid_type="verbatim"`. `solve_ivp` defaults to
  `grid_type="combinatorial"`.”
- `docs/source/user_guide/configuration.rst:4-9` becomes “The entry-point
  parameters of :func:`~cubie.solve_ivp`, :class:`~cubie.Solver`, and
  :meth:`~cubie.Solver.solve` route to output, memory, step-control, algorithm,
  loop, cache, and kernel settings groups.”
- `docs/source/user_guide/drivers.rst:4-12` becomes “Use the time symbol `t`
  for explicit time dependence in a system equation. Use a driver for a named
  time-dependent input supplied as sampled values.”
- `docs/source/user_guide/drivers.rst:74-87` becomes “Use sampled drivers when
  forcing values are available at discrete times. CuBIE interpolates those
  values for fixed or adaptive steps. Sampling cadence determines the stored
  driver-array length.”
- `docs/source/user_guide/drivers.rst:120-122` becomes “Declare each driver name
  in the system's `drivers` argument. Reference the driver by its bare name in
  the function body.”
- `docs/source/user_guide/drivers.rst:135-152` becomes “`order` sets the spline
  polynomial degree and defaults to `3`. With `wrap=True`, samples repeat
  periodically. With `wrap=False`, the interpolator adds transition segments
  between zero and the endpoint values outside the sampled interval.
  `boundary_condition` accepts `"natural"`, `"periodic"`, `"clamped"`, or
  `"not-a-knot"`. It defaults to `"periodic"` with wrapping and `"clamped"`
  without wrapping.”
- `docs/source/user_guide/memory.rst:4-6` becomes “CuBIE manages VRAM and
  divides batches that exceed the configured memory budget.”
- `docs/source/user_guide/optional_arguments.rst:4-15` becomes “Pass solver
  options as keywords to :func:`~cubie.solve_ivp`, the :class:`~cubie.Solver`
  constructor, or :meth:`~cubie.batchsolving.solver.Solver.solve`. CuBIE routes
  each accepted keyword to its settings group. Omitted or `None` values use the
  default. Unknown names raise `KeyError`.”
- Delete `docs/source/user_guide/optional_arguments.rst:20`.
- `docs/source/user_guide/optional_arguments.rst:69-73` becomes
  “`step_controller` accepts `"fixed"`, `"i"`, `"pi"`, `"pid"`, or
  `"gustafsson"`. Each algorithm defines its default controller.”
- `docs/source/user_guide/optional_arguments.rst:91-94` becomes “Proposed gains
  inside the interval from `deadband_min` through `deadband_max` are replaced
  with `1.0`. Set both values to `1.0` to disable the interval.”
- `docs/source/user_guide/optional_arguments.rst:177-181` becomes
  “`minimal_residual` minimizes the residual along its search direction.
  `steepest_descent` uses the gradient direction. `bicgstab` selects the
  BiCGSTAB solver.”
- `docs/source/user_guide/optional_arguments.rst:185-187` becomes
  “`preconditioner_order` sets the number of terms in the truncated Neumann
  series. Each additional term adds one operator application.”
- `docs/source/user_guide/optional_arguments.rst:195-201` becomes “`beta` and
  `gamma` change the implicit equations. A mass matrix belongs to the system
  definition and requires an implicit algorithm.”
- `docs/source/user_guide/optional_arguments.rst:230-239` becomes
  “`save_variables` and `summarise_variables` restrict output to named states
  or observables. Their index-based counterparts select the same variables.
  Selection
  lists default to all indices, while `output_types` determines which output
  categories are active.”
- `docs/source/user_guide/optional_arguments.rst:244-251` becomes “Each
  `*_location` compile setting selects local or shared memory for one working
  buffer. Defaults are component-specific. Benchmark a location override on
  the target GPU.”
- `docs/source/user_guide/results.rst:11` and
  `docs/source/user_guide/results.rst:47` are deleted. Retain the existing
  section headings and definitions.
- `docs/source/user_guide/results.rst:68-78` deletes “Points to note.” Replace
  the GPU-result instruction with “Use a normal host solve when host arrays are
  required.”
- `docs/source/user_guide/results.rst:130-131` becomes “Summary outputs compute
  metrics without saving the full trajectory.”
- `docs/source/user_guide/results.rst:179-184` becomes “CuBIE combines requested
  metrics that share accumulators. The result legend preserves each requested
  metric name.”
- `docs/source/user_guide/results.rst:208-210` becomes “Requesting
  `iteration_counters` records Newton iterations, Krylov iterations, accepted
  steps, and rejected steps at each save point.”
- `docs/source/user_guide/solving.rst:4-7` becomes “Solve a defined ODE system
  with :func:`~cubie.batchsolving.solver.solve_ivp` or a reusable
  :class:`~cubie.batchsolving.solver.Solver`.”
- `docs/source/user_guide/solving.rst:12-13` becomes “The `solve_ivp` function
  constructs a solver and executes one batch. Call it with the Lotka-Volterra
  system and one initial condition.”
- `docs/source/user_guide/solving.rst:74-78` becomes “Create one
  :class:`~cubie.Solver` and call
  :meth:`~cubie.batchsolving.solver.Solver.solve` repeatedly to retain compiled
  objects and managed allocations.”
- `docs/source/user_guide/solving.rst:100-106` becomes “`Solver.update`
  reconfigures a live solver. `Solver.build_grid` constructs an input grid for
  reuse by solves with the same batch layout.”
- `docs/source/user_guide/solving.rst:117-120` becomes “`duration` sets the
  recorded integration length. `t0` sets the start time. `settling_time`
  integrates before output recording starts.”
- `docs/source/user_guide/systems.rst:100-124` becomes “Equation strings define
  assignments directly. A left-hand side of `dx` defines state `x`. Undeclared
  right-hand-side symbols are inferred as parameters with value `0.0` and emit
  a warning. Explicit `states`, `parameters`, and `constants` mappings assign
  roles and default values.”
- `docs/source/user_guide/systems.rst:145-151` becomes “Default values are
  optional. Unlisted assigned auxiliaries remain in the symbolic expressions.
  Their trajectories are not saved.”
- `docs/source/user_guide/timing.rst:111-119` becomes “Updating a derived step
  parameter recalculates dependent values when validation rules require it.
  `Solver.update` preserves an explicitly supplied `dt_min` when a later `dt`
  is below it.”
- `docs/source/user_guide/timing.rst:278-280` becomes “`settling_time` integrates
  from `t0` without recording output. Use it to exclude the initial transient
  from recorded data.”
- `docs/source/user_guide/timing.rst:311-315` uses the heading “Combined timing
  configuration” and deletes the sentence introducing the example.
- `docs/source/user_guide/troubleshooting.rst:44-48` becomes “Reduce the number
  of saved variables or request summaries instead of trajectories. Lower
  `mem_proportion` to reserve VRAM for other processes. Reduce the batch size
  when host memory is the limiting resource.”
- `docs/source/user_guide/troubleshooting.rst:64-69` becomes “CUDASIM executes
  CUDA kernels on one CPU thread and does not require a GPU. Use it for
  CPU-only logic checks. It is available only with the deprecated numba-cuda
  backend.”
- `docs/source/user_guide/userfunctions.rst:4-11` becomes “System equations
  accept supplied Python function and CUDA device function calls. CuBIE inlines
  Python functions that evaluate on symbolic arguments. A device function
  remains an opaque call. Supply its derivative function when an algorithm
  generates Jacobian terms.”
- `docs/source/user_guide/userfunctions.rst:159-166` becomes “A device function
  without a derivative helper cannot be used by an algorithm that generates
  Jacobian terms. CuBIE inlines Python functions that evaluate on symbolic
  arguments. Functions that do not evaluate on symbolic arguments remain named
  calls and require a derivative for differentiation.”

### UFR-S02: Remove prose em dashes and semicolons

For every prose occurrence listed in the companion inventory, replace a dash or
semicolon joining independent clauses with a full stop. Keep punctuation that
is required by Python, MATLAB, CSS, Make, URLs, RST roles and directives, or
mathematical notation. Keep a colon only for a literal mapping, signature,
directive, table field, or a short list whose introduction is a complete
sentence.

The following high-density files should be rewritten by paragraph rather than
with character substitution.

- `docs/source/user_guide/optional_arguments.rst`: replace dash-led definition
  entries with RST definition-list terms. Split all semicolon-joined defaults
  and constraints into sentences.
- `docs/source/user_guide/results.rst`: replace dash-separated accessor
  definitions with a table. Split clauses at lines 18, 31, 42, 58-59, 70-75,
  84-86, 183, and 209.
- `docs/source/user_guide/coming_from_matlab.rst` and
  `docs/source/user_guide/coming_from_scipy.rst`: retain semicolons inside source
  code. Split them in narrative prose.
- `docs/source/user_guide/configuration.rst`: replace punctuation-heavy table
  descriptions with one factual sentence per field.
- `docs/source/user_guide/optional_arguments.rst`: convert the fragment
  lead-ins at lines 75, 98, 128, 148, 183, and 195 into RST subsection
  headings without colons.
- `docs/source/user_guide/results.rst`: delete the framed lead-ins “Key
  attributes,” “Convenience accessors,” and “Points to note.” The existing
  section heading and following definitions carry the structure.
- `docs/source/user_guide/timing.rst`: convert “Adaptive controllers,” “Fixed
  controllers,” “Example usage,” and the complete-example lead-in into RST
  headings without colons.
- `docs/source/user_guide/choosing_algorithms.rst`: replace subjective labels
  such as “good default,” “excellent,” “most common,” and “sensible” with
  method properties such as order, stability, stage count, and embedded-error
  availability.
- `readme.md:98-115`: replace acknowledgement dashes with complete sentences
  under one heading per project.

## Completeness additions

### UFR-C01: Add an environment and backend page

Document `CUBIE_CUDA_BACKEND`, `NUMBA_ENABLE_CUDASIM`, `CUBIE_LINEINFO`,
`CUBIE_CACHE_DIR`, `CUBIE_KERNEL_CACHE_DIR`, and
`CUBIE_MAX_CACHE_ENTRIES`. State import-time versus call-time behavior.

### UFR-C02: Add memory ownership and cleanup

Document solver and result lifetime, host buffer loans, pinned or memmap spill,
`auto_memory`, stream-group ownership, `Solver.close`, and context-manager use.

### UFR-C03: Add DAE workflow documentation

Document implicit equations, algebraic unknowns, higher-order derivatives,
automatic structural simplification, `simplify`, `state_priority`,
`irreducible`, `simplify_options`, mass matrices, and the requirement for an
implicit algorithm after tearing.

### UFR-C04: Generate algorithm and setting tables from registries

Generate alias, order, stage, embedded-estimate, and default-controller tables
from the tableau and algorithm registries during the docs build.

### UFR-C05: Add result status and device-result lifecycle pages

Document every `CUBIE_RESULT_CODES` member, bitwise combinations,
`status_messages`, `DeviceSolveResult`, `on_device`, stream ordering, host
materialization, and the live-solver requirement.

## Changelog decision

`CHANGELOG.md` is current through version 0.4.0 and was read in full. No entry
was found to contradict its historical release context. No style rewrite is
proposed because repository policy forbids editing the changelog.
