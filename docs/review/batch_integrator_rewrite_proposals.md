# Batch solving, integrator, and developer-guide rewrite proposals

## Scope and method

This audit covers every line in these 69 pages, totalling 1,917 lines.

- `docs/source/developer_guide/**/*.rst`
- `docs/source/API_reference/index.rst`
- `docs/source/API_reference/batchsolving/**/*.rst`
- `docs/source/API_reference/integrators/**/*.rst`

The audit follows every `autoclass`, `autofunction`, and `autodata`
directive into the source docstrings rendered by `:members:`. There are 58
autodoc directives. Fifty-four targets resolve and four are stale. The
inventory artifact records the result for each page and the missing public API
surface. A prospective audit covers docstrings for four proposed missing
API targets before those targets are added.

This artifact records proposals. No source documentation was changed.

## Developer guide

### DG-00: contributor-guide opening carries framing

Location: `docs/source/developer_guide/index.rst:4-7`

Current text:

```rst
This section is for contributors and users who want to extend CuBIE with
new algorithms, metrics, or other components. It assumes familiarity with
the :doc:`User Guide </user_guide/index>` and a working knowledge of
Python, NumPy, and CUDA concepts.
```

Replacement:

```rst
Implement algorithm, metric, code-generation, and CUDA-buffer extensions
with these instructions. The examples require Python, NumPy, and CUDA
knowledge.
```

Reason: the current opening frames the section and describes reader identity.
The replacement states the prerequisites and available instructions.

### DG-01: the tableau example uses stale construction details

Location: `docs/source/developer_guide/adding_algorithms.rst:4-26`

Current text:

```rst
CuBIE's algorithm system is designed to make adding new integration
methods straightforward, from simple tableau entries to entirely new
algorithm families.

The simplest case.  Add a new entry to ``ERK_TABLEAU_REGISTRY`` in
``src/cubie/integrators/algorithms/generic_erk_tableaus.py``:

.. code-block:: python

   ERK_TABLEAU_REGISTRY["my_method_43"] = ERKTableau(
       a=np.array([...]),       # Butcher A matrix (lower-triangular)
       b=np.array([...]),       # weights
       b_hat=np.array([...]),   # embedded weights (or None if non-adaptive)
       c=np.array([...]),       # nodes
       order=4,
       name="my_method_43",
   )

The method is immediately available as
``Solver(system, algorithm="my_method_43")``.
```

Replacement:

```rst
Add integration tableaus to the registry for their algorithm family.
Add an ERK tableau to ``ERK_TABLEAU_REGISTRY`` in
``src/cubie/integrators/algorithms/generic_erk_tableaus.py``.

.. code-block:: python

   MY_METHOD_43_TABLEAU = ERKTableau(
       a=((...), (...)),
       b=(...),
       b_hat=(...),
       c=(...),
       order=4,
   )

   ERK_TABLEAU_REGISTRY["my_method_43"] = MY_METHOD_43_TABLEAU

Select the method with ``Solver(system, algorithm="my_method_43")``.
```

Reason: current tableaus are frozen attrs objects with tuple coefficients and
no `name` field. The registry supplies the public name. See
`src/cubie/integrators/algorithms/base_algorithm_step.py:217-257` and
`src/cubie/integrators/algorithms/generic_erk_tableaus.py:751-771`.

### DG-02: tableau-family and new-family instructions are stale and incomplete

Location: `docs/source/developer_guide/adding_algorithms.rst:28-60`

Current text:

```rst
The process is analogous.  Each family has its own tableau registry and
dataclass:

- ``DIRK_TABLEAU_REGISTRY`` in ``generic_dirk_tableaus.py``
- ``FIRK_TABLEAU_REGISTRY`` in ``generic_firk_tableaus.py``
- ``ROSENBROCK_TABLEAUS`` in ``generic_rosenbrockw_tableaus.py``

FIRK tableaus must also supply a transformation matrix ``T`` and its
inverse for the stage-coupled solver.

For an entirely new algorithm type:

1. **Subclass** ``ODEExplicitStep`` or ``ODEImplicitStep`` (in
   ``src/cubie/integrators/algorithms/``).

2. **Implement** ``build_step()``, which returns a CUDA device function
   that performs one integration step.

3. **Register buffers** via the buffer registry for any working arrays
   the step function needs (see :doc:`buffer_registry`).

4. **Register** the new class in ``_ALGORITHM_REGISTRY`` in
   ``src/cubie/integrators/algorithms/__init__.py``.

5. **Declare defaults** by setting ``StepControlDefaults`` on the class
   to specify preferred controller settings (tolerances, step-size
   bounds, controller type).
```

Replacement:

```rst
Add DIRK, FIRK, and Rosenbrock tableaus to ``DIRK_TABLEAU_REGISTRY``,
``FIRK_TABLEAU_REGISTRY``, and ``ROSENBROCK_TABLEAUS``. The tableau
classes are frozen attrs classes. Supply the coefficient fields declared
by the selected class. ``FIRKTableau`` uses the base ``a``, ``b``,
``b_hat``, ``c``, and ``order`` fields and does not accept transformation
matrices.

Add a new algorithm family with these changes.

1. Subclass ``ODEExplicitStep`` or ``ODEImplicitStep``.
2. Implement the abstract build and resource properties required by the
   base class, including ``build_step()``.
3. Register working buffers before a build requests their allocators.
4. Add accepted settings to ``ALL_ALGORITHM_STEP_PARAMETERS``.
5. Add the class to ``_ALGORITHM_REGISTRY``.
6. Construct the algorithm family's ``StepControlDefaults`` and pass it to
   the base constructor.
7. Add a CPU reference implementation and compare GPU results with it in
   the algorithm tests.
```

Reason: `FIRKTableau` declares no transformation fields
(`src/cubie/integrators/algorithms/generic_firk_tableaus.py:52-59`). The
factory filters settings through `ALL_ALGORITHM_STEP_PARAMETERS`
(`src/cubie/integrators/algorithms/base_algorithm_step.py:62-123`). Existing
families pass defaults during construction rather than assigning a class
attribute, for example
`src/cubie/integrators/algorithms/generic_dirk.py:81-99`.

### DG-03: the implicit-helper example calls a removed API

Location: `docs/source/developer_guide/adding_algorithms.rst:65-73`

Current text:

```rst
Implicit algorithms typically need Jacobian--vector products and solver
infrastructure.  Use ``build_implicit_helpers()`` to request these from
the ODE system's code generator:

.. code-block:: python

   helpers = self.system.get_solver_helper("linear_operator_cached")
```

Replacement:

```rst
Implicit algorithms obtain generated helpers through the configured
request callback.

.. code-block:: python

   from cubie.odesystems.solver_helpers import (
       SolverHelperKind,
       SolverHelperRequest,
   )

   operator = self.compile_settings.get_solver_helper_fn(
       SolverHelperRequest(
           kind=SolverHelperKind.LINEAR_OPERATOR_CACHED,
       )
   ).device_function

Use ``build_implicit_helpers()`` when the standard implicit-step helper
chain matches the algorithm.
```

Reason: helper lookup takes an immutable `SolverHelperRequest`; implicit
steps retrieve `.device_function` from the result. See
`src/cubie/odesystems/solver_helpers.py:288-338` and
`src/cubie/integrators/algorithms/ode_implicitstep.py:587-625`.

### DG-04: the custom metric imports and metadata are wrong

Location: `docs/source/developer_guide/adding_metrics.rst:4-44`

Current text:

```rst
CuBIE's summary metrics system is extensible via the
``@register_metric`` decorator.  This page walks through implementing a
custom metric.

   from cubie.outputhandling.summarymetrics.metrics import (
       SummaryMetric,
       MetricFuncCache,
       register_metric,
       summary_metrics,
   )

   @register_metric(summary_metrics)
   class MyCustomMetric(SummaryMetric):
       def __init__(self, precision):
           super().__init__(
               buffer_size=2,
               output_size=1,
               name="my_custom",
               precision=precision,
               unit_modification="[custom]",
               sample_summaries_every=None,
           )

       def build(self) -> MetricFuncCache:
           # Return a MetricFuncCache with update and save callables
           ...
```

Replacement:

```rst
Register a ``SummaryMetric`` subclass with ``@register_metric``.

   from cubie.outputhandling.summarymetrics import summary_metrics
   from cubie.outputhandling.summarymetrics.metrics import (
       MetricFuncCache,
       SummaryMetric,
       register_metric,
   )

   @register_metric(summary_metrics)
   class MyCustomMetric(SummaryMetric):
       def __init__(self, precision):
           super().__init__(
               buffer_size=2,
               output_size=1,
               name="my_custom",
               precision=precision,
               unit_modification="[unit] custom",
           )

       def build(self) -> MetricFuncCache:
           ...
```

Reason: `summary_metrics` is created in the package initializer, not
`metrics.py` (`src/cubie/outputhandling/summarymetrics/__init__.py:27-38`).
`unit_modification` is a format string containing `[unit]`, and the sampling
interval defaults to `0.01`
(`src/cubie/outputhandling/summarymetrics/metrics.py:162-189`).

### DG-05: metric parameter descriptions and callback signatures are stale

Location: `docs/source/developer_guide/adding_metrics.rst:49-84`

Current text:

```rst
``buffer_size``
   Number of scratch elements per variable needed during accumulation.
   Can be an ``int`` or a callable that receives the number of
   summarised variables.

``unit_modification``
   Label suffix appended to variable names in the summary legend
   (e.g. ``"[mean]"``, ``"[max]"``).

``sample_summaries_every``
   If not ``None``, the metric requests sub-step sampling at this
   interval.

``update(buffer, value, t, dt)``
   Called at every summary sample point during integration.  Accumulates
   data into ``buffer``.

``save(buffer, output, n_samples)``
   Called at the end of the solve.  Finalises the accumulated data and
   writes to ``output``.
```

Replacement:

```rst
``buffer_size``
   Number of scratch elements per variable. A callable receives the
   metric parameter parsed from the output specification.

``output_size``
   Number of persisted output elements per variable. A callable receives
   the metric parameter.

``unit_modification``
   Legend format string. Use ``[unit]`` where the source unit belongs.

``sample_summaries_every``
   Time interval between summary samples. Derivative metrics use the
   interval to scale finite differences.

``update(value, buffer, current_index, customisable_variable)``
   Accumulates one sampled value.

``save(buffer, output_array, summarise_every, customisable_variable)``
   Writes one summary and resets any scratch state required for the next
   summary interval.
```

Reason: the base class documents the sampling interval for every metric at
`src/cubie/outputhandling/summarymetrics/metrics.py:148-149`. Derivative
metrics use it for finite-difference scaling as documented at
`src/cubie/outputhandling/summarymetrics/metrics.py:80-82`. The callback
contracts are at `src/cubie/outputhandling/summarymetrics/metrics.py:125-159`
and `src/cubie/outputhandling/summarymetrics/metrics.py:224-240`.

### DG-06: the Count metric does not compile against the current callback API

Location: `docs/source/developer_guide/adding_metrics.rst:86-118`

Current text:

```rst
Example: Implementing a "Count" Metric
----------------------------------------

A metric that counts how many samples were accumulated:

.. code-block:: python

   from numba import cuda

   @register_metric(summary_metrics)
   class CountMetric(SummaryMetric):
       def __init__(self, precision):
           super().__init__(
               buffer_size=1,
               output_size=1,
               name="count",
               precision=precision,
               unit_modification="[count]",
           )

       def build(self) -> MetricFuncCache:
           @cuda.jit(device=True)
           def update(buffer, value, t, dt):
               buffer[0] += 1.0

           @cuda.jit(device=True)
           def save(buffer, output, n_samples):
               output[0] = buffer[0]

           return MetricFuncCache(update=update, save=save)

After adding the file, the metric is available as
``output_types=["count"]``.
```

Replacement:

```rst
Count metric
------------

.. code-block:: python

   from cubie.cuda_simsafe import cuda

   @register_metric(summary_metrics)
   class CountMetric(SummaryMetric):
       def __init__(self, precision):
           super().__init__(
               buffer_size=1,
               output_size=1,
               name="count",
               precision=precision,
               unit_modification="[unit] count",
           )

       def build(self) -> MetricFuncCache:
           precision = self.compile_settings.precision

           @cuda.jit(
               device=True,
               inline=True,
               **self.jit_kwargs,
           )
           def update(
               value,
               buffer,
               current_index,
               customisable_variable,
           ):
               buffer[0] += precision(1.0)

           @cuda.jit(
               device=True,
               inline=True,
               **self.jit_kwargs,
           )
           def save(
               buffer,
               output_array,
               summarise_every,
               customisable_variable,
           ):
               output_array[0] = buffer[0]
               buffer[0] = precision(0.0)

           return MetricFuncCache(update=update, save=save)

Import the new module once in
``cubie.outputhandling.summarymetrics.__init__`` after ``summary_metrics``
is constructed. The import registers ``output_types=["count"]``.
```

Reason: CUDA imports must use `cuda_simsafe`. Built-in metrics use `inline=True`,
the factory `jit_kwargs`, current argument order, and buffer reset. See
`src/cubie/outputhandling/summarymetrics/mean.py:62-79` and
`src/cubie/outputhandling/summarymetrics/mean.py:99-136`. Registration happens
through initializer imports at
`src/cubie/outputhandling/summarymetrics/__init__.py:37-55`.

### DG-07: the ownership tree and CUDAFactory claim are inaccurate

Location: `docs/source/developer_guide/architecture.rst:4-22`

Current text:

```rst
This page describes CuBIE's internal design patterns.  Understanding
these is essential for contributing new algorithms, metrics, or other
components.

A typical solve involves the following hierarchy::

   Solver
   └── BatchSolverKernel
       └── SingleIntegratorRun
           ├── IVPLoop
           ├── OutputFunctions
           ├── Algorithm step (e.g. ERKStep, FIRKStep)
           └── Controller (PI, PID, Gustafsson)

Each component is a :class:`~cubie.CUDAFactory.CUDAFactory` subclass that
generates CUDA device functions.
```

Replacement:

```rst
CuBIE composes a solve through this ownership tree::

   Solver
   └── BatchSolverKernel
       ├── ArrayInterpolator
       └── SingleIntegratorRun
           ├── IVPLoop
           ├── OutputFunctions
           ├── Algorithm step
           │   └── Matrix-free solver for implicit methods
           └── Step controller

``Solver`` owns the host orchestration API. CUDA-generating descendants
derive from :class:`~cubie.CUDAFactory.CUDAFactory` and expose cached
compiled-function properties.
```

Reason: `Solver` is not a `CUDAFactory`. `BatchSolverKernel` constructs and
owns the driver interpolator at
`src/cubie/batchsolving/BatchSolverKernel.py:303-312`.
`Solver.driver_interpolator` is a passthrough to the kernel-owned object at
`src/cubie/batchsolving/solver.py:608-611`.

### DG-08: the lifecycle promises a generic device-function property

Location: `docs/source/developer_guide/architecture.rst:27-47`

Current text:

```rst
Every CUDA-generating component follows the same pattern:

2. **Build.**  The ``build()`` method generates and returns compiled
   device functions.  Subclasses override this method.

3. **Cache.**  The result of ``build()`` is cached.  Subsequent accesses
   via the ``device_function`` property return the cached result without
   rebuilding.

.. important::

   Never call ``build()`` directly.  Always access compiled functions
   through properties (e.g. ``device_function``), which handle caching.
```

Replacement:

```rst
CUDA-generating components use this lifecycle.

2. **Build.** The ``build()`` method returns the component's dispatcher
   cache object.

3. **Cache.** A public compiled-function property calls
   ``get_cached_output(name)`` and returns the named field from the
   component's dispatcher cache. Property names are component-specific.

Call compiled-function properties. Do not call ``build()`` directly.
```

Reason: `get_cached_output(name)` builds an invalid cache and returns the named
cache field at `src/cubie/CUDAFactory.py:578-604`. Algorithm steps expose that
protocol through properties such as `step_function` at
`src/cubie/integrators/algorithms/base_algorithm_step.py:877-880`.

### DG-09: config comparison and update arity are stale

Location: `docs/source/developer_guide/architecture.rst:52-55`

Current text:

```rst
``_CubieConfigBase`` provides ``values_hash`` (a hex digest of the frozen
config) and ``values_tuple`` for comparison.  ``update()`` returns two
sets: the names of fields that were *requested* to change and those that
*actually* changed.  Only actual changes invalidate the cache.
```

Replacement:

```rst
``_CubieConfigBase.values_hash`` is the canonical digest of a frozen
configuration snapshot. ``update()`` returns a replacement snapshot, the
recognized setting names, and the names whose converted values changed.
A change to a converted value invalidates the factory cache.
```

Reason: `values_tuple` does not exist, and `update()` returns a three-item
tuple. See `src/cubie/CUDAFactory.py:194-276` and
`src/cubie/CUDAFactory.py:278-300`.

### DG-10: MultipleInstanceCUDAFactory is described as a collection

Location: `docs/source/developer_guide/architecture.rst:60-63`

Current text:

```rst
Some components appear multiple times in the tree (e.g. one algorithm
step per stage in a multi-stage method).  ``MultipleInstanceCUDAFactory``
manages a collection of identically-typed factories distinguished by a
prefix string, so their buffers and settings don't collide.
```

Replacement:

```rst
``MultipleInstanceCUDAFactory`` represents one factory instance whose
selected configuration keys use an instance prefix. For example,
``krylov_atol`` maps to the internal ``atol`` field of the Krylov solver.
The prefix separates settings for coexisting solver instances.
```

Reason: the class does not own a collection. Prefix mapping occurs in the
single instance's config update method. See
`src/cubie/CUDAFactory.py:706-832` and `src/cubie/CUDAFactory.py:835-878`.

### DG-11: buffer-registry scope and memory descriptions overstate behavior

Location: `docs/source/developer_guide/buffer_registry.rst:4-30`

Current text:

```rst
CuBIE centralises GPU memory management through the
:mod:`~cubie.buffer_registry` module.  Every CUDA-generating component
registers its memory requirements, and the registry computes a layout
that is allocated once at kernel launch time.

``location``
   ``"shared"`` (on-chip, fast, limited to ~48 KB per block) or
   ``"local"`` (per-thread, in registers/L1).

``persistent``
   If ``True``, the buffer survives across steps (e.g. state arrays).
   Non-persistent buffers can be aliased.
```

Replacement:

```rst
The :mod:`~cubie.buffer_registry` module computes per-run scratch layouts
for CUDA factories. Batch input and output allocation is handled by the
memory-manager layer.

``location``
   ``"shared"`` selects dynamic shared memory. ``"local"`` selects a
   per-thread CUDA local array.

``persistent``
   A persistent local buffer occupies a stable slice of the top-level
   per-run persistent array across integration steps. Non-persistent
   Register non-persistent buffers as aliases when their lifetimes do not
   overlap.
```

Reason: the registry stores layout metadata and builds slice allocators; it is
not the allocator for all GPU arrays. CUDA local arrays are not guaranteed to
remain in registers or L1. See `src/cubie/buffer_registry.py:649-700` and
`src/cubie/buffer_registry.py:703-710`.

### DG-12: registration timing and allocator signatures are wrong

Location: `docs/source/developer_guide/buffer_registry.rst:38-69`

Current text:

```rst
Components register buffers in their ``build()`` method:

``get_allocator(name, parent)``
   Returns a function that, given a base pointer and thread index,
   returns a typed array slice for the named buffer.

``get_child_allocators(parent, child)``
   Delegates a region of the parent's allocation to a child component.
```

Replacement:

```rst
Register buffers before requesting layouts or compiled allocators.
Factories normally register buffers during construction and refresh their
sizes when compile settings change.

``get_allocator(name, parent)``
   Returns a device function with the signature
   ``(shared, persistent) -> array``.

``get_child_allocators(parent, child, name=None)``
   Registers the child footprint on the parent and returns the child's
   shared and persistent slice allocators.
```

Reason: allocator generation reads an existing layout and takes shared and
persistent base arrays, not a thread index
(`src/cubie/buffer_registry.py:649-700`). The child allocator accepts an
optional name and performs registration
(`src/cubie/buffer_registry.py:1173-1208`).

### DG-13: shared-memory limits and layout-query units are wrong

Location: `docs/source/developer_guide/buffer_registry.rst:71-95`

Current text:

```rst
Shared memory is limited (~48 KB default, configurable up to 100 KB on
some architectures).  When the total exceeds the budget, CuBIE reduces
the block size (fewer threads per block) to fit.

``buffer_registry.shared_buffer_size(parent)``
   Total shared memory bytes for a component and its children.

``buffer_registry.local_buffer_size(parent)``
   Total local memory bytes (non-persistent).

``buffer_registry.persistent_local_buffer_size(parent)``
   Total local memory bytes (persistent).

Use these to verify that your component's memory footprint is reasonable.
```

Replacement:

```rst
The performance stage reduces block size toward a dynamic shared-memory
target under 32 KiB and stops at one warp. The 32 KiB target does not
apply after the block reaches one warp. The hardware stage then applies
the device's per-block limit and reduces to fewer than 32 threads when
required for a valid launch.

``buffer_registry.shared_buffer_size(parent)``
   Number of shared-memory elements in the parent's layout, including
   synthetic shared rollups for registered children.

``buffer_registry.local_buffer_size(parent)``
   Number of non-persistent local elements registered on the parent.

``buffer_registry.persistent_local_buffer_size(parent)``
   Number of persistent local elements in the parent's layout, including
   synthetic persistent rollups for registered children.

Multiply each element count by the buffer dtype's item size to calculate
bytes.
```

Reason: the current block-size policy is implemented at
`src/cubie/batchsolving/BatchSolverKernel.py:810-853`. Registry query methods
return element counts at `src/cubie/buffer_registry.py:584-618`. Child
registration creates shared and persistent synthetic entries at
`src/cubie/buffer_registry.py:1118-1167`; it creates no non-persistent local
rollup.

### DG-14: code-generation opening and parser scope are padded and incomplete

Location: `docs/source/developer_guide/codegen.rst:4-33`

Current text:

```rst
CuBIE transforms symbolic ODE definitions into compiled CUDA device
functions.  SymPy is the parse layer for string and SymPy input; every
expression converts to a lightweight interned expression IR (the
``engine`` package) at the parse boundary, and every later stage —
classification, structural simplification, differentiation, CSE,
hashing, and printing — runs on the IR.  This page describes the
pipeline.

The parser in ``src/cubie/odesystems/symbolic/parsing/`` tokenises the
equation strings, identifies states (variables with ``d<name>`` on the
left-hand side), parameters, constants, and observables, and converts
every expression to engine IR before returning.
```

Replacement:

```rst
CuBIE parses string, SymPy, callable, and CellML systems and emits CUDA
device-function factories. SymPy input is converted to the interned
``engine`` expression IR at the parse boundary. Classification,
structural simplification, differentiation, common-subexpression
elimination, hashing, and printing operate on that IR.

The parsing package normalises accepted system inputs, classifies explicit
ODE and DAE equations, applies CellML substitutions, and converts symbolic
expressions to engine IR.
```

Reason: the public normalisation path accepts more than equation strings and
performs structural DAE processing. The pipeline modules are routed from
`src/cubie/odesystems/symbolic/symbolicODE.py:318-352` and structural DAE
simplification is described at
`src/cubie/odesystems/symbolic/symbolicODE.py:391-415`.

### DG-15: helper and stage-utility descriptions use removed behavior

Location: `docs/source/developer_guide/codegen.rst:53-81`

Current text:

```rst
``src/cubie/odesystems/symbolic/engine/`` holds the expression IR the
compute phase runs on: nodes are interned (structurally
identical expressions are the same object), so substitution,
differentiation, and common-subexpression elimination are single passes
over a DAG.  The engine's printer renders IR as Numba-CUDA source:

``get_solver_helper()`` dispatches to the appropriate code-generation
path based on the requested helper name (``"linear_operator"``,
``"prepare_jac"``, ``"calculate_cached_jvp"``,
``"time_derivative_rhs"``).

``_stage_utils`` provides helpers for FIRK methods that need to generate
code for multiple coupled stages simultaneously, including
block-structured linear algebra and transformation matrices.
```

Replacement:

```rst
``src/cubie/odesystems/symbolic/engine/`` holds the interned expression
IR. Structural equality reuses the same nodes. Substitution,
differentiation, and common-subexpression elimination traverse the DAG.
The printer renders IR as Numba CUDA source.

``get_solver_helper(request, cache_policy=None)`` accepts a
``SolverHelperRequest`` containing a ``SolverHelperKind`` and the
coefficient metadata required by that kind. The result carries the
generated device function and helper metadata. ``get_solver_helper_fn`` is
the algorithm-config callback field. Use ``solver_helper_getter`` to bind
a cache policy before passing the callback to the algorithm.

``_stage_utils`` normalises FIRK stage coefficients and nodes and creates
IR symbol assignments for generated coupled-stage helpers.
```

Reason: the system API retains `get_solver_helper(request, cache_policy=None)`
at `src/cubie/odesystems/baseODE.py:440-444` and
`src/cubie/odesystems/symbolic/symbolicODE.py:941-945`. The policy-bound
getter is built at `src/cubie/odesystems/baseODE.py:415-438`.
`get_solver_helper_fn` is an algorithm-config callback field at
`src/cubie/integrators/algorithms/base_algorithm_step.py:614-640`.
`_stage_utils` provides only stage-data normalisation and metadata construction
at `src/cubie/odesystems/symbolic/codegen/_stage_utils.py:37-108`.

### DG-16: generated-file location and compilation timing are overstated

Location: `docs/source/developer_guide/codegen.rst:83-89`

Current text:

```rst
Each system uses one generated Python module at
``generated/<system_name>/<system_name>.py``. Solver helpers are added
to that module as they are requested. Numba compiles the factories on
import.
```

Replacement:

```rst
Each symbolic system stores one generated Python module at
``<cache_root>/<system_name>/<system_name>.py``. The cache root is the
process override, ``CUBIE_CACHE_DIR``, or ``<cwd>/generated`` in that
order. Requested helper factories are appended to the module. Importing
the module loads the Python factory. A compiled-function property invokes
the factory and caches the resulting CUDA dispatcher.
```

Reason: cache-root precedence is implemented at `src/cubie/cache_root.py:34-50`.
`ODEFile` writes and imports generated Python factories but does not JIT them
on import (`src/cubie/odesystems/symbolic/odefile.py:27-45` and
`src/cubie/odesystems/symbolic/odefile.py:133-166`).

### DG-17: test commands are not the project commands and are not PowerShell

Location: `docs/source/developer_guide/testing.rst:9-22`

Current text:

```rst
.. code-block:: bash

   # Full suite (requires CUDA GPU)
   python -m pytest

   # CPU-only tests (CUDASIM mode)
   export NUMBA_ENABLE_CUDASIM=1
   python -m pytest -m "not nocudasim and not cupy"
```

Replacement:

```rst
.. code-block:: powershell

   # Simulator first pass
   $env:NUMBA_ENABLE_CUDASIM = "1"
   pytest -m "not nocudasim and not cupy and not specific_algos"

   # Real GPU verification
   Remove-Item Env:NUMBA_ENABLE_CUDASIM -ErrorAction SilentlyContinue
   pytest -m "not specific_algos and not sim_only"

   # Specific module
   pytest tests/integrators/

   # Specific test pattern
   pytest -k test_solver
```

Reason: the repository shell is PowerShell. The marker set and automatic
parallel, coverage, and report options are defined at `pyproject.toml:95-118`.
The simulator and real-GPU selections must exclude their incompatible marker
sets.

### DG-18: marker and fixture claims are stale

Location: `docs/source/developer_guide/testing.rst:27-60`

Current text:

```rst
``nocudasim``
   Test requires a real GPU; skip in CUDASIM mode.

``cupy``
   Test requires CuPy; skip if not installed.

Key fixtures include pre-built ODE systems (Lotka--Volterra, van der Pol,
linear test systems) and solver factories parameterised by algorithm.
```

Replacement:

```rst
``nocudasim``
   Requires a real GPU and fails under CUDASIM.

``cupy``
   Requires CuPy and fails when it is unavailable.

``slow``
   Marks slow tests. Exclude them with ``-m "not slow"``.

``sim_only``
   Runs only in CUDASIM.

``specific_algos``
   Covers non-default algorithm tableau aliases.

``tests/conftest.py`` supplies session-scoped systems, solver settings,
and tolerance fixtures. Use their default parameter sets unless a test
requires an explicit exception.
```

Reason: marker definitions say these tests fail, not automatically skip
(`pyproject.toml:100-105`). The shared `system` fixture currently provides the
models listed at `tests/conftest.py:303-361`; it does not list Lotka-Volterra
or van der Pol systems.

### DG-19: test rules are framed as philosophy and omit binding constraints

Location: `docs/source/developer_guide/testing.rst:62-72`

Current text:

```rst
Philosophy
----------

- **Failing tests are good.**  Tests should assert intended behaviour.
  If a test fails, fix the code, not the test (unless the test itself is
  wrong).
- **Prefer real objects over mocks.**  Use actual CuBIE objects and real
  (small) ODE systems.  Never patch ``is_device`` checks or CUDA
  availability.
- **Never mark tests ``xfail``** or use ``importorskip`` to hide
  failures.
```

Replacement:

```rst
Test constraints
----------------

- Assert the exact intended behavior. Do not loosen assertions or
  tolerances to accept a failure.
- Use the shared session-scoped fixtures and their default parameter sets.
- Add mocks or patches only with an explicit user exception.
- Do not patch device checks or CUDA availability.
- Do not use ``xfail`` or ``importorskip`` to suppress failures.
- Verify device behavior on a real GPU. CUDASIM results are a first pass.
```

Reason: the current prose qualifies the rule with an undefined exception and
omits the project's fixture, mock, exact-assertion, and real-GPU requirements.
The real fixture surface begins at `tests/conftest.py:303`.

## API reference and batch solving

### API-01: root and batch introductions add framing and mislabel solve_ivp

Locations: `docs/source/API_reference/index.rst:4-6` and
`docs/source/API_reference/batchsolving/index.rst:22-40`

Current text:

```rst
The :mod:`cubie` API is organized into subpackages. The main entry point is
:class:`cubie.batchsolving.Solver` for launching batch runs. The following
pages describe each top-level module inside :mod:`cubie`.

The :class:`Solver` interface is the main entry point. It brings together batch
grids, array managers, GPU program builds, and system details into a coordinated
integration pipeline, while :func:`solve_ivp` offers a shortcut for common
workflows. Supporting modules prepare the GPU programs, describe solver
outputs, and provide data checks used by attrs-based containers.

* :doc:`solve_ivp <solve_ivp>` – convenience wrapper for single-run solver configuration.
```

Replacement:

```rst
:class:`cubie.batchsolving.Solver` configures and launches batched solves.
The module pages document the public package API.

:class:`Solver` owns system configuration, batch-grid construction, memory
managers, compiled kernels, and results. :func:`solve_ivp` constructs a
temporary ``Solver`` and launches the same batched solve path.

* :doc:`solve_ivp <solve_ivp>`. Convenience wrapper for batch solver construction and execution.
```

Reason: `solve_ivp` forwards to `Solver` and accepts batched grids. See
`src/cubie/batchsolving/solver.py:215-315`.

### API-02: the arrays page describes per-launch mirroring that does not occur

Location: `docs/source/API_reference/batchsolving/arrays.rst:23-28`

Current text:

```rst
The ``batchsolving.arrays`` package coordinates host and device array
management for batch solver runs. ``InputArrays`` and ``OutputArrays`` are the
key classes: they collect stride metadata, request GPU allocations via
:mod:`cubie.memory`, and expose helpers for copying data between the CPU and
CUDA kernels. ``OutputArrays`` mirrors requested state, observable, and summary
buffers on the host and GPU for every solver launch.
```

Replacement:

```rst
The ``batchsolving.arrays`` package owns host slots, device slots, shape
metadata, chunk transfers, caller-supplied device inputs, and output-buffer
loans. ``InputArrays`` stages pageable inputs through pinned buffers when
needed. ``OutputArrays`` allocates active output slots and loans completed
host buffers to ``SolveResult`` until the result releases them.
```

Reason: arrays are reused, reallocated only when their signatures change, and
can remain device-only. Output buffers are loaned to results. See
`src/cubie/batchsolving/arrays/BatchOutputArrays.py:311-364` and
`src/cubie/batchsolving/arrays/BatchInputArrays.py:450-507`.

### API-03: batch package public objects are absent from the page tree

Location: `docs/source/API_reference/batchsolving/index.rst:9-49`

Current text:

```rst
.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   solver
   solve_ivp
   solve_result
   solve_spec
   batch_solver_config
   batch_solver_kernel
   system_interface

* :doc:`Solver <solver>` – high-level manager that drives CUDA kernel launches.
* :doc:`solve_ivp <solve_ivp>` – convenience wrapper for single-run solver configuration.
* :doc:`SolveResult <solve_result>` – captures state, summaries, and diagnostic metadata.
* :doc:`SolveSpec <solve_spec>` – checked configuration describing a solver invocation.
```

Replacement:

```rst
   array_types
   batch_input_handler
   device_solve_result

* :doc:`ArrayTypes <array_types>`. Accepted host and device array types.
* :doc:`BatchInputHandler <batch_input_handler>`. Normalizes batch grids and direct array inputs.
* :doc:`DeviceSolveResult <device_solve_result>`. Holds views of the solver's device output buffers and its stream. Synchronize the stream before reading. Copy data that must survive the next solve.

Move ``ActiveOutputs`` from the arrays subtree to the
``BatchSolverConfig`` supporting-infrastructure group.
```

Reason: all three omitted names are public exports at
`src/cubie/batchsolving/__init__.py:54-74`. `DeviceSolveResult` holds device
views and the solver stream but defines no conversion method. Its lifetime and
synchronization contract is at `src/cubie/batchsolving/solveresult.py:925-944`
and its fields are populated at
`src/cubie/batchsolving/solveresult.py:1002-1029`. `ActiveOutputs` resolves to
`src/cubie/batchsolving/BatchSolverConfig.py:35`.

## Integrator reference

### INT-01: package index identifies an internal factory as the primary API

Location: `docs/source/API_reference/integrators/index.rst:17-24`

Current text:

```rst
The :class:`SingleIntegratorRun` interface is the primary entry point. It creates
a loop callable by combining controller, algorithm, and loop submodules as
directed by the supplied :class:`IntegratorRunSettings`.
```

Replacement:

```rst
``cubie.integrators`` contains the compiled components assembled by
:class:`cubie.batchsolving.Solver`. ``SingleIntegratorRun`` combines an
algorithm, controller, output callbacks, and ``IVPLoop`` into the per-run
device function. ``IntegratorRunSettings`` is an internal settings type and
is not exported by ``cubie.integrators``.
```

Reason: the public package export list contains `SingleIntegratorRun` but not
`IntegratorRunSettings`; end users launch integration through the batch solver.
See `src/cubie/integrators/__init__.py:68-93`.

### INT-02: result-code API is named incorrectly

Locations: `docs/source/API_reference/integrators/index.rst:39-40`,
`docs/source/API_reference/integrators/integrator_return_codes.rst:1-8`, and
`docs/source/API_reference/integrators/matrix_free_solvers/solver_ret_codes.rst:1-8`

Current text:

```rst
* :doc:`IntegratorReturnCodes <integrator_return_codes>` – enumerates loop exit statuses.

.. autoclass:: cubie.integrators.IntegratorReturnCodes

.. autoclass:: cubie.integrators.matrix_free_solvers.SolverRetCodes
```

Replacement:

```rst
* :doc:`CUBIE_RESULT_CODES <integrator_return_codes>`. Bit flags reported by loops and matrix-free solvers.

.. autoclass:: cubie.integrators.CUBIE_RESULT_CODES
    :members:
    :member-order: bysource

.. autoclass:: cubie.integrators.matrix_free_solvers.CUBIE_RESULT_CODES
    :members:
    :member-order: bysource
```

Reason: `IntegratorReturnCodes` and `SolverRetCodes` do not exist.
`CUBIE_RESULT_CODES` is an `IntFlag` class defined at
`src/cubie/result_codes.py:19-58` and exported by both packages. See
`src/cubie/integrators/__init__.py:28-36` and
`src/cubie/integrators/matrix_free_solvers/__init__.py:29-42`.

### INT-03: the linear-solver autodoc page points at removed classes

Location: `docs/source/API_reference/integrators/matrix_free_solvers/linear_solver_factory.rst:1-16`

Current text:

```rst
.. autoclass:: LinearSolver
    :members:

.. autoclass:: LinearSolverConfig
    :members:

.. autoclass:: LinearSolverCache
    :members:
```

Replacement:

```rst
.. autoclass:: MRLinearSolver
    :members:
    :show-inheritance:

.. autoclass:: MRLinearSolverConfig
    :members:
    :show-inheritance:

.. autoclass:: BiCGSTABSolver
    :members:
    :show-inheritance:

.. autoclass:: BiCGSTABSolverConfig
    :members:
    :show-inheritance:

.. autoclass:: LinearSolverCache
    :members:
```

Reason: only `LinearSolverCache` still resolves. The current public linear
solvers and configs are exported at
`src/cubie/integrators/matrix_free_solvers/__init__.py:29-42`.

### INT-04: matrix-free overview names removed factories and nonexistent damping

Location: `docs/source/API_reference/integrators/matrix_free_solvers.rst:18-56`

Current text:

```rst
The ``matrix_free_solvers`` package gathers factories that build CUDA device
functions for matrix-free linear and nonlinear solves. These factories are used
by the integrator loops to update implicit states without forming Jacobian
matrices. The solvers rely on :mod:`numba.cuda` for device kernels and perform
warp-synchronisation via lightweight vote helpers.

* :doc:`linear_solver_factory <matrix_free_solvers/linear_solver_factory>` – emits steepest-descent/minimal-residual CUDA
  solvers that operate on matrix-free operators.
* :doc:`newton_krylov_solver_factory <matrix_free_solvers/newton_krylov_solver_factory>` – wraps the linear solver to construct
  damped Newton–Krylov iterations for implicit steps.
* :doc:`SolverRetCodes <matrix_free_solvers/solver_ret_codes>` – enumerates solver completion status codes.

``linear_solver``
^^^^^^^^^^^^^^^^^

Constructs preconditioned steepest-descent and minimal-residual solvers that
operate on matrix-free operators. The factory emits CUDA device functions that
maintain only vector workspaces, which keeps GPU memory usage predictable.

``newton_krylov``
^^^^^^^^^^^^^^^^^

Wraps the linear solver factory to assemble damped Newton–Krylov iterations for
implicit integration steps. The solver encodes completion status using the
:class:`SolverRetCodes` enumeration.
```

Replacement:

```rst
The package builds matrix-free linear and nonlinear CUDA device functions.
Implicit algorithm steps own the solver objects and supply generated
operators, preconditioners, residuals, and scaled norms. The implementations
import CUDA primitives from :mod:`cubie.cuda_simsafe`.

* :doc:`Linear solvers <matrix_free_solvers/linear_solver_factory>`.
  ``MRLinearSolver`` and ``BiCGSTABSolver`` apply matrix-free operators.
* :doc:`NewtonKrylov <matrix_free_solvers/newton_krylov_solver_factory>`.
  Newton iteration delegates each correction to an owned linear solver.
* :doc:`CUBIE_RESULT_CODES <matrix_free_solvers/solver_ret_codes>`.
  Shared status flags returned by device solvers and integration loops.
```

Reason: the package exports MR and BiCGSTAB solvers, and no damping or line
search is implemented. See
`src/cubie/integrators/matrix_free_solvers/__init__.py:29-42` and the Newton
configuration at `src/cubie/integrators/matrix_free_solvers/newton_krylov.py:69-140`.

### INT-05: algorithm-family overview treats Rosenbrock as nonlinear

Location: `docs/source/API_reference/integrators/algorithms.rst:33-39`

Current text:

```rst
Factories in :mod:`cubie.integrators.algorithms` assemble explicit and implicit
step implementations that plug into the CUDA-based integrator loop. Explicit
steps wrap direct right-hand-side evaluations, while implicit steps couple
user-supplied device callbacks with matrix-free Newton–Krylov helpers to
satisfy nonlinear solves. Precision handling is central: every factory
propagates the configured precision through compiled device helpers and the
shared linear and nonlinear solver stack.
```

Replacement:

```rst
Algorithm factories build device functions consumed by the integration
loop. Explicit methods evaluate the right-hand side directly. Backward
Euler, Crank-Nicolson, DIRK, and FIRK solve nonlinear stage equations with
Newton and an owned matrix-free linear solver. Rosenbrock-W methods perform
one matrix-free linear solve per stage without Newton iteration. Each
factory compiles for its configured precision.
```

Reason: Rosenbrock overrides helper construction with a direct linear solver at
`src/cubie/integrators/algorithms/generic_rosenbrock_w.py:307-364`.

### INT-06: adaptive-controller selection is wrong for DIRK

Location: `docs/source/API_reference/integrators/algorithms.rst:98-109`

Current text:

```rst
CuBIE selects the family's tuned controller if the scheme provides
an estimate — a PI controller for explicit Runge–Kutta, the
Gustafsson predictive controller for the implicit families — and a
fixed-step controller if it does not.
This is why ``dirk`` and ``firk`` run fixed-step out of the box:
their default tableaus carry no embedded estimate. Aliases that do
(``radau``, for example) enable the family's adaptive defaults
automatically.
```

Replacement:

```rst
When ``step_controller`` is unset, an estimate-bearing ERK or DIRK
tableau selects a PI controller. Estimate-bearing FIRK and Rosenbrock
tableaus select a Gustafsson controller. A tableau without embedded
weights selects fixed stepping. The default DIRK and FIRK tableaus have no
embedded weights. The ``radau`` FIRK alias has embedded weights and selects
the FIRK adaptive defaults.
```

Reason: DIRK has order-dependent PI defaults at
`src/cubie/integrators/algorithms/generic_dirk.py:71-93`. FIRK uses
Gustafsson at `src/cubie/integrators/algorithms/generic_firk.py:71-84`.

### INT-07: the implicit deadband statement incorrectly includes DIRK

Location: `docs/source/API_reference/integrators/algorithms.rst:111-117`

Current text:

```rst
The implicit family defaults use a 1.0–1.2 deadband (step-size
increases smaller than 20% are skipped; decreases always apply),
matching RADAU5's step-freeze band; the explicit ``erk`` defaults
apply no deadband. Every value can be overridden per solve — see
:doc:`/user_guide/configuration` for how kwargs reach the controller
and :doc:`/user_guide/optional_arguments` for the full parameter
list.
```

Replacement:

```rst
Adaptive FIRK, Rosenbrock, and Crank-Nicolson defaults use a 1.0 to 1.2
deadband and a maximum gain of 8. Adaptive DIRK defaults use a
``1 / 1.2`` to 1.0 deadband and a maximum gain of 10. ERK defaults have
no deadband. Controller keyword arguments override these family defaults.
```

Reason: DIRK values are defined at
`src/cubie/integrators/algorithms/generic_dirk.py:81-91`; FIRK and Rosenbrock
values are at `src/cubie/integrators/algorithms/generic_firk.py:71-79` and
`src/cubie/integrators/algorithms/generic_rosenbrock_w.py:71-79`.

### INT-08: implicit-solver tolerances are derived rather than fixed at 1e-6

Location: `docs/source/API_reference/integrators/algorithms.rst:133-148`

Current text:

```rst
   * - ``newton_atol`` / ``newton_rtol``
     - ``1e-6``
   * - ``newton_max_iters``
     - ``100``
   * - ``krylov_atol`` / ``krylov_rtol``
     - ``1e-6``
   * - ``krylov_max_iters``
     - ``100``
```

Replacement:

```rst
   * - ``newton_atol`` / ``newton_rtol``
     - Controller ``atol`` / ``rtol`` divided by 10 when unset
   * - ``newton_max_iters``
     - ``100``
   * - ``krylov_atol`` / ``krylov_rtol``
     - Controller ``atol`` / ``rtol`` when unset
   * - ``krylov_residual_reduction``
     - Tightest positive controller ``rtol`` when adaptive. Divide it by
       100 for linearly implicit steps. Use machine epsilon for
       non-adaptive or pure-absolute control.
   * - ``krylov_residual_floor``
     - Square root of machine epsilon when unset
   * - ``krylov_max_iters``
     - ``100``
```

Reason: residual-reduction derivation is implemented at
`src/cubie/integrators/SingleIntegratorRunCore.py:408-421`. The other
tolerance defaults are at
`src/cubie/integrators/SingleIntegratorRunCore.py:368-407`. The iteration
default is `100` at
`src/cubie/integrators/matrix_free_solvers/base_solver.py:69-75`, and residual
fallbacks are defined at
`src/cubie/integrators/matrix_free_solvers/linear_solver_base.py:52-63`.

### INT-09: StepCache and get_algorithm_step descriptions are stale

Locations: `docs/source/API_reference/integrators/algorithms.rst:150-155` and
`docs/source/API_reference/integrators/algorithms.rst:179-182`

Current text:

```rst
* :doc:`StepCache <algorithms/step_cache>` – container storing compiled kernels and loop scratch buffers.

* :doc:`get_algorithm_step <algorithms/get_algorithm_step>` – resolves step factories by enum or name.
```

Replacement:

```rst
* :doc:`StepCache <algorithms/step_cache>`. Stores the compiled step function and optional nonlinear-solver function.

* :doc:`get_algorithm_step <algorithms/get_algorithm_step>`. Resolves a registered string alias or a supplied ``ButcherTableau`` instance.
```

Reason: `StepCache` has only `step` and `nonlinear_solver` fields
(`src/cubie/integrators/algorithms/base_algorithm_step.py:685-702`). The
factory accepts strings or `ButcherTableau` instances, not enums
(`src/cubie/integrators/algorithms/__init__.py:190-207`).

### INT-10: registry tables omit current aliases and schemes

Locations:

- `docs/source/API_reference/integrators/algorithms/generic_erk_tableaus.rst:29-49`
- `docs/source/API_reference/integrators/algorithms/generic_dirk_tableaus.rst:28-42`
- `docs/source/API_reference/integrators/algorithms/generic_firk_tableaus.rst:27-32`

Current text:

```rst
   * - ``"heun-21"``
     - Heun's improved Euler method (order 2).
     - [Heun1900]_
   * - ``"ralston-33"``
     - Ralston's third-order method with minimized error constants.
     - [Ralston1962]_
   * - ``"bogacki-shampine-32"``
     - Bogacki--Shampine embedded 3(2) pair with FSAL property.
     - [BogackiShampine1993]_
   * - ``"dormand-prince-54"`` and ``"dopri54"``
     - Dormand--Prince embedded 5(4) pair with FSAL property.
     - [DormandPrince1980]_
   * - ``"classical-rk4"``
     - Classical fourth-order Runge--Kutta scheme.
     - [Kutta1901]_
   * - ``"cash-karp-54"``
     - Cash--Karp embedded 5(4) pair with adaptive control weights.
     - [CashKarp1990]_
   * - ``"fehlberg-45"``
     - Runge--Kutta--Fehlberg embedded 5(4) pair.
     - [Fehlberg1969]_

   * - ``"implicit_midpoint"``
     - Single-stage implicit midpoint rule with symplectic structure.
     - [SanzSerna1988]_
   * - ``"trapezoidal_dirk"``
     - Two-stage trapezoidal (Crank--Nicolson) ESDIRK scheme.
     - [CrankNicolson1947]_
   * - ``"sdirk_2_2"``
     - Alexander's L-stable SDIRK pair with embedded error weights.
     - [Alexander1977]_
   * - ``"l_stable_dirk_3"``
     - Three-stage, third-order L-stable SDIRK scheme with stiff accuracy.
     - [MOOSELStableDirk3]_
   * - ``"l_stable_sdirk_4"``
     - Hairer--Wanner five-stage, fourth-order L-stable SDIRK tableau.
     - [HairerWanner1996]_

   * - ``"firk_gauss_legendre_2"``
     - Two-stage fourth-order Gauss--Legendre scheme with symplectic structure.
     - [HairerLubichWanner2006FIRK]_
   * - ``"radau_iia_5"`` and ``"radau"``
     - Three-stage fifth-order Radau IIA method with stiff accuracy.
     - [HairerWanner1996FIRK]_
```

Replacement:

```rst
Add a row for each registry alias. Group aliases that map to the same tableau.
Document ``dormand-prince-853`` as order 8, ``tsit5`` as the Tsitouras
5(4) pair, ``vern7`` as the Verner 7(6) pair, ``kvaerno3`` and
``kvaerno5`` as their registered ESDIRK tableaus, and
``firk_gauss_legendre_4`` as the four-stage order-eight fixed-step
Gauss-Legendre method.
```

Reason: exact registry keys are at
`src/cubie/integrators/algorithms/generic_erk_tableaus.py:751-771`,
`src/cubie/integrators/algorithms/generic_dirk_tableaus.py:429-438`, and
`src/cubie/integrators/algorithms/generic_firk_tableaus.py:219-224`. The
order-eight FIRK definition is at
`src/cubie/integrators/algorithms/generic_firk_tableaus.py:157-207`.

### INT-11: DIRK and FIRK descriptions imply every tableau is adaptive

Locations: `docs/source/API_reference/integrators/algorithms/generic_dirk_step.rst:6-11`
and `docs/source/API_reference/integrators/algorithms/generic_firk_step.rst:6-11`

Current text:

```rst
:class:`DIRKStep` provides a diagonally implicit Runge--Kutta integrator that
solves stage systems with the cached Newton--Krylov helpers supplied by
:mod:`cubie.integrators.matrix_free_solvers`. The factory consumes
:class:`~cubie.integrators.algorithms.generic_dirk.DIRKTableau` instances,
exposing L-stable SDIRK and ESDIRK schemes for stiff problems while preserving
adaptive error control through embedded weights.

:class:`FIRKStep` provides a fully implicit Runge--Kutta integrator that solves
coupled stage systems with the cached Newton--Krylov helpers supplied by
:mod:`cubie.integrators.matrix_free_solvers`. The factory consumes
:class:`~cubie.integrators.algorithms.generic_firk_tableaus.FIRKTableau`
instances, exposing high-order Gauss--Legendre and Radau IIA schemes for stiff
problems while preserving adaptive error control through embedded weights.
```

Replacement:

```rst
``DIRKStep`` solves each stage with its owned Newton and matrix-free linear
solver. A tableau with embedded weights produces the error estimate used
by adaptive control.
A tableau without embedded weights uses fixed stepping by default.

``FIRKStep`` solves all stages as one coupled Newton system. Radau IIA-5
has embedded weights. The registered Gauss-Legendre tableaus do not and
use fixed stepping by default.
```

Reason: controller selection tests `tableau.has_error_estimate`; the default
DIRK and FIRK tableaus have no `b_hat`. See
`src/cubie/integrators/algorithms/generic_dirk.py:81-99` and
`src/cubie/integrators/algorithms/generic_firk.py:71-84`.

### INT-12: Rosenbrock page claims a Jacobian factorization

Location: `docs/source/API_reference/integrators/algorithms/generic_rosenbrock_step.rst:6-11`

Current text:

```rst
:class:`GenericRosenbrockWStep` provides a linearly implicit Rosenbrock-W
integrator that requires only one Jacobian factorisation per step. The factory
consumes
:class:`~cubie.integrators.algorithms.generic_rosenbrockw_tableaus.RosenbrockTableau`
instances and couples user-supplied device callbacks with matrix-free linear
solves from :mod:`cubie.integrators.matrix_free_solvers`.
```

Replacement:

```rst
:class:`GenericRosenbrockWStep` prepares cached Jacobian auxiliaries once
per step and performs one matrix-free linear solve per stage. It does not
materialize or factor a Jacobian matrix. The factory consumes
:class:`~cubie.integrators.algorithms.generic_rosenbrockw_tableaus.RosenbrockTableau`
instances.
```

Reason: helper construction requests `PREPARE_JAC` metadata and a cached
matrix-free operator, then installs the linear-solver device function. See
`src/cubie/integrators/algorithms/generic_rosenbrock_w.py:307-364`.

### INT-13: suggested controller table does not describe the API or defaults

Location: `docs/source/API_reference/integrators/step_control.rst:69-107`

Current text:

```rst
Suggested controller parameters
-------------------------------

The default proportional, integral, and derivative gains mirror the
recommendations from Söderlind and Wang while matching the guidance in
`OrdinaryDiffEq.jl <https://github.com/SciML/OrdinaryDiffEq.jl>`_. Common
choices include:

.. list-table::
   :header-rows: 1

   * - Controller
     - ``beta1``
     - ``beta2``
     - ``beta3``
   * - basic
     - 1.00
     - 0.00
     - 0
   * - PI42
     - 0.60
     - -0.20
     - 0
   * - PI33
     - 2/3
     - -1/3
     - 0
   * - PI34
     - 0.70
     - -0.40
     - 0
   * - H211PI
     - 1/6
     - 1/6
     - 0
   * - H312PID
     - 1/18
     - 1/9
     - 1/18
```

Replacement:

```rst
Configuration defaults
----------------------

The adaptive base config defaults to ``dt_min=1e-6``, ``dt_max=1.0``,
``min_gain=0.3``, ``max_gain=2.0``, ``safety=0.9``, and a 1.0 to 1.2
deadband. ``PIStepControlConfig`` adds ``kp=0.7`` and ``ki=-0.4``.
``PIDStepControlConfig`` adds ``kd=0.0``. Gustafsson adds
``newton_target_iters=20``. Algorithm-family configs override these base
values at construction.
```

Reason: current config fields use `kp`, `ki`, and `kd`, not beta names. See
`src/cubie/integrators/step_control/adaptive_step_controller.py:84-118`,
`src/cubie/integrators/step_control/adaptive_PI_controller.py:48-69`,
`src/cubie/integrators/step_control/adaptive_PID_controller.py:51-60`, and
`src/cubie/integrators/step_control/gustafsson_controller.py:53-74`.

### INT-14: ODELoopConfig does not itself describe computed layouts

Location: `docs/source/API_reference/integrators/loops.rst:7-12`

Current text:

```rst
Supporting
configuration classes in :mod:`cubie.integrators.loops.ode_loop_config` describe
shared and persistent local memory layouts expected during kernel launches.
```

Replacement:

```rst
``ODELoopConfig`` stores loop dimensions, callbacks, tolerances, and buffer
location settings. The buffer registry derives shared and persistent
layouts from the registered loop and child requirements.
```

Reason: layout computation belongs to `BufferRegistry`; its query and allocator
methods are at `src/cubie/buffer_registry.py:948-997` and
`src/cubie/buffer_registry.py:1079-1110`.

### INT-15: exported integrator API and supporting capabilities lack pages

Locations: `docs/source/API_reference/integrators/index.rst:9-40`,
`docs/source/API_reference/integrators/matrix_free_solvers.rst:9-31`, and
`docs/source/API_reference/integrators/algorithms.rst:9-31`

Current text:

```rst
.. toctree::
   :hidden:
   :maxdepth: 2
   :titlesonly:

   single_integrator_run
   integrator_return_codes

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   matrix_free_solvers/linear_solver_factory
   matrix_free_solvers/newton_krylov_solver_factory
   matrix_free_solvers/solver_ret_codes

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   algorithms/base_step_config
   algorithms/base_algorithm_step
   algorithms/step_cache
   algorithms/explicit_step_config
   algorithms/explicit_euler_step
   algorithms/implicit_step_config
   algorithms/backwards_euler_step
   algorithms/backwards_euler_pc_step
   algorithms/crank_nicolson_step
   algorithms/generic_erk_step
   algorithms/generic_erk_tableaus
   algorithms/generic_dirk_step
   algorithms/generic_dirk_tableaus
   algorithms/generic_firk_step
   algorithms/generic_firk_tableaus
   algorithms/generic_rosenbrock_step
   algorithms/generic_rosenbrock_tableaus
   algorithms/get_algorithm_step
```

Replacement:

```rst
Add resolved API entries for ``MRLinearSolver``,
``MRLinearSolverConfig``, ``BiCGSTABSolver``, ``BiCGSTABSolverConfig``,
``CUBIE_RESULT_CODES``, and ``algorithm_is_adaptive``. Add internal
architecture references for ``SingleIntegratorRunCore``,
``IntegratorRunSettings``, scaled norms, dense stage prediction, and
automatic memory heuristics where those settings are explained.
```

Reason: the public exports are listed at `src/cubie/integrators/__init__.py:68-93`
and `src/cubie/integrators/algorithms/__init__.py:31-53`. Current internal
composition also depends on `SingleIntegratorRunCore`, `norms`,
`stage_predictors`, and `memory_heuristics`.

## Punctuation and AI-language sweep for authored RST prose

### STYLE-RST-01: every em dash and semicolon in scoped RST prose

Each row gives the exact current line and exact replacement. RST directive
syntax, citation-title colons, and code tokens are excluded because they are not
authored sentence punctuation.

| Location | Current text | Replacement |
|---|---|---|
| `algorithms.rst:48` | `(``algorithm="erk"``) uses the defaults below; requesting a specific` | `(``algorithm="erk"``) selects the family defaults. A specific` |
| `algorithms.rst:84` | `- Fixed step — the default tableau has no error estimate` | `- Fixed step because the default tableau has no error estimate` |
| `algorithms.rst:89` | `- Fixed step — the default tableau has no error estimate` | `- Fixed step because the default tableau has no error estimate` |
| `algorithms.rst:101` | `an estimate — a PI controller for explicit Runge–Kutta, the` | `an estimate. ERK and DIRK use a PI controller. FIRK and Rosenbrock use the` |
| `algorithms.rst:102` | `Gustafsson predictive controller for the implicit families — and a` | `Gustafsson predictive controller. A scheme without an estimate uses a` |
| `algorithms.rst:112` | `increases smaller than 20% are skipped; decreases always apply),` | `increases smaller than 20% are skipped. Decreases always apply.` |
| `algorithms.rst:113` | `matching RADAU5's step-freeze band; the explicit ``erk`` defaults` | `The explicit ``erk`` defaults` |
| `algorithms.rst:114` | `apply no deadband. Every value can be overridden per solve — see` | `apply no deadband. Controller keyword arguments override these values.` |
| `backwards_euler_pc_step.rst:10` | ``:doc:`BackwardsEulerStep <backwards_euler_step>` — order 1,`` | ``:doc:`BackwardsEulerStep <backwards_euler_step>`. It uses order 1,`` |
| `backwards_euler_pc_step.rst:12` | ``(:ref:`algorithm-defaults`) — adding only an explicit forward-Euler`` | ``(:ref:`algorithm-defaults`). The method adds an explicit forward-Euler`` |
| `backwards_euler_step.rst:10-11` | `backward-Euler update under fixed-step control; the method carries`<br>`no embedded error estimate, so adaptive control is unavailable. Each` | `backward-Euler update under fixed-step control.`<br>`The method carries no embedded error estimate, so adaptive control is unavailable. Each` |
| `crank_nicolson_step.rst:10-11` | `update and runs a backward-Euler companion solve each step; the`<br>`difference between the two supplies an embedded error estimate, so` | `update and runs a backward-Euler companion solve each step.`<br>`The difference between the two supplies an embedded error estimate, so` |
| `generic_dirk_step.rst:18` | `no embedded error estimate, so the default is fixed-step control;` | `no embedded error estimate, so the default is fixed-step control.` |
| `generic_dirk_tableaus.rst:15` | `The :class:\`DIRKStep\` factory defaults to \`\`"l_stable_dirk_3"\`\`—a` | `The :class:\`DIRKStep\` factory defaults to \`\`"l_stable_dirk_3"\`\`, a` |
| `generic_erk_step.rst:16` | `tableau — explicit Dormand–Prince 5(4), seven stages, with an` | `tableau. Dormand-Prince 5(4) is explicit, has seven stages, and provides an` |
| `generic_erk_step.rst:17` | `embedded fourth-order error estimate — under adaptive PI control` | `embedded fourth-order error estimate. It uses adaptive PI control` |
| `generic_erk_step.rst:20` | `the same scheme; the other named schemes in the` | `the same scheme. The other named schemes in the` |
| `generic_firk_step.rst:21` | `control; tableaus that provide one — ``radau`` (Radau IIA-5), for` | `control. ``radau`` provides an estimate. It is the Radau IIA-5 method and` |
| `generic_firk_step.rst:22` | `example — enable the family's Gustafsson predictive defaults` | `selects the family's Gustafsson predictive defaults` |
| `generic_rosenbrock_step.rst:19` | `0.2–8.0×). Rosenbrock-W methods are linearly implicit —` | `0.2 to 8.0 times). Rosenbrock-W methods are linearly implicit.` |
| `generic_rosenbrock_step.rst:21` | `iteration — so of the solver settings in` | `iteration. Of the solver settings in` |
| `generic_rosenbrock_step.rst:23` | `defaults apply; the ``newton_*`` parameters do not.` | `defaults apply. The ``newton_*`` parameters do not.` |
| `buffer_registry.rst:34` | `precision; buffers that differ (e.g. ``np.int32`` iteration` | `precision. Buffers that differ, such as ``np.int32`` iteration` |
| `codegen.rst:5` | `functions.  SymPy is the parse layer for string and SymPy input; every` | `functions. SymPy parses string and SymPy input. Every` |
| `codegen.rst:7` | ```engine`` package) at the parse boundary, and every later stage —` | ```engine`` package) at the parse boundary. Every later stage, including` |
| `codegen.rst:9` | `hashing, and printing — runs on the IR.  This page describes the` | `hashing and printing, runs on the IR.` |
| `codegen.rst:65` | ```print_cuda_multiple`` renders a list of assignments as source lines;` | ```print_cuda_multiple`` renders a list of assignments as source lines.` |
| `testing.rst:28` | `Test requires a real GPU; skip in CUDASIM mode.` | `Requires a real GPU and fails under CUDASIM.` |
| `testing.rst:31` | `Test requires CuPy; skip if not installed.` | `Requires CuPy and fails when it is unavailable.` |
| `testing.rst:34` | `Long-running test; useful with ``-m "not slow"`` for quick` | `Marks a slow test. Exclude it with ``-m "not slow"``.` |

Factual replacements take precedence where line ranges overlap.

### STYLE-RST-02: prose colons

Rewrite these non-structural uses.

| Location | Current text | Disposition |
|---|---|---|
| `architecture.rst:53-55` | ``update()`` returns two sets: the names... | Merged into DG-09. |
| `codegen.rst:54-57` | `compute phase runs on: nodes are interned...` and `source:` | Merged into DG-15. |
| `batchsolving/arrays.rst:25-28` | `key classes: they collect...` | Merged into API-02. |
| `integrators/algorithms.rst:37-39` | `Precision handling is central: every factory...` | Merged into INT-05. |
| `integrators/algorithms.rst:76` | `Gustafsson predictive: gain clamp...` | `Gustafsson predictive with gain clamp 0.2 to 8.0` |
| `integrators/algorithms.rst:80` | `PI: ``kp=...`` ` | `PI with ``kp=...`` ` |
| `integrators/algorithms.rst:93` | `Gustafsson predictive: gain clamp...` | `Gustafsson predictive with gain clamp 0.2 to 8.0` |
| `integrators/algorithms.rst:104` | `This is why ... out of the box:` | Merged into INT-06. |
| `integrators/algorithms.rst:125` | `linearly implicit: it performs...` | `linearly implicit. It performs...` |

Retain these colon categories because they perform required syntax or introduce
a directly adjacent code block, list, or definition list.

- RST directives, roles, options, and literal-block markers.
- Numpydoc-style field lists rendered through autodoc.
- `adding_algorithms.rst:12,32,44,67` and
  `adding_metrics.rst:19,73,89`, which introduce code or enumerated content.
- `architecture.rst:11,27`, `buffer_registry.rst:12,41,58`, and
  `step_control.rst:75`, which introduce literal, code, or definition lists.
- Citation titles at `generic_dirk_tableaus.rst:65` and
  `generic_firk_tableaus.rst:47`.

### STYLE-FRAME-01: detected AI-language tells in RST prose

| Location | Tell | Disposition |
|---|---|---|
| `developer_guide/index.rst:4-7` | Reader framing | Merged into DG-00. |
| `adding_algorithms.rst:4-6` | “designed to make ... straightforward” | Merged into DG-01. |
| `adding_algorithms.rst:11` | “The simplest case.” | Merged into DG-01. |
| `adding_metrics.rst:4-6` | “This page walks through” | Merged into DG-04. |
| `architecture.rst:4-6` | “Understanding these is essential” | Merged into DG-07. |
| `API_reference/index.rst:4-6` | “main entry point” and “following pages” framing | Merged into API-01. |
| `batchsolving/index.rst:22-26` | “main entry point” and “Supporting modules” framing | Merged into API-01. |
| `batchsolving/arrays.rst:25` | “key classes” intensifier | Merged into API-02. |
| `integrators/index.rst:17-19` | “primary entry point” | Merged into INT-01. |
| `integrators/algorithms.rst:37` | “Precision handling is central” | Merged into INT-05. |
| `integrators/algorithms.rst:104` | “This is why” | Merged into INT-06. |
| `developer_guide/buffer_registry.rst:78-81` | “the main tool” | `Alias buffers whose lifetimes do not overlap to reduce shared-memory use. Register intermediate stage vectors as aliases of the output writeback buffer when their lifetimes do not overlap.` |
| `algorithms/generic_erk_tableaus.rst:6-10` | “human-friendly”, “well-known”, and “without manually specifying” | ``ERK_TABLEAU_REGISTRY`` maps string aliases to ``ERKTableau`` instances. Pass a registered alias to ``get_algorithm_step`` to select its coefficients. |
| `algorithms/generic_erk_tableaus.rst:15-18` | “so existing controller presets continue to work” | `The default ``ERKStep`` tableau is Dormand-Prince 5(4). ``dopri54``, ``rk45``, and ``ode45`` select the same tableau as ``dormand-prince-54``.` |
| `algorithms/generic_rosenbrock_tableaus.rst:6-10` | “human-friendly” and “without manually specifying” | ``ROSENBROCK_TABLEAUS`` maps string aliases to ``RosenbrockTableau`` instances. Pass a registered alias to ``get_algorithm_step`` to select its coefficients. |

No false-drama staccato of the form “That's it” or “That's the tweet” appears in
the scoped RST prose.

## Autodoc-rendered source docstrings

Source locations in the STYLE-DOC tables are relative to `src/cubie`. Paths
beginning with `arrays/` are relative to `src/cubie/batchsolving`, paths
beginning with `algorithms/`, `loops/`, `matrix_free_solvers/`, or
`step_control/` are relative to `src/cubie/integrators`, and unprefixed batch
module names are relative to `src/cubie/batchsolving`.

### STYLE-DOC-01: all seven rendered em-dash lines

| Location | Current text | Replacement |
|---|---|---|
| `src/cubie/batchsolving/BatchSolverKernel.py:845` | `no alternative — there a sub-warp block is a launchability` | `no alternative. A sub-warp block is then a launchability` |
| `src/cubie/batchsolving/BatchSolverKernel.py:1633` | ``ArrayInterpolator.coefficients_shape`` — the exact` | ``ArrayInterpolator.coefficients_shape``, which is the exact` |
| `src/cubie/batchsolving/BatchSolverKernel.py:1635` | `the compiled driver evaluators — so supplied coefficient` | `the compiled driver evaluators. Supplied coefficient` |
| `src/cubie/batchsolving/solver.py:377` | `— everything smaller is pageable RAM the operating system` | `Everything smaller uses pageable RAM that the operating system` |
| `src/cubie/batchsolving/solver.py:685` | ``SolveResult`` owning the solve's host output buffers —` | ``SolveResult`` that owns the solve's host output buffers.` |
| `src/cubie/batchsolving/solveresult.py:198` | `The result takes the solve's host buffers wholesale — no copies.` | `The result takes ownership of the solve's host buffers without copying them.` |
| `src/cubie/batchsolving/solveresult.py:470` | `buffer — no copy, no RAM beyond what the solve already used.` | `buffer without copying it or allocating another in-memory array.` |

Reason: these source docstrings render into the scoped API pages through
`:members:`. The replacements remove the disallowed punctuation without
changing behavior.

### STYLE-DOC-02: all 32 rendered semicolon lines

| Location | Current text | Replacement |
|---|---|---|
| `BatchSolverConfig.py:146` | `occupancy to one block per SM); capping trades spill traffic` | `occupancy to one block per SM). Capping trades spill traffic` |
| `BatchSolverConfig.py:152` | ``ArrayInterpolator.coefficients_shape``; input sizing and` | ``ArrayInterpolator.coefficients_shape``. Input sizing and` |
| `BatchSolverConfig.py:159` | ``{algorithm}_{system name}``; the LTO state is appended as` | ``{algorithm}_{system name}``. The LTO state is appended as` |
| `arrays/BaseArrayManager.py:567` | `Owner's spill threshold; None without an owner.` | `Return the owner's spill threshold or ``None`` without an owner.` |
| `arrays/BaseArrayManager.py:574` | `Owner's spill directory; None without an owner.` | `Return the owner's spill directory or ``None`` without an owner.` |
| `arrays/BaseArrayManager.py:922` | `to their slots and are reused; otherwise the next solve` | `to their slots and are reused. Otherwise the next solve` |
| `arrays/BaseArrayManager.py:1003` | `values are ignored. Used for output arrays, which the kernel` | `values are ignored. Output arrays use this path because the kernel` |
| `BatchSolverKernel.py:575` | `or device arrays are accepted; device arrays are used in` | `or device arrays are accepted. Device arrays are used in` |
| `BatchSolverKernel.py:594` | `so results stay in the device output buffers; the run must` | `so results stay in the device output buffers. The run must` |
| `BatchSolverKernel.py:603` | `memory-manager stream (:attr:`stream`); there is no per-run` | `memory-manager stream (:attr:`stream`). There is no per-run` |
| `BatchSolverKernel.py:839` | `32 kiB footprint (three blocks per SM on CC7* hardware;` | `32 kiB footprint, which fits three blocks per SM on CC7* hardware.` |
| `arrays/BatchInputArrays.py:335` | `supplied on device; otherwise the host array.` | `supplied on device. Otherwise it returns the host array.` |
| `arrays/BatchInputArrays.py:346` | `supplied on device; otherwise the host array.` | `supplied on device. Otherwise it returns the host array.` |
| `solver.py:242` | `parameter names and defaults from a ``parameters`` dict; for` | `parameter names and defaults from a ``parameters`` dict. For` |
| `solver.py:294` | `representations on demand; disk-backed results release their` | `representations on demand. Disk-backed results release their` |
| `solver.py:373` | `Memory configuration; each key may also be a keyword argument.` | `Memory configuration. Each setting accepts a corresponding keyword argument.` |
| `solver.py:375` | `result arrays are disk-backed instead of held in RAM; by` | `result arrays are disk-backed instead of held in RAM. By` |
| `solver.py:381` | `directory); point it at a fast disk for large spilled runs.` | `directory). Use a fast disk for large spilled runs.` |
| `solver.py:383` | `another solver faces a genuine VRAM shortage; the evicted` | `another solver faces a VRAM shortage. The evicted` |
| `solver.py:396` | `calibrated per GPU architecture; cards without a calibrated` | `calibrated per GPU architecture. Cards without a calibrated` |
| `solver.py:398` | `arguments always take precedence; pass ``False`` to keep` | `arguments always take precedence. Pass ``False`` to keep` |
| `solver.py:647` | `Numba) are used in place with no host-to-device transfer;` | `Numba) are used in place with no host-to-device transfer.` |
| `solver.py:677` | `solve ran on; see Notes. Default ``False``.` | `solve ran on. Default ``False``.` |
| `solver.py:689` | `and ``as_pandas`` build RAM representations on demand;` | `and ``as_pandas`` build RAM representations on demand.` |
| `solver.py:1114` | `names set in that run's status word; successful runs are` | `names set in that run's status word. Successful runs are` |
| `solveresult.py:330-331` | `after the result has been garbage collected; while the result`<br>`lives, the next run allocates fresh backing.` | `after the result has been garbage collected.`<br>`Starting another solve before releasing the result allocates fresh backing.` |
| `algorithms/generic_dirk.py:225` | `Request dense stage prediction; ignored when the tableau` | `Request dense stage prediction. The setting is ignored when the tableau` |
| `algorithms/generic_dirk_tableaus.py:97` | `solve from that stage's converged increment; every other` | `solve from that stage's converged increment. Every other` |
| `algorithms/generic_firk.py:224` | `Request dense stage prediction; ignored when the tableau` | `Request dense stage prediction. The setting is ignored when the tableau` |
| `loops/ode_loop_config.py:251` | `(≤ 0.01) is accepted with a warning; larger deviations raise` | `(≤ 0.01) is accepted with a warning. Larger deviations raise` |
| `matrix_free_solvers/newton_krylov.py:135` | `are not included here; access them via solver.newton_atol` | `are not included here. Access them through ``solver.newton_atol``` |
| `step_control/base_step_controller.py:177` | `error norms with it; every controller carries it so implicit` | `error norms with it. Every controller carries it so implicit` |

Reason: all lines render through the 54 resolved autodoc targets. The
replacement sentences preserve the documented behavior.

### STYLE-DOC-03: rendered prose colons and framing tells

Rewrite these non-structural colons.

| Location | Current text | Replacement |
|---|---|---|
| `BatchSolverConfig.py:96` | `Maps OutputCompileFlags to ActiveOutputs:` | `Map output compile flags to active outputs.` |
| `arrays/BaseArrayManager.py:942` | `A live owner keeps its buffers: the loan record is dropped so` | `A live owner keeps its buffers. The loan record is dropped so` |
| `BatchSolverKernel.py:841` | `floored at one warp: profiling shows sub-warp blocks starve` | `floored at one warp. Profiling shows sub-warp blocks starve` |
| `arrays/BatchInputArrays.py:94` | `Memory type for host arrays: "pinned" or "host".` | `Memory type for host arrays. Accepted values are ``"pinned"`` and ``"host"``.` |
| `arrays/BatchInputArrays.py:466` | `method never blocks on the stream: with the pool deep enough,` | `method never blocks on the stream. With a sufficiently deep pool,` |
| `arrays/BatchOutputArrays.py:117` | `Memory type for host arrays: "pinned" or "host".` | `Memory type for host arrays. Accepted values are ``"pinned"`` and ``"host"``.` |
| `solver.py:702` | `Device arrays are used in place: no grid construction and` | `Device arrays are used in place. No grid construction or` |
| `solver.py:708` | ``on_device=True`` returns without synchronizing: buffer` | ``on_device=True`` returns without synchronizing. Buffer` |
| `algorithms/ode_implicitstep.py:105` | `The mass matrix is not an algorithm parameter: it belongs to the` | `The mass matrix belongs to the ODE system.` |
| `matrix_free_solvers/linear_solver_base.py:83` | `against: ``"state"`` ...` | `against. Use ``"state"`` ...` |
| `matrix_free_solvers/newton_krylov.py:134` | `Configuration dictionary. Note: newton_atol and newton_rtol` | `Configuration dictionary. ``newton_atol`` and ``newton_rtol``` |

Retain the other 132 colon-bearing rendered lines. They are Sphinx roles,
numpydoc field declarations, or direct introductions to adjacent lists and
examples. The retained direct list introductions are
`solver.py:412,696`, `SystemInterface.py:71`,
`generic_dirk.py:234,236,239`, `generic_erk.py:200,202,204`,
`generic_firk.py:233,235,237`, and
`generic_rosenbrock_w.py:207,209,212`.

Rewrite these framing tells from rendered docstrings.

| Location | Current text | Replacement |
|---|---|---|
| `arrays/BaseArrayManager.py:414-416` | `This method sets the num_runs attribute to specify the total number of runs in the batch. This value is used during allocation to determine chunking behavior.` | `Set ``num_runs`` on every managed array for allocation and chunking.` |
| `arrays/BatchOutputArrays.py:315-318` | `Create an OutputArrays instance from a solver. Does not allocate arrays, just sets up size specifications.` | `Create output-array metadata from a solver without allocating arrays.` |
| `solveresult.py:326-331` | `Create a SolveResult owning the solver's buffers. The solver's host output buffers are handed to the result without copying. The solver reuses them on its next run only after the result has been garbage collected; while the result lives, the next run allocates fresh backing.` | `Transfer ownership of the solver's host output buffers to a new ``SolveResult`` without copying them. The solver reuses the buffers after the result is collected. Starting another solve before releasing the result allocates new backing.` |
| `SystemInterface.py:139-140` | `The method attempts to update both parameters and states. Updates are applied to whichever :class:\`SystemValues\` object recognizes each key.` | `Apply each recognized key to the matching parameter or state ``SystemValues`` object.` |
| `algorithms/base_algorithm_step.py:708-711` | `The class exposes properties and an ``update`` helper shared by concrete explicit and implicit algorithms. Concrete subclasses implement ``build`` to compile device helpers and provide metadata about resource usage.` | `Provide cache, update, and resource interfaces shared by concrete algorithm steps. Concrete subclasses implement ``build`` and resource metadata.` |
| `BatchSolverKernel.py:1072-1073` | `The method applies updates to the single integrator before refreshing compile-critical settings so the kernel rebuild picks up new metadata.` | `Apply updates to the single integrator, then refresh compile-critical kernel settings.` |

No false-drama staccato was found in the rendered source docstrings.

### STYLE-PROSPECTIVE-01: docstrings for proposed missing API pages

The proposed `BatchInputHandler`, `DeviceSolveResult`, `BiCGSTABSolver`, and
`BiCGSTABSolverConfig` pages would render these additional findings.

| Location | Current text | Replacement |
|---|---|---|
| `BatchInputHandler.py:513` | `Disk-backing size in bytes; ``None`` = the RAM default.` | `Disk-backing size in bytes. ``None`` uses the RAM default.` |
| `BatchInputHandler.py:515` | `Directory for disk-backed arrays; ``None`` = temp dir.` | `Directory for disk-backed arrays. ``None`` uses the temporary directory.` |
| `BatchInputHandler.py:565` | `Disk-backing size in bytes; ``None`` = the RAM default.` | `Disk-backing size in bytes. ``None`` uses the RAM default.` |
| `BatchInputHandler.py:567` | `Directory for disk-backed arrays; ``None`` = temp dir.` | `Directory for disk-backed arrays. ``None`` uses the temporary directory.` |
| `matrix_free_solvers/bicgstab_solver.py:94-98` | `(default) auto-selects: ``"shared"`` when ``n * itemsize`` lies in the measured DRAM-bound window [``SHARED_WITNESS_MIN_BYTES``, ``SHARED_WITNESS_MAX_BYTES``], ``"local"`` otherwise. Pass ``"local"`` or ``"shared"`` to override.` | `The default selects ``"shared"`` when ``n * itemsize`` lies in the measured DRAM-bound window [``SHARED_WITNESS_MIN_BYTES``, ``SHARED_WITNESS_MAX_BYTES``]. It selects ``"local"`` otherwise. Pass either value to override the selection.` |
| `matrix_free_solvers/bicgstab_solver.py:172` | `Uses 5 work vectors: r0_hat, p, v, tmp, s_hat.` | `Uses five work vectors named ``r0_hat``, ``p``, ``v``, ``tmp``, and ``s_hat``.` |

Replace the full `DeviceSolveResult` class opening at
`src/cubie/batchsolving/solveresult.py:928-944`.

Current text:

```python
Returned by :meth:`Solver.solve` when ``on_device=True``. Nothing
is copied to the host: the fields are the solver's device output
buffers plus the kernel's memory-manager stream — the stream
every launch and transfer for that solver runs on. The solve does
not synchronize before returning — buffer contents are valid once
:attr:`stream` has been synchronized, and work queued on that
stream executes in order after the solve.

This class holds handles only; it performs no stream or memory
operations. Callers queue follow-up device work on
:attr:`stream`, and read on the host by synchronizing that stream
first (or run a normal host solve instead).

The handles are views into the solver's working buffers: the next
``solve()`` on the same solver overwrites their contents, and a
reallocation or memory-pressure eviction detaches them from the
solver. Copy anything that must outlive the next solve.
```

Replacement:

```python
Returned by :meth:`Solver.solve` when ``on_device=True``. The fields
hold the solver's device output buffers and the memory-manager stream
used by solver launches and transfers. The solve returns without
synchronizing. Buffer contents are valid after :attr:`stream` is
synchronized. Work queued on that stream executes in order after the
solve.

The object performs no stream or memory operations. Queue follow-up
device work on :attr:`stream`. Synchronize that stream before reading
the buffers on the host. A normal host solve performs host transfers.

The handles are views into the solver's working buffers. The next
``solve()`` on the same solver overwrites their contents. Reallocation
or memory-pressure eviction detaches them from the solver. Copy data
that must survive the next solve.
```

Reason: these four public objects are proposed in API-03 and INT-03, so their
rendered docstrings must pass the same style audit before pages are added. The
current behavior and lifetime contract for `DeviceSolveResult` is defined at
`src/cubie/batchsolving/solveresult.py:925-944`.
