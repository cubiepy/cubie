# Systems and output documentation rewrite proposals

## Scope

This audit covers 51 reStructuredText pages and every source docstring
rendered by their autodoc directives. Existing documentation and source files
remain unchanged.

Issue references use the line numbers in the current working tree.

## Memory

### SOD-001

Location: `docs/source/API_reference/memory.rst:21`

Current text:

```rst
The memory package coordinates GPU allocations across cubie. It exposes the
package-level :class:`~cubie.memory.mem_manager.MemoryManager` through
``default_memmgr`` so integrators can request array buffers and register CUDA
streams without rewriting the coordination code. CuPy is the single device
allocation provider on a real GPU: device arrays come from CuPy's memory
pool and host staging buffers from CuPy's pinned memory pool. Supporting
modules describe allocation requests, track chunked response metadata, and
manage stream groups.
```

Replacement:

```rst
``cubie.memory`` coordinates device and host allocations for solver runs.
``default_memmgr`` is the process-wide
:class:`~cubie.memory.mem_manager.MemoryManager` used by solvers. On a real
GPU, CuPy's asynchronous pool backs Numba device arrays and CuPy's pinned
pool backs pinned host arrays. :class:`ArrayRequest` defines an allocation,
:class:`ArrayResponse` reports its arrays and chunk layout, and
:class:`StreamGroups` assigns registered instances to CUDA streams.
```

Reason: removes framing and the prose colon while naming the asynchronous
device pool and the actual responsibilities. Evidence:
`src/cubie/memory/__init__.py:24`, `src/cubie/memory/__init__.py:35`,
`src/cubie/memory/cupy_emm.py:24`, `src/cubie/memory/array_requests.py:45`,
`src/cubie/memory/array_requests.py:109`,
`src/cubie/memory/stream_groups.py:34`.

### SOD-002

Location: `docs/source/API_reference/memory.rst:33`

Current text:

```rst
``default_memmgr`` creates :class:`cubie.memory.mem_manager.MemoryManager`
with stream grouping ready to configure. Typical callers obtain this
singleton, register their instance identifier, and submit
:class:`cubie.memory.array_requests.ArrayRequest` objects that describe the
arrays they need.
```

Replacement:

```rst
``default_memmgr`` is created at package import. A solver registers itself,
queues labelled :class:`cubie.memory.array_requests.ArrayRequest` objects,
and receives an :class:`cubie.memory.array_requests.ArrayResponse` through
its allocation callback.
```

Reason: removes "ready", "Typical callers", and the unsupported claim that
the singleton creates stream grouping. Registration creates a group stream
on demand. Evidence: `src/cubie/memory/__init__.py:33`,
`src/cubie/memory/__init__.py:35`, `src/cubie/memory/mem_manager.py:725`,
`src/cubie/memory/mem_manager.py:1889`,
`src/cubie/memory/stream_groups.py:77`.

### SOD-003

Locations: `docs/source/API_reference/memory.rst:42`, `:44`, `:49`, `:50`,
`:55`, `:60`

Replace each current list item exactly as follows.

| Current text | Replacement |
| --- | --- |
| `* :doc:\`default_memmgr <memory/default_memmgr>\` – default :class:\`MemoryManager\` instance shared across` plus line 43 | `* :doc:\`default_memmgr <memory/default_memmgr>\` is the process-wide :class:\`MemoryManager\` instance used by solvers.` |
| `* :doc:\`MemoryManager <memory/memory_manager>\` – handles allocation requests and stream registration.` | `* :doc:\`MemoryManager <memory/memory_manager>\` registers owners, allocates arrays, selects host backing, and chunks run-axis allocations.` |
| `* :doc:\`ArrayRequest <memory/array_request>\` – describes requested buffers, precision factories, and chunking.` | `* :doc:\`ArrayRequest <memory/array_request>\` defines shape, dtype, placement, chunk axis, and run count.` |
| `* :doc:\`ArrayResponse <memory/array_response>\` – returns allocated buffers and metadata for callers.` | `* :doc:\`ArrayResponse <memory/array_response>\` contains allocated arrays, chunk count, chunk length, and per-chunk shapes.` |
| `* :doc:\`StreamGroups <memory/stream_groups>\` – assigns host instances to CUDA streams and manages synchronisation policies.` | `* :doc:\`StreamGroups <memory/stream_groups>\` assigns instances to named groups that share dedicated CUDA streams.` |
| `* :doc:\`current_cupy_stream <memory/current_cupy_stream>\` – context manager that binds a Numba stream to a CuPy stream so CuPy allocations and copies stay ordered with the Numba-launched kernel.` | `* :doc:\`current_cupy_stream <memory/current_cupy_stream>\` forwards a non-default Numba stream to CuPy for ordered allocations.` |

Reason: removes six separator en dashes and corrects the stale "precision
factories" and "synchronisation policies" descriptions. Evidence:
`src/cubie/memory/array_requests.py:50`,
`src/cubie/memory/array_requests.py:81`,
`src/cubie/memory/array_requests.py:114`,
`src/cubie/memory/stream_groups.py:35`,
`src/cubie/memory/mem_manager.py:347`,
`tests/memory/test_memmgmt.py:1382`.

### SOD-004

Location: `docs/source/API_reference/memory.rst:65`

Current text:

```rst
The package requires :mod:`numba.cuda` for kernel launch, stream management,
and context access. CuPy is required on a real GPU — it is CuBIE's single
device memory allocator, imported at package import time through
:mod:`cubie.cuda_simsafe`. Under the CUDA simulator (which never touches
device memory) CuPy is not required and the import is skipped.
```

Replacement:

```rst
The active CUDA backend supplies launches, streams, and contexts through
:mod:`cubie.cuda_simsafe`. Real-GPU execution requires CuPy for device and
pinned-host allocation. CUDA simulation does not import CuPy.
```

Reason: removes the only em dash in the scoped RST corpus and avoids naming
`numba.cuda` as the only backend. Evidence: `src/cubie/cuda_simsafe.py:1`,
`src/cubie/memory/cupy_emm.py:19`, `src/cubie/memory/cupy_emm.py:24`,
`src/cubie/memory/cupy_emm.py:94`.

### SOD-005

Location: `docs/source/API_reference/memory/current_cupy_stream.rst:6`

Current text:

```rst
.. autofunction:: current_cupy_stream
```

Replacement:

```rst
.. autoclass:: current_cupy_stream
   :members:
```

Reason: the target is a class context manager, not a function. Runtime
resolution reported `TYPE-MISMATCH expected function`. Evidence:
`src/cubie/memory/mem_manager.py:347`,
`tests/memory/test_memmgmt.py:1391`.

### SOD-006

Location: `src/cubie/memory/mem_manager.py:582`

Current text:

```text
Singleton interface coordinating GPU memory allocation and
stream usage.
```

Replacement:

```text
Coordinate GPU memory allocation and stream usage.
```

Reason: `MemoryManager` does not enforce singleton construction. Only
`default_memmgr` is the process-wide instance by convention. Evidence:
`src/cubie/memory/mem_manager.py:581`, `src/cubie/memory/__init__.py:35`.

### SOD-007

Location: `src/cubie/memory/mem_manager.py:601`

Current text:

```text
The manager accepts ArrayRequest objects and returns ArrayResponse instances
that reference allocated arrays and chunking information. Active mode
enforces per-instance VRAM proportions while passive mode mirrors standard
allocation behaviour using chunking only when necessary.

Construction succeeds without a device: the manager stays unsized and every
decision needing a byte figure reprobes, then raises NoCudaDeviceError if no
device answers.
```

Replacement:

```text
Queued ArrayRequest objects produce ArrayResponse objects containing allocated
arrays and chunk metadata. Active mode applies per-instance VRAM caps.
Passive mode uses device-wide free memory when calculating chunk sizes.

Construction without a CUDA device leaves the manager unsized. Operations
that need a device-memory size reprobe and raise NoCudaDeviceError if the
probe fails.
```

Reason: removes the prose colon and vague "standard allocation" wording.
Evidence: `src/cubie/memory/mem_manager.py:581`,
`src/cubie/memory/mem_manager.py:677`,
`src/cubie/memory/mem_manager.py:1577`,
`tests/memory/test_memmgmt.py:1463`.

### SOD-008

Location: `src/cubie/memory/stream_groups.py:39`

Current text:

```text
groups
    Dictionary mapping group names to lists of instance identifiers. When
    omitted, an empty mapping is created and populated with the "default"
    group.
streams
    Dictionary mapping group names to CUDA streams. When omitted, each
    group, including "default", receives a dedicated stream from
    numba.cuda.stream on first use.
...
Each group has an associated CUDA stream that all instances in the group
share for coordinated operations. The "default" group is created
automatically.
```

Replacement:

```text
groups
    Dictionary mapping group names to instance identifiers. The default is
    an empty mapping.
streams
    Dictionary mapping group names to CUDA streams. A stream is created when
    a group is first requested.
...
Instances in one group share its dedicated CUDA stream. Registering an
instance in the "default" group creates that group and stream on demand.
```

Reason: default construction does not create a `default` group. The current
rendered docstring contradicts the implementation and its explicit regression
test. Evidence: `src/cubie/memory/stream_groups.py:64`,
`src/cubie/memory/stream_groups.py:72`,
`src/cubie/memory/stream_groups.py:158`,
`tests/memory/test_stream_groups.py:18`.

## ODE systems and symbolic generation

### SOD-009

Location: `docs/source/API_reference/odesystems/index.rst:22`

Current text:

```rst
The :func:`create_ODE_system` helper is the main entry point. It consumes
symbolic :mod:`sympy` equations and creates :class:`SymbolicODE` instances that
inherit CUDA compilation behaviour from :class:`cubie.CUDAFactory`.
:class:`BaseODE` sets the abstract requirements, and ``SymbolicODE`` is
currently its concrete implementation. Base classes and data containers expose
the precision-aware metadata required by integrator factories.
```

Replacement:

```rst
:func:`create_ODE_system` accepts equation strings, SymPy equations, or an
explicit Python callable and returns :class:`SymbolicODE`. Symbolic DAE input
is structurally simplified before CUDA source generation. :class:`BaseODE`
defines the CUDA-backed system interface. :class:`ODEData`,
:class:`SystemValues`, and :class:`SystemSizes` hold the values and dimensions
used by integrator factories.
```

Reason: the current text omits strings, callables, and DAE simplification and
uses removable framing. Evidence:
`src/cubie/odesystems/symbolic/symbolicODE.py:102`,
`src/cubie/odesystems/symbolic/parsing/parser.py:651`,
`src/cubie/odesystems/baseODE.py:77`.

### SOD-010

Locations: `docs/source/API_reference/odesystems/index.rst:39` through `:52`

Replace the seven current en-dash list items with:

```rst
* :doc:`create_ODE_system <create_ode_system>` accepts strings, SymPy
  equations, and explicit Python callables.
* :doc:`SymbolicODE <symbolic_ode>` generates and compiles device functions
  for parsed systems.
* :doc:`BaseODE <base_ode>` defines the CUDA-backed system interface.
* :doc:`ODEData <ode_data>` stores states, parameters, constants,
  observables, driver count, precision, and the mass matrix.
* :doc:`SystemValues <system_values>` maps named system values to a packed
  precision-specific array.
* :doc:`SystemSizes <system_sizes>` stores state, observable, parameter,
  constant, and driver counts.
* :doc:`ODECache <ode_cache>` stores compiled right-hand-side and observable
  functions plus lazily generated solver helpers.
```

The rendered `SystemSizes` class docstring also needs this replacement.

Location: `src/cubie/odesystems/ODEData.py:117`

Current text:

```text
This data class is passed to CUDA kernels so they can size device buffers
and shared-memory structures correctly.
```

Replacement:

```text
Integrator, batch-solver, and output-sizing components read these counts from
the ODE system.
```

Reason: removes separator en dashes, corrects the incomplete ODEData,
SystemValues, and SystemSizes descriptions, and removes class and correctness
framing from the rendered `SystemSizes` page. Evidence:
`src/cubie/odesystems/ODEData.py:101`,
`src/cubie/odesystems/ODEData.py:117`,
`src/cubie/odesystems/ODEData.py:131`,
`src/cubie/odesystems/SystemValues.py:44`,
`src/cubie/odesystems/baseODE.py:57`,
`src/cubie/integrators/SingleIntegratorRunCore.py:161`,
`src/cubie/outputhandling/output_sizes.py:274`.

### SOD-011

Location: `docs/source/API_reference/odesystems/index.rst:57`

Current text:

```rst
- :class:`SymbolicODE` subclasses :class:`cubie.CUDAFactory` so integrator loops
  can request compiled CUDA device functions directly.
- Precision handling relies on :mod:`cubie._utils` helpers and
  :mod:`cubie.cuda_simsafe` to provide simulator-safe coercions.
- Generated kernels are consumed by :mod:`cubie.integrators` factories during
  loop construction.
```

Replacement:

```rst
* :class:`SymbolicODE` inherits :class:`BaseODE`, which inherits
  :class:`cubie.CUDAFactory`.
* :mod:`cubie.cuda_simsafe` supplies backend-neutral CUDA symbols.
* Integrator factories consume the compiled system functions and requested
  solver helpers.
```

Reason: states the inheritance chain and helper interface directly. Evidence:
`src/cubie/odesystems/symbolic/symbolicODE.py:213`,
`src/cubie/odesystems/baseODE.py:77`,
`src/cubie/odesystems/baseODE.py:440`.

### SOD-012

Location: `docs/source/API_reference/odesystems/symbolic.rst:9`

Current text:

```rst
The symbolic subpackage implements the SymPy-driven pipeline that generates CUDA
kernels for right-hand-side evaluations and Newton–Krylov helpers. It parses
symbolic systems, emits CUDA ``dxdt`` kernels, and packages metadata required by
:class:`cubie.odesystems.SymbolicODE` so integrator loops can consume compiled
functions directly.

Key helpers
-----------

* ``builders`` – utilities that assemble CUDA source strings from SymPy
  expressions.
* ``codegen`` – orchestrates SymPy code generation for device kernels.
* ``templates`` – shared Jinja templates for kernel emission.
* ``transforms`` – symbolic transformations applied before code generation.

Dependencies
------------

The code generation workflow relies on :mod:`sympy` for symbolic manipulation
and :mod:`numba.cuda` for compiling emitted kernels. Generated code is cached via
:class:`cubie.CUDAFactory` and consumed by :mod:`cubie.integrators` during loop
assembly.
```

Replacement:

```rst
String and SymPy input is converted to the interned expression IR in
``engine`` at the parse boundary. ``parsing`` normalises equations and routes
DAE-shaped systems through ``structural`` simplification. ``codegen`` derives
Jacobian and Jacobian-vector-product IR expressions and renders Python source
for right-hand-side, observable, residual, linear-operator, and preconditioner
device functions. ``generate_jacobian`` returns row-major IR expression rows.
Operator and preconditioner code generators consume Jacobian or
Jacobian-vector-product IR. Codegen does not emit a standalone Jacobian device
function.
:class:`cubie.odesystems.SymbolicODE` stores generated source through
``ODEFile`` and compiles requested functions through the active CUDA backend.

Components
----------

* ``engine`` defines expression nodes, differentiation, substitution, common
  subexpression elimination, dependency ordering, and CUDA printing.
* ``parsing`` accepts strings, SymPy equations, Python callables, and CellML.
* ``structural`` simplifies and tears DAE systems.
* ``codegen`` emits source factories from engine IR.

Dependencies
------------

SymPy is used at the parse boundary. Generated modules import CUDA symbols
from :mod:`cubie.cuda_simsafe`. Source files are stored by ``ODEFile`` under
the configured cache root.
```

Reason: `builders`, `templates`, and `transforms` do not exist. SymPy is no
longer the compute representation, Jinja is not used, and generated modules
import `cuda_simsafe`. The replacement also removes all five en dashes.
Evidence: `src/cubie/odesystems/symbolic/engine/__init__.py:1`,
`src/cubie/odesystems/symbolic/parsing/parser.py:627`,
`src/cubie/odesystems/symbolic/codegen/__init__.py:1`,
`src/cubie/odesystems/symbolic/codegen/jacobian.py:15`,
`src/cubie/odesystems/symbolic/codegen/linear_operators.py:44`,
`src/cubie/odesystems/symbolic/codegen/preconditioners.py:623`,
`src/cubie/odesystems/symbolic/odefile.py:20`.

### SOD-013

Location: `src/cubie/odesystems/symbolic/symbolicODE.py:120`

Current text:

```text
Create a SymbolicODE from SymPy definitions.
```

Replacement:

```text
Create a SymbolicODE from equation strings, SymPy equations, or an explicit
Python callable.
```

Reason: the rendered function summary omits two supported input paths.
Evidence: `src/cubie/odesystems/symbolic/symbolicODE.py:102`,
`src/cubie/odesystems/symbolic/symbolicODE.py:124`,
`src/cubie/odesystems/symbolic/parsing/parser.py:651`.

### SOD-014

Location: `src/cubie/odesystems/symbolic/symbolicODE.py:162`

Current text:

```text
Force MTK-style structural simplification (alias elimination, index
reduction, tearing) before code generation, even for systems already in
explicit form. DAE input — implicit equations (0 = g(...)), higher-order
derivatives, derivative terms inside expressions, and algebraic unknowns —
enables it automatically. Torn systems carry a singular mass matrix and
require an implicit algorithm.
```

Replacement:

```text
Force structural simplification before code generation for an explicit
system. Implicit equations, higher-order derivatives, derivative terms inside
expressions, and algebraic unknowns enable structural simplification without
this option. Torn systems carry a singular mass matrix and require an implicit
algorithm.
```

Reason: removes two em dashes and the parenthetical process list while
preserving the DAE behavior. Evidence:
`src/cubie/odesystems/symbolic/parsing/parser.py:690`,
`src/cubie/odesystems/symbolic/parsing/parser.py:722`.

### SOD-015

Location: `src/cubie/odesystems/symbolic/symbolicODE.py:179`

Current text:

```text
Solver mass matrix for hand-formulated semi-explicit DAEs, paired row-for-row
with the declared state order; None implies identity. Part of the system
definition — fixed at construction; algorithms read it from the system.
Singular matrices require an implicit algorithm. Incompatible with structural
simplification, which derives its own.
```

Replacement:

```text
Solver mass matrix for a hand-formulated semi-explicit DAE, paired row for row
with the declared state order. None selects the identity matrix. The matrix is
fixed at system construction and is read by implicit algorithms. Structural
simplification derives its own mass matrix and cannot be combined with this
argument.
```

Reason: removes one em dash and two semicolons. Evidence:
`src/cubie/odesystems/ODEData.py:194`,
`src/cubie/odesystems/symbolic/symbolicODE.py:192`.

### SOD-016

Locations: `src/cubie/odesystems/symbolic/symbolicODE.py:189` and
`src/cubie/odesystems/symbolic/symbolicODE.py:419`

Current text:

```text
Fully constructed symbolic system ready for compilation.
```

Replacement:

```text
The created symbolic system.
```

Reason: removes empty intensification and readiness framing.

### SOD-017

Location: `src/cubie/odesystems/baseODE.py:188`

Current text:

```text
Compile the dxdt system as a CUDA device function.
...
Cache containing the built dxdt function. Subclasses may add further solver
helpers to this cache as needed.
```

Replacement:

```text
Compile the system's base device functions.
...
Cache containing the right-hand-side and observable functions plus an empty
solver-helper member cache.
```

Reason: `ODECache` contains right-hand-side, observable, and helper members.
Helpers are requested lazily rather than optionally added during `build`.
Evidence: `src/cubie/odesystems/baseODE.py:57`,
`src/cubie/odesystems/symbolic/symbolicODE.py:944`.

## Output handling and metrics

### SOD-018

Location: `docs/source/API_reference/outputhandling/index.rst:25`

Current text:

```rst
The output handling package manages CUDA device callbacks that save state
trajectories and summary calculations from integration loops. It turns loop
settings into checked configuration, builds the device functions through the
CUDA factory tools, and provides sizing helpers so callers can allocate host and
device buffers without repeating dimension logic.
```

Replacement:

```rst
``cubie.outputhandling`` validates requested outputs, compiles device
functions that save time-domain values and summary metrics, and calculates
the buffer dimensions used by batch solvers.
```

Reason: removes package framing and the purpose clause. Evidence:
`src/cubie/outputhandling/output_config.py:199`,
`src/cubie/outputhandling/output_functions.py:125`,
`src/cubie/outputhandling/output_sizes.py:53`.

### SOD-019

Location: `docs/source/API_reference/outputhandling/index.rst:41`

Current text:

```rst
:doc:`OutputFunctions <output_functions>` is the main interface. Create it with loop
dimensions and requested outputs to compile CUDA functions that save solver
state, refresh summary metrics, and write reductions back to host arrays. The
factory keeps an :class:`OutputConfig` instance and rebuilds compiled callbacks
when configuration changes.
```

Replacement:

```rst
:doc:`OutputFunctions <output_functions>` converts loop dimensions and output
requests into :class:`OutputConfig`. Its cached device functions save selected
values, update summary buffers, and write summary results to output arrays.
Changing a compile setting invalidates the cached functions.
```

Reason: summary functions write device output slices, not host arrays. The
replacement names the three function roles. Evidence:
`src/cubie/outputhandling/output_functions.py:100`,
`src/cubie/outputhandling/output_functions.py:245`,
`src/cubie/outputhandling/save_summaries.py:189`.

### SOD-020

Locations: `docs/source/API_reference/outputhandling/index.rst:50` through
`:67`

Replace the ten current en-dash list items with:

```rst
* :doc:`OutputConfig <output_config>` validates requested values, indices,
  output types, summary metrics, cadence, and precision.
* :doc:`OutputCompileFlags <output_compile_flags>` carries boolean flags used
  by compiled output functions.
* :doc:`OutputArrayHeights <output_array_heights>` stores per-array heights.
* :doc:`SingleRunOutputSizes <single_run_output_sizes>` stores single-run
  output shapes.
* :doc:`BatchInputSizes <batch_input_sizes>` stores batch input shapes.
* :doc:`BatchOutputSizes <batch_output_sizes>` stores batch output shapes.
* :doc:`OutputFunctions <output_functions>` compiles the save and summary
  callbacks.
* :doc:`OutputFunctionCache <output_function_cache>` stores the three compiled
  callbacks for one factory configuration.
* :doc:`summary_metrics <summary_metrics>` is the process-wide metric registry.
* :doc:`register_metric <register_metric>` creates a decorator for a specified
  metric registry.
```

Reason: removes ten separator en dashes and corrects the cache and decorator
descriptions. Evidence: `src/cubie/outputhandling/output_functions.py:100`,
`src/cubie/outputhandling/summarymetrics/metrics.py:98`.

### SOD-021

Location: `docs/source/API_reference/outputhandling/index.rst:72`

Current text:

```rst
* Compiles CUDA callables through :class:`cubie.CUDAFactory` and
  :mod:`numba.cuda`.
* Loop buffers and output slices align with expectations from
  :mod:`cubie.integrators.loops` and related algorithm factories.
```

Replacement:

```rst
* :class:`cubie.CUDAFactory` caches compiled output functions.
* :mod:`cubie.cuda_simsafe` supplies the active CUDA backend.
* Integrator loops pass per-run output slices to the compiled functions.
```

Reason: avoids naming `numba.cuda` as the only backend and removes vague
"align with expectations" wording. Evidence:
`src/cubie/outputhandling/output_functions.py:125`,
`src/cubie/outputhandling/save_state.py:25`.

### SOD-022

Location: `docs/source/API_reference/outputhandling/summarymetrics.rst:36`

Current text:

```rst
The ``summarymetrics`` package houses the summary metric registry used by output
handling to accumulate reductions during integration. Importing the package
creates :data:`summary_metrics` and eagerly imports the built-in metrics so that
each registers its CUDA device update and save functions. External packages
extend the system by decorating new metric classes with :func:`register_metric`.
```

Replacement:

```rst
Importing ``cubie.outputhandling.summarymetrics`` creates
:data:`summary_metrics` and imports 18 built-in metric classes. The decorators
instantiate the classes and register each metric object's name, sizes, output
metadata, and object reference. :class:`SummaryMetrics` retrieves update and
save device functions from those objects when the functions are requested. A
custom metric subclasses :class:`SummaryMetric` and uses
``@register_metric(summary_metrics)``.
```

Reason: removes package framing and states the decorator's required registry
argument. Imports register metric instances and metadata. They do not eagerly
retrieve the compiled update and save functions. Evidence:
`src/cubie/outputhandling/summarymetrics/__init__.py:35`,
`src/cubie/outputhandling/summarymetrics/__init__.py:38`,
`src/cubie/outputhandling/summarymetrics/metrics.py:117`,
`src/cubie/outputhandling/summarymetrics/metrics.py:403`,
`src/cubie/outputhandling/summarymetrics/metrics.py:765`,
`src/cubie/outputhandling/summarymetrics/metrics.py:788`,
`tests/outputhandling/summarymetrics/test_summary_metrics.py:672`.

### SOD-023

Locations: `docs/source/API_reference/outputhandling/summarymetrics.rst:45`
through `:49`

Replacement:

```rst
* :doc:`summary_metrics <summary_metrics>` stores registered metric objects and
  dispatches sizes, offsets, labels, units, and device functions.
* :doc:`register_metric <register_metric>` creates a class decorator bound to a
  :class:`SummaryMetrics` registry.
* :doc:`SummaryMetric <summarymetrics/metrics/summary_metric>` defines the
  metric compile settings and update/save function contract.
* :doc:`SummaryMetrics <summarymetrics/metrics/summary_metrics>` dispatches
  requested metrics and applies combined-metric substitution.
* :doc:`MetricFuncCache <summarymetrics/metrics/metric_func_cache>` stores one
  metric's update and save device functions.
```

Current text: the same five list entries use an en dash between each link and
description.

Reason: removes five separator en dashes and replaces generic descriptions
with current responsibilities. Evidence:
`src/cubie/outputhandling/summarymetrics/metrics.py:58`,
`src/cubie/outputhandling/summarymetrics/metrics.py:125`,
`src/cubie/outputhandling/summarymetrics/metrics.py:280`.

### SOD-024

Locations: `docs/source/API_reference/outputhandling/summarymetrics.rst:54`
through `:69`

Replace the current 16 en-dash entries with the following 18 entries:

```rst
* :doc:`Mean <summarymetrics/metrics/mean>` computes the arithmetic mean.
* :doc:`Max <summarymetrics/metrics/max>` computes the maximum.
* :doc:`Min <summarymetrics/metrics/min>` computes the minimum.
* :doc:`RMS <summarymetrics/metrics/rms>` computes the root mean square.
* :doc:`Std <summarymetrics/metrics/std>` computes standard deviation.
* :doc:`MaxMagnitude <summarymetrics/metrics/max_magnitude>` computes the
  maximum absolute value.
* :doc:`Extrema <summarymetrics/metrics/extrema>` computes maximum and minimum.
* :doc:`Peaks <summarymetrics/metrics/peaks>` records local maxima.
* :doc:`NegativePeaks <summarymetrics/metrics/negative_peaks>` records local
  minima.
* :doc:`MeanStd <summarymetrics/metrics/mean_std>` computes mean and standard
  deviation from one buffer.
* :doc:`MeanStdRms <summarymetrics/metrics/mean_std_rms>` computes mean,
  standard deviation, and root mean square from one buffer.
* :doc:`StdRms <summarymetrics/metrics/std_rms>` computes standard deviation
  and root mean square from one buffer.
* :doc:`DxdtMax <summarymetrics/metrics/dxdt_max>` computes the maximum first
  derivative.
* :doc:`DxdtMin <summarymetrics/metrics/dxdt_min>` computes the minimum first
  derivative.
* :doc:`DxdtExtrema <summarymetrics/metrics/dxdt_extrema>` computes both first
  derivative extrema.
* :doc:`D2xdt2Max <summarymetrics/metrics/d2xdt2_max>` computes the maximum
  second derivative.
* :doc:`D2xdt2Min <summarymetrics/metrics/d2xdt2_min>` computes the minimum
  second derivative.
* :doc:`D2xdt2Extrema <summarymetrics/metrics/d2xdt2_extrema>` computes both
  second derivative extrema.
```

Reason: `MeanStd` and `StdRms` are registered built-ins but have no API pages
or overview entries. The replacement removes 16 separator en dashes and the
unsupported intensifier "efficiently". Evidence:
`src/cubie/outputhandling/summarymetrics/__init__.py:47`,
`src/cubie/outputhandling/summarymetrics/mean_std.py:29`,
`src/cubie/outputhandling/summarymetrics/std_rms.py:28`,
`tests/outputhandling/summarymetrics/test_summary_metrics.py:672`.

### SOD-025

Location: `docs/source/API_reference/outputhandling/summarymetrics.rst:74`

Current text:

```rst
* Compiles device functions via :class:`cubie.CUDAFactory` and :mod:`numba.cuda`.
* Consumes save/update cadence configuration from :mod:`cubie.outputhandling`.
```

Replacement:

```rst
* :class:`cubie.CUDAFactory` caches each metric's update and save functions.
* :mod:`cubie.cuda_simsafe` supplies the active CUDA backend.
* :class:`cubie.outputhandling.OutputFunctions` propagates precision and
  summary sampling cadence to registered metrics before compilation.
```

Reason: names the backend abstraction and the actual configuration boundary.
Evidence: `src/cubie/outputhandling/output_functions.py:261`,
`src/cubie/outputhandling/summarymetrics/metrics.py:357`.

### SOD-026

Location: `src/cubie/outputhandling/output_config.py:208`

Current text:

```text
saved_state_indices
    Indices of state variables to save. Defaults to an empty collection that
    resolves to all states.
saved_observable_indices
    Indices of observable variables to save. Defaults to an empty collection
    that resolves to all observables.
```

Replacement:

```text
saved_state_indices
    State indices to save. An empty collection disables state output.
saved_observable_indices
    Observable indices to save. An empty collection disables observable
    output.
```

Reason: empty arrays do not expand to all variables. The property gates output
on a non-empty array. Evidence: `src/cubie/outputhandling/output_config.py:369`,
`src/cubie/outputhandling/output_config.py:374`,
`tests/outputhandling/test_output_config.py:305`.

### SOD-027

Location: `src/cubie/outputhandling/output_config.py:228`

Current text:

```text
Private attributes store numpy arrays so that properties can manage circular
dependencies between index validation and flag updates. The post-initialisation
hook applies default indices, validates bounds, and ensures at least one output
path is active.
```

Replacement:

```text
Post-initialisation derives flags from output_types, validates each index array,
and rejects configurations without an active output.
```

Reason: removes implementation framing and a stale circular-dependency claim.
Evidence: `src/cubie/outputhandling/output_config.py:290`.

### SOD-028

Location: `src/cubie/outputhandling/output_config.py:746`

Current text:

```text
This class method provides a convenient interface for creating OutputConfig
objects from the parameter format used by integrator classes. It handles None
values appropriately by converting them to empty arrays.
```

Replacement:

```text
None index arguments are converted to empty int32 arrays before OutputConfig
construction.
```

Reason: removes convenience framing and the empty qualifier "appropriately".
Evidence: `src/cubie/outputhandling/output_config.py:756`.

### SOD-029

Locations: `src/cubie/outputhandling/output_sizes.py:163`, `:233`, `:287`

| Current text | Replacement |
| --- | --- |
| `This class provides 2D array sizes (time × variable) for output arrays from a single integration run.` | `Shapes use the order time sample by variable.` |
| `This class specifies the sizes of input arrays needed for batch processing, including initial conditions, parameters, and forcing terms.` | `Batch inputs contain initial-state, parameter, and driver-coefficient arrays.` |
| `This class provides 3D array sizes (time × variable × run) for output arrays from batch integration runs.` | `Batch output shapes use the order time sample by variable by run.` |

Reason: removes three class-framing sentences. Evidence:
`src/cubie/outputhandling/output_sizes.py:160`,
`src/cubie/outputhandling/output_sizes.py:230`,
`src/cubie/outputhandling/output_sizes.py:284`.

### SOD-030

Locations: `src/cubie/outputhandling/output_functions.py:216` and `:253`

Current text:

```text
Use this method for coordinated configuration updates alongside other
components by passing silent=True so unrelated keys fall through without
raising.

This method is invoked lazily by CUDAFactory the first time a compiled function
is requested. The resulting cache is reused until configuration settings
change.
```

Replacement:

```text
silent=True ignores keys that are not OutputFunctions settings.

CUDAFactory invokes build on the first compiled-function request and reuses the
result until a compile setting changes.
```

Reason: removes method framing while preserving behavior. Evidence:
`src/cubie/outputhandling/output_functions.py:189`,
`src/cubie/outputhandling/output_functions.py:245`.

### SOD-031

Location: `src/cubie/outputhandling/summarymetrics/metrics.py:484`

Current text:

```text
Invalid metric names trigger a warning and are removed from the returned list.
Combined metrics are automatically substituted when multiple individual
metrics can be computed more efficiently together.
```

Replacement:

```text
Invalid metric names trigger a warning and are removed. A registered combined
metric replaces its complete constituent set.
```

Reason: removes the unqualified efficiency claim and automaticity emphasis.
Evidence: `src/cubie/outputhandling/summarymetrics/metrics.py:410`,
`tests/outputhandling/summarymetrics/test_summary_metrics.py:331`.

### SOD-032

Locations: source metric class docstrings at
`src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py:33`,
`d2xdt2_max.py:33`, `d2xdt2_min.py:33`, `dxdt_extrema.py:33`,
`dxdt_max.py:33`, `dxdt_min.py:33`, `extrema.py:33`,
`mean_std_rms.py:33`, and `std.py:33`

Replace each `Uses ... buffer slots:` or `The metric uses ... buffer slots:`
construction with direct slot assignments. Apply these exact sentence forms:

```text
D2xdt2Extrema uses four buffer slots. Slots 0 and 1 store the two previous
values. Slots 2 and 3 store the maximum and minimum unscaled second
derivatives. The outputs are the maximum and minimum second derivatives.

D2xdt2Max and D2xdt2Min use three buffer slots. Slots 0 and 1 store the two
previous values. Slot 2 stores the unscaled second-derivative extremum.

DxdtExtrema uses three buffer slots. Slot 0 stores the previous value. Slots 1
and 2 store the maximum and minimum unscaled derivatives. The outputs are the
maximum and minimum derivatives.

DxdtMax and DxdtMin use two buffer slots. Slot 0 stores the previous value.
Slot 1 stores the unscaled derivative extremum.

Extrema uses two buffer slots. Slot 0 stores the maximum and slot 1 stores the
minimum. The outputs use the same order.

MeanStdRms uses three buffer slots. They store the shift, the sum of shifted
values, and the sum of squared shifted values.

Std uses three buffer slots. They store the shift, the sum of shifted values,
and the sum of squared shifted values.
```

Current text: the cited docstrings use a colon after the buffer-slot count and,
for combined metrics, a second colon before the output order.

Reason: removes eleven avoidable prose colons without dropping buffer layout.
Evidence: the cited class definitions and
`tests/outputhandling/summarymetrics/test_summary_metrics.py:691`.

## Rendered source-docstring punctuation

### SOD-033

The following rendered public-member sentences contain every remaining em dash
or semicolon in the scoped autodoc corpus. Apply the exact replacements below.

| Location | Current text | Replacement |
| --- | --- | --- |
| `src/cubie/memory/mem_manager.py:851` | `Because each process owns its own manager, this stream is private to the process; synchronizing it never orders against the device-wide default stream.` | `Each process owns its manager and its group stream. Synchronizing the group stream does not order against the device-wide default stream.` |
| `src/cubie/memory/mem_manager.py:1119` | `Garbage-collection finalizers call this instead of deregistering directly: GC runs inside whatever allocation triggered it — possibly while this manager iterates its registry, possibly on the transfer watcher's own thread — so the callback only appends.` | `Garbage-collection finalizers append a teardown instead of deregistering. Collection can run during a registry iteration or on the transfer watcher thread.` |
| `src/cubie/memory/mem_manager.py:1433` | `Directory for "memmap" arrays; None = temp dir.` | `Directory for memory-mapped arrays. None uses the system temporary directory.` |
| `src/cubie/memory/mem_manager.py:1488` | `Disk-backing size in bytes; None = the RAM default.` | `Disk-backing threshold in bytes. None uses 80 percent of total RAM.` |
| `src/cubie/memory/mem_manager.py:1490` | `Permit the "pinned" choice; chunked staging passes False.` | `Permit pinned host allocation. Chunked staging passes False.` |
| `src/cubie/memory/stream_groups.py:210` | `Removing an instance that is in no group is a no-op; the group and its stream are kept for the remaining members.` | `Removing an unregistered instance has no effect. Removing a registered instance keeps its group and stream.` |
| `src/cubie/odesystems/baseODE.py:423` | `The consumer's cache policy, forwarded with every request made through the returned callable. Service context only — it never enters any identity.` | `The consumer's cache policy is forwarded with each request and is excluded from helper identity.` |
| `src/cubie/odesystems/baseODE.py:447` | `Helpers that consume a mass matrix read the system's own mass; the matrix is part of the system definition, not an algorithm parameter.` | `Mass-consuming helpers read the system's mass. The mass matrix is part of the system definition and is not an algorithm parameter.` |
| `src/cubie/odesystems/ODEData.py:307` | `Solver mass matrix; None implies identity.` | `Solver mass matrix. None selects the identity matrix.` |
| `src/cubie/odesystems/symbolic/symbolicODE.py:395` | `DAE input — implicit equations, higher-order derivatives, derivative terms inside expressions, and algebraic unknowns — enables it automatically.` | `Implicit equations, higher-order derivatives, derivative terms inside expressions, and algebraic unknowns enable structural simplification.` |
| `src/cubie/odesystems/symbolic/symbolicODE.py:410` | `Solver mass matrix for hand-formulated semi-explicit DAEs, paired row-for-row with the declared state order; None implies identity. The matrix is part of the system definition: it is fixed at construction and algorithms read it from the system.` | `The solver mass matrix for a hand-formulated semi-explicit DAE is paired row for row with the declared state order. None selects the identity matrix. The matrix is fixed at system construction and read by algorithms.` |
| `src/cubie/odesystems/symbolic/symbolicODE.py:966` | `An exact repeated request returns the same member object; different bindings that share emitted source reuse one generated factory.` | `An identical request returns the same member object. Bindings with the same emitted source reuse one generated factory.` |
| `src/cubie/odesystems/SystemValues.py:221` | `Values stay writable for containers whose stored values are runtime data (parameters, states, observables); the constants container seals fully because constant values are compile-critical.` | `Parameter, state, and observable values remain writable. Constant values are sealed because compilation captures them.` |

Reason: removes all five em dashes and all ten semicolons found in rendered
class and member docstrings. SOD-014 and SOD-015 remove the additional three em
dashes and two semicolons in the rendered `create_ODE_system` docstring.

## Missing pages

### SOD-034

Location: absent API page for `NoCudaDeviceError`

Current text: absent.

Replacement file `docs/source/API_reference/memory/no_cuda_device_error.rst`:

```rst
NoCudaDeviceError
=================

.. currentmodule:: cubie.memory

.. autoclass:: NoCudaDeviceError
```

Add `memory/no_cuda_device_error` to `memory.rst`.

Reason: `NoCudaDeviceError` is in `cubie.memory.__all__` and controls the
documented no-device behavior. Evidence: `src/cubie/memory/__init__.py:37`,
`src/cubie/memory/mem_manager.py:122`,
`tests/memory/test_memmgmt.py:1463`.

### SOD-035

Location: absent API page for `CuPyAsyncNumbaManager`

Current text: absent.

Replacement file `docs/source/API_reference/memory/cupy_async_numba_manager.rst`:

```rst
CuPyAsyncNumbaManager
=====================

.. currentmodule:: cubie.memory.cupy_emm

.. py:class:: CuPyAsyncNumbaManager(context)

   External memory manager available on real-GPU builds. It allocates Numba
   device memory through CuPy's asynchronous memory pool.

   ``CuPyAsyncNumbaManager`` is ``None`` when CUDA simulation is enabled.
```

Add `memory/cupy_async_numba_manager` to `memory.rst`.

Reason: the class is exported on real GPU builds and implements the sole
device allocator. A manual Python-domain declaration remains build-valid under
CUDA simulation, where the symbol is assigned `None` and `autoclass` cannot
resolve a class. Evidence: `src/cubie/memory/__init__.py:24`,
`src/cubie/memory/cupy_emm.py:26`,
`src/cubie/memory/cupy_emm.py:94`.

### SOD-036

Location: absent API page for `load_cellml_model`

Current text: absent.

Replacement file `docs/source/API_reference/odesystems/load_cellml_model.rst`:

```rst
load_cellml_model
=================

.. currentmodule:: cubie.odesystems

.. autofunction:: load_cellml_model
```

Add `load_cellml_model` to the ODE-system toctree and Core API list.

Reason: `load_cellml_model` is a package-level public function and has only a
user-guide mention. Evidence: `src/cubie/odesystems/__init__.py:24`,
`src/cubie/odesystems/symbolic/parsing/cellml.py:184`,
`tests/odesystems/symbolic/test_cellml.py:14`.

Before adding the page, revise every affected part of its newly rendered
docstring.

Location: `src/cubie/odesystems/symbolic/parsing/cellml.py:194`

Current text:

```text
Load a CellML model and return an initialized SymbolicODE system.

This function uses the cellmlmanip library to parse CellML files
and converts them into a ready-to-use SymbolicODE system with all
differential equations and algebraic constraints properly configured.
```

Replacement:

```text
Load a CellML model into a SymbolicODE.
```

Location: `src/cubie/odesystems/symbolic/parsing/cellml.py:221`

Current text:

```text
voltage_variable : str, optional
    Name of the membrane voltage variable used by the singularity
    rewrite. If None while fix_singularities is True, the
    voltage state is auto-detected by name; when none is found a
    UserWarning is issued and the rewrite is skipped. Ignored when
    fix_singularities is False.
```

Replacement:

```text
voltage_variable : str, optional
    Name of the membrane voltage variable used by the singularity
    rewrite. When omitted and fix_singularities is True, state-name
    matching selects the voltage variable. If no match exists, a
    UserWarning is issued and the rewrite is skipped. Ignored when
    fix_singularities is False.
```

Location: `src/cubie/odesystems/symbolic/parsing/cellml.py:233`

Current text:

```text
SymbolicODE
    Initialized ODE system ready for use with solve_ivp.
    State variables are configured with initial values from the
    CellML model, and algebraic equations are set up according
    to the parameters and observables specifications.
```

Replacement:

```text
SymbolicODE
    Symbolic system containing the model equations, selected parameters and
    observables, and CellML initial values.
```

Remove the stale Raises entry at
`src/cubie/odesystems/symbolic/parsing/cellml.py:241`:

```text
Current: ImportError. If cellmlmanip is not installed. Install with pip install cellmlmanip.
Replacement: remove this Raises entry.
```

`cellmlmanip` is vendored and imported unconditionally from
`cubie.vendored`. Evidence:
`src/cubie/odesystems/symbolic/parsing/cellml.py:24`,
`src/cubie/odesystems/symbolic/parsing/cellml.py:36`.

Location: `src/cubie/odesystems/symbolic/parsing/cellml.py:254`

Current text:

```text
Load a CellML model and run a simulation:
```

Replacement:

```text
Load and simulate a CellML model.
```

Replace the Notes block at
`src/cubie/odesystems/symbolic/parsing/cellml.py:269` with:

```text
Notes
-----
CellML differential equations become states. Requested algebraic variables
become observables. Other algebraic expressions remain auxiliaries. CellML
initial values are retained.
```

The existing notes include implementation framing, an unqualified Physiome
compatibility claim, and a claim that the vendored parser handles "complex"
XML. The replacement retains the public conversion behavior. The current
85-line docstring contains ten colons and one semicolon. These replacements
retain eight numpydoc type colons and remove both prose colons and the
semicolon.

### SOD-037

Locations: absent API pages for `MeanStd` and `StdRms`

Current text: absent.

Replacement files:

```rst
MeanStd
-------

.. currentmodule:: cubie.outputhandling.summarymetrics.mean_std

.. autoclass:: MeanStd
   :members:
   :show-inheritance:
```

```rst
StdRms
======

.. currentmodule:: cubie.outputhandling.summarymetrics.std_rms

.. autoclass:: StdRms
   :members:
   :show-inheritance:
```

Add both files to the summary-metrics toctree.

Before adding the pages, remove the prose colon from each newly rendered class
docstring.

Location: `src/cubie/outputhandling/summarymetrics/mean_std.py:34`

Current text:

```text
Uses three buffer slots: shift (first value), sum of shifted values, and
sum of squares of shifted values. The shift technique improves numerical
stability for the variance calculation.

The output array contains [mean, std] in that order.
```

Replacement:

```text
The buffer stores the first value, the sum of shifted values, and the sum of
squared shifted values. Shifting reduces cancellation in the variance
calculation. The output order is mean followed by standard deviation.
```

Location: `src/cubie/outputhandling/summarymetrics/std_rms.py:33`

Current text:

```text
Uses three buffer slots: shift (first value), sum of shifted values, and
sum of squares of shifted values. The shift technique improves numerical
stability for the variance calculation.

The output array contains [std, rms] in that order.
```

Replacement:

```text
The buffer stores the first value, the sum of shifted values, and the sum of
squared shifted values. Shifting reduces cancellation in the variance
calculation. The output order is standard deviation followed by RMS.
```

Reason: both classes are registered built-ins and are tested as combined
metric substitutions. Each current ten-line class docstring contains one
prose colon. The replacements contain none. Evidence:
`src/cubie/outputhandling/summarymetrics/__init__.py:48`,
`src/cubie/outputhandling/summarymetrics/__init__.py:49`,
`src/cubie/outputhandling/summarymetrics/mean_std.py:34`,
`src/cubie/outputhandling/summarymetrics/std_rms.py:33`,
`tests/outputhandling/summarymetrics/test_summary_metrics.py:339`,
`tests/outputhandling/summarymetrics/test_summary_metrics.py:343`.

## Agent-instruction mirror

### SOD-038

Location: `src/cubie/odesystems/symbolic/engine/CLAUDE.md:1`

Current text:

```text
AGENTS.md
```

Current Git mode: `100644` regular file.

Replacement: store the same `AGENTS.md` link target with Git mode `120000`,
matching every other nested `CLAUDE.md` mirror.

Reason: the regular file displays the link target as its entire instruction
document instead of resolving the 80-line engine guidance. Adjacent mirrors
use mode `120000`. The authoritative guidance is
`src/cubie/odesystems/symbolic/engine/AGENTS.md:1` through `:80` and agrees
with the current engine source at
`src/cubie/odesystems/symbolic/engine/expr.py:62`,
`src/cubie/odesystems/symbolic/engine/assignments.py:32`, and
`src/cubie/odesystems/symbolic/engine/printer.py:95`.

## Regular instruction files

### SOD-039

Location: `src/cubie/memory/AGENTS.md:149` through `:154`

Current text lists `test_memmgmt.py` twice and attaches the real-GPU CuPy
qualification to the duplicate filename.

Replacement:

```markdown
### Testing

`tests/memory/` contains `test_array_requests.py`,
`test_chunk_buffer_pool.py`, `test_memmgmt.py`, and
`test_stream_groups.py`. The native-device-allocation and CuPy stream tests
in `test_memmgmt.py` use the `cupy` and `nocudasim` markers. Other real-device
tests use `nocudasim` as required.
```

Reason: gives the exact four-test-module inventory and identifies the marked
tests without duplicating a filename. Evidence:
`tests/memory/test_memmgmt.py:399`,
`tests/memory/test_memmgmt.py:1380`,
`tests/memory/test_array_requests.py:1`,
`tests/memory/test_chunk_buffer_pool.py:1`, and
`tests/memory/test_stream_groups.py:1`.

### SOD-040

Locations: `src/cubie/odesystems/AGENTS.md:6` through `:13`,
`src/cubie/odesystems/AGENTS.md:22`, and
`src/cubie/odesystems/AGENTS.md:124`

Replacement for the affected purpose and key-file claims:

```markdown
`BaseODE(CUDAFactory)` defines the shared system interface and owns the
`CUDAFactory` cache interaction, `ODEData` compile settings, and `SystemValues`
mappings. `SymbolicODE` generates source for `dxdt`, observables, and requested
solver helpers. Jacobian generation returns IR expression rows used by helper
source generators. It does not emit a standalone Jacobian device function.
`ODECache` stores compiled `dxdt`, observables, and solver-helper caches.
`SystemSizes` stores component counts read by integrator, batch-solver, and
output-sizing components.
```

Replacement dependency entry:

```markdown
- `attrs`, `numpy`, and `sympy`. CUDA compilation uses the active backend
  exposed through `cubie.cuda_simsafe`.
```

Reason: removes the nonexistent Jacobian device function, the claim that the
`SystemSizes` object is passed to kernels, and the deprecated-backend-only
dependency description. Evidence:
`src/cubie/odesystems/symbolic/codegen/jacobian.py:233`,
`src/cubie/odesystems/symbolic/codegen/preconditioners.py:623`,
`src/cubie/integrators/SingleIntegratorRunCore.py:161`,
`src/cubie/outputhandling/output_sizes.py:274`, and
`src/cubie/cuda_simsafe.py:1`.

### SOD-041

Location: `src/cubie/odesystems/symbolic/AGENTS.md:117`

Current text:

```markdown
- `sympy`; `numpy` (`float32`, `ndarray`); `numba`/`numba.cuda` (generated header + precision types).
```

Replacement:

```markdown
- `sympy` and `numpy`. Generated modules import CUDA symbols and integer types
  from `cubie.cuda_simsafe`, which exposes the active backend.
```

Reason: generated modules do not import `numba.cuda` directly. Evidence:
`src/cubie/odesystems/symbolic/odefile.py:20` through `:23` and
`src/cubie/cuda_simsafe.py:1`.

### SOD-042

Locations: `src/cubie/odesystems/symbolic/codegen/AGENTS.md:6` through `:21`
and `src/cubie/odesystems/symbolic/codegen/AGENTS.md:145` through `:147`

Replacement purpose text:

```markdown
Source-emitter functions return Python strings defining factories for CUDA
device functions. `jacobian.py` is different. `generate_jacobian` returns
row-major IR expression rows, and `generate_analytical_jvp` returns structured
JVP assignments. Operator and preconditioner source generators consume those
expressions. `SymbolicODE.get_solver_helper()` writes source-emitter output
through `ODEFile`, imports the generated module, and calls the selected
factory.

The emitted callbacks implement the linear operator, nonlinear residual, and
requested preconditioners used by implicit solvers. The operator is the
Jacobian of the residual with respect to the stage increment. Their coefficient
placement follows the equations in the sign-and-coefficient section.
```

Replacement dependency entry:

```markdown
- SymPy is used at the conversion boundary. Emitted modules obtain CUDA
  symbols from `cubie.cuda_simsafe` and compile with the active backend.
```

Reason: not every public `generate_*` returns source, and codegen does not emit
a standalone analytic Jacobian device function. The replacement also removes
deprecated-backend-only wording. Evidence:
`src/cubie/odesystems/symbolic/codegen/jacobian.py:233`,
`src/cubie/odesystems/symbolic/codegen/jacobian.py:292`,
`src/cubie/odesystems/symbolic/codegen/linear_operators.py:44`,
`src/cubie/odesystems/symbolic/codegen/preconditioners.py:40`, and
`src/cubie/odesystems/symbolic/odefile.py:20` through `:23`.

### SOD-043

Locations: `src/cubie/odesystems/symbolic/parsing/AGENTS.md:80` through `:88`,
`src/cubie/odesystems/symbolic/parsing/AGENTS.md:111` through `:115`, and
`src/cubie/odesystems/symbolic/parsing/AGENTS.md:124` through `:127`

Replacement:

```markdown
### CellML

`cellmlmanip` is vendored under `cubie.vendored.cellmlmanip` and imported at
module scope. `load_cellml_model` therefore has no missing-cellmlmanip
`ImportError` path. Numeric Dummy atoms become `sp.Float` or `sp.Integer`.
Algebraic equations with numeric right-hand sides become constants or selected
parameters. Other algebraic equations become observables or auxiliaries.
`CellMLCache` stores at most five configurations per model under the configured
cache root and keys entries by file content and serialized arguments.

### Testing

The parsing tests run without a GPU. CellML tests use the vendored parser and
its core runtime dependencies.

### External dependencies

- `sympy`, `attrs`, and `numpy`. The vendored CellML parser uses the core
  `lxml`, `networkx`, `Pint`, and `rdflib` dependencies.
```

Reason: the instruction still describes the pre-vendoring optional import and
an `ImportError` that cannot occur from an absent external `cellmlmanip`
package. Evidence:
`src/cubie/odesystems/symbolic/parsing/cellml.py:24` through `:36` and
`pyproject.toml:29` through `:37`.

### SOD-044

Location: `src/cubie/outputhandling/AGENTS.md:102` through `:104`

Replacement:

```markdown
### External

- `attrs` and `numpy`. Device-function modules import CUDA symbols and integer
  types from `cubie.cuda_simsafe`.
```

Reason: output device factories do not import `numba.cuda` directly. Evidence:
`src/cubie/outputhandling/save_state.py:19`,
`src/cubie/outputhandling/update_summaries.py:29`, and
`src/cubie/outputhandling/save_summaries.py:29`.

### SOD-045

Locations:
`src/cubie/outputhandling/summarymetrics/AGENTS.md:93` through `:94` and
`src/cubie/outputhandling/summarymetrics/AGENTS.md:112` through `:114`

Current shift description:

```markdown
`buffer[0]` holds the window's first sample as the shift, set when
`current_index == 0`.
```

Replacement:

```markdown
`buffer[0]` is initialized from the first summary sample when
`current_index == 0`. Each save replaces it with the completed window's mean,
which becomes the shift for the next window.
```

Replacement dependency entry:

```markdown
### External

- `numpy`, `attrs`, and `math`. Metric modules import CUDA symbols from
  `cubie.cuda_simsafe`.
```

Reason: the shift is not reset to the first sample of each window. It carries
the previous window's mean after the initial sample. Metric modules also use
the backend-neutral CUDA import. Evidence:
`src/cubie/outputhandling/summarymetrics/std.py:103`,
`src/cubie/outputhandling/summarymetrics/std.py:152`,
`src/cubie/outputhandling/summarymetrics/mean_std.py:104`,
`src/cubie/outputhandling/summarymetrics/mean_std.py:160`, and
`src/cubie/outputhandling/summarymetrics/mean.py:16`.

### SOD-046

Locations: all ten regular instruction files listed in the matching inventory
manifest.

The 1,068-line corpus contains 94 em dashes, 184 semicolons, 99 colons, 36
Markdown bold spans, and nine Markdown italic spans. Proposed normalization:

* Replace all 94 em dashes with sentence boundaries or direct conjunctions.
* Split every semicolon-joined clause. None of the 184 semicolons is required
  by Markdown or an executable code block.
* Remove 83 prose colons. Retain the ten `Parent:` metadata colons and six
  inline-code colons in dictionary, type-annotation, and lambda syntax.
* Remove all 45 emphasis spans. Preserve behavior-changing imperatives such as
  `never call build directly` as unformatted sentences.
* Replace framing such as "the odd one out", "algebraically locked together",
  "the path is live, not disabled", "the expensive step", and "numerically
  critical" with the factual behavior already stated beside it.

The inventory records exact character counts and affected line numbers per
file. No listed empty-framing phrase or false-drama staccato example occurs in
this corpus.

## Retained punctuation

The scoped RST files contain 591 colon characters. All but the prose colon
removed by SOD-001 are required reStructuredText directive, option, or role
syntax. Examples include `.. autoclass::`, `:members:`, and `:doc:`. Removing
them changes document semantics.

The currently rendered source docstrings contain 137 colon characters.
SOD-007, SOD-032, and SOD-033 remove every avoidable prose colon. The retained
colons occur in
reStructuredText roles, numpydoc `name : type` declarations, and literal
dictionary examples such as ``{name: value}``. They carry syntax or type
information.

The three docstrings that would be exposed by SOD-036 and SOD-037 contain 12
colon characters and one semicolon before rewriting. SOD-036 and SOD-037
remove four prose colons and the semicolon. The remaining eight colons are
numpydoc parameter type declarations in `load_cellml_model`.

The 49 en dashes in scoped RST comprise 48 list separators and the
`Newton–Krylov` compound. SOD-003, SOD-010, SOD-012, SOD-020, SOD-023, and
SOD-024 remove all 49. The scoped RST corpus contains no semicolons. The one em
dash is removed by SOD-004.

No scoped RST or rendered target docstring contains the listed empty framing
phrases or false-drama staccato examples. SOD-016, SOD-018, SOD-028, SOD-029,
SOD-030, SOD-031, and SOD-036 remove the other detected readiness,
convenience, automaticity, and efficiency framing. SOD-010 removes class and
correctness framing from `SystemSizes`.

The regular instruction corpus has its own punctuation accounting in SOD-046.
Its 94 em dashes, 184 semicolons, 83 prose colons, and 45 emphasis spans are all
proposed for removal. Sixteen colons remain because they encode parent
metadata or inline code syntax.
