# Root and batch supplemental documentation proposals

## Scope

These findings are new after reconciliation with the six existing review
artifacts. Findings already recorded as DG, API, INT, STYLE, UFR, or SOD items
are not repeated here. No source or existing documentation file was changed.

## RB-01 Batch solver config mirror omits fields

Location `src/cubie/batchsolving/AGENTS.md:22`

Current text

```markdown
| `BatchSolverConfig.py` | `BatchSolverConfig(CUDAFactoryConfig)` — holds `precision`, `loop_fn`, `compile_flags`, `driver_coefficients_shape`. Cache policy is **not** a compile setting — it lives with the kernel's `CubieCacheHandler`. `ActiveOutputs(_CubieConfigBase)` — booleans for which output arrays are produced, built via `ActiveOutputs.from_compile_flags(...)`. |
```

Proposed replacement

```markdown
| `BatchSolverConfig.py` | `BatchSolverConfig(CUDAFactoryConfig)` holds `precision`, `loop_fn`, `compile_flags`, `max_registers`, `driver_coefficients_shape`, and `kernel_name`. Cache policy belongs to the kernel's `CubieCacheHandler` and is not a compile setting. `ActiveOutputs(_CubieConfigBase)` stores the enabled output arrays and is built by `ActiveOutputs.from_compile_flags(...)`. |
```

`max_registers` and `kernel_name` are fields on the frozen compile settings.
The current mirror lists neither field. `max_registers` is also accepted by the
public solver route.

Evidence `src/cubie/batchsolving/BatchSolverConfig.py:132-191`,
`src/cubie/batchsolving/solver.py:501-505`, and
`tests/batchsolving/test_solver.py:1546-1556`.

## RB-02 Result mirror assigns one lifecycle to two different classes

Location `src/cubie/batchsolving/AGENTS.md:25`

Current text

```markdown
| `solveresult.py` | `SolveSpec` (attrs config snapshot); `SolveResult` — owns the solve's host buffers via `OutputArrays.loan_host_arrays` (zero copy), applies NaN-on-error masking in place, carries the solve's `stream`, and derives `time`/`time_domain_array`/`summaries_array` plus `as_numpy`/`as_numpy_per_summary`/`as_pandas` lazily; `DeviceSolveResult` — device-array handles to the solve's output buffers plus the kernel's stream, returned by `Solver.solve(on_device=True)` with no D2H copy. Both are pure data containers: no stream or memory operations happen in this module. |
```

Proposed replacement

```markdown
| `solveresult.py` | `SolveSpec` stores the solve configuration. `SolveResult` owns the solve's host buffers through `OutputArrays.loan_host_arrays`, masks failed trajectories when requested, derives combined representations lazily, and releases disk-backed arrays through the memory manager on close, context exit, or collection. `DeviceSolveResult` stores device-array handles and the kernel stream without performing stream or memory operations. |
```

`SolveResult.close()` calls the memory manager to release spill mappings.
`DeviceSolveResult` is the handles-only container. The current sentence applies
the device result's behavior to both classes.

Evidence `src/cubie/batchsolving/solveresult.py:102-111`,
`src/cubie/batchsolving/solveresult.py:522-545`,
`tests/batchsolving/test_solveresult.py:57-78`, and
`tests/batchsolving/test_solver_teardown.py:213-245`.

## RB-03 Rendered timing defaults disagree with signatures

Locations `src/cubie/batchsolving/solver.py:278`,
`src/cubie/batchsolving/solver.py:389`, and
`src/cubie/time_logger.py:194`

Current text

```text
time_logging_level : str or None, default='default'
time_logging_level : str or None, default='default'
verbosity : str or None, default='default'
```

Proposed replacement

```text
time_logging_level : str or None, default=None
time_logging_level : str or None, default=None
verbosity : str or None, default=None
```

The first two lines render through the current `solve_ivp` and `Solver` API
pages. All three signatures default to `None`. `None` disables timing output
until a level is selected.

Evidence `src/cubie/batchsolving/solver.py:215-229`,
`src/cubie/batchsolving/solver.py:420-434`,
`src/cubie/time_logger.py:189-230`,
`tests/batchsolving/test_solver.py:1911-1919`, and
`tests/test_time_logger.py:62-86`.

## RB-04 Configuration index omits accepted memory keys

Location `docs/source/user_guide/configuration.rst:175-177`

Current text

```rst
   * - Memory
     - ``mem_proportion``, ``stream_group``
     - :doc:`memory`
```

Proposed replacement

```rst
   * - Memory
     - ``memory_manager``, ``stream_group``, ``mem_proportion``,
       ``host_spill_threshold``, ``spill_directory``
     - :doc:`memory`
```

Add these definitions to the same page.

```rst
``host_spill_threshold``
   Host arrays larger than this byte count use disk-backed storage. ``None``
   uses the memory manager default.

``spill_directory``
   Existing directory used for disk-backed host arrays. ``None`` uses the
   system temporary directory.
```

UFR-012 already corrects the group count and the cache and kernel rows. It
does not add the three missing memory keys. The solver accepts all five names
through `ALL_MEMORY_MANAGER_PARAMETERS`.

Evidence `src/cubie/memory/mem_manager.py:96-106`,
`src/cubie/batchsolving/solver.py:372-384`,
`src/cubie/batchsolving/solver.py:473-477`,
`tests/batchsolving/test_solver.py:1559-1582`, and
`tests/batchsolving/test_memory_pressure.py:82-166`.

## RB-05 SystemInterface index description assigns CUDA execution work

Location `docs/source/API_reference/batchsolving/index.rst:46-49`

Current text

```rst
* :doc:`SystemInterface <system_interface>` – adapts :class:`cubie.odesystems.baseODE.BaseODE`
  instances for CUDA execution.
```

Proposed replacement

```rst
* :doc:`SystemInterface <system_interface>`. Resolves state, observable, and
  parameter labels and indices from a system's ``SystemValues`` objects and
  merges label-based and index-based output selections.
```

`SystemInterface` does not compile or launch CUDA work. It owns host-side
label, index, and value-routing operations.

Evidence `src/cubie/batchsolving/SystemInterface.py:48-99`,
`src/cubie/batchsolving/SystemInterface.py:166-379`, and
`tests/batchsolving/test_system_interface.py:12-376`.

## RB-06 Top-level timing exports have no API page

Location `docs/source/API_reference/index.rst:8-17`

Current text

```rst
.. toctree::
   :maxdepth: 2
   :caption: Modules
   :titlesonly:

   batchsolving/index
   odesystems/index
   integrators/index
   outputhandling/index
   memory
   gui/index
```

Proposed addition to the toctree

```rst
   time_logger
```

Proposed new page `docs/source/API_reference/time_logger.rst`

```rst
Time logging
============

.. currentmodule:: cubie

.. autoclass:: TimeLogger
   :members:

.. autodata:: default_timelogger
```

Prospective rendered source rewrites for the `:members:` page

| Location | Current text | Proposed replacement |
|---|---|---|
| `src/cubie/time_logger.py:560` | `- 'verbose': Inline timing already printed; category summaries at end` | `- 'verbose': Event timings print inline. Category summaries print at the end.` |
| `src/cubie/time_logger.py:561` | `- 'debug': Individual start/stop already printed; category summaries` | `- 'debug': Individual start and stop messages print during events. Category summaries print at the end.` |
| `src/cubie/time_logger.py:683-685` | `This method is called by CUDAFactory subclasses to register timing events they will track. The category helps organize timing reports by operation type.` | `Register timing events for CUDAFactory subclasses. The category groups timing reports by operation type.` |
| `src/cubie/time_logger.py:716` | `This method centralizes all printing logic in TimeLogger.` | `Messages print when the configured verbosity meets ``min_verbosity``.` |

Apply RB-03 before rendering this page. `TimeLogger` and
`default_timelogger` are top-level public exports. Current prose mentions both
objects but the reference tree contains no definition page.

Evidence `src/cubie/__init__.py:36-49`,
`src/cubie/time_logger.py:189-230`,
`src/cubie/time_logger.py:633-698`,
`src/cubie/time_logger.py:801-803`, and
`tests/test_time_logger.py:59-416`.

## RB-07 Documented direct-use interpolator has no API page

Locations `docs/source/API_reference/index.rst:8-17` and
`docs/source/examples/array_interpolation_example.py:17`

Current text

```python
from cubie.array_interpolator import ArrayInterpolator
```

The API-reference toctree quoted in RB-06 contains no interpolator page.

Proposed addition to the toctree

```rst
   array_interpolator
```

Proposed new page `docs/source/API_reference/array_interpolator.rst`

```rst
Array interpolation
===================

.. currentmodule:: cubie.array_interpolator

.. autoclass:: ArrayInterpolator
   :members: evaluation_function, coefficients, update_from_dict, get_interpolated, plot_interpolated
```

Prospective `update_from_dict` docstring rewrite

Location `src/cubie/array_interpolator.py:203-247`

Current text

```text
## Input dictionary
input_dict fields must include:

    - ``"time"``: 1D float array of sample times corresponding to
    input array values, or
        - ``"driver_sample_period"``: uniform spacing between samples, and
        - ``"t0"``: starting time of the input samples.
    - ``[input_name]``: one-dimensional float array of samples for
    each input, where ``input_name`` is the name of the input signal
    as entered in the system definition.

    Fields may optionally include:

    - ``"order"``: polynomial order for spline interpolation,
    default 3.
    - ``"wrap"``: whether the input should wrap past the final
    value when the last time index is exceeded. When False the
    interpolator clamps to zero before ``t0`` and after the final
    sample.
    - ``"boundary_condition"``: boundary condition for splines.
    Defaults to ``"clamped"`` when ``"wrap"`` is False and to
    ``"periodic"`` when wrapping is enabled.

The input arrays must all be one-dimensional and of the same length.

The final interpolation result is an array of polynomial
coefficients with shape (num_segments, num_inputs, order + 1),
where num_segments is one less than the number of samples provided.

## Interpolation behaviour
If ``"boundary_condition"`` is None, then spline coefficients are
calculated in segments, with no continuity constraints. Otherwise,
the spline coefficients are fit simultaneously for all segments,
and end conditions are enforced according to the boundary condition:

- ``"natural"``: second derivative at the ends of the curve is set
to zero.
- ``"periodic"``: the first and last segments are identical. For
this condition, the first and last samples must match. This is the
default when "wrap" is True, to avoid introducing a discontinuity on
wrap.

These boundary conditions are identical to those in [SciPy's
CubicSpline interpolator]<https://docs.scipy.org/doc/scipy/reference
/generated/scipy.interpolate.CubicSpline.html>
```

Proposed replacement

```rst
Input dictionary
~~~~~~~~~~~~~~~~

Supply one one-dimensional sample array for each input name. All input
arrays must have the same length. The sample count must be at least
``order + 1``.

Choose one sampling-time representation.

``time``
   One-dimensional sample-time array. Its length must match the input
   arrays. Values must be strictly increasing and uniformly spaced.

``driver_sample_period``
   Scalar spacing between samples. Do not supply ``time`` with this
   field.

``t0``
   Start time used with ``driver_sample_period``. The default is ``0.0``.

The remaining fields configure interpolation.

``order``
   Polynomial order. The default is ``3``.

``wrap``
   Repeat the sampled input outside its time range when ``True``. Return
   zero outside the range when ``False``. The default is ``True``.

``boundary_condition``
   One of ``"natural"``, ``"periodic"``, ``"clamped"``, or
   ``"not-a-knot"``. Omission selects ``"periodic"`` when ``wrap`` is
   true and ``"clamped"`` when ``wrap`` is false. Periodic interpolation
   requires matching first and last samples. A non-wrapping clamped input
   adds a zero-valued sample at each end and two transition segments.

The coefficient array has shape ``(num_segments, num_inputs, order + 1)``.
The boundary-condition names follow
`SciPy CubicSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_.
```

The replacement applies UFR-013's timing correction and converts the Markdown
headings, nested list, and link to RST. The example constructs this class
directly and the solver exposes its instance through
`Solver.driver_interpolator`.

Evidence `src/cubie/array_interpolator.py:151-283`,
`src/cubie/array_interpolator.py:625-790`,
`src/cubie/batchsolving/solver.py:608-622`,
`tests/test_array_interpolator.py:432-480`, and
`tests/test_array_interpolator.py:1306-1406`.

## Style and freshness receipt

No additional false-drama staccato or intensifying frame was found in current
rendered batch pages after accounting for STYLE-FRAME-01 and STYLE-DOC-03.
All current em dash, semicolon, and non-structural colon findings in those pages
are already recorded in STYLE-RST-01, STYLE-RST-02, STYLE-DOC-01,
STYLE-DOC-02, and STYLE-DOC-03. The proposed replacement text above introduces
no em dash or semicolon.
