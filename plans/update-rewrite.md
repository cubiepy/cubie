# Implementation plan: `update` distributes, consumers pull products

Base: PRs 938 and 939 merged. Supersedes PR 934.

## 1. Invariants

- A parent's `update` does three things, in a hard-coded order: update each
  child with the incoming dict, merge that child's `products` into the dict
  before the next child sees it, then `update_compile_settings` on itself with
  the same dict. Nothing else: no key picking, no renaming, no size arithmetic.
- `products` is the child's cache, by field name (`CUDAFactory.products`,
  PR 938). Reading it builds the child if its cache is invalid, so a parent's
  `update` is where children rebuild. `build()` reads its own
  `compile_settings` and nothing else; it never calls
  `update_compile_settings`, and nothing calls `_invalidate_cache` or `build`.
- A consumer field and the product that fills it share one name. Where a
  product feeds two consumers the consumers share the name too.
- Everything a child delivers upward is a cache field: compiled device
  functions, derived sizes and flags, and the resolved values of settings other
  components read (`dt`, `is_adaptive`, `mass_flags`, ...). The cache is
  rebuilt only through `update_compile_settings`, so a delivered value is
  stale exactly when the child's config says so.
- A parent's config captures its last child's product with a
  `device_function_field` (`loop_fn` on the run and the kernel), so the
  parent invalidates through the same path as every other consumer.
- Givenness records (inner tolerances, performance keys, timing) are written
  from the dict as it enters the user-facing `update`, before any product is
  merged; derived defaults stay in the run and run between child updates.
- The unrecognised-key check compares the recognised sets against the
  user's keys only; product keys are never user keys.

## 2. Products and consumers

Cache fields per factory. "Consumers" lists who reads the product by name.

| Producer | Cache fields | Consumers |
|---|---|---|
| system (`ODECache`) | `dxdt_fn`, `observables_fn`, `helpers`, `operation_counts`, `get_solver_helper_fn`, `n_states`, `n_parameters`, `n_observables`, `n_drivers`, `mass_flags`, `precision` | step (`dxdt_fn`, `observables_fn`, `get_solver_helper_fn`, `n_states`, `n_drivers`, `precision`); controller (`n_states`, `mass_flags`, `precision`); initialiser (`get_solver_helper_fn`, `n_states`, `mass_flags`, `precision`); output functions (`n_states`, `n_observables`, `precision`); loop (`observables_fn`, `n_*`, `precision`) |
| output functions | `save_state_fn`, `update_summaries_fn`, `save_summaries_fn`, `compile_flags`, `n_counters`, `state_summaries_buffer_height`, `observable_summaries_buffer_height`, `output_array_heights`, `summary_legend_per_variable`, `summary_unit_modifications` | loop (`*_fn`, `compile_flags`, `n_counters`, heights); run cache (`compile_flags`, `output_array_heights`, legends) |
| step | `step_fn`, `nonlinear_solver_fn`, `threads_per_step`, `n_error`, `algorithm_order`, `has_error_estimate`, `is_implicit`, `helper_operation_counts`, `performance_defaults` | controller (`algorithm_order`); loop (`step_fn`, `n_error`); run cache (`threads_per_step`) |
| controller | `step_controller_fn`, `is_adaptive`, `dt`, `dt_min`, `dt_max`, `atol`, `rtol` | step (`is_adaptive`); loop (`step_controller_fn`, `is_adaptive`, `dt`, `dt_min`, `dt_max`); run derivations (`atol`, `rtol`) |
| initialiser | `initialise_state_fn` | loop |
| loop | `loop_fn`, `shared_memory_elements`, `persistent_local_elements` | run config (`loop_fn`); run cache (sizes) |
| run | `loop_fn`, `compile_flags`, `threads_per_step`, `shared_memory_elements`, `persistent_local_elements`, `output_array_heights` | kernel config (`loop_fn`, `compile_flags`); kernel build (sizes) |
| interpolator | `drivers_fn`, `driver_derivative_fn`, `coefficients_shape` (`coefficients` once PR 935 lands) | step and loop (`drivers_fn`, `driver_derivative_fn`); kernel config (`driver_coefficients_shape`, renamed `coefficients_shape`) |
| linear solvers | `linear_solver_fn` | Newton (`linear_solver_fn`); initialiser (`linear_solver_fn`); linearly-implicit step (`linear_solver_fn`) |
| Newton | `nonlinear_solver_fn` | step (`nonlinear_solver_fn`) |
| norms | `norm_fn` | Newton, initialiser, adaptive controller |
| predictor | `predictor_fn` | DIRK and FIRK steps |
| kernel (`BatchSolverCache`) | `solver_kernel`, `launch_geometries`, `duration_counts`, `output_array_heights`, `time_domain_legend`, `summaries_legend` (PR 931) | Solver |

Consumer renames the table needs (mechanical, one PR, same shape as 939):

- `n` becomes `n_states` on the step, controller and initialiser configs and
  in `ALL_ALGORITHM_STEP_PARAMETERS`; `max_states`/`max_observables` on
  `OutputConfig` become `n_states`/`n_observables`. `solver_width` is a
  different quantity and keeps its name; the step and controller keep
  deriving it for their solver and norm children.
- `mass_diagonal_flags` (system property) is delivered as `mass_flags`.
- `controller_order` (step) becomes `algorithm_order`.
- `solver_function` (step) splits into `nonlinear_solver_fn` (Newton child)
  and `linear_solver_fn` (linearly-implicit steps); `build_step` reads the
  one its family uses.
- `driver_coefficients_shape` (kernel) becomes `coefficients_shape`.

One product cannot route by name: the step's `error_solver` is a second
linear solver whose product is also `linear_solver_fn`, consumed as
`error_solver_fn`. Both it and the Newton's inner solver carry
`instance_label="krylov"`, so a label prefix does not separate them either.
The step's `update` writes that one field explicitly:
`{"error_solver_fn": self.error_solver.products["linear_solver_fn"]}`.

### Field helpers

`device_function_field()` (PR 938) stays for device functions. A sibling
`product_field(**kwargs)` declares a hashed config field that a sibling's
product fills (`n_states`, `mass_flags`, `is_adaptive`, `algorithm_order`,
`compile_flags`, buffer heights, `dt`, ...): `metadata={"product": True}`.
`_CubieConfigBase.init_kwargs` skips tagged fields, so `settings_dict` emits
only what the user can give. `_INJECTED_KEYS` on the run goes.

## 3. The run's `update`

```python
def update(self, updates_dict=None, silent=False, **kwargs):
    updates, unpacked = unpack_dict_values(merge(updates_dict, kwargs))
    if not updates:
        return set()
    user_keys = set(updates)
    self._record_givenness(updates)

    recognised = self._system.update(updates, silent=True)
    updates |= self._system.products
    recognised |= self._output_functions.update(updates, silent=True)
    updates |= self._output_functions.products

    recognised |= self._switch_algos(updates)          # constructs the step
    recognised |= self._switch_controllers(updates)    # constructs the controller
    self.check_compatibility()                         # may swap in "fixed"

    recognised |= self._step_controller.update(updates, silent=True)
    updates |= self._step_controller.products          # is_adaptive, dt, atol, rtol
    recognised |= self._algo_step.update(updates, silent=True)
    recognised |= self._apply_algorithm_step_defaults()
    recognised |= self._apply_dae_linear_solve_defaults()
    recognised |= self._apply_inner_tolerance_defaults()
    recognised |= self._apply_performance_defaults()
    updates |= self._algo_step.products                # step_fn, n_error, algorithm_order
    recognised |= self._step_controller.update(updates, silent=True)
    updates |= self._step_controller.products          # step_controller_fn at the final order

    recognised |= self._dae_initialiser.update(updates, silent=True)
    updates |= self._dae_initialiser.products
    self._register_loop_children()
    updates |= self._loop_timing(updates)
    recognised |= self._loop.update(updates, silent=True)
    updates |= self._loop.products
    recognised |= self.update_compile_settings(updates, silent=True)

    unrecognised = user_keys - recognised
    if unrecognised and not silent:
        raise KeyError(...)
    return recognised | unpacked
```

- The controller updates twice: once so the step reads `is_adaptive` (a class
  constant on every controller, so the first pass costs a build only when
  the controller's own settings changed), once with the step's
  `algorithm_order`. Both are the plain child `update`.
- `check_compatibility` runs after both swaps and before any child update, on
  the constructed objects (`has_error_estimate` is a tableau property,
  `is_adaptive` a class constant). Today it runs last, after the loop was
  already given `is_adaptive=True`; the replacement controller reaches the
  loop but the loop's `is_adaptive` does not flip.
- `_apply_algorithm_step_defaults`, `_apply_dae_linear_solve_defaults`,
  `_apply_inner_tolerance_defaults` and `_apply_performance_defaults` stay in
  the run. They read child attributes and push to the step through its
  `update`; they do not touch caches. `_apply_performance_defaults` moves out
  of `build()`: the step's helper operation counts are known after its
  `update` (§4).
- `_register_loop_children` runs once per `update`, after the step's products
  (an implicit step's solver sizes are final once it has built).
- `__init__` constructs the children with their static settings and then runs
  the same routine once, so construction and update wire products identically.
- `build()` returns `SingleIntegratorRunCache(loop_fn=config.loop_fn, ...)`
  with the sizes from `_loop.products` and `_output_functions.products`.
  Deleted: `_step_device_functions`, the `compiled_functions` dict and the
  system-function comparisons in `build()`, `instantiate_loop`'s hand-built
  kwargs, the duplicated `register_child` calls, `_INJECTED_KEYS`, the manual
  `_invalidate_cache`.

## 4. Step and initialiser: helper wiring is an update

`build_implicit_helpers` (implicit steps, with the DIRK and FIRK overrides)
and `build_solver_helpers` (initialiser) request device helpers from the
system's `get_solver_helper_fn`, push them into the solver children, then
write the children's products into their own config. That is an `update`,
not a `build`: it moves into `update`, after the children's own `update` and
before the products merge. `build()` then reads `nonlinear_solver_fn`,
`prepare_jacobian_fn`, `predictor_fn`, `error_solver_fn`, `apply_mass_fn`,
`inverse_mass_dxdt_fn` from its config. The wiring runs when
`get_solver_helper_fn` is set (it arrives as a system product on the run's
first pass); a build without it raises. `__init__` runs the wiring once for
standalone construction. `helper_operation_counts` stays an `eq=False`
config field written by the wiring.

`buffer_registry.update_buffer("cached_auxiliaries", ...)` moves with it.

## 5. The kernel's `update`

```python
recognised = self.driver_interpolator.update(updates, silent=True)
updates |= self.driver_interpolator.products
recognised |= self.single_integrator.update(updates, silent=True)
updates |= self.single_integrator.products
recognised |= buffer_registry.update(self.single_integrator._loop, updates, silent=True)
recognised |= self.update_compile_settings(updates, silent=True)
```

- `BatchSolverConfig.loop_fn` is a `device_function_field`; `build_kernel`
  reads `config.loop_fn`. Construction seeds it through the same routine.
- `_prepare_batch` pushes nothing per solve. With a duration-dependent
  summary schedule it calls `self.update({"duration": duration}, silent=True)`
  so the new `loop_fn` reaches the kernel's config; `duration` is a routed key
  the run's `_loop_timing` consumes, never a config field. Closes #932. The
  `host_overhead` gate row must not move: the update walk runs only on the
  duration-dependent path.
- `configure_drivers` keeps `update_from_dict` for the arrays and then calls
  `self.update(self.driver_interpolator.products, silent=True)`. PR 935
  reshapes this method; whichever lands second adapts.

## 6. Duration-derived summary schedule

- `_loop_timing(updates)` derives the six timing keys from `_user_timing`, the
  output types and a `duration` key in `updates` (`summarise_every =
  duration`, `sample_summaries_every = duration / 100`,
  `summarise_regularly = True`).
- `is_duration_dependent = summary_outputs_requested and
  _user_timing["summarise_every"] is None`.

## 7. Measured costs (main, RTX 4070, mlir backend)

`products` builds eagerly at update time, so this is what each update pays.

| Factory | Cold build | Rebuild after `update(dt=...)` |
|---|---|---|
| system (`dxdt_fn`, codegen) | 31-35 ms; 0.01 ms when the generated file is cached | 34 ms on a constant change (euler); 0.03 ms cached |
| output functions | 0.8-1.5 ms | not invalidated |
| explicit step | 0.5-0.7 ms | not invalidated |
| controller | 0.3-1.0 ms | 0.2-1.0 ms |
| initialiser (no-op) | 0.1-0.3 ms | not invalidated |
| loop | 2.6-7 ms | 2.6-6.7 ms |
| implicit run (step helpers + loop), cold | 54 (backwards_euler), 71 (kvaerno3), 65 (rosenbrock23), 160 (radau) ms | 3.6-6.6 ms |
| kernel object (`build_kernel`, no launch) | 1.3-16 ms | |

The cold costs move from first solve to construction; a settings change costs
one loop rebuild at the update instead of at the next solve. `Solver` and
kernel construction currently issue several sequential updates (`lineinfo`,
`unroll_settings`, `kernel_settings`); each now rebuilds what it touches, so
they collapse into the constructor's single wiring pass.

## 8. Tests

- `test_CUDAFactory.py`: producer/consumer pair; the parent's `update` fills a
  tagged field from `products` by name, invalidates on identity change, stays
  valid otherwise; `product_field` excluded from `init_kwargs`.
- `test_SingleIntegratorRunCore.py`: `update(dt=...)` gives
  `compile_settings.loop_fn is _loop.device_function` and `cache_valid`
  False; equal values keep it valid; `update({"duration": 2.0})` on
  `SUMMARY_ONLY_NO_TIMING` rebuilds, the same duration keeps it valid; an
  errorless algorithm with an adaptive controller leaves the loop's
  `is_adaptive` False after `update`.
- `test_ode_implicitstep.py`: `update` with a new `get_solver_helper_fn`
  fills `nonlinear_solver_fn`; `build` without one raises.
- `test_solver.py`: `SUMMARY_ONLY_NO_TIMING` at two durations, each
  `array_equal` to the explicit schedule; the repeat-solve reuse test
  unchanged; `host_overhead` gate row within threshold.

## 9. PR split

1. `chore(integrators)`: products on every child (cache fields in §2),
   `product_field`, consumer renames, and the step and initialiser helper
   wiring in `update` (§4). Mechanical plus one behavioural move.
2. `fix(integrators)`: the run's `update` (§3, §6); closes #932.
3. `fix(batchsolving)`: the kernel's `update` and `loop_fn` seeding (§5).

Full simulator and GPU suites per PR; the gate on each (kernel rows and the
`host_overhead` wall row).
