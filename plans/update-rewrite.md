# Implementation plan: `update` distributes, consumers pull products

Base: PRs 938 and 939 merged, then the rename PR (§2). Supersedes PR 934.

## 1. Invariants

- A parent's `update`, in a hard-coded order: update each child with the
  dict, merge the child's `products` into the dict, then
  `update_compile_settings` on itself with the same dict. No key picking,
  renaming or size arithmetic.
- `products` is the child's cache by field name (`CUDAFactory.products`);
  reading it builds an invalid cache. `build()` reads its own
  `compile_settings` only; nothing calls `update_compile_settings` from a
  build, `_invalidate_cache`, or `build`.
- A consumer field and the product that fills it share one name.
- Everything a child delivers upward is a cache field: device functions,
  derived sizes and flags, and resolved settings other components read
  (`dt`, `is_adaptive`, `mass_flags`, ...).
- A parent's config captures its last child's product with a
  `device_function_field` (`loop_fn` on the run and the kernel).
- Givenness records are written from the dict as it enters the user-facing
  `update`, before any product is merged. Family, DAE and inner-tolerance
  defaults are the run's; performance defaults are the kernel's (§5).
- The unrecognised-key check uses the user's keys only.

## 2. Products and consumers

Cache fields per factory. "Consumers" lists who reads the product by name.

| Producer | Cache fields | Consumers |
|---|---|---|
| system (`ODECache`) | `dxdt_fn`, `observables_fn`, `helpers`, `operation_counts`, `get_solver_helper_fn`, `n_states`, `n_parameters`, `n_observables`, `n_drivers`, `mass_flags`, `precision` | step (`dxdt_fn`, `observables_fn`, `get_solver_helper_fn`, `n_states`, `n_drivers`, `precision`); controller (`n_states`, `mass_flags`, `precision`); initialiser (`get_solver_helper_fn`, `n_states`, `mass_flags`, `precision`); output functions (`n_states`, `n_observables`, `precision`); loop (`observables_fn`, `n_*`, `precision`) |
| output functions | `save_state_fn`, `update_summaries_fn`, `save_summaries_fn`, `compile_flags`, `n_counters`, `state_summaries_buffer_height`, `observable_summaries_buffer_height`, `output_array_heights`, `summary_legend_per_variable`, `summary_unit_modifications` | loop (`*_fn`, `compile_flags`, `n_counters`, heights); run cache (`compile_flags`, `output_array_heights`, legends) |
| step | `step_fn`, `nonlinear_solver_fn`, `threads_per_step`, `n_error`, `algorithm_order`, `has_error_estimate`, `is_implicit`, `helper_operation_counts`, `performance_defaults` | controller (`algorithm_order`); loop (`step_fn`, `n_error`); run cache (`threads_per_step`, counts and `performance_defaults` for the kernel) |
| controller | `step_controller_fn`, `is_adaptive`, `dt`, `dt_min`, `dt_max`, `atol`, `rtol` | step (`is_adaptive`); loop (`step_controller_fn`, `is_adaptive`, `dt`, `dt_min`, `dt_max`); run derivations (`atol`, `rtol`) |
| initialiser | `initialise_state_fn` | loop |
| loop | `loop_fn`, `shared_memory_elements`, `persistent_local_elements` | run config (`loop_fn`); run cache (sizes) |
| run | `loop_fn`, `compile_flags`, `threads_per_step`, `shared_memory_elements`, `persistent_local_elements`, `output_array_heights`, `operation_counts`, `helper_operation_counts`, `performance_defaults`, `is_implicit` | kernel config (`loop_fn`, `compile_flags`); kernel build (sizes); kernel performance defaults (counts, `performance_defaults`) |
| interpolator | `drivers_fn`, `driver_derivative_fn`, `coefficients_shape` (`coefficients` once PR 935 lands) | step and loop (`drivers_fn`, `driver_derivative_fn`); kernel config (`coefficients_shape`) |
| linear solvers | `linear_solver_fn` | Newton (`linear_solver_fn`); initialiser (`linear_solver_fn`); linearly-implicit step (`linear_solver_fn`) |
| Newton | `nonlinear_solver_fn` | step (`nonlinear_solver_fn`) |
| norms | `norm_fn` | Newton, initialiser, adaptive controller |
| predictor | `predictor_fn` | DIRK and FIRK steps |
| kernel (`BatchSolverCache`) | `solver_kernel`, `launch_geometries`, `duration_counts`, `output_array_heights`, `time_domain_legend`, `summaries_legend` (PR 931) | Solver |

Consumer renames, one mechanical PR stacked on 939:

- `n` becomes `n_states` on the step, controller, initialiser and norm
  configs and in `ALL_ALGORITHM_STEP_PARAMETERS`; `max_states` and
  `max_observables` on `OutputConfig` become `n_states` and
  `n_observables`. `solver_width` keeps its name; the step and controller
  keep deriving it for their solver and norm children.
- `mass_diagonal_flags` (system property) is delivered as `mass_flags`.
- `controller_order` (step) becomes `algorithm_order`.
- `solver_function` (step) splits into `nonlinear_solver_fn` (Newton child)
  and `linear_solver_fn` (linearly-implicit steps); `build_step` reads the
  one its family uses.
- `driver_coefficients_shape` (kernel) becomes `coefficients_shape`.

Open: `error_solver_fn` (the step's second `krylov` linear solver). Either:

1. `MultipleInstanceCUDAFactory.products` prefixes its cache fields with
   `instance_label`; the error solver takes `instance_label="error"` and
   reads `error_*` settings. Consumer fields: `newton_nonlinear_solver_fn`
   (step), `krylov_linear_solver_fn` (Newton, linear step),
   `error_linear_solver_fn` (step), `newton_norm_fn`, `krylov_norm_fn`;
   the controller's unlabelled norm stays `norm_fn`.
2. Products unprefixed; the step's `update` writes
   `{"error_solver_fn": self.error_solver.products["linear_solver_fn"]}`.

### Field helpers

`product_field(**kwargs)` declares a hashed config field a sibling's product
fills (`n_states`, `mass_flags`, `is_adaptive`, `algorithm_order`,
`compile_flags`, buffer heights, `dt`, ...): `metadata={"product": True}`;
`_CubieConfigBase.init_kwargs` skips it. `_INJECTED_KEYS` on the run goes.

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

    recognised |= self._switch_algos(updates)
    recognised |= self._switch_controllers(updates)

    recognised |= self._step_controller.update(updates, silent=True)
    updates |= self._step_controller.products          # is_adaptive, dt, atol, rtol
    recognised |= self._algo_step.update(updates, silent=True)
    recognised |= self._apply_algorithm_step_defaults()
    recognised |= self._apply_dae_linear_solve_defaults()
    recognised |= self._apply_inner_tolerance_defaults()
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

Units:

- `_switch_algos(updates)`: construct a new step when `algorithm` changed;
  merge the family's step and controller defaults into the dict where the
  key is absent.
- `_switch_controllers(updates)`: resolve the effective controller name
  (given name, else gain promotion within `i`/`pi`/`pid`, else the family
  default; `fixed` with a warning when the step has no error estimate),
  construct it when it changed, dropping the gains on a family change.
  `check_compatibility` and `_promote_controller` fold into this unit.
- The controller updates twice: first so the step reads `is_adaptive` (a
  class constant), then with the step's `algorithm_order`.
- The three `_apply_*_defaults` derivations push to the step through its
  `update`.
- `_register_loop_children` runs once per `update`, after the step's products.
- `__init__` constructs the children with their static settings, then runs
  the same routine once.
- `build()` returns `SingleIntegratorRunCache(loop_fn=config.loop_fn, ...)`
  with the sizes and counts from the children's products.
- Delete: `_step_device_functions`, `compiled_functions`,
  `instantiate_loop`'s kwargs, the extra `register_child` calls,
  `_INJECTED_KEYS`, `_invalidate_cache` calls, `_apply_performance_defaults`,
  `auto_performance`, `_user_given_keys`, `optimisation_candidates`.

## 4. Step and initialiser: helper wiring is an update

`build_implicit_helpers` (implicit steps, DIRK and FIRK overrides) and
`build_solver_helpers` (initialiser) move into `update`, after the
children's own `update` and before the products merge: request helpers from
`get_solver_helper_fn`, push them into the solver children, write the
children's products into the config. `build()` reads `nonlinear_solver_fn`,
`prepare_jacobian_fn`, `predictor_fn`, `error_solver_fn`, `apply_mass_fn`,
`inverse_mass_dxdt_fn` from its config. The wiring runs when
`get_solver_helper_fn` is set; a build without it raises. `__init__` runs
the wiring once. `helper_operation_counts` stays an `eq=False` config field
written by the wiring.

`buffer_registry.update_buffer("cached_auxiliaries", ...)` moves with it.

## 5. The kernel's `update`

```python
user_keys = set(updates)
self._record_performance_givenness(updates)
recognised = self.driver_interpolator.update(updates, silent=True)
updates |= self.driver_interpolator.products
recognised |= self.single_integrator.update(updates, silent=True)
updates |= self.single_integrator.products
derived = self._performance_defaults(updates)         # unroll and placement keys
if derived:
    recognised |= self.single_integrator.update(derived, silent=True)
    updates |= self.single_integrator.products
recognised |= buffer_registry.update(self.single_integrator._loop, updates, silent=True)
recognised |= self.update_compile_settings(updates, silent=True)
```

- `BatchSolverConfig.loop_fn` is a `device_function_field`; `build_kernel`
  reads `config.loop_fn`. Construction seeds it through the same routine,
  with `lineinfo`, `unroll_settings` and `kernel_settings` in that one pass.
- Performance defaults are the kernel's: `auto_performance` moves to
  `BatchSolverConfig` (`eq=False`); `_performance_defaults(updates)` reads
  the run's products (`operation_counts`, `helper_operation_counts`,
  `performance_defaults`, `is_implicit`) and `device_hardware()`, filters by
  the kernel's record of user-given performance keys, and returns the unroll
  and placement keys. `optimisation_candidates` and the performance filter
  in `settings_dict` move with it.
- `_prepare_batch` pushes nothing per solve. With summaries requested and
  no `summarise_every` given it calls
  `self.update({"duration": duration}, silent=True)`; the run's
  `_loop_timing` derives `summarise_every=duration`,
  `sample_summaries_every=duration / 100`, `summarise_regularly=True` for
  the loop. `duration` is a routed key, never a config field. Closes #932.
- `configure_drivers` keeps `update_from_dict` for the arrays, then calls
  `self.update(self.driver_interpolator.products, silent=True)`. PR 935
  reshapes this method; whichever lands second adapts.

## 6. Tests

- `test_CUDAFactory.py`: producer/consumer pair; the parent's `update` fills a
  tagged field from `products` by name, invalidates on identity change, stays
  valid otherwise; `product_field` excluded from `init_kwargs`.
- `test_SingleIntegratorRunCore.py`: `update(dt=...)` gives
  `compile_settings.loop_fn is _loop.device_function` and `cache_valid`
  False; equal values keep it valid; `update({"duration": 2.0})` on
  `SUMMARY_ONLY_NO_TIMING` rebuilds, the same duration keeps it valid; an
  errorless algorithm named with an adaptive controller in one `update`
  leaves the loop's `is_adaptive` False.
- `test_ode_implicitstep.py`: `update` with a new `get_solver_helper_fn`
  fills `nonlinear_solver_fn`; `build` without one raises.
- `test_SolverKernel.py`: the performance-default tests from
  `test_performance_defaults.py`, driven through the kernel.
- `test_solver.py`: `SUMMARY_ONLY_NO_TIMING` at two durations, each
  `array_equal` to the explicit schedule; the repeat-solve reuse test
  unchanged; `host_overhead` gate row within threshold.

## 7. PR split

0. `chore`: the consumer renames (§2), stacked on 939.
1. `chore(integrators)`: products on every child (§2), `product_field`, and
   the step and initialiser helper wiring in `update` (§4).
2. `fix(integrators)`: the run's `update` (§3); closes #932 with PR 3.
3. `fix(batchsolving)`: the kernel's `update`, `loop_fn` seeding, the
   duration push and the performance defaults (§5).

Full simulator and GPU suites per PR; the gate on each (kernel rows and the
`host_overhead` wall row).
