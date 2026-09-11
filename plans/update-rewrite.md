# Implementation plan: `update` distributes, consumers pull products

Base: PR 938 and PR 939 merged. Supersedes PR 934.

## 1. Invariants

- `update`: record givenness from the incoming dict, update each child in order with the dict plus earlier children's products, then `update_compile_settings` on self.
- Consumers pull by name: a `<name>_fn` field fills from the product `<name>_fn`; a size field fills from the same-named cache or settings field of an earlier child.
- Invalidation happens only through `update_compile_settings`; a parent's config captures its last child's product (`loop_fn`).
- Each level declares its own attributes; consumers combine (`uses_error = has_error_estimate and is_adaptive` in the loop).
- Derived values ride `update`; givenness records are written only from the dict as it enters the user-facing `update`.
- `build()` reads its own `compile_settings` only.

## 2. Products and settings per child

| Child | Products (cache fields) | Settings read by consumers |
|---|---|---|
| system | `dxdt_fn`, `observables_fn`, `helpers`, `operation_counts` | `sizes`, `mass_diagonal_flags`, `precision` |
| output functions | `save_state_fn`, `update_summaries_fn`, `save_summaries_fn`, `output_array_heights`, buffer sizes | `compile_flags`, `sample_summaries_every` |
| step | `step_fn`, `nonlinear_solver_fn`, `threads_per_step`, `n_error`, `algorithm_order`, `has_error_estimate`, `is_implicit`, operation counts, `performance_defaults` | `dt`, unroll and placement keys |
| controller | `step_controller_fn`, `is_adaptive` | `dt`, `dt_min`, `dt_max`, `atol`, `rtol` |
| initialiser | `initialise_state_fn` | — |
| loop | `loop_fn`, `shared_memory_elements`, `persistent_local_elements` | timing keys and flags |
| interpolator | `drivers_fn`, `driver_derivative_fn`, `coefficients_shape` | — |
| run | `loop_fn`, `output_compile_flags`, `threads_per_step` | `algorithm`, `step_controller`, timing |

- Properties that are products today become cache fields set in `build()`.
- A product and its consuming field share a name; the step's `controller_order` becomes `algorithm_order`.

## 3. Ordering

- system → output functions → step (swap first) → controller (swap first) → initialiser → loop → self.
- `is_adaptive` no longer reaches the step; the loop combines it with `has_error_estimate`.
- `check_compatibility` runs with `_switch_controllers`, before the controller's `update`.
- The three `register_child` calls run once in the run's `update`, before the loop's `update`.

## 4. The run's `update`

```python
def update(self, updates_dict=None, silent=False, **kwargs):
    updates = flatten(updates_dict, kwargs)
    self._record_givenness(updates)

    recognised = self._system.update(updates, silent=True)
    updates |= self._system.declared()        # n, n_drivers, layout sizes, mass_flags, precision
    recognised |= self._output_functions.update(updates, silent=True)
    updates |= self._output_functions.products

    recognised |= self._switch_algos(updates)
    updates |= self._system.products          # dxdt_fn, observables_fn, helpers
    recognised |= self._algo_step.update(updates, silent=True)
    updates |= self._algo_step.products       # step_fn, nonlinear_solver_fn, n_error,
                                              # threads_per_step, algorithm_order, has_error_estimate

    recognised |= self._switch_controllers(updates)
    recognised |= self._step_controller.update(updates, silent=True)
    updates |= self._step_controller.products | self._step_controller.declared()

    recognised |= self._dae_initialiser.update(updates, silent=True)
    updates |= self._dae_initialiser.products

    self._register_loop_children()
    updates |= self._loop_timing(updates)
    recognised |= self._loop.update(updates, silent=True)

    recognised |= self.update_compile_settings(
        {**updates, "loop_fn": self._loop.device_function}, silent=True
    )
    if not silent and (set(updates_dict) - recognised):
        raise KeyError(...)
    return recognised
```

- `declared()`: per-factory property returning the settings other components read.
- `build()` returns `SingleIntegratorRunCache(loop_fn=self.compile_settings.loop_fn)`; `IntegratorRunSettings` gains `loop_fn = device_function_field()`.
- Deleted: manual `_invalidate_cache()`, the `compiled_functions` dict and system-function comparisons in `build()`, duplicated `register_child` calls, `_step_device_functions()`.

## 5. Derivations hoisted to the Solver

- `_apply_performance_defaults` and `_apply_inner_tolerance_defaults` become `Solver._derived_settings()`, called from `__init__` and `update`: read the run's products and settings, filter by its givenness records, `kernel.update(derived, silent=True)`.
- `auto_performance` moves from `IntegratorRunSettings` to `Solver`.
- `_apply_algorithm_step_defaults` and `_apply_dae_linear_solve_defaults` stay in `_switch_algos`.

## 6. Duration-derived summary schedule

- `_loop_timing(updates)` derives the six timing keys from `_user_timing`, the output types and a `duration` key in `updates` (`summarise_every = duration`, `sample_summaries_every = duration / 100`, `summarise_regularly = True`).
- `duration` is a routed key, never a config field.
- `is_duration_dependent = summary_outputs_requested and _user_timing["summarise_every"] is None`.
- `_prepare_batch`: `if integrator.is_duration_dependent: integrator.update({"duration": duration}, silent=True)`.
- Closes #932.

## 7. The kernel's `update`

- `driver_interpolator.update`, merge its products and `driver_coefficients_shape`, `single_integrator.update`, `buffer_registry.update(loop, ...)`, `update_compile_settings({**updates, "loop_fn": ..., "compile_flags": ...})`.
- `configure_drivers` calls `update` with the interpolator's products.
- `_prepare_batch` pushes `loop_fn` per solve; `build_kernel` reads `config.loop_fn`; construction seeds it.

## 8. Tests

- `test_CUDAFactory.py`: producer/consumer pair; `update` fills a tagged field by name, invalidates on identity change, stays valid otherwise.
- `test_SingleIntegratorRunCore.py`: `update(dt=...)` → `compile_settings.loop_fn is _loop.device_function`, `cache_valid` False; equal values keep it valid; `update({"duration": 2.0})` on `SUMMARY_ONLY_NO_TIMING` rebuilds, same duration keeps it valid.
- `test_solver.py`: `SUMMARY_ONLY_NO_TIMING` at two durations, each `array_equal` to the explicit schedule; `test_repeat_solve_reuses_the_build_state` unchanged.
- Hoisting: derived `unroll_newton_exits` and `newton_atol` on an implicit algorithm; a given `newton_atol` survives `update(atol=...)`.

## 9. PR split

1. `chore(integrators)`: products and `declared()` on the children.
2. `fix(integrators)`: the run's `update` (§3, §4, §6); closes #932.
3. `fix(batchsolving)`: the kernel's `update` and `loop_fn` seeding (§7).
4. `chore(batchsolving)`: defaults derived by the Solver (§5).

Full simulator and GPU suites per PR; gates deferred to the review step.
