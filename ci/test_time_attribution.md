# GPU test-time attribution, 2026-09-07 → 2026-09-30

Baseline: CUDA tests run 34156642070 (5f31d6c2, 2026-09-07). Current: run
36774161837 (ff1d6a11, 2026-09-30). Windows legs only (g5.xlarge every run).
Times are JUnit per-test seconds, summed over the 4 xdist workers, for tests
executed on the GPU leg (banked CPU-runner cases excluded). JUnit time includes
fixture setup, so a session fixture's build lands on the first test using it.

## Pytest step wall time (Windows, mean of cuda12 and cuda13 legs)

| Run date | py3.11 | py3.14 |
|---|---|---|
| 2026-08-17 | 152 s | 149 s |
| 2026-09-07 | 154 s | 148 s |
| 2026-09-11 | 186 s | 158 s |
| 2026-09-16 | 270 s | 208 s |
| 2026-09-30 | 236 s | 164 s |

## Summed per-test time

| Leg | 09-07 | 09-30 | New tests | Existing tests | Removed |
|---|---|---|---|---|---|
| windows-py3.14-cuda12 | 148 s | 319 s | +113 s | +71 s | −13 s |
| windows-py3.11-cuda12 | 246 s | 548 s | +242 s | +90 s | −30 s |

## New tests by introducing commit

| Commit | Tests | py3.14 | py3.11 | Largest (py3.14 / py3.11 s) |
|---|---|---|---|---|
| #927 b7e6a110 auto_performance defaults span machines | 17 | 32.9 | 92.4 | test_large_direct_step_rolls_newton_and_stays_local 8.1/25.1; test_large_krylov_firk_keeps_stage_increment_local 6.4/17.5; test_derived_defaults_stay_free_axes_on_a_copy 6.3/19.2; test_auto_launch_of_a_local_kernel 2.8/9.8 |
| #951 74d2bd7e default derivation at top-level Solver | 25 | 13.4 | 21.9 | test_solver_settings cases, 1–4 s each |
| #917 59335780 auto_performance heuristics | 11 | 12.0 | 34.0 | test_defaults_rerun_after_algorithm_update 9.4/30.5 |
| #979 71f29f96 algorithm defaults resolve in resolve() | 6 | 7.1 | 12.7 | test_error_solver_settings_resolve_and_survive_an_update 3.8/6.1 |
| #967 886636ed precompile plugin, timing tests | 5 | 6.4 | 8.6 | test_optimize_times_candidates_and_applies_the_fastest 3.8/4.5 |
| #919 be507c21 Solver.copy from settings_dict | 4 | 5.6 | 13.5 | test_copy_rederives_what_the_parent_derived 4.2/10.8 |
| #937 23ca4bb9 optimize auto-sizes runs | 6 | 5.4 | 4.3 | test_optimize_takes_device_grids 4.8/3.3 |
| #976 3d8d8a20 None sets the declared default | 1 | 3.7 | 10.5 | test_auto_performance_off_returns_flags_to_their_defaults |
| #993 347573a8 stock-numba memory | 17 | 3.1 | 3.1 | each under 0.5 s |
| unmatched (mostly renamed in #942) | 72 | 7.2 | 11.6 | |

## Existing tests, growth by module (py3.14 / py3.11 s)

| Module | py3.14 | py3.11 |
|---|---|---|
| test_step_controllers.TestControllerNumerical | +26.1 | +17.4 |
| test_SingleIntegratorRunCore | +10.7 | +19.0 |
| algorithms.test_ode_implicitstep | +4.9 | +9.4 |
| integrated test_step_algorithms | +4.8 | +10.9 |
| step_control.test_controllers | +4.2 | +6.8 |
| symbolic.structural | +3.4 | +4.1 |
| algorithms.test_generic_dirk | +3.2 | +5.3 |
| batchsolving.test_solver | +3.1 | +6.0 |

TestControllerNumerical::test_matches_cpu[*-i]: four cases, 0.0 → 7.3 s each
on py3.14.

## Coverage tracing cost

Same 153 GPU tests, `-n 4`, warm caches, precompile plugin, RTX 4070 SUPER:

| | untraced | traced |
|---|---|---|
| py3.11 (C tracer) | 13.5 s | 30 s |
| py3.14 (sys.monitoring) | 13.7 s | 14.7 s |
