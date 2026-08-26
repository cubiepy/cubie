"""Solvers sharing a stream group keep independent run partitions."""

import numpy as np

from tests._utils import _build_solver_instance

STREAM_GROUP = "sibling_allocation"


def _build_solver(system, solver_settings, driver_settings, manager):
    return _build_solver_instance(
        system=system,
        solver_settings={**solver_settings, "stream_group": STREAM_GROUP},
        driver_settings=driver_settings,
        memory_manager=manager,
    )


def test_repeat_solve_with_live_result_after_sibling_construction(
    system,
    solver_settings,
    driver_settings,
    batch_input_arrays,
    thread_mem_manager,
):
    """A held result plus newly built siblings leaves the batch intact.

    Building a solver queues a one-run placeholder allocation in the
    shared group. The base solver's next solve must not size its own
    run partition from those placeholders, or its later reallocation
    (forced here by holding the previous result) launches the full
    batch against one-run output buffers.
    """
    y0, params = batch_input_arrays
    n_runs = y0.shape[1]
    solve_kwargs = dict(drivers=driver_settings, duration=0.1)
    base = _build_solver(
        system, solver_settings, driver_settings, thread_mem_manager
    )
    siblings = []
    try:
        base.solve(y0, params, **solve_kwargs)
        siblings = [
            _build_solver(
                system, solver_settings, driver_settings, thread_mem_manager
            )
            for _ in range(2)
        ]
        held = base.solve(y0, params, **solve_kwargs)
        for sibling in siblings:
            sibling.solve(y0, params, **solve_kwargs)

        repeat = base.solve(y0, params, **solve_kwargs)

        outputs = base.kernel.output_arrays
        assert outputs.device_state.shape[2] == n_runs
        assert outputs.device_status_codes.shape[0] == n_runs
        assert base.kernel.chunks == 1
        np.testing.assert_array_equal(repeat.state, held.state)
        np.testing.assert_array_equal(
            repeat.status_codes, held.status_codes
        )
    finally:
        base.close()
        for sibling in siblings:
            sibling.close()
