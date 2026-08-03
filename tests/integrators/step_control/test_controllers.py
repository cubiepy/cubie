"""Structural tests for step controllers."""

import numpy as np
import pytest

from cubie.result_codes import CUBIE_RESULT_CODES
from tests._utils import run_controller_device_step
from tests._utils import (
    DT_CLAMP_CASES,
    CONTROLLER_TOLERANCE_SETS,
    HISTORY_CONTROLLER_TOLERANCE_SETS,
)



@pytest.mark.parametrize(
    "solver_settings_override, step_setup",
    [
        (settings, case)
        for settings in CONTROLLER_TOLERANCE_SETS.values()
        for case in DT_CLAMP_CASES.values()
    ],
    ids=[
        f"{controller}-{case}"
        for controller in CONTROLLER_TOLERANCE_SETS
        for case in DT_CLAMP_CASES
    ],
    indirect=True,
)
def test_dt_clamps(
    step_controller_settings,
    step_setup,
    device_step_results,
    tolerance,
):
    dt0 = step_setup["dt0"]
    dt_min = step_controller_settings["dt_min"]
    dt_max = step_controller_settings["dt_max"]
    if dt0 < dt_min:
        expected = dt_min
    elif dt0 > dt_max:
        expected = dt_max
    else:
        expected = dt0
    assert device_step_results.dt == pytest.approx(
        expected,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    list(CONTROLLER_TOLERANCE_SETS.values()),
    ids=list(CONTROLLER_TOLERANCE_SETS),
    indirect=True,
)
class TestControllers:
    def test_controller_builds(self, step_controller, precision):
        assert callable(step_controller.device_function)

    @pytest.mark.parametrize(
        "flag, value",
        [("afn", False), ("lto", False), ("lineinfo", True)],
    )
    def test_jit_flag_updates_reach_compile_settings(
        self, step_controller, flag, value
    ):
        """Jit-flag settings route into the controller's config."""
        original = getattr(step_controller.compile_settings.jit_flags, flag)
        recognized = step_controller.update_compile_settings({flag: value})
        try:
            assert flag in recognized
            assert getattr(
                step_controller.compile_settings.jit_flags, flag
            ) == value
        finally:
            step_controller.update_compile_settings({flag: original})

    def test_rejected_step_never_grows_dt(
        self, step_controller, precision, system
    ):
        """A rejected step shrinks dt after a solver failure."""
        device_func = step_controller.device_function
        n = system.sizes.states
        state = np.ones(n, dtype=precision)

        # First step: solver-failure error injection (loop uses 1e16).
        huge_error = np.full(n, 1e16, dtype=precision)
        first = run_controller_device_step(
            device_func,
            precision,
            0.017,
            huge_error,
            state=state,
            state_prev=state,
        )
        assert first.accepted == 0

        # Second step: moderate rejection (nrm2 just above one), with
        # the controller history the first launch left behind.
        dt_before = precision(0.017)
        moderate_error = np.full(n, 1.23e-3, dtype=precision)
        second = run_controller_device_step(
            device_func,
            precision,
            dt_before,
            moderate_error,
            state=state,
            state_prev=state,
            local_mem=first.local_mem,
        )
        assert second.accepted == 0
        assert second.dt < dt_before

    def test_truncated_accepted_step_freezes_controller(
        self, step_controller, precision, system
    ):
        """An accepted truncated step rescales nothing: dt and the
        error history are unchanged."""
        device_func = step_controller.device_function
        n = system.sizes.states
        dt0 = precision(0.017)
        tiny_error = np.full(n, 1e-12, dtype=precision)
        state = np.ones(n, dtype=precision)

        frozen = run_controller_device_step(
            device_func,
            precision,
            dt0,
            tiny_error,
            state=state,
            state_prev=state,
            truncated=True,
        )
        assert frozen.accepted == 1
        assert frozen.dt == dt0
        assert np.all(frozen.local_mem == precision(0.0))

        unforced = run_controller_device_step(
            device_func,
            precision,
            dt0,
            tiny_error,
            state=state,
            state_prev=state,
            truncated=False,
        )
        assert unforced.accepted == 1
        assert unforced.dt > dt0

    def test_truncated_rejected_step_still_shrinks_dt(
        self, step_controller, precision, system
    ):
        """A rejected truncated step still walks dt down."""
        device_func = step_controller.device_function
        n = system.sizes.states
        dt0 = precision(0.017)
        huge_error = np.full(n, 1e16, dtype=precision)
        state = np.ones(n, dtype=precision)

        result = run_controller_device_step(
            device_func,
            precision,
            dt0,
            huge_error,
            state=state,
            state_prev=state,
            truncated=True,
        )
        assert result.accepted == 0
        assert result.dt < dt0

    def test_truncated_accepted_step_at_dt_min_returns_success(
        self, step_controller, precision, system
    ):
        """An accepted truncated step at dt_min reports SUCCESS.

        Its sub-unity gain would otherwise propose dt <= dt_min and
        end the run as irrecoverable.
        """
        device_func = step_controller.device_function
        n = system.sizes.states
        dt_min = precision(step_controller.dt_min)
        # Error norm just below 1: accepted, with gain < 1.
        near_unity_error = np.full(n, 0.999e-3, dtype=precision)
        state = np.ones(n, dtype=precision)

        result = run_controller_device_step(
            device_func,
            precision,
            dt_min,
            near_unity_error,
            state=state,
            state_prev=state,
            truncated=True,
        )
        assert result.accepted == 1
        assert result.dt == dt_min
        assert result.status == int(CUBIE_RESULT_CODES.SUCCESS)


@pytest.mark.parametrize(
    "solver_settings_override",
    list(HISTORY_CONTROLLER_TOLERANCE_SETS.values()),
    ids=list(HISTORY_CONTROLLER_TOLERANCE_SETS),
    indirect=True,
)
class TestControllerHistory:
    @pytest.mark.parametrize(
        "accepted, truncated",
        (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ),
        ids=(
            "rejection",
            "truncated-rejection",
            "acceptance",
            "truncated-acceptance",
        ),
    )
    def test_history_commits_only_untruncated_acceptance(
        self,
        step_controller,
        step_controller_settings,
        precision,
        system,
        accepted,
        truncated,
    ):
        """History advances only after an ordinary accepted step."""
        controller_name = step_controller_settings["step_controller"]
        dt0 = precision(0.017)
        seeds = {
            "pi": np.asarray([0.25], dtype=precision),
            "pid": np.asarray([0.25, 0.5], dtype=precision),
            "gustafsson": np.asarray([0.0125, 0.25], dtype=precision),
        }
        # The accepted error equals atol with rtol zero, so the
        # committed norm is exactly one and history slots either take
        # these values verbatim or stay bit-identical to the seed.
        accepted_history = {
            "pi": np.asarray([1.0], dtype=precision),
            "pid": np.asarray([1.0, 0.25], dtype=precision),
            "gustafsson": np.asarray([dt0, 1.0], dtype=precision),
        }
        error_value = 1e-3 if accepted else 2e-3
        error = np.full(system.sizes.states, error_value, dtype=precision)
        state = np.ones(system.sizes.states, dtype=precision)
        seed = seeds[controller_name]

        result = run_controller_device_step(
            step_controller.device_function,
            precision,
            dt0,
            error,
            local_mem=seed.copy(),
            state=state,
            state_prev=state,
            truncated=truncated,
        )

        assert result.accepted == int(accepted)
        expected = (
            accepted_history[controller_name]
            if accepted and not truncated
            else seed
        )
        np.testing.assert_array_equal(result.local_mem, expected)
