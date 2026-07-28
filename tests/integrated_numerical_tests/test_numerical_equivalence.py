"""Full-run numerical equivalence: device loop vs CPU reference."""

from __future__ import annotations

import pytest

from tests._utils import (
    ALGORITHM_PARAM_SETS,
    assert_integration_outputs,
)


@pytest.mark.parametrize(
    "solver_settings_override",
    [ALGORITHM_PARAM_SETS[0]],
    indirect=True,
)
def test_numerical_equivalence(
    single_integrator_run,
    cpu_loop_outputs,
    device_loop_outputs,
    tolerance,
):
    """Device loop output matches CPU reference."""

    assert device_loop_outputs.status == cpu_loop_outputs["status"]
    assert_integration_outputs(
        reference=cpu_loop_outputs,
        device=device_loop_outputs,
        output_functions=single_integrator_run._output_functions,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
    )
