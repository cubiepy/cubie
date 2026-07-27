"""An interpolated driver must reproduce the same drive in-equation."""

import pytest

from tests._utils import (
    SINUSOID_DRIVER_SAMPLES,
    TIME_DRIVER_SETTINGS,
    assert_integration_outputs,
    run_device_loop,
)


@pytest.mark.parametrize(
    "solver_settings_override", [TIME_DRIVER_SETTINGS], indirect=True
)
@pytest.mark.parametrize(
    "driver_settings_override", [SINUSOID_DRIVER_SAMPLES], indirect=True
)
def test_time_driver_array_matches_function(
    precision,
    system,
    solver_settings,
    single_integrator_run,
    driver_array,
    output_functions,
    time_function_driver_system,
    time_function_driver_run,
):
    """The spline-driven twin matches the sinusoid-in-equations twin."""
    reference_result = run_device_loop(
        time_function_driver_run,
        system=time_function_driver_system,
        initial_state=(
            time_function_driver_system.initial_values.values_array.astype(
                precision, copy=True
            )
        ),
        solver_config=solver_settings,
    )
    driver_result = run_device_loop(
        single_integrator_run,
        system=system,
        initial_state=system.initial_values.values_array.astype(
            precision, copy=True
        ),
        solver_config=solver_settings,
        driver_array=driver_array,
    )

    assert_integration_outputs(
        reference=reference_result,
        device=driver_result,
        output_functions=output_functions,
        rtol=1e-5,
        atol=1e-5,
    )
