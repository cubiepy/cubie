"""An interpolated driver must reproduce the same drive in-equation."""

import numpy as np
import pytest

from tests._utils import assert_integration_outputs, run_device_loop


_DRIVER_DURATION = 2.0 * np.pi
_DRIVER_SAMPLES = 127
# One full period on a uniform grid whose last knot lands exactly on
# the duration, so the periodic wrap closes on itself.
_DRIVER_SAMPLE_PERIOD = _DRIVER_DURATION / (_DRIVER_SAMPLES - 1)
_DRIVER_TIMES = np.arange(_DRIVER_SAMPLES) * _DRIVER_SAMPLE_PERIOD
_DRIVER_VALUES = np.sin(_DRIVER_TIMES)
_DRIVER_VALUES[-1] = _DRIVER_VALUES[0]

TIME_DRIVER_SETTINGS = {
    "system_type": "time_array_driver",
    "duration": _DRIVER_DURATION,
    "dt_min": 0.05,
    "dt_max": 0.05,
    "save_every": 0.05,
    "summarise_every": 0.1,
    "sample_summaries_every": 0.05,
    "saved_state_indices": [0],
    "saved_observable_indices": [0],
    "summarised_state_indices": [0],
    "summarised_observable_indices": [0],
    "output_types": ["state", "observables", "time"],
    "driverspline_wrap": True,
    "driverspline_boundary_condition": "periodic",
}

SINUSOID_DRIVER_SAMPLES = {
    "drive": _DRIVER_VALUES,
    "driver_sample_period": _DRIVER_SAMPLE_PERIOD,
}


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
