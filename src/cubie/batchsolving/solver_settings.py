"""All settings accepted by a top-level solver.

Published Classes
-----------------
:class:`SolverSettings`
    One field per setting a Solver accepts; ``None`` means not given.
:class:`EffectiveSettings`
    The settings in effect: the given ones plus the resolved ones.
"""

from typing import Any, Dict, List, Optional, Set

from attrs import Attribute, field, fields, frozen
from numpy import array as np_array
from numpy import ndarray

from cubie.CUDAFactory import _CubieConfigBase


def _optional(**kwargs: Any) -> Any:
    """Return a setting field whose ``None`` is passed to the children."""
    return field(default=None, metadata={"passes_none": True}, **kwargs)


def owned(value: Any) -> Any:
    """Return arrays as read-only copies and lists as tuples."""
    if isinstance(value, ndarray):
        value = np_array(value)
        value.setflags(write=False)
        return value
    if isinstance(value, list):
        return tuple(value)
    return value


def _owned_fields(cls: type, attributes: List[Attribute]) -> List[Attribute]:
    """Give every init field the ``owned`` converter."""
    return [
        attribute.evolve(converter=owned) if attribute.init else attribute
        for attribute in attributes
    ]


@frozen(field_transformer=_owned_fields)
class SolverSettings(_CubieConfigBase):
    """Every setting a Solver accepts; ``None`` means not given."""

    # -- Solver -------------------------------------------------------
    algorithm: Any = None
    tableau: Any = None
    time_logging_level: Optional[str] = None
    save_variables: Optional[List[str]] = None
    summarise_variables: Optional[List[str]] = None

    # -- System -------------------------------------------------------
    precision: Any = None
    operation_ordering: Optional[str] = None

    # -- Step algorithm -----------------------------------------------
    attempt_dense_prediction: Optional[bool] = None
    dae_initialisation: Optional[str] = None
    linear_correction_type: Optional[str] = None
    preconditioner_type: Optional[str] = None
    preconditioner_order: Optional[int] = None
    inexact_newton: Optional[bool] = None
    prefactored: Optional[bool] = None
    use_smoothed_error: Optional[bool] = None
    krylov_atol: Any = None
    krylov_rtol: Any = None
    krylov_max_iters: Optional[int] = None
    krylov_residual_floor: Optional[float] = None
    krylov_residual_reduction: Optional[float] = None
    newton_atol: Any = None
    newton_rtol: Any = None
    newton_max_iters: Optional[int] = None
    error_atol: Any = None
    error_rtol: Any = None
    error_max_iters: Optional[int] = None
    error_residual_floor: Optional[float] = None
    error_residual_reduction: Optional[float] = None

    # -- Step buffer placement ----------------------------------------
    accumulator_location: Optional[str] = None
    base_state_placeholder_location: Optional[str] = None
    cached_auxiliaries_location: Optional[str] = None
    delta_location: Optional[str] = None
    dxdt_location: Optional[str] = None
    increment_cache_location: Optional[str] = None
    krylov_iters_local_location: Optional[str] = None
    krylov_iters_out_location: Optional[str] = None
    lu_factor_location: Optional[str] = None
    p_location: Optional[str] = None
    preconditioned_vec_location: Optional[str] = None
    predictor_previous_values_location: Optional[str] = None
    predictor_transform_location: Optional[str] = None
    prev_theta_location: Optional[str] = None
    previous_step_size_location: Optional[str] = None
    r0_hat_location: Optional[str] = None
    residual_location: Optional[str] = None
    s_hat_location: Optional[str] = None
    stage_accumulator_location: Optional[str] = None
    stage_base_location: Optional[str] = None
    stage_driver_stack_location: Optional[str] = None
    stage_increment_history_location: Optional[str] = None
    stage_increment_location: Optional[str] = None
    stage_rhs_location: Optional[str] = None
    stage_state_location: Optional[str] = None
    stage_store_location: Optional[str] = None
    temp_location: Optional[str] = None
    tmp_location: Optional[str] = None
    v_location: Optional[str] = None

    # -- Step controller ----------------------------------------------
    step_controller: Optional[str] = None
    dt: Optional[float] = None
    dt_min: Optional[float] = None
    dt_max: Optional[float] = None
    atol: Any = None
    rtol: Any = None
    safety: Optional[float] = None
    min_step_shrink: Optional[float] = None
    max_step_growth: Optional[float] = None
    deadband_min: Optional[float] = None
    deadband_max: Optional[float] = None
    integral_gain: Any = None
    proportional_gain: Any = None
    derivative_gain: Any = None
    filter_coefficients: Any = None
    newton_target_iters: Optional[int] = None
    timestep_memory_location: Optional[str] = None

    # -- Loop ---------------------------------------------------------
    save_every: Optional[float] = _optional()
    summarise_every: Optional[float] = _optional()
    sample_summaries_every: Optional[float] = _optional()
    accept_step_location: Optional[str] = None
    counters_location: Optional[str] = None
    drivers_location: Optional[str] = None
    dt_location: Optional[str] = None
    error_location: Optional[str] = None
    observable_summary_location: Optional[str] = None
    observables_location: Optional[str] = None
    parameters_location: Optional[str] = None
    proposed_counters_location: Optional[str] = None
    proposed_drivers_location: Optional[str] = None
    proposed_observables_location: Optional[str] = None
    proposed_state_location: Optional[str] = None
    state_location: Optional[str] = None
    state_summary_location: Optional[str] = None

    # -- Outputs ------------------------------------------------------
    output_types: Optional[List[str]] = None
    saved_state_indices: Any = None
    saved_observable_indices: Any = None
    summarised_state_indices: Any = None
    summarised_observable_indices: Any = None

    # -- Kernel -------------------------------------------------------
    auto_performance: Optional[bool] = None
    blocksize: Optional[int] = None
    max_registers: Optional[int] = None
    kernel_name: Optional[str] = None
    cache: Any = None
    cache_enabled: Optional[bool] = None
    cache_mode: Optional[str] = None
    cache_dir: Any = None
    max_cache_entries: Optional[int] = None

    # -- Memory -------------------------------------------------------
    memory_manager: Any = None
    stream_group: Optional[str] = None
    mem_proportion: Optional[float] = _optional()

    # -- Driver interpolation -----------------------------------------
    drivers: Any = None
    order: Optional[int] = None
    wrap: Optional[bool] = None
    boundary_condition: Any = None

    # -- Loop unrolling -----------------------------------------------
    unroll_stage: Any = None
    unroll_step_element: Any = None
    unroll_accumulator: Any = None
    unroll_solver_element: Any = None
    unroll_norms: Any = None
    unroll_other_small: Any = None
    unroll_newton_exits: Any = None
    unroll_krylov_exits: Any = None

    # -- JIT flags ----------------------------------------------------
    lineinfo: Optional[bool] = None
    nsz: Optional[bool] = None
    contract: Optional[bool] = None
    arcp: Optional[bool] = None
    afn: Optional[bool] = None
    ftz: Optional[bool] = None
    lto: Optional[bool] = None

    def is_given(self, name: str) -> bool:
        """Return whether ``name`` was given."""
        return getattr(self, name) is not None

    def as_kwargs(self) -> Dict[str, Any]:
        """Return the given settings as Solver keyword arguments."""
        return {
            fld.name: getattr(self, fld.name)
            for fld in fields(type(self))
            if fld.init and getattr(self, fld.name) is not None
        }


@frozen(field_transformer=_owned_fields)
class EffectiveSettings(SolverSettings):
    """The settings in effect: the given ones plus the resolved ones."""

    is_adaptive: Optional[bool] = None
    save_last: Optional[bool] = None
    save_regularly: Optional[bool] = None
    summarise_last: Optional[bool] = None
    summarise_regularly: Optional[bool] = None

    def as_kwargs(self) -> Dict[str, Any]:
        """Return the settings in effect, ``None`` intervals included."""
        return {
            fld.name: getattr(self, fld.name)
            for fld in fields(type(self))
            if fld.init
            and (
                getattr(self, fld.name) is not None
                or fld.metadata.get("passes_none")
            )
        }


def clashing_names(names: Any) -> Set[str]:
    """Return the entries of ``names`` that are Solver settings."""
    settings = {fld.name for fld in fields(SolverSettings) if fld.init}
    return {str(name) for name in names} & settings
