"""Tests for cubie.batchsolving.BatchSolverConfig."""

from __future__ import annotations

import numpy as np
import pytest

from cubie.batchsolving.BatchSolverConfig import (
    ActiveOutputs,
    BatchSolverConfig,
)
from cubie.outputhandling.output_config import OutputCompileFlags


# -- ActiveOutputs: defaults and fields -------------------------------- #

def test_active_outputs_defaults_all_false():
    """All six fields default to False."""
    ao = ActiveOutputs()
    for field_name in (
        "state",
        "observables",
        "state_summaries",
        "observable_summaries",
        "status_codes",
        "iteration_counters",
    ):
        assert getattr(ao, field_name) is False


def test_active_outputs_fields_accept_true():
    """All six fields can be set to True and retain their values."""
    ao = ActiveOutputs(
        state=True,
        observables=True,
        state_summaries=True,
        observable_summaries=True,
        status_codes=True,
        iteration_counters=True,
    )
    for field_name in (
        "state",
        "observables",
        "state_summaries",
        "observable_summaries",
        "status_codes",
        "iteration_counters",
    ):
        assert getattr(ao, field_name) is True


# -- ActiveOutputs.from_compile_flags --------------------------------- #

@pytest.mark.parametrize(
    "flag_kwargs, expected",
    [
        pytest.param(
            {
                "save_state": True,
                "save_observables": False,
                "summarise_state": False,
                "summarise_observables": False,
                "save_counters": False,
            },
            {
                "state": True,
                "observables": False,
                "state_summaries": False,
                "observable_summaries": False,
                "status_codes": True,
                "iteration_counters": False,
            },
            id="save_state_only",
        ),
        pytest.param(
            {
                "save_state": False,
                "save_observables": True,
                "summarise_state": False,
                "summarise_observables": False,
                "save_counters": False,
            },
            {
                "state": False,
                "observables": True,
                "state_summaries": False,
                "observable_summaries": False,
                "status_codes": True,
                "iteration_counters": False,
            },
            id="save_observables_only",
        ),
        pytest.param(
            {
                "save_state": False,
                "save_observables": False,
                "summarise_state": True,
                "summarise_observables": False,
                "save_counters": False,
            },
            {
                "state": False,
                "observables": False,
                "state_summaries": True,
                "observable_summaries": False,
                "status_codes": True,
                "iteration_counters": False,
            },
            id="summarise_state_only",
        ),
        pytest.param(
            {
                "save_state": False,
                "save_observables": False,
                "summarise_state": False,
                "summarise_observables": True,
                "save_counters": False,
            },
            {
                "state": False,
                "observables": False,
                "state_summaries": False,
                "observable_summaries": True,
                "status_codes": True,
                "iteration_counters": False,
            },
            id="summarise_observables_only",
        ),
        pytest.param(
            {
                "save_state": False,
                "save_observables": False,
                "summarise_state": False,
                "summarise_observables": False,
                "save_counters": True,
            },
            {
                "state": False,
                "observables": False,
                "state_summaries": False,
                "observable_summaries": False,
                "status_codes": True,
                "iteration_counters": True,
            },
            id="save_counters_only",
        ),
        pytest.param(
            {
                "save_state": False,
                "save_observables": False,
                "summarise_state": False,
                "summarise_observables": False,
                "save_counters": False,
            },
            {
                "state": False,
                "observables": False,
                "state_summaries": False,
                "observable_summaries": False,
                "status_codes": True,
                "iteration_counters": False,
            },
            id="all_false_status_codes_still_true",
        ),
    ],
)
def test_from_compile_flags_mapping(flag_kwargs, expected):
    """from_compile_flags maps each flag correctly; status_codes always True.
    """
    flags = OutputCompileFlags(**flag_kwargs)
    ao = ActiveOutputs.from_compile_flags(flags)
    for attr_name, expected_val in expected.items():
        assert getattr(ao, attr_name) is expected_val


# -- BatchSolverConfig ------------------------------------------------ #

def test_batch_solver_config_defaults():
    """loop_fn defaults to None; compile_flags defaults to
    OutputCompileFlags().
    """
    cfg = BatchSolverConfig(precision=np.float32)
    assert cfg.loop_fn is None
    assert cfg.compile_flags == OutputCompileFlags()


def test_batch_solver_config_precision():
    """Precision is stored from CUDAFactoryConfig parent."""
    cfg = BatchSolverConfig(precision=np.float64)
    assert cfg.precision == np.float64


# -- active_outputs property ------------------------------------------ #

def test_active_outputs_property_derives_from_compile_flags():
    """active_outputs property returns ActiveOutputs matching compile_flags."""
    flags = OutputCompileFlags(
        save_state=True,
        save_observables=True,
        summarise_state=True,
        summarise_observables=True,
        save_counters=True,
    )
    cfg = BatchSolverConfig(precision=np.float32, compile_flags=flags)
    ao = cfg.active_outputs
    expected = ActiveOutputs.from_compile_flags(flags)
    assert ao == expected
    # Verify concrete values to ensure correctness
    assert ao.state is True
    assert ao.observables is True
    assert ao.state_summaries is True
    assert ao.observable_summaries is True
    assert ao.status_codes is True
    assert ao.iteration_counters is True
