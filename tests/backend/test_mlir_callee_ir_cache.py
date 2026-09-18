"""The inline worker's callee IR cache lives and dies with its pipeline."""

import gc
import weakref
from importlib import import_module
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.mlir_only, pytest.mark.nocudasim]


def _callee(x):
    return x + 1


def _worker():
    compat = import_module("cubie.backend._mlir_compat")
    descriptor = import_module("numba_cuda_mlir.descriptor")
    flags = import_module("numba_cuda_mlir.numba_cuda.flags")
    pipeline = SimpleNamespace()
    worker = compat._nb_icc.InlineWorker(
        descriptor.mlir_target.typing_context,
        descriptor.mlir_target.target_context,
        {},
        pipeline,
        flags.CUDAFlags(),
        compat._nb_icc.callee_ir_validator,
    )
    return compat, pipeline, worker


def test_callee_ir_is_cached_per_pipeline_and_cloned_per_call_site():
    compat, pipeline, worker = _worker()
    first = worker._fresh_callee_ir(_callee)
    second = worker._fresh_callee_ir(_callee)
    cache = getattr(pipeline, compat._PIPELINE_CALLEE_IR_CACHE_ATTR)

    assert first is not second
    assert [key[0] for key in cache] == [_callee]


def test_callee_ir_is_released_with_its_pipeline():
    compat, pipeline, worker = _worker()
    worker._fresh_callee_ir(_callee)
    cache = getattr(pipeline, compat._PIPELINE_CALLEE_IR_CACHE_ATTR)
    canonical_ir = weakref.ref(next(iter(cache.values())))
    del cache, pipeline, worker
    gc.collect()

    assert canonical_ir() is None
