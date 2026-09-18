"""The inline worker's callee IR cache lives and dies with its pipeline."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from cubie.backend import _mlir_compat
from numba_cuda_mlir.descriptor import mlir_target
from numba_cuda_mlir.numba_cuda.core import inline_closurecall
from numba_cuda_mlir.numba_cuda.flags import CUDAFlags

pytestmark = [pytest.mark.mlir_only, pytest.mark.nocudasim]


def _callee(x):
    return x + 1


def _worker():
    pipeline = SimpleNamespace()
    worker = inline_closurecall.InlineWorker(
        mlir_target.typing_context,
        mlir_target.target_context,
        {},
        pipeline,
        CUDAFlags(),
        inline_closurecall.callee_ir_validator,
    )
    return pipeline, worker


def test_callee_ir_is_cached_per_pipeline_and_cloned_per_call_site():
    pipeline, worker = _worker()
    first = worker._fresh_callee_ir(_callee)
    second = worker._fresh_callee_ir(_callee)
    cache = getattr(pipeline, _mlir_compat._PIPELINE_CALLEE_IR_CACHE_ATTR)

    assert first is not second
    assert [key[0] for key in cache] == [_callee]


def test_callee_ir_is_released_with_its_pipeline():
    pipeline, worker = _worker()
    worker._fresh_callee_ir(_callee)
    cache = getattr(pipeline, _mlir_compat._PIPELINE_CALLEE_IR_CACHE_ATTR)
    canonical_ir = weakref.ref(next(iter(cache.values())))
    del cache, pipeline, worker
    gc.collect()

    assert canonical_ir() is None
