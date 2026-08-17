"""MLIR-backend typing and lowering for cubie device utilities."""

import numpy as np

from numba_cuda_mlir._mlir import ir as _ir
from numba_cuda_mlir._mlir.dialects import llvm as _llvm
from numba_cuda_mlir.extending import type_callable
from numba_cuda_mlir.lowering.builtins import lower as _lower, type_convert
from numba_cuda_mlir.lowering_utilities import try_extract_constant
from numba_cuda_mlir.numba_cuda import types


def narrow_f64(value):
    """Narrow a float64 to float32 without flushing subnormal results."""

    return np.float32(value)


@type_callable(narrow_f64)
def _type_narrow_f64(context):
    def typer(value):
        if value == types.float64:
            return types.float32

    return typer


@_lower(narrow_f64, types.Float)
def _lower_narrow_f64(builder, target, args, kwargs):
    value = builder.load_var(args[0])
    if (
        isinstance(value, _ir.Value)
        and isinstance(value.type, _ir.F64Type)
        and try_extract_constant(value) is None
    ):
        result = _llvm.inline_asm(
            _ir.F32Type.get(), [value], "cvt.rn.f32.f64 $0, $1;", "=f,d"
        )
        builder.store_var(target, result)
        return
    type_convert(builder, target, args, kwargs)
