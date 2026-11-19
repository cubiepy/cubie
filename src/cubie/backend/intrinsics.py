"""Cubie device intrinsics with numba-cuda-mlir typing and lowering.

Published Objects
-----------------
:func:`narrow_f64`
    Narrow a float64 to float32 without flushing subnormal results.
:func:`unroll_if`
    ``unroll_if(range(n), flag[, count])``: ``flag`` sets whether MLIR
    adds a loop-unroll hint, ``count`` its unroll count. The AST pass
    in :mod:`cubie.backend._mlir_cubie_extensions` consumes the call.
:data:`UnrollFlag`
    One ``(unroll, count)`` pair, the ``flag`` argument.
:func:`bf16x2_pack`, :func:`bf16x2_high`, :func:`bf16x2_low`
    Round two float32 to a bfloat16 pair in one ``uint32``, and read
    either half back as a float32.
:func:`bf16x2_fma`
    Lane-wise bfloat16 fused multiply-add of three pairs.
:func:`bf16x2_insert_high`, :func:`bf16x2_insert_low`
    Replace one half of a bfloat16 pair.
:func:`float_bits`, :func:`bits_float`, :func:`multiply_high`
    Reinterpret float32 bits as int32 and back; the high word of an
    int32 product.
:func:`fast_reciprocal`
    A float32 reciprocal from a bit-pattern estimate and two Newton
    steps, with no hardware reciprocal.
"""

from typing import Optional, Tuple

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


_INLINE_ASM_TYPES = {
    types.float32: _ir.F32Type.get,
    types.int32: lambda: _ir.IntegerType.get_signless(32),
    types.uint32: lambda: _ir.IntegerType.get_signless(32),
}


def _inline_asm_intrinsic(stub, result, arguments, asm, constraints):
    """Type and lower ``stub`` as one inline PTX instruction."""

    @type_callable(stub)
    def _typer(context):
        def typer(*actual):
            if tuple(actual) == tuple(arguments):
                return result

        return typer

    @_lower(stub, *arguments)
    def _lowering(builder, target, args, kwargs):
        values = [builder.load_var(arg) for arg in args]
        value = _llvm.inline_asm(
            _INLINE_ASM_TYPES[result](), values, asm, constraints
        )
        builder.store_var(target, value)

    return stub


def _bfloat16_bits(value) -> np.uint32:
    """Return the bfloat16 nearest ``value`` (ties to even) as raw bits."""
    bits = int(np.float32(value).view(np.uint32))
    if (bits & 0x7F800000) == 0x7F800000 and bits & 0x007FFFFF:
        return np.uint32((bits >> 16) | 0x0040)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return np.uint32((rounded >> 16) & 0xFFFF)


def _bfloat16_value(bits) -> np.float32:
    """Return the float32 holding the bfloat16 ``bits``."""
    return np.uint32((int(bits) & 0xFFFF) << 16).view(np.float32)


def bf16x2_pack(high, low):
    """Return ``high`` and ``low`` rounded to bfloat16 in one word.

    ``high`` takes bits 16-31 and ``low`` bits 0-15.
    """
    return np.uint32(
        (int(_bfloat16_bits(high)) << 16) | int(_bfloat16_bits(low))
    )


def bf16x2_high(packed):
    """Return the high bfloat16 of ``packed`` as a float32."""
    return _bfloat16_value(int(packed) >> 16)


def bf16x2_low(packed):
    """Return the low bfloat16 of ``packed`` as a float32."""
    return _bfloat16_value(int(packed) & 0xFFFF)


def bf16x2_fma(a, b, c):
    """Return the lane-wise bfloat16 ``a * b + c`` of three packed words."""
    lanes = []
    for unpack in (bf16x2_high, bf16x2_low):
        # The float64 product and sum are exact for bfloat16 operands.
        exact = float(unpack(a)) * float(unpack(b)) + float(unpack(c))
        # Round to float32 towards odd, so that the bfloat16 rounding
        # below rounds once from the exact value.
        nearest = np.float32(exact)
        if float(nearest) != exact and np.isfinite(nearest):
            if abs(float(nearest)) > abs(exact):
                nearest = np.nextafter(nearest, np.float32(0.0))
            nearest = (nearest.view(np.uint32) | np.uint32(1)).view(
                np.float32
            )
        lanes.append(nearest)
    return bf16x2_pack(lanes[0], lanes[1])


def bf16x2_insert_low(packed, value):
    """Return ``packed`` with its low bfloat16 replaced by ``value``."""
    return np.uint32(
        (int(packed) & 0xFFFF0000) | int(_bfloat16_bits(value))
    )


def bf16x2_insert_high(packed, value):
    """Return ``packed`` with its high bfloat16 replaced by ``value``."""
    return np.uint32(
        (int(packed) & 0x0000FFFF) | (int(_bfloat16_bits(value)) << 16)
    )


def multiply_high(a, b):
    """Return the high 32 bits of the 64-bit product of two int32."""
    return np.int32((int(a) * int(b)) >> 32)


RECIPROCAL_ESTIMATE_MAGIC = 0x7EF311C3
"""Bits that, less a positive float32, estimate its reciprocal.

The estimate's relative error is under 12.5%; each Newton step squares
it.
"""

RECIPROCAL_SMALLEST = np.float32(2.0**-126)
"""Smallest magnitude :func:`fast_reciprocal` inverts."""

RECIPROCAL_LARGEST = np.float32(2.0**126)
"""Largest magnitude :func:`fast_reciprocal` inverts."""


def fast_reciprocal(value):
    """Return ``1 / value`` from a bit estimate and two Newton steps.

    The magnitude is clamped to ``[2**-126, 2**126]`` first, so zero
    gives ``2**126`` and infinity ``2**-126``, each with the sign of
    ``value``. Inside that range the relative error is under 3e-4.
    """
    value = np.float32(value)
    magnitude = min(
        max(abs(value), RECIPROCAL_SMALLEST), RECIPROCAL_LARGEST
    )
    magnitude_bits = int(np.float32(magnitude).view(np.int32))
    bits = (RECIPROCAL_ESTIMATE_MAGIC - magnitude_bits) & 0xFFFFFFFF
    estimate = np.uint32(bits).view(np.float32)
    for _ in range(2):
        error = np.float32(magnitude * estimate - np.float32(1.0))
        estimate = np.float32(estimate - estimate * error)
    return -estimate if np.signbit(value) else estimate


def float_bits(value):
    """Return the bits of a float32 as an int32."""
    return np.float32(value).view(np.int32)


def bits_float(bits):
    """Return the float32 whose bits are the int32 ``bits``."""
    return np.int32(bits).view(np.float32)


_inline_asm_intrinsic(
    bf16x2_pack, types.uint32, (types.float32, types.float32),
    "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f",
)
_inline_asm_intrinsic(
    bf16x2_high, types.float32, (types.uint32,),
    "and.b32 $0, $1, 0xffff0000;", "=f,r",
)
_inline_asm_intrinsic(
    bf16x2_low, types.float32, (types.uint32,),
    "shl.b32 $0, $1, 16;", "=f,r",
)
_inline_asm_intrinsic(
    bf16x2_fma, types.uint32, (types.uint32, types.uint32, types.uint32),
    "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r",
)
# The packed conversion runs on the arithmetic pipe; the scalar
# cvt.rn.bf16.f32 runs on the slower conversion unit.
_inline_asm_intrinsic(
    bf16x2_insert_low, types.uint32, (types.uint32, types.float32),
    "{.reg .b32 pair; cvt.rn.bf16x2.f32 pair, 0f00000000, $2;"
    " prmt.b32 $0, pair, $1, 0x7610;}",
    "=r,r,f",
)
_inline_asm_intrinsic(
    bf16x2_insert_high, types.uint32, (types.uint32, types.float32),
    "{.reg .b32 pair; cvt.rn.bf16x2.f32 pair, $2, 0f00000000;"
    " prmt.b32 $0, $1, pair, 0x7610;}",
    "=r,r,f",
)
_inline_asm_intrinsic(
    multiply_high, types.int32, (types.int32, types.int32),
    "mul.hi.s32 $0, $1, $2;", "=r,r,r",
)
_inline_asm_intrinsic(
    fast_reciprocal, types.float32, (types.float32,),
    "{.reg .b32 bits, sign; .reg .f32 magnitude, estimate, error, negative;"
    " abs.ftz.f32 magnitude, $1;"
    " max.ftz.f32 magnitude, magnitude, 0f00800000;"
    " min.ftz.f32 magnitude, magnitude, 0f7E800000;"
    " mov.b32 bits, magnitude;"
    " sub.s32 bits, 0x7EF311C3, bits;"
    " mov.b32 estimate, bits;"
    " fma.rn.ftz.f32 error, magnitude, estimate, 0fBF800000;"
    " neg.f32 negative, error;"
    " fma.rn.ftz.f32 estimate, negative, estimate, estimate;"
    " fma.rn.ftz.f32 error, magnitude, estimate, 0fBF800000;"
    " neg.f32 negative, error;"
    " fma.rn.ftz.f32 estimate, negative, estimate, estimate;"
    " mov.b32 bits, estimate;"
    " mov.b32 sign, $1;"
    " and.b32 sign, sign, 0x80000000;"
    " or.b32 bits, bits, sign;"
    " mov.b32 $0, bits;}",
    "=f,f",
)
_inline_asm_intrinsic(
    float_bits, types.int32, (types.float32,), "mov.b32 $0, $1;", "=r,f",
)
_inline_asm_intrinsic(
    bits_float, types.float32, (types.int32,), "mov.b32 $0, $1;", "=f,r",
)


UnrollFlag = Tuple[bool, Optional[int]]
"""Loop-group flag: ``(unroll, count)``."""


def unroll_if(iterable, flag, count=None):
    """Return ``iterable``; the UnrollIf pass consumes the call."""
    return iterable
