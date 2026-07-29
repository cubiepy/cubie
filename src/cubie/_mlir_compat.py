"""Lowering registrations that fill gaps in numba-cuda-mlir.

numba-cuda-mlir 0.4.0 registers bitwise binary operators
(``&``, ``|``, ``^`` and their in-place forms) for
``(Integer, Integer)`` signatures only; Boolean operands raise
``NotImplementedError`` during MLIR lowering. CuBIE device code uses
branch-free Boolean flag updates (``finished &= save_finished``),
so this module registers the missing Boolean signatures using the
same ``arith.andi``/``ori``/``xori`` code generation the package
uses for integers (all three operate on ``i1``).

numba-cuda-mlir also widens signless integers with zero-extension
(``arith.extui``) by default, corrupting negative signed operands of
``%`` and ``//`` on runtime values, and its integer ``%`` emits a
truncated remainder (``arith.remsi``) rather than Python's floored
modulo. This module registers ``(Integer, Integer)`` lowerings for
``%`` and ``//`` with sign-aware widening and floored semantics.

numba-cuda-mlir also lowers shared memory to zero-sized
internal-linkage globals: ``cuda.shared.array(0)`` becomes a private
zero-length ``memref.global`` and ``gpu.dynamic_shared_memory``'s
base becomes ``llvm.mlir.global internal @__dynamic_shmem__N :
!llvm.array<0 x i8>``. Indexing a zero-length internal object is
undefined behaviour, so optimizers (libnvvm -O3 and the LTO cubin
link at opt > 0) legally sink or delete stores staged through
dynamic shared memory. This module reroutes zero-length shared
arrays through a ``gpu.dynamic_shared_memory`` view and rewrites the
``__dynamic_shmem__`` globals to external linkage, which is the
"LTO store erasure" fix (upstream warning in #117 blames the
linker's fp16 handling; the actual defect is this lowering).

numba-cuda-mlir also materializes frozen numpy closure arrays with a
per-thread device ``malloc`` and no ``free`` (element-wise
``tensor.from_elements`` bufferized to a heap allocation). The 8 MB
default device heap is exhausted around 64k threads, so any kernel
capturing an array literal — every cubie save path freezes its
saved-index arrays — faulted with ``CUDA_ERROR_ILLEGAL_ADDRESS`` at
production batch sizes. This module reroutes array literals to
internal constant globals carrying the array's raw bytes.

numba-cuda-mlir 0.4.0 also converts memrefs to raw LLVM pointers from
the aligned allocation base, dropping the view offset and assuming
dense row-major strides, so cache-hint stores (``cuda.stwt``) and
aligned vector accesses through sliced views hit the wrong elements.
This module reroutes the conversion through
``memref.extract_strided_metadata``, mirroring the upstream fix
"Preserve memref offsets in pointer casts" (#73); the shim no-ops on
builds that already carry it.

numba-cuda-mlir also uses ABI storage types for multiply-assigned
compiler locals. Boolean locals therefore cross an i1/i8 boundary on
every stack load and store. This module keeps scalar and tuple locals
in their semantic value types. External arrays and ABI-facing data
retain their storage types.

numba-cuda-mlir also gives Python ``min`` and ``max`` NaN-propagating
float semantics and leaves constant-zero floating powers for NVVM.
This module selects the non-NaN operand and folds those powers to one.

numba-cuda-mlir also rejects the compile-time-empty tail view
``arr[n:n]`` of a size-``n`` array: the optimization pipeline folds
the frozen bounds and parent shape static after inlining, and the
``memref.subview`` bounds check then fails ("offset 0 is
out-of-bounds: n >= n") although numpy-style slicing allows the
view. This module anchors statically empty slices at offset zero,
which stays in bounds under any folding (the registry's zero-length
buffer views hit this on every statically sized shared or
persistent scratch parent).

Import this module before compiling any kernel; registrations are
picked up when the MLIR target context refreshes its registries.
These are stop-gaps that belong upstream in numba-cuda-mlir; patch
branches exist in the ccam80/numba-cuda-mlir fork. Fixes in the
backend's native (C++) code cannot be patched from here; they ship
prebuilt in the ``cubie-numba-cuda-mlir`` wheel the ``mlir*``
extras install (built from the fork's ``cubie-wheel`` branch), so
the installed wheel, not upstream source, is what compiles device
code. The Python-side branches are
(fix-boolean-bitwise-invert-lowering, fix-boolean-comparison-lowering,
fix-numpy-scalar-constants, fix-nested-tuple-dynamic-getitem,
fix-integer-mod-floordiv-lowering, fix-dynamic-shared-memory-ub,
fix-frozen-array-device-malloc, ssa-iterative-def-search,
selective-fastmath, fix-float-minmax-lowering, fix-pow-zero-fold,
ftz-standalone-flag, perf-drop-math-uplift-to-fma,
fix-lto-link-fastmath-knobs).

The iterative SSA def-search shim removes the RecursionError that
large flattened kernels hit inside ``reconstruct_ssa``, and the
selective fastmath shims accept numba-cuda's per-flag ``fastmath``
form (bool | set | dict), stamping ``#arith.fastmath`` per op and
rewriting ``arcp`` division / ``afn`` tanh to their hardware
approximations; both groups no-op on builds that carry the fixes
natively. The standalone-ftz shim enables the libnvvm
denormal-flush knob for every truthy flag set — numba-cuda's
module-knob behaviour — until the fork's ftz-standalone-flag
branch ships in the wheel, and the fma-uplift shim strips the
math-uplift-to-fma pass from the optimization pipeline so libnvvm
owns floating-point contraction (the fork's
perf-drop-math-uplift-to-fma branch). The LTO-link fastmath shim
passes the per-flag fma/prec-div/prec-sqrt codegen knobs (and
ftz under full ``fast``) to the production kernel link: final
code generation for LTO-IR inputs happens in nvJitLink, which
applies precise-math defaults unless the link passes the knobs,
so LTO cubins lost fastmath codegen while the ``get_lto_ptx()``
diagnostic linker carried it (the fork's
fix-lto-link-fastmath-knobs branch).
Remove the corresponding shim once each lands upstream. With the
shared-memory shim in place CuBIE requests LTO-link optimization
explicitly; set
NUMBA_CUDA_MLIR_DISABLE_LTO_OPT=1 to force opt_level=0 on the LTO
link.
"""

import copy
import inspect
import operator
import weakref
from collections import defaultdict

import numpy

from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir import mlir_lowering as _mlir_lowering
from numba_cuda_mlir import mlir_optimization as _mlir_optimization
from numba_cuda_mlir import optimization as _optimization
from numba_cuda_mlir._mlir import ir as _ir
from numba_cuda_mlir._mlir.dialects import (
    arith,
    builtin as _builtin,
    llvm as _llvm,
    memref as _memref,
)
from numba_cuda_mlir._mlir.extras import types as _T
from numba_cuda_mlir.mlir.dialect_exts import llvm as _llvm_ext
from numba_cuda_mlir.lowering_utilities.type_conversions import (
    to_numba_type as _to_numba_type,
)
from numba_cuda_mlir.lowering import builtins as _lowering_builtins
from numba_cuda_mlir.lowering import cuda as _lowering_cuda
from numba_cuda_mlir.lowering import math as _lowering_math
from numba_cuda_mlir.lowering import numpy as _lowering_numpy
from numba_cuda_mlir.lowering.numpy import registry as _np_registry
from numba_cuda_mlir.lowering.math import (
    eq_cg,
    ge_cg,
    gt_cg,
    le_cg,
    lt_cg,
    ne_cg,
    registry as _math_registry,
)
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import (
    analysis as _nb_analysis,
    errors as _nb_errors,
    inline_closurecall as _nb_icc,
    ir as _nb_ir,
    ir_utils as _nb_ir_utils,
    postproc as _nb_postproc,
    ssa as _nb_ssa,
    transforms as _nb_transforms,
)


def _make_bitwise_cg(mlir_op):
    """Build a code generator emitting ``mlir_op`` on cast operands.

    Parameters
    ----------
    mlir_op
        MLIR ``arith`` dialect builder (``andi``, ``ori`` or
        ``xori``) applied to the two operands.

    Returns
    -------
    Callable
        Lowering function with the ``(builder, target, args,
        kwargs)`` signature expected by the lowering registry.
    """

    def _cg(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        target_mlir_type = builder.get_mlir_type(target_type)
        lhs, rhs = args
        lhs = lowering_utilities.convert(
            builder.load_var(lhs), target_mlir_type
        )
        rhs = lowering_utilities.convert(
            builder.load_var(rhs), target_mlir_type
        )
        builder.store_var(target, mlir_op(lhs, rhs))

    return _cg


_BITWISE_OPS = {
    operator.and_: arith.andi,
    operator.iand: arith.andi,
    operator.or_: arith.ori,
    operator.ior: arith.ori,
    operator.xor: arith.xori,
    operator.ixor: arith.xori,
}

_BOOLEAN_SIGNATURES = (
    (types.Boolean, types.Boolean),
    (types.Boolean, types.Integer),
    (types.Integer, types.Boolean),
)


def _invert_cg(builder, target, args, kwargs):
    """Lower ``~x``: xor with all-ones (-1 for ints, true for bools)."""

    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    operand = lowering_utilities.convert(
        builder.load_var(args[0]), target_mlir_type
    )
    fill = 1 if isinstance(target_type, types.Boolean) else -1
    ones = lowering_utilities.constant(fill, operand.type)
    builder.store_var(target, arith.xori(operand, ones))


def register_boolean_bitwise_lowerings() -> None:
    """Register bitwise lowerings absent from numba-cuda-mlir."""

    for op, mlir_op in _BITWISE_OPS.items():
        cg = _make_bitwise_cg(mlir_op)
        for lhs_type, rhs_type in _BOOLEAN_SIGNATURES:
            _math_registry.lower(op, lhs_type, rhs_type)(cg)
    _math_registry.lower(operator.invert, types.Boolean)(_invert_cg)
    _math_registry.lower(operator.invert, types.Integer)(_invert_cg)


register_boolean_bitwise_lowerings()


_COMPARISON_CGS = {
    operator.eq: eq_cg,
    operator.ne: ne_cg,
    operator.lt: lt_cg,
    operator.le: le_cg,
    operator.gt: gt_cg,
    operator.ge: ge_cg,
}


def _bin_op_cg_boolean_unsigned(op, builder, target, args, kwargs):
    """Copy of upstream ``_bin_op_cg`` with Boolean-aware signedness.

    Booleans lower to signless i1 and must select unsigned operations,
    otherwise True (all-ones, -1) orders below False in ordering
    comparisons. Mirrors the ``_is_unsigned_operand`` change on the
    fix-boolean-comparison-lowering branch; the body is otherwise
    identical to upstream.
    """

    assert not kwargs, "add_cg does not accept any keyword arguments"
    assert len(args) == 2, "add_cg expects 2 arguments"
    lhs, rhs = args
    target_type, lhs_type, rhs_type = (
        builder.get_numba_type(target.name),
        builder.get_numba_type(lhs.name),
        builder.get_numba_type(rhs.name),
    )
    target_mlir_type = builder.get_mlir_type(target_type)

    def _is_unsigned_operand(operand_type):
        if isinstance(operand_type, types.Boolean):
            return True
        return (
            isinstance(operand_type, types.Integer)
            and not operand_type.signed
        )

    is_unsigned = (
        _is_unsigned_operand(lhs_type) and _is_unsigned_operand(rhs_type)
    )

    lhs, rhs = builder.load_var(lhs), builder.load_var(rhs)

    # Handle cases where load_var returns Python/numpy scalars instead
    # of MLIR values (module-level constants).
    if not isinstance(lhs, _ir.Value):
        if hasattr(lhs, "item"):
            lhs = lhs.item()
        lhs = lowering_utilities.constant(lhs, target_mlir_type)
    if not isinstance(rhs, _ir.Value):
        if hasattr(rhs, "item"):
            rhs = rhs.item()
        rhs = lowering_utilities.constant(rhs, target_mlir_type)

    unified_type = lowering_utilities.numpy_implicit_type_promotion(
        lhs.type, rhs.type
    )

    if found_op := _lowering_math._get_operation_for_op_and_type(
        op, unified_type, is_unsigned
    ):
        info, op = found_op
        assert op is not None, "Expected operation"
        if info.cast_to_return_type:
            lhs = lowering_utilities.convert(lhs, target_mlir_type)
            rhs = lowering_utilities.convert(rhs, target_mlir_type)
        else:
            lhs, rhs = (
                lowering_utilities.coerce_numpy_scalars_for_binary_op(
                    lhs, rhs
                )
            )
        res = op(lhs, rhs)
    else:
        raise ValueError(
            f"No operation found for {op=} and {target_mlir_type=}"
        )

    res = builder.mlir_convert(res, target_mlir_type)
    builder.store_var(target, res)


def register_boolean_comparison_lowerings() -> None:
    """Register comparisons on Boolean operands with unsigned semantics.

    Upstream registers eq/ne/lt/le/gt/ge for ``(Number, Number)``
    only; Boolean operands raise ``NotImplementedError`` during
    lowering. Matching the fix-boolean-comparison-lowering branch:
    the shared ``_bin_op_cg`` is replaced so Boolean operands select
    unsigned operations (the code generators resolve it as a module
    global at call time), and the six comparison code generators gain
    ``(Boolean, Boolean)`` signatures. Mixed ``(Boolean, Number)`` /
    ``(Number, Boolean)`` pairs are registered as well — Python
    promotes ``bool`` to ``int`` in comparisons, and the shared code
    generator already unifies operand types before comparing.

    Builds that already carry the operand-signedness rework expose a
    two-parameter ``_get_operation_for_op_and_type(op, type)`` that
    reads signedness off the MLIR type; the replacement body calls
    the stock three-parameter form, so it is only installed when that
    form is present.
    """

    import inspect

    stock_params = inspect.signature(
        _lowering_math._get_operation_for_op_and_type
    ).parameters
    if len(stock_params) >= 3:
        _lowering_math._bin_op_cg = _bin_op_cg_boolean_unsigned
    for op, cg in _COMPARISON_CGS.items():
        _math_registry.lower(op, types.Boolean, types.Boolean)(cg)
        _math_registry.lower(op, types.Boolean, types.Number)(cg)
        _math_registry.lower(op, types.Number, types.Boolean)(cg)


register_boolean_comparison_lowerings()


_original_try_extract_constant = lowering_utilities.try_extract_constant


def _try_extract_constant_numpy(value):
    """Unwrap numpy scalar constants before constant extraction.

    numba freezes closure constants such as ``numpy.bool_(True)`` into
    kernels, but numba-cuda-mlir 0.4.0's ``try_extract_constant`` only
    matches Python ``int``/``float``/``bool`` and crashes constructing
    ``ir.Value`` from a numpy scalar. Convert via ``.item()`` first.
    """

    if isinstance(value, numpy.generic):
        value = value.item()
    return _original_try_extract_constant(value)


def register_numpy_constant_shim() -> None:
    """Patch try_extract_constant in every module that imported it."""

    for module in (
        lowering_utilities,
        _lowering_builtins,
        _lowering_cuda,
        _lowering_numpy,
    ):
        module.try_extract_constant = _try_extract_constant_numpy


register_numpy_constant_shim()


_original_lower_const_assign = _mlir_lowering.MLIRLower.lower_const_assign


def _lower_const_assign_numpy(self, target, const):
    """Convert numpy scalar constants before const-assign lowering.

    ``lower_const_assign`` gates on ``isinstance(value, (bool, int,
    float, np.number))``, but ``numpy.bool_`` is not ``np.number``,
    so frozen numpy bool constants fall through to
    ``NotImplementedError``. Normalise to Python scalars first.
    """

    if isinstance(const.value, numpy.generic):
        const = copy.copy(const)
        const.value = const.value.item()
    return _original_lower_const_assign(self, target, const)


_mlir_lowering.MLIRLower.lower_const_assign = _lower_const_assign_numpy


_original_load_var = _mlir_lowering.MLIRLower.load_var


def _load_var_numpy(self, var):
    """Unwrap numpy scalars read out of the lowering varmap.

    Frozen closure/global numpy scalars reach the varmap unconverted
    and crash downstream utilities (``unverified_convert``,
    ``try_extract_constant``) that only accept Python scalars or MLIR
    values. Convert at the single read chokepoint.
    """

    result = _original_load_var(self, var)
    if isinstance(result, numpy.generic):
        result = result.item()
    return result


_mlir_lowering.MLIRLower.load_var = _load_var_numpy


@lowering_utilities.unverified_convert.register(numpy.generic)
def _unverified_convert_numpy_scalar(value, target_type, *, signed=False):
    """Convert numpy scalar values via their Python equivalents.

    ``unverified_convert`` dispatches on the value type and has no
    overload for numpy scalars, which reach it through frozen closure
    constants in multi-stage algorithm loops.
    """

    return lowering_utilities.unverified_convert(
        value.item(), target_type, signed=signed
    )


_original_lower_global_assign = _mlir_lowering.MLIRLower.lower_global_assign


def _lower_global_assign_numpy(self, target, glob):
    """Convert numpy scalar globals before global-assign lowering.

    ``lower_global_assign`` gates on ``isinstance(value, (bool, int,
    float, np.number))``, but ``numpy.bool_`` is not ``np.number``,
    so module-level numpy bool globals referenced in kernels fall
    through to the unsupported-global path. Normalise to Python
    scalars first, matching the widened gate on the
    fix-numpy-scalar-constants branch.
    """

    if isinstance(glob.value, numpy.generic):
        glob = copy.copy(glob)
        glob.value = glob.value.item()
    return _original_lower_global_assign(self, target, glob)


_mlir_lowering.MLIRLower.lower_global_assign = _lower_global_assign_numpy


_original_lower_literal_if_needed = (
    _mlir_lowering.MLIRLower.lower_literal_if_needed
)


def _lower_literal_if_needed_numpy(self, value, numba_type=None):
    """Normalise ``numpy.bool_`` literals before literal lowering.

    ``lower_literal_if_needed`` matches ``np.number()`` literals, but
    ``numpy.bool_`` is not an ``np.number`` and falls through to
    ``NotImplementedError``. Convert to a Python bool, which lowers
    to the same i1 constant the fix-numpy-scalar-constants branch
    emits through its widened match case.
    """

    if isinstance(value, numpy.bool_):
        value = value.item()
    return _original_lower_literal_if_needed(self, value, numba_type)


_mlir_lowering.MLIRLower.lower_literal_if_needed = (
    _lower_literal_if_needed_numpy
)


def _lower_tuple_getitem_nested(builder, target, args, kwargs):
    """Dynamic tuple getitem with nested decomposition and bounds check.

    Port of ``lower_uni_tuple_getitem`` from the
    fix-nested-tuple-dynamic-getitem branch. ``scf.index_switch``
    results must be scalar MLIR types, so when the selected element is
    itself a tuple (e.g. indexing a tuple of coefficient rows with a
    loop variable) the selection is decomposed recursively into one
    switch per leaf position and the result is stored as a Python
    tuple of switch results. A dedicated zero-result switch sets the
    kernel ``IndexError`` code for out-of-range dynamic indices, so
    selections with no leaf switches (empty-tuple elements) are still
    checked.
    """
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir.errors import InternalCompilerError
    from numba_cuda_mlir.lowering.numpy import (
        KERNEL_ERROR_CODES,
        set_error_code_if_zero,
    )
    from numba_cuda_mlir.lowering_utilities import convert, index_of
    from numba_cuda_mlir.mlir.dialect_exts import scf
    from numba_cuda_mlir.numba_cuda.core import ir as numba_ir

    target_type = builder.get_numba_type(target.name)
    tup = builder.load_var(args[0])
    index = (
        builder.load_var(args[1])
        if isinstance(args[1], numba_ir.Var)
        else args[1]
    )
    assert isinstance(tup, tuple), f"Expected Python tuple, got {type(tup)}"
    match tup, index:
        case tuple(), ir.Value():
            tup = builder.lower_literal_if_needed(tup)
            index = index_of(index)
            error_memref = builder._get_or_create_error_global()
            cases = ir.DenseI64ArrayAttr.get(range(len(tup)))

            if error_memref is not None:
                # The bounds check lives in its own zero-result switch
                # so that selections with no leaf switches (empty-tuple
                # elements) are still checked; the leaf switches below
                # only select.
                def oob_default(op):
                    set_error_code_if_zero(
                        error_memref, KERNEL_ERROR_CODES[IndexError]
                    )
                    scf.yield_([])

                def oob_case(op, case_index, case_value):
                    scf.yield_([])

                scf.index_switch(
                    results=[],
                    arg=index,
                    cases=cases,
                    default_body_builder=oob_default,
                    case_body_builder=oob_case,
                )

            def select(candidates, element_type):
                # candidates[i] is the value this selection yields when
                # the runtime index equals i.
                if isinstance(element_type, types.BaseTuple):
                    sub_types = (
                        [element_type.dtype] * element_type.count
                        if isinstance(element_type, types.UniTuple)
                        else list(element_type.types)
                    )
                    return tuple(
                        select(
                            [candidate[i] for candidate in candidates],
                            sub_type,
                        )
                        for i, sub_type in enumerate(sub_types)
                    )

                result_type = builder.get_mlir_type(element_type)

                def default(op):
                    scf.yield_([convert(candidates[0], result_type)])

                def case_builder(op, case_index, case_value):
                    scf.yield_(
                        [convert(candidates[case_value], result_type)]
                    )

                return scf.index_switch(
                    results=[result_type],
                    arg=index,
                    cases=cases,
                    default_body_builder=default,
                    case_body_builder=case_builder,
                )

            result = select(list(tup), target_type)
            builder.store_var(target, result)
            if isinstance(result, ir.Value):
                builder.incref(target_type, result)
        case tuple(), int() as index:
            val = tup[index]
            builder.store_var(target, val)
            if isinstance(val, ir.Value):
                builder.incref(target_type, val)
        case _:
            raise InternalCompilerError(
                f"Tuple index must be an integer, got {type(args[1])}"
            )


def register_nested_tuple_getitem_lowering() -> None:
    """Register the nested-tuple getitem over upstream's version."""

    for container in (types.UniTuple, types.Tuple):
        _np_registry.lower(operator.getitem, container, types.Number)(
            _lower_tuple_getitem_nested
        )


register_nested_tuple_getitem_lowering()


def _int_mod_cg(builder, target, args, kwargs):
    """Lower integer ``%`` with Python floored-modulo semantics.

    Upstream's ``mod_cg`` widens operands through a conversion that
    zero-extends signless integers, so negative ``int32`` dividends
    become huge positive ``int64`` values, and it emits a bare
    ``arith.remsi`` (truncated remainder) where Python requires
    floored modulo. Sign-extend signed operands and add the divisor
    when the remainder's sign disagrees with it.
    """

    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    signed = getattr(target_type, "signed", True)
    lhs, rhs = args
    lhs = lowering_utilities.convert(
        builder.load_var(lhs), target_mlir_type, signed=signed
    )
    rhs = lowering_utilities.convert(
        builder.load_var(rhs), target_mlir_type, signed=signed
    )
    if signed:
        rem = arith.remsi(lhs, rhs)
        zero = lowering_utilities.constant(0, rem.type)
        sign_mismatch = arith.cmpi(
            arith.CmpIPredicate.slt, arith.xori(rem, rhs), zero
        )
        rem_nonzero = arith.cmpi(arith.CmpIPredicate.ne, rem, zero)
        needs_fix = arith.andi(sign_mismatch, rem_nonzero)
        correction = arith.select(needs_fix, rhs, zero)
        result = arith.addi(rem, correction)
    else:
        result = arith.remui(lhs, rhs)
    builder.store_var(target, result)


def _int_floordiv_cg(builder, target, args, kwargs):
    """Lower integer ``//`` with sign-aware operand widening.

    Upstream's ``floordiv_cg`` emits the correct floored division
    (``arith.floordivsi``) but widens operands with zero-extension,
    corrupting any negative runtime operand. Sign-extend signed
    operands; use ``arith.divui`` for unsigned ones.
    """

    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    signed = getattr(target_type, "signed", True)
    lhs, rhs = args
    lhs = lowering_utilities.convert(
        builder.load_var(lhs), target_mlir_type, signed=signed
    )
    rhs = lowering_utilities.convert(
        builder.load_var(rhs), target_mlir_type, signed=signed
    )
    if signed:
        result = arith.floordivsi(lhs, rhs)
    else:
        result = arith.divui(lhs, rhs)
    builder.store_var(target, result)


def register_integer_division_lowerings() -> None:
    """Register floored ``%`` and sign-safe ``//`` for integers."""

    for op in (operator.mod, operator.imod):
        _math_registry.lower(op, types.Integer, types.Integer)(
            _int_mod_cg
        )
    for op in (operator.floordiv, operator.ifloordiv):
        _math_registry.lower(op, types.Integer, types.Integer)(
            _int_floordiv_cg
        )


register_integer_division_lowerings()


_original_static_shared = _lowering_cuda.cuda_static_shared_memory


def _dynamic_region_view(lower, mr_type):
    """Build a view of the whole dynamic shared region at offset 0.

    A private zero-length shared ``memref.global`` is undefined
    behaviour to index, so optimizers sink or delete stores staged
    through it. A view over ``gpu.dynamic_shared_memory`` at byte
    offset zero, sized at runtime from the region's extent, matches
    numba's convention that every zero-length shared array aliases
    the dynamic region base. The view does not advance the running
    byte offset used by runtime-shaped shared arrays.
    """

    with lower.alloca_insertion_point():
        shm_base = lower._get_shared_memory_base()
        zero = arith.constant(result=_T.index(), value=0)
        element_bytes = arith.constant(
            result=_T.index(),
            value=mr_type.element_type.width // 8,
        )
        num_elements = arith.divui(
            _memref.dim(shm_base, zero), element_bytes
        )
        return _memref.view(
            result=mr_type,
            source=shm_base,
            byte_shift=zero,
            sizes=[num_elements],
        )


def _dynamic_region_shared_memory(lower, target, dtype):
    """Lower ``cuda.shared.array(0)`` to the dynamic shared region."""

    element_type = lower.get_storage_type(
        _lowering_cuda._resolve_numba_dtype(lower, dtype)
    )
    mr_type = _ir.MemRefType.get(
        shape=[_ir.ShapedType.get_dynamic_size()],
        element_type=element_type,
        memory_space=lower._get_shared_address_space(),
    )
    lower.store_var(target, _dynamic_region_view(lower, mr_type))


def _static_shared_memory_shim(lower, target, static_shape, dtype, alignas):
    """Route zero-length 1-D shared arrays to the dynamic region."""

    if len(static_shape) == 1 and static_shape[0] == 0:
        return _dynamic_region_shared_memory(lower, target, dtype)
    return _original_static_shared(
        lower, target, static_shape, dtype, alignas
    )


def _request_shared_memory_shim(self, sizes, mr_type):
    """Emit runtime-shaped shared views at the current insertion point.

    Upstream inserts at the end of the entry block, which raises an
    insertion error once the block has a terminator (any shared
    request lowered after control flow) and cannot see size operands
    computed after a branch.
    """

    match mr_type.element_type:
        case _ir.IntegerType() | _ir.FloatType() as t:
            element_bytes = t.width // 8
        case _T.index:
            element_bytes = 8
        case _:
            raise NotImplementedError(
                f"NotImplemented shared memory type {mr_type}."
            )
    assert self.mlir_funcOp
    bytes_op = arith.constant(result=_T.index(), value=element_bytes)
    for size in sizes:
        size = self.mlir_convert(size, _T.index())
        bytes_op = arith.muli(lhs=bytes_op, rhs=size)
    shm_base = self._get_shared_memory_base()
    if self._total_shared_memory_bytes is None:
        self._total_shared_memory_bytes = arith.constant(
            result=_T.index(), value=0
        )
    view = _memref.view(
        result=mr_type,
        source=shm_base,
        byte_shift=self._total_shared_memory_bytes,
        sizes=sizes,
    )
    self._total_shared_memory_bytes = arith.addi(
        lhs=self._total_shared_memory_bytes, rhs=bytes_op
    )
    return view


def _request_dynamic_shared_memory_shim(self, mr_type):
    """Emit the dynamic-region view at the current insertion point.

    Upstream inserts at the end of the entry block, which raises an
    insertion error once the block has a terminator (any
    ``shared.array(0)`` lowered after control flow), offsets the
    view by the running byte total, and consumes the whole region;
    every zero-length shared array must instead alias the region
    base at byte offset zero.
    """

    view = _dynamic_region_view(self, mr_type)
    self._dynamic_shared_memory_values.append(view)
    return view


def _make_dynamic_shared_memory_external(module):
    """Rewrite ``__dynamic_shmem__*`` globals to external linkage.

    With internal linkage the optimizer may assume the zero-length
    object really is zero bytes long, making every indexed access
    out of bounds; external linkage makes the size unknown and
    restores conservative aliasing, matching CUDA C's
    ``extern __shared__`` declaration.
    """

    external = _ir.Attribute.parse("#llvm.linkage<external>")

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    if child.operation.name == "llvm.mlir.global":
                        sym = str(child.attributes["sym_name"])
                        if "__dynamic_shmem__" in sym:
                            child.attributes["linkage"] = external
                    walk(child.operation)

    walk(module.operation)


_original_pre_codegen = _mlir_optimization.run_pre_codegen_patterns


def _pre_codegen_with_external_shmem(module, *args, **kwargs):
    result = _original_pre_codegen(module, *args, **kwargs)
    _make_dynamic_shared_memory_external(module)
    return result


def register_dynamic_shared_memory_shims() -> None:
    """Install the dynamic-shared-memory UB fixes."""

    _lowering_cuda.cuda_static_shared_memory = _static_shared_memory_shim
    _mlir_lowering.MLIRLower._request_shared_memory = (
        _request_shared_memory_shim
    )
    if hasattr(_mlir_lowering.MLIRLower, "_request_dynamic_shared_memory"):
        _mlir_lowering.MLIRLower._request_dynamic_shared_memory = (
            _request_dynamic_shared_memory_shim
        )
    _mlir_optimization.run_pre_codegen_patterns = (
        _pre_codegen_with_external_shmem
    )


register_dynamic_shared_memory_shims()


_original_lower_array_literal = (
    _mlir_lowering.MLIRLower.lower_array_literal
)
_array_literal_count = 0


def _lower_array_literal_shim(self, value):
    """Lower frozen array literals as internal constant globals.

    The upstream lowering materializes each frozen numpy closure
    array with a per-thread device ``malloc`` and no ``free``,
    exhausting the 8 MB default device heap around 64k threads. The
    array's raw bytes are emitted instead as an internal constant
    ``llvm.mlir.global`` (the encoding string constants use, since
    dense-attribute initializers are dropped by the LLVM-7
    translation backend) wrapped in a memref descriptor with an
    identity-layout memref type. Dtypes whose
    storage width differs from the numpy itemsize, and rank-0
    arrays, keep the upstream lowering.
    """

    global _array_literal_count
    dtype_numba = _to_numba_type(value.dtype)
    dtype = self.get_storage_type(dtype_numba)
    raw_byte_storage = (
        isinstance(dtype, (_ir.IntegerType, _ir.FloatType))
        and dtype.width == value.dtype.itemsize * 8
        and value.ndim > 0
    )
    if not raw_byte_storage:
        return _original_lower_array_literal(self, value)

    contiguous = numpy.ascontiguousarray(value)
    databytes = contiguous.tobytes()
    name = f"__cubie_array_literal_{_array_literal_count}"
    _array_literal_count += 1
    array_type = _ir.Type.parse(f"!llvm.array<{len(databytes)} x i8>")
    gpu_block = self.mlir_gpu_module.bodyRegion.blocks[0]
    with _ir.InsertionPoint.at_block_begin(gpu_block):
        _llvm.GlobalOp(
            array_type,
            name,
            _ir.Attribute.parse("#llvm.linkage<internal>"),
            addr_space=0,
            constant=True,
            value=_ir.StringAttr.get(databytes),
            alignment=value.dtype.itemsize,
        )

    ndim = contiguous.ndim
    strides, running = [], 1
    for dim in reversed(contiguous.shape):
        strides.insert(0, running)
        running *= dim
    # The memref type must use the identity layout: memref.copy of a
    # memref whose layout carries symbolic strides lowers to a call to
    # the memrefCopy runtime function, which does not exist on device.
    memref_type = _ir.MemRefType.get(
        shape=contiguous.shape, element_type=dtype
    )
    struct_type = _ir.Type.parse(
        f"!llvm.struct<(ptr, ptr, i64, array<{ndim} x i64>,"
        f" array<{ndim} x i64>)>"
    )

    def insert(container, element, *position):
        return _llvm.insertvalue(
            container=container,
            value=element,
            position=_ir.DenseI64ArrayAttr.get(list(position)),
        )

    with self.alloca_insertion_point():
        pointer = _llvm_ext.addressof(name)
        descriptor = _llvm.UndefOp(struct_type).result
        descriptor = insert(descriptor, pointer, 0)
        descriptor = insert(descriptor, pointer, 1)
        descriptor = insert(descriptor, arith.constant(_T.i64(), 0), 2)
        for axis, dim in enumerate(contiguous.shape):
            descriptor = insert(
                descriptor, arith.constant(_T.i64(), dim), 3, axis
            )
        for axis, stride in enumerate(strides):
            descriptor = insert(
                descriptor, arith.constant(_T.i64(), stride), 4, axis
            )
        return _builtin.unrealized_conversion_cast(
            [memref_type], [descriptor]
        )


def register_array_literal_shim() -> None:
    """Install the constant-global array-literal lowering."""

    _mlir_lowering.MLIRLower.lower_array_literal = (
        _lower_array_literal_shim
    )


register_array_literal_shim()


def _memref_to_llvm_ptr_strided(array, indices, element_type):
    """Convert memref + indices to an LLVM pointer via strided metadata.

    The stock 0.4.0 helper extracts the aligned base pointer and drops
    the memref's offset, so cache-hint and vector accesses through
    sliced views read and write relative to the allocation base
    instead of the slice start. It also recomputes strides from dim
    sizes assuming a dense row-major layout, corrupting genuinely
    strided views. Mirrors the upstream fix (#73, merged 2026-06-15):
    the element offset is the metadata offset plus the index-stride
    products; a scalar index into a higher-rank memref stays a linear
    element index.
    """

    gep_dynamic = getattr(
        lowering_utilities, "GEP_DYNAMIC_INDEX", -2147483648
    )
    metadata = _memref.extract_strided_metadata(array)
    rank = _ir.MemRefType(array.type).rank
    base_ptr_idx = _memref.extract_aligned_pointer_as_index(array)
    base_ptr = lowering_utilities.convert(
        base_ptr_idx, _llvm.PointerType.get()
    )

    linear_idx = lowering_utilities.convert(metadata[1], _T.i64())
    ndim = len(indices)
    if ndim == 1 and rank != 1:
        idx = lowering_utilities.convert(indices[0], _T.i64())
        linear_idx = arith.addi(linear_idx, idx)
    else:
        if ndim != rank:
            raise ValueError(
                f"Expected either a scalar linear index or {rank} "
                f"indices for {array.type}, got {ndim}"
            )
        for dim in range(ndim):
            idx = lowering_utilities.convert(indices[dim], _T.i64())
            stride = lowering_utilities.convert(
                metadata[2 + rank + dim], _T.i64()
            )
            linear_idx = arith.addi(
                linear_idx, arith.muli(idx, stride)
            )
    return _llvm.getelementptr(
        _llvm.PointerType.get(),
        base_ptr,
        [linear_idx],
        [gep_dynamic],
        element_type,
        None,
    )


def register_memref_pointer_offset_shim() -> None:
    """Route memref-to-pointer conversion through strided metadata.

    Rebinds ``memref_to_llvm_ptr`` in every module that imported it by
    name (the cache-hint lowerings in ``lowering.cuda`` and the
    aligned-vector lowerings in ``lowering.vector``). Builds that
    already carry the upstream fix expose
    ``memref_data_pointer_as_index``; the shim no-ops there.
    """

    if hasattr(lowering_utilities, "memref_data_pointer_as_index"):
        return
    from numba_cuda_mlir.lowering import vector as _lowering_vector

    lowering_utilities.memref_to_llvm_ptr = _memref_to_llvm_ptr_strided
    _lowering_cuda.memref_to_llvm_ptr = _memref_to_llvm_ptr_strided
    _lowering_vector.memref_to_llvm_ptr = _memref_to_llvm_ptr_strided


register_memref_pointer_offset_shim()


# The dynamic-shared-memory shims above make explicit LTO safe. Set
# NUMBA_CUDA_MLIR_DISABLE_LTO_OPT=1 to force opt_level=0 on the link.


def register_semantic_local_stack_slots():
    """Keep multiply-assigned compiler locals in value types."""
    lower_class = _mlir_lowering.MLIRLower
    marker = "_cubie_semantic_local_stack_slots"
    if getattr(lower_class, marker, None):
        return

    method_names = (
        "_allocate_stack_slot_for_type",
        "allocate_stack_space_for_vars_with_multiple_assigns",
        "_load_stack_slot",
        "_store_stack_slot",
        "_load_var",
        "store_var",
    )
    try:
        sources = {
            name: inspect.getsource(getattr(lower_class, name))
            for name in method_names
        }
    except (AttributeError, OSError, TypeError) as exc:
        raise RuntimeError(
            "cubie._mlir_compat: cannot inspect numba-cuda-mlir's "
            "local stack-slot lowering; update the compatibility "
            "check for this release."
        ) from exc

    semantic_signatures = (
        "get_mlir_type(var_type)"
        in sources["_allocate_stack_slot_for_type"],
        "get_mlir_type(var_type.dtype)"
        in sources[
            "allocate_stack_space_for_vars_with_multiple_assigns"
        ],
        "get_mlir_type(var_type)"
        in sources[
            "allocate_stack_space_for_vars_with_multiple_assigns"
        ],
        (
            "get_mlir_type(var_type)" in sources["_load_stack_slot"]
            and "from_storage" not in sources["_load_stack_slot"]
        ),
        (
            "as_storage" not in sources["_store_stack_slot"]
            and "from_storage" not in sources["_store_stack_slot"]
            and "value=value" in sources["_store_stack_slot"]
        ),
        (
            "from_storage" not in sources["_load_var"]
            and "memref.load" in sources["_load_var"]
        ),
        (
            "as_storage" not in sources["store_var"]
            and "memref.store" in sources["store_var"]
        ),
    )
    if all(semantic_signatures):
        setattr(lower_class, marker, "upstream")
        return

    stock_fragments = {
        "_allocate_stack_slot_for_type": (
            "self.get_storage_type(var_type)",
        ),
        "allocate_stack_space_for_vars_with_multiple_assigns": (
            "self.get_storage_type(var_type.dtype)",
            "self.get_storage_type(var_type)",
        ),
        "_load_stack_slot": (
            "self.from_storage(var_type, loadOp)",
            "self.get_storage_type(var_type)",
        ),
        "_store_stack_slot": (
            "self.from_storage(var_type",
            "self.as_storage(var_type, value)",
        ),
        "_load_var": (
            "self.from_storage(",
            "var_type.dtype",
        ),
        "store_var": (
            "self.as_storage(var_type.dtype, elem)",
        ),
    }
    if any(
        fragment not in sources[name]
        for name, fragments in stock_fragments.items()
        for fragment in fragments
    ):
        raise RuntimeError(
            "cubie._mlir_compat: numba-cuda-mlir's local stack-slot "
            "lowering no longer matches the storage-type implementation; "
            "update the semantic local-slot shim for this release."
        )

    def allocate_stack_slot_for_type(self, var_type):
        if isinstance(var_type, types.BaseTuple):
            return tuple(
                self._allocate_stack_slot_for_type(element_type)
                for element_type in self._tuple_element_types(var_type)
            )

        slot_type = self.get_mlir_type(var_type)
        if not _mlir_lowering._is_valid_memref_element_type(slot_type):
            return self.alloca(slot_type, count=1)

        memref_type = _ir.MemRefType.get(
            shape=[1], element_type=slot_type
        )
        return _memref.alloca(
            memref=memref_type,
            dynamic_sizes=[],
            symbol_operands=[],
        )

    def allocate_stack_space(self, var_assign_count):
        _mlir_lowering.trace()
        for var_name, count in var_assign_count.items():
            if count <= 1:
                continue
            var_type = self.get_numba_type(var_name)
            if isinstance(var_type, types.NoneType):
                continue
            if isinstance(var_type, types.UniTuple):
                element_type = self.get_mlir_type(var_type.dtype)
                memref_type = _ir.MemRefType.get(
                    shape=[var_type.count],
                    element_type=element_type,
                )
                self.varmap[var_name] = _memref.alloca(
                    memref=memref_type,
                    dynamic_sizes=[],
                    symbol_operands=[],
                )
                continue

            slot = self._allocate_stack_slot_for_type(var_type)
            self.varmap[var_name] = slot
            if isinstance(slot, tuple):
                continue
            if isinstance(slot.type, _ir.MemRefType):
                self._tag_alloca_for_deferred_dbg_declare(
                    var_name, slot
                )
            else:
                _mlir_lowering.trace(
                    "Allocated LLVM stack space for %s variable %s",
                    type(var_type).__name__,
                    var_name,
                )
        if (
            self._debug_full
            and self._di_builder is not None
            and self._di_builder.valid
        ):
            self._allocate_poly_dbg_slots()

    def load_stack_slot(self, var_type, slot):
        if isinstance(var_type, types.BaseTuple):
            assert isinstance(slot, tuple)
            return tuple(
                self._load_stack_slot(element_type, element_slot)
                for element_type, element_slot in zip(
                    self._tuple_element_types(var_type), slot
                )
            )

        if isinstance(slot.type, _ir.MemRefType):
            index = lowering_utilities.index_of(0)
            return _memref.load(memref=slot, indices=[index])

        return _llvm.load(
            res=self.get_mlir_type(var_type), addr=slot
        )

    def store_stack_slot(self, var_type, slot, value):
        if isinstance(var_type, types.BaseTuple):
            assert isinstance(slot, tuple)
            assert isinstance(value, (tuple, list))
            for element_type, element_slot, element_value in zip(
                self._tuple_element_types(var_type), slot, value
            ):
                self._store_stack_slot(
                    element_type, element_slot, element_value
                )
            return

        if isinstance(var_type, types.Optional) and not isinstance(
            value, (_ir.Value, _ir.OpView)
        ):
            value = self._cast_to_optional(
                types.NoneType("none"), var_type, None
            )

        if self.nrt.type_has_nrt_meminfo(var_type) and isinstance(
            value, _ir.Value
        ):
            old = self._load_stack_slot(var_type, slot)
            self.decref(var_type, old)

        if isinstance(slot.type, _ir.MemRefType):
            _memref.store(
                value=value,
                memref=slot,
                indices=[lowering_utilities.index_of(0)],
            )
        else:
            _mlir_lowering.trace(
                "Storing %s to LLVM stack slot",
                type(var_type).__name__,
            )
            _llvm.store(value=value, addr=slot)

    original_load_var = lower_class._load_var

    def load_var(self, var):
        if (
            var.name in self.var_assign_count
            and self.var_assign_count[var.name] > 1
            and not self._is_poly_debug_var(var.name)
        ):
            var_type = self.get_numba_type(var.name)
            slot = self.varmap[var.name]
            if isinstance(var_type, types.UniTuple) and not isinstance(
                slot, tuple
            ):
                return tuple(
                    _memref.load(
                        memref=slot,
                        indices=[lowering_utilities.index_of(index)],
                    )
                    for index in range(var_type.count)
                )
        return original_load_var(self, var)

    original_store_var = lower_class.store_var

    def store_var(self, var, value):
        if (
            var.name in self.var_assign_count
            and self.var_assign_count[var.name] > 1
            and not (
                self._debug_full
                and self._poly_dbg_alloca.get(
                    self._canonical_dbg_var_name(var.name)
                )
                is not None
            )
        ):
            var_type = self.get_numba_type(var.name)
            slot = self.varmap[var.name]
            if isinstance(var_type, types.UniTuple) and not isinstance(
                slot, tuple
            ):
                assert isinstance(value, (tuple, list))
                for index, element in enumerate(value):
                    _memref.store(
                        value=element,
                        memref=slot,
                        indices=[
                            lowering_utilities.index_of(index)
                        ],
                    )
                return
        original_store_var(self, var, value)

    lower_class._allocate_stack_slot_for_type = (
        allocate_stack_slot_for_type
    )
    lower_class.allocate_stack_space_for_vars_with_multiple_assigns = (
        allocate_stack_space
    )
    lower_class._load_stack_slot = load_stack_slot
    lower_class._store_stack_slot = store_stack_slot
    lower_class._load_var = load_var
    lower_class.store_var = store_var
    setattr(lower_class, marker, "shim")


register_semantic_local_stack_slots()


def _lower_builtin_extrema(float_op, integer_op, name):
    """Build a scalar numeric lowering for Python min or max."""

    def lower(builder, target, args, kwargs):
        left = builder.load_var(args[0])
        right = builder.load_var(args[1])
        left, right = lowering_utilities.coerce_numpy_scalars_for_binary_op(
            left, right
        )
        if isinstance(left.type, _ir.FloatType):
            result = float_op(left, right)
        elif isinstance(left.type, _ir.IntegerType):
            result = integer_op(left, right)
        else:
            raise NotImplementedError(
                f"{name} not implemented for type {left.type}"
            )
        builder.store_var(target, result)

    return lower


def register_float_minmax_semantics() -> None:
    """Use Python's non-NaN operand semantics for float min and max."""

    registry = _lowering_builtins.registry
    marker = "_cubie_float_minmax_semantics"
    if getattr(registry, marker, None):
        return

    lower_max_source = inspect.getsource(_lowering_builtins.lower_max)
    lower_min_source = inspect.getsource(_lowering_builtins.lower_min)
    native = (
        "arith.maxnumf" in lower_max_source,
        "arith.minnumf" in lower_min_source,
    )
    if all(native):
        setattr(registry, marker, "upstream")
        return
    if (
        any(native)
        or "arith.maximumf" not in lower_max_source
        or ("arith.minimumf" not in lower_min_source)
    ):
        raise RuntimeError(
            "cubie._mlir_compat: numba-cuda-mlir's float min/max "
            "lowering no longer matches the stock implementation; update "
            "the compatibility shim for this release."
        )

    registry.lower(max, types.Number, types.Number)(
        _lower_builtin_extrema(arith.maxnumf, arith.maxsi, "max")
    )
    registry.lower(min, types.Number, types.Number)(
        _lower_builtin_extrema(arith.minnumf, arith.minsi, "min")
    )
    setattr(registry, marker, "shim")


register_float_minmax_semantics()


def _fold_zero_powers(operation):
    """Replace floating-point powers with zero exponents by one."""

    pow_ops = []

    def collect(op):
        if op.name != "math.powf":
            return _ir.WalkResult.ADVANCE
        exponent_op = op.operands[1].owner
        if getattr(exponent_op, "name", None) != "arith.constant":
            return _ir.WalkResult.ADVANCE
        value = exponent_op.attributes["value"]
        if (
            isinstance(value, _ir.FloatAttr)
            and _ir.FloatAttr(value).value == 0.0
        ):
            pow_ops.append(op)
        return _ir.WalkResult.ADVANCE

    operation.walk(collect)
    for op in pow_ops:
        with _ir.InsertionPoint(op), op.location:
            result = arith.constant(op.results[0].type, 1.0)
        op.results[0].replace_all_uses_with(result)
        op.erase()


class _ZeroPowerPassManager:
    """Run the base pipeline around the constant-zero power fold."""

    def __init__(self, pass_manager, pipeline):
        marker = "convert-math-to-nvvm,"
        before_math, after_math = pipeline.split(marker, maxsplit=1)
        before_math = before_math.rstrip().removesuffix(",")
        self._before_math = pass_manager.parse(before_math + ")")
        self._after_math = pass_manager.parse(
            "builtin.module(" + marker + after_math
        )

    def enable_ir_printing(self, **kwargs):
        self._before_math.enable_ir_printing(**kwargs)
        self._after_math.enable_ir_printing(**kwargs)

    def run(self, operation):
        self._before_math.run(operation)
        _fold_zero_powers(operation)
        self._after_math.run(operation)


def register_zero_power_fold() -> None:
    """Fold zero powers after inlining and before math lowering."""

    marker = "_cubie_zero_power_fold"
    if getattr(_mlir_optimization, marker, None):
        return

    native = (
        hasattr(_mlir_optimization, "get_base_pipeline_parts"),
        hasattr(_optimization, "fold_zero_powers"),
    )
    if all(native):
        setattr(_mlir_optimization, marker, "upstream")
        return
    if any(native):
        raise RuntimeError(
            "cubie._mlir_compat: numba-cuda-mlir carries only part of "
            "the zero-power fold; update the compatibility shim for this "
            "release."
        )

    pipeline = _mlir_optimization.get_base_pipeline()
    pipeline_marker = "convert-math-to-nvvm,"
    optimize_source = inspect.getsource(_mlir_optimization.optimize)
    if pipeline.count(pipeline_marker) != 1 or (
        "PassManager.parse(get_base_pipeline())" not in optimize_source
    ):
        raise RuntimeError(
            "cubie._mlir_compat: numba-cuda-mlir's base pipeline no "
            "longer matches the stock math-lowering path; update the "
            "zero-power shim for this release."
        )

    pass_manager = _mlir_optimization.PassManager

    class CompatPassManager:
        @staticmethod
        def parse(candidate):
            if candidate == _mlir_optimization.get_base_pipeline():
                return _ZeroPowerPassManager(pass_manager, candidate)
            return pass_manager.parse(candidate)

    _mlir_optimization.PassManager = CompatPassManager
    setattr(_mlir_optimization, marker, "shim")


register_zero_power_fold()


# ------------------------------------------------------------------ #
# Compile-time performance patches (numba_cuda frontend)             #
# ------------------------------------------------------------------ #
# The shims below rebind the compiler-frontend performance changes
# carried on the cubie_patch branch of the ccam80/numba-cuda-mlir
# fork so they apply to the stock wheel: lazy PostProcessor liveness,
# string-only error markup, SSA sweeps restricted to def/use blocks,
# memoised callee IR with a structural clone (including the
# preserve_ir form of inline_ir), and bitset liveness fix points.
# All are behaviour-preserving; only compile time changes.
# Each group feature-detects the installed package and no-ops when
# the change is already present (a patched build, or a future release
# that merged it). Upstream PRs: perf-lazy-postproc-liveness (#200),
# perf-lazy-error-markup (#201), perf-ssa-restricted-sweeps (#199),
# perf-inline-callee-ir-cache (#197), perf-liveness-bitsets (#198).
# The former targetconfig-hash and callconstraint-memo groups were
# removed: no measurable effect.
# The numba-cuda lowering-side patches (call-type cache, linear
# singly-assigned scan) have no analogue here: MLIRBackend replaces
# the LLVM lowering entirely. The NumbaError double-highlight fix is
# also inapplicable: the vendored NumbaError inherits Exception
# directly, so no base class re-highlights the message.


_BYTE_BITS = tuple(
    tuple(bit for bit in range(8) if value & (1 << bit))
    for value in range(256)
)


def _compute_live_map(cfg, blocks, var_use_map, var_def_map):
    """
    Find variables that must be alive at the ENTRY of each block.

    The two fix points (forward definition reach, backward liveness)
    run on bitsets: every variable gets a bit index and per-block sets
    become arbitrary-size integers, so the union/intersection work in
    each sweep is machine-word bignum arithmetic instead of hash-set
    element traversal. Large flattened functions have tens of
    thousands of variables live across thousands of blocks, where set
    objects made this analysis dominate compilation.
    """
    index = {}
    names = []
    for use_def_map in (var_def_map, var_use_map):
        for name_set in use_def_map.values():
            for name in name_set:
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
    nbytes = (len(names) + 7) // 8

    def to_bits(name_set):
        buf = bytearray(nbytes)
        for name in name_set:
            i = index[name]
            buf[i >> 3] |= 1 << (i & 7)
        return int.from_bytes(buf, "little")

    offsets = list(blocks.keys())
    def_bits = {offset: to_bits(var_def_map[offset]) for offset in offsets}
    use_bits = {offset: to_bits(var_use_map[offset]) for offset in offsets}

    successors = {
        offset: [out_blk for out_blk, _ in cfg.successors(offset)]
        for offset in offsets
    }
    predecessors = {
        offset: [inc_blk for inc_blk, _ in cfg.predecessors(offset)]
        for offset in offsets
    }

    # Forward: definitions (and uses) of every block that can reach a
    # block, itself included. Ascending label order approximates a
    # topological order, so this converges in a couple of sweeps.
    def_reach_map = {
        offset: def_bits[offset] | use_bits[offset] for offset in offsets
    }
    changed = True
    while changed:
        changed = False
        for offset in offsets:
            cur = def_reach_map[offset]
            for out_blk in successors[offset]:
                merged = def_reach_map[out_blk] | cur
                if merged != def_reach_map[out_blk]:
                    def_reach_map[out_blk] = merged
                    changed = True

    # Backward: push variable usage to predecessors, restricted to
    # variables a definition can reach and not defined in the
    # predecessor itself. Reverse label order approximates a reverse
    # topological order for the same fast convergence.
    live_bits = {offset: use_bits[offset] for offset in offsets}
    changed = True
    while changed:
        changed = False
        for offset in reversed(offsets):
            live_vars = live_bits[offset]
            for inc_blk in predecessors[offset]:
                incoming = (
                    live_vars & def_reach_map[inc_blk]
                ) & ~def_bits[inc_blk]
                merged = live_bits[inc_blk] | incoming
                if merged != live_bits[inc_blk]:
                    live_bits[inc_blk] = merged
                    changed = True

    live_map = {}
    for offset in offsets:
        blob = live_bits[offset].to_bytes(nbytes, "little")
        live = set()
        for byte_pos, byte in enumerate(blob):
            if byte:
                base = byte_pos << 3
                for bit in _BYTE_BITS[byte]:
                    live.add(names[base + bit])
        live_map[offset] = live
    return live_map


def _patch_live_map():
    if hasattr(_nb_analysis, "_BYTE_BITS"):
        return
    stock = _nb_analysis.compute_live_map
    _nb_analysis._BYTE_BITS = _BYTE_BITS
    _nb_analysis.compute_live_map = _compute_live_map
    # ir_utils imports the function by name at module import time.
    if getattr(_nb_ir_utils, "compute_live_map", None) is stock:
        _nb_ir_utils.compute_live_map = _compute_live_map


def _patch_postproc():
    src = inspect.getsource(_nb_postproc.PostProcessor.run)
    if "Only generator info consumes" in src:
        return  # already lazy

    def run(self, emit_dels: bool = False, extend_lifetimes: bool = False):
        """
        Run the following passes over Numba IR:
        - canonicalize the CFG
        - emit explicit `del` instructions for variables
        - compute lifetime of variables
        - compute generator info (if function is a generator function)
        """
        self.func_ir.blocks = _nb_transforms.canonicalize_cfg(
            self.func_ir.blocks
        )
        vlt = _nb_postproc.VariableLifetime(self.func_ir.blocks)
        self.func_ir.variable_lifetime = vlt

        if self.func_ir.is_generator:
            # Only generator info consumes the entry-liveness result
            # (via get_block_entry_vars); non-generator consumers of
            # liveness use the lazily computed properties on
            # VariableLifetime instead, so the fix-point analyses are
            # not run eagerly for them.
            bev = _nb_analysis.compute_live_variables(
                vlt.cfg,
                self.func_ir.blocks,
                vlt.usedefs.defmap,
                vlt.deadmaps.combined,
            )
            for offset, ir_block in self.func_ir.blocks.items():
                self.func_ir.block_entry_vars[ir_block] = bev[offset]

            self.func_ir.generator_info = _nb_postproc.GeneratorInfo()
            self._compute_generator_info()
        else:
            self.func_ir.generator_info = None

        # Emit del nodes, do this last as the generator info parsing
        # generates and then strips dels as part of its analysis.
        if emit_dels:
            self._insert_var_dels(extend_lifetimes=extend_lifetimes)

    _nb_postproc.PostProcessor.run = run


def _patch_error_markup():
    scheme_cls = getattr(_nb_errors, "HighlightColorScheme", None)
    if scheme_cls is None or "ColorShell" not in inspect.getsource(
        scheme_cls._markup
    ):
        return
    from colorama import Style

    def _markup(self, msg, color=None, style=Style.BRIGHT):
        # This only builds a string; it does not write to a stream.
        # Wrapping the standard streams with colorama (ColorShell) is
        # unnecessary for that and was undone before anything printed
        # the string, yet its init/deinit dominated error
        # construction when typing speculatively instantiates many
        # exceptions. Emit the same bytes without touching the
        # streams.
        features = ""
        if color:
            features += color
        if style:
            features += style
        return features + msg + Style.RESET_ALL

    scheme_cls._markup = _markup


def _ssa_find_defs_violators(blocks, cfg):
    """
    Returns
    -------
    res : Tuple[Dict[str, None], Mapping, Mapping]
        The SSA violators in a dictionary of variable names, the
        per-variable definition map (name -> [(assign, label)]) and
        the per-variable use-block map (name -> {label}).
    """
    defs = defaultdict(list)
    uses = defaultdict(set)
    states = dict(defs=defs, uses=uses)
    _nb_ssa._run_block_analysis(blocks, states, _nb_ssa._GatherDefsHandler())
    violators = {k: None for k, vs in defs.items() if len(vs) > 1}
    doms = cfg.dominators()
    for k, use_blocks in uses.items():
        if k not in violators:
            for label in use_blocks:
                dom = doms[label]
                def_labels = {label for _assign, label in defs[k]}
                if not def_labels.intersection(dom):
                    violators[k] = None
                    break
    return violators, defs, uses


def _ssa_run_block_rewrite(blocks, states, handler, relevant_labels=None):
    newblocks = {}
    for label, blk in blocks.items():
        if relevant_labels is not None and label not in relevant_labels:
            # The handler can only change statements that mention the
            # variable being processed, so blocks without a def/use
            # of it pass through unchanged.
            newblocks[label] = blk
            continue
        newblk = _nb_ir.Block(scope=blk.scope, loc=blk.loc)
        newbody = []
        states["label"] = label
        states["block"] = blk
        for stmt in _nb_ssa._run_ssa_block_pass(states, blk, handler):
            assert stmt is not None
            newbody.append(stmt)
        newblk.body = newbody
        newblocks[label] = newblk
    return newblocks


def _ssa_fresh_vars(blocks, varname, def_labels):
    """Rewrite to put fresh variable names"""
    states = _nb_ssa._make_states(blocks)
    states["varname"] = varname
    states["defmap"] = defmap = defaultdict(list)
    newblocks = _ssa_run_block_rewrite(
        blocks, states, _nb_ssa._FreshVarHandler(), def_labels
    )
    return newblocks, defmap


def _ssa_fix_ssa_vars(
    blocks, varname, defmap, cfg, df_plus, cache_list_vars, use_labels
):
    """Rewrite all uses to ``varname`` given the definition map"""
    states = _nb_ssa._make_states(blocks)
    states["varname"] = varname
    states["defmap"] = defmap
    states["phimap"] = phimap = defaultdict(list)
    states["cfg"] = cfg
    states["phi_locations"] = _nb_ssa._compute_phi_locations(df_plus, defmap)
    newblocks = _ssa_run_block_rewrite(
        blocks, states, _nb_ssa._FixSSAVars(cache_list_vars), use_labels
    )
    # insert phi nodes
    for label, philist in phimap.items():
        curblk = newblocks[label]
        # Prepend PHI nodes to the block. Build a fresh block rather
        # than mutating in place: phi locations include pass-through
        # blocks, and input block objects must never be mutated.
        newblk = _nb_ir.Block(scope=curblk.scope, loc=curblk.loc)
        newblk.body = philist + curblk.body
        newblocks[label] = newblk
    return newblocks


def _ssa_run_ssa(blocks):
    """Run SSA reconstruction on IR blocks of a function."""
    if not blocks:
        return {}
    cfg = _nb_ssa.compute_cfg_from_blocks(blocks)
    df_plus = _nb_ssa._iterated_domfronts(cfg)
    violators, defs, uses = _ssa_find_defs_violators(blocks, cfg)
    cache_list_vars = _nb_ssa._CacheListVars()

    for varname in violators:
        # Only blocks that define or use the variable can be changed
        # by its rewrite passes; every other block passes through
        # untouched. The def/use block sets collected up front stay
        # valid throughout: the passes rename assignment targets and
        # uses of the current variable only, and phi nodes introduce
        # only freshly versioned names. The uses map excludes a
        # variable's use on the RHS of an assignment to itself
        # (e.g. ``x = x + 1``), but such a use can only appear in a
        # statement that assigns the variable, so its block is always
        # a def block; the fix pass therefore visits the union.
        def_labels = {label for _assign, label in defs[varname]}
        use_labels = uses[varname] | def_labels
        blocks, defmap = _ssa_fresh_vars(blocks, varname, def_labels)
        blocks = _ssa_fix_ssa_vars(
            blocks,
            varname,
            defmap,
            cfg,
            df_plus,
            cache_list_vars,
            use_labels,
        )

    cfg_post = _nb_ssa.compute_cfg_from_blocks(blocks)
    if cfg_post != cfg:
        raise _nb_errors.CompilerError("CFG mutated in SSA pass")
    return blocks


def _patch_ssa():
    params = inspect.signature(_nb_ssa._fresh_vars).parameters
    if "def_labels" in params:
        return
    _nb_ssa._find_defs_violators = _ssa_find_defs_violators
    _nb_ssa._run_block_rewrite = _ssa_run_block_rewrite
    _nb_ssa._fresh_vars = _ssa_fresh_vars
    _nb_ssa._fix_ssa_vars = _ssa_fix_ssa_vars
    _nb_ssa._run_ssa = _ssa_run_ssa


_callee_ir_cache = weakref.WeakKeyDictionary()


def _clone_callee_ir(func_ir):
    """Structural clone of ``func_ir`` for use as an inline callee.

    Equivalent in effect to deep-copying the IR blocks, but far
    cheaper: a fresh single Scope is created (with its redefinition
    state), every Var is recreated in it, and every statement,
    expression and mutable container is rebuilt. Immutable leaves are
    shared: Loc objects, constant/global/freevar payloads, and any
    non-IR values held in expressions. The clone can be freely
    relabelled, renamed and spliced by ``inline_ir`` without mutating
    the source IR.
    """
    blocks = func_ir.blocks
    old_scope = next(iter(blocks.values())).scope
    new_scope = _nb_ir.Scope(parent=old_scope.parent, loc=old_scope.loc)
    new_scope.redefined.update(old_scope.redefined)
    for name, versions in old_scope.var_redefinitions.items():
        new_scope.var_redefinitions[name] = set(versions)

    varmap = {}
    for name, var in old_scope.localvars._con.items():
        varmap[name] = new_scope.define(name, var.loc)

    def clone_value(value):
        if isinstance(value, _nb_ir.Var):
            return varmap[value.name]
        if isinstance(value, _nb_ir.Expr):
            new_expr = copy.copy(value)
            new_expr._kws = {
                key: clone_value(item) for key, item in value._kws.items()
            }
            return new_expr
        if isinstance(value, list):
            return [clone_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_value(item) for item in value)
        if isinstance(value, dict):
            return {key: clone_value(item) for key, item in value.items()}
        return value

    def clone_stmt(stmt):
        new_stmt = copy.copy(stmt)
        for name, value in tuple(new_stmt.__dict__.items()):
            cloned = clone_value(value)
            if cloned is not value:
                new_stmt.__dict__[name] = cloned
        return new_stmt

    new_blocks = {}
    for label, block in blocks.items():
        new_block = _nb_ir.Block(scope=new_scope, loc=block.loc)
        new_block.body = [clone_stmt(stmt) for stmt in block.body]
        new_blocks[label] = new_block

    new_ir = copy.copy(func_ir)
    new_ir.blocks = new_blocks
    new_ir.block_entry_vars = {}
    return new_ir


def _make_inline_ir():
    def inline_ir(
        self, caller_ir, block, i, callee_ir, callee_freevars,
        arg_typs=None, preserve_ir=True,
    ):
        """Inlines the callee_ir in the caller_ir at statement index i
        of block `block`, callee_freevars are the free variables for
        the callee_ir. If the callee_ir is derived from a function
        `func` then this is `func.__code__.co_freevars`. If `arg_typs`
        is given and the InlineWorker instance was initialized with a
        typemap and calltypes then they will be appropriately updated
        based on the arg_typs. If `preserve_ir` is True, the callee_ir
        object will be copied before mutating, otherwise it will be
        mutated in place.
        """
        # Save a reference to the incoming callee_ir
        callee_ir_original = callee_ir

        if preserve_ir:
            def copy_ir(the_ir):
                kernel_copy = the_ir.copy()
                kernel_copy.blocks = {}
                for block_label, block in the_ir.blocks.items():
                    new_block = copy.deepcopy(the_ir.blocks[block_label])
                    kernel_copy.blocks[block_label] = new_block
                return kernel_copy

            callee_ir = copy_ir(callee_ir)

        if self.validator is not None:
            self.validator(callee_ir)

        scope = block.scope
        instr = block.body[i]
        call_expr = instr.value
        callee_blocks = callee_ir.blocks

        # 1. relabel callee_ir by adding an offset
        max_label = max(
            _nb_ir_utils._the_max_label.next(),
            max(caller_ir.blocks.keys()),
        )
        callee_blocks = _nb_icc.add_offset_to_labels(
            callee_blocks, max_label + 1
        )
        callee_blocks = _nb_icc.simplify_CFG(callee_blocks)
        callee_ir.blocks = callee_blocks
        min_label = min(callee_blocks.keys())
        max_label = max(callee_blocks.keys())
        _nb_ir_utils._the_max_label.update(max_label)
        self.debug_print("After relabel")
        _nb_icc._debug_dump(callee_ir)

        # 2. rename all local variables in callee_ir with new locals
        # created in caller_ir
        callee_scopes = _nb_icc._get_all_scopes(callee_blocks)
        self.debug_print("callee_scopes = ", callee_scopes)
        assert len(callee_scopes) == 1
        callee_scope = callee_scopes[0]
        var_dict = {}
        for var in tuple(callee_scope.localvars._con.values()):
            if var.name not in callee_freevars:
                inlined_name = _nb_icc._created_inlined_var_name(
                    callee_ir.func_id.unique_name, var.name
                )
                new_var = scope.redefine(inlined_name, loc=var.loc)
                callee_scope.redefine(inlined_name, loc=var.loc)
                var_dict[var.name] = new_var
        self.debug_print("var_dict = ", var_dict)
        _nb_icc.replace_vars(callee_blocks, var_dict)
        self.debug_print("After local var rename")
        _nb_icc._debug_dump(callee_ir)

        # 3. replace formal parameters with actual arguments
        callee_func = callee_ir.func_id.func
        args = _nb_icc._get_callee_args(
            call_expr, callee_func, block.body[i].loc, caller_ir
        )

        # 4. Update typemap
        if self._permit_update_type_and_call_maps:
            if arg_typs is None:
                raise TypeError("arg_typs should have a value not None")
            self.update_type_and_call_maps(callee_ir, arg_typs)
            callee_blocks = callee_ir.blocks

        self.debug_print("After arguments rename: ")
        _nb_icc._debug_dump(callee_ir)

        _nb_icc._replace_args_with(callee_blocks, args)
        # 5. split caller blocks into two
        new_blocks = []
        new_block = _nb_ir.Block(scope, block.loc)
        new_block.body = block.body[i + 1 :]
        new_label = _nb_icc.next_label()
        caller_ir.blocks[new_label] = new_block
        new_blocks.append((new_label, new_block))
        block.body = block.body[:i]
        block.body.append(_nb_ir.Jump(min_label, instr.loc))

        # 6. replace Return with assignment to LHS
        topo_order = _nb_icc.find_topo_order(callee_blocks)
        _nb_icc._replace_returns(callee_blocks, instr.target, new_label)

        if (
            instr.target.name in caller_ir._definitions
            and call_expr in caller_ir._definitions[instr.target.name]
        ):
            caller_ir._definitions[instr.target.name].remove(call_expr)

        # 7. insert all new blocks, and add back definitions
        for label in topo_order:
            block = callee_blocks[label]
            block.scope = scope
            _nb_icc._add_definitions(caller_ir, block)
            caller_ir.blocks[label] = block
            new_blocks.append((label, block))
        self.debug_print("After merge in")
        _nb_icc._debug_dump(caller_ir)

        return callee_ir_original, callee_blocks, var_dict, new_blocks

    return inline_ir


def _patch_inline_worker():
    if hasattr(_nb_icc, "_clone_callee_ir"):
        return
    _nb_icc._clone_callee_ir = _clone_callee_ir
    _nb_icc._callee_ir_cache = _callee_ir_cache

    worker = _nb_icc.InlineWorker
    if "preserve_ir" not in inspect.signature(worker.inline_ir).parameters:
        worker.inline_ir = _make_inline_ir()

    def inline_function(self, caller_ir, block, i, function, arg_typs=None):
        """Inlines the function in the caller_ir at statement index i
        of block `block`. If `arg_typs` is given and the InlineWorker
        instance was initialized with a typemap and calltypes then
        they will be appropriately updated based on the arg_typs.
        """
        callee_ir = self._fresh_callee_ir(function)
        freevars = function.__code__.co_freevars
        return self.inline_ir(
            caller_ir, block, i, callee_ir, freevars,
            arg_typs=arg_typs, preserve_ir=False,
        )

    def _fresh_callee_ir(self, function, enable_ssa=False):
        """Return callee IR that is safe for ``inline_ir`` to mutate.

        The canonical IR produced by the untyped pipeline for a given
        function and flags configuration is cached, and each call
        site receives a structural clone of it. Running the untyped
        pipeline is far more expensive than cloning, and deeply
        nested inline='always' functions otherwise recompile their
        whole subtree at every transitive call site.
        """
        per_func = _callee_ir_cache.setdefault(function, {})
        key = (str(self.flags), enable_ssa)
        canonical_ir = per_func.get(key)
        if canonical_ir is None:
            canonical_ir = self.run_untyped_passes(function, enable_ssa)
            per_func[key] = canonical_ir
        return _clone_callee_ir(canonical_ir)

    worker.inline_function = inline_function
    worker._fresh_callee_ir = _fresh_callee_ir


_PERF_PATCH_GROUPS = {
    "liveness": _patch_live_map,
    "postproc": _patch_postproc,
    "errors": _patch_error_markup,
    "ssa": _patch_ssa,
    "inline": _patch_inline_worker,
}


def apply_compiler_perf_patches() -> None:
    """Apply all frontend perf patch groups the installed wheel needs.

    Set CUBIE_DISABLE_NUMBA_PERF_PATCHES=1 to skip every group, for
    A/B benchmarking and for isolating suspected patch regressions.
    Set CUBIE_NUMBA_PERF_PATCH_GROUPS to a comma-separated subset of
    liveness, postproc, errors, ssa, inline to apply only those
    groups (per-feature A/B).
    """
    import os

    if os.environ.get("CUBIE_DISABLE_NUMBA_PERF_PATCHES", "0") == "1":
        return
    selected = os.environ.get("CUBIE_NUMBA_PERF_PATCH_GROUPS", "all")
    if selected.strip().lower() == "all":
        names = list(_PERF_PATCH_GROUPS)
    else:
        names = [n.strip() for n in selected.split(",") if n.strip()]
        unknown = [n for n in names if n not in _PERF_PATCH_GROUPS]
        if unknown:
            raise ValueError(
                f"Unknown perf patch group(s) {unknown}; valid: "
                f"{sorted(_PERF_PATCH_GROUPS)}"
            )
    for name in names:
        _PERF_PATCH_GROUPS[name]()


apply_compiler_perf_patches()


# ------------------------------------------------------------------ #
# Iterative SSA reaching-definition search                            #
# ------------------------------------------------------------------ #
# The stock reaching-definition search in numba_cuda/core/ssa.py is a
# mutual recursion between _find_def_from_top and
# _find_def_from_bottom at two Python frames per CFG block, so large
# flattened kernels (exactly the shape cubie's generated loops take
# after inlining) raise RecursionError inside reconstruct_ssa. These
# methods mirror the ssa-iterative-def-search branch of the
# ccam80/numba-cuda-mlir fork: an explicit worklist bounds the search
# by memory instead of the interpreter recursion limit, and
# predecessors are pushed in reverse so phi creation order — and thus
# fresh-variable numbering — matches the recursive formulation
# exactly.


def _ssa_find_def_from_top(self, states, label, loc):
    """Find definition reaching the top of the block at ``label``."""

    return self._find_def_iteratively(states, label, loc, from_top=True)


def _ssa_find_def_from_bottom(self, states, label, loc):
    """Find definition from within the block at ``label``."""

    return self._find_def_iteratively(
        states, label, loc, from_top=False
    )


def _ssa_find_def_iteratively(self, states, label, loc, from_top):
    """Drive the def search on an explicit worklist.

    Each ``pending`` item is a ``(phinode, pred, loc)`` triple whose
    resolved incoming definition must be appended to ``phinode``.
    """

    pending = []
    result = self._walk_def_chain(states, label, loc, from_top, pending)
    while pending:
        phinode, pred, philoc = pending.pop()
        incoming_def = self._walk_def_chain(
            states, pred, philoc, False, pending
        )
        phinode.value.incoming_values.append(incoming_def.target)
        phinode.value.incoming_blocks.append(pred)
    return result


def _ssa_walk_def_chain(self, states, label, loc, from_top, pending):
    """Walk one def-search chain without recursion.

    Alternates the *from-bottom* step (take the block's last
    definition, if any) with the *from-top* step (insert a phi node,
    or hop to the immediate dominator). A phi node is registered in
    ``defmap`` before its predecessors are resolved, so a chain that
    revisits the block terminates there; resolution of the phi's
    incoming values is deferred onto ``pending``.
    """

    cfg = states["cfg"]
    defmap = states["defmap"]
    phimap = states["phimap"]
    phi_locations = states["phi_locations"]

    while True:
        if not from_top:
            defs = defmap[label]
            if defs:
                return defs[-1]
            from_top = True

        if label in phi_locations:
            scope = states["scope"]
            loc = states["block"].loc
            freshvar = scope.redefine(states["varname"], loc=loc)
            phinode = _nb_ir.Assign(
                target=freshvar,
                value=_nb_ir.Expr.phi(loc=loc),
                loc=loc,
            )
            defmap[label].insert(0, phinode)
            phimap[label].append(phinode)
            # Defer the search for the phi's incoming values;
            # reversed so they resolve in predecessor order.
            preds = [pred for pred, _ in cfg.predecessors(label)]
            for pred in reversed(preds):
                pending.append((phinode, pred, loc))
            return phinode
        else:
            idom = cfg.immediate_dominators()[label]
            if idom == label:
                _nb_ssa._warn_about_uninitialized_variable(
                    states["varname"], loc
                )
                return _nb_ssa.UndefinedVariable
            label = idom
            from_top = False


def register_iterative_ssa_def_search() -> None:
    """Make the SSA reaching-definition search iterative.

    No-ops on builds whose ``_FixSSAVars`` already carries
    ``_walk_def_chain`` (a patched build, or a future release that
    merged the fix).
    """

    fixer = _nb_ssa._FixSSAVars
    if hasattr(fixer, "_walk_def_chain"):
        return
    fixer._walk_def_chain = _ssa_walk_def_chain
    fixer._find_def_iteratively = _ssa_find_def_iteratively
    fixer._find_def_from_top = _ssa_find_def_from_top
    fixer._find_def_from_bottom = _ssa_find_def_from_bottom


register_iterative_ssa_def_search()


# ------------------------------------------------------------------ #
# Selective fastmath (Python-side subset)                             #
# ------------------------------------------------------------------ #
# The stock wheel coerces ``fastmath`` to a bool and consumes it only
# as module-wide knobs, so numba-cuda's selective form
# (``fastmath={"arcp", ...}``) is rejected and no per-operation
# fast-math semantics reach codegen. The shims below carry the
# Python-side portion of the selective-fastmath branch of the
# ccam80/numba-cuda-mlir fork:
#
# - bool | set | dict | FastMathOptions accepted and normalised;
# - every fastmath-capable arith/math op stamped with
#   ``#arith.fastmath<...>``;
# - f32 division under ``arcp``/``fast`` rewritten to
#   ``__nv_fast_fdividef`` (so it compiles to ``div.approx.f32``);
# - f32 ``math.tanh`` under ``afn``/``fast`` on sm_75+ rewritten to
#   the hardware ``tanh.approx.f32``;
# - the module-level knobs (``#nvvm.target flags={fast}``, libnvvm
#   options on the modern path, LTO-link ftz/fma/prec options)
#   implied per-flag instead of by truthiness.
#
# The native half of the branch (transferring per-instruction flags
# through the LLVM 7 translation so libnvvm sees them) lives in the
# MLIRToLLVM70 library and cannot be shimmed from Python; on the
# stock wheel the per-op attributes still drive the two MLIR-level
# rewrites above, which are the effects cubie's flag set uses. The
# group no-ops when the installed package provides
# ``numba_cuda_mlir.fastmath`` natively.

_FASTMATH_FLAG_ORDER = (
    "reassoc",
    "nnan",
    "ninf",
    "nsz",
    "arcp",
    "contract",
    "afn",
)
_FAST_FDIVIDEF = "__nv_fast_fdividef"
_fastmath_capable_names_cache = None


def _fastmath_flags(value):
    """Return the LLVM flag-name set for a user-facing value."""

    from numba_cuda_mlir.numba_cuda.core.options import FastMathOptions

    if value is None:
        return set()
    return FastMathOptions(value).flags


def _fastmath_attr(flags):
    """Build ``#arith.fastmath<...>`` for the given flag set."""

    if "fast" in flags:
        mnemonic = "fast"
    else:
        mnemonic = ",".join(
            f for f in _FASTMATH_FLAG_ORDER if f in flags
        )
    return _ir.Attribute.parse(f"#arith.fastmath<{mnemonic}>")


def _fastmath_capable_op_names():
    """Names of arith/math ops that carry a ``fastmath`` attribute.

    Discovered from the generated Python bindings so the set tracks
    the bundled MLIR version.
    """

    global _fastmath_capable_names_cache
    if _fastmath_capable_names_cache is not None:
        return _fastmath_capable_names_cache
    from numba_cuda_mlir._mlir.dialects import _arith_ops_gen
    from numba_cuda_mlir._mlir.dialects import _math_ops_gen

    names = set()
    for module in (_arith_ops_gen, _math_ops_gen):
        for cls in vars(module).values():
            if (
                inspect.isclass(cls)
                and hasattr(cls, "OPERATION_NAME")
                and isinstance(
                    inspect.getattr_static(cls, "fastmath", None),
                    property,
                )
            ):
                names.add(cls.OPERATION_NAME)
    _fastmath_capable_names_cache = frozenset(names)
    return _fastmath_capable_names_cache


def _stamp_fastmath_attrs(func_op, flags):
    """Stamp the fastmath attribute onto every capable nested op."""

    attr = _fastmath_attr(flags)
    capable = _fastmath_capable_op_names()

    def stamp(op):
        if op.name in capable:
            op.attributes["fastmath"] = attr
        return _ir.WalkResult.ADVANCE

    func_op.operation.walk(stamp)


def _chip_number(chip):
    """Return 89 for ``sm_89``; 0 when the chip is unknown."""

    if not chip:
        return 0
    digits = "".join(c for c in str(chip) if c.isdigit())
    return int(digits) if digits else 0


def _rewrite_approx_tanh(func_op, flags, chip):
    """Rewrite f32 ``math.tanh`` to ``tanh.approx.f32`` on sm_75+.

    Runs at stamping time because ``convert-math-to-nvvm`` lowers
    ``math.tanh`` to a plain libdevice call and drops the fastmath
    attribute in the process.
    """

    if not (flags & {"afn", "fast"}) or _chip_number(chip) < 75:
        return

    tanh_ops = []

    def collect(op):
        if op.name == "math.tanh" and isinstance(
            op.results[0].type, _ir.F32Type
        ):
            tanh_ops.append(op)
        return _ir.WalkResult.ADVANCE

    func_op.operation.walk(collect)

    for op in tanh_ops:
        with _ir.InsertionPoint(op), op.location:
            result = _llvm.inline_asm(
                op.results[0].type,
                [op.operands[0]],
                "tanh.approx.f32 $0, $1;",
                "=f,f",
            )
        op.results[0].replace_all_uses_with(result)
        op.erase()


def _fastmath_flag_set_of_op(op):
    """Flag names from an op's ``#llvm.fastmath<...>`` attribute."""

    attrs = op.operation.attributes
    if "fastmathFlags" not in attrs:
        return frozenset()
    text = str(attrs["fastmathFlags"])
    inner = text[text.index("<") + 1 : text.rindex(">")]
    return frozenset(
        flag.strip() for flag in inner.split(",") if flag.strip()
    )


def _rewrite_fast_divisions(module):
    """Lower flagged f32 ``llvm.fdiv`` to ``__nv_fast_fdividef``.

    Mirrors numba-cuda, whose fastmath float32 division calls
    ``__nv_fast_fdividef`` (libnvvm never selects ``div.approx``
    from instruction flags or ``-prec-div=0`` alone). Gating on the
    per-instruction ``arcp``/``fast`` flag keeps the transform
    selective per compiled function.
    """

    from numba_cuda_mlir._mlir.dialects import gpu as _gpu

    worklist = []

    def collect(op):
        if (
            op.name == "llvm.fdiv"
            and isinstance(op.results[0].type, _ir.F32Type)
            and _fastmath_flag_set_of_op(op) & {"fast", "arcp"}
        ):
            worklist.append(op)
        return _ir.WalkResult.ADVANCE

    module.operation.walk(collect)
    if not worklist:
        return

    for gpu_module in module.body:
        if isinstance(gpu_module, _gpu.GPUModuleOp):
            block = gpu_module.regions[0].blocks[0]
            has_decl = any(
                getattr(op, "sym_name", None)
                and op.sym_name.value == _FAST_FDIVIDEF
                for op in block
            )
            if not has_decl:
                decl = _ir.Operation.parse(
                    f"llvm.func @{_FAST_FDIVIDEF}(f32, f32) -> f32"
                )
                _ir.InsertionPoint.at_block_begin(block).insert(decl)

    for op in worklist:
        loc = op.operation.location
        with _ir.InsertionPoint(op), loc:
            call = _llvm.CallOp(
                result=op.results[0].type,
                callee_operands=[op.operands[0], op.operands[1]],
                op_bundle_operands=[],
                op_bundle_sizes=[],
                callee=_FAST_FDIVIDEF,
            )
        op.results[0].replace_all_uses_with(call.results[0])
        op.operation.erase()


def _fastmath_nvvm_knobs(value):
    """Module-level libnvvm/ptxas knobs implied by a flag set.

    A key is absent when the flag set does not speak to that knob:
    ``arcp`` implies ``prec_div=False``, ``afn`` implies
    ``prec_sqrt=False``, ``contract`` implies ``fma=True``, and
    denormal flushing has no per-instruction flag so ``ftz=True`` is
    implied only by full ``fast`` (which enables all four, matching
    numba-cuda's bool-form gating).
    """

    flags = _fastmath_flags(value)
    knobs = {}
    if "fast" in flags:
        knobs["ftz"] = True
    if flags & {"contract", "fast"}:
        knobs["fma"] = True
    if flags & {"arcp", "fast"}:
        knobs["prec_div"] = False
    if flags & {"afn", "fast"}:
        knobs["prec_sqrt"] = False
    return knobs


def register_selective_fastmath_shims() -> None:
    """Install the Python-side selective fastmath support.

    No-ops when the installed package provides
    ``numba_cuda_mlir.fastmath`` natively.
    """

    try:
        import numba_cuda_mlir.fastmath  # noqa: F401

        return
    except ImportError:
        pass

    import dataclasses

    from numba_cuda_mlir import decorators as _decorators
    from numba_cuda_mlir.numba_cuda.core.options import FastMathOptions

    # Defining __eq__ without __hash__ leaves the class unhashable;
    # normalised targetoptions values must stay hashable for
    # dispatch caching.
    if getattr(FastMathOptions, "__hash__", None) is None:
        FastMathOptions.__hash__ = lambda self: hash(
            frozenset(self.flags)
        )

    def verify_fastmath_value(value, targetoptions):
        if value is None:
            return None
        try:
            FastMathOptions(value)
        except ValueError as error:
            return str(error)
        return None

    original_get_schema = _decorators._get_schema

    def get_schema_with_selective_fastmath():
        schema = []
        for option in original_get_schema():
            if option.name in ("fastmath", "fast_math"):
                types = option.types
                if not isinstance(types, tuple):
                    types = (types,)
                option = dataclasses.replace(
                    option,
                    types=types + (set, dict, FastMathOptions),
                    extra_verification=verify_fastmath_value,
                )
            schema.append(option)
        return tuple(schema)

    _decorators._get_schema = get_schema_with_selective_fastmath

    original_verify = _decorators.verify_target_options

    def verify_target_options_normalized(kws):
        targetoptions = original_verify(kws)
        targetoptions["fastmath"] = FastMathOptions(
            targetoptions.get("fastmath") or False
        )
        return targetoptions

    _decorators.verify_target_options = verify_target_options_normalized

    # The #nvvm.target module flag is all-or-nothing; selective
    # subsets are expressed per-op, so the flag keys off full 'fast'
    # only. setup_func_op gates it on truthiness, so it sees a
    # bool-shaped view of the option.
    original_setup_func_op = _mlir_lowering.MLIRLower.setup_func_op

    def setup_func_op_selective(self):
        saved = self.targetoptions
        adjusted = dict(saved)
        adjusted["fastmath"] = "fast" in _fastmath_flags(
            saved.get("fastmath") or False
        )
        self.targetoptions = adjusted
        try:
            return original_setup_func_op(self)
        finally:
            self.targetoptions = saved

    _mlir_lowering.MLIRLower.setup_func_op = setup_func_op_selective

    # Stamp per-op attributes right after the body is lowered, before
    # lower_capi_thunks clones the function, so the C-ABI clone
    # inherits them. Device callees are compiled under their own
    # target options and cloned in pre-stamped, so flags scope
    # per-function exactly as numba-cuda's per-instruction flags do.
    original_lower_body = _mlir_lowering.MLIRLower.lower_function_body

    def lower_function_body_with_fastmath(self):
        result = original_lower_body(self)
        flags = _fastmath_flags(
            self.targetoptions.get("fastmath") or False
        )
        if flags:
            _stamp_fastmath_attrs(self.mlir_funcOp, flags)
            _rewrite_approx_tanh(
                self.mlir_funcOp, flags, self.targetoptions.get("chip")
            )
        return result

    _mlir_lowering.MLIRLower.lower_function_body = (
        lower_function_body_with_fastmath
    )

    previous_pre_codegen = _mlir_optimization.run_pre_codegen_patterns

    def pre_codegen_with_fast_divisions(module, *args, **kwargs):
        result = previous_pre_codegen(module, *args, **kwargs)
        _rewrite_fast_divisions(module)
        return result

    _mlir_optimization.run_pre_codegen_patterns = (
        pre_codegen_with_fast_divisions
    )

    original_nvvm_options = _mlir_optimization._nvvm_options
    nvvm_knob_verified = False

    def verify_nvvm_knob():
        # nvvm_options_selective is a frozen copy of the stock
        # function, differing only in the knob gating, so a wheel
        # that reworks _nvvm_options without providing
        # numba_cuda_mlir.fastmath would be silently overridden by a
        # stale copy. Checked on the first call, like the get_lto_ptx
        # guard below, so `import cubie` stays importable.
        nonlocal nvvm_knob_verified
        if nvvm_knob_verified:
            return
        stock_knob = 'if target_options.get("fastmath"):'
        if stock_knob not in inspect.getsource(original_nvvm_options):
            raise RuntimeError(
                "cubie._mlir_compat: numba-cuda-mlir's _nvvm_options "
                "no longer matches the stock fastmath knob gating; "
                "update the selective fastmath shim for this release."
            )
        nvvm_knob_verified = True

    def nvvm_options_selective(cc, target_options=None, **extra):
        verify_nvvm_knob()
        opts = {"arch": f"compute_{cc}", **extra}
        if target_options is None:
            return opts
        opts.update(
            _fastmath_nvvm_knobs(target_options.get("fastmath", False))
        )
        # -g / -generate-line-info stay omitted: the MLIR pipeline
        # embeds DWARF metadata itself and libnvvm rejects the
        # combination.
        opt = target_options.get("opt")
        if opt is False or opt == 0:
            opts["opt"] = 0
        return opts

    _mlir_optimization._nvvm_options = nvvm_options_selective

    original_get_lto_ptx = _mlir_optimization.get_lto_ptx
    stock_knob_verified = False

    def verify_stock_knob():
        # Checked on the first LTO compile rather than at import so a
        # wheel that reworks get_lto_ptx without providing
        # numba_cuda_mlir.fastmath fails loudly at the point the
        # wrapped knob gating matters, not on every `import cubie`.
        nonlocal stock_knob_verified
        if stock_knob_verified:
            return
        stock_knob = 'target_options.get("fastmath") or None'
        if stock_knob not in inspect.getsource(original_get_lto_ptx):
            raise RuntimeError(
                "cubie._mlir_compat: numba-cuda-mlir's get_lto_ptx no "
                "longer matches the stock fastmath knob gating; update "
                "the selective fastmath shim for this release."
            )
        stock_knob_verified = True

    def get_lto_ptx_selective(cres, linker=None, target_options=None):
        verify_stock_knob()
        if target_options is None:
            target_options = cres.metadata["targetoptions"]
        if (
            linker is None
            and not cres.metadata.get("lto_ptx")
            and cres.metadata.get("linker") is None
        ):
            from numba_cuda_mlir.linker import Linker
            from numba_cuda_mlir.tools import (
                get_gpu_compute_capability,
                parse_compute_capability,
            )

            chip = target_options.get("chip")
            if chip:
                cc = parse_compute_capability(chip)
                arch = chip
            else:
                cc = get_gpu_compute_capability(tuple)
                arch = get_gpu_compute_capability(str)
            knobs = _fastmath_nvvm_knobs(
                target_options.get("fastmath", False)
            )
            linker = Linker(
                cc=cc,
                arch=arch,
                verbose=target_options.get("dump", False),
                debug=target_options.get("debug", False),
                lineinfo=target_options.get("lineinfo", False),
                lto=True,
                ftz=knobs.get("ftz"),
                prec_div=knobs.get("prec_div"),
                prec_sqrt=knobs.get("prec_sqrt"),
                fma=knobs.get("fma"),
                optimization_level=int(
                    target_options.get("opt_level", 3)
                ),
                ptxas_options=target_options.get(
                    "ptxas_options", None
                ),
                max_registers=target_options.get(
                    "max_registers", None
                ),
            )
        return original_get_lto_ptx(cres, linker, target_options)

    _mlir_optimization.get_lto_ptx = get_lto_ptx_selective


register_selective_fastmath_shims()


# ------------------------------------------------------------------ #
# Standalone ftz module knob                                          #
# ------------------------------------------------------------------ #
# numba-cuda enables the libnvvm ftz knob for any truthy ``fastmath``
# value, so identical JITFlags flush denormals there. The installed
# numba-cuda-mlir gates ftz on full ``fast``, which leaves cubie's
# selective sets on the denormal-safe libdevice paths (the
# ``__nv_fast_fdividef`` overflow guard alone costs three extra
# hot-loop instructions per division) and the two backends disagree
# numerically on denormals. The fork's ftz-standalone-flag branch
# accepts ``ftz`` as a module-only flag name; until a wheel carries
# it, wrap ``nvvm_fastmath_options`` so every truthy flag set enables
# the knob, matching numba-cuda's behaviour for the values cubie
# passes. All three consumers (both nvvmCompileProgram paths and the
# LTO-link knobs) import the function lazily, so patching the module
# attribute covers every path.


def register_standalone_ftz_shim() -> None:
    """Enable the libnvvm ftz knob for truthy fastmath flag sets.

    No-ops when the installed package accepts ``ftz`` natively or
    when the required submodule is absent.
    """
    try:
        import numba_cuda_mlir.fastmath as fastmath_module
        from numba_cuda_mlir.numba_cuda.core.options import (
            FastMathOptions,
        )
    except ImportError:
        return

    try:
        FastMathOptions({"ftz"})
        return
    except ValueError:
        pass

    stock_options = fastmath_module.nvvm_fastmath_options

    def nvvm_options_with_ftz(value):
        knobs = dict(stock_options(value))
        if FastMathOptions(value).flags:
            knobs["ftz"] = True
        return knobs

    fastmath_module.nvvm_fastmath_options = nvvm_options_with_ftz


register_standalone_ftz_shim()


# ------------------------------------------------------------------ #
# Contraction belongs to libnvvm                                      #
# ------------------------------------------------------------------ #
# The wheel's optimization pipeline runs math-uplift-to-fma, which
# rewrites contract-flagged mul+add pairs into opaque __nv_fmaf
# calls while the module still sits in alloca/load form — before
# libnvvm has run CSE/GVN — so a product shared by several
# additions is frozen into one FMA per use and computed that many
# times. Leaving flagged mul/add in place lets libnvvm decide
# contraction late with value numbering done: on the gate's Radau
# adaptive kernel this preserves 61 shared products (PTX fma.rn
# 298 -> 214) and cuts kernel cycles by 18%. Only contract-flagged
# ops ever uplift, so stripping the pass is a no-op for unflagged
# code. Mirrors the fork's perf-drop-math-uplift-to-fma branch.


def register_fma_uplift_removal_shim() -> None:
    """Strip math-uplift-to-fma from the optimization pipeline.

    No-ops when the installed package's pipeline already omits the
    pass.
    """
    from numba_cuda_mlir import mlir_optimization

    stock_pipeline = mlir_optimization.get_base_pipeline
    if "math-uplift-to-fma" not in stock_pipeline():
        return

    def pipeline_without_fma_uplift():
        p = stock_pipeline()
        p = p.replace(",math-uplift-to-fma", "")
        p = p.replace("math-uplift-to-fma,", "")
        return p

    mlir_optimization.get_base_pipeline = pipeline_without_fma_uplift


register_fma_uplift_removal_shim()


# ------------------------------------------------------------------ #
# Fastmath knobs on the production LTO link                           #
# ------------------------------------------------------------------ #
# Final code generation for LTO-IR inputs happens in nvJitLink at
# link time; the fastmath knobs handed to libnvvm only govern
# LTO-IR generation, and nvJitLink applies its precise-math
# defaults (ftz=0, prec-div=1, prec-sqrt=1) unless the link passes
# ftz/fma/prec-div/prec-sqrt itself. The wheel passes them to the
# get_lto_ptx() diagnostic linker but builds the production kernel
# linker from _linker_config, which never carries them, so LTO
# cubins revert to precise-math codegen. Knobs derive per flag
# through nvvm_fastmath_options (picking up the standalone-ftz
# wrap above) and are injected only when the link consumes LTO-IR
# inputs — they are meaningless on a PTX-only link. cuda.core
# renders the four knobs as ``-ftz=true``/``false``, which the
# libnvvm parser inside nvJitLink rejects (it accepts ``=0``/
# ``=1`` only), so the rendered option list is rewritten
# numerically as well. Mirrors the fork's
# fix-lto-link-fastmath-knobs branch.
#
# The link-stage knobs deliberately bypass the standalone-ftz wrap:
# ftz keys off full ``fast`` only, per the native per-flag mapping.
# Under link-time ``-ftz=1`` the f32 arithmetic just takes its
# same-throughput .FTZ opcode forms, but the f64->f32 narrowing of
# the loop's time accumulator gains a guard sequence — a magnitude
# compare on the 1/32-rate f64 pipe plus predicated
# denormal-scaling and x1.0 flush multiplies — on the per-step
# dependency chain: +24% kernel time on the gate's explicit RK4
# config, knob-decomposed to ftz alone (fma/prec knobs: +-0%).
# Production LTO cubins have never flushed denormals, so no
# accuracy contract depends on it. The NVVM-stage wrap keeps
# numba-cuda's module-knob behaviour for the non-LTO path, where
# flushing is free.


def _lto_link_fastmath_knobs(fastmath):
    """Return the LTO-link codegen knobs a fastmath value implies.

    Uses this file's per-flag mapping — never the standalone-ftz
    wrap — so link-stage ftz is implied by full ``fast`` only. The
    guard sequences link-time ftz=1 forces around f64->f32
    narrowing cost 24% kernel time on the gate's explicit RK4
    config; see the section comment.
    """

    return _fastmath_nvvm_knobs(fastmath or False)


_LTO_CODEGEN_BOOLEAN_OPTIONS = ("-ftz", "-prec-div", "-prec-sqrt", "-fma")


def _render_lto_codegen_options_numeric(options):
    """Rewrite boolean LTO codegen options to their numeric form."""

    stock_prepare = options._prepare_nvjitlink_options

    def prepare_numeric(as_bytes=False):
        rendered = []
        for opt in stock_prepare(as_bytes=False):
            name, _, value = opt.partition("=")
            if name in _LTO_CODEGEN_BOOLEAN_OPTIONS and value in (
                "true",
                "false",
            ):
                opt = f"{name}={1 if value == 'true' else 0}"
            rendered.append(opt.encode() if as_bytes else opt)
        return rendered

    options._prepare_nvjitlink_options = prepare_numeric


def register_lto_link_fastmath_shim() -> None:
    """Pass fastmath codegen knobs to the production LTO link.

    No-ops on builds whose ``_create_linker`` already derives the
    knobs (the fork's fix-lto-link-fastmath-knobs branch, or a
    release that merged it).
    """

    from numba_cuda_mlir.numba_cuda.cudadrv import driver as _ncm_driver

    if not hasattr(_ncm_driver, "_render_lto_codegen_options_numeric"):
        stock_get_options = _ncm_driver._Linker._get_linker_options

        def get_options_numeric(self, ptx):
            options = stock_get_options(self, ptx)
            _render_lto_codegen_options_numeric(options)
            return options

        _ncm_driver._Linker._get_linker_options = get_options_numeric
        _ncm_driver._render_lto_codegen_options_numeric = (
            _render_lto_codegen_options_numeric
        )

    lower_class = _mlir_lowering.MLIRLower
    marker = "_cubie_lto_link_fastmath"
    if getattr(lower_class, marker, None):
        return

    source = inspect.getsource(lower_class._create_linker)
    if "ftz" in source:
        setattr(lower_class, marker, "upstream")
        return
    fragments = (
        "**self._linker_config",
        "lto=link_plan.compile_new_inputs_as_ltoir",
        "optimize_unused_variables=True",
    )
    if any(fragment not in source for fragment in fragments):
        raise RuntimeError(
            "cubie._mlir_compat: numba-cuda-mlir's kernel linker "
            "construction no longer matches the stock "
            "implementation; update the LTO-link fastmath shim for "
            "this release."
        )

    def _create_linker_with_fastmath(self, link_plan):
        linker_config = dict(self._linker_config)
        if (
            link_plan.compile_new_inputs_as_ltoir
            or link_plan.has_ltoir_link_items
        ):
            linker_config.update(
                _lto_link_fastmath_knobs(
                    self.targetoptions.get("fastmath")
                )
            )
        return _mlir_lowering.Linker(
            **linker_config,
            lto=link_plan.compile_new_inputs_as_ltoir,
            optimize_unused_variables=True,
        )

    lower_class._create_linker = _create_linker_with_fastmath
    setattr(lower_class, marker, "shim")


register_lto_link_fastmath_shim()


def _lower_array_slice_getitem_empty_safe(builder, target, args, kwargs):
    """Lower 1-D array slices, anchoring statically empty ones at 0.

    A compile-time-empty slice ``arr[n:n]`` (bounds frozen into the
    closure, as the buffer registry does for every zero-length
    buffer view) survives initial lowering — the parent crosses the
    device-function boundary as a dynamically shaped memref — but
    the optimization pipeline's inlining and canonicalization fold
    the parent shape and the slice bounds static again, and a
    mid-pipeline verification then rejects the ``memref.subview``
    bounds: "offset 0 is out-of-bounds: n >= n". numpy-style slicing
    allows the empty tail view, and an empty view has no addressable
    elements, so its anchor is arbitrary: statically empty slices
    are rewritten to offset zero with zero size before the upstream
    lowering logic runs. Offset zero stays in bounds no matter how
    far canonicalization folds. SSA-normalizing the original bounds
    instead does not survive: the canonicalizer folds
    ``index_cast(constant)`` chains back into static attributes. The
    body otherwise mirrors upstream.
    """
    from numba_cuda_mlir.lowering.numpy import trace as _np_trace

    _np = _lowering_numpy
    _np_trace()
    mr = builder.load_var(args[0])
    mr_type = mr.type
    dtype = mr_type.element_type
    rank = mr_type.rank
    slc = builder.load_var(args[1])
    start, stop, step = slc.start, slc.stop, slc.step

    def _static_bound(value):
        if value is None or isinstance(value, int):
            return value
        return lowering_utilities.try_extract_constant(value)

    static_start = _static_bound(start)
    static_stop = _static_bound(stop)
    static_step = _static_bound(step)
    statically_empty = (
        isinstance(static_start, int)
        and isinstance(static_stop, int)
        and static_stop <= static_start
        and (step is None or static_step == 1)
    )
    if statically_empty:
        start = _np.index_of(0)
        stop = _np.index_of(0)
        step = _np.index_of(1)

    if start is None:
        start = _np.arith.index_cast(
            _np.arith.constant(result=_np.T.i64(), value=0),
            to=_np.T.index(),
        )
    if stop is None:
        stop = _np.memref.dim(mr, _np.index_of(0))
    if step is None:
        step = _np.index_of(1)
    starts, stops, steps = [start], [stop], [step]
    for i in range(1, rank):
        starts.append(_np.index_of(0))
        stops.append(_np.memref.dim(mr, _np.index_of(i)))
        steps.append(_np.index_of(1))

    dyn = _np.ir.ShapedType.get_dynamic_stride_or_offset()
    source_strides, _ = mr_type.get_strides_and_offset()
    result_strides = []
    for src_stride, step_value in zip(source_strides, steps):
        step_const = lowering_utilities.try_extract_constant(step_value)
        if step_const is not None and src_stride != dyn:
            result_strides.append(src_stride * step_const)
        else:
            result_strides.append(dyn)
    layout = _np.ir.StridedLayoutAttr.get(
        offset=dyn, strides=result_strides
    )
    mrt = _np.ir.MemRefType.get(
        element_type=dtype,
        shape=[dyn for _ in range(rank)],
        layout=layout,
        memory_space=mr_type.memory_space,
    )
    sizes = [
        (stop_v - start_v) // step_v
        for start_v, stop_v, step_v in zip(starts, stops, steps)
    ]
    view = _np.memref.subview(
        mr, offsets=starts, sizes=sizes, strides=steps, result_type=mrt
    )
    builder.store_var(target, view)


def register_empty_slice_anchor_shim() -> None:
    """Anchor statically empty array slices at offset zero.

    Verifies the stock 1-D slice lowering still matches the copied
    body before overriding it, mirroring the other source-checked
    shims in this module.
    """

    stock_source = inspect.getsource(
        _lowering_numpy.lower_array_slice_getitem
    )
    fragments = (
        "starts, stops, steps = [start], [stop], [step]",
        "memref.subview(mr, offsets=starts",
        "(stop - start) // step",
    )
    if any(fragment not in stock_source for fragment in fragments):
        raise RuntimeError(
            "cubie._mlir_compat: numba-cuda-mlir's array slice "
            "lowering no longer matches the stock implementation; "
            "update the empty-slice anchor shim for this release."
        )
    _np_registry.lower(
        operator.getitem, types.Array, types.SliceType
    )(_lower_array_slice_getitem_empty_safe)


register_empty_slice_anchor_shim()
