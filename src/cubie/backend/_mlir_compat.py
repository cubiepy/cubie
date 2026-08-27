"""Lowering registrations that fill gaps in numba-cuda-mlir.

numba-cuda-mlir lowers shared memory to zero-sized
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

numba-cuda-mlir also uses ABI storage types for multiply-assigned
compiler locals. Boolean locals therefore cross an i1/i8 boundary on
every stack load and store. This module keeps scalar and tuple locals
in their semantic value types. External arrays and ABI-facing data
retain their storage types.

numba-cuda-mlir also gives Python ``min`` and ``max`` NaN-propagating
float semantics. This module selects the non-NaN operand.

numba-cuda-mlir also registers comparisons for matching operand kinds
only. This module adds the mixed ``(Boolean, Number)`` pairs Python's
bool-to-int promotion allows.

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
(fix-dynamic-shared-memory-ub, ssa-iterative-def-search,
fix-float-minmax-lowering).

The iterative SSA def-search shim removes the RecursionError that
large flattened kernels hit inside ``reconstruct_ssa``; it no-ops
on builds that carry the fix natively. Remove each shim once its
fix lands upstream. With the shared-memory shim in place CuBIE
requests LTO-link optimization explicitly; set
NUMBA_CUDA_MLIR_DISABLE_LTO_OPT=1 to force opt_level=0 on the LTO
link.

numba-cuda-mlir also applies its AST transforms (``consteval``) only
to the function ``compile_mlir`` receives; inlined callees are built
from their untransformed ``py_func``. This module transforms each
callee before the inline worker builds its IR, and fills statement
bodies the transforms leave empty with ``pass``.

Modified numba-cuda-mlir source: (c) NVIDIA CORPORATION; Apache 2.0.
"""

import ast
import copy
import inspect
import operator
import types as types_module
import warnings
import weakref
from collections import defaultdict

from numba_cuda_mlir import ast_transforms as _ast_transforms
from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir import mlir_lowering as _mlir_lowering
from numba_cuda_mlir import mlir_optimization as _mlir_optimization
from numba_cuda_mlir.ast_transforms import (
    ASTTransformPass,
    apply_ast_transforms,
)
from numba_cuda_mlir.cuda.experimental import consteval
from numba_cuda_mlir._mlir import ir as _ir
from numba_cuda_mlir._mlir.dialects import (
    arith,
    llvm as _llvm,
    memref as _memref,
)
from numba_cuda_mlir._mlir.extras import types as _T
from numba_cuda_mlir.lowering import builtins as _lowering_builtins
from numba_cuda_mlir.lowering import cuda as _lowering_cuda
from numba_cuda_mlir.lowering import numpy as _lowering_numpy
from numba_cuda_mlir.lowering.math import (
    eq_cg,
    ge_cg,
    gt_cg,
    le_cg,
    lt_cg,
    ne_cg,
    registry as _math_registry,
)
from numba_cuda_mlir.lowering.numpy import registry as _np_registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import (
    errors as _nb_errors,
    inline_closurecall as _nb_icc,
    ir as _nb_ir,
    ir_utils as _nb_ir_utils,
    ssa as _nb_ssa,
)


_COMPARISON_CGS = {
    operator.eq: eq_cg,
    operator.ne: ne_cg,
    operator.lt: lt_cg,
    operator.le: le_cg,
    operator.gt: gt_cg,
    operator.ge: ge_cg,
}


def register_mixed_boolean_comparison_lowerings() -> None:
    """Register comparisons for mixed Boolean/Number operand pairs; upstream
    covers matching pairs only.
    """

    for op, cg in _COMPARISON_CGS.items():
        _math_registry.lower(op, types.Boolean, types.Number)(cg)
        _math_registry.lower(op, types.Number, types.Boolean)(cg)


register_mixed_boolean_comparison_lowerings()


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
        num_elements = arith.divui(_memref.dim(shm_base, zero), element_bytes)
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
    return _original_static_shared(lower, target, static_shape, dtype, alignas)


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
            "cubie.backend._mlir_compat: cannot inspect numba-cuda-mlir's "
            "local stack-slot lowering; update the compatibility "
            "check for this release."
        ) from exc

    semantic_signatures = (
        "get_mlir_type(var_type)" in sources["_allocate_stack_slot_for_type"],
        "get_mlir_type(var_type.dtype)"
        in sources["allocate_stack_space_for_vars_with_multiple_assigns"],
        "get_mlir_type(var_type)"
        in sources["allocate_stack_space_for_vars_with_multiple_assigns"],
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
        "_allocate_stack_slot_for_type": ("self.get_storage_type(var_type)",),
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
        "store_var": ("self.as_storage(var_type.dtype, elem)",),
    }
    if any(
        fragment not in sources[name]
        for name, fragments in stock_fragments.items()
        for fragment in fragments
    ):
        raise RuntimeError(
            "cubie.backend._mlir_compat: numba-cuda-mlir's local stack-slot "
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

        memref_type = _ir.MemRefType.get(shape=[1], element_type=slot_type)
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
                self._tag_alloca_for_deferred_dbg_declare(var_name, slot)
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

        return _llvm.load(res=self.get_mlir_type(var_type), addr=slot)

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
                        indices=[lowering_utilities.index_of(index)],
                    )
                return
        original_store_var(self, var, value)

    lower_class._allocate_stack_slot_for_type = allocate_stack_slot_for_type
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
            "cubie.backend._mlir_compat: numba-cuda-mlir's float min/max "
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


# ------------------------------------------------------------------ #
# Compile-time performance patches (numba_cuda frontend)             #
# ------------------------------------------------------------------ #
# The shims below rebind the compiler-frontend performance changes
# carried on the cubie_patch branch of the ccam80/numba-cuda-mlir
# fork so they apply to the stock wheel: SSA sweeps restricted to
# def/use blocks and memoised callee IR with a structural clone
# (including the preserve_ir form of inline_ir).
# All are behaviour-preserving; only compile time changes.
# Each group feature-detects the installed package and no-ops when
# the change is already present (a patched build, or a future release
# that merged it). Upstream PRs: perf-ssa-restricted-sweeps (#199),
# perf-inline-callee-ir-cache (#197). The lazy-postproc-liveness,
# lazy-error-markup, and liveness-bitsets groups were removed after
# upstream merged #200, #201, and #198; the former targetconfig-hash
# and callconstraint-memo groups were removed: no measurable effect.
# The numba-cuda lowering-side patches (call-type cache, linear
# singly-assigned scan) have no analogue here: MLIRBackend replaces
# the LLVM lowering entirely. The NumbaError double-highlight fix is
# also inapplicable: the vendored NumbaError inherits Exception
# directly, so no base class re-highlights the message.


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
        self,
        caller_ir,
        block,
        i,
        callee_ir,
        callee_freevars,
        arg_typs=None,
        preserve_ir=True,
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
            caller_ir,
            block,
            i,
            callee_ir,
            freevars,
            arg_typs=arg_typs,
            preserve_ir=False,
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
    "ssa": _patch_ssa,
    "inline": _patch_inline_worker,
}


def apply_compiler_perf_patches() -> None:
    """Apply all frontend perf patch groups the installed wheel needs.

    Set CUBIE_DISABLE_NUMBA_PERF_PATCHES=1 to skip every group, for
    A/B benchmarking and for isolating suspected patch regressions.
    Set CUBIE_NUMBA_PERF_PATCH_GROUPS to a comma-separated subset of
    ssa, inline to apply only those groups (per-feature A/B).
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
# consteval on inlined device functions                              #
# ------------------------------------------------------------------ #

_CONSTEVAL_OPTIONS = {"experimental_ast_transforms": True}
_consteval_transformed_callees = weakref.WeakKeyDictionary()


class _EmptyBodyRepair(ast.NodeTransformer):
    """Fill statement bodies the transforms emptied with ``pass``."""

    def __init__(self):
        self.modified = False

    def generic_visit(self, node):
        node = super().generic_visit(node)
        body = getattr(node, "body", None)
        if isinstance(body, list) and not body:
            node.body = [ast.copy_location(ast.Pass(), node)]
            self.modified = True
        return node


class _EmptyBodyRepairPass(ASTTransformPass):
    """Pipeline pass restoring compilable bodies after unrolling."""

    @property
    def name(self) -> str:
        return "EmptyBodyRepair"

    def transform(self, tree, context):
        repair = _EmptyBodyRepair()
        tree = repair.visit(tree)
        ast.fix_missing_locations(tree)
        return tree, repair.modified


def _zero_trip_loop_probe(flag, out):
    if flag:
        for i in consteval(range(0)):
            out[i] = 0


def _wheel_repairs_empty_bodies() -> bool:
    """Return whether the wheel's transforms compile emptied bodies."""
    try:
        apply_ast_transforms(_zero_trip_loop_probe, dict(_CONSTEVAL_OPTIONS))
    except ValueError:
        return False
    return True


def _patch_empty_body_repair() -> None:
    """Append the empty-body repair pass to the transform pipeline."""
    if _wheel_repairs_empty_bodies():
        return
    stock_create_default_pipeline = _ast_transforms.create_default_pipeline

    def create_default_pipeline():
        pipeline = stock_create_default_pipeline()
        pipeline.add_pass(_EmptyBodyRepairPass())
        return pipeline

    _ast_transforms.create_default_pipeline = create_default_pipeline


_patch_empty_body_repair()


def _consteval_transformed(function):
    """Return ``function`` with ``consteval`` applied, memoised."""
    if not isinstance(function, types_module.FunctionType):
        return function
    if getattr(function, "_cubie_consteval_transformed", False):
        return function
    cached = _consteval_transformed_callees.get(function)
    if cached is not None:
        return cached
    transformed, source = apply_ast_transforms(
        function, dict(_CONSTEVAL_OPTIONS)
    )
    if source is None:
        function._cubie_consteval_transformed = True
        return function
    # Keep lineinfo on the defining file's lines.
    transformed.__code__ = transformed.__code__.replace(
        co_firstlineno=function.__code__.co_firstlineno
    )
    transformed._cubie_consteval_transformed = True
    _consteval_transformed_callees[function] = transformed
    return transformed


def _wheel_transforms_inlined_callees() -> bool:
    """Return whether the wheel already transforms inlined callees."""
    from numba_cuda_mlir.numba_cuda.core import untyped_passes

    probes = (
        _nb_icc.InlineWorker.inline_function,
        _nb_icc.InlineWorker.run_untyped_passes,
        untyped_passes.InlineInlinables._do_work,
    )
    return any("ast_transform" in inspect.getsource(p) for p in probes)


def _patch_inline_consteval() -> None:
    """Transform callees before the inline worker builds their IR."""
    if _wheel_transforms_inlined_callees():
        return
    worker = _nb_icc.InlineWorker
    stock_inline_function = worker.inline_function
    stock_run_untyped_passes = worker.run_untyped_passes

    def inline_function(self, caller_ir, block, i, function, arg_typs=None):
        return stock_inline_function(
            self,
            caller_ir,
            block,
            i,
            _consteval_transformed(function),
            arg_typs=arg_typs,
        )

    def run_untyped_passes(self, func, enable_ssa=False):
        return stock_run_untyped_passes(
            self, _consteval_transformed(func), enable_ssa
        )

    worker.inline_function = inline_function
    worker.run_untyped_passes = run_untyped_passes


_patch_inline_consteval()


def register_typed_block_scheduler() -> None:
    """Register cubie's typed-IR block scheduler with the backend.

    Warns and no-ops on wheels without the typed-planner hook;
    no-ops silently when ``CUBIE_BLOCK_SCHEDULE`` is ``source``.
    The registered policy folds into the kernel-cache fingerprint.
    """
    from cubie.backend._block_schedule_policies import (
        BLOCK_SCHEDULE_POLICIES,
    )
    from cubie._env import (
        block_schedule_default,
        set_active_block_schedule,
    )

    policy = block_schedule_default()
    if policy == "source":
        return
    if policy not in BLOCK_SCHEDULE_POLICIES:
        raise ValueError(
            f"CUBIE_BLOCK_SCHEDULE={policy!r} is not recognised; "
            f"valid values: {sorted(BLOCK_SCHEDULE_POLICIES)}"
        )
    try:
        from numba_cuda_mlir.extending import (
            register_typed_planner,
        )
    except ImportError:
        warnings.warn(
            f"The block-schedule policy {policy!r} is configured "
            "but the installed numba-cuda-mlir wheel has no "
            "typed-planner hook; kernels compile in source order. "
            "Install cubie-numba-cuda-mlir with the "
            "register_typed_planner hook to enable scheduling."
        )
        return
    from cubie.backend._typed_block_scheduler import (
        TypedBlockScheduler,
    )

    TypedBlockScheduler.policy = policy
    register_typed_planner(TypedBlockScheduler)
    set_active_block_schedule(policy)


register_typed_block_scheduler()


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

    return self._find_def_iteratively(states, label, loc, from_top=False)


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
    layout = _np.ir.StridedLayoutAttr.get(offset=dyn, strides=result_strides)
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

    stock_source = inspect.getsource(_lowering_numpy.lower_array_slice_getitem)
    fragments = (
        "starts, stops, steps = [start], [stop], [step]",
        "memref.subview(mr, offsets=starts",
        "(stop - start) // step",
    )
    if any(fragment not in stock_source for fragment in fragments):
        raise RuntimeError(
            "cubie.backend._mlir_compat: numba-cuda-mlir's array slice "
            "lowering no longer matches the stock implementation; "
            "update the empty-slice anchor shim for this release."
        )
    _np_registry.lower(operator.getitem, types.Array, types.SliceType)(
        _lower_array_slice_getitem_empty_safe
    )


register_empty_slice_anchor_shim()
