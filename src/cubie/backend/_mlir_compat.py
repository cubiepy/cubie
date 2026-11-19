"""Shims mirroring the open numba-cuda-mlir pull requests cubie uses.

Each section applies one fork branch, as merged onto the pinned
``cubie-numba-cuda-mlir`` wheel, at import. Changed lowerings are
registered again for their signatures, replacing the stock ones. The
module then registers cubie's typed-IR block scheduler.

Modified numba-cuda-mlir source: (c) NVIDIA CORPORATION; Apache 2.0.
"""

import ast
import contextvars
import copy
import functools
import inspect
import math
import operator
from collections import defaultdict

from numba_cuda_mlir import ast_transforms as _ast_transforms
from numba_cuda_mlir import descriptor as _descriptor
from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir import mlir_compiler as _mlir_compiler
from numba_cuda_mlir import mlir_lowering as _ml
from numba_cuda_mlir.ast_transforms import (
    apply_ast_transforms,
)
from numba_cuda_mlir.ast_transforms import common as _ast_common
from numba_cuda_mlir.ast_transforms import consteval as _ast_consteval
from numba_cuda_mlir.extending import register_typed_planner
from numba_cuda_mlir.lowering import builtins as _lbuiltins
from numba_cuda_mlir.lowering import math as _lmath
from numba_cuda_mlir.lowering import numpy as _lnumpy
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import (
    errors as _nb_errors,
)
from numba_cuda_mlir.numba_cuda.core import (
    inline_closurecall as _nb_icc,
)
from numba_cuda_mlir.numba_cuda.core import (
    ir as _nb_ir,
)
from numba_cuda_mlir.numba_cuda.core import (
    ir_utils as _nb_ir_utils,
)
from numba_cuda_mlir.numba_cuda.core import (
    ssa as _nb_ssa,
)
from numba_cuda_mlir.numba_cuda.core import (
    untyped_passes as _nb_untyped_passes,
)
from numba_cuda_mlir.numba_cuda import typing as _nb_typing
from numba_cuda_mlir.numba_cuda.typing import templates as _nb_templates
from numba_cuda_mlir.numba_cuda.typing.templates import signature
from numba_cuda_mlir.typing import math as _tmath

# ------------------------------------------------------------------ #
# ccam80/mixed-bool-number                                           #
# ------------------------------------------------------------------ #


# numba-cuda-mlir lowers Boolean/integer comparisons but not
# Boolean/Float ones.


def apply_mixed_bool_number() -> None:
    """Register Boolean/Float comparisons in both operand orders."""
    for op, cg in (
        (operator.ne, _lmath.ne_cg),
        (operator.eq, _lmath.eq_cg),
        (operator.lt, _lmath.lt_cg),
        (operator.le, _lmath.le_cg),
        (operator.gt, _lmath.gt_cg),
        (operator.ge, _lmath.ge_cg),
    ):
        _lmath.registry.lower(op, types.Boolean, types.Float)(cg)
        _lmath.registry.lower(op, types.Float, types.Boolean)(cg)


apply_mixed_bool_number()


# ------------------------------------------------------------------ #
# fix-local-boolean-stack-slots                                      #
# ------------------------------------------------------------------ #


def _allocate_stack_slot_for_type(self, var_type):
    if isinstance(var_type, types.BaseTuple):
        return tuple(
            self._allocate_stack_slot_for_type(elem_type)
            for elem_type in self._tuple_element_types(var_type)
        )

    mlir_type = self.get_mlir_type(var_type)
    if not _ml._is_valid_memref_element_type(mlir_type):
        return self.alloca(mlir_type, count=1)

    memref_type = _ml.ir.MemRefType.get(shape=[1], element_type=mlir_type)
    return _ml.memref.alloca(
        memref=memref_type, dynamic_sizes=[], symbol_operands=[]
    )


def allocate_stack_space_for_vars_with_multiple_assigns(
    self, var_assign_count
):
    _ml.trace()
    for var_name, count in var_assign_count.items():
        if count > 1:
            var_type = self.get_numba_type(var_name)
            if isinstance(var_type, types.NoneType):
                continue
            if isinstance(var_type, types.UniTuple):
                elem_mlir_type = self.get_mlir_type(var_type.dtype)
                memref_type = _ml.ir.MemRefType.get(
                    shape=[var_type.count], element_type=elem_mlir_type
                )
                self.varmap[var_name] = _ml.memref.alloca(
                    memref=memref_type, dynamic_sizes=[], symbol_operands=[]
                )
                continue
            if isinstance(var_type, types.BaseTuple):
                self.varmap[var_name] = self._allocate_stack_slot_for_type(
                    var_type
                )
                continue
            mlir_type = self.get_mlir_type(var_type)

            if not _ml._is_valid_memref_element_type(mlir_type):
                self.varmap[var_name] = self.alloca(mlir_type, count=1)
                _ml.trace(
                    f"Allocated LLVM stack space for "
                    f"{type(var_type).__name__} "
                    f"variable {var_name} (mlir type {mlir_type})"
                )
            else:
                memref_type = _ml.ir.MemRefType.get(
                    shape=[1], element_type=mlir_type
                )
                self.varmap[var_name] = _ml.memref.alloca(
                    memref=memref_type, dynamic_sizes=[], symbol_operands=[]
                )
                self._tag_alloca_for_deferred_dbg_declare(
                    var_name, self.varmap[var_name]
                )
    if (
        self._debug_full
        and self._di_builder is not None
        and self._di_builder.valid
    ):
        self._allocate_poly_dbg_slots()


def _load_stack_slot(self, var_type, slot):
    if isinstance(var_type, types.BaseTuple):
        assert isinstance(slot, tuple)
        return tuple(
            self._load_stack_slot(elem_type, elem_slot)
            for elem_type, elem_slot in zip(
                self._tuple_element_types(var_type), slot
            )
        )

    if isinstance(slot.type, _ml.MemRefType):
        _ml.trace("")
        index = _ml.index_of(0)
        _ml.trace("index=%s", index)
        loadOp = _ml.memref.load(memref=slot, indices=[index])
        _ml.trace("loadOp=%s", loadOp)
        return loadOp

    _ml.trace("Loading %s from LLVM stack slot", type(var_type).__name__)
    return _ml.llvm.load(res=self.get_mlir_type(var_type), addr=slot)


def _load_var(self, var):
    """
    Load the value from the given numba variable.
    """
    _ml.trace("var=%s", var)
    if isinstance(var, (list, tuple)):
        return [self.load_var(v) for v in var]

    assert self.var_lowered(var), f"Var {var.name} not found in varmap."

    if self._is_poly_debug_var(var.name):
        return self._load_poly_debug_var(var.name)

    if (
        var.name in self.var_assign_count
        and self.var_assign_count[var.name] > 1
    ):
        # if variable is stack allocated (multiple assigned),
        # load the variable from stack
        var_type = self.get_numba_type(var.name)
        slot = self.varmap[var.name]

        # UniTuple multi-assign uses a packed memref; heterogeneous
        # BaseTuple multi-assign uses per-element stack slots.
        if isinstance(var_type, types.UniTuple) and not isinstance(
            slot, tuple
        ):
            return tuple(
                _ml.memref.load(memref=slot, indices=[_ml.index_of(i)])
                for i in range(var_type.count)
            )

        return self._load_stack_slot(var_type, slot)
    elif var.name in self._debug_forced_alloca:
        # variable forced to memref.alloca for debug info.
        return _ml.memref.load(
            memref=self.varmap[var.name], indices=[_ml.index_of(0)]
        )
    else:
        _ml.trace("")
        # the variable is promoted to register,
        # load the value from varmap
        return self.varmap[var.name]


def _store_stack_slot(self, var_type, slot, value):
    if isinstance(var_type, types.BaseTuple):
        assert isinstance(slot, tuple)
        assert isinstance(value, (tuple, list))
        for elem_type, elem_slot, elem_value in zip(
            self._tuple_element_types(var_type), slot, value
        ):
            self._store_stack_slot(elem_type, elem_slot, elem_value)
        return

    if isinstance(var_type, types.Optional) and not isinstance(
        value, (_ml.ir.Value, _ml.ir.OpView)
    ):
        value = self._cast_to_optional(types.NoneType("none"), var_type, None)

    if self.nrt.type_has_nrt_meminfo(var_type) and isinstance(
        value, _ml.ir.Value
    ):
        if isinstance(slot.type, _ml.MemRefType):
            old = _ml.memref.load(memref=slot, indices=[_ml.index_of(0)])
        else:
            old = _ml.llvm.load(res=self.get_mlir_type(var_type), addr=slot)
        self.decref(var_type, old)

    if isinstance(slot.type, _ml.MemRefType):
        _ml.memref.store(value=value, memref=slot, indices=[_ml.index_of(0)])
    else:
        _ml.trace("Storing %s to LLVM stack slot", type(var_type).__name__)
        _ml.llvm.store(value=value, addr=slot)


def store_var(self, var, value):
    """
    Store the value (MLIR Op) into the given variable.
    """
    _ml.trace("var=%s value=%s", var, value)
    if self._debug_full:
        base_name = self._canonical_dbg_var_name(var.name)
        if self._poly_dbg_alloca.get(base_name) is not None:
            self._store_poly_dbg_var(var.name, value)
            return
    if (
        var.name in self.var_assign_count
        and self.var_assign_count[var.name] > 1
    ):
        # if variable is stack allocated (multiple assigned),
        # store the value to stack
        assert self.var_lowered(var), (
            f"Stack allocated var {var.name} not found in varmap."
        )

        slot = self.varmap[var.name]
        var_type = self.get_numba_type(var.name)

        # UniTuple multi-assign uses a packed memref; heterogeneous
        # BaseTuple multi-assign uses per-element stack slots.
        if isinstance(var_type, types.UniTuple) and not isinstance(
            slot, tuple
        ):
            assert isinstance(value, (tuple, list))
            for i, elem in enumerate(value):
                _ml.memref.store(
                    value=elem, memref=slot, indices=[_ml.index_of(i)]
                )
            return

        self._store_stack_slot(var_type, slot, value)
    else:
        # the value can be safely stored in register,
        # register the value in varmap
        assert not self.var_lowered(var) or isinstance(
            self.get_numba_type(var.name), types.BaseTuple
        ), f"Var {var.name} already defined in varmap."
        numba_type = self._get_numba_type_for_dbg_var(var.name)
        if (
            self._debug_full
            and isinstance(numba_type, types.Complex)
            and isinstance(value, (_ml.ir.Value, _ml.ir.OpView))
        ):
            # Force single-assign complex vars onto stack so deferred
            # dbg.declare has a stable pointer location after
            # memref->LLVM lowering.
            mlir_value = (
                value.result if isinstance(value, _ml.ir.OpView) else value
            )
            memref_type = _ml.ir.MemRefType.get(
                shape=[1], element_type=mlir_value.type
            )
            alloca_op = _ml.memref.alloca(
                memref=memref_type, dynamic_sizes=[], symbol_operands=[]
            )
            _ml.memref.store(
                value=mlir_value, memref=alloca_op, indices=[_ml.index_of(0)]
            )
            self.varmap[var.name] = alloca_op
            self._debug_forced_alloca.add(var.name)
            self._tag_alloca_for_deferred_dbg_declare(var.name, alloca_op)
        else:
            self.varmap[var.name] = value

    self._emit_dbg_value(var.name, value)


def apply_fix_local_boolean_stack_slots() -> None:
    """Keep multiply-assigned compiler locals in their value types."""
    lower_class = _ml.MLIRLower
    lower_class._allocate_stack_slot_for_type = _allocate_stack_slot_for_type
    lower_class.allocate_stack_space_for_vars_with_multiple_assigns = (
        allocate_stack_space_for_vars_with_multiple_assigns
    )
    lower_class._load_stack_slot = _load_stack_slot
    lower_class._load_var = _load_var
    lower_class._store_stack_slot = _store_stack_slot
    lower_class.store_var = store_var


apply_fix_local_boolean_stack_slots()


# ------------------------------------------------------------------ #
# perf-ssa-restricted-sweeps                                         #
# ------------------------------------------------------------------ #


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
    enter, leave = _ssa_dominator_tree_intervals(cfg)
    for k, use_blocks in uses.items():
        if k not in violators:
            def_labels = [
                label for _assign, label in defs[k] if label in enter
            ]
            for label in use_blocks:
                use_enter = enter[label]
                if not any(
                    enter[d] <= use_enter < leave[d] for d in def_labels
                ):
                    violators[k] = None
                    break
    return violators, defs, uses


def _ssa_dominator_tree_intervals(cfg):
    """Dominator-tree preorder intervals (enter/leave numbers)."""
    domtree = cfg.dominator_tree()
    enter = {}
    leave = {}
    counter = 0
    stack = [(cfg.entry_point(), False)]
    while stack:
        node, done = stack.pop()
        if done:
            leave[node] = counter
            continue
        enter[node] = counter
        counter += 1
        stack.append((node, True))
        stack.extend((child, False) for child in domtree[node])
    return enter, leave


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


def apply_perf_ssa_restricted_sweeps() -> None:
    """Restrict SSA rewrites to blocks defining or using a name."""
    _nb_ssa._find_defs_violators = _ssa_find_defs_violators
    _nb_ssa._run_block_rewrite = _ssa_run_block_rewrite
    _nb_ssa._fresh_vars = _ssa_fresh_vars
    _nb_ssa._fix_ssa_vars = _ssa_fix_ssa_vars
    _nb_ssa._run_ssa = _ssa_run_ssa


apply_perf_ssa_restricted_sweeps()


# ------------------------------------------------------------------ #
# perf-inline-callee-ir-cache                                        #
# ------------------------------------------------------------------ #

_PIPELINE_CALLEE_IR_CACHE_ATTR = "_numba_cuda_callee_ir_cache"


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


def apply_perf_inline_callee_ir_cache() -> None:
    """Cache callee IR per pipeline and clone it for each call site."""
    _nb_icc._clone_callee_ir = _clone_callee_ir
    worker = _nb_icc.InlineWorker
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

        The canonical IR for a function and flags configuration is
        cached on the current compiler pipeline and each call site
        receives a structural clone of it.
        """
        # enable_ssa is rewritten by the pipeline; pin it so keys match.
        self.flags.enable_ssa = enable_ssa
        holder = self.pipeline if self.pipeline is not None else self
        cache = getattr(holder, _PIPELINE_CALLEE_IR_CACHE_ATTR, None)
        if cache is None:
            cache = {}
            setattr(holder, _PIPELINE_CALLEE_IR_CACHE_ATTR, cache)
        key = (function, str(self.flags), enable_ssa)
        canonical_ir = cache.get(key)
        if canonical_ir is None:
            canonical_ir = self.run_untyped_passes(function, enable_ssa)
            cache[key] = canonical_ir
        return _clone_callee_ir(canonical_ir)

    worker.inline_function = inline_function
    worker._fresh_callee_ir = _fresh_callee_ir


apply_perf_inline_callee_ir_cache()


# ------------------------------------------------------------------ #
# inlined-callee-ast-transforms-pr                                   #
# ------------------------------------------------------------------ #


@classmethod
def _consteval_is_consteval_call(cls, node):
    """Check if a node is a call to consteval or literally."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in cls.CONSTEVAL_NAMES
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr in cls.CONSTEVAL_NAMES
    return False


def reads_parameters(func):
    """Return whether a consteval in ``func`` names one of its parameters.

    We look inside ``consteval(...)`` arguments and ``with consteval():``
    bodies, because the consteval pass reads parameter types nowhere else.
    """
    tree = _ast_common.get_function_ast(func)
    if tree is None:
        return False

    is_consteval = _ast_consteval.ConstevalTransformer._is_consteval_call
    parameters = set(inspect.signature(func).parameters)
    for node in ast.walk(tree):
        if is_consteval(node):
            scanned = node.args
        elif isinstance(node, ast.With) and any(
            is_consteval(i.context_expr) for i in node.items
        ):
            scanned = node.body
        else:
            continue
        for part in scanned:
            if any(
                isinstance(n, ast.Name) and n.id in parameters
                for n in ast.walk(part)
            ):
                return True
    return False


def transform_inline_callee(pyfunc, targetoptions, argtypes=None):
    """Apply AST transforms to an inlinee under the caller's options and
    its call's argument types.

    Without ``argtypes`` we return None if a consteval in the inlinee
    names one of its parameters, because the transform then needs the
    argument types.
    """
    if not targetoptions.get("experimental_ast_transforms", False):
        return pyfunc
    if argtypes is None and reads_parameters(pyfunc):
        return None

    transformed, _ = apply_ast_transforms(pyfunc, targetoptions, argtypes)
    return transformed


# While InlineInlinables partially types a caller, it stores a function
# here. MLIRDispatcher.get_call_template asks that function for a call's
# signature, which it gets by typing the callee's transformed IR instead
# of compiling the callee.
inlinee_typer = contextvars.ContextVar("inlinee_typer", default=None)


class InlineeNeedsArgTypes(Exception):
    """Raised inside an inlinee's untyped run to ask the enclosing
    InlineInlinables to type the call and inline the callee again."""


def _pipeline_callee_cache(worker):
    """Return the callee cache of ``worker``'s compiler pipeline.

    Store the cache on the pipeline, or on ``worker`` when it has no
    pipeline, so that Python frees it when the compilation ends.
    """
    holder = worker.pipeline if worker.pipeline is not None else worker
    cache = getattr(holder, _PIPELINE_CALLEE_IR_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(holder, _PIPELINE_CALLEE_IR_CACHE_ATTR, cache)
    return cache


def _inline_worker_fresh_callee_ir(
    self, function, enable_ssa=False, arg_typs=None
):
    """Return callee IR that is safe for ``inline_ir`` to mutate.

    Cache the canonical IR for each function, flags configuration and
    argument types on the current compiler pipeline, and return a
    structural clone of it for each call site.
    """
    # Pin enable_ssa before building the key, because
    # run_untyped_passes rewrites it.
    self.flags.enable_ssa = enable_ssa
    cache = _pipeline_callee_cache(self)
    key = (function, str(self.flags), enable_ssa, arg_typs)
    canonical_ir = cache.get(key)
    if canonical_ir is None:
        canonical_ir = self.run_untyped_passes(
            function, enable_ssa, arg_typs=arg_typs
        )
        cache[key] = canonical_ir
    return _clone_callee_ir(canonical_ir)


def _inline_worker_inline_function(
    self, caller_ir, block, i, function, arg_typs=None
):
    """Inline ``function`` into ``caller_ir`` at statement ``i`` of
    ``block``.

    Run the function's untyped passes with ``arg_typs`` as its argument
    types, or take their result from the pipeline's callee IR cache.
    """
    callee_ir = self._fresh_callee_ir(function, arg_typs=arg_typs)
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


def _inline_worker_type_callee(self, function, arg_typs):
    """Run the untyped passes over a callee in SSA form and type it for
    arg_typs.

    Returns
    -------
    tuple
        The callee's IR and its ``type_inference_stage`` results.
    """
    from numba_cuda_mlir.numba_cuda.core.typed_passes import (
        type_inference_stage,
    )

    callee_ir = self.run_untyped_passes(
        function, enable_ssa=True, arg_typs=arg_typs
    )
    typing = type_inference_stage(
        self.typingctx, self.targetctx, callee_ir, arg_typs, None
    )
    return callee_ir, typing


def _inline_worker_transform_inlinee(self, function, arg_typs=None):
    """Transform an inlinee with the compiling function's inlinee
    transform.

    Return ``function`` unchanged when no inlinee transform is
    configured. Otherwise apply the transform with ``arg_typs`` and
    return its result: ``function`` itself, a new function, or None if
    ``arg_typs`` is None and one of the inlinee's parameters appears
    inside a consteval.

    Store each result in the pipeline's callee cache and return the
    stored function at later call sites of the same inlinee.
    ``_fresh_callee_ir`` keys its IR on that function object.
    """
    if self.inlinee_transform is None:
        return function
    cache = _pipeline_callee_cache(self)
    key = ("transform", function, id(self.targetoptions), arg_typs)
    if key not in cache:
        cache[key] = self.inlinee_transform(
            function, self.targetoptions, arg_typs
        )
    return cache[key]


def _inline_worker_run_untyped_passes(
    self, func, enable_ssa=False, arg_typs=None
):
    """Run the untyped passes over ``func`` and return its Numba IR.

    Without ``arg_typs`` we type every argument as pyobject.
    InlineInlinables reads pyobject arguments as the sign that it is
    running inside such an inline.
    """
    from numba_cuda_mlir.numba_cuda.core.compiler import (
        StateDict,
        _CompileStatus,
    )
    from numba_cuda_mlir.numba_cuda.core.untyped_passes import (
        ExtractByteCode,
    )
    from numba_cuda_mlir.numba_cuda.core import bytecode

    state = StateDict()
    state.func_ir = None
    state.typingctx = self.typingctx
    state.targetctx = self.targetctx
    state.locals = self.locals
    state.pipeline = self.pipeline
    state.flags = self.flags
    state.flags.enable_ssa = enable_ssa

    state.func_id = bytecode.FunctionIdentity.from_function(func)

    state.typemap = None
    state.calltypes = None
    state.type_annotation = None
    state.status = _CompileStatus(False)
    state.return_type = None
    state.metadata = {}
    if self.targetoptions is not None:
        state.metadata["targetoptions"] = self.targetoptions
    if self.inlinee_transform is not None:
        state.metadata["inlinee_transform"] = self.inlinee_transform

    ExtractByteCode().run_pass(state)
    if arg_typs is not None:
        state.args = tuple(arg_typs)
    else:
        # Placeholder args for the object-mode lifting path.
        state.args = len(state.bc.func_id.pysig.parameters) * (types.pyobject,)

    pm = self._compiler_pipeline(state)

    pm.finalize()
    pm.run(state)
    return state.func_ir


def _inline_inlinables_run_pass(self, state):
    """Run inlining of inlinables"""
    if self._DEBUG:
        print("before inline".center(80, "-"))
        print(state.func_ir.dump())
        print("".center(80, "-"))

    inline_worker = _nb_icc.InlineWorker(
        state.typingctx,
        state.targetctx,
        state.locals,
        state.pipeline,
        state.flags,
        validator=_nb_icc.callee_ir_validator,
        targetoptions=state.metadata.get("targetoptions"),
        inlinee_transform=state.metadata.get("inlinee_transform"),
    )
    # We type the caller at most once in this run, in _call_argtypes.
    state.typemap = None

    modified = False
    # use a work list, look for call sites via `ir.Expr.op == call`
    # and pass these to `self._do_work` to decide on inlining.
    work_list = list(state.func_ir.blocks.items())
    while work_list:
        label, block = work_list.pop()
        for i, instr in enumerate(block.body):
            if isinstance(instr, _nb_ir.Assign):
                expr = instr.value
                if isinstance(expr, _nb_ir.Expr) and expr.op == "call":
                    if _nb_untyped_passes.guard(
                        self._do_work,
                        state,
                        work_list,
                        block,
                        i,
                        expr,
                        inline_worker,
                    ):
                        modified = True
                        break  # because block structure changed

    # reset type inference now we are done with the partial results
    state.typemap = None
    state.calltypes = None

    if modified:
        # clean up unconditional branches that appear due to inlined
        # functions introducing blocks
        cfg = _nb_untyped_passes.compute_cfg_from_blocks(state.func_ir.blocks)
        for dead in cfg.dead_nodes():
            del state.func_ir.blocks[dead]
        post_proc = _nb_untyped_passes.postproc.PostProcessor(state.func_ir)
        post_proc.run()
        state.func_ir.blocks = _nb_untyped_passes.simplify_CFG(
            state.func_ir.blocks
        )

    if self._DEBUG:
        print("after inline".center(80, "-"))
        print(state.func_ir.dump())
        print("".center(80, "-"))
    return True


def _inline_inlinables_callee(self, func_ir, expr):
    """Return the object that a call expression calls.

    We return False when the call has no definition we can follow, or
    when it calls a closure, which another pass inlines.
    """
    try:
        to_inline = func_ir.get_definition(expr.func)
    except Exception:
        if self._DEBUG:
            print("Cannot find definition for %s" % expr.func)
        return False
    # Closure inlining belongs to another pass.
    if getattr(to_inline, "op", False) == "make_function":
        return False

    if getattr(to_inline, "op", False) == "getattr":
        return _nb_untyped_passes.resolve_func_from_module(func_ir, to_inline)
    # getattr on an ir.Expr looks in _kws and may fail.
    try:
        return getattr(to_inline, "value", False)
    except Exception:
        raise _nb_untyped_passes.GuardException


def _inline_inlinables_do_work(
    self, state, work_list, block, i, expr, inline_worker
):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numba_cuda.core.options import InlineOptions

    val = self._callee(state.func_ir, expr)

    if val:
        # Dispatcher-like values carry the jit kwargs in targetoptions.
        topt = getattr(val, "targetoptions", False)
        if topt:
            inline_type = topt.get("inline", None)
            if inline_type is not None:
                inline_opt = InlineOptions(inline_type)
                if not inline_opt.is_never_inline:
                    do_inline = True
                    pyfunc = val.py_func
                    if inline_opt.has_cost_model:
                        py_func_ir = run_frontend(pyfunc)
                        do_inline = inline_type(
                            expr, state.func_ir, py_func_ir
                        )
                    if do_inline:
                        new_blocks = self._inline(
                            state, block, i, expr, val, inline_worker
                        )
                        if work_list is not None:
                            for blk in new_blocks:
                                work_list.append(blk)
                        return True
    return False


def _inline_inlinables_inline(
    self, state, block, i, expr, dispatcher, inline_worker
):
    """Inline a call, transforming the callee for the call's argument
    types when it needs them.

    We first inline the callee without argument types. The transform
    returns None if a consteval in the callee names one of its
    parameters, and the callee's untyped run raises InlineeNeedsArgTypes
    if a callee nested in it does. Both happen before we change the
    caller's IR, so we can then type the call and inline the callee for
    those types.
    """
    function = inline_worker.transform_inlinee(dispatcher.py_func)
    if function is not None:
        try:
            return inline_worker.inline_function(
                state.func_ir, block, i, function
            )[3]
        except InlineeNeedsArgTypes:
            pass
    if any(isinstance(argtype, types.PyObject) for argtype in state.args):
        # This is the untyped run of an enclosing inlinee, so its caller
        # types the call.
        raise InlineeNeedsArgTypes(dispatcher.py_func.__name__)
    argtypes = self._call_argtypes(state, expr, dispatcher, inline_worker)
    function = inline_worker.transform_inlinee(dispatcher.py_func, argtypes)
    _, _, _, new_blocks = inline_worker.inline_function(
        state.func_ir, block, i, function, arg_typs=argtypes
    )
    return new_blocks


def _inline_inlinables_call_argtypes(
    self, state, expr, dispatcher, inline_worker
):
    """Fold the call's argument types, as a call would, from a partial
    typing of the caller.

    We type the caller once per run of this pass and reuse the types for
    every later call. That is safe because inlining renames only the
    callee's variables, and the call arguments are the caller's own
    variables.
    """
    from numba_cuda_mlir.numba_cuda.core.typed_passes import (
        PartialTypeInference,
    )

    if state.typemap is None:
        token = inlinee_typer.set(self._inlinee_typer(state, inline_worker))
        try:
            PartialTypeInference().run_pass(state)
        finally:
            inlinee_typer.reset(token)
    typemap = state.typemap

    argtypes = [typemap.get(var.name, types.unknown) for var in expr.args]
    kwtypes = {
        name: typemap.get(var.name, types.unknown) for name, var in expr.kws
    }
    if types.unknown in argtypes or types.unknown in kwtypes.values():
        for error in state.typing_errors or ():
            raise error
        raise _nb_errors.TypingError(
            "Cannot determine the argument types of the call to "
            f"{dispatcher.py_func.__name__}",
            loc=expr.loc,
        )
    return tuple(
        dispatcher._compiler.fold_argument_types(argtypes, kwtypes)[1]
    )


def _inline_inlinables_inlinee_typer(self, state, inline_worker):
    """Return a function that types a call to one of the caller's callees
    from its transformed IR.

    The function returns None for any other dispatcher, so that an
    overload template still compiles its implementation. It also returns
    None for a callee it is already typing, so that a recursive callee is
    compiled.
    """
    calls = (
        call
        for block in state.func_ir.blocks.values()
        for call in block.find_exprs("call")
    )
    callees = {
        callee
        for callee in (
            _nb_untyped_passes.guard(self._callee, state.func_ir, call)
            for call in calls
        )
        if hasattr(callee, "py_func")
    }
    typing = set()
    signatures = {}

    def type_inlinee(dispatcher, argtypes):
        if dispatcher not in callees or dispatcher in typing:
            return None
        if (dispatcher, argtypes) in signatures:
            return signatures[dispatcher, argtypes]
        typing.add(dispatcher)
        try:
            function = inline_worker.transform_inlinee(
                dispatcher.py_func, argtypes
            )
            _, results = inline_worker.type_callee(function, argtypes)
        finally:
            typing.discard(dispatcher)
        signatures[dispatcher, argtypes] = _nb_templates.Signature(
            results.return_type, argtypes, None
        )
        return signatures[dispatcher, argtypes]

    return type_inlinee


def _dispatcher_get_call_template(self, args, kws):
    """Resolve return type when this dispatcher is called from another
    jit function. Always compile as a device function so we don't emit
    kernel metadata for callees."""
    pysig, args = self._compiler.fold_argument_types(args, kws)
    kws = {}
    typer = inlinee_typer.get()
    try:
        signature = None if typer is None else typer(self, tuple(args))
    except _nb_errors.ForceLiteralArg as e:

        def folded(args, kws):
            return self._compiler.fold_argument_types(args, kws)[1]

        raise e.bind_fold_arguments(folded)
    if signature is not None:
        signatures = [signature]
    else:
        if self._can_compile:
            self._compile_as_device_callee(tuple(args))
        signatures = self.nopython_signatures
    func_name = self.py_func.__name__
    name = "CallTemplate({0})".format(func_name)
    call_template = _nb_typing.make_concrete_template(
        name, key=func_name, signatures=signatures
    )
    return call_template, pysig, args, kws


def _overload_template_generic(self, args, kws):
    """
    Type the overloaded function by compiling the appropriate
    implementation for the given args.
    """
    from numba_cuda_mlir.numba_cuda.core.typed_passes import (
        PreLowerStripPhis,
    )

    disp, new_args = self._get_impl(args, kws)
    if disp is None:
        return
    # Compile and type it for the given types
    disp_type = types.Dispatcher(disp)
    # Store the compiled overload for use in the lowering phase if there's
    # no inlining required (else functions are being compiled which will
    # never be used as they are inlined)
    if not self._inline.is_never_inline:
        # need to run the compiler front end up to type inference to
        # compute a signature
        from numba_cuda_mlir.numba_cuda.flags import Flags

        fcomp = disp._compiler
        flags = Flags()

        # spoof a compiler pipline like the one that will be in use
        tyctx = fcomp.targetdescr.typing_context
        tgctx = fcomp.targetdescr.target_context

        compiler_inst = fcomp.pipeline_class(
            tyctx,
            tgctx,
            None,
            None,
            None,
            flags,
            None,
        )
        inline_worker = _nb_icc.InlineWorker(
            tyctx,
            tgctx,
            fcomp.locals,
            compiler_inst,
            flags,
            None,
        )

        # Resolve first, so that a `literally` call in the inlinee raises
        # ForceLiteralArg through the dispatcher before the pipeline runs.
        resolve = disp_type.dispatcher.get_call_template
        template, pysig, folded_args, kws = resolve(new_args, kws)
        ir, (typemap, return_type, calltypes, _) = inline_worker.type_callee(
            disp_type.dispatcher.py_func, folded_args
        )
        ir = PreLowerStripPhis()._strip_phi_nodes(ir)
        ir._definitions = _nb_ir_utils.build_definitions(ir.blocks)

        sig = _nb_templates.Signature(return_type, folded_args, None)
        # this stores a load of info for the cost model function if
        # supplied it by default is None
        self._inline_overloads[sig.args] = {"folded_args": folded_args}
        # this stores the compiled overloads, if there's no compiled
        # overload available i.e. function is always inlined, the key
        # still needs to exist for type resolution

        # NOTE: If lowering is failing on a `_EmptyImplementationEntry`,
        #       the inliner has failed to inline this entry correctly.
        impl_init = _nb_templates._EmptyImplementationEntry("always inlined")
        self._compiled_overloads[sig.args] = impl_init
        if not self._inline.is_always_inline:
            # this branch is here because a user has supplied a function
            # to determine whether to inline or not. As a result both
            # compiled function and inliner info needed.
            sig = disp_type.get_call_type(self.context, new_args, kws)
            self._compiled_overloads[sig.args] = disp_type.get_overload(sig)
            # store the inliner information, it's used later in the cost
            # model function call
        iinfo = _nb_templates._inline_info(ir, typemap, calltypes, sig)
        self._inline_overloads[sig.args] = {
            "folded_args": folded_args,
            "iinfo": iinfo,
        }
    else:
        sig = disp_type.get_call_type(self.context, new_args, kws)
        if sig is None:  # can't resolve for this target
            return None
        self._compiled_overloads[sig.args] = disp_type.get_overload(sig)
    # Keyed on sig.args, which the return type does not take part in, so
    # the registrations above stay addressable by the same key.
    return_type = _nb_templates.strip_star_args(sig.return_type)
    return (
        sig
        if return_type == sig.return_type
        else sig.replace(return_type=return_type)
    )


def apply_inlined_callee_ast_transforms() -> None:
    """Transform inlined callees under the calling kernel's options and,
    when a consteval reads a parameter, its call's argument types."""
    consteval_transformer = _ast_consteval.ConstevalTransformer
    consteval_transformer._is_consteval_call = _consteval_is_consteval_call
    _ast_consteval.reads_parameters = reads_parameters
    _ast_transforms.reads_parameters = reads_parameters
    _ast_transforms.transform_inline_callee = transform_inline_callee

    stock_recompile_function = _ast_common.recompile_function

    def recompile_function(func, tree, stored_values=None):
        # Shift the dedented tree onto the function's file lines.
        ast.increment_lineno(tree, func.__code__.co_firstlineno - 1)
        new_func = stock_recompile_function(func, tree, stored_values)
        new_func.__kwdefaults__ = func.__kwdefaults__
        return new_func

    _ast_common.recompile_function = recompile_function
    _ast_transforms.recompile_function = recompile_function

    stock_get_compiler_class = _mlir_compiler.get_compiler_class

    @functools.wraps(stock_get_compiler_class)
    def get_compiler_class(*args, **kwargs):
        compiler_class = stock_get_compiler_class(*args, **kwargs)
        stock_init = compiler_class.__init__

        @functools.wraps(stock_init)
        def __init__(self, *init_args, **init_kwargs):
            stock_init(self, *init_args, **init_kwargs)
            self.state.metadata["inlinee_transform"] = transform_inline_callee

        compiler_class.__init__ = __init__
        return compiler_class

    _mlir_compiler.get_compiler_class = get_compiler_class
    _mlir_compiler.transform_inline_callee = transform_inline_callee

    _nb_icc.inlinee_typer = inlinee_typer
    _nb_icc.InlineeNeedsArgTypes = InlineeNeedsArgTypes
    worker = _nb_icc.InlineWorker
    stock_worker_init = worker.__init__

    @functools.wraps(stock_worker_init)
    def worker_init(
        self, *args, targetoptions=None, inlinee_transform=None, **kwargs
    ):
        stock_worker_init(self, *args, **kwargs)
        self.targetoptions = targetoptions
        self.inlinee_transform = inlinee_transform

    worker.__init__ = worker_init
    worker._fresh_callee_ir = _inline_worker_fresh_callee_ir
    worker.inline_function = _inline_worker_inline_function
    worker.type_callee = _inline_worker_type_callee
    worker.transform_inlinee = _inline_worker_transform_inlinee
    worker.run_untyped_passes = _inline_worker_run_untyped_passes

    passes = _nb_untyped_passes.InlineInlinables
    passes.run_pass = _inline_inlinables_run_pass
    passes._callee = _inline_inlinables_callee
    passes._do_work = _inline_inlinables_do_work
    passes._inline = _inline_inlinables_inline
    passes._call_argtypes = _inline_inlinables_call_argtypes
    passes._inlinee_typer = _inline_inlinables_inlinee_typer

    _descriptor.MLIRDispatcher.get_call_template = (
        _dispatcher_get_call_template
    )
    _nb_templates._OverloadFunctionTemplate.generic = (
        _overload_template_generic
    )


apply_inlined_callee_ast_transforms()


# ------------------------------------------------------------------ #
# slice-python-parity                                                #
# ------------------------------------------------------------------ #


def _slice_init(self, start=None, stop=None, step=None):
    def as_index(bound):
        if bound is None or isinstance(bound, _lnumpy.ir.NoneType):
            return None
        return lowering_utilities.convert(bound, _lnumpy.T.index())

    self.start = as_index(start)
    self.stop = as_index(stop)
    self.step = as_index(step)


def _resolve_slice(builder, slc, mr, dim_index=0):
    """Return (start, length, step) index values with Python slice
    semantics for dim dim_index."""
    np_ = _lnumpy
    start, stop, step = slc.start, slc.stop, slc.step
    c_step = 1 if step is None else np_.try_extract_constant(step)
    if c_step == 0:
        raise ValueError("slice step cannot be zero")
    step = np_.index_of(1) if step is None else step
    extent = mr.type.shape[dim_index]
    dynamic = np_.ir.ShapedType.get_dynamic_size()
    ext = (
        np_.index_of(extent)
        if extent != dynamic
        else np_.memref.dim(mr, np_.index_of(dim_index))
    )
    zero = np_.index_of(0)
    one = np_.index_of(1)
    neg1 = np_.index_of(-1)

    if c_step is None:
        is_zero = np_.arith.cmpi(np_.arith.CmpIPredicate.eq, step, zero)
        error_memref = builder._get_or_create_error_global()
        if error_memref is not None:
            with np_.scf.if_ctx_manager(is_zero):
                np_.set_error_code_if_zero(
                    error_memref, np_.KERNEL_ERROR_CODES[ValueError]
                )
                np_.scf.yield_([])
        # The flagged zero step still reaches the length division;
        # substitute one.
        step = np_.arith.select(is_zero, one, step)

    is_negative_step = np_.arith.cmpi(np_.arith.CmpIPredicate.slt, step, zero)
    extent_minus_one = np_.arith.subi(ext, one)
    lower = np_.arith.select(is_negative_step, neg1, zero)
    upper = np_.arith.select(is_negative_step, extent_minus_one, ext)

    def fix_bound(bound, default):
        if bound is None:
            return default
        is_negative = np_.arith.cmpi(np_.arith.CmpIPredicate.slt, bound, zero)
        wrapped = np_.arith.select(
            is_negative, np_.arith.addi(bound, ext), bound
        )
        return np_.arith.minsi(np_.arith.maxsi(wrapped, lower), upper)

    resolved_start = fix_bound(
        start, np_.arith.select(is_negative_step, extent_minus_one, zero)
    )
    resolved_stop = fix_bound(
        stop, np_.arith.select(is_negative_step, neg1, ext)
    )

    delta = np_.arith.subi(resolved_stop, resolved_start)
    dividend = np_.arith.select(
        is_negative_step,
        np_.arith.addi(delta, one),
        np_.arith.subi(delta, one),
    )
    nominal_length = np_.arith.addi(one, np_.arith.divsi(dividend, step))
    is_empty = np_.arith.select(
        is_negative_step,
        np_.arith.cmpi(np_.arith.CmpIPredicate.sge, delta, zero),
        np_.arith.cmpi(np_.arith.CmpIPredicate.sle, delta, zero),
    )
    length = np_.arith.select(is_empty, zero, nominal_length)
    return resolved_start, length, step


def _strided_view(array, offsets, sizes, strides, dims_to_drop=None):
    """Build a strided view; dimensions marked in dims_to_drop are
    dropped from the result."""
    np_ = _lnumpy
    rank = array.type.rank
    metadata = np_.memref_dialect.extract_strided_metadata(array)
    source_strides = list(metadata[2 + rank : 2 + 2 * rank])
    result_offset = metadata[1]
    result_strides = []
    result_sizes = []
    if dims_to_drop is None:
        dims_to_drop = [False] * rank

    for offset, size, stride, source_stride, drop in zip(
        offsets, sizes, strides, source_strides, dims_to_drop
    ):
        result_offset = np_.arith.addi(
            result_offset,
            np_.arith.muli(np_.index_of(offset), np_.index_of(source_stride)),
        )
        if not drop:
            result_sizes.append(np_.index_of(size))
            result_strides.append(
                np_.arith.muli(
                    np_.index_of(source_stride), np_.index_of(stride)
                )
            )

    dynamic_size = np_.ir.ShapedType.get_dynamic_size()
    dynamic_stride = np_.ir.ShapedType.get_dynamic_stride_or_offset()
    result_type = np_.ir.MemRefType.get(
        [dynamic_size] * len(result_sizes),
        array.type.element_type,
        layout=np_.ir.StridedLayoutAttr.get(
            dynamic_stride, [dynamic_stride] * len(result_sizes)
        ),
        memory_space=array.type.memory_space,
    )
    return np_.memref_dialect.reinterpret_cast(
        result_type,
        array,
        offsets=[result_offset],
        sizes=result_sizes,
        strides=result_strides,
        static_offsets=[dynamic_stride],
        static_sizes=[dynamic_size] * len(result_sizes),
        static_strides=[dynamic_stride] * len(result_sizes),
    )


def lower_array_getitem(builder, target, args, kwargs):
    np_ = _lnumpy
    np_.trace()

    # Check if this is a record array
    array_numba_type = builder.get_numba_type(args[0].name)
    from numba_cuda_mlir.types import NestedArray

    if isinstance(array_numba_type, NestedArray):
        from numba_cuda_mlir.lowering.record import (
            lower_nested_array_getitem_int,
        )

        return lower_nested_array_getitem_int(builder, target, args, kwargs)

    if isinstance(array_numba_type.dtype, np_.Record):
        return np_._lower_record_array_getitem(builder, target, args, kwargs)

    array = builder.load_var(args[0])
    # Handle both variable and constant indices
    if isinstance(args[1], int):
        index = args[1]
    else:
        index = builder.load_var(args[1])
    array_type = array.type

    if not array_type.has_rank:
        raise NotImplementedError("NYI: unranked memrefs")

    index = np_._normalize_negative_index(
        array,
        index,
        0,
        None
        if isinstance(args[1], int)
        else builder.get_numba_type(args[1].name),
    )
    if array_type.rank == 1:
        value = lowering_utilities.array_element_value_load(
            array_numba_type,
            array,
            [index],
            dynamic_shared_memory=builder._is_dynamic_shared_memory(array),
        )
    else:
        rank = array_type.rank
        sv_offsets = [index] + [np_.index_of(0)] * (rank - 1)
        sv_sizes = [np_.index_of(1)] + [
            np_.memref.dim(array, np_.index_of(i)) for i in range(1, rank)
        ]
        sv_strides = [np_.index_of(1)] * rank
        dims_to_drop = [True] + [False] * (rank - 1)
        value = _strided_view(
            array, sv_offsets, sv_sizes, sv_strides, dims_to_drop
        )

    builder.store_var(target, value)


def lower_array_slice_getitem(builder, target, args, kwargs):
    np_ = _lnumpy
    np_.trace()
    mr = builder.load_var(args[0])
    rank = mr.type.rank
    slc = builder.load_var(args[1])
    start, length, step = _resolve_slice(builder, slc, mr)

    offsets = [start] + [np_.index_of(0) for _ in range(1, rank)]
    sizes = [length] + [
        np_.memref.dim(mr, np_.index_of(i)) for i in range(1, rank)
    ]
    strides = [step] + [np_.index_of(1) for _ in range(1, rank)]
    view = _strided_view(mr, offsets, sizes, strides)
    builder.store_var(target, view)


def lower_array_slice_setitem(builder, target, args, kwargs):
    """Lower arr[slice] = value: fill the sliced region."""
    np_ = _lnumpy
    np_.trace()
    array = builder.load_var(args[0])
    slice_val = builder.load_var(args[1])
    value = builder.load_var(args[2])

    array_numba_type = builder.get_numba_type(args[0].name)
    mr_type = array.type
    rank = mr_type.rank

    start, length, step = _resolve_slice(builder, slice_val, array)

    # Map a forward loop onto the resolved slice.
    starts = [np_.index_of(0)] * rank
    stops = [length] + [
        np_.memref.dim(array, np_.index_of(i + 1)) for i in range(rank - 1)
    ]
    steps = [np_.index_of(1)] * rank

    @np_.scf.forall_(starts, stops, steps)
    def fill_all(*indices):
        idx0 = np_.arith.addi(start, np_.arith.muli(indices[0], step))
        lowering_utilities.array_element_value_store(
            array_numba_type,
            array,
            [idx0, *indices[1:]],
            value,
            dynamic_shared_memory=builder._is_dynamic_shared_memory(array),
        )


def lower_array_tuple_getitem(builder, target, args, kwargs):
    np_ = _lnumpy
    if len(args) != 2:
        raise np_.InternalCompilerError(
            f"Tuple getitem takes exactly two arguments, got {len(args)}"
        )

    # Check if this is a nested array (embedded in a record)
    from numba_cuda_mlir.types import NestedArray

    array_numba_type = builder.get_numba_type(args[0].name)
    if isinstance(array_numba_type, NestedArray):
        from numba_cuda_mlir.lowering.record import (
            lower_nested_array_getitem_tuple,
        )

        return lower_nested_array_getitem_tuple(builder, target, args, kwargs)

    array = builder.load_var(args[0])
    tuple_indices = builder.load_var(args[1])

    array_type = array.type
    if (
        not isinstance(
            array_type, (np_.ir.MemRefType, np_.ir.RankedTensorType)
        )
        or not array_type.has_rank
    ):
        raise np_.InternalCompilerError(
            "Array must be a statically-ranked memref or tensor, "
            f"got {array_type}"
        )

    if not isinstance(tuple_indices, tuple):
        raise np_.InternalCompilerError(
            f"Tuple indices must be a tuple, got {type(tuple_indices)}"
        )

    target_type = builder.get_numba_type(target.name)
    source_rank = array_type.rank
    n_indexed = len(tuple_indices)
    n_trailing = source_rank - n_indexed

    dims = [np_.memref.dim(array, np_.index_of(i)) for i in range(source_rank)]

    tuple_numba_type = builder.get_numba_type(args[1].name)
    offsets, sizes, strides, is_scalar = [], [], [], []
    for dim, index in enumerate(tuple_indices):
        if isinstance(index, slice) and index == slice(None, None, -1):
            offsets.append(np_.arith.subi(dims[dim], np_.index_of(1)))
            sizes.append(dims[dim])
            strides.append(np_.index_of(-1))
            is_scalar.append(False)
            continue
        match index:
            case np_.Slice() as slc:
                if not isinstance(target_type, types.Array):
                    raise TypeError(
                        f"Target type {target_type} is not an array, but "
                        "a slice was used to index it"
                    )
                s_start, s_length, s_step = _resolve_slice(
                    builder, slc, array, dim
                )
                offsets.append(s_start)
                sizes.append(s_length)
                strides.append(s_step)
                is_scalar.append(False)
            case int() | np_.ir.Value() as value:
                index_type = tuple_numba_type.types[dim]
                offsets.append(
                    np_._normalize_negative_index(
                        array, value, dim, index_type
                    )
                )
                sizes.append(1)
                strides.append(1)
                is_scalar.append(True)
            case _:
                raise np_.InternalCompilerError(
                    f"Tuple indices must be a slice or int, got {type(index)}"
                )

    # Extend with full-extent entries for unindexed trailing dimensions
    trailing_dims = [
        np_.memref.dim(array, np_.index_of(n_indexed + i))
        for i in range(n_trailing)
    ]
    full_offsets = list(offsets) + [np_.index_of(0)] * n_trailing
    full_sizes = list(sizes) + trailing_dims
    full_strides = list(strides) + [np_.index_of(1)] * n_trailing
    full_is_scalar = list(is_scalar) + [False] * n_trailing

    match target_type:
        case types.Array():
            n_kept = sum(1 for sc in full_is_scalar if not sc)
            if n_kept != target_type.ndim:
                raise np_.InternalCompilerError(
                    f"Result rank {n_kept} does not match target type ndim "
                    f"{target_type.ndim}"
                )
            value = _strided_view(
                array, full_offsets, full_sizes, full_strides, full_is_scalar
            )
            builder.store_var(target, value)
        case (
            types.Number()
            | types.Boolean()
            | types.NPDatetime()
            | types.NPTimedelta()
        ):
            value = lowering_utilities.array_element_value_load(
                array_numba_type,
                array,
                full_offsets,
                dynamic_shared_memory=builder._is_dynamic_shared_memory(array),
            )
            builder.store_var(target, value)
        case _:
            raise np_.InternalCompilerError(
                f"Target type {target_type} is not an array or number, but "
                "a tuple was used to index it"
            )


def apply_slice_python_parity() -> None:
    """Resolve slices with Python semantics and build views by
    reinterpreting the parent array."""
    _lnumpy.Slice.__init__ = _slice_init
    _lnumpy._resolve_slice = _resolve_slice
    _lnumpy._strided_view = _strided_view
    np_lower = _lnumpy.registry.lower
    np_lower(operator.getitem, types.Array, types.Number)(lower_array_getitem)
    np_lower(operator.getitem, types.Array, types.Integer)(lower_array_getitem)
    np_lower(operator.getitem, types.Buffer, types.Integer)(
        lower_array_getitem
    )
    np_lower(operator.getitem, types.Array, types.SliceType)(
        lower_array_slice_getitem
    )
    np_lower(operator.setitem, types.Array, types.SliceType, types.Any)(
        lower_array_slice_setitem
    )
    np_lower(operator.getitem, types.Array, types.UniTuple)(
        lower_array_tuple_getitem
    )
    np_lower(operator.getitem, types.Array, types.Tuple)(
        lower_array_tuple_getitem
    )

    lower_class = _ml.MLIRLower
    stock_lower_literal = lower_class.lower_literal_if_needed

    @functools.wraps(stock_lower_literal)
    def lower_literal_if_needed(self, value, numba_type=None):
        if isinstance(value, slice):
            # Materialize frozen slices like inline slices.
            return _lnumpy.Slice(
                *(
                    self.lower_literal_if_needed(bound)
                    if bound is not None
                    else None
                    for bound in (value.start, value.stop, value.step)
                )
            )
        return stock_lower_literal(self, value, numba_type)

    lower_class.lower_literal_if_needed = lower_literal_if_needed


apply_slice_python_parity()


# ------------------------------------------------------------------ #
# float-rounding-parity-pr                                           #
# ------------------------------------------------------------------ #

# Exponent types that convert safely to int32
POW_INT32_EXPONENTS = (
    types.int8,
    types.int16,
    types.int32,
    types.uint8,
    types.uint16,
)


def _math_result_type(*arg_types):
    """Float type a math function computes in: integers become float64,
    the widest float wins."""
    floats = [
        types.float64 if isinstance(ty, types.Integer) else ty
        for ty in arg_types
    ]
    return max(floats, key=lambda ty: getattr(ty, "bitwidth", 0))


def _pow_result_type(base, exponent):
    """A float base raised to an exponent that converts safely to int32
    keeps the base type."""
    if isinstance(base, types.Float) and exponent in POW_INT32_EXPONENTS:
        return base
    return _math_result_type(base, exponent)


def _unary_math_generic(acceptable, return_type_fn):
    def generic(self, args, kws):
        if len(args) == 1 and isinstance(args[0], acceptable):
            return_fn = return_type_fn or _math_result_type
            return signature(return_fn(args[0]), args[0])

    return generic


def _binary_math_generic(return_type_fn):
    def generic(self, args, kws):
        if (
            len(args) == 2
            and isinstance(args[0], types.Number)
            and isinstance(args[1], types.Number)
        ):
            return_fn = return_type_fn or _math_result_type
            return signature(return_fn(args[0], args[1]), args[0], args[1])

    return generic


def _get_range_object(builder, args, int_type):
    """Load the range bounds converted to the range's resolved integer
    type."""
    int_of = _lbuiltins.int_of
    int_mlir_type = builder.get_mlir_type(int_type)

    def bound(var):
        signed = _lbuiltins.get_conversion_signedness(
            builder.get_numba_type(var.name), int_type
        )
        return int_of(builder.load_var(var), ty=int_mlir_type, signed=signed)

    match args:
        case [stop]:
            return (
                int_of(0, ty=int_mlir_type),
                bound(stop),
                int_of(1, ty=int_mlir_type),
            )
        case [start, stop]:
            return bound(start), bound(stop), int_of(1, ty=int_mlir_type)
        case [start, stop, step]:
            return bound(start), bound(stop), bound(step)
        case _:
            raise ValueError(f"Invalid arguments for range: {args}")


def lower_range(builder, target, args, kwargs):
    int_type = builder.get_numba_type(target.name).dtype
    start, stop, step = _get_range_object(builder, args, int_type)
    ro = _lbuiltins.RangeObject(builder, start, stop, step)
    builder.store_var(target, ro)


def _ensure_float(value, source_type=None):
    """Ensure value is floating-point; integers become float64."""
    if isinstance(value.type, _lmath.ir.IntegerType) or isinstance(
        value.type, _lmath.ir.IndexType
    ):
        signed = None
        if source_type is not None:
            signed = _lmath.get_conversion_signedness(
                source_type, types.float64
            )
        return lowering_utilities.convert(value, _lmath.T.f64(), signed=signed)
    return value


def math_ceil_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, (
        "math_ceil_intrinsic does not accept any keyword arguments"
    )
    value = mlir_lower.load_var(args[0])
    if _lmath._is_integer_type(value.type):
        # ceil of an integer is the integer itself, as float64
        result = _ensure_float(value, mlir_lower.get_numba_type(args[0].name))
    else:
        result = _lmath.math_dialect.ceil(value)
    mlir_lower.store_var(target, result)


def math_floor_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_floor does not accept any keyword arguments"
    value = mlir_lower.load_var(args[0])
    if _lmath._is_integer_type(value.type):
        # floor of an integer is the integer itself, as float64
        result = _ensure_float(value, mlir_lower.get_numba_type(args[0].name))
    else:
        result = _lmath.math_dialect.floor(value)
    mlir_lower.store_var(target, result)


def math_trunc_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_trunc does not accept any keyword arguments"
    value = mlir_lower.load_var(args[0])
    if _lmath._is_integer_type(value.type):
        # trunc of an integer is the integer itself, as float64
        result = _ensure_float(value, mlir_lower.get_numba_type(args[0].name))
    else:
        result = _lmath.math_dialect.trunc(value)
    mlir_lower.store_var(target, result)


def math_pow_cg(mlir_lower, target, args, kwargs):
    """math.pow(x, y) - x raised to power y"""
    assert not kwargs, "math_pow does not accept any keyword arguments"
    assert len(args) == 2, "math_pow expects 2 arguments"
    target_type = mlir_lower.get_numba_type(target.name)
    float_type = mlir_lower.get_mlir_type(target_type)
    x = _lmath._load_and_convert_operand(
        mlir_lower, args[0], target_type, float_type
    )
    exponent_type = mlir_lower.get_numba_type(args[1].name)
    if (
        isinstance(target_type, types.Float)
        and exponent_type in POW_INT32_EXPONENTS
    ):
        y = _lmath._load_and_convert_operand(
            mlir_lower, args[1], types.int32, _lmath.T.i32()
        )
        result = _lmath.math_dialect.fpowi(x, y)
    else:
        y = _lmath._load_and_convert_operand(
            mlir_lower, args[1], target_type, float_type
        )
        result = _lmath.math_dialect.powf(x, y)
    mlir_lower.store_var(target, result)


def apply_float_rounding_parity() -> None:
    """Math functions return the float type they compute in."""
    _tmath.POW_INT32_EXPONENTS = POW_INT32_EXPONENTS
    _tmath._math_result_type = _math_result_type
    _tmath._pow_result_type = _pow_result_type
    return_type_fns = {
        "ceil": None,
        "floor": None,
        "trunc": None,
        "pow": _pow_result_type,
    }
    for name, template in _tmath._math_functions.items():
        qualname = template.__qualname__
        stock = template.__dict__.get("generic")
        if stock is None or stock.__closure__ is None:
            continue
        cells = dict(zip(stock.__code__.co_freevars, stock.__closure__))
        return_type_fn = return_type_fns.get(
            name, cells["return_type_fn"].cell_contents
        )
        if qualname.startswith("_make_unary_math_template."):
            template.generic = _unary_math_generic(
                cells["acceptable"].cell_contents, return_type_fn
            )
        elif qualname.startswith("_make_binary_math_template."):
            template.generic = _binary_math_generic(return_type_fn)

    _lbuiltins._get_range_object = _get_range_object
    builtins_lower = _lbuiltins.registry.lower
    builtins_lower(range, types.Number)(lower_range)
    builtins_lower(range, types.Number, types.Number)(lower_range)
    builtins_lower(range, types.Number, types.Number, types.Number)(
        lower_range
    )

    _lmath._ensure_float = _ensure_float
    _lmath.POW_INT32_EXPONENTS = POW_INT32_EXPONENTS
    math_lower = _lmath.registry.lower
    math_lower(math.ceil, types.Number)(math_ceil_cg)
    math_lower(math.floor, types.Number)(math_floor_cg)
    math_lower(math.trunc, types.Number)(math_trunc_cg)
    math_lower(math.pow, types.Number, types.Number)(math_pow_cg)


apply_float_rounding_parity()


# ------------------------------------------------------------------ #
# perf-device-array-readonly                                         #
# ------------------------------------------------------------------ #


def apply_device_array_readonly() -> None:
    """Read a device array's readonly flag without its array interface."""
    stock = _descriptor._array_readonly

    def array_readonly(arg, ty=None):
        # Numba checks every array argument on each launch, and it
        # builds a device array's interface dict only to report it
        # writable.
        if isinstance(arg, _descriptor.DeviceNDArrayBase):
            return False
        return stock(arg, ty)

    _descriptor._array_readonly = array_readonly


apply_device_array_readonly()


# ------------------------------------------------------------------ #
# Typed-IR block scheduler                                           #
# ------------------------------------------------------------------ #


def register_typed_block_scheduler() -> None:
    """Register cubie's typed-IR block scheduler with the backend.

    No-ops when ``CUBIE_BLOCK_SCHEDULE`` is ``source``. The registered
    policy folds into the kernel-cache fingerprint.
    """
    from cubie._env import (
        block_schedule_default,
        set_active_block_schedule,
    )
    from cubie.backend._block_schedule_policies import (
        BLOCK_SCHEDULE_POLICIES,
    )

    policy = block_schedule_default()
    if policy == "source":
        return
    if policy not in BLOCK_SCHEDULE_POLICIES:
        raise ValueError(
            f"CUBIE_BLOCK_SCHEDULE={policy!r} is not recognised; "
            f"valid values: {sorted(BLOCK_SCHEDULE_POLICIES)}"
        )
    from cubie.backend._typed_block_scheduler import (
        TypedBlockScheduler,
    )

    TypedBlockScheduler.policy = policy
    register_typed_planner(TypedBlockScheduler)
    set_active_block_schedule(policy)


register_typed_block_scheduler()
