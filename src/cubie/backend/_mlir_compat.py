"""Shims mirroring the open numba-cuda-mlir pull requests cubie uses.

Each section reproduces one open pull request branch of the
ccam80/numba-cuda-mlir fork, as merged onto the pinned
``cubie-numba-cuda-mlir`` wheel, and applies it at import. Changed
functions are copied from the branch with module names qualified;
changed lowerings are registered again for the same signatures, which
replaces the stock implementation. Remove a section when its pull
request merges and the wheel carries it.

The module then registers cubie's typed-IR block scheduler with the
wheel's typed-planner hook.

Modified numba-cuda-mlir source: (c) NVIDIA CORPORATION; Apache 2.0.
"""

import ast
import copy
import functools
import inspect
import math
import operator
from collections import defaultdict

from numba_cuda_mlir import ast_transforms as _ast_transforms
from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir import mlir_compiler as _mlir_compiler
from numba_cuda_mlir import mlir_lowering as _ml
from numba_cuda_mlir import mlir_optimization as _mlir_optimization
from numba_cuda_mlir._mlir.dialects import arith
from numba_cuda_mlir.ast_transforms import (
    ASTTransformPass,
    apply_ast_transforms,
)
from numba_cuda_mlir.ast_transforms import common as _ast_common
from numba_cuda_mlir.extending import register_typed_planner
from numba_cuda_mlir.lowering import builtins as _lbuiltins
from numba_cuda_mlir.lowering import cuda as _lcuda
from numba_cuda_mlir.lowering import math as _lmath
from numba_cuda_mlir.lowering import numpy as _lnumpy
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import (
    analysis as _nb_analysis,
)
from numba_cuda_mlir.numba_cuda.core import (
    controlflow as _nb_controlflow,
)
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
from numba_cuda_mlir.numba_cuda.typing.templates import signature
from numba_cuda_mlir.typing import math as _tmath

# ------------------------------------------------------------------ #
# ccam80/mixed-bool-number                                           #
# ------------------------------------------------------------------ #


def apply_mixed_bool_number() -> None:
    """Register mixed Boolean/Number comparisons; ``!=`` unordered."""
    for op, cg in (
        (operator.ne, _lmath.ne_cg),
        (operator.eq, _lmath.eq_cg),
        (operator.lt, _lmath.lt_cg),
        (operator.le, _lmath.le_cg),
        (operator.gt, _lmath.gt_cg),
        (operator.ge, _lmath.ge_cg),
    ):
        _lmath.registry.lower(op, types.Boolean, types.Number)(cg)
        _lmath.registry.lower(op, types.Number, types.Boolean)(cg)
    # _operator_mapping is cached; its dict is the mapping in use.
    _lmath._operator_mapping()[operator.ne] = _lmath.OpForType(
        _lmath._make_fcmp(arith.CmpFPredicate.UNE),
        _lmath._make_icmp(arith.CmpIPredicate.ne),
        _lmath._make_icmp(arith.CmpIPredicate.ne),
        None,
    )


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
# frontend-set-copies-pr                                             #
# ------------------------------------------------------------------ #


def _analysis_compute_dead_maps(cfg, blocks, live_map, var_def_map):
    """End-of-life maps from the block-entry ``live_map``."""
    escaping_dead_map = defaultdict(set)
    internal_dead_map = defaultdict(set)
    exit_dead_map = defaultdict(set)

    for offset, ir_block in blocks.items():
        cur_live_set = live_map[offset] | var_def_map[offset]
        outgoing_live_map = dict(
            (out_blk, live_map[out_blk])
            for out_blk, _data in cfg.successors(offset)
        )
        terminator_liveset = set(
            v.name for v in ir_block.terminator.list_vars()
        )
        combined_liveset = functools.reduce(
            operator.or_, outgoing_live_map.values(), set()
        )
        combined_liveset |= terminator_liveset
        internal_set = cur_live_set - combined_liveset
        internal_dead_map[offset] = internal_set
        escaping_live_set = cur_live_set - internal_set
        for out_blk, new_live_set in outgoing_live_map.items():
            new_live_set = new_live_set | var_def_map[out_blk]
            escaping_dead_map[out_blk] |= escaping_live_set - new_live_set
        if not outgoing_live_map:
            exit_dead_map[offset] = terminator_liveset

    all_vars = set().union(*live_map.values())
    internal_dead_vars = set().union(*internal_dead_map.values())
    escaping_dead_vars = set().union(*escaping_dead_map.values())
    exit_dead_vars = set().union(*exit_dead_map.values())
    dead_vars = internal_dead_vars | escaping_dead_vars | exit_dead_vars
    missing_vars = all_vars - dead_vars
    if missing_vars:
        if not cfg.exit_points():
            pass
        else:
            msg = "liveness info missing for vars: {0}".format(missing_vars)
            raise RuntimeError(msg)

    combined = dict(
        (k, internal_dead_map[k] | escaping_dead_map[k]) for k in blocks
    )

    return _nb_analysis._dead_maps_result(
        internal=internal_dead_map,
        escaping=escaping_dead_map,
        combined=combined,
    )


def _ir_utils_fixup_var_define_in_scope(blocks):
    """Define every referenced ir.Var in every scope the blocks use."""
    used_var = {}
    for blk in blocks.values():
        for inst in blk.body:
            for var in inst.list_vars():
                used_var[var] = inst
    scopes = {id(blk.scope): blk.scope for blk in blocks.values()}
    for scope in scopes.values():
        for var in used_var.keys():
            if var.name not in scope.localvars:
                scope.localvars.define(var.name, var)


def apply_frontend_set_copies() -> None:
    """Build dead maps and scope repairs without set copies."""
    _nb_analysis.compute_dead_maps = _analysis_compute_dead_maps
    _nb_ir_utils.fixup_var_define_in_scope = (
        _ir_utils_fixup_var_define_in_scope
    )
    _nb_untyped_passes.fixup_var_define_in_scope = (
        _ir_utils_fixup_var_define_in_scope
    )


apply_frontend_set_copies()


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
# ssa-iterative-def-search-pr                                        #
# ------------------------------------------------------------------ #


def _find_def_from_top(self, states, label, loc):
    """Find definition reaching the top of the block at ``label``,
    inserting phi nodes where necessary.

    Runs on an explicit worklist so the search depth does not grow
    with the CFG. Each ``pending`` item is a ``(phinode, pred, loc)``
    triple whose resolved incoming definition is appended to
    ``phinode``; predecessors are pushed in reverse so phi nodes are
    created, and fresh variables numbered, in depth-first order.
    """
    pending = []
    result = self._walk_def_chain(states, label, loc, True, pending)
    while pending:
        phinode, pred, philoc = pending.pop()
        incoming_def = self._walk_def_chain(
            states,
            pred,
            philoc,
            False,
            pending,
        )
        _nb_ssa._logger.debug("incoming_def %s", incoming_def)
        phinode.value.incoming_values.append(incoming_def.target)
        phinode.value.incoming_blocks.append(pred)
    return result


def _walk_def_chain(self, states, label, loc, from_top, pending):
    """Walk a single def-search chain.

    Alternates between the *from_bottom* step (take the last
    definition in the block, if any) and the *from_top* step (insert
    a phi node, or hop to the immediate dominator).  A phi node is
    registered in ``defmap`` before its predecessors are resolved,
    so a chain revisiting the block terminates there; resolution of
    the phi's incoming values is deferred onto ``pending``.
    """
    cfg = states["cfg"]
    defmap = states["defmap"]
    phimap = states["phimap"]
    phi_locations = states["phi_locations"]

    while True:
        if not from_top:
            _nb_ssa._logger.debug("find_def_from_bottom label %r", label)
            defs = defmap[label]
            if defs:
                return defs[-1]
            from_top = True

        _nb_ssa._logger.debug("find_def_from_top label %r", label)
        if label in phi_locations:
            scope = states["scope"]
            loc = states["block"].loc
            # fresh variable
            freshvar = scope.redefine(states["varname"], loc=loc)
            # insert phi
            phinode = _nb_ir.Assign(
                target=freshvar,
                value=_nb_ir.Expr.phi(loc=loc),
                loc=loc,
            )
            _nb_ssa._logger.debug("insert phi node %s at %s", phinode, label)
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
                # We have searched to the top of the idom tree.
                # Since we still cannot find a definition,
                # we will warn.
                _nb_ssa._warn_about_uninitialized_variable(
                    states["varname"], loc
                )
                return _nb_ssa.UndefinedVariable
            _nb_ssa._logger.debug("idom %s from label %s", idom, label)
            label = idom
            from_top = False


def apply_ssa_iterative_def_search() -> None:
    """Search SSA reaching definitions on an explicit worklist."""
    fixer = _nb_ssa._FixSSAVars
    fixer._find_def_from_top = _find_def_from_top
    fixer._walk_def_chain = _walk_def_chain


apply_ssa_iterative_def_search()


# ------------------------------------------------------------------ #
# topo-order-iterative-pr                                            #
# ------------------------------------------------------------------ #


def topo_sort(self, nodes, reverse=False):
    """
    Iterate over the *nodes* in topological order (ignoring back edges).
    The sort isn't guaranteed to be stable.
    """
    nodes = set(nodes)
    if not nodes:
        return
    it = self._topo_order
    if reverse:
        it = reversed(it)
    for n in it:
        if n in nodes:
            yield n


def _find_topo_order(self):
    succs = self._succs
    back_edges = self._back_edges
    post_order = []
    seen = set()

    # Successors pushed in reverse so the stack visits them in
    # iteration order.
    def visit(node):
        if node not in seen:
            seen.add(node)
            stack.append((post_order.append, node))
            forward = [
                dest for dest in succs[node] if (node, dest) not in back_edges
            ]
            stack.extend((visit, dest) for dest in reversed(forward))

    stack = [(visit, self._entry_point)]
    while stack:
        cb, node = stack.pop()
        cb(node)

    post_order.reverse()
    return post_order


def apply_topo_order_iterative() -> None:
    """Compute CFGraph topological order without recursion."""
    graph = _nb_controlflow.CFGraph
    graph.topo_sort = topo_sort
    graph._find_topo_order = _find_topo_order


apply_topo_order_iterative()


# ------------------------------------------------------------------ #
# empty-body-repair-pr                                               #
# ------------------------------------------------------------------ #


class EmptyBodyRepairer(ast.NodeTransformer):
    """Fill empty statement bodies with ``pass``."""

    def __init__(self):
        self.modified = False

    def visit_Module(self, node: ast.Module) -> ast.Module:
        return super().generic_visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        node = super().generic_visit(node)
        body = getattr(node, "body", None)
        if isinstance(body, list) and not body:
            node.body = [ast.copy_location(ast.Pass(), node)]
            self.modified = True
        return node


def repair_empty_bodies(tree: ast.Module) -> tuple[ast.Module, bool]:
    """Insert ``pass`` into every empty statement body."""
    repairer = EmptyBodyRepairer()
    new_tree = repairer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, repairer.modified


class EmptyBodyRepairPass(ASTTransformPass):
    """Pipeline pass filling bodies emptied by earlier passes."""

    @property
    def name(self) -> str:
        return "EmptyBodyRepair"

    def transform(self, tree, context):
        return repair_empty_bodies(tree)


def apply_empty_body_repair() -> None:
    """Append the empty-body repair pass to the transform pipeline."""
    stock_create_default_pipeline = _ast_transforms.create_default_pipeline

    def create_default_pipeline():
        pipeline = stock_create_default_pipeline()
        pipeline.add_pass(EmptyBodyRepairPass())
        return pipeline

    _ast_transforms.create_default_pipeline = create_default_pipeline


apply_empty_body_repair()


# ------------------------------------------------------------------ #
# inlined-callee-ast-transforms-pr                                   #
# ------------------------------------------------------------------ #

_INLINEE_PARAMETER = object()


def transform_inline_callee(pyfunc, targetoptions):
    """Apply AST transforms to an inlinee under the caller's options.

    Parameters resolve to a placeholder.
    """
    if not targetoptions.get("experimental_ast_transforms", False):
        return pyfunc

    argtypes = (_INLINEE_PARAMETER,) * len(
        inspect.signature(pyfunc).parameters
    )
    transformed, _ = apply_ast_transforms(pyfunc, targetoptions, argtypes)
    return transformed


def _inline_worker_transform_inlinee(self, function):
    """Apply the configured target-specific transform to an inlinee."""
    if self.inlinee_transform is None:
        return function
    return self.inlinee_transform(function, self.targetoptions)


def _inline_worker_run_untyped_passes(self, func, enable_ssa=False):
    """Run the untyped passes over ``func`` and return its Numba IR."""
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


def _inline_inlinables_do_work(
    self, state, work_list, block, i, expr, inline_worker
):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numba_cuda.core.options import InlineOptions

    to_inline = None
    try:
        to_inline = state.func_ir.get_definition(expr.func)
    except Exception:
        if self._DEBUG:
            print("Cannot find definition for %s" % expr.func)
        return False
    # Closure inlining belongs to another pass.
    if getattr(to_inline, "op", False) == "make_function":
        return False

    if getattr(to_inline, "op", False) == "getattr":
        val = _nb_untyped_passes.resolve_func_from_module(
            state.func_ir, to_inline
        )
    else:
        # getattr on an ir.Expr looks in _kws and may fail.
        try:
            val = getattr(to_inline, "value", False)
        except Exception:
            raise _nb_untyped_passes.GuardException

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
                        pyfunc = inline_worker.transform_inlinee(pyfunc)
                        _, _, _, new_blocks = inline_worker.inline_function(
                            state.func_ir,
                            block,
                            i,
                            pyfunc,
                        )
                        if work_list is not None:
                            for blk in new_blocks:
                                work_list.append(blk)
                        return True
    return False


def apply_inlined_callee_ast_transforms() -> None:
    """Transform inlined callees under the calling kernel's options."""
    _ast_transforms._INLINEE_PARAMETER = _INLINEE_PARAMETER
    _ast_transforms.transform_inline_callee = transform_inline_callee

    stock_recompile_function = _ast_common.recompile_function

    def recompile_function(func, tree, stored_values=None):
        # Shift the dedented tree onto the function's file lines.
        ast.increment_lineno(tree, func.__code__.co_firstlineno - 1)
        return stock_recompile_function(func, tree, stored_values)

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
    worker.transform_inlinee = _inline_worker_transform_inlinee
    worker.run_untyped_passes = _inline_worker_run_untyped_passes
    passes = _nb_untyped_passes.InlineInlinables
    passes.run_pass = _inline_inlinables_run_pass
    passes._do_work = _inline_inlinables_do_work


apply_inlined_callee_ast_transforms()


# ------------------------------------------------------------------ #
# slice-python-parity                                                #
# ------------------------------------------------------------------ #


def lower_strides(_, mlir_lower, target, array):
    from numba_cuda_mlir.lowering_utilities import index_of

    array_numba_type = mlir_lower.get_numba_type(array.name)
    array = mlir_lower.load_var(array)
    array_type = array.type
    rank = array_type.rank
    element_size = _lcuda.storage_itemsize_bytes(array_numba_type)

    if isinstance(array_type, _lcuda.ir.MemRefType):
        metadata = _lcuda.memref.extract_strided_metadata(array)
        element_strides = metadata[2 + rank : 2 + 2 * rank]
        strides = [
            _lcuda.arith.muli(index_of(stride), index_of(element_size))
            for stride in element_strides
        ]
    elif isinstance(array_type, _lcuda.ir.RankedTensorType):
        dims = [
            _lcuda.tensor.dim(
                source=array,
                index=_lcuda.arith.constant(result=_lcuda.T.index(), value=i),
            )
            for i in range(rank)
        ]
        strides = [None] * rank
        strides[-1] = index_of(element_size)
        for i in range(rank - 2, -1, -1):
            strides[i] = _lcuda.arith.muli(strides[i + 1], dims[i + 1])
    else:
        raise NotImplementedError(f"strides not implemented for {array_type}")

    mlir_lower.store_var(target, tuple(strides))


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

    index = np_._normalize_negative_index(array, index, 0)
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

    offsets, sizes, strides, is_scalar = [], [], [], []
    for dim, index in enumerate(tuple_indices):
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
                offsets.append(
                    np_._normalize_negative_index(array, value, dim)
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
        case types.Number() | types.Boolean():
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


def _request_dynamic_shared_memory(self, mr_type):
    bytes = _ml.get_type_size_bytes(mr_type.element_type)
    assert self.mlir_funcOp
    # Emit at the current insertion point: the entry block may
    # already have a terminator once the request appears after
    # control flow. The shared-memory base itself is still created
    # at the entry block's start by _get_shared_memory_base.
    bytes_op = _ml.arith.constant(result=_ml.T.index(), value=bytes)
    shm_base = self._get_shared_memory_base()
    total_shared_memory_bytes = self._load_total_shared_memory_bytes()
    # LLVM 7 has no dynamic_smem_size intrinsic; read the sreg directly.
    smem_size = _ml.llvm.inline_asm(
        _ml.T.i32(),
        [],
        "mov.u32 $0, %dynamic_smem_size;",
        "=r",
    )
    dynamic_shared_bytes = self.mlir_convert(smem_size, _ml.T.index())
    remaining_bytes = _ml.arith.subi(
        lhs=dynamic_shared_bytes, rhs=total_shared_memory_bytes
    )
    size = _ml.arith.divui(lhs=remaining_bytes, rhs=bytes_op)
    view = _ml.memref.view(
        result=mr_type,
        source=shm_base,
        byte_shift=total_shared_memory_bytes,
        sizes=[size],
    )
    self._dynamic_shared_memory_values.append(view)
    return view


def _externalize_dynamic_shared_globals(module):
    """Give zero-length ``__dynamic_shmem__`` globals external linkage
    so their size is unknown."""
    ir = _ml.ir
    external = ir.Attribute.parse("#llvm.linkage<external>")

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    if child.operation.name == "llvm.mlir.global":
                        sym = ir.StringAttr(child.attributes["sym_name"]).value
                        if sym.startswith("__dynamic_shmem__"):
                            child.attributes["linkage"] = external
                    walk(child.operation)

    walk(module.operation)


def apply_slice_python_parity() -> None:
    """Resolve slices with Python semantics and alias dynamic shared
    arrays at the region base."""
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
    _lcuda.registry.lower_getattr(types.Array, "strides")(lower_strides)

    lower_class = _ml.MLIRLower
    lower_class._request_dynamic_shared_memory = _request_dynamic_shared_memory
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

    stock_pre_codegen = _mlir_optimization.run_pre_codegen_patterns

    @functools.wraps(stock_pre_codegen)
    def run_pre_codegen_patterns(module, *args, **kwargs):
        result = stock_pre_codegen(module, *args, **kwargs)
        _externalize_dynamic_shared_globals(module)
        return result

    _mlir_optimization.run_pre_codegen_patterns = run_pre_codegen_patterns


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
