"""Performance and lineinfo patches for stock numba-cuda.

CuBIE JIT-compiles deeply nested ``inline='always'`` device-function
stacks; several numba-cuda frontend algorithms are superlinear or
redundant on the large flattened functions this produces, making
kernel compilation 5-15x slower than necessary. This module rebinds
the affected functions and methods with the implementations carried
on the ``cubie_patch`` branch of the ccam80/numba-cuda fork, so the
improvements ship with cubie while awaiting upstream acceptance.

Every performance patch is behaviour-preserving: live maps, SSA form,
inferred signatures and generated PTX are identical (verified
bit-exact on saturating production batches); only compile time
changes. The ``fix/lineinfo-cross-file`` group deliberately changes
emitted debug metadata; that change is the fix.

Patch groups, each with its fork feature branch:

``compat/arrayobj-row-stack-alias`` (no fork branch)
    ``numba.cuda.np.arrayobj`` imports on numpy 2.5, which does not
    define the ``np.row_stack`` it registers a lowering against.
``feat/lazy-postproc-liveness``
    ``PostProcessor.run`` computes entry liveness only for generator
    functions; all other consumers use the lazily evaluated
    ``VariableLifetime`` properties.
``feat/lazy-error-markup``
    Error markup builds strings without re-wrapping the terminal via
    colorama per call, and the numba.core base class no longer
    re-highlights already-highlighted messages.
``feat/ssa-restricted-sweeps``
    SSA rewrite passes visit only blocks that define or use the
    variable being processed.
``feat/inline-callee-ir-cache``
    Callee IR for inlining is memoised per (function, flags) and each
    call site receives a fast structural clone; includes the
    ``preserve_ir`` form of ``InlineWorker.inline_ir``.
``feat/lowering-call-type-cache``
    Signatures resolved while lowering getitem/setitem/binop
    instructions are cached per function. (The delitem site inside
    ``lower_inst`` is not patched here — copying that whole method
    was not worth a construct cubie kernels never emit.)
``feat/liveness-bitsets`` (stacked on the loop-invariant hoist and
reverse-order sweep commits)
    ``compute_live_map`` runs both dataflow fix points on bitsets in
    reverse-topological sweep order.
``perf: Lower._find_singly_assigned_variable`` (cubie_patch)
    Linear in block size instead of quadratic.
``fix/lineinfo-cross-file`` (fork branch ``fix-lineinfo-cross-file``,
proposed upstream)
    Debug locations for code inlined at the numba IR level from other
    source files keep their own file (scoped through a
    DILexicalBlockFile) and their own line numbers (the prologue line
    clamp only applies to lines in the kernel's own file), so
    ``lineinfo=True`` attributes every PTX line to the source file
    that produced it instead of mis-filing or dropping cross-file
    lines.

Import this module before compiling any kernel (cubie imports it at
package import). Each group self-detects whether the installed
numba-cuda already contains its change (the cubie_patch fork, or a
future upstream release) and becomes a no-op. Under
``NUMBA_ENABLE_CUDASIM=1`` the module does nothing. Remove each shim
once the corresponding change lands upstream.
"""

import copy
import importlib
import inspect
import linecache
import operator
import os
import weakref
from collections import defaultdict

import numpy as np

if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") != "1":
    from numba.cuda import types
    from numba.cuda.core import (
        analysis as _analysis,
        errors as _errors,
        inline_closurecall as _icc,
        ir,
        ir_utils as _ir_utils,
        postproc as _postproc,
        ssa as _ssa,
        transforms as _transforms,
    )
    from numba.cuda import debuginfo as _debuginfo
    from numba.cuda import lowering as _lowering

    _PATCHES_ACTIVE = True
else:  # pragma: no cover - simulator has no compiler frontend
    _PATCHES_ACTIVE = False


# ----------------------------------------------------------------- #
# compat/arrayobj-row-stack-alias: import on numpy >= 2.5            #
# ----------------------------------------------------------------- #


def _row_stack_removed(*args, **kwargs):  # pragma: no cover
    """Stand in for ``np.row_stack``, absent from numpy 2.5."""
    raise AttributeError("module 'numpy' has no attribute 'row_stack'")


def _patch_row_stack_overload():
    """Import ``numba.cuda.np.arrayobj`` on numpy 2.5 and later.

    The module registers a vstack lowering against ``np.row_stack``,
    which numpy 2.5 does not define. The stand-in is bound to that
    name for the one import, so the registration lands on it and
    array expressions keep typing against ``np.vstack``.
    """
    if hasattr(np, "row_stack"):
        return  # numpy still carries the alias
    np.row_stack = _row_stack_removed
    try:
        importlib.import_module("numba.cuda.np.arrayobj")
    finally:
        del np.row_stack


# ----------------------------------------------------------------- #
# feat/liveness-bitsets: compute_live_map on bitsets                 #
# ----------------------------------------------------------------- #

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
                incoming = (live_vars & def_reach_map[inc_blk]) & ~def_bits[
                    inc_blk
                ]
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
    if hasattr(_analysis, "_BYTE_BITS"):
        return
    stock = _analysis.compute_live_map
    _analysis._BYTE_BITS = _BYTE_BITS
    _analysis.compute_live_map = _compute_live_map
    # ir_utils imports the function by name at module import time.
    if getattr(_ir_utils, "compute_live_map", None) is stock:
        _ir_utils.compute_live_map = _compute_live_map


# ----------------------------------------------------------------- #
# feat/lazy-postproc-liveness: PostProcessor.run                     #
# ----------------------------------------------------------------- #


def _patch_postproc():
    src = inspect.getsource(_postproc.PostProcessor.run)
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
        self.func_ir.blocks = _transforms.canonicalize_cfg(self.func_ir.blocks)
        vlt = _postproc.VariableLifetime(self.func_ir.blocks)
        self.func_ir.variable_lifetime = vlt

        if self.func_ir.is_generator:
            # Only generator info consumes the entry-liveness result
            # (via get_block_entry_vars); non-generator consumers of
            # liveness use the lazily computed properties on
            # VariableLifetime instead, so the fix-point analyses are
            # not run eagerly for them.
            bev = _analysis.compute_live_variables(
                vlt.cfg,
                self.func_ir.blocks,
                vlt.usedefs.defmap,
                vlt.deadmaps.combined,
            )
            for offset, ir_block in self.func_ir.blocks.items():
                self.func_ir.block_entry_vars[ir_block] = bev[offset]

            self.func_ir.generator_info = _postproc.GeneratorInfo()
            self._compute_generator_info()
        else:
            self.func_ir.generator_info = None

        # Emit del nodes, do this last as the generator info parsing
        # generates and then strips dels as part of its analysis.
        if emit_dels:
            self._insert_var_dels(extend_lifetimes=extend_lifetimes)

    _postproc.PostProcessor.run = run


# ----------------------------------------------------------------- #
# feat/lazy-error-markup: string-only markup, single highlight pass  #
# ----------------------------------------------------------------- #


def _patch_error_markup():
    scheme_cls = getattr(_errors, "HighlightColorScheme", None)
    if scheme_cls is not None and "ColorShell" in inspect.getsource(
        scheme_cls._markup
    ):
        from colorama import Style

        def _markup(self, msg, color=None, style=Style.BRIGHT):
            # This only builds a string; it does not write to a
            # stream. Wrapping the standard streams with colorama
            # (ColorShell) is unnecessary for that and was undone
            # before anything printed the string, yet its init/deinit
            # dominated error construction when typing speculatively
            # instantiates many exceptions. Emit the same bytes
            # without touching the streams.
            features = ""
            if color:
                features += color
            if style:
                features += style
            return features + msg + Style.RESET_ALL

        scheme_cls._markup = _markup

    if "highlighting=False" in inspect.getsource(_errors.NumbaError.__init__):
        return  # base-class double highlight already suppressed

    def __init__(self, msg, loc=None, highlighting=True):
        self.msg = msg
        self.loc = loc
        if highlighting:
            highlight = _errors.termcolor().errmsg
        else:

            def highlight(x):
                return x

        if loc:
            new_msg = "%s\n%s\n" % (msg, loc.strformat())
        else:
            new_msg = "%s" % (msg,)
        # The message is already highlighted here; tell the
        # numba.core base class not to run its own termcolor pass
        # over it again.
        import numba.core.errors

        numba.core.errors.NumbaError.__init__(
            self, highlight(new_msg), highlighting=False
        )

    _errors.NumbaError.__init__ = __init__


# ----------------------------------------------------------------- #
# feat/ssa-restricted-sweeps: per-variable block restriction         #
# ----------------------------------------------------------------- #


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
    _ssa._run_block_analysis(blocks, states, _ssa._GatherDefsHandler())
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
        newblk = ir.Block(scope=blk.scope, loc=blk.loc)
        newbody = []
        states["label"] = label
        states["block"] = blk
        for stmt in _ssa._run_ssa_block_pass(states, blk, handler):
            assert stmt is not None
            newbody.append(stmt)
        newblk.body = newbody
        newblocks[label] = newblk
    return newblocks


def _ssa_fresh_vars(blocks, varname, def_labels):
    """Rewrite to put fresh variable names"""
    states = _ssa._make_states(blocks)
    states["varname"] = varname
    states["defmap"] = defmap = defaultdict(list)
    newblocks = _ssa_run_block_rewrite(
        blocks, states, _ssa._FreshVarHandler(), def_labels
    )
    return newblocks, defmap


def _ssa_fix_ssa_vars(
    blocks, varname, defmap, cfg, df_plus, cache_list_vars, use_labels
):
    """Rewrite all uses to ``varname`` given the definition map"""
    states = _ssa._make_states(blocks)
    states["varname"] = varname
    states["defmap"] = defmap
    states["phimap"] = phimap = defaultdict(list)
    states["cfg"] = cfg
    states["phi_locations"] = _ssa._compute_phi_locations(df_plus, defmap)
    newblocks = _ssa_run_block_rewrite(
        blocks, states, _ssa._FixSSAVars(cache_list_vars), use_labels
    )
    # insert phi nodes
    for label, philist in phimap.items():
        curblk = newblocks[label]
        # Prepend PHI nodes to the block. Build a fresh block rather
        # than mutating in place: phi locations include pass-through
        # blocks, and input block objects must never be mutated.
        newblk = ir.Block(scope=curblk.scope, loc=curblk.loc)
        newblk.body = philist + curblk.body
        newblocks[label] = newblk
    return newblocks


def _ssa_run_ssa(blocks):
    """Run SSA reconstruction on IR blocks of a function."""
    if not blocks:
        return {}
    cfg = _ssa.compute_cfg_from_blocks(blocks)
    df_plus = _ssa._iterated_domfronts(cfg)
    violators, defs, uses = _ssa_find_defs_violators(blocks, cfg)
    cache_list_vars = _ssa._CacheListVars()

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

    cfg_post = _ssa.compute_cfg_from_blocks(blocks)
    if cfg_post != cfg:
        raise _errors.CompilerError("CFG mutated in SSA pass")
    return blocks


def _patch_ssa():
    params = inspect.signature(_ssa._fresh_vars).parameters
    if "def_labels" in params:
        return
    _ssa._find_defs_violators = _ssa_find_defs_violators
    _ssa._run_block_rewrite = _ssa_run_block_rewrite
    _ssa._fresh_vars = _ssa_fresh_vars
    _ssa._fix_ssa_vars = _ssa_fix_ssa_vars
    _ssa._run_ssa = _ssa_run_ssa


# ----------------------------------------------------------------- #
# feat/inline-callee-ir-cache (incl. preserve_ir inline_ir)          #
# ----------------------------------------------------------------- #

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
    new_scope = ir.Scope(parent=old_scope.parent, loc=old_scope.loc)
    new_scope.redefined.update(old_scope.redefined)
    for name, versions in old_scope.var_redefinitions.items():
        new_scope.var_redefinitions[name] = set(versions)

    varmap = {}
    for name, var in old_scope.localvars._con.items():
        varmap[name] = new_scope.define(name, var.loc)

    def clone_value(value):
        if isinstance(value, ir.var_types):
            return varmap[value.name]
        if isinstance(value, ir.expr_types):
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
        new_block = ir.Block(scope=new_scope, loc=block.loc)
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
            _ir_utils._the_max_label.next(),
            max(caller_ir.blocks.keys()),
        )
        callee_blocks = _icc.add_offset_to_labels(callee_blocks, max_label + 1)
        callee_blocks = _icc.simplify_CFG(callee_blocks)
        callee_ir.blocks = callee_blocks
        min_label = min(callee_blocks.keys())
        max_label = max(callee_blocks.keys())
        _ir_utils._the_max_label.update(max_label)
        self.debug_print("After relabel")
        _icc._debug_dump(callee_ir)

        # 2. rename all local variables in callee_ir with new locals
        # created in caller_ir
        callee_scopes = _icc._get_all_scopes(callee_blocks)
        self.debug_print("callee_scopes = ", callee_scopes)
        assert len(callee_scopes) == 1
        callee_scope = callee_scopes[0]
        var_dict = {}
        for var in tuple(callee_scope.localvars._con.values()):
            if var.name not in callee_freevars:
                inlined_name = _icc._created_inlined_var_name(
                    callee_ir.func_id.unique_name, var.name
                )
                new_var = scope.redefine(inlined_name, loc=var.loc)
                callee_scope.redefine(inlined_name, loc=var.loc)
                var_dict[var.name] = new_var
        self.debug_print("var_dict = ", var_dict)
        _icc.replace_vars(callee_blocks, var_dict)
        self.debug_print("After local var rename")
        _icc._debug_dump(callee_ir)

        # 3. replace formal parameters with actual arguments
        callee_func = callee_ir.func_id.func
        args = _icc._get_callee_args(
            call_expr, callee_func, block.body[i].loc, caller_ir
        )

        # 4. Update typemap
        if self._permit_update_type_and_call_maps:
            if arg_typs is None:
                raise TypeError("arg_typs should have a value not None")
            self.update_type_and_call_maps(callee_ir, arg_typs)
            callee_blocks = callee_ir.blocks

        self.debug_print("After arguments rename: ")
        _icc._debug_dump(callee_ir)

        _icc._replace_args_with(callee_blocks, args)
        # 5. split caller blocks into two
        new_blocks = []
        new_block = ir.Block(scope, block.loc)
        new_block.body = block.body[i + 1 :]
        new_label = _icc.next_label()
        caller_ir.blocks[new_label] = new_block
        new_blocks.append((new_label, new_block))
        block.body = block.body[:i]
        block.body.append(ir.Jump(min_label, instr.loc))

        # 6. replace Return with assignment to LHS
        topo_order = _icc.find_topo_order(callee_blocks)
        _icc._replace_returns(callee_blocks, instr.target, new_label)

        if (
            instr.target.name in caller_ir._definitions
            and call_expr in caller_ir._definitions[instr.target.name]
        ):
            caller_ir._definitions[instr.target.name].remove(call_expr)

        # 7. insert all new blocks, and add back definitions
        for label in topo_order:
            block = callee_blocks[label]
            block.scope = scope
            _icc._add_definitions(caller_ir, block)
            caller_ir.blocks[label] = block
            new_blocks.append((label, block))
        self.debug_print("After merge in")
        _icc._debug_dump(caller_ir)

        return callee_ir_original, callee_blocks, var_dict, new_blocks

    return inline_ir


def _patch_inline_worker():
    if hasattr(_icc, "_clone_callee_ir"):
        return
    _icc._clone_callee_ir = _clone_callee_ir
    _icc._callee_ir_cache = _callee_ir_cache

    worker = _icc.InlineWorker
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


# ----------------------------------------------------------------- #
# feat/lowering-call-type-cache + linear singly-assigned scan        #
# ----------------------------------------------------------------- #


def _patch_lowering():
    base = _lowering.BaseLower
    if hasattr(base, "_resolve_call_type_cached"):
        return

    orig_base_init = base.__init__

    def __init__(self, context, library, fndesc, func_ir, metadata=None):
        # Signatures resolved while lowering individual instructions.
        # The typing context is frozen during lowering, so a
        # (callable type, argument types) pair always resolves to the
        # same signature; without the cache every getitem/setitem/
        # binop instruction repeated full template matching. Set
        # before the original __init__ because it invokes the
        # subclass init() hook.
        self._call_type_cache = {}
        orig_base_init(self, context, library, fndesc, func_ir, metadata)

    def _resolve_call_type_cached(self, fnop, args):
        key = (fnop, args)
        sig = self._call_type_cache.get(key)
        if sig is None:
            sig = fnop.get_call_type(self.context.typing_context, args, {})
            self._call_type_cache[key] = sig
        return sig

    base.__init__ = __init__
    base._resolve_call_type_cached = _resolve_call_type_cached

    lower = _lowering.Lower

    def lower_setitem(self, target_var, index_var, value_var, signature):
        target = self.loadvar(target_var.name)
        value = self.loadvar(value_var.name)
        index = self.loadvar(index_var.name)

        targetty = self.typeof(target_var.name)
        valuety = self.typeof(value_var.name)
        indexty = self.typeof(index_var.name)

        op = operator.setitem
        fnop = self.context.typing_context.resolve_value_type(op)
        callsig = self._resolve_call_type_cached(fnop, signature.args)
        impl = self.context.get_function(fnop, callsig)

        # Convert argument to match
        if isinstance(targetty, types.Optional):
            target = self.context.cast(
                self.builder, target, targetty, targetty.type
            )
        else:
            ul = types.unliteral
            assert ul(targetty) == ul(signature.args[0])

        index = self.context.cast(
            self.builder, index, indexty, signature.args[1]
        )
        value = self.context.cast(
            self.builder, value, valuety, signature.args[2]
        )

        return impl(self.builder, (target, index, value))

    def lower_getitem(self, resty, expr, value, index, signature):
        baseval = self.loadvar(value.name)
        indexval = self.loadvar(index.name)
        # Get implementation of getitem
        op = operator.getitem
        fnop = self.context.typing_context.resolve_value_type(op)
        callsig = self._resolve_call_type_cached(fnop, signature.args)
        impl = self.context.get_function(fnop, callsig)

        argvals = (baseval, indexval)
        argtyps = (self.typeof(value.name), self.typeof(index.name))
        castvals = [
            self.context.cast(self.builder, av, at, ft)
            for av, at, ft in zip(argvals, argtyps, signature.args)
        ]
        res = impl(self.builder, castvals)
        return self.context.cast(
            self.builder, res, signature.return_type, resty
        )

    from numba.cuda import typing
    from numba.cuda.core.errors import TypingError

    _lit_or_omitted = _lowering._lit_or_omitted

    def lower_binop(self, resty, expr, op):
        # if op in utils.OPERATORS_TO_BUILTINS:
        # map operator.the_op => the corresponding types.Function()
        # TODO: is this looks dodgy ...
        op = self.context.typing_context.resolve_value_type(op)

        lhs = expr.lhs
        rhs = expr.rhs
        static_lhs = expr.static_lhs
        static_rhs = expr.static_rhs
        lty = self.typeof(lhs.name)
        rty = self.typeof(rhs.name)
        lhs = self.loadvar(lhs.name)
        rhs = self.loadvar(rhs.name)

        # Convert argument to match
        signature = self.fndesc.calltypes[expr]
        lhs = self.context.cast(self.builder, lhs, lty, signature.args[0])
        rhs = self.context.cast(self.builder, rhs, rty, signature.args[1])

        def cast_result(res):
            return self.context.cast(
                self.builder, res, signature.return_type, resty
            )

        # First try with static operands, if known
        def try_static_impl(tys, args):
            if any(a is ir.UNDEFINED for a in args):
                return None
            try:
                if isinstance(op, types.Function):
                    static_sig = op.get_call_type(
                        self.context.typing_context, tys, {}
                    )
                else:
                    static_sig = typing.signature(signature.return_type, *tys)
            except TypingError:
                return None
            try:
                static_impl = self.context.get_function(op, static_sig)
                return static_impl(self.builder, args)
            except NotImplementedError:
                return None

        res = try_static_impl(
            (_lit_or_omitted(static_lhs), _lit_or_omitted(static_rhs)),
            (static_lhs, static_rhs),
        )
        if res is not None:
            return cast_result(res)

        res = try_static_impl(
            (_lit_or_omitted(static_lhs), rty),
            (static_lhs, rhs),
        )
        if res is not None:
            return cast_result(res)

        res = try_static_impl(
            (lty, _lit_or_omitted(static_rhs)),
            (lhs, static_rhs),
        )
        if res is not None:
            return cast_result(res)

        # Normal implementation for generic arguments

        sig = self._resolve_call_type_cached(op, signature.args)
        impl = self.context.get_function(op, sig)
        res = impl(self.builder, (lhs, rhs))
        return cast_result(res)

    lower.lower_setitem = lower_setitem
    lower.lower_getitem = lower_getitem
    lower.lower_binop = lower_binop

    # Linear-time singly-assigned-variable scan (quadratic upstream).
    def _find_singly_assigned_variable(self):
        func_ir = self.func_ir
        blocks = func_ir.blocks

        sav = set()

        if not self.func_ir.func_id.is_generator:
            use_defs = _analysis.compute_use_defs(blocks)
            alloca_vars = _lowering.must_use_alloca(blocks)

            # Compute where variables are defined
            var_assign_map = defaultdict(set)
            for blk, vl in use_defs.defmap.items():
                for var in vl:
                    var_assign_map[var].add(blk)

            # Compute where variables are used
            var_use_map = defaultdict(set)
            for blk, vl in use_defs.usemap.items():
                for var in vl:
                    var_use_map[var].add(blk)

            # Per-block count of assignments per target name. Computed
            # once per block (lazily) so the singly-assigned check
            # below is an O(1) lookup instead of rescanning the whole
            # block body for every variable (which is quadratic in
            # block size on large kernels).
            assign_counts_cache = {}

            def assign_counts(defblk):
                counts = assign_counts_cache.get(defblk)
                if counts is None:
                    counts = defaultdict(int)
                    for stmt in self.blocks[defblk].body:
                        if isinstance(stmt, ir.assign_types):
                            counts[stmt.target.name] += 1
                    assign_counts_cache[defblk] = counts
                return counts

            # Keep only variables that are defined locally and used
            # locally
            for var in var_assign_map:
                if var not in alloca_vars and len(var_assign_map[var]) == 1:
                    # Usemap does not keep locally defined variables.
                    if len(var_use_map[var]) == 0:
                        # Ensure that the variable is not defined
                        # multiple times in the block
                        [defblk] = var_assign_map[var]
                        if assign_counts(defblk)[var] == 1:
                            sav.add(var)

        self._singly_assigned_vars = sav
        self._blk_local_varmap = {}

    if "assign_counts" not in inspect.getsource(
        lower._find_singly_assigned_variable
    ):
        lower._find_singly_assigned_variable = _find_singly_assigned_variable


# ----------------------------------------------------------------- #
# fix/lineinfo-cross-file: per-file line info for inlined code       #
# ----------------------------------------------------------------- #


def _patch_cross_file_lineinfo():
    """Attribute inlined cross-file lines to their own file and line.

    ``inline='always'`` device functions are folded into the caller at
    the numba IR level with their source locations intact, but stock
    lowering discards the file (``mark_location`` receives only a line
    number, and every DILocation is scoped to the kernel's
    DISubprogram) and ``_adjust_line_if_prologue`` rewrites any line
    numerically below the kernel's ``def`` line to the ``def`` line, a
    prologue heuristic that is only valid for lines in the kernel's
    own file. Together these mis-file or destroy every line of an
    inlined cross-file device function.

    The fix scopes locations whose file differs from the subprogram's
    through a DILexicalBlockFile referencing the correct DIFile, and
    restricts the prologue clamp to same-file lines. The DIBuilder
    learns the current location from a weak back-reference to the
    Lower installed at construction, because the stock
    ``mark_location`` signature carries no file information.
    """
    lower = _lowering.Lower
    base = _lowering.BaseLower
    if "filename" in inspect.getsource(lower._adjust_line_if_prologue):
        return  # already file-aware (cubie_patch fork or upstream fix)

    def _adjust_line_if_prologue(self, line):
        """Adjust prologue line numbers to use the 'def' line.

        The function prologue doesn't have corresponding source lines.
        These instructions inherit line numbers from co_firstlineno,
        which points to the decorator line when decorators are
        present. This method redirects such lines to the 'def' line.
        Lines from other source files (inlined device functions) are
        never prologue lines and pass through unchanged.
        """
        # Every call site passes line == self.loc.line, so self.loc
        # names the file the line belongs to.
        filename = getattr(self.loc, "filename", None)
        if filename is not None and filename != self.defn_loc.filename:
            return line
        return self.defn_loc.line if line < self.defn_loc.line else line

    lower._adjust_line_if_prologue = _adjust_line_if_prologue

    orig_init = base.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # The DIBuilder needs the file of the location being marked,
        # which only the Lower knows (self.loc); mark_location's
        # signature cannot carry it without forking the call sites.
        # Guarded so a future numba-cuda that leaves debuginfo unset
        # degrades to stock behaviour instead of failing every compile.
        if getattr(self, "debuginfo", None) is not None:
            self.debuginfo._cubie_lower = weakref.ref(self)

    base.__init__ = __init__

    def mark_location(self, builder, line):
        builder.debug_metadata = self._cubie_add_location(line)

    def _cubie_add_location(self, line):
        lower_ref = getattr(self, "_cubie_lower", None)
        lower_inst = lower_ref() if lower_ref is not None else None
        loc = getattr(lower_inst, "loc", None)
        filename = getattr(loc, "filename", None)
        if filename is None or filename == self.filepath:
            return self._add_location(line)
        if filename.startswith("<") and filename.endswith(">"):
            # Pseudo-paths from exec'd sources (e.g. notebook cells)
            # get a file entry only when linecache can serve their
            # source; otherwise keep the subprogram scope, matching
            # stock attribution for lines no tool could display.
            if not linecache.getlines(filename):
                return self._add_location(line)
        else:
            # self.filepath is stored absolute; normalize so a
            # relative spelling of the subprogram's own file is not
            # treated as a different file.
            filename = os.path.abspath(filename)
            if filename == self.filepath:
                return self._add_location(line)
        # llvmlite dedupes debug info nodes, so re-adding the same
        # DIFile/DILexicalBlockFile returns the cached node.
        difile = self.module.add_debug_info(
            "DIFile",
            {
                "directory": os.path.dirname(filename),
                "filename": os.path.basename(filename),
            },
        )
        scope = self.module.add_debug_info(
            "DILexicalBlockFile",
            {
                "scope": self.subprograms[-1],
                "file": difile,
                "discriminator": 0,
            },
        )
        return self.module.add_debug_info(
            "DILocation",
            {"line": line, "column": 1, "scope": scope},
        )

    _debuginfo.DIBuilder.mark_location = mark_location
    _debuginfo.DIBuilder._cubie_add_location = _cubie_add_location


# ----------------------------------------------------------------- #
# application                                                        #
# ----------------------------------------------------------------- #


def apply_patches():
    """Apply all patch groups that the installed numba-cuda needs."""
    if not _PATCHES_ACTIVE:
        return
    _patch_row_stack_overload()
    _patch_live_map()
    _patch_postproc()
    _patch_error_markup()
    _patch_ssa()
    _patch_inline_worker()
    _patch_lowering()
    _patch_cross_file_lineinfo()


apply_patches()
