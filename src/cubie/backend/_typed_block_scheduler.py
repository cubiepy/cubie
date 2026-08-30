"""Intra-block statement scheduling of typed Numba IR (MLIR backend).

``TypedBlockScheduler`` registers through
``numba_cuda_mlir.extending.register_typed_planner`` and reorders
each basic block of the fully inlined typed IR under a dependency
DAG (flow edges, name chains, per-element memory chains, effect
barriers, Del pins). Policies (see
:mod:`cubie.backend._block_schedule_policies`): ``source``,
``anchor_dfs`` (default), ``dfs``, ``liveness``,
``longlived_dfs``, ``inject``.
Knobs: ``CUBIE_BLOCK_SCHEDULE`` (policy),
``CUBIE_BLOCK_SCHEDULE_DUMP`` (gzip graph dump),
``CUBIE_BLOCK_SCHEDULE_ORDER`` (JSON orders for ``inject``).
Import only via ``cubie.backend._mlir_compat`` after hook
detection.
"""

import gzip
import json
import os

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import ir
from numba_cuda_mlir.extending import TypedWholeFunctionPlanner

from cubie.backend._block_schedule_policies import (
    BLOCK_SCHEDULE_POLICIES,
    ScheduleNode,
    modeled_peak,
    order_nodes,
)

#: Dump the dependency graph of every large block to this gzip path.
_DUMP_ENV = "CUBIE_BLOCK_SCHEDULE_DUMP"
#: Read explicit per-block orders (JSON: label -> node order) here.
_INJECT_ENV = "CUBIE_BLOCK_SCHEDULE_ORDER"
_DUMP_MIN_STATEMENTS = 2000

_METADATA_KEY = "typed_block_scheduler"

_PURE_EXPR_OPS = frozenset(
    {
        "binop",
        "inplace_binop",
        "unary",
        "cast",
        "exhaust_iter",
        "build_tuple",
        "getattr",
        "null",
        "undef",
    }
)

_PURE_CALL_MODULE_ROOTS = frozenset(
    {
        "math",
        "cmath",
        "builtins",
        "operator",
        "numpy",
        "numba",
        "numba_cuda_mlir",
        "cubie",
    }
)

_IMPURE_CALL_MARKERS = (
    "sync",
    "atomic",
    "fence",
    "print",
    "random",
    "stwt",
    "vote",
    "shfl",
    "ballot",
    "match_any",
    "match_all",
)

_BARRIER_STATEMENT_TYPES = tuple(
    statement_type
    for statement_type in (
        getattr(ir, "Print", None),
        getattr(ir, "SetAttr", None),
        getattr(ir, "EnterWith", None),
        getattr(ir, "PopBlock", None),
        getattr(ir, "Raise", None),
        getattr(ir, "StaticRaise", None),
        getattr(ir, "DynamicRaise", None),
    )
    if statement_type is not None
)


def _expr_op(statement):
    if isinstance(statement, ir.Assign) and isinstance(
        statement.value, ir.Expr
    ):
        return statement.value.op
    return None


class TypedBlockScheduler(TypedWholeFunctionPlanner):
    """Reorder statements inside each block of the typed IR."""

    #: Ordering policy applied to every block; see module docstring.
    policy = "anchor_dfs"
    #: The active policy is part of the kernel-cache fingerprint.
    cache_safe = True

    def run(self) -> bool:
        policy = self.state.metadata.get(
            "typed_block_scheduler_policy", type(self).policy
        )
        if policy not in BLOCK_SCHEDULE_POLICIES:
            raise ValueError(
                f"unknown block schedule policy {policy!r}"
            )
        func_ir = self.state.func_ir
        typemap = self.state.typemap
        roots = self._alias_roots(func_ir, typemap)
        scalar_names = {
            name
            for name, numba_type in typemap.items()
            if not isinstance(numba_type, types.Array)
        }
        live_out = self._block_live_out(func_ir)
        dump_path = os.environ.get(_DUMP_ENV)
        inject_orders = {}
        if policy == "inject":
            with open(
                os.environ[_INJECT_ENV], "r", encoding="utf-8"
            ) as fh:
                inject_orders = {
                    int(label): order
                    for label, order in json.load(fh).items()
                }
        dumped_blocks = {}
        modified = False
        stats = {
            "blocks": 0,
            "reordered_blocks": 0,
            "moved_statements": 0,
            "statements": 0,
            "largest_block": 0,
            "modeled_peak_source": 0,
            "modeled_peak_scheduled": 0,
        }
        for label, block in func_ir.blocks.items():
            stats["blocks"] += 1
            stats["statements"] += len(block.body)
            stats["largest_block"] = max(
                stats["largest_block"], len(block.body)
            )
            if policy == "source" and dump_path is None:
                continue
            scheduled = self._schedule_block(
                block,
                policy,
                roots,
                typemap,
                scalar_names,
                live_out.get(label, frozenset()),
                inject_orders.get(label),
                dumped_blocks if dump_path is not None else None,
                label,
            )
            if scheduled is None:
                continue
            order, block_stats = scheduled
            stats["modeled_peak_source"] = max(
                stats["modeled_peak_source"],
                block_stats["peak_source"],
            )
            stats["modeled_peak_scheduled"] = max(
                stats["modeled_peak_scheduled"],
                block_stats["peak_scheduled"],
            )
            if policy == "source":
                continue
            body = block.body
            new_body = [body[index] for index in order] + [body[-1]]
            moved = sum(
                1
                for position, index in enumerate(order)
                if index != position
            )
            if moved:
                block.body = new_body
                stats["reordered_blocks"] += 1
                stats["moved_statements"] += moved
                modified = True
        if dump_path is not None and dumped_blocks:
            with gzip.open(dump_path, "wt", encoding="utf-8") as fh:
                json.dump(dumped_blocks, fh)
        stats["policy"] = policy
        self.state.metadata[_METADATA_KEY] = stats
        return modified

    # -- alias analysis ------------------------------------------------

    def _alias_roots(self, func_ir, typemap):
        """Map each array-typed name to ``(root, element offset)``."""

        roots = {}

        def view_offset(expression):
            index = getattr(expression, "index", None)
            if isinstance(index, ir.Var):
                try:
                    definition = func_ir.get_definition(index)
                except Exception:
                    return None
                if not isinstance(
                    definition, (ir.Const, ir.Global, ir.FreeVar)
                ):
                    return None
                index = definition.value
            if isinstance(index, slice):
                if (
                    index.step in (None, 1)
                    and isinstance(index.start, int)
                    and index.start >= 0
                ):
                    return index.start
                if index == slice(None):
                    return 0
                return None
            return None

        def resolve(name, seen):
            if name in roots:
                return roots[name]
            if name in seen:
                return (None, None)
            seen.add(name)
            definitions = func_ir._definitions.get(name, [])
            if len(definitions) != 1:
                entry = (None, None)
            else:
                value = definitions[0]
                if isinstance(value, ir.Arg):
                    entry = (("arg", value.index), 0)
                elif isinstance(value, ir.Var):
                    entry = resolve(value.name, seen)
                elif isinstance(value, ir.Expr):
                    if value.op == "cast":
                        parent = value.value
                        entry = (
                            resolve(parent.name, seen)
                            if isinstance(parent, ir.Var)
                            else (None, None)
                        )
                    elif value.op in ("getitem", "static_getitem"):
                        parent = value.value
                        if isinstance(parent, ir.Var):
                            root, offset = resolve(parent.name, seen)
                            parent_type = typemap.get(parent.name)
                            step = (
                                view_offset(value)
                                if getattr(parent_type, "ndim", 0)
                                == 1
                                else None
                            )
                            if offset is None or step is None:
                                entry = (root, None)
                            else:
                                entry = (root, offset + step)
                        else:
                            entry = (None, None)
                    elif value.op == "call":
                        entry = (("alloc", id(value)), 0)
                    else:
                        entry = (None, None)
                else:
                    entry = (None, None)
            roots[name] = entry
            return entry

        for name, numba_type in typemap.items():
            if isinstance(numba_type, types.Array):
                resolve(name, set())
        return roots

    # -- liveness ------------------------------------------------------

    def _block_live_out(self, func_ir):
        """Names referenced by any other block or a terminator."""

        blocks_referencing = {}
        block_names = {}
        for label, block in func_ir.blocks.items():
            names = set()
            for statement in block.body:
                for var in statement.list_vars():
                    names.add(var.name)
            block_names[label] = names
            for name in names:
                blocks_referencing[name] = (
                    blocks_referencing.get(name, 0) + 1
                )
        live_out = {}
        for label, block in func_ir.blocks.items():
            names = block_names[label]
            external = {
                name
                for name in names
                if blocks_referencing[name] > 1
            }
            external |= {
                var.name for var in block.terminator.list_vars()
            }
            live_out[label] = frozenset(external)
        return live_out

    # -- statement classification --------------------------------------

    def _resolve_callee(self, func_ir, expr):
        """Return the Python object a call targets, or None."""

        attributes = []
        value = expr.func
        for _ in range(8):
            try:
                definition = func_ir.get_definition(value)
            except Exception:
                return None
            if isinstance(definition, (ir.Global, ir.FreeVar)):
                target = definition.value
                for attribute in reversed(attributes):
                    try:
                        target = getattr(target, attribute)
                    except AttributeError:
                        return None
                return target
            if (
                isinstance(definition, ir.Expr)
                and definition.op == "getattr"
            ):
                attributes.append(definition.attr)
                value = definition.value
                continue
            return None
        return None

    def _call_is_pure(self, func_ir, typemap, expr):
        """Whether a call expression is safe to reorder freely."""

        for argument in expr.list_vars():
            if argument is expr.func:
                continue
            if isinstance(typemap.get(argument.name), types.Array):
                return False
        callee = self._resolve_callee(func_ir, expr)
        if callee is None:
            return False
        qualname = (
            getattr(callee, "__qualname__", "")
            or getattr(callee, "__name__", "")
            or repr(callee)
        ).lower()
        module = getattr(callee, "__module__", "") or ""
        haystack = f"{module}.{qualname}"
        if any(marker in haystack for marker in _IMPURE_CALL_MARKERS):
            return False
        return module.split(".")[0] in _PURE_CALL_MODULE_ROOTS

    def _constant_index(self, func_ir, operation):
        """Return a non-negative constant index, or ``None``."""

        index = getattr(operation, "index", None)
        if isinstance(index, ir.Var):
            try:
                definition = func_ir.get_definition(index)
            except Exception:
                return None
            if not isinstance(
                definition, (ir.Const, ir.Global, ir.FreeVar)
            ):
                return None
            index = definition.value
        if isinstance(index, tuple):
            if all(
                isinstance(item, int) and item >= 0
                for item in index
            ):
                return index[0] if len(index) == 1 else index
            return None
        if isinstance(index, int) and not isinstance(index, bool):
            return index if index >= 0 else None
        return None

    @staticmethod
    def _absolute_index(index_key, base_offset):
        """Combine an access index with its view's element offset."""

        if index_key is None or base_offset is None:
            return None
        if isinstance(index_key, tuple):
            return index_key if base_offset == 0 else None
        return index_key + base_offset

    # -- graph construction --------------------------------------------

    def _schedule_block(
        self,
        block,
        policy,
        roots,
        typemap,
        scalar_names,
        live_out,
        injected_order=None,
        dump_sink=None,
        label=None,
    ):
        body = block.body
        if len(body) < 3:
            return None
        statements = body[:-1]

        pinned = 0
        for statement in statements:
            if isinstance(statement, ir.Assign) and isinstance(
                statement.value, ir.Arg
            ):
                pinned += 1
                continue
            break
        movable = statements[pinned:]
        if len(movable) < 2:
            return None
        if any(
            _expr_op(statement) == "phi" for statement in movable
        ):
            return None

        nodes = []
        for offset, statement in enumerate(movable):
            defs = set()
            uses = []
            if isinstance(statement, ir.Assign):
                defs.add(statement.target.name)
                value = statement.value
                if isinstance(value, ir.Expr):
                    uses = [var.name for var in value.list_vars()]
                elif isinstance(value, ir.Var):
                    uses = [value.name]
            elif isinstance(statement, ir.Del):
                uses = [statement.value]
            else:
                uses = [var.name for var in statement.list_vars()]
            nodes.append(
                ScheduleNode(
                    offset,
                    type(statement).__name__,
                    defs,
                    set(uses),
                )
            )

        def add_edge(before, after):
            before.add_edge_to(after)

        # Chain multi-def and use-before-def names in original order.
        def_site = {}
        chained = set()
        for node in nodes:
            for name in node.defs:
                if name in def_site:
                    chained.add(name)
                def_site.setdefault(name, node)
        for node in nodes:
            for name in node.uses:
                site = def_site.get(name)
                if site is not None and site.index > node.index:
                    chained.add(name)
        for node in nodes:
            for name in node.uses:
                site = def_site.get(name)
                if site is not None and site.index < node.index:
                    add_edge(site, node)
        touching = {}
        for node in nodes:
            for name in node.defs | node.uses:
                if name in chained:
                    previous = touching.get(name)
                    if previous is not None:
                        add_edge(previous, node)
                    touching[name] = node

        # Del statements follow everything that referenced their name.
        references = {}
        for node in nodes:
            statement = movable[node.index]
            if isinstance(statement, ir.Del):
                name = statement.value
                for other_index in references.get(name, ()):
                    add_edge(nodes[other_index], node)
            for name in node.defs | node.uses:
                references.setdefault(name, []).append(node.index)

        # Memory chains keyed by (alias root, constant index).
        func_ir = self.state.func_ir
        last_store = {}
        last_unknown_store = {}
        loads = {}
        loads_unknown = {}
        root_keys = {}
        last_barrier = None
        memory_nodes = []

        for node in nodes:
            statement = movable[node.index]
            kind = None
            root = None
            index_key = None
            if isinstance(statement, (ir.SetItem, ir.StaticSetItem)):
                kind = "store"
                root, base_offset = roots.get(
                    statement.target.name, (None, None)
                )
                index_key = self._absolute_index(
                    self._constant_index(func_ir, statement),
                    base_offset,
                )
            elif isinstance(statement, _BARRIER_STATEMENT_TYPES):
                kind = "barrier"
            elif isinstance(statement, ir.Assign):
                op = _expr_op(statement)
                if op in (
                    "getitem",
                    "static_getitem",
                    "typed_getitem",
                ):
                    if isinstance(
                        typemap.get(statement.target.name),
                        types.Array,
                    ):
                        kind = None  # view creation: address math
                    else:
                        kind = "load"
                        root, base_offset = roots.get(
                            statement.value.value.name, (None, None)
                        )
                        index_key = self._absolute_index(
                            self._constant_index(
                                func_ir, statement.value
                            ),
                            base_offset,
                        )
                elif op == "call":
                    if self._call_is_pure(
                        func_ir, typemap, statement.value
                    ):
                        kind = None
                    else:
                        kind = "barrier"
                elif (
                    op in _PURE_EXPR_OPS
                    or op is None
                    or op == "phi"
                ):
                    kind = None
                else:
                    kind = "barrier"
            elif isinstance(statement, ir.Del):
                # An array del serialises against its whole root.
                if isinstance(
                    typemap.get(statement.value), types.Array
                ):
                    kind = "store"
                    root, _ = roots.get(
                        statement.value, (None, None)
                    )
                else:
                    kind = None
            else:
                kind = "barrier"

            if kind is not None and kind != "barrier" and root is None:
                kind = "barrier"
            if kind is not None:
                node.memory = (kind, repr(root), repr(index_key))
            if kind is None:
                continue
            if kind == "barrier":
                # Earlier nodes already order through the last barrier.
                for other in memory_nodes:
                    add_edge(other, node)
                last_barrier = node
                memory_nodes = [node]
                continue
            if last_barrier is not None:
                add_edge(last_barrier, node)
            memory_nodes.append(node)
            keys = root_keys.setdefault(root, set())
            unknown_store = last_unknown_store.get(root)
            if kind == "store" and index_key is None:
                for key in keys:
                    store = last_store.get((root, key))
                    if store is not None:
                        add_edge(store, node)
                    for load_index in loads.get((root, key), ()):
                        add_edge(nodes[load_index], node)
                    loads[(root, key)] = []
                if unknown_store is not None:
                    add_edge(unknown_store, node)
                for load_index in loads_unknown.get(root, ()):
                    add_edge(nodes[load_index], node)
                loads_unknown[root] = []
                last_unknown_store[root] = node
            elif kind == "store":
                keys.add(index_key)
                store = last_store.get((root, index_key))
                if store is not None:
                    add_edge(store, node)
                if unknown_store is not None:
                    add_edge(unknown_store, node)
                for load_index in loads.get((root, index_key), ()):
                    add_edge(nodes[load_index], node)
                for load_index in loads_unknown.get(root, ()):
                    add_edge(nodes[load_index], node)
                last_store[(root, index_key)] = node
                loads[(root, index_key)] = []
            elif index_key is None:
                for key in keys:
                    store = last_store.get((root, key))
                    if store is not None:
                        add_edge(store, node)
                if unknown_store is not None:
                    add_edge(unknown_store, node)
                loads_unknown.setdefault(root, []).append(node.index)
            else:
                keys.add(index_key)
                store = last_store.get((root, index_key))
                if store is not None:
                    add_edge(store, node)
                if unknown_store is not None:
                    add_edge(unknown_store, node)
                loads.setdefault((root, index_key), []).append(
                    node.index
                )

        if (
            dump_sink is not None
            and len(nodes) >= _DUMP_MIN_STATEMENTS
        ):
            dump_sink[str(label)] = {
                "pinned": pinned,
                "live_out": sorted(live_out),
                "nodes": [
                    {
                        "i": node.index,
                        "kind": node.statement_kind,
                        "op": _expr_op(movable[node.index]),
                        "defs": sorted(node.defs & scalar_names),
                        "all_defs": sorted(node.defs),
                        "uses": sorted(node.uses),
                        "mem": node.memory,
                        "succ": sorted(node.successors),
                    }
                    for node in nodes
                ],
            }
        if policy == "inject":
            order = injected_order
            if order is not None:
                expected = set(range(len(nodes)))
                if sorted(order) != sorted(expected):
                    raise ValueError(
                        f"injected order for block {label} is not a "
                        f"permutation of {len(nodes)} nodes"
                    )
                position = {
                    index: pos for pos, index in enumerate(order)
                }
                for node in nodes:
                    for successor in node.successors:
                        if (
                            position[successor]
                            < position[node.index]
                        ):
                            raise ValueError(
                                f"injected order for block {label} "
                                "violates dependency "
                                f"{node.index}->{successor}"
                            )
        elif policy == "source":
            order = None
        else:
            order = order_nodes(nodes, policy, live_out)
        peak_source = modeled_peak(
            nodes, range(len(nodes)), scalar_names, live_out
        )
        peak_scheduled = (
            modeled_peak(nodes, order, scalar_names, live_out)
            if order is not None
            else peak_source
        )
        block_stats = {
            "peak_source": peak_source,
            "peak_scheduled": peak_scheduled,
        }
        if order is None:
            if policy == "source":
                return (
                    list(range(len(statements))),
                    block_stats,
                )
            return None
        return (
            list(range(pinned))
            + [index + pinned for index in order],
            block_stats,
        )


__all__ = ["TypedBlockScheduler"]
