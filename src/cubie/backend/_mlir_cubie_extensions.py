"""The ``unroll_if`` AST pass: closure flags become ``cuda.unroll`` hints."""

import ast

from numba_cuda_mlir import ast_transforms as _ast_transforms
from numba_cuda_mlir import cuda as _ncm_cuda
from numba_cuda_mlir.ast_transforms import ASTTransformPass
from numba_cuda_mlir.ast_transforms.common import get_function_context


def _is_consteval_call(node):
    """Return whether ``node`` is a one-argument consteval call."""
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return False
    func = node.func
    name = (
        func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    )
    return name in ("consteval", "literally")


def _is_unroll_if_call(node):
    """Return whether ``node`` is a two- or three-argument unroll_if call."""
    if not isinstance(node, ast.Call) or len(node.args) not in (2, 3):
        return False
    func = node.func
    name = (
        func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    )
    return name == "unroll_if"


_UNROLL_HINT_NAME = "_cubie_unroll"
"""Global bound to the wheel's ``cuda.unroll`` in rewritten functions."""


def _unroll_hint():
    """Return the wheel's ``cuda.unroll`` loop-hint function."""
    unroll = getattr(_ncm_cuda, "unroll", None)
    if unroll is None:
        raise RuntimeError(
            "unroll_if loops need cuda.unroll (the llvm.loop.unroll "
            "hint) from cubie-numba-cuda-mlir; the installed wheel "
            "lacks it"
        )
    return unroll


class _StripConstevalOf(ast.NodeTransformer):
    """Replace ``consteval(expr)`` by ``expr`` when ``expr`` reads a name."""

    def __init__(self, names):
        self.names = names

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if _is_consteval_call(node):
            used = {
                child.id
                for child in ast.walk(node.args[0])
                if isinstance(child, ast.Name)
            }
            if used & self.names:
                return node.args[0]
        return node


class _UnrollIf(ast.NodeTransformer):
    """Rewrite ``unroll_if`` loops to unroll hints or plain loops."""

    def __init__(self, func):
        self.func = func
        self._env = None
        self.plain_vars = set()
        self.modified = False

    @property
    def env(self):
        if self._env is None:
            self._env = get_function_context(self.func)
        return self._env

    def _closure_value(self, node, role):
        """Resolve a closure name or ``name.attr`` node to its value."""
        func_name = getattr(self.func, "__qualname__", repr(self.func))
        if isinstance(node, ast.Attribute) and isinstance(
            node.value, ast.Name
        ):
            name, attr = node.value.id, node.attr
        elif isinstance(node, ast.Name):
            name, attr = node.id, None
        else:
            raise TypeError(
                f"unroll_if {role} in {func_name} must be a closure "
                "name or an attribute of one"
            )
        if name not in self.env:
            raise NameError(
                f"unroll_if {role} {name!r} is not in the closure or "
                f"globals of {func_name}"
            )
        value = self.env[name]
        return getattr(value, attr) if attr is not None else value

    def _hint(self, args):
        """Return ``(unroll, count)`` for one unroll_if argument list."""
        # cuda_simsafe imports after the backend patches.
        from cubie.cuda_simsafe import unroll_flag_converter

        unroll, count = unroll_flag_converter(
            self._closure_value(args[1], "flag")
        )
        if len(args) == 3:
            count_node = args[2]
            if isinstance(count_node, ast.Constant):
                explicit = count_node.value
            else:
                explicit = self._closure_value(count_node, "count")
            if unroll and explicit is not None:
                _, count = unroll_flag_converter((True, explicit))
        return unroll, count

    def visit_For(self, node):
        rewritten = False
        if _is_unroll_if_call(node.iter):
            args = node.iter.args
            unroll, count = self._hint(args)
            if not unroll:
                # No hint: the backend decides.
                node.iter = args[0]
            else:
                hint_args = [args[0]]
                if count is not None:
                    hint_args.append(ast.Constant(value=count))
                node.iter = ast.copy_location(
                    ast.Call(
                        func=ast.Name(id=_UNROLL_HINT_NAME, ctx=ast.Load()),
                        args=hint_args,
                        keywords=[],
                    ),
                    node.iter,
                )
            if isinstance(node.target, ast.Name):
                self.plain_vars.add(node.target.id)
            self.modified = True
            rewritten = True
        node = self.generic_visit(node)
        if rewritten and self.plain_vars:
            node = _StripConstevalOf(self.plain_vars).visit(node)
        return node


class _UnrollIfPass(ASTTransformPass):
    """Pipeline pass resolving unroll_if loops ahead of consteval."""

    @property
    def name(self) -> str:
        return "UnrollIf"

    def transform(self, tree, context):
        rewriter = _UnrollIf(context.func)
        tree = rewriter.visit(tree)
        ast.fix_missing_locations(tree)
        if rewriter.modified:
            context.stored_values[_UNROLL_HINT_NAME] = _unroll_hint()
        return tree, rewriter.modified


def _patch_unroll_if() -> None:
    """Run the unroll_if pass ahead of the wheel's consteval pass."""
    stock_create_default_pipeline = _ast_transforms.create_default_pipeline

    def create_default_pipeline():
        pipeline = stock_create_default_pipeline()
        pipeline._passes.insert(0, _UnrollIfPass())
        return pipeline

    _ast_transforms.create_default_pipeline = create_default_pipeline


_patch_unroll_if()
