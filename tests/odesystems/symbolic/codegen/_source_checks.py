"""Shared source-level assertions for generated helper factories."""

import ast
from typing import Set, Tuple

# Modules the generated file's header imports; factory sources may
# reference them without defining them.
_MODULE_SCOPE_NAMES = {"cuda", "math"}


def factory_name_bindings(code: str) -> Tuple[Set[str], Set[str]]:
    """Return the names a factory source references and defines.

    Parameters
    ----------
    code
        Generated factory source, as returned by a ``generate_*``
        function.

    Returns
    -------
    tuple of set
        Referenced names and defined names. Call-position names
        (``exp``, ``selp``, ``precision``, ...) come from the
        generated file's header and count as defined, as do the
        header modules and every argument or assignment target in
        the source.
    """
    tree = ast.parse(code)
    defined: Set[str] = set(_MODULE_SCOPE_NAMES)
    referenced: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
            defined.update(arg.arg for arg in node.args.args)
        elif isinstance(node, ast.Call) and isinstance(
            node.func, ast.Name
        ):
            defined.add(node.func.id)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            else:
                referenced.add(node.id)
    return referenced, defined
