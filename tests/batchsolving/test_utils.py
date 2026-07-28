"""Tests for cubie.batchsolving._utils."""

from __future__ import annotations

from cubie.batchsolving._utils import name_and_compile_kernel


# ── name_and_compile_kernel ─────────────────────────────── #


def test_name_and_compile_kernel_renames_last_qualname_segment():
    """Helper renames the function and its innermost qualname part."""

    def outer():
        def placeholder_kernel():
            return None

        return placeholder_kernel

    kernel_function = outer()
    name_and_compile_kernel(kernel_function, "renamed_kernel", {})

    assert kernel_function.__name__ == "renamed_kernel"
    assert kernel_function.__qualname__ == (
        "test_name_and_compile_kernel_renames_last_qualname_segment."
        "<locals>.outer.<locals>.renamed_kernel"
    )
