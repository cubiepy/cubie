"""Source-level tests for the remaining codegen branches.

Covers the Jacobian cache-key dict path and non-CSE JVP emission, the
CUDA printer's function-dispatch branches, the ``dxdt``/observables/
time-derivative non-CSE paths, and ``ODEFile.import_function``
re-initialising a stale cache file. Equation-set fixtures live in the
local conftest.
"""

import ast
import uuid

import sympy as sp

from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie.odesystems.symbolic.parsing import ParsedEquations
from cubie.odesystems.symbolic.codegen import print_cuda
from cubie.odesystems.symbolic.engine import call, sym
from cubie.odesystems.symbolic.codegen.jacobian import (
    generate_analytical_jvp,
    get_cache_key,
)
from cubie.odesystems.symbolic.codegen.dxdt import (
    generate_dxdt_lines,
    generate_observables_lines,
)
from cubie.odesystems.symbolic.codegen.time_derivative import (
    generate_time_derivative_fac_code,
)
from cubie.odesystems.symbolic.odefile import ODEFile


# ── jacobian cache key / non-CSE JVP ────────────────────────────── #

def test_get_cache_key_accepts_mapping():
    """A dict of equations hashes to the same key as its item tuple.

    The key includes ordering and ends with sorted ``derivative_names``
    tuple, which is empty unless the kwarg is supplied.
    """
    x, y = sp.symbols("x y")
    equations = {x: y, y: x}
    order = {x: 0, y: 1}
    key = get_cache_key(
        equations, order, order, cse=True, operation_ordering="kahn"
    )
    assert key[0] == tuple(equations.items())
    assert key[3] is True
    assert key[4] == "kahn"
    assert key[-1] == ()

    alternative_keys = {
        get_cache_key(
            equations,
            order,
            order,
            cse=True,
            operation_ordering=operation_ordering,
        )
        for operation_ordering in ("greedy", "dfs", "liveness_auto")
    }
    assert key not in alternative_keys
    assert len(alternative_keys) == 3

    named_key = get_cache_key(
        equations,
        order,
        order,
        cse=True,
        derivative_names={"b": "d_b", "a": "d_a"},
    )
    assert named_key[-1] == (("a", "d_a"), ("b", "d_b"))


def test_generate_analytical_jvp_without_cse():
    """Non-CSE JVP generation orders assignments via topological sort.

    Uses distinct state names (not the shared fixture's ``x, y``) so
    the jacobian module's memoisation cache cannot serve a CSE'd
    result computed by another test.
    """
    ib = IndexedBases.from_user_inputs(
        states=["p", "q"],
        parameters=["k"],
        constants=[],
        observables=[],
        drivers=[],
    )
    p = ib.states.symbol_map["p"]
    q = ib.states.symbol_map["q"]
    k = ib.parameters.symbol_map["k"]
    dp = ib.dxdt.symbol_map["dp"]
    dq = ib.dxdt.symbol_map["dq"]
    equations = ParsedEquations.from_equations(
        [(dp, k * p * q), (dq, k * q * q)], ib
    )
    jvp = generate_analytical_jvp(
        equations,
        input_order=ib.states.index_map,
        output_order=ib.dxdt.index_map,
        observables=ib.observable_symbols,
        cse=False,
    )
    # Two outputs produce two jvp[...] assignments.
    assert set(jvp.jvp_terms.keys()) == {0, 1}


# ── CUDA printer function dispatch ──────────────────────────────── #

def test_print_cuda_maps_known_function():
    """A CUDA-mapped applied function resolves to its ``math.*`` name.

    An undefined function whose name matches a ``CUDA_FUNCTIONS`` entry
    routes through the printer's function dispatch (SymPy-native
    functions bypass it), exercising the CUDA-known branch.
    """
    x = sp.Symbol("x")
    cuda_named = sp.Function("expm1")
    assert print_cuda(cuda_named(x)) == "math.expm1(x)"


def test_print_cuda_maps_native_abs():
    """SymPy's native ``Abs`` prints via the CUDA mapping.

    ``PythonCodePrinter`` would print the ``abs`` builtin; the
    ``_print_Abs`` override routes it to ``math.fabs`` per
    ``CUDA_FUNCTIONS``.
    """
    x = sp.Symbol("x")
    assert print_cuda(sp.Abs(x)) == "math.fabs(x)"


def test_print_cuda_passes_through_derivative_function():
    """A registered derivative helper prints inside a precision cast."""
    expression = call("d_helper", sym("x"))
    aliases = {"d_helper": "d_helper"}
    assert print_cuda(
        expression, function_aliases=aliases
    ) == "precision(d_helper(x))"


# ── dxdt / observables / time-derivative non-CSE ────────────────── #

def test_generate_dxdt_lines_without_cse(
    bare_nonlinear_equations, bare_indexed_bases
):
    """dxdt lines emit in topological order when CSE is disabled."""
    lines = generate_dxdt_lines(
        bare_nonlinear_equations, index_map=bare_indexed_bases, cse=False
    )
    assert any("out[0]" in line for line in lines)


def test_generate_observables_lines_without_cse(
    single_observable_equations, single_observable_indexed_bases
):
    """Observable lines emit in topological order when CSE is disabled."""
    lines = generate_observables_lines(
        single_observable_equations,
        single_observable_indexed_bases,
        cse=False,
    )
    assert any("observables[0]" in line for line in lines)


def test_generate_time_derivative_without_cse(
    bare_nonlinear_equations, bare_indexed_bases
):
    """Time-derivative factory parses when CSE is disabled."""
    code = generate_time_derivative_fac_code(
        bare_nonlinear_equations, bare_indexed_bases, cse=False
    )
    ast.parse(code)
    assert "def time_derivative_rhs(" in code


# ── ODEFile stale-file re-initialisation ────────────────────────── #

def test_import_function_reinitialises_stale_file(codegen_dir):
    """import_function rewrites the file when the stored hash is stale."""
    name = f"test_{uuid.uuid4().hex}"
    odf = ODEFile(name, 111)
    # Simulate a definition change: the stored hash no longer matches.
    odf.fn_hash = 222
    assert not odf.cached_file_valid(odf.fn_hash)

    code = (
        "\ndef sample_factory():\n"
        "    def sample():\n"
        "        return 1\n"
        "    return sample\n"
    )
    factory, was_cached = odf.import_function("sample_factory", code)
    assert was_cached is False
    assert factory()() == 1
    # The file was re-initialised under the new hash.
    assert odf.cached_file_valid(222)
