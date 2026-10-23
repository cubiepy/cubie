import sympy as sp

import numpy as np
import pytest

from cubie.odesystems.symbolic.codegen.dxdt import (
    DXDT_TEMPLATE,
    generate_dxdt_fac_code,
    generate_observables_fac_code,
)
from cubie.odesystems.symbolic.parsing import (
    ParsedEquations,
    parse_system,
)
from cubie.odesystems.layout import SystemLayout
from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
from tests._utils import run_device_dxdt, run_device_observables


class TestDxdtTemplate:
    """Test the DXDT_TEMPLATE constant."""

    def test_dxdt_template_format(self):
        """Test that DXDT_TEMPLATE can be formatted correctly."""
        func_name = "test_factory"
        body = "    # test body"

        formatted = DXDT_TEMPLATE.format(
            func_name=func_name, const_lines="", body=body
        )

        assert func_name in formatted
        assert "test body" in formatted
        assert "@cuda.jit" in formatted
        assert "def dxdt(" in formatted
        assert "out, t" in formatted
        assert "return dxdt" in formatted

    def test_dxdt_template_structure(self):
        """Test the structure of DXDT_TEMPLATE."""
        lines = DXDT_TEMPLATE.splitlines()

        # Should contain proper function definition
        assert any("def {func_name}" in line for line in lines)
        assert any("@cuda.jit" in line for line in lines)
        assert any("def dxdt(" in line for line in lines)
        assert any("return dxdt" in line for line in lines)


class TestGenerateDxdtFacCode:
    """Test the generate_dxdt_fac_code function."""

    def test_simple_equations(self, simple_equations, layout):
        """Test code generation with simple equations."""
        code = generate_dxdt_fac_code(simple_equations, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code
        assert "@cuda.jit" in code
        assert "def dxdt(" in code
        assert "out[" in code
        assert "dxdt[" not in code
        assert "return dxdt" in code

    def test_custom_function_name(self, simple_equations, layout):
        """Test code generation with custom function name."""
        func_name = "custom_dxdt_factory"
        code = generate_dxdt_fac_code(
            simple_equations, layout, func_name
        )

        assert f"def {func_name}" in code
        assert "def dxdt_factory" not in code

    def test_complex_equations(self, complex_equations, layout):
        """Test code generation with complex equations containing auxiliaries.
        """
        code = generate_dxdt_fac_code(complex_equations, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code
        assert len(code) > 100  # Should be substantial code

    def test_linear_system(self, linear_system_equations, layout):
        """Test code generation with linear system."""
        code = generate_dxdt_fac_code(linear_system_equations, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_nonlinear_equations(self, nonlinear_equations, layout):
        """Test code generation with nonlinear equations."""
        code = generate_dxdt_fac_code(nonlinear_equations, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_empty_equations(self, layout):
        """Test code generation with empty equations list."""
        empty_equations = ParsedEquations.from_equations([], layout)
        code = generate_dxdt_fac_code(empty_equations, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_single_equation(self, layout):
        """Test code generation with single equation."""
        x = sp.Symbol("x", real=True)
        a = sp.Symbol("a", real=True)
        dx = sp.Symbol("dx", real=True)
        single_eq = ParsedEquations.from_equations(
            [(dx, a * x)],
            layout,
        )

        code = generate_dxdt_fac_code(single_eq, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_equations_with_numeric_constants(self, layout):
        """Test equations that reference numeric constants."""
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        dx = sp.Symbol("dx", real=True)
        dy = sp.Symbol("dy", real=True)
        equations = ParsedEquations.from_equations(
            [
                (dx, sp.pi * x + sp.E * y),
                (dy, sp.sqrt(2) * x - y),
            ],
            layout,
        )

        code = generate_dxdt_fac_code(equations, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_equations_with_functions(self, layout):
        """Test equations with mathematical functions."""
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        a = sp.Symbol("a", real=True)
        dx = sp.Symbol("dx", real=True)
        dy = sp.Symbol("dy", real=True)
        equations = ParsedEquations.from_equations(
            [
                (dx, sp.sin(a * x) + sp.cos(y)),
                (dy, sp.exp(-a * x) * y),
            ],
            layout,
        )

        code = generate_dxdt_fac_code(equations, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_code_contains_cuda_signatures(
        self, simple_equations, layout
    ):
        """Test that generated code contains proper CUDA signatures."""
        code = generate_dxdt_fac_code(simple_equations, layout)

        # Should contain proper parameter types
        assert "precision[::1]" in code
        assert "device=True" in code
        assert "inline=True" in code

    def test_code_parameter_order(self, simple_equations, layout):
        """Test that function parameters are in correct order."""
        code = generate_dxdt_fac_code(simple_equations, layout)

        # Find the function signature line
        lines = code.splitlines()
        func_def_line = None
        for line in lines:
            if "def dxdt(" in line:
                func_def_line = line
                break

        assert func_def_line is not None
        assert "state" in func_def_line
        assert "parameters" in func_def_line
        assert "driver" in func_def_line
        assert "observables" in func_def_line
        assert "out" in func_def_line
        assert "t" in func_def_line

    def test_fixed_parameters_never_named(self, layout):
        """Folded parameters emit no load binding in the factory."""
        x = sp.Symbol("x", real=True)
        c = sp.Symbol("c", real=True)
        dx = sp.Symbol("dx", real=True)
        equations = ParsedEquations.from_equations(
            [(dx, c * x)],
            layout,
        )
        code = generate_dxdt_fac_code(equations, layout)
        assert "_cubie_codegen_const_" not in code
        assert "constants[" not in code


class TestDxdtIntegration:
    """Integration tests for DXDT functionality."""

    def test_realistic_ode_system(self):
        """Test with a realistic ODE system (Lotka-Volterra)."""
        # Lotka-Volterra predator-prey model
        states = ["prey", "predator"]
        parameters = ["alpha", "beta", "gamma", "delta"]
        observables = []
        drivers = []

        layout = SystemLayout(
            states=sorted(states),
            parameters=sorted(parameters),
            observables=sorted(observables),
            drivers=sorted(drivers),
        )

        # Define equations symbolically
        prey, predator = sp.symbols("prey predator")
        alpha, beta, gamma, delta = sp.symbols("alpha beta gamma delta")

        dx_prey = sp.Symbol("dprey", real=True)
        dx_predator = sp.Symbol("dpredator", real=True)
        equations = ParsedEquations.from_equations(
            [
                (dx_prey, alpha * prey - beta * prey * predator),
                (
                    dx_predator,
                    gamma * prey * predator - delta * predator,
                ),
            ],
            layout,
        )

        code = generate_dxdt_fac_code(equations, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code
        assert len(code) > 200  # Should be substantial

    def test_with_auxiliary_variables(self):
        """Test system with auxiliary variables."""
        states = ["x", "y"]
        parameters = ["k1", "k2", "c1"]
        observables = ["total"]
        drivers = []

        layout = SystemLayout(
            states=sorted(states),
            parameters=sorted(parameters),
            observables=sorted(observables),
            drivers=sorted(drivers),
        )

        x, y = sp.symbols("x y")
        k1, k2, c1 = sp.symbols("k1 k2 c1")

        total = sp.Symbol("total", real=True)
        dx = sp.Symbol("dx", real=True)
        dy = sp.Symbol("dy", real=True)
        equations = ParsedEquations.from_equations(
            [
                (total, x + y),
                (dx, k1 * total - k2 * x + c1),
                (dy, -k1 * total + k2 * x),
            ],
            layout,
        )

        code = generate_dxdt_fac_code(equations, layout)

        assert isinstance(code, str)
        assert "def dxdt_factory" in code

    def test_dxdt_never_reads_observables_buffer(self):
        """Observable definitions are inlined into consuming
        dynamics, so the generated dxdt neither reads nor assigns
        the observables array (its values are stale during stage
        evaluation)."""

        dxdt_lines = [
            "a1 = x1**2",
            "a2 = x1 * x2",
            "a3 = p1 * x3 * x2",
            "o1 = a3**2 + x1",
            "o2 = x2 * a2",
            "dx1 = o2 * p2",
            "dx2 = o1 * a1 * p2",
            "dx3 = x3 * a1",
        ]

        specialised = parse_system(
            dxdt=dxdt_lines,
            states={"x1": 0.0, "x2": 0.0, "x3": 0.0},
            observables=["o1", "o2"],
            parameters={"p1": 0.0, "p2": 0.0},
            drivers=[],
            strict=False,
        ).specialise()
        layout = specialised.layout
        equations = specialised.equations

        code = generate_dxdt_fac_code(equations, layout)

        body = code.split("def dxdt(", 1)[1]
        assert "observables[" not in body


class TestGenerateObservablesFacCode:
    """Tests for observables-only factory generation."""

    def test_observables_factory_structure(self, layout):
        """Generated code should expose the observables factory."""

        x, y = sp.symbols("x y", real=True)
        a = sp.symbols("a", real=True)
        obs = sp.Symbol("obs1", real=True)
        dx = sp.Symbol("dx", real=True)
        equations = ParsedEquations.from_equations(
            [
                (obs, x + a),
                (dx, y + x),
            ],
            layout,
        )

        code = generate_observables_fac_code(equations, layout)

        assert "def observables" in code
        assert "def get_observables" in code
        assert "observables[" in code
        assert "out[" not in code
        assert "observables, t" in code

    def test_observables_factory_includes_cse_dependencies(
        self, layout
    ):
        """CSE expressions shared with dxdt should still be emitted."""

        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        repeated = (x + y) ** 2
        obs = sp.Symbol("obs1", real=True)
        dx = sp.Symbol("dx", real=True)
        equations = ParsedEquations.from_equations(
            [
                (obs, repeated),
                (dx, repeated + x),
            ],
            layout,
        )

        code = generate_observables_fac_code(equations, layout)

        # Power replacement optimization converts **2 to explicit
        # multiplication
        assert ("(state[0] + state[1])*(state[0] + state[1])" in code)
        assert "out[" not in code


class TestObservablesDeviceParity:
    """Ensure observables helpers mirror dx/dt side effects."""

    @pytest.mark.parametrize(
        "state_values,param_values,driver_value",
        [
            ([1.2, -0.4], [0.3, 0.7], 0.25),
            ([-0.6, 1.5], [0.9, -0.2], -0.4),
        ],
    )
    def test_dxdt_preserves_observables(
        self,
        observables_kernel_system,
        state_values,
        param_values,
        driver_value,
        precision,
        tolerance,
    ):
        """dxdt should consume, not overwrite, the observables buffer."""

        system = observables_kernel_system
        dxdt_dev = system.dxdt_fn
        observables_fn = system.observables_fn

        state = np.array(state_values, dtype=precision)
        parameters = np.array(param_values, dtype=precision)
        drivers = np.array([driver_value], dtype=precision)

        state_kernel = state.copy()
        parameters_kernel = parameters.copy()
        drivers_kernel = drivers.copy()

        observables_buffer = np.full(
            system.num_observables, precision(-1.0), dtype=precision
        )
        out = np.zeros(system.num_states, dtype=precision)

        run_device_observables(
            observables_fn,
            state_kernel,
            parameters_kernel,
            drivers_kernel,
            observables_buffer,
            precision(0.0),
        )
        run_device_dxdt(
            dxdt_dev,
            state_kernel,
            parameters_kernel,
            drivers_kernel,
            observables_buffer,
            out,
            precision(0.0),
        )

        state_idx = {
            name: row for row, name in enumerate(system.states.names)
        }
        param_idx = {
            name: row for row, name in enumerate(system.parameters.names)
        }
        driver_names = system.layout.driver_names
        drive_idx = driver_names.index("drive")
        const_value = precision(system.parameters.values_dict["c0"])

        x_val = state_kernel[state_idx["x"]]
        y_val = state_kernel[state_idx["y"]]
        alpha = parameters_kernel[param_idx["alpha"]]
        beta = parameters_kernel[param_idx["beta"]]
        drive_val = drivers_kernel[drive_idx]

        expected_rate = alpha * x_val + const_value
        expected_total = expected_rate + beta * y_val + drive_val
        expected_observables = np.array(
            [expected_rate, expected_total], dtype=precision
        )
        expected_dx = expected_total - y_val + alpha * drive_val
        expected_dy = expected_rate * x_val + const_value

        np.testing.assert_allclose(
            observables_buffer,
            expected_observables,
            rtol=tolerance.rel_tight,
            atol=tolerance.abs_tight,
        )
        np.testing.assert_allclose(
            out,
            np.array([expected_dx, expected_dy], dtype=precision),
            rtol=tolerance.rel_tight,
            atol=tolerance.abs_tight,
        )


def test_recompile_updates_fixed_parameters(precision, tolerance):
    """A new default for a fixed parameter updates the device function."""

    system = SymbolicODE.create(
        dxdt=["dx = c * x"],
        states={"x": precision(1.0)},
        parameters={"c": precision(2.0)},
        drivers=[],
        observables=[],
        precision=precision,
        strict=True,
        name="recompile_fixed_parameters",
    )

    def run_dxdt(current_system: SymbolicODE) -> float:
        dxdt_func = current_system.dxdt_fn

        state = np.array([precision(1.0)], dtype=precision)
        parameters = np.zeros(current_system.num_parameters, dtype=precision)
        drivers = np.zeros(current_system.num_drivers, dtype=precision)
        observables = np.zeros(
            current_system.num_observables, dtype=precision
        )
        out = np.zeros(current_system.num_states, dtype=precision)
        run_device_dxdt(
            dxdt_func,
            state,
            parameters,
            drivers,
            observables,
            out,
            precision(0.0),
        )
        return float(out[0])

    res1 = run_dxdt(system)
    system.set_default_parameters({"c": precision(3.0)})
    res2 = run_dxdt(system)

    assert res1 == pytest.approx(
        2.0,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
    assert res2 == pytest.approx(
        3.0,
        rel=tolerance.rel_tight,
        abs=tolerance.abs_tight,
    )
