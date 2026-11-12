User functions and derivatives
==============================

Cubie lets you call your own functions inside dx/dt equations. There are two
main cases:

- Pure Python callables: These are treated like ordinary functions. When
  possible, Cubie inlines them symbolically.
- CUDA device functions: These are detected and treated as opaque calls in
  generated code. For differentiation (Jacobian/JVP/VJP), provide a
  corresponding derivative function.

Providing functions
-------------------

Pass a mapping of names to callables via user_functions to parse_input. If your
function name collides with a SymPy built-in (e.g., exp), the user function
takes precedence.

Example (Python function):

- Define a simple function:

  .. code-block:: python

     def ex_squared(x):
         return x**2

- Use it in equations:

  .. code-block:: python

     index_map, symbols, funcs, eqs, fn_hash = parse_input(
         dxdt=["dx = ex_squared(a)", "y = x"],
         user_functions={"ex_squared": ex_squared}
     )

- print_cuda_multiple(eqs, symbols) will emit a*a in code if it can inline the
  function (i.e. it's simple and uses Sympy-compatible logic, otherwise it will
  call ex_squared(a) in code. If it calls ex_squared(a), it will also call
  ex_squared_grad(a, 0) in Jacobian terms, so you need to provide that function
  in user_function_derivatives.

Device functions and derivatives
--------------------------------

CUDA device functions are detected automatically if they are created with
numba.cuda.jit(..., device=True). For differentiation, also provide a derivative
function in user_function_derivatives with the same key as the original function
name.

- The derivative callable signature must be: d_userfunc(funcargs..., argindex)
  where argindex is 0-based index of the argument with respect to which the
  derivative is taken.
- The derivative callable’s __name__ is used in generated code, so choose a
  descriptive name (e.g., myfunc_grad).

Example:

- Define device function and its derivative name:

  .. code-block:: python

     from numba import cuda

     @cuda.jit(device=True)
     def myfunc(a, b):
         return a * b

     # This can be device or pure Python; codegen only needs the name
     def myfunc_grad(a, b, index):
         if index == 0:
             return b
         elif index == 1:
             return a
         return 0

- Parse equations with both maps:

  .. code-block:: python

     index_map, symbols, funcs, eqs, fn_hash = parse_input(
         dxdt=["dx = myfunc(x, y)", "dy = x"],
         states=["x", "y"], parameters=[], constants=[], observables=[],
         user_functions={"myfunc": myfunc},
         user_function_derivatives={"myfunc": myfunc_grad}
     )

- Generate JVP code:

  .. code-block:: python

     from cubie.odesystems.symbolic.codegen.linear_operators import (
         generate_cached_jvp_code,
     )

     code = generate_cached_jvp_code(eqs, index_map)
     # The code will contain calls to myfunc_grad(..., argindex) in the
     # Jacobian terms.

  (This is internal plumbing — when you solve through
  :func:`~cubie.solve_ivp` or :class:`~cubie.Solver`, the JVP code is
  generated for you.)

Name collisions with SymPy
--------------------------

If your user function has the same name as a SymPy function, Cubie ensures your
function wins. Internally it renames your function to a safe symbolic token
during parsing and maps it back to your original name when printing code.

Tips
----

- If you don’t provide a derivative for a device function, auto-generated
  jacobians will not work.
- Pure Python user functions that can be evaluated on SymPy symbols may be
  inlined symbolically; otherwise they are called by name in code, and you'll
  need to provide a derivative function for differentiation.
