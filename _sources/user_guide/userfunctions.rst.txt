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
- The derivative must itself be a CUDA device function
  (``@cuda.jit(device=True)``) — it is called from generated device code
  when an implicit method builds Jacobian terms.

Example:

- Define device function and its derivative name:

  .. code-block:: python

     from numba import cuda

     @cuda.jit(device=True)
     def myfunc(a, b):
         return a * b

     @cuda.jit(device=True)
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

End-to-end example
------------------

Device functions are supported end-to-end: systems whose equations call
them can be built and solved directly. The device callables are made
available to the generated module automatically, in both the string and
Python-function input forms.

.. code-block:: python

   import numpy as np
   from numba import cuda

   from cubie import create_ODE_system, solve_ivp

   @cuda.jit(device=True)
   def cubed(x):
       return x * x * x

   @cuda.jit(device=True)
   def d_cubed(x, index):
       return 3.0 * x * x

   system = create_ODE_system(
       "dx = -cubed(x)",
       states={"x": 2.0},
       user_functions={"cubed": cubed},
       user_function_derivatives={"cubed": d_cubed},
   )

   result = solve_ivp(
       system,
       y0={"x": 2.0},
       method="backwards_euler",
       duration=0.5,
       dt=0.01,
       save_every=0.05,
   )
   print(result.time_domain_array[:, 0, 0])

The derivative mapping is only needed for implicit methods, which build
Jacobian terms; explicit methods such as ``euler`` need only
``user_functions``.

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
