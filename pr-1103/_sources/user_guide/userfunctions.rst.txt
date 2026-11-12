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

Pass a mapping of names to callables as ``user_functions`` when you build the
system with :func:`~cubie.create_ODE_system`. If your function name collides
with a SymPy built-in (e.g., exp), the user function takes precedence.

Example (Python function):

- Define a simple function:

  .. code-block:: python

     def ex_squared(x):
         return x**2

- Use it in equations:

  .. code-block:: python

     system = create_ODE_system(
         dxdt=["dx = ex_squared(a)", "y = x"],
         user_functions={"ex_squared": ex_squared},
     )

- If Cubie can inline the function (it is simple and uses SymPy-compatible
  logic), it writes a*a into the generated code. Otherwise it writes a call to
  ex_squared(a), and a call to the derivative of ex_squared in the Jacobian
  terms, so you need to provide that function in user_function_derivatives.

Device functions and derivatives
--------------------------------

CUDA device functions are detected automatically if they are created with
cuda.jit(..., device=True), with ``cuda`` from ``cubie._cudasim_extensions``. For differentiation, also provide a derivative
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

- Build the system with both maps:

  .. code-block:: python

     system = create_ODE_system(
         dxdt=["dx = myfunc(x, y)", "dy = x"],
         states=["x", "y"],
         user_functions={"myfunc": myfunc},
         user_function_derivatives={"myfunc": myfunc_grad},
     )

- When you solve with an implicit method, Cubie writes calls to
  myfunc_grad(..., argindex) into the Jacobian terms.

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
