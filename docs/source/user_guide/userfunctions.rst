User functions and derivatives
==============================

You can call your own functions inside your equations. Pass them as
``user_functions`` when you build the system, as a dict that maps the name
you call in the equations to your function. This works for both Python
function and equation-string systems. If your function has the same name as
one of the maths functions cubie knows, such as ``exp``, cubie uses yours.

You can give cubie two kinds of function: a plain Python function, or a
CUDA device function.

Python functions
----------------

cubie calls a plain Python function with symbols instead of numbers, and
writes the expression it returns into your equations. Your function has to
follow the same rules as a system function, so it can only contain
arithmetic and the maths functions cubie knows. cubie can't evaluate a
function that branches on its argument, such as ``if x > 0:``; write that
one as a CUDA device function instead.

.. code-block:: python

   import numpy as np
   from cubie import create_ODE_system, solve_ivp

   def squared(x):
       return x ** 2

   system = create_ODE_system(
       "dx = -squared(x)",
       states={"x": 2.0},
       user_functions={"squared": squared},
   )

   result = solve_ivp(
       system,
       y0={"x": np.linspace(0.5, 2.0, 10000)},
       method="backwards_euler",
       duration=0.5,
       dt=0.01,
   )

Because cubie writes ``x ** 2`` straight into the equations, it can
differentiate it for an implicit algorithm such as ``backwards_euler``
without any help from you.

CUDA device functions
---------------------

Write a CUDA device function with ``@cuda.jit(device=True)``, importing
``cuda`` from ``numba_cuda_mlir``, the CUDA backend that installs with
cubie. cubie calls your device function directly from the GPU code.

.. code-block:: python

   import numpy as np
   from numba_cuda_mlir import cuda
   from cubie import create_ODE_system, solve_ivp

   @cuda.jit(device=True)
   def cubed(x):
       return x * x * x

   system = create_ODE_system(
       "dx = -cubed(x)",
       states={"x": 2.0},
       user_functions={"cubed": cubed},
   )

   result = solve_ivp(
       system,
       y0={"x": np.linspace(0.5, 2.0, 10000)},
       method="ode45",
       duration=0.5,
   )

Function derivatives
--------------------

If you pass a device function, you will also need to provide a function
that evaluates its derivative if you want to use an implicit algorithm. If
you're solving a DAE, you might need to provide a function for each
higher-order derivative required after tearing. cubie raises an error that
tells you which orders you need when you leave one out.

Pass the derivatives as ``user_function_derivatives``, a dict keyed by the
same names as ``user_functions``:

- Write each derivative as a CUDA device function too. The first
  derivative takes your function's arguments followed by the 0-based index
  of the argument to differentiate by.
- For higher orders, pass a list of functions, lowest order first. The
  order-``n`` derivative takes your function's arguments followed by ``n``
  argument indices.
- Give each derivative function its own name, different from every user
  function's name, because cubie finds it by its ``__name__`` in the code
  it generates.

.. code-block:: python

   import numpy as np
   from numba_cuda_mlir import cuda
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
       y0={"x": np.linspace(0.5, 2.0, 10000)},
       method="backwards_euler",
       duration=0.5,
       dt=0.01,
   )

Explicit algorithms such as ``ode45`` don't need derivatives, so you only
need ``user_function_derivatives`` when you use an implicit algorithm.
