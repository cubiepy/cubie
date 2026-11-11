Getting Started with Cubie
==========================

cubie is a fast numerical integration engine with a large set of integration algorithms. It works on Windows and Linux (MacOS untested) and is as close to a drop-in replacement of SciPy/MATLAB as seems possible. For advanced users, it allows customisation down to minute algorithm details and BYO-tableaux. For everyone else, you just give it a problem and a starting point and then watch it do a full night's work in a couple of seconds. 

Installation
------------
# This isn't correct really, the we should lead with cuda13 here and have a point underneath: If you already have a CUDA toolkit installed, omit the [cuda13] backend option, and just run pip install cubie.


.. code-block:: bash

   pip install cubie[cuda12]  # CUDA 12 toolkit
   # pip install cubie[cuda13]  # CUDA 13 toolkit

The extra installs the CUDA toolkit wheels alongside cubie's CUDA
backend (numba-cuda-mlir). If your machine already has a system CUDA
toolkit, ``pip install cubie`` installs the backend alone.

Basic Usage
-----------

To use Cubie, you need to:

1. Define a system of ODEs
2. Solve it from some starting states and parameters

.. code-block:: python
   :caption: Creating and solving a system of ODEs.
# We need to decide on the cubie-native form of inputs. I first made string recognition and am not sure if that offers any actual advantages - check this. Dict inputs i think are the native format, with arrays as faster or when programatically generated. all examples should be in native-form unless demonstrating another form.
   import cubie as qb
   import numpy as np

   def lotka_volterra(t, y, p):
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -p.c * y.y + p.d * y.x * y.y
       return [dx, dy]

   LV = qb.create_ODE_system(
           lotka_volterra,
           states={'x': 100, 'y': 100},
           parameters={'a': 0.01, 'b': 1, 'c': 0.01, 'd': 1},
           name="LotkaVolterra")

   solution = qb.solve_ivp(LV,
                           {'x': np.arange(100), 'y': np.arange(100)},
                           duration=1.0)

This runs 10,000 different IVPs of the Lotka-Volterra equations, starting from
every combination of x and y each ranging from 0 to 99.

# This is stupid. the top example is already a function, this point was supposed to be the function-form alternative to a string-form system, but it looks like it's really only showing the create ode system difference which seems irrelevant. making your own system just allows the setting of defaults, so it's not part of this.. I think we'd be best to settle on the canonical form, and call solve_ivp by default. reusing a solver can introduce the solver form and that should probably go in "solving", and tutorials should come after the user guides section.

Systems can also be defined as plain Python functions in the style of
``scipy.integrate.solve_ivp``, and passed straight to ``solve_ivp`` for a
one-call solve:

.. code-block:: python
   :caption: One-call solve from a SciPy-style function.

   def lotka_volterra(t, y, a, b, c, d):
       dx = a * y[0] - b * y[0] * y[1]
       dy = -c * y[1] + d * y[0] * y[1]
       return [dx, dy]

   solution = qb.solve_ivp(
       lotka_volterra,
       y0={'x': np.arange(100), 'y': np.arange(100)},
       parameters={'a': 0.01, 'b': 1, 'c': 0.01, 'd': 1},
       duration=1.0)

The one-call form rebuilds and recompiles the system every time it is
called. For repeated solves of the same system with different parameters
or initial values, use the two-step form: build the system once with
``create_ODE_system`` and keep a ``Solver``, so each subsequent solve
reuses the compiled GPU kernel:

.. code-block:: python
   :caption: Reusing a Solver for repeated batches.

   LV = qb.create_ODE_system(
           lotka_volterra,
           states={'x': 100, 'y': 100},
           parameters={'a': 0.01, 'b': 1, 'c': 0.01, 'd': 1})
   solver = qb.Solver(LV, algorithm="euler")

   for a_values in ([0.01, 0.02], [0.05, 0.1]):
       solution = solver.solve(
           {'x': np.arange(100), 'y': np.arange(100)},
           {'a': a_values},
           duration=1.0,
           grid_type="combinatorial")

Coming from another ecosystem? See
:doc:`Coming from SciPy <coming_from_scipy>` and
:doc:`Coming from MATLAB <coming_from_matlab>` for direct,
line-by-line ports.

# This stuff all looks like it actually belongson the top index page
Features
--------

* GPU-Accelerated forward simulation of non-stiff and stiff systems of ODEs
* Explicit, implicit, and semi-implicit Runge-Kutta methods
* Rosenbrock-W methods
* Adaptive time-stepping
* Interpolation of arbitrary time-domain inputs (forcing terms)
* Internal derivation of Jacobians using algorithmic symbolic differentiation with explicit chain-rule accumulation.
* Internal codegen of jacobian product evaluation and time-derivative functions.

Requirements
------------

* Python >= 3.11
* NumPy>=2.0
* numba-cuda-mlir
* attrs
* SymPy>= 1.13.0

Optional Dependencies
---------------------

* Pandas: For DataFrame output support
* Matplotlib: For plotting support. Only used to plot an interpolated driver function for sanity-checks (see
  :doc:`Drivers </user_guide/drivers>`), but generally useful for visualizing results.

GPU Requirements
~~~~~~~~~~~~~~~~

cubie requires an NVIDIA GPU with compute capability 6.0 or higher and an up-to-date NVIDIA driver.

.. toctree::
   :hidden:

   coming_from_scipy
   coming_from_matlab
