Getting Started with Cubie
==========================

cubie is a Python library designed to provide an easy entry point for users to
perform large-scale batch integration of ivps: from many initial values or with
many different parameter sets, or both, cubie don't care.

Installation
------------

Install cubie using pip, selecting the toolkit that matches your setup:

.. code-block:: bash

   pip install cubie[mlir-cuda12]  # CUDA 12 toolkit
   # pip install cubie[mlir-cuda13]  # CUDA 13 toolkit

The extra is required: it installs cubie's CUDA backend
(numba-cuda-mlir) alongside the matching toolkit wheels, and a bare
``pip install cubie`` has no backend to compile with. If your machine
already has a system CUDA toolkit, ``pip install cubie[mlir]``
installs the backend alone.

The previous default backend (numba-cuda) is deprecated but still
available through the ``cuda12``/``cuda13`` extras (bare ``cuda`` for
a system toolkit). MLIR is faster; try numba-cuda if you run into
unexpected errors, or if you need Python 3.10 or the CUDA simulator,
which only exist on numba-cuda.

Basic Usage
-----------

To use Cubie, you need to:

1. Define a system of ODEs
2. Solve a batch of IVPs

.. code-block:: python
   :caption: Creating and solving a system of ODEs.

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
:doc:`Coming from SciPy <user_guide/coming_from_scipy>` and
:doc:`Coming from MATLAB <user_guide/coming_from_matlab>` for direct,
line-by-line ports.

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

* Python >= 3.11 (>= 3.10 with the deprecated numba-cuda backend)
* NumPy>=2.0
* Numba
* numba-cuda-mlir (or the deprecated numba-cuda)
* attrs
* SymPy>= 1.13.0

Optional Dependencies
---------------------

* Pandas: For DataFrame output support
* Matplotlib: For plotting support. Only used to plot an interpolated driver function for sanity-checks (see
  :doc:`Drivers <user_guide/drivers>`), but generally useful for visualizing results.

GPU Requirements
~~~~~~~~~~~~~~~~

cubie requires an NVIDIA GPU with compute capability 6.0 or higher (see nvidia's
documentation for details) and an up-to-date NVIDIA driver.
