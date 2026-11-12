Getting Started with Cubie
==========================

cubie is a fast numerical integration engine with a large set of
integration algorithms. It works on Windows and Linux (MacOS untested) and
is as close to a drop-in replacement of SciPy/MATLAB as seems possible. For
advanced users, it allows customisation down to minute algorithm details
and BYO-tableaux. For everyone else, you just give it a problem and a
starting point and then watch it do a full night's work in a couple of
seconds.

Installation
------------

Install cubie with pip:

.. code-block:: bash

   pip install cubie[cuda13]

With the ``[cuda13]`` extra, pip installs a CUDA 13 toolkit along with
cubie. If you
already have a CUDA toolkit installed, leave the extra out and run
``pip install cubie``. Use ``cubie[cuda12]`` instead to install a CUDA 12
toolkit.

Basic Usage
-----------

To use Cubie, you need to:

1. Define a system of ODEs
2. Solve it from some starting states and parameters

.. code-block:: python
   :caption: Defining and solving a system of ODEs.

   import cubie as qb
   import numpy as np

   def lotka_volterra(t, y, p):
       dx = p.a * y.x - p.b * y.x * y.y
       dy = -p.c * y.y + p.d * y.x * y.y
       return [dx, dy]

   result = qb.solve_ivp(
       lotka_volterra,
       y0={"x": np.linspace(1.0, 50.0, 100), "y": np.linspace(1.0, 50.0, 100)},
       parameters={"a": 0.1, "b": 0.02, "c": 0.3, "d": 0.01},
       method="ode45",
       duration=100.0,
   )

This example solves 10,000 different IVPs of the Lotka-Volterra equations, one for
every combination of the 100 starting values of ``x`` and the 100 starting
values of ``y``. 

Coming from another ecosystem? See
:doc:`Coming from SciPy <coming_from_scipy>` and
:doc:`Coming from MATLAB <coming_from_matlab>` for direct,
line-by-line ports.

.. toctree::
   :hidden:

   coming_from_scipy
   coming_from_matlab
