Cubie documentation
===================

`cubie` provides GPU-accelerated tools for integrating large batches of initial
value problems. The source code is available on `Github
<https://github.com/ccam80/cubie>`__. If you're just getting started, check out
:doc:`Getting Started <getting_started/index>` and the rest of the
:doc:`User Manual <user_guide/index>`. If you need specific details about a
function or class, the :doc:`Reference Manual <API_reference/index>` is the
place to go.

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
* NumPy >= 2.0
* cubie-numba-cuda-mlir
* attrs
* SymPy >= 1.13.0
* lxml, networkx, Pint >= 0.24 and rdflib, for CellML import

pip installs all of these with cubie.

Optional Dependencies
---------------------

* Pandas: For DataFrame output support
* Matplotlib: For plotting support. Only used to plot an interpolated driver function for sanity-checks (see
  :doc:`Drivers </user_guide/drivers>`), but generally useful for visualizing results.
* qtpy with a Qt binding: For the CellML variable editor (``show_gui``).

GPU Requirements
~~~~~~~~~~~~~~~~

cubie requires an NVIDIA GPU with compute capability 6.0 or higher and an up-to-date NVIDIA driver.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started/index
   user_guide/index
   tutorials/index
   troubleshooting
   theory/index
   developer_guide/index
   API_reference/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
