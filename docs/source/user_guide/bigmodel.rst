BigModel Models
===============

BigModel is an XML-based markup language for describing systems of
ordinary differential equations.  CuBIE can import BigModel files and
convert them into
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE` objects.

Loading a Bundled Model
-----------------------

CuBIE ships a handful of BigModel systems.  Load one by name:

.. code-block:: python

   import cubie as qb

   qb.available_bigmodels()
   system = qb.load_bigmodel("FL")

``load_bigmodel`` returns an unbuilt system; pass it to
:func:`~cubie.batchsolving.solver.solve_ivp` or
:class:`~cubie.batchsolving.solver.Solver` as usual.

Loading a File
--------------

.. code-block:: python

   import cubie as qb

   system = qb.load_bigmodel_file(
       "path/to/model.bigmodel",
       parameters=["g_Cr", "g_Az"],
       observables=["I_Cr", "I_Az"],
   )

Variables with differential equations become states.  Of the
remaining (algebraic) variables, those defined as plain numbers become
constants — or parameters, if you list them in ``parameters``.
Variables defined by expressions become anonymous auxiliaries unless
you list them in ``observables``, in which case their trajectories can
be saved.

Optional arguments:

``precision``
   ``np.float32`` (default) or ``np.float64``.

``name``
   Override the system name (defaults to the filename).

``fix_singularities`` (default ``True``)
   Rewrite removable singularities of the form ``U/(exp(U) - 1)``
   before parsing.  These otherwise destabilise Newton–Krylov solves,
   especially in ``float32``.

``voltage_variable``
   Name of the potential variable the singular terms are written in,
   used by the singularity fix.  Auto-detected if omitted.

``show_gui``
   Launch the interactive variable-classification editor.

BigModel parsing is handled by ``bigmodelmanip``, which ships vendored
inside CuBIE — no extra install is needed.

Known Caveats
-------------

- Only ODE-based BigModel models are supported: CuBIE extracts the
  differential equations as states, so a DAE or algebraic-only model
  has nothing to integrate.
- Some BigModel 2.0 features may not be fully handled by
  ``bigmodelmanip``.
- Large BigModel models (hundreds of states) may take noticeable time to
  parse and differentiate on first use; subsequent loads of the same
  file and settings come from an on-disk cache.
