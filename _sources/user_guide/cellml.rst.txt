CellML Models
=============

`CellML <https://www.cellml.org/>`__ is an XML-based markup language for
describing mathematical models of biological systems.  CuBIE can import
CellML files and convert them into
:class:`~cubie.odesystems.symbolic.symbolicODE.SymbolicODE` objects.

Loading a CellML Model
-----------------------

.. code-block:: python

   import cubie as qb

   system = qb.load_cellml_model(
       "path/to/model.cellml",
       parameters=["g_Na", "g_K"],
       observables=["I_Na", "I_K"],
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
   Rewrite removable singularities of the Goldman–Hodgkin–Katz form
   ``U/(exp(U) - 1)`` before parsing.  These otherwise destabilise
   Newton–Krylov solves, especially in ``float32``.

``voltage_variable``
   Name of the membrane-voltage variable, used by the singularity
   fix.  Auto-detected if omitted.

``show_gui``
   Launch the interactive variable-classification editor.

CellML parsing is handled by ``cellmlmanip``, which ships vendored
inside CuBIE — no extra install is needed.

Known Caveats
-------------

- Only ODE-based CellML models are supported: CuBIE extracts the
  differential equations as states, so a DAE or algebraic-only model
  has nothing to integrate.
- Some CellML 2.0 features may not be fully handled by ``cellmlmanip``.
- Large CellML models (hundreds of states) may take noticeable time to
  parse and differentiate on first use; subsequent loads of the same
  file and settings come from an on-disk cache.
