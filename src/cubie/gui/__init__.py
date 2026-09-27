# no cover: start
"""Qt-based GUI utilities for CuBIE ODE systems.

Graphical editors for managing parameters and initial
state values in :class:`~cubie.odesystems.symbolic.SymbolicODE`
systems.  The editors use the public API of ``SymbolicODE`` and are
loosely coupled from the rest of the library.

Requires one of: PyQt6, PyQt5, PySide6, or PySide2 (via ``qtpy``).

Published Classes
-----------------
:class:`ParametersEditor`
    Dialog for viewing and editing parameter values.

    >>> from cubie.gui import ParametersEditor
    >>> editor = ParametersEditor(ode)
    >>> editor.exec()

:class:`StatesEditor`
    Dialog for viewing and editing initial state values.

    >>> from cubie.gui import StatesEditor
    >>> editor = StatesEditor(ode)
    >>> editor.exec()

See Also
--------
:mod:`cubie.gui.parameters_editor`
    Parameters editor and pre-parse editor.
:mod:`cubie.gui.states_editor`
    Initial-states editor.
:class:`~cubie.odesystems.symbolic.SymbolicODE`
    ODE system class consumed by the editors.
"""

from cubie.gui.parameters_editor import ParametersEditor
from cubie.gui.states_editor import StatesEditor

__all__ = ["ParametersEditor", "StatesEditor"]
# no cover: end
