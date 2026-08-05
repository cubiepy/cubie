# no cover: start
"""Qt GUI for editing initial state values in a SymbolicODE.

Table-based editor for viewing and modifying initial state values
in a :class:`~cubie.odesystems.symbolic.SymbolicODE`.

Published Classes
-----------------
:class:`StatesEditor`
    Modal dialog for editing initial state values on a live
    ``SymbolicODE``.

    >>> editor = StatesEditor(ode)
    >>> editor.exec()

Module-Level Functions
----------------------
:func:`show_states_editor`
    Convenience wrapper to display a :class:`StatesEditor`.

    >>> show_states_editor(ode)

See Also
--------
:mod:`cubie.gui.constants_editor`
    Companion editor for constants and parameters.
:class:`~cubie.odesystems.symbolic.SymbolicODE`
    ODE system class consumed by the editor.
"""

from typing import TYPE_CHECKING, Optional

from qtpy.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QLabel, QHeaderView, QMessageBox,
    QWidget,
)
from qtpy.QtCore import Qt

from cubie.gui.constants_editor import FloatLineEdit

if TYPE_CHECKING:
    from cubie.odesystems.symbolic import SymbolicODE


class StatesEditor(QDialog):
    """Dialog for editing initial state values in a SymbolicODE.

    Parameters
    ----------
    ode
        The SymbolicODE instance to edit.
    parent
        Optional parent widget.

    Example
    -------
    >>> from cubie.odesystems.symbolic import load_bigmodel_file
    >>> ode = load_bigmodel_file("model.bigmodel")
    >>> editor = StatesEditor(ode)
    >>> editor.exec()  # Modal dialog
    """

    def __init__(
        self,
        ode: "SymbolicODE",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._ode = ode
        self._value_edits: dict[str, FloatLineEdit] = {}

        self._setup_ui()
        self._populate_table()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle("Initial States Editor")
        self.setMinimumSize(500, 400)

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Edit initial values for state variables.\n"
            "These values are used as the starting point for simulations."
        )
        layout.addWidget(info_label)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Name", "Initial Value", "Unit"])

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self._table.setColumnWidth(1, 150)
        self._table.setColumnWidth(2, 100)

        layout.addWidget(self._table)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._on_ok)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

    def _populate_table(self) -> None:
        """Populate the table with state variables."""
        states = self._ode.get_states_info()

        states.sort(key=lambda x: x['name'])

        self._table.setRowCount(len(states))

        for row, state in enumerate(states):
            name = state['name']
            value = state['value']
            unit = state['unit']

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            value_edit = FloatLineEdit(value)
            self._value_edits[name] = value_edit
            self._table.setCellWidget(row, 1, value_edit)

            unit_item = QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 2, unit_item)

    def _on_ok(self) -> None:
        """Apply all changes and close the dialog."""
        invalid_edits = [
            name for name, edit in self._value_edits.items()
            if not edit.isValid()
        ]
        if invalid_edits:
            QMessageBox.warning(
                self,
                "Invalid Values",
                f"The following entries have invalid values: "
                f"{', '.join(invalid_edits)}"
            )
            return

        errors = []

        for name, ve in self._value_edits.items():
            try:
                self._ode.set_initial_value(name, ve.value())
            except Exception as e:
                errors.append(f"{name}: {e}")

        if errors:
            QMessageBox.warning(
                self,
                "Errors Applying Changes",
                "Some changes could not be applied:\n"
                + "\n".join(errors)
            )
        else:
            self.accept()


def show_states_editor(
    ode: "SymbolicODE",
    blocking: bool = True,
) -> Optional[StatesEditor]:
    """Show the initial states editor dialog.

    Parameters
    ----------
    ode
        The SymbolicODE instance to edit.
    blocking
        If True, block until the dialog is closed. If False, return
        the dialog instance immediately.

    Returns
    -------
    StatesEditor or None
        The dialog instance if non-blocking, None if blocking.

    Examples
    --------
    >>> show_states_editor(ode)          # blocking
    >>> editor = show_states_editor(ode, blocking=False)
    """
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    editor = StatesEditor(ode)

    if blocking:
        editor.exec() if hasattr(editor, 'exec') else editor.exec_()
        if created_app:
            app.quit()
        return None
    else:
        editor.show()
        return editor
# no cover: end
