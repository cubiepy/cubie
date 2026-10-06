# no cover: start
"""Qt GUI for editing parameter values in a SymbolicODE.

Table-based editors for viewing and modifying the parameters of a
:class:`~cubie.odesystems.symbolic.SymbolicODE`.

Published Classes
-----------------
:class:`FloatLineEdit`
    Line edit widget with forgiving float validation.

    >>> edit = FloatLineEdit(1.5e-3)
    >>> edit.value()
    0.0015

:class:`ParametersEditor`
    Modal dialog for editing parameter values on a live
    ``SymbolicODE``.

    >>> editor = ParametersEditor(ode)
    >>> editor.exec()

:class:`PreParseEditor`
    Modal dialog for editing parameter and initial values before
    parsing (operates on raw dictionaries).

Module-Level Functions
----------------------
:func:`edit_pre_parse_dicts`
    Show a :class:`PreParseEditor` and return the modified dicts.

    >>> params, inits = edit_pre_parse_dicts({"k": 1.0}, {"x": 0.0})

:func:`show_parameters_editor`
    Convenience wrapper to display a :class:`ParametersEditor`.

    >>> show_parameters_editor(ode)

See Also
--------
:mod:`cubie.gui.states_editor`
    Companion editor for initial state values.
:class:`~cubie.odesystems.symbolic.SymbolicODE`
    ODE system class consumed by the editors.
"""

from typing import TYPE_CHECKING, Optional

from qtpy.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QLineEdit, QPushButton, QLabel, QHeaderView,
    QMessageBox, QWidget,
)
from qtpy.QtCore import Qt

if TYPE_CHECKING:
    from cubie.odesystems.symbolic import SymbolicODE


class FloatLineEdit(QLineEdit):
    """Line edit with forgiving float validation.

    Accepts standard floats, scientific notation (``1e-5``), and
    Fortran-style ``d`` exponents (``1.5d3``).  Invalid input is
    highlighted with a red background.

    Parameters
    ----------
    value
        Initial numeric value.
    parent
        Optional parent widget.
    """

    def __init__(self, value: float = 0.0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._value = value
        self.setText(self._format_value(value))
        self.editingFinished.connect(self._on_editing_finished)
        self._valid = True

    def _format_value(self, value: float) -> str:
        """Format value for display."""
        if abs(value) < 1e-4 or abs(value) >= 1e6:
            return f"{value:.6e}"
        return f"{value:.6g}"

    def _on_editing_finished(self) -> None:
        """Validate and update value when editing finishes."""
        text = self.text().strip()
        try:
            self._value = self._parse_float(text)
            self._valid = True
            self.setStyleSheet("")
            self.setText(self._format_value(self._value))
        except ValueError:
            self._valid = False
            self.setStyleSheet("background-color: #ffcccc;")

    def _parse_float(self, text: str) -> float:
        """Parse text to float with forgiving validation.

        Accepts:
        - Standard floats: 1.5, -2.3, .5
        - Scientific: 1e-5, 1.5E+10
        - Leading/trailing whitespace
        """
        text = text.strip().lower()
        if not text:
            return 0.0

        text = text.replace('d', 'e')

        return float(text)

    def value(self) -> float:
        """Return the current value."""
        return self._value

    def setValue(self, value: float) -> None:
        """Set the value."""
        self._value = value
        self.setText(self._format_value(value))
        self._valid = True
        self.setStyleSheet("")

    def isValid(self) -> bool:
        """Return whether the current text is valid."""
        return self._valid


def _value_table(parent_layout: QVBoxLayout, value_header: str):
    """Add a Name/value/Unit table to ``parent_layout``."""
    table = QTableWidget()
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels(["Name", value_header, "Unit"])
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.Stretch)
    header.setSectionResizeMode(1, QHeaderView.Fixed)
    header.setSectionResizeMode(2, QHeaderView.Fixed)
    table.setColumnWidth(1, 150)
    table.setColumnWidth(2, 100)
    parent_layout.addWidget(table)
    return table


def _fill_value_table(
    table: QTableWidget,
    values: dict[str, float],
    units: dict[str, str],
) -> dict[str, FloatLineEdit]:
    """Fill ``table`` with sorted rows and return the value editors."""
    edits = {}
    items = sorted(values.items())
    table.setRowCount(len(items))
    for row, (name, value) in enumerate(items):
        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
        table.setItem(row, 0, name_item)

        edit = FloatLineEdit(value)
        edits[name] = edit
        table.setCellWidget(row, 1, edit)

        unit_item = QTableWidgetItem(units.get(name, ""))
        unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
        table.setItem(row, 2, unit_item)
    return edits


def _invalid_names(*edit_maps: dict[str, FloatLineEdit]) -> list[str]:
    """Return the names whose editor holds an invalid value."""
    return [
        name
        for edits in edit_maps
        for name, edit in edits.items()
        if not edit.isValid()
    ]


class ParametersEditor(QDialog):
    """Dialog for editing parameter values in a SymbolicODE.

    Parameters
    ----------
    ode
        The SymbolicODE instance to edit.
    parent
        Optional parent widget.

    Example
    -------
    >>> from cubie.odesystems.symbolic import load_cellml_model
    >>> ode = load_cellml_model("model.cellml")
    >>> editor = ParametersEditor(ode)
    >>> editor.exec()  # Modal dialog
    """

    def __init__(
        self,
        ode: "SymbolicODE",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._ode = ode
        self.setWindowTitle("Parameters Editor")
        self.setMinimumSize(500, 400)

        layout = QVBoxLayout(self)
        info = ode.get_parameters_info()
        self._table = _value_table(layout, "Value")
        self._value_edits = _fill_value_table(
            self._table,
            {item['name']: item['value'] for item in info},
            {item['name']: item['unit'] for item in info},
        )

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._on_ok)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

    def _on_ok(self) -> None:
        """Set the edited values as defaults and close the dialog."""
        invalid = _invalid_names(self._value_edits)
        if invalid:
            QMessageBox.warning(
                self,
                "Invalid Values",
                f"The following entries have invalid values: "
                f"{', '.join(invalid)}"
            )
            return
        defaults = self._ode.parameters.values_dict
        changed = {
            name: edit.value()
            for name, edit in self._value_edits.items()
            if edit.value() != defaults[name]
        }
        try:
            self._ode.set_default_parameters(changed)
        except Exception as error:
            QMessageBox.warning(
                self,
                "Errors Applying Changes",
                f"The values could not be applied:\n{error}",
            )
            return
        self.accept()


class PreParseEditor(QDialog):
    """Dialog for editing parameter and initial values before parsing.

    Operates on raw dictionaries rather than a constructed SymbolicODE,
    so the edited values feed ``parse_input()`` and the codegen cache
    key.

    Parameters
    ----------
    parameters_dict
        ``{name: value}`` for parameters.
    initial_values
        ``{name: value}`` for state initial values.
    parameter_units
        ``{name: unit_str}`` for parameter units.
    state_units
        ``{name: unit_str}`` for state units.
    parent
        Optional parent widget.
    """

    def __init__(
        self,
        parameters_dict: dict[str, float],
        initial_values: dict[str, float],
        parameter_units: Optional[dict[str, str]] = None,
        state_units: Optional[dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._parameters = dict(parameters_dict)
        self._initial_values = dict(initial_values)
        self._accepted = False
        self.setWindowTitle("CellML Model Setup")
        self.setMinimumSize(650, 500)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Parameters"))
        self._value_edits = _fill_value_table(
            _value_table(layout, "Value"),
            self._parameters,
            parameter_units or {},
        )
        layout.addWidget(QLabel("Initial State Values"))
        self._state_edits = _fill_value_table(
            _value_table(layout, "Initial Value"),
            self._initial_values,
            state_units or {},
        )

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._on_ok)
        button_layout.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    @property
    def accepted(self) -> bool:
        """Whether the user clicked OK."""
        return self._accepted

    @property
    def result_parameters(self) -> dict[str, float]:
        """Parameters dict after user edits."""
        return dict(self._parameters)

    @property
    def result_initial_values(self) -> dict[str, float]:
        """Initial values dict after user edits."""
        return dict(self._initial_values)

    def _on_ok(self) -> None:
        """Collect edits into result dicts and accept."""
        invalid = _invalid_names(self._value_edits, self._state_edits)
        if invalid:
            QMessageBox.warning(
                self, "Invalid Values",
                "Invalid entries: " + ", ".join(invalid),
            )
            return

        for name, edit in self._value_edits.items():
            self._parameters[name] = edit.value()
        for name, edit in self._state_edits.items():
            self._initial_values[name] = edit.value()

        self._accepted = True
        self.accept()


def edit_pre_parse_dicts(
    parameters_dict: dict[str, float],
    initial_values: dict[str, float],
    parameter_units: Optional[dict[str, str]] = None,
    state_units: Optional[dict[str, str]] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Show the pre-parse editor and return modified dicts.

    Parameters
    ----------
    parameters_dict
        ``{name: value}`` parameters.
    initial_values
        ``{name: value}`` state initial values.
    parameter_units
        Optional ``{name: unit}`` for parameters.
    state_units
        Optional ``{name: unit}`` for states.

    Returns
    -------
    tuple of (dict, dict)
        ``(parameters_dict, initial_values)`` after user edits. If
        the user cancels, the original dicts are returned unchanged.

    Examples
    --------
    >>> params, inits = edit_pre_parse_dicts({"k": 1.0}, {"x": 0.0})
    """
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    editor = PreParseEditor(
        parameters_dict, initial_values, parameter_units, state_units,
    )
    editor.exec() if hasattr(editor, 'exec') else editor.exec_()

    if created_app:
        app.quit()

    if editor.accepted:
        return editor.result_parameters, editor.result_initial_values
    return parameters_dict, initial_values


def show_parameters_editor(
    ode: "SymbolicODE",
    blocking: bool = True,
) -> Optional[ParametersEditor]:
    """Show the parameters editor dialog.

    Parameters
    ----------
    ode
        The SymbolicODE instance to edit.
    blocking
        If True, block until the dialog is closed. If False, return
        the dialog instance immediately.

    Returns
    -------
    ParametersEditor or None
        The dialog instance if non-blocking, None if blocking.

    Examples
    --------
    >>> show_parameters_editor(ode)          # blocking
    >>> editor = show_parameters_editor(ode, blocking=False)
    """
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    editor = ParametersEditor(ode)

    if blocking:
        editor.exec() if hasattr(editor, 'exec') else editor.exec_()
        if created_app:
            app.quit()
        return None
    else:
        editor.show()
        return editor
# no cover: end
