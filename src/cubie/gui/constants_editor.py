# no cover: start
"""Qt GUI for editing named values in a SymbolicODE.

Table-based editors for viewing and modifying the named values
(swept parameters and folded constants alike) of a
:class:`~cubie.odesystems.symbolic.SymbolicODE`.

Published Classes
-----------------
:class:`FloatLineEdit`
    Line edit widget with forgiving float validation.

    >>> edit = FloatLineEdit(1.5e-3)
    >>> edit.value()
    0.0015

:class:`ConstantsEditor`
    Modal dialog for editing named values on a live ``SymbolicODE``.

    >>> editor = ConstantsEditor(ode)
    >>> editor.exec()

:class:`PreParseEditor`
    Modal dialog for editing named values and initial states before
    parsing (operates on raw dictionaries).

Module-Level Functions
----------------------
:func:`edit_pre_parse_dicts`
    Show a :class:`PreParseEditor` and return the modified dicts.

    >>> values, iv = edit_pre_parse_dicts(
    ...     {"g": 9.81, "k": 1.0}, {"x": 0.0},
    ... )

:func:`show_constants_editor`
    Convenience wrapper to display a :class:`ConstantsEditor`.

    >>> show_constants_editor(ode)

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
    QTableWidgetItem, QLineEdit, QPushButton, QLabel,
    QHeaderView, QMessageBox, QWidget,
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


class ConstantsEditor(QDialog):
    """Dialog for editing named values in a SymbolicODE.

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
    >>> editor = ConstantsEditor(ode)
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
        self.setWindowTitle("Values Editor")
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Edit the model's named values. Values are swept or "
            "folded per solve, driven by the batch grid."
        )
        layout.addWidget(info_label)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Value", "Unit"]
        )

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
        """Populate the table with the system's named values."""
        all_items = (
            self._ode.get_constants_info()
            + self._ode.get_parameters_info()
        )
        all_items.sort(key=lambda x: x['name'])

        self._table.setRowCount(len(all_items))

        for row, item in enumerate(all_items):
            name = item['name']
            value = item['value']
            unit = item['unit']

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

        constant_names = set(self._ode.indices.constant_names)
        parameter_names = set(self._ode.indices.parameter_names)

        constant_updates = {}
        parameter_updates = {}
        for name, edit in self._value_edits.items():
            value = edit.value()
            if name in constant_names:
                constant_updates[name] = value
            elif name in parameter_names:
                parameter_updates[name] = value

        errors = []
        try:
            if constant_updates:
                self._ode.set_constants(constant_updates)
        except Exception as e:
            errors.append(str(e))
        for name, value in parameter_updates.items():
            try:
                self._ode.parameters[name] = value
                self._ode.indices.parameters.update_values({name: value})
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


class PreParseEditor(QDialog):
    """Dialog for editing named values before parsing.

    Operates on raw dictionaries rather than a constructed SymbolicODE,
    allowing the user's edits to feed into ``parse_input()`` and the
    codegen cache key.

    Parameters
    ----------
    values_dict
        ``{name: value}`` for the model's named values.
    initial_values
        ``{name: value}`` for state initial values.
    value_units
        ``{name: unit_str}`` for named-value units.
    state_units
        ``{name: unit_str}`` for state units.
    parent
        Optional parent widget.
    """

    def __init__(
        self,
        values_dict: dict[str, float],
        initial_values: dict[str, float],
        value_units: Optional[dict[str, str]] = None,
        state_units: Optional[dict[str, str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._values = dict(values_dict)
        self._initial_values = dict(initial_values)
        self._value_units = value_units or {}
        self._state_units = state_units or {}
        self._value_edits: dict[str, FloatLineEdit] = {}
        self._state_edits: dict[str, FloatLineEdit] = {}
        self._accepted = False

        self._setup_ui()
        self._populate_values_table()
        self._populate_states_table()

    @property
    def accepted(self) -> bool:
        """Whether the user clicked OK."""
        return self._accepted

    @property
    def result_values(self) -> dict[str, float]:
        """Named-values dict after user edits."""
        return dict(self._values)

    @property
    def result_initial_values(self) -> dict[str, float]:
        """Initial values dict after user edits."""
        return dict(self._initial_values)

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle("CellML Model Setup")
        self.setMinimumSize(650, 500)

        layout = QVBoxLayout(self)

        # --- Named values section ---
        cp_label = QLabel(
            "Named Values\n"
            "Values are swept or folded per solve, driven by the "
            "batch grid."
        )
        layout.addWidget(cp_label)

        self._cp_table = QTableWidget()
        self._cp_table.setColumnCount(3)
        self._cp_table.setHorizontalHeaderLabels(
            ["Name", "Value", "Unit"]
        )
        header = self._cp_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self._cp_table.setColumnWidth(1, 150)
        self._cp_table.setColumnWidth(2, 100)
        layout.addWidget(self._cp_table)

        # --- Initial states section ---
        st_label = QLabel("Initial State Values")
        layout.addWidget(st_label)

        self._st_table = QTableWidget()
        self._st_table.setColumnCount(3)
        self._st_table.setHorizontalHeaderLabels(
            ["Name", "Initial Value", "Unit"]
        )
        st_header = self._st_table.horizontalHeader()
        st_header.setSectionResizeMode(0, QHeaderView.Stretch)
        st_header.setSectionResizeMode(1, QHeaderView.Fixed)
        st_header.setSectionResizeMode(2, QHeaderView.Fixed)
        self._st_table.setColumnWidth(1, 150)
        self._st_table.setColumnWidth(2, 100)
        layout.addWidget(self._st_table)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._on_ok)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _populate_values_table(self) -> None:
        """Populate the named-values table from the dict."""
        items = sorted(self._values.items())
        self._cp_table.setRowCount(len(items))

        for row, (name, value) in enumerate(items):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._cp_table.setItem(row, 0, name_item)

            ve = FloatLineEdit(value)
            self._value_edits[name] = ve
            self._cp_table.setCellWidget(row, 1, ve)

            unit = self._value_units.get(name, "")
            unit_item = QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self._cp_table.setItem(row, 2, unit_item)

    def _populate_states_table(self) -> None:
        """Populate the initial-states table from dict."""
        items = sorted(self._initial_values.items())
        self._st_table.setRowCount(len(items))

        for row, (name, value) in enumerate(items):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._st_table.setItem(row, 0, name_item)

            ve = FloatLineEdit(value)
            self._state_edits[name] = ve
            self._st_table.setCellWidget(row, 1, ve)

            unit = self._state_units.get(name, "")
            unit_item = QTableWidgetItem(unit)
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self._st_table.setItem(row, 2, unit_item)

    def _on_ok(self) -> None:
        """Collect edits into result dicts and accept."""
        invalid = [n for n, e in self._value_edits.items()
                   if not e.isValid()]
        invalid += [n for n, e in self._state_edits.items()
                    if not e.isValid()]
        if invalid:
            QMessageBox.warning(
                self, "Invalid Values",
                "Invalid entries: " + ", ".join(invalid),
            )
            return

        for name, ve in self._value_edits.items():
            self._values[name] = ve.value()

        for name, ve in self._state_edits.items():
            self._initial_values[name] = ve.value()

        self._accepted = True
        self.accept()


def edit_pre_parse_dicts(
    values_dict: dict[str, float],
    initial_values: dict[str, float],
    value_units: Optional[dict[str, str]] = None,
    state_units: Optional[dict[str, str]] = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Show the pre-parse editor and return modified dicts.

    Parameters
    ----------
    values_dict
        ``{name: value}`` named values.
    initial_values
        ``{name: value}`` state initial values.
    value_units
        Optional ``{name: unit}`` for named values.
    state_units
        Optional ``{name: unit}`` for states.

    Returns
    -------
    tuple of (dict, dict)
        ``(values_dict, initial_values)`` after user edits. If the
        user cancels, the original dicts are returned unchanged.

    Examples
    --------
    >>> values, inits = edit_pre_parse_dicts(
    ...     {"g": 9.81, "k": 1.0}, {"x": 0.0},
    ... )
    """
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    editor = PreParseEditor(
        values_dict, initial_values, value_units, state_units,
    )
    editor.exec() if hasattr(editor, 'exec') else editor.exec_()

    if created_app:
        app.quit()

    if editor.accepted:
        return (
            editor.result_values,
            editor.result_initial_values,
        )
    return (values_dict, initial_values)


def show_constants_editor(
    ode: "SymbolicODE",
    blocking: bool = True,
) -> Optional[ConstantsEditor]:
    """Show the named-values editor dialog.

    Parameters
    ----------
    ode
        The SymbolicODE instance to edit.
    blocking
        If True, block until the dialog is closed. If False, return
        the dialog instance immediately.

    Returns
    -------
    ConstantsEditor or None
        The dialog instance if non-blocking, None if blocking.

    Examples
    --------
    >>> show_constants_editor(ode)          # blocking
    >>> editor = show_constants_editor(ode, blocking=False)
    """
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    editor = ConstantsEditor(ode)

    if blocking:
        editor.exec() if hasattr(editor, 'exec') else editor.exec_()
        if created_app:
            app.quit()
        return None
    else:
        editor.show()
        return editor
# no cover: end
