<!-- Parent: ../AGENTS.md -->

# gui

## Purpose
Qt editor dialogs for a `SymbolicODE`'s constants, parameters, and initial states.
`ConstantsEditor` and `StatesEditor` edit a live `SymbolicODE`; `PreParseEditor` edits raw dicts
*before* a `SymbolicODE` is built, so its result can feed `parse_input()`. The Qt binding is
abstracted through `qtpy` (PyQt6/PyQt5/PySide6/PySide2).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Re-exports `ConstantsEditor` and `StatesEditor` (the only package exports). |
| `constants_editor.py` | `FloatLineEdit`, `ConstantsEditor` (live constants/parameters), `PreParseEditor` (raw-dict categorisation), and the `edit_pre_parse_dicts()` / `show_constants_editor()` wrappers. |
| `states_editor.py` | `StatesEditor` (live initial-state editing) + `show_states_editor()`. |

## SymbolicODE coupling
- `ConstantsEditor` calls `get_constants_info`/`get_parameters_info`,
  `set_constant_value`/`set_parameter_value`, `make_parameter`/`make_constant` and
  `indices.constant_names`/`parameter_names`; `StatesEditor` calls
  `get_states_info`/`set_initial_value`. `SymbolicODE` is imported under
  `TYPE_CHECKING` only.
- `PreParseEditor` takes and returns plain dicts; its `result_*` properties are valid
  only after OK, and `edit_pre_parse_dicts` returns its inputs unchanged on cancel.
- GUI tests run headless (`QT_QPA_PLATFORM=offscreen`).

## Dependencies
- `qtpy` (needs one of PyQt6/PyQt5/PySide6/PySide2); `cubie.odesystems.symbolic.SymbolicODE`
  (`TYPE_CHECKING` only).
