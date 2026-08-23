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
| `constants_editor.py` | `FloatLineEdit`, `ConstantsEditor` (live named-value editing), `PreParseEditor` (raw-dict value editing), and the `edit_pre_parse_dicts()` / `show_constants_editor()` wrappers. |
| `states_editor.py` | `StatesEditor` (live initial-state editing) + `show_states_editor()`. |

## For AI Agents
- The dialogs are the GUI's only coupling to `SymbolicODE`, through its public methods
  (duck-typed; `SymbolicODE` imported under `TYPE_CHECKING`): `ConstantsEditor` uses
  `get_constants_info`/`get_parameters_info`, `set_constants`,
  `indices.constant_names`/`parameter_names`, and in-place parameter-value writes;
  `StatesEditor` uses `get_states_info`/`set_initial_value`. The editors set values only —
  the swept/folded partition is owned by the solve-time grid, not the GUI.
- `PreParseEditor` is the odd one out — plain dicts in/out (no `SymbolicODE`), so it can run
  before parsing; its `result_*` properties are valid only after OK, and `edit_pre_parse_dicts`
  returns its inputs unchanged on cancel.
- No `tests/gui/`; run any new tests headless (`QT_QPA_PLATFORM=offscreen`).

## Dependencies
- `qtpy` (needs one of PyQt6/PyQt5/PySide6/PySide2); `cubie.odesystems.symbolic.SymbolicODE`
  (`TYPE_CHECKING` only).
