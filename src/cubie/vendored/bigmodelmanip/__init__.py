"""Main module for loading, parsing and manipulating BigModel models.

Vendored into CuBIE from bigmodelmanip 0.3.6
(BSD 3-Clause; see the
adjacent LICENSE file). Local modifications: absolute intra-package
imports (``bigmodelmanip.x``) rewritten to relative (``.x``), and a
Pint >= 0.20 compatibility fallback in ``units.py``.
"""
from ._config import __version__, __version_int__, version  # noqa
from .main import load_model  # noqa
