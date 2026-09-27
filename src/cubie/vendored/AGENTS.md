<!-- Parent: ../AGENTS.md -->

# vendored

## Purpose
Third-party code: the CUDA simulator from numba-cuda (on numba-cuda-mlir) and cellmlmanip.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package docstring only; no exports. |
| `cudasim/` | The CUDA simulator from NVIDIA/numba-cuda 0.30.4 (`numba_cuda/numba/cuda/simulator`, plus `is_available`/`cuda_error` from `simulator_init.py`), snapshot 2026-09-27. Runs device code as Python threads on the CPU. `cubie_cudasim_extensions` extends it and serves it as `cuda` when `NUMBA_ENABLE_CUDASIM=1`. |
| `cellmlmanip/` | Vendored snapshot of cellmlmanip 0.3.6 (ModellingWebLab, BSD 3-Clause; `LICENSE` kept alongside). Parses CellML into SymPy via `load_model`. Consumed by `odesystems/symbolic/parsing/cellml.py`. |

## cudasim
- `numba.cuda.*` imports point at `numba_cuda_mlir.numba_cuda.*`; the kernel-time
  module swap replaces globals bound to this package.
- Local modifications: `__init__.py` drops the `sys.modules["numba.cuda.*"]` aliasing
  and imports its submodules unconditionally. Cubie's additions to the simulator
  live in `cubie/cubie_cudasim_extensions.py`, applied at its import.
- To update, re-snapshot upstream and re-apply the import rewrite and the `__init__.py`
  change.
- BSD 2-Clause; notice in each upstream file header and `THIRD_PARTY_LICENSES`.

## cellmlmanip
- Local modifications: intra-package imports made relative (`from .x`), and a
  `try/except ImportError` fallback in `units.py` for Pint>=0.20 (`ScaleConverter`/
  `UnitDefinition` in `pint.facets.plain`; `UnitDefinition` takes a `reference`). To
  update, re-snapshot upstream and re-apply those two changes.
- Its data files (`data/*.rng`/`.rnc`/`.txt`, `version.txt`, `LICENSE`) ship through
  `[tool.setuptools.package-data]` and load via `os.path.dirname(__file__)`. Its runtime
  dependencies (`lxml`, `networkx`, `Pint`, `rdflib`) are cubie dependencies.

## Dependencies
- `cudasim/`: none internal (consumed by `cubie_cudasim_extensions`). External: `numpy`,
  `numba_cuda_mlir.numba_cuda` (config, types, typing helpers).
- `cellmlmanip/`: external `lxml`, `networkx`, `Pint`, `rdflib`, `sympy`; consumed by
  `cubie.odesystems.symbolic.parsing.cellml`.
