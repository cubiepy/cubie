<!-- Parent: ../AGENTS.md -->

# vendored

## Purpose
A pinned snapshot of numba-cuda's `Cache`/`CUDACache` that gives CuBIE a stable base to extend
with its own file-based caching (`CUBIECache` in `cubie_cache.py`) without tracking upstream
churn, and that compiles under CUDASIM. The upstream source path and snapshot date are recorded
in the module docstring.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package docstring only; no exports. |
| `numba_cuda_cache.py` | Snapshot from NVIDIA/numba-cuda, 2026-07-03: `_Cache`/`Cache` from `numba_cuda/numba/cuda/core/caching.py`; `CUDACache` from `numba_cuda/numba/cuda/dispatcher.py` (its `load_overload` override under `utils.numba_target_override()` plus the launch-config API — `is_launch_config_sensitive`/`mark_launch_config_sensitive`/`set_launch_config_key`/`flush`). |
| `cellmlmanip/` | Vendored snapshot of cellmlmanip 0.3.6 (ModellingWebLab, BSD 3-Clause; `LICENSE` kept alongside). Parses CellML into SymPy via `load_model`. Consumed by `odesystems/symbolic/parsing/cellml.py`. |

## numba-cuda cache snapshot
- The snapshot is unmodified. CuBIE's behaviour lives in `CUBIECache(CUDACache)`
  (`cubie_cache.py`), which sets `_impl_class = CUBIECacheImpl`; the vendored `CUDACache`
  (`_impl_class = None`) is never instantiated. `cuda_simsafe` picks this base or
  `numba.cuda.dispatcher.CUDACache` depending on CUDASIM.
- `CUBIECache.__init__` does not chain to `CUDACache.__init__` (it takes system hashes, not
  a `py_func`), so it replicates the base's per-instance state: the launch-config fields
  `_launch_config_key`, `_launch_config_sensitive_flag` and
  `_launch_config_marker_path`, which the dispatcher reads on every cached launch.
  Re-check them when re-snapshotting.
- To update, replace the file with a newer upstream snapshot and bump the date.
- numba-cuda is BSD 2-Clause; the notice is in the module docstring and
  `THIRD_PARTY_LICENSES`.

## cellmlmanip
- Local modifications: intra-package imports made relative (`from .x`), and a
  `try/except ImportError` fallback in `units.py` for Pint>=0.20 (`ScaleConverter`/
  `UnitDefinition` in `pint.facets.plain`; `UnitDefinition` takes a `reference`). To
  update, re-snapshot upstream and re-apply those two changes.
- Its data files (`data/*.rng`/`.rnc`/`.txt`, `version.txt`, `LICENSE`) ship through
  `[tool.setuptools.package-data]` and load via `os.path.dirname(__file__)`. Its runtime
  dependencies (`lxml`, `networkx`, `Pint`, `rdflib`) are cubie dependencies.

## Dependencies
- `numba_cuda_cache.py`: none internal (consumed by `cubie_cache`/`cuda_simsafe`). Live upstream
  imports: `numba.cuda.core.caching` (`IndexDataCacheFile`), `numba.cuda.serialize` (`dumps`),
  `numba.cuda.utils` (`numba_target_override`).
- `cellmlmanip/`: external `lxml`, `networkx`, `Pint`, `rdflib`, `sympy`; consumed by
  `cubie.odesystems.symbolic.parsing.cellml`.
