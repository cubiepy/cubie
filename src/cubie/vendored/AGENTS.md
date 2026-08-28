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
| `numba_cuda_cache.py` | Snapshot from NVIDIA/numba-cuda, 2026-07-03: `_Cache`/`Cache` from `numba_cuda/numba/cuda/core/caching.py`; `CUDACache` from `numba_cuda/numba/cuda/dispatcher.py` (its `load_overload` override under `utils.numba_target_override()` plus the launch-config API — `is_launch_config_sensitive`/`mark_launch_config_sensitive`/`set_launch_config_key`/`flush`, added upstream in PR #804). |
| `cellmlmanip/` | Vendored snapshot of cellmlmanip 0.3.6 (ModellingWebLab, BSD 3-Clause; `LICENSE` kept alongside). Parses CellML into SymPy via `load_model`. Consumed by `odesystems/symbolic/parsing/cellml.py`. |

## For AI Agents
- No inline modifications — it reads as a direct snapshot. CuBIE's customization lives in the
  subclass `CUBIECache(CUDACache)` (`cubie_cache.py`), which sets `_impl_class = CUBIECacheImpl`
  and adds the file-based caching; the vendored `CUDACache` leaves `_impl_class = None` and is
  never instantiated directly. `cuda_simsafe` selects this vendored base vs the real
  `numba.cuda.dispatcher.CUDACache` depending on CUDASIM.
- **`CUBIECache.__init__` does not chain to `CUDACache.__init__`** (it takes system hashes, not a
  `py_func`), so any per-instance state the base constructor sets must be replicated there. It
  currently mirrors `CUDACache`'s launch-config init (`_launch_config_key`,
  `_launch_config_sensitive_flag`, `_launch_config_marker_path`); the dispatcher reads these on
  every cached launch, so re-check this when re-snapshotting a newer numba-cuda.
- Only `_Cache`/`Cache`/`CUDACache` are vendored; the file imports live from
  `numba.cuda.core.caching`, `numba.cuda.serialize`, and `numba.cuda.utils`.
- To update, replace it with a newer upstream snapshot and bump the date; extend behaviour in
  `CUBIECache` rather than editing the snapshot.
- numba-cuda is BSD 2-Clause; the notice is in the module docstring and `THIRD_PARTY_LICENSES`.

### cellmlmanip
- Vendored because cellmlmanip pins `Pint<0.20`, which is incompatible with the `numpy>=2`
  that numba-cuda-mlir requires. Vendoring drops the metadata pin so a modern Pint can be used.
- Local modifications only: absolute intra-package imports (`from cellmlmanip.x`) rewritten to
  relative (`from .x`); a `try/except ImportError` fallback in `units.py` for Pint>=0.20
  (`ScaleConverter`/`UnitDefinition` moved to `pint.facets.plain`, `UnitDefinition` gained a
  required `reference` arg). Do NOT otherwise edit — to update, re-snapshot upstream and re-apply
  these two mechanical changes.
- Data files (`data/*.rng`/`.rnc`/`.txt`, `version.txt`, `LICENSE`) ship via
  `[tool.setuptools.package-data]` in `pyproject.toml`; they load through
  `os.path.dirname(__file__)`, so the vendored layout needs no code change.
- Runtime deps moved into cubie's `dependencies`: `lxml`, `networkx`, `Pint`, `rdflib`.

## Dependencies
- `numba_cuda_cache.py`: none internal (consumed by `cubie_cache`/`cuda_simsafe`). Live upstream
  imports: `numba.cuda.core.caching` (`IndexDataCacheFile`), `numba.cuda.serialize` (`dumps`),
  `numba.cuda.utils` (`numba_target_override`).
- `cellmlmanip/`: external `lxml`, `networkx`, `Pint`, `rdflib`, `sympy`; consumed by
  `cubie.odesystems.symbolic.parsing.cellml`.
