# CuBIE — CUDA Batch Integration Engine

CuBIE JIT-compiles CUDA kernels with Numba to integrate large batches of ODE/SDE systems in
parallel on NVIDIA GPUs — compiled-CUDA speed without writing CUDA, behind a SciPy/MATLAB-like
interface (`solve_ivp`, `Solver`). Domain-agnostic; built for large parameter/initial-condition
sweeps and summary-only extraction (e.g. likelihood-free inference).

## Documentation map
Architecture is documented per directory under `src/cubie/**/AGENTS.md` (each mirrored to a
`CLAUDE.md` symlink). **Start at `src/cubie/AGENTS.md`** — the package root, which defines the
`CUDAFactory` cached-compilation spine and the invariants every subpackage builds on. Device-code
optimisation conventions live in `src/cubie/writing_cuda_functions.md`.

## Setup
- `pip install -e .[dev]` from the repo root (use a venv; some deps are version-pinned).
- **Python 3.11-3.14** (3.10 only on the deprecated numba-cuda backend), **CUDA 12 or 13**
  (via the `mlir-cuda12`/`mlir-cuda13` extras, or a system toolkit), **NVIDIA GPU (compute
  capability ≥6.0)**.
- CPU-only dev/test without a GPU: set `NUMBA_ENABLE_CUDASIM=1` (Numba's CUDA simulator).
  **CUDASIM is not production.** Behaviour under the simulator must never be considered when
  evaluating code: designs, fixes, and diagnostics are judged solely on their real-GPU
  behaviour. A path that works under CUDASIM but degrades or disappears on hardware is broken.
- Dev shell is **PowerShell on Windows** — chain with `;`, not `&&`. Staying Windows-compatible is
  a project goal.

## Testing
Run `pytest` from the repo root. `pyproject.toml` `addopts` already applies coverage and
`-n logical` (xdist), so bare `pytest` is parallel + covered. Only run files relevant to your
change — the full suite is slow. Run the complete simulator and real-GPU suites before opening or
updating a PR; targeted subsets miss cross-cutting tests.
- **Simulator (CPU, matches nocuda CI) — a first pass only:**
  `NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not cupy and not specific_algos"`
- **Real GPU (matches CUDA CI; CUDASIM off) — always run to verify results.** The simulator does
  not guarantee on-device correctness; a change is only verified once the real-GPU tests pass:
  `pytest -m "not specific_algos and not sim_only"`
- Markers (`pyproject.toml`): `nocudasim` (real-GPU only), `sim_only` (simulator-only debug),
  `cupy` (needs CuPy), `slow`, `specific_algos` (non-default tableau aliases).
- **Use the shared session-scoped fixtures in `tests/conftest.py`** with their default parameter
  sets unless the user explicitly excepts a case; don't hand-roll fixtures. **Mocks/patches may
  only be added with an explicit user exception.** Don't type-hint tests.
- **A failing test is a good test.** Never soften a test, loosen a tolerance, or use inexact/lax
  assertions to make it pass — even while developing. Assert the exact intended behaviour.

## Lint & build
- `ruff` (line-length 79, max-doc-length 72, docstring-code-format) and `flake8`. CI's blocking
  gate: `flake8 . --select=E9,F63,F7,F82 --show-source`.
- Coverage config and pytest markers live in `pyproject.toml`.

## Code style
- PEP8: 79-char lines, 71-char comments. Descriptive names, not abbreviations.
- Type hints on function/method **signatures** only (PEP484) — no inline variable annotations, no
  `from __future__ import annotations` (min Python 3.10). numpydoc docstrings on public API.
- Comments describe current behaviour, not change history ("now", "no longer", "changed from" →
  removed). **Never edit `changelog.md`** (plugin-managed).

## Commits & PRs
- **Conventional Commit format**; description in **present-state changelog language** (describe the
  resulting state, e.g. "nested AGENTS.md files created…"). Types: `fix`, `feat` (rare), `test`,
  `docs`, `chore`.
- **Agents:** every fix or feature is developed on its own branch off `main`. When the work is
  done and verified, commit, push the branch, and open a PR.
- Capture `ab_gate.py` stdout untruncated.
- **Performance gate (every PR):** run `python benchmarks/ab_gate.py` and paste its table into
  the PR message. One command compares A (`origin/main`, an ephemeral `git worktree`) against B
  (the working tree) on every installed CUDA backend — both `numba-cuda` and `numba-cuda-mlir`
  should be in the venv. Per backend it starts one persistent worker per side (each compiles and
  builds its grid once) and ping-pongs short solve blocks between them in ABBA order with
  randomised idle gaps — continuous load pins the GPU at its power limit and the kernel-time
  floor dithers, so the rest between blocks keeps it in a repeatable boost state, and the
  per-block jitter stops a concurrent periodic GPU load phase-locking with the rhythm and
  biasing one side coherently. Each block reports the mean of
  its lowest `k` per-solve kernel times (CUDA-event, kernel-only: the fastest solves track the
  kernel's intrinsic cost); the two blocks of a pair run seconds apart and share clock state, so
  the verdict per config is the **median paired delta** against `--threshold` (default 0.50%),
  with non-zero exit on regression. The `host_overhead` config (a constant 8 trajectories,
  guarding the per-call host cost of `Solver.solve`) gates on its wall statistic only, in
  absolute milliseconds against `--host-overhead-threshold` (default 0.25 ms). A default run
  takes ~3 minutes for two backends on a quiet GPU. A row marked DISTRUST means the per-pair deltas disagreed. Retry the gate once, with more
  `--pairs`; publish the PR with the retry's table as-is, DISTRUST rows and all. Constant
  background load inflates absolute times but cancels out of the deltas. `--calibrate` measures
  the A-vs-A null for setting the threshold on a new machine;
  `--n-runs 1024` smoke-tests the harness cheaply.
- ** Any changes left uncommitted or unstaged will be programatically deleted **. The only place to
  store work is in a branch off origin, pushed to main, with a PR open. PRs are the only format
  reviewed by the user. Don't leave PRs draft, they must be marked ready and reviewed by Greptile
  before the user can review.

## Cross-cutting code rules (details in `src/cubie/AGENTS.md`)
- Never call a `CUDAFactory.build()` directly — access compiled functions via the cached properties.
- Never set/modify env vars in source (esp. `NUMBA_ENABLE_CUDASIM`); set them externally.
- Module-scoped imports belong in the file header only; deliberate lazy imports of optional deps
  (Qt) stay function-local. cupy/cupyx are required on a real GPU and imported
  once, conditionally, in `cuda_simsafe` — import them from there (`from cubie.cuda_simsafe
  import cupy, cupyx`), never directly and never lazily.
- In `CUDAFactory`/device-code files, use explicit imports with the project aliasing (`np_`,
  `attrsval_`, `attrs`-prefixed); store float config fields underscored and expose via a
  precision-casting property.
- Device-code optimisation patterns (predicated commit, warp-coherent loops, …) live in
  `src/cubie/writing_cuda_functions.md`.
- No backwards-compatibility burden — breaking changes are expected pre-1.0.

## Dependencies
- **Core:** numpy>=2.0, numba, attrs, sympy>=1.13.0. cellmlmanip is vendored under
  `src/cubie/vendored/cellmlmanip` (its `lxml`/`networkx`/`Pint>=0.24`/`rdflib` runtime deps are core).
- **CUDA backend (installed by extra, so installs stay clean):** numba-cuda-mlir is the
  default (Python >= 3.11; `mlir`/`mlir-cuda12`/`mlir-cuda13` extras — these install
  `cubie-numba-cuda-mlir`, cubie's own build carrying the native-code fixes pending
  upstream, with the same `numba_cuda_mlir` import package; never co-install it with the
  stock wheel, and treat the installed wheel, not upstream numba-cuda-mlir source, as
  ground truth when debugging how device code compiles). `numba-cuda`
  (`cuda`/`cuda12`/`cuda13` extras) is deprecated but kept as a fallback for unexpected
  errors, Python 3.10, and the CUDA simulator. A backendless install fails at
  `import cubie` with instructions. `cubie.cuda_backend` resolves the active backend at
  import: `CUBIE_CUDA_BACKEND` overrides; otherwise the installed backend is used,
  preferring mlir when an environment ends up with both and numba-cuda under CUDASIM.
  The CUDA simulator exists only on numba-cuda.
- **CUDA toolkit:** supplied by the `cuda12`/`cuda13`/`mlir-cuda12`/`mlir-cuda13` extras or an
  existing system install (the bare `cuda`/`mlir` extras use whatever toolkit the backend finds).
- **CuPy is required for real-GPU execution** — it is cubie's single device memory allocator.
  The toolkit extras pull in the matching cupy build alongside the toolkit wheels.
  It is imported at `import cubie` through `cubie.cuda_simsafe`; the CUDA simulator
  (`NUMBA_ENABLE_CUDASIM=1`) never requires it.
- **Optional:** pandas (DataFrame output), matplotlib (driver plots).
- CI installs a patched `numba-cuda` fork (`ccam80/numba-cuda@cubie_patch`) for faster compile;
  stock `numba-cuda` works for local dev.
