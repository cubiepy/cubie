# CuBIE — CUDA Batch Integration Engine

CuBIE JIT-compiles CUDA kernels with Numba to integrate large batches of ODE/SDE systems in
parallel on NVIDIA GPUs — compiled-CUDA speed without writing CUDA, behind a SciPy/MATLAB-like
interface (`solve_ivp`, `Solver`). Domain-agnostic; built for large parameter/initial-condition
sweeps and summary-only extraction (e.g. likelihood-free inference).

## Documentation map
Architecture is documented per directory under `src/cubie/**/AGENTS.md` (each mirrored to a
`CLAUDE.md` symlink). **Start at `src/cubie/AGENTS.md`** — the package root, which defines the
`CUDAFactory` cached-compilation spine and the invariants every subpackage builds on, including
the device-code optimisation conventions.

## Setup
- `pip install -e .[dev-cuda13]` from the repo root (use a venv; some deps are
  version-pinned). `dev` uses a system CUDA toolkit; `dev-cuda12`/`dev-cuda13` install one.
- Every worktree runs in its own `.venv`. If you run `git worktree add`, run
  `python ci/tools/worktree_setup.py` (or `python3 ...` where there is no
  `python`) in the new worktree before anything else. The script builds the
  venv but does not activate it, so run every command through the venv's
  interpreter: `.venv/Scripts/python -m pytest` on Windows,
  `.venv/bin/python -m pytest` elsewhere.
- **Python 3.11-3.14**, **CUDA 12 or 13** (via the `cuda12`/`cuda13` extras, or a
  system toolkit), **NVIDIA GPU (compute capability ≥6.0)**.
- CPU-only dev/test without a GPU: set `NUMBA_ENABLE_CUDASIM=1` (the CUDA simulator
  vendored from numba-cuda under `src/cubie/vendored/cudasim`; the `dev` installs cover it).
  **CUDASIM is not production.** Behaviour under the simulator must never be considered when
  evaluating code: designs, fixes, and diagnostics are judged solely on their real-GPU
  behaviour. A path that works under CUDASIM but degrades or disappears on hardware is broken.
- **float64 is not a realistic test case for this package, ever.** If you are forced to use f64 to verify something, the problem is actually elsewhere.
- Dev shell is **PowerShell on Windows** — chain with `;`, not `&&`. Staying Windows-compatible is
  a project goal.

## Testing
Run `.venv/Scripts/python -m pytest` from the repo root. `pyproject.toml` `addopts` already
applies coverage and `-n logical` (xdist), so a bare run is parallel + covered. Only run files relevant to your
change — the full suite is slow. Run the complete simulator and real-GPU suites before opening or
updating a PR; targeted subsets miss cross-cutting tests.
- **Simulator (CPU, matches nocuda CI) — a first pass only:**
  `NUMBA_ENABLE_CUDASIM=1 .venv/Scripts/python -m pytest -m "not nocudasim and not specific_algos"`
- **Real GPU (matches CUDA CI; CUDASIM off) — always run to verify results.** The simulator does
  not guarantee on-device correctness; a change is only verified once the real-GPU tests pass:
  `.venv/Scripts/python -m pytest -m "not specific_algos and not sim_only"`
- **Use the shared session-scoped fixtures in `tests/conftest.py`** with their default parameter
  sets unless the user explicitly excepts a case; don't hand-roll fixtures. **Mocks/patches may
  only be added with an explicit user exception.** Don't type-hint tests.
- **A failing test is a good test.** Never soften a test, loosen a tolerance, or use inexact/lax
  assertions to make it pass — even while developing. Assert the exact intended behaviour.
- **No negative-presence tests.** Assert positive behaviour, never that a key or field is absent.
- **Timing/performance measurements run ≥ 2 full occupancy waves** (runs ≥ 2 × SMs × resident
  blocks/SM × runs/block at the compiled geometry); smaller batches are invalid.

## Lint & build
- `ruff` (line-length 79, max-doc-length 72, docstring-code-format) and `flake8`. CI's blocking
  gate: `flake8 . --select=E9,F63,F7,F82 --show-source`.

## Code style
- PEP8: 79-char lines, 71-char comments. Descriptive names, not abbreviations.
- Type hints on function/method **signatures** only (PEP484) — no inline variable annotations, no
  `from __future__ import annotations` (min Python 3.11). numpydoc docstrings on public API.
- Write comments as the programmer explaining the code to a colleague, in the imperative
  ("Place driver derivatives after drivers in the buffer") or as narrative ("We sort first so
  the cache key is stable"). Data and objects are never the actor: "Driver derivatives take the
  last slots; drivers carry priority" is unreadable.
- Use whole sentences in plain words, comment only where the code is hard to read, and say why
  when the reason is not visible in the code. Never describe what used to exist ("now", "no
  longer", "changed from"), a bug avoided, or an alternative rejected.
- Docstrings describe what the code does and any non-obvious behaviour, in the same plain
  sentences. `AGENTS.md` files tell the reader what they need to use, extend or debug the code.
- Cite a source only for directly ported code (see `odesystems/symbolic/structural/AGENTS.md`).
- **Never edit `changelog.md`** (plugin-managed).

## Commits & PRs
- **Conventional Commit format**; description in **present-state changelog language** (describe the
  resulting state, e.g. "nested AGENTS.md files created…"). Types: `fix`, `feat` (rare), `test`,
  `docs`, `chore`.
- **Agents:** every fix or feature is developed on its own branch off `main`. When the work is
  done and verified, commit, push the branch, and open a PR.
- Capture `ab_gate.py` stdout untruncated.
- **Performance gate (every PR that touches `src/`):** run `python benchmarks/ab_gate.py` and
  paste its table into the PR message. One command compares A (`origin/main`, an ephemeral
  `git worktree`) against B (the working tree). It starts one persistent worker per side (each
  compiles and builds its grid once) and ping-pongs short solve blocks between them in ABBA
  order with randomised idle gaps — continuous load pins the GPU at its power limit and the kernel-time
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
  reviewed by the user. Don't leave PRs draft, they must be marked ready.

## Cross-cutting code rules (details in `src/cubie/AGENTS.md`)
- Never call a `CUDAFactory.build()` directly — access compiled functions via the cached properties.
- Never set/modify env vars in source (esp. `NUMBA_ENABLE_CUDASIM`); set them externally.
- Module-scoped imports belong in the file header only; deliberate lazy imports of optional deps
  (Qt) stay function-local. Import `cuda` from `cubie._cudasim_extensions`; simulator
  stand-ins live there too.
- In `CUDAFactory`/device-code files, use explicit imports with the project aliasing (`np_`,
  `attrsval_`, `attrs`-prefixed); store float config fields underscored and expose via a
  precision-casting property.
- Device-code optimisation patterns (predicated commit, compile-time branching,
  warp-coherent loop exit) live in `src/cubie/AGENTS.md`.
- No backwards-compatibility burden — breaking changes are expected pre-1.0.

## Dependencies
- **Core:** numpy>=2.0, attrs, sympy>=1.13.0. cellmlmanip is vendored under
  `src/cubie/vendored/cellmlmanip` (its `lxml`/`networkx`/`Pint>=0.24`/`rdflib` runtime deps are core).
- **CUDA backend (core):** `cubie-numba-cuda-mlir`, cubie's own build of numba-cuda-mlir
  carrying the native-code fixes pending upstream, with the same `numba_cuda_mlir` import
  package; never co-install it with the stock wheel, and treat the installed wheel, not
  upstream numba-cuda-mlir source, as ground truth when debugging how device code
  compiles. A backendless environment fails at `import cubie` with instructions. The CUDA
  simulator is vendored (`cubie.vendored.cudasim`) and runs on the same install.
- **CUDA toolkit:** supplied by the `cuda12`/`cuda13` extras or an existing system
  install (a plain `pip install cubie` uses whatever toolkit the backend finds).
- **Device memory** comes from Numba and the device's stream-ordered pool through
  `cuda.bindings` (a dependency of numba-cuda-mlir).
- **Optional:** pandas (DataFrame output), matplotlib (driver plots).
