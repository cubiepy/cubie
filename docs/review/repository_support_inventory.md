# Repository support inventory

## Scope

This lane covers the tracked complement left after the documentation,
package, integrator, and test lanes were assigned. The manifest was derived
from `git ls-files` and reconciled against those lane boundaries.

The complement contains 61 files, 65,778 text lines, and 2,306,969 bytes.
All files decode as strict UTF-8.

| Class | Files | Lines | Bytes | Review method |
|---|---:|---:|---:|---|
| Human-authored prose, code, and configuration | 44 | 11,478 | 420,194 | Every line read |
| Generated, schema, fixture, and inventory data | 17 | 54,300 | 1,886,775 | Every byte traversed and structure checked |
| Total | 61 | 65,778 | 2,306,969 | Reconciled with the tracked manifest |

The complement excludes all `docs/**` files, mapped documentation pages,
the Python package and test lanes, and every `AGENTS.md` or `CLAUDE.md`
instruction mirror. Those files appear in the other review inventories.

## Findings

Four freshness findings and one generated-artifact finding remain after a
comparison with the other files in `docs/review`.

| ID | Evidence | Result |
|---|---|---|
| D3-F01 | `benchmarks/lorenz_mean_runtime.py:520-527`, `benchmarks/lorenz_mean_runtime.py:651-654` | Command help says the wall statistic is a median. The implementation computes the mean of the lowest `k` wall times. |
| D3-F02 | `benchmarks/memory_location_sweep.py:1088`, `src/cubie/integrators/memory_heuristics.py:1` | Printed paste instructions omit the repository's `src/` directory. |
| D3-F03 | `scratch_profile/bench_ncu_bicgstab.py:40`, `scratch_profile/bench_ncu_bigsys.py:29` | Both profiling scripts prepend a nonexistent personal worktree path. |
| D3-F04 | `tests/_inventory.json:8483-8727`, `src/cubie/odesystems/symbolic/engine/printer.py:26`, `src/cubie/odesystems/symbolic/engine/printer.py:95` | The inventory describes the removed `numba_cuda_printer.py` and `CUDAPrinter`. Current code exports `IRPrinter` from `engine/printer.py`. |
| D3-F05 | `tests/_inventory.json:21716`, `src/cubie/batchsolving/_utils.py:13-16`, `tests/query_inventory.py:25-33` | The inventory calls `_utils.py` empty although it exports `name_and_compile_kernel`. Its query tool points to the missing `tests/_convert_inventory.py`. |

The inventory drift is wider than D3-F04 and D3-F05. A section-symbol
comparison found 16 buckets containing names absent from their matched
current source file. Confirmed examples include `CacheConfig` at
`tests/_inventory.json:3701-3715`, `_check_saved_indices` at
`tests/_inventory.json:17585-17592`, and `_align_run_counts` at
`tests/_inventory.json:22319`. Current definitions are visible at
`src/cubie/cubie_cache.py:680`,
`src/cubie/outputhandling/output_config.py:290-337`, and
`src/cubie/batchsolving/BatchInputHandler.py:808-841`. The generated
inventory needs a complete refresh from current source and tests.

No existing review artifact mentions these findings. A search covered every
file already present in `docs/review`.

## Language scan

The supplied false-drama and framing phrases have zero literal matches in
this complement.

Twenty-eight human-authored lines contain an em dash. Twenty-seven use it as
sentence punctuation and have proposed rewrites. The remaining occurrence at
`benchmarks/ncu_algorithm_comparison.py:427` is a table missing-value glyph.
It is not sentence punctuation and remains valid.

The semicolon review separated prose from syntax. Syntax uses remain valid in
shell snippets, Python one-line commands, content-security-policy strings,
MIME types, and generated result delimiters. Every prose semicolon and every
framing colon found in the benchmark and support descriptions has a rewrite
in `repository_support_supplemental_proposals.md`.

The generated test inventory contains 8 semicolons and 331 colons in
`section` or `text` values. Its stale content makes line edits unsafe. The
proposal regenerates it and applies the prose rule at the source of
generation.

## Validation receipts

The read-only structure checks produced these results.

- Python AST parsing passed for 15 support scripts.
- YAML parsing passed for 11 GitHub configuration files.
- HCL parsing passed for the Terraform lock file, four Terraform files, and
  the Packer file.
- TOML parsing passed for `pyproject.toml`.
- PowerShell parsing passed for all three PowerShell tools.
- Bash syntax checking passed for `cloudshell-iam.sh`.
- JavaScript syntax checking passed for `cost_dashboard.js`.
- HTML parsing passed for `cost_dashboard.html`.
- `cellml_1_0.rng` compiled as Relax NG. `mathml2.rng` compiled through an
  include wrapper with a `mathml.math` start rule.
- All nine CellML documents parsed and validated against
  `cellml_1_0.rng`. They contain no duplicate component names, duplicate
  variable names, or dangling mapped-variable references.
- Both RNC files have balanced braces, parentheses, and brackets.
  `cellml_1_0.rnc` includes `mathml2.rnc`.
- `cellml_units.txt` loaded into an empty Pint registry with 32 units and
  the `mks` system.
- `version.txt` contains the valid semantic version `0.3.6`.
- `tests/_inventory.json` parsed into 100 buckets and 3,482 entries. Every
  entry has the required typed fields and its `file` field matches its
  bucket key. Content freshness failed as recorded above.

## File manifest

| File | Lines | Bytes | Review |
|---|---:|---:|---|
| `.gitattributes` | 4 | 211 | Read, no documentation issue |
| `.github/dependabot.yml` | 11 | 536 | Read, no documentation issue |
| `.github/runs-on.yml` | 60 | 3,096 | Read, no documentation issue |
| `.github/workflows/build-windows-gpu-ami.yml` | 210 | 8,963 | Read, no documentation issue |
| `.github/workflows/ci_cuda_tests.yml` | 698 | 35,264 | Read, no documentation issue |
| `.github/workflows/ci_nocuda_tests.yml` | 66 | 2,412 | Read, no documentation issue |
| `.github/workflows/ci_semver_manage.yml` | 93 | 3,332 | Read, no documentation issue |
| `.github/workflows/cleanup-ami-builder.yml` | 38 | 1,327 | Read, no documentation issue |
| `.github/workflows/copilot-setup-steps.yml` | 40 | 1,309 | Read, no documentation issue |
| `.github/workflows/documentation.yml` | 48 | 1,402 | Read, no documentation issue |
| `.github/workflows/test_pypi.yml` | 76 | 2,415 | Read, no documentation issue |
| `.github/workflows/todo-actions.yml` | 16 | 456 | Read, no documentation issue |
| `.gitignore` | 73 | 671 | Read, no documentation issue |
| `LICENSE` | 21 | 1,089 | Read, no documentation issue |
| `benchmarks/ab_gate.py` | 574 | 23,388 | Prose rewrite candidates |
| `benchmarks/dense_prediction_ratio_sweep.py` | 689 | 24,522 | Prose rewrite candidates |
| `benchmarks/lorenz_mean_runtime.py` | 704 | 26,006 | D3-F01 and prose rewrite candidates |
| `benchmarks/memory_location_sweep.py` | 1,126 | 39,217 | D3-F02 and prose rewrite candidates |
| `benchmarks/ncu_algorithm_comparison.py` | 821 | 25,659 | Prose rewrite candidates |
| `benchmarks/ncu_algorithm_worker.py` | 299 | 9,380 | Prose rewrite candidates |
| `benchmarks/vendor_julia_reference.py` | 136 | 5,700 | Prose rewrite candidates |
| `blackbox/__init__.py` | 56 | 1,876 | Prose rewrite candidates |
| `blackbox/blackbox1` | 1,201 | 41,946 | CellML structure clean |
| `blackbox/blackbox2` | 7,062 | 298,860 | CellML structure clean |
| `blackbox_solve_2.py` | 309 | 12,018 | Prose rewrite candidates |
| `ci/requirements.ci.txt` | 13 | 156 | Read, no documentation issue |
| `ci/tools/install_gpu_driver.ps1` | 154 | 6,055 | Read, no documentation issue |
| `ci/tools/merge_junit.py` | 94 | 2,982 | Prose rewrite candidate |
| `ci/tools/populate_uv_cache.ps1` | 97 | 3,502 | Read, no documentation issue |
| `ci/tools/prepare_ci_image.ps1` | 208 | 7,807 | Read, no documentation issue |
| `infra/fleet/.gitignore` | 7 | 92 | Read, no documentation issue |
| `infra/fleet/.terraform.lock.hcl` | 76 | 4,344 | HCL structure clean |
| `infra/fleet/bootstrap/cloudshell-iam.sh` | 553 | 18,862 | Read and syntax checked |
| `infra/fleet/cost_dashboard.html` | 138 | 4,831 | Read and structure checked |
| `infra/fleet/cost_dashboard.js` | 609 | 18,922 | Read and syntax checked |
| `infra/fleet/cost_dashboard.py` | 2,238 | 84,104 | Prose rewrite candidates |
| `infra/fleet/main.tf` | 191 | 6,105 | Read and structure checked |
| `infra/fleet/outputs.tf` | 14 | 474 | Read and structure checked |
| `infra/fleet/terraform.tfvars.example` | 13 | 528 | Read and structure checked |
| `infra/fleet/variables.tf` | 50 | 1,658 | Read and structure checked |
| `noxfile.py` | 61 | 2,547 | Prose rewrite candidates |
| `packer/windows-gpu.pkr.hcl` | 177 | 5,834 | Read and structure checked |
| `pyproject.toml` | 179 | 6,141 | Prose rewrite candidates |
| `scratch_profile/bench_ncu_bicgstab.py` | 169 | 6,388 | D3-F03 and prose rewrite candidates |
| `scratch_profile/bench_ncu_bigsys.py` | 156 | 5,387 | D3-F03 and prose rewrite candidates |
| `src/cubie/vendored/cellmlmanip/LICENSE` | 34 | 1,821 | Read, no documentation issue |
| `src/cubie/vendored/cellmlmanip/data/cellml_1_0.rnc` | 508 | 18,320 | RNC structure clean |
| `src/cubie/vendored/cellmlmanip/data/cellml_1_0.rng` | 989 | 34,262 | Relax NG structure clean |
| `src/cubie/vendored/cellmlmanip/data/cellml_units.txt` | 47 | 1,103 | Pint definitions load cleanly |
| `src/cubie/vendored/cellmlmanip/data/mathml2.rnc` | 1,278 | 47,972 | RNC structure clean |
| `src/cubie/vendored/cellmlmanip/data/mathml2.rng` | 3,450 | 102,803 | Relax NG include structure clean |
| `src/cubie/vendored/cellmlmanip/version.txt` | 1 | 7 | Semantic version clean |
| `tests/_inventory.json` | 27,201 | 801,479 | D3-F04 and D3-F05 |
| `tests/fixtures/cellml/Fabbri_Linder.cellml` | 7,158 | 314,345 | CellML structure clean |
| `tests/fixtures/cellml/_make_blackbox.py` | 155 | 5,749 | Read and syntax checked |
| `tests/fixtures/cellml/basic_ode.cellml` | 23 | 721 | CellML structure clean |
| `tests/fixtures/cellml/beeler_reuter_model_1977.cellml` | 1,246 | 46,832 | CellML structure clean |
| `tests/fixtures/cellml/demir_clark_giles_1999.cellml` | 3,951 | 170,288 | CellML structure clean |
| `tests/fixtures/cellml/ghk_singularity.cellml` | 51 | 1,741 | CellML structure clean |
| `tests/fixtures/cellml/two_time_variables.cellml` | 29 | 874 | CellML structure clean |
| `tests/fixtures/cellml/underscore_names.cellml` | 29 | 878 | CellML structure clean |
