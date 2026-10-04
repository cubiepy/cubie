# AGPL inventory of `structural/`

AGPL-derived code in `src/cubie/odesystems/symbolic/structural/`, the loss from its MIT equivalent, and the decision for each piece.

## How the code was classified

`structural/` was ported in #605 (2026-07-14) from the July 2026 masters of three Julia packages:

| Source | Licence |
|---|---|
| StateSelection.jl, including its `lib/ModelingToolkitTearing` subpackage | AGPL-3.0 since 2025-12-01; last MIT commit 74df007e |
| ModelingToolkit.jl (MTK) | MIT at every commit. c4177c335 (2025-12-02) is the last commit before its tearing and reassembly code moved into ModelingToolkitTearing |
| BipartiteGraphs.jl | MIT |

Code matching an MIT release is usable under MIT; code with no MIT counterpart came from the AGPL code.

## Files that are MIT-clean

| File | MIT source | Note |
|---|---|---|
| `bipartite.py` | BipartiteGraphs.jl | |
| `diffgraph.py`, `clil.py`, `pantelides.py`, `consistency.py`, `errors.py` | StateSelection.jl 74df007e | |
| `digraph.py` | BipartiteGraphs.jl; the incremental cycle tracker is from Graphs.jl | Graphs.jl is BSD-2-Clause and has no entry in `THIRD_PARTY_LICENSES` |
| `derivative_block.py` | cubie's own (#843) | |

## Files that contain AGPL-derived code

Each entry lists the AGPL-derived pieces; the rest of the file matches the MIT releases.

| File | AGPL-derived pieces | MIT-clean remainder |
|---|---|---|
| `tearing.py` | Carpanzano tearing; exact integer-linear SCC matching; Modia trying candidate variables in state-priority order; the full-matching repair after tearing (`update_full_var_eq_matching`) | Modia tearing core (from Modia.jl via MTK), `TearingResult`, `free_equations`, `contract_variables` |
| `singularity_removal.py` | pivot choice by state priority and fewest nonzeros; the row presort (`_MMSortKey`, `sort_mm_rows`); elimination split into groups of rows sharing no variable; the restricted pivot search used by exact SCC matching; `PivotInfo`; `get_new_mm` (rebasing the integer matrix after removals, with alias substitution) | tiered Bareiss elimination, `force_var_to_zero`, `IgnoreUnderconstrainedVariable`, `structural_singularity_removal` |
| `system_structure.py` | canonical variable ranks and the equation sort key; admitting any integer coefficient up to 127 into the integer subsystem (`as_small_int`, in `symbolics.py`); keeping the integer matrix in sync when Pantelides differentiates equations; `rm_eqs_vars` and its index-map helpers; trivial-tearing candidate selection (`possibly_explicit_equations`, `trivial_tearing_postprocess`) | equation/variable bookkeeping, incidence and solvability analysis, `_build_state_priorities` (MTK c4177c335), differentiation of equations and variables |
| `alias_elimination.py` | `trivial_tearing`'s two extra exclusions (never tear a variable with positive state priority; never tear an equation whose other variables include a derivative) and its integer-matrix update (unreachable from `structural_simplify`) | perfect-alias elimination and the alias-elimination driver (MTK, MIT, 2026 commits) |
| `dummy_derivatives.py` | breaking state-priority ties by canonical rank when choosing dummy derivatives; Carpanzano as the default tearing | dummy-derivative selection, partial state selection |
| `reassemble.py` | where the solve block for a lowered derivative (`D(x) = x_t`) is placed; the elimination of already-solved rows inside linear-SCC inlining | everything else, including the linear-SCC inlining option itself (MTK c4177c335) |
| `simplify.py` | writing exact-SCC-matching results back into the equations (`_apply_linear_rewrites`) | the pipeline (MTK `_mtkcompile!`) |
| `symbolics.py` | `as_small_int` | the other primitives (Symbolics.jl / MTK equivalents, MIT) and cubie's own `DerivativeRegistry` and `linear_dependencies` |

## What each AGPL piece costs

"MIT version" means replacing the piece with what the MIT release does. "Rewrite" means reimplementing the current behaviour from a requirement written without the AGPL code; that loses no behaviour if the rewrite matches, and costs the implementation work.

Measurements: each replacement was emulated on top of the decided design (Carpanzano removed, Modia tearing, exact SCC matching kept), then run through the full simulator test suite and on three DAE problems:
- NAND gate: Test Set; 14 node voltages, `C(y) y' = f(y, t)`.
- Ring modulator, index 2: Test Set II-3 with the four capacitors removed.
- Transistor amplifier: Test Set II-2; 8 node voltages, float32 radau_iia_5, compared with the Test Set reference at t = 0.2.

| Piece | MIT version | Loss with the MIT version | Decision / recommendation |
|---|---|---|---|
| Carpanzano tearing | Modia tearing | Different choice of which variables Newton iterates on; same number of them. All three problems solve as before. | Decided: drop Carpanzano |
| Exact integer-linear SCC matching | none | Blocks of integer-coefficient linear equations stay as equations Newton solves instead of being solved exactly. `dx + dy = -x; dy + dz = -y; 0 = x + y - z` becomes a DAE needing an implicit solver instead of an explicit ODE. | Decided: rewrite |
| Linear-SCC inlining | MTK c4177c335 version | Off by default and unused by the benchmarks | Decided: remove |
| Integer coefficients up to 127 in the integer subsystem | only ±1 coefficients | Relations with other integer coefficients lose exact checking. `dz = w; 0 = x + y + w; 0 = 2x + 2y - w; 0 = w^5 + w - z` (forces `w = 0`, leaves `x`, `y` undetermined) is accepted and fails at solve time instead of being rejected at construction. `dx = -3x; dy = -3y; 0 = x - y` (consistent, redundant) is rejected as having too many equations instead of being reduced to one state. NAND, ring modulator, amplifier: unchanged. | Rewrite (the requirement is one sentence) |
| Canonical variable ranks (tie-breaks between variables of equal state priority) | no ranks; ties go to variable order | The transistor amplifier integrates the diode currents instead of node voltages 2 and 5, so Newton must invert the exponential diode law for those voltages every step; every run fails (Newton divergence, step too small). NAND, ring modulator: unchanged. | Rewrite: a rank scheme designed from scratch (sort by name, then derivative order) gives the amplifier the same simplified system as the current ranks, which solves to the reference with max error 3.6e-4 |
| Equation sort key | sort equations by their printed form | NAND: equations come out in a different order, same states and residuals. Others unchanged. No test changes. | MIT version |
| Bareiss pivot choice, row presort, grouped elimination | first row with one nonzero, then two, then any; no presort; one elimination over all rows | Ring modulator: one constraint is kept in a different but equivalent form. Others unchanged. No test changes. | MIT version |
| Modia priority order and full-matching repair | MTK c4177c335 Modia | None measured. By reading: Modia no longer prefers keeping high-priority variables as Newton unknowns. | MIT version |
| Trivial-tearing exclusions | MTK c4177c335 `trivial_tearing!` | None measured. By reading: a variable with positive state priority that is defined by an explicit equation can be eliminated despite the priority. | MIT version, or rewrite the priority exclusion if state priority must always be honoured |
| Placement of the `D(x) = x_t` solve block | just before the block `D(x)` was in (MTK c4177c335) | None measured | MIT version |
| Integer-matrix rebasing with alias substitution (`get_new_mm`) | drop rows holding removed variables | None measured | MIT version |
| Integer-matrix sync during Pantelides | none | Required by exact SCC matching | Rewrite with exact SCC matching |
| Index-map helpers, `rm_eqs_vars` | MTK c4177c335 `alias_elimination!` / `trivial_tearing!` bookkeeping, BipartiteGraphs.jl `delete_srcs!` | Same behaviour | MIT version |
| `PivotInfo`, `return_pivots`, `trivial_tearing`'s integer-matrix branch | none | Unused by the pipeline | Decided: remove, with all other code nothing in `src` calls |

Test fallout of the MIT versions (beyond the tests that read Carpanzano's variable names): ±1-only coefficients fail `test_singular_integer_block_raises`, `test_alias_target_prefers_priority` and `test_exact_scc_matching_singular_warns`; removing canonical ranks fails `test_sum_of_parameters_coefficient_reduces_like_a_number` (the amplifier grows from 8 to 10 states). Every other MIT version passes the full simulator suite.

## Notices

`THIRD_PARTY_LICENSES` needs entries for StateSelection.jl (MIT, 74df007e), ModelingToolkit.jl (MIT), Modia.jl (MIT, via the Modia tearing core) and Graphs.jl (BSD-2-Clause, the incremental cycle tracker in `digraph.py`). BipartiteGraphs.jl is already listed.
