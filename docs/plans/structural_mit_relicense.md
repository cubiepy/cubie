# Structural package MIT relicense plan

Goal: every file under `src/cubie/odesystems/symbolic/structural/` derives only from MIT- or BSD-licensed sources or from cubie's own design.

## Sources

`structural/` was ported in #605 from the July 2026 versions of these packages; code with no MIT counterpart is AGPL-derived.

| Source | Licence | Usable |
|---|---|---|
| `JuliaComputing/StateSelection.jl` | MIT up to 74df007e; AGPL-3.0 from e33acb22 (2025-12-01) | 74df007e and earlier |
| ModelingToolkitTearing (`lib/` of the StateSelection.jl repository, created 2025-12-04) | AGPL-3.0 | none |
| `SciML/ModelingToolkit.jl` (MTK) | MIT at every commit | any commit; c4177c335 (2025-12-02) is the last with tearing, reassembly and `trivial_tearing!` in-tree |
| `SciML/BipartiteGraphs.jl` | MIT | any commit |
| `ModiaSim/Modia.jl` (Modia tearing core, via MTK and StateSelection.jl) | MIT | any commit |
| `JuliaGraphs/Graphs.jl` (incremental cycle tracker) | BSD-2-Clause | any commit |

## Rules

- Ports come from the usable sources above. Rewrites are implemented from the requirements in this document, not from cubie's current code for that piece or any AGPL code.
- Do not open StateSelection.jl after 74df007e or ModelingToolkitTearing at any commit.
- Keep `structural_simplify(StructuralState) -> SimplifiedSystem` and the `SimplifiedSystem` contract.

## Inventory

MIT-clean and unchanged: `bipartite.py` (BipartiteGraphs.jl); `diffgraph.py`, `clil.py`, `pantelides.py`, `consistency.py`, `errors.py` (StateSelection.jl 74df007e); `digraph.py` (BipartiteGraphs.jl, Graphs.jl); `derivative_block.py` (cubie, #843).

AGPL-derived pieces in the other files; the rest of each file stays.

| File | AGPL-derived piece | Decision |
|---|---|---|
| `tearing.py` | Carpanzano tearing (`find_single_solvable_eq`, `carpanzano_tear_scc`, `CarpanzanoTearing`) | Remove (item 1) |
| `tearing.py` | exact integer-linear SCC matching (`exact_scc_matching`) | Rewrite (item 2) |
| `tearing.py` | Modia trying candidates in state-priority order; `update_full_var_eq_matching` | MIT version (item 4) |
| `singularity_removal.py` | `RestrictedBareissContext` | Rewrite with item 2 |
| `singularity_removal.py` | pivot choice by priority and fewest nonzeros (`find_first_linear_variable`, `BareissContext`, `do_bareiss`); row presort (`_MMSortKey`, `sort_mm_rows`); grouped elimination (`_uf_find`, `_uf_union`, grouping in `aag_bareiss`) | MIT version (item 5) |
| `singularity_removal.py` | `get_new_mm`, `_add_row_coeffs` | MIT behaviour, new implementation (item 6) |
| `singularity_removal.py` | `PivotInfo`, `return_pivots` | Remove (item 9) |
| `system_structure.py` | `_canonical_sort_key`, `_build_canonical_ranks` | Rewrite (item 3) |
| `system_structure.py` | equation sort key (`_equation_sort_key`, `_expression_sort_key`, `__expression_sort_key`, `_num_float`, `_ieee_pow`) | MIT version (item 5) |
| `system_structure.py`, `symbolics.py` | integer coefficients up to 127 in the integer subsystem (`as_small_int` in `find_eq_solvables`) | Rewrite (item 3) |
| `system_structure.py` | integer-matrix sync in `eq_derivative`; `_eq_derivative_mm` | Rewrite with item 2 |
| `system_structure.py` | `get_old_to_new_idxs`, `default_rm_eqs_vars`, `rm_eqs_vars`, `possibly_explicit_equations`, `trivial_tearing_postprocess` | MIT version (item 6) |
| `alias_elimination.py` | `trivial_tearing` exclusions (positive state priority; derivative neighbour) | MIT version (item 6) |
| `alias_elimination.py` | `trivial_tearing` integer-matrix branch | Remove (item 9) |
| `dummy_derivatives.py` | rank tie-break among equal state priorities | Keep, on item 3's ranks |
| `dummy_derivatives.py` | Carpanzano as default tearing | Remove (item 1) |
| `reassemble.py` | placement of the `D(x) = x_t` solve block | MIT version (item 5) |
| `reassemble.py` | linear-SCC inlining (`_get_linear_scc_linsol`) | Remove (item 8) |
| `simplify.py` | `_apply_linear_rewrites` | Rewrite with item 2 |

## Work items

### 1. Remove Carpanzano tearing

Delete `find_single_solvable_eq`, `carpanzano_tear_scc` and `CarpanzanoTearing`. Tearing after dummy-derivative selection is `ModiaTearing` (item 4) with item 2's exact matching tried on each SCC first; an SCC matched exactly is not torn further.

### 2. Rewrite exact integer-linear SCC matching

Delete `exact_scc_matching`, `RestrictedBareissContext`, `_eq_derivative_mm`, the integer-matrix code in `eq_derivative` and `_apply_linear_rewrites`. Requirements:
- Applies to a square SCC of n >= 2 equations that are all homogeneous integer-linear (`sum(c_i * v_i) = 0`), each with incidence equal to its integer-matrix row.
- Decide exactly, with fraction-free arithmetic, whether the SCC's coefficient block is nonsingular over the SCC's own variables.
- Nonsingular: match each equation to a distinct variable so the reduced equations solve their variables explicitly in sequence, preferring derivative variables; replace the equations, their integer-matrix rows, the incidence graph and the solvable graph with the reduced forms before reassembly.
- Singular: warn and leave the SCC to Modia tearing.
- When Pantelides differentiates an equation whose derivative is homogeneous integer-linear, the derivative's row is in the integer matrix.

### 3. Rewrite variable ranks and integer-coefficient admission

Delete `_canonical_sort_key`, `_build_canonical_ranks` and `as_small_int`'s use in `find_eq_solvables`. Requirements:
- Variable rank: position of the variable in the unknowns sorted by (base name, derivative order). A derivative variable added during reassembly takes the rank of the variable it replaces.
- Ranks break ties between equal state priorities in dummy-derivative selection and in alias-target choice (MTK `pick_alias_target`).
- An unknown enters an equation's integer-matrix row when its coefficient is an integer of magnitude at most 127.

### 4. Modia tearing from MTK c4177c335

Port `ModiaTearing`, `TearingResult` and `free_equations` from MTK c4177c335 (`tearing.jl`, `bipartite_tearing/modia_tearing.jl`), including its final tear of the free equations on overdetermined systems. Delete `update_full_var_eq_matching` and the state-priority ordering in `tear_equations`.

### 5. MIT versions of pivot choice, equation order and solve-block placement

- Bareiss (StateSelection.jl 74df007e `singularity_removal.jl`): one elimination over all integer-matrix rows in their stored order; pivot = first row with one nonzero, then two, then any, taking that row's first column allowed by the current tier. Delete `_MMSortKey`, `sort_mm_rows`, `_uf_find`, `_uf_union`, the grouping in `aag_bareiss` and the priority logic in `find_first_linear_variable`, `BareissContext` and `do_bareiss`.
- Equation order at `StructuralState` construction (MTK c4177c335 `TearingState`): sorted by printed form. Delete `_equation_sort_key`, `_expression_sort_key`, `__expression_sort_key`, `_num_float`, `_ieee_pow`.
- Solve-block placement (MTK c4177c335 `generate_derivative_variables!`): the singleton SCC solving `D(x) = x_t` goes immediately before the SCC `D(x)` was in.

### 6. MIT versions of index compaction, matrix rebasing and trivial tearing

- `rm_eqs_vars` and its index maps: port from the `old_to_new_eq` loop and graph rebuild in MTK c4177c335 `alias_elimination!` and BipartiteGraphs.jl `delete_srcs!`/`delete_dsts!`; it also renumbers `fullvars`, `state_priorities`, `canonical_ranks`, `always_present`, `eqs` and `original_eqs`.
- `get_new_mm(aliases, old_to_new_eq, old_to_new_var, mm)` (interface fixed by MTK's call sites): rows of removed equations drop; rows holding a removed variable drop; the rest are renumbered. `aliases` is unused.
- `trivial_tearing`: port MTK c4177c335 `trivial_tearing!`. Delete `possibly_explicit_equations` and `trivial_tearing_postprocess`.

### 7. Tests

- Delete the assertions that pin which variables Carpanzano picks, keeping the numerical checks: `test_conservative_excludes_nonunit_rows` (that `y` is observed in terms of `x`); `test_transistor_amplifier_init_and_reference` (the mass-flag tuple, the algebraic-set equality and the initial-derivative checks keyed by dummy-derivative names); `test_ring_modulator_index2_backwards_euler` reads its diode voltages from the saved states or observables, whichever holds them.
- Delete tests that call deleted functions: `test_exact_scc_matching_singular_warns`, `test_inline_linear_scc_solves_analytically`.

### 8. Remove linear-SCC inlining

Delete `inline_linear_sccs` and `analytical_linear_scc_limit` from `structural_simplify`, `default_reassemble` and `generate_system_equations`, `_get_linear_scc_linsol` and `symbolics.solve_linear_system`.

### 9. Remove unused code

Nothing in `src` calls these: `PivotInfo` and `return_pivots`; `trivial_tearing`'s `mm` parameter and branch; `tearing_with_dummy_derivatives`; `partial_state_selection_graph`, `_partial_state_selection_graph`, `_pss_graph_modia`, `_ascend_dg`, `_DiffData`; `BipartiteGraph.delete_dsts`; `SparseMatrixCLIL.getindex`; `DerivativeRegistry.register_known`; `SystemStructure.isalgvar`, `isdiffvar`, `algeqs`.

## Measured effect

Items 1 and 3 to 6 emulated together on current `main` (current exact matching for item 2), on the full simulator suite and three DAE problems:
- NAND gate: Test Set; 14 node voltages, `C(y) y' = f(y, t)`.
- Ring modulator, index 2: Test Set II-3 with the four capacitors removed.
- Transistor amplifier: Test Set II-2; 8 node voltages; float32 radau_iia_5, against the Test Set reference at t = 0.2.

Simulator suite: all tests pass except `test_conservative_excludes_nonunit_rows` and `test_ring_modulator_index2_backwards_euler`, whose pinned variable choices item 7 deletes.

| Problem | Effect |
|---|---|
| NAND gate | Same states and Newton unknowns; equations emitted in a different order |
| Ring modulator, index 2 | Newton iterates on `U6` and diode voltages 2-4 instead of `U5` and diode voltages 1-3; three residuals are current balances and one a diode-voltage relation, with diode currents computed directly from the exponential law |
| Transistor amplifier | Newton iterates on `y1_t, y4_t, y7_t` instead of `y2_t, y5_t, y7_t`; max error against the reference 3.6e-4 (current 8.2e-4) |

Float32 solves of 1024 trajectories (first 1024 points of each benchmark's parameter sweep) on an RTX 4070 SUPER:

| Problem and solver | Current | This plan |
|---|---|---|
| NAND gate, crank_nicolson, PI, tolerance 1e-2 | 0.20% failed | 0.10% failed |
| NAND gate, crank_nicolson, PI, tolerance 1e-3 | 6.15% failed | 5.57% failed |
| Ring modulator index 2, backwards_euler, fixed 1e-7 s | 100% failed (Newton) | 100% failed (Newton) |
| Ring modulator index 2, radau_iia_5, fixed 1e-7 s | 100% failed (Newton) | 100% failed (Newton) |

The index-2 ring modulator fails every float32 trajectory on current `main` (also radau_iia_5 and rodas3p adaptive at 1e-4 and 1e-6).

What the rewrites in items 2 and 3 keep, measured by dropping each instead:

| Dropped | Loss |
|---|---|
| Exact matching (item 2) | `dx + dy = -x; dy + dz = -y; 0 = x + y - z` becomes a DAE needing an implicit solver instead of an explicit ODE |
| Ranks (item 3) | The transistor amplifier integrates the diode currents instead of node voltages 2 and 5; every solve fails (Newton divergence, step too small) |
| Coefficients up to 127 (item 3), with ±1 only | `dz = w; 0 = x + y + w; 0 = 2x + 2y - w; 0 = w^5 + w - z` is accepted and fails at solve time instead of being rejected at construction; `dx = -3x; dy = -3y; 0 = x - y` is rejected as having too many equations instead of reduced to one state |

## Notices

- `THIRD_PARTY_LICENSES` gains StateSelection.jl (MIT, Copyright (c) JuliaHub, Inc. and other contributors), ModelingToolkit.jl (MIT, Copyright (c) 2018-2026 Yingbo Ma, Christopher Rackauckas, Julia Computing, and contributors), Modia.jl (MIT, Copyright (c) 2017-2018 ModiaSim developers) and Graphs.jl (BSD-2-Clause, Copyright (c) 2015 Seth Bromberger and other contributors).
- Module docstrings and `structural/AGENTS.md` name only the sources above and the Pantelides and Mattsson-Söderlind papers.

## Sequence

1. Items 8 and 9.
2. Items 1 and 4.
3. Items 5 and 6.
4. Items 2 and 3.
5. Item 7, then the structural tests, then the full simulator and real-GPU suites.

## Done criteria

- Every function in `structural/` traces to a usable source above, cubie's own history, or a requirement in this document.
- Simulator and real-GPU suites pass.
- Notices are in place.
