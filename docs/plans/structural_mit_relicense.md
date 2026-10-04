# Structural package MIT relicense plan

Goal: every file under `src/cubie/odesystems/symbolic/structural/` derives only from MIT-licensed sources or from cubie's own design, so cubie ships under MIT.

Line references are against `main` at e59cab6c.

## Sources

| Source | Licence | Usable range |
|---|---|---|
| `JuliaComputing/StateSelection.jl` | MIT up to 74df007e (2025-12-01); AGPL-3.0 from e33acb22 (same day) | 74df007e and earlier |
| `ModelingToolkitTearing` (`lib/` of the StateSelection.jl repository, created 8588d0c 2025-12-04) | AGPL-3.0 | none |
| `SciML/ModelingToolkit.jl` | MIT at every commit, including HEAD | any commit. c4177c335 (2025-12-02) is the last commit carrying `src/structural_transformation/` tearing, reassembly and `trivial_tearing!`; perfect-alias elimination (`eliminate_perfect_aliases!`, `union_with_sign!`, `pick_alias_target`, `alias_elimination!`) is in `src/systems/alias_elimination.jl` at later commits (c49d619f6 2026-03-20 onward) |
| `SciML/BipartiteGraphs.jl` | MIT | any commit |
| `ModiaSim/Modia.jl` | MIT (Copyright (c) 2017-2018 ModiaSim developers) | the Modia tearing core, as carried in StateSelection.jl 74df007e `src/modia_tearing.jl` |
| Carpanzano, E. (2000), "Order Reduction of General Nonlinear DAE Systems by Automatic Tearing", Math. Comput. Model. Dyn. Syst. 6(2):145-168 | paper | algorithm and heuristics |

MTK code after c4177c335 calls `StateSelection.rm_eqs_vars!`, `StateSelection.get_new_mm` and `StateSelection.trivial_tearing!`; the call sites (arguments, return values, order of calls in `_mtkcompile!`) are MIT, the implementations are AGPL.

## Rules for the rewrite

- Work only from the sources above, cubie's own history, and the specifications in this document. Do not open StateSelection.jl after 74df007e or ModelingToolkitTearing at any commit.
- Each rewritten function is specified here by behaviour; implement from the specification or the named MIT source.
- Keep the public entry point `structural_simplify(StructuralState) -> SimplifiedSystem` and the `SimplifiedSystem` contract unchanged.
- Determinism (results independent of declaration and equation order) is preserved by cubie's own ordering scheme (item 6).

## Work items

### 1. Carpanzano tearing (`tearing.py:511-806`, `dummy_derivatives.py:337-380`)

No MIT ancestor. Delete `find_single_solvable_eq`, `carpanzano_tear_scc`, `CarpanzanoTearing`. Reimplement from Carpanzano (2000).

Specification:
- Input: one SCC (variables, equations), the incidence graph, the solvable-edge subgraph, the matching (all SCC variables unassigned on entry).
- Variables solvable from no equation of the SCC leave the candidate set first.
- Loop while candidates remain: match any equation incident on exactly one candidate via a solvable edge, derivative variables first; otherwise choose a tear variable: among the minimum-incidence equations, a candidate present but not solvable there; else the candidate of maximum incidence, then fewest solvable equations; ties broken by item-6 rank. Remove the chosen variable from the candidates.
- Output: matching with solved variables assigned and tear variables unassigned.
- Default tearing after dummy-derivative selection stays Carpanzano.

### 2. Exact integer-linear SCC matching (`tearing.py:409-508`, `singularity_removal.py:183-233`, `system_structure.py:563-677`, `simplify.py:194-205`)

No MIT ancestor; MIT `eq_derivative!` (MTK c4177c335 `symbolics_tearing.jl`) does not maintain `mm`. Delete `exact_scc_matching`, `RestrictedBareissContext`, `_eq_derivative_mm`, the `mm` append in `eq_derivative` (`system_structure.py:617-627`) and `_apply_linear_rewrites`. Reimplement:
- Square SCC (n >= 2) whose equations are all rows of `mm`, each row's columns equal to the equation's incidence: fraction-free Bareiss (`clil.bareiss`) on those rows with pivot columns restricted to the SCC's variables, derivative variables preferred; no unrestricted tier.
- Full rank: replace each equation's row in `mm`, the incidence graph and the solvable graph by its reduced row; match each equation to its pivot; return `(equation, columns, coefficients)` rewrites, which `structural_simplify` writes into `eqs` and `original_eqs` as `0 ~ sum(coefficient * variable)` before reassembly.
- Rank deficient: warn and fall back to item 1.
- Pantelides differentiation keeps `mm` in sync: a differentiated `mm` row has the same coefficients on each variable's derivative column; a differentiated equation that is not an `mm` row but whose derivative is integer-linear in its incidence is appended to `mm`. A row with a variable that has no derivative column raises.

### 3. Bareiss pivot policy (`singularity_removal.py:47-127, 130-180, 279-402, 442-453`)

The MIT baseline (StateSelection.jl 74df007e `find_masked_pivot`) takes the first row with one nonzero, then two, then any, and the first masked column of that row. Delete `_MMSortKey`, `sort_mm_rows`, `_uf_find`, `_uf_union`, `PivotInfo`, the `var_priorities` logic in `find_first_linear_variable`, `BareissContext` and `do_bareiss`, and the connected-component partition in `aag_bareiss`. Reimplement:
- Pivot search per elimination step: among rows not yet used, candidate columns filtered by the current tier mask; choose the row with the fewest nonzeros, then the column with the lowest item-6 rank.
- Rows are processed in the order (nonzero count, item-6 ranks of columns, coefficients, equation index) before elimination.
- Rows sharing no column are eliminated as separate groups, groups ordered by their smallest row index; tier ranks and pivots are concatenated tier by tier across groups.
- Return values: `(rank1, rank2, rank3, pivots)`; `structural_singularity_removal` keeps its `return_pivots` form.

### 4. Integer-matrix rebasing (`singularity_removal.py:502-587`, `alias_elimination.py:480-527`)

The interface `get_new_mm(aliases, old_to_new_eq, old_to_new_var, mm)` is fixed by MIT MTK call sites; the implementation has no MIT ancestor. Delete `get_new_mm` and `_add_row_coeffs`. Reimplement one routine: given `mm`, an old-to-new equation map, an old-to-new variable map, and `aliases` (removed variable -> surviving variable, or -> integer linear combination of surviving variables), produce the rebased `mm`: rows of removed equations drop; each removed variable in a kept row is substituted by its alias; a kept row containing a removed variable with no alias drops; columns are merged and sorted, zero coefficients removed.

`trivial_tearing`'s `mm` branch (`alias_elimination.py:480-527`) is unreachable: `structural_simplify` calls `trivial_tearing(state)` without `mm` (`simplify.py:337`), as MTK's `_mtkcompile!` does. Delete the branch and the `mm` parameter.

### 5. Index compaction and trivial-tearing candidates (`system_structure.py:238-335, 831-900`, `alias_elimination.py:398-478`)

`get_old_to_new_idxs` and the graph rebuild in `default_rm_eqs_vars` have MIT ancestors: the `old_to_new_eq` loop and `set_neighbors!` rebuild in MTK c4177c335 `alias_elimination!`, and `delete_srcs!`/`delete_dsts!` in BipartiteGraphs.jl. Port them from those. `rm_eqs_vars` (renumbering `fullvars`, `state_priorities`, `canonical_ranks`, `always_present`, `eqs`, `original_eqs`, returning both maps with `-1` for removed entries) is reimplemented from this specification.

`trivial_tearing` is ported from MTK c4177c335 `trivial_tearing!` (MIT): candidate `var ~ expr` equations whose LHS is an unknown that is not irreducible, not differentiated and not a derivative, is not on its own RHS, appears in no other untorn equation, and whose every other variable appears in another untorn equation; repeat to a fixed point; torn equations become observed. Two conditions in cubie have no MIT ancestor and are kept as cubie's own: a variable with positive state priority is never torn, and an equation is not torn when another of its variables is a derivative. Delete `possibly_explicit_equations` and `trivial_tearing_postprocess`.

### 6. Deterministic ordering (`system_structure.py:337-343, 427-475, 504-512, 913-1059`; consumers `dummy_derivatives.py:158`, `alias_elimination.py:152`, `tearing.py:565`, `reassemble.py:342-343, 970-974`)

MIT MTK consumes `canonical_ranks` (`pick_alias_target`); its construction is AGPL. Delete `_canonical_sort_key`, `_build_canonical_ranks`, `_num_float`, `_ieee_pow`, `_expression_sort_key`, `__expression_sort_key`, `_equation_sort_key`. Reimplement:
- Variable rank: position in the list of `fullvars` sorted by (base symbol name, derivative order); a derivative variable created later takes its base's rank plus its order.
- Equation order at `StructuralState` construction: sort by the tuple of sorted incident variable ranks, then by the tuple of integer coefficients on those variables (0 for non-linear incidence), then by original index.
- All tie-breaks (alias target choice, dummy-derivative column order, tear-variable choice, pivot choice) use the variable rank.

### 7. Full-matching consistency and Modia priority order (`tearing.py:234-235, 265-307, 319-322, 362`)

`TearingResult`, `ModiaTearing`, `free_equations` and the overdetermined-system handling are in MTK c4177c335 (`tearing.jl`, `bipartite_tearing/modia_tearing.jl`, MIT); port them from there, including its final tear of the free equations on overdetermined systems. Delete `update_full_var_eq_matching` and rewrite: after tearing an SCC, copy the torn assignments into `full_var_eq_matching`; match each still-unassigned SCC variable to an unmatched SCC equation incident on it, else to the first remaining equation. Modia tries candidate variables lowest state priority first (cubie's own).

### 8. Reassembly (`reassemble.py:296-329, 548-726, 727-898`)

- Dummy-derivative singleton SCC insertion: MTK c4177c335 inserts the singleton SCC solving `D(x) = x_t` immediately before the SCC `D(x)` was in. cubie's rule: before the earlier of that SCC and the SCC containing `D(D(x))`; at equal positions, longer derivative chains first. Rewrite from this rule.
- Linear-SCC inlining: `inline_linear_sccs`, `analytical_linear_scc_limit`, `find_alg_eqs_vars`, `is_linear_scc` and `get_linear_scc_linsol` are in MTK c4177c335 `symbolics_tearing.jl` (MIT). Port the option from there. cubie's elimination of torn rows inside the SCC (`reassemble.py:779-859`) has no MIT ancestor; reimplement it from this specification: each SCC equation already matched to an SCC variable defines that variable as a linear combination of the others; substitute those definitions into the remaining rows and solve the reduced system when its size is at most `analytical_linear_scc_limit`, subject to the `allow_symbolic`/`allow_parameter` division policy.

## Unchanged

Ancestry below was checked by the presence of the corresponding functions in the named source, not line by line.

- MIT ancestry: `bipartite.py`, `digraph.py` (BipartiteGraphs.jl); `diffgraph.py`, `clil.py`, `pantelides.py`, `consistency.py`, `errors.py` (StateSelection.jl 74df007e); `reassemble.py` outside item 8 (MTK c4177c335 `symbolics_tearing.jl`); `alias_elimination.py` perfect-alias elimination and the `alias_elimination` driver (MTK `src/systems/alias_elimination.jl` after c49d619f6); `dummy_derivatives.py` outside items 1 and 6 (StateSelection.jl 74df007e `partial_state_selection.jl`, MTK c4177c335); `simplify.py` pipeline order (MTK `_mtkcompile!`); `contract_variables`, the Modia core of `tearing.py`; `aag_bareiss` tiers, `force_var_to_zero`, `find_linear_variables`, `IgnoreUnderconstrainedVariable`, `structural_singularity_removal`.
- cubie's own: `derivative_block.py` and `symbolics.linear_dependencies` (#843); `symbolics.py` (listed unchanged by the original plan; not re-checked); `DerivativeRegistry`; incidence, solvability, derivative-graph and `_build_state_priorities` code in `system_structure.py`.

## Functionality loss

Measured by running `structural_simplify` and the structural test directory with each item replaced in-process by an emulation of its specification (items 1, 3, 6), or removed (items 1, 2):

| Change | Structural tests (real GPU, 129) | Behaviour change |
|---|---|---|
| Items 1, 3, 6 as specified above | 129 pass | Item 3 changes which integer-linear combination a reduced row keeps (ring modulator index 2, below); none observed elsewhere |
| Item 1 not reimplemented (Modia in its place) | 4 fail: `test_conservative_excludes_nonunit_rows`, `test_integer_constraint_block_solves_explicitly`, `test_transistor_amplifier_init_and_reference`, `test_ring_modulator_index2_backwards_euler` | Different torn variables; the amplifier selects different algebraic states (its mass flags differ) |
| Item 2 not reimplemented | 2 fail: `test_exact_scc_matching_singular_warns`, `test_integer_constraint_block_solves_explicitly` | Integer-linear SCCs (e.g. `u + 2v = x; 3u - v = 1`) stay as residual rows instead of solving explicitly |
| Item 8 inlining deleted instead of ported | `test_inline_linear_scc_solves_analytically` | `structural_simplify(..., inline_linear_sccs=True)` no longer exists |

Without item 6's spec the ordering is still order-independent on the systems below: 12 shuffles of equations and unknowns gave one result each under current and specified ordering.

## Measured effect on the GPUODEBenchmarks DAE problems

NAND gate (14 states, `c9` swept) and ring modulator with `Cs = 0` (index 2, `Uin1_amplitude` swept), as defined in `GPUODEBenchmarks/runner_scripts/cubie_systems.py`.

| | NAND gate | Ring modulator, index 2 |
|---|---|---|
| Current result | 14 differential, 8 algebraic (`y3_t`, `y4_t`, `y5_t`, `y8_t`, `y9_t`, `y10_t`, `y13_t`, `y14_t`), 33 observed | 10 differential, 4 algebraic (`U5`, `UD1`, `UD2`, `UD3`), 12 observed |
| Carpanzano SCCs (variables, torn) | (3, 3), (5, 3) | (8, 3), (8, 1) |
| Exact SCC matching | never eligible (no integer-linear SCC) | never eligible |
| Items 1, 3, 6 as specified | identical system | same states; one residual row changes form (`-U5 - U6 - UD1 - UD4` becomes `U6 - U4 - U7 - UD2 - Uin2`; residual operations 25 -> 26), `U4`/`U6` observed through the other constraint |
| Item 1 replaced by Modia | identical system | algebraic `U6`, `UD2`, `UD3`, `UD4`; residuals become the linear current balances with the diode exponentials observed (residual operations 25 -> 18) |
| Item 2 removed | identical system | identical system |
| Linear-SCC inlining enabled (off by default and in the benchmarks) | 6 algebraic | 3 algebraic |

Float32 solves of 1024 trajectories on an RTX 4070 SUPER:

| Configuration | Current | Items 1, 3, 6 as specified | Modia | Item 2 removed |
|---|---|---|---|---|
| NAND, crank_nicolson, PI, tol 1e-2 | 0.20% failed | 0.20% | 0.20% | 0.20% |
| NAND, crank_nicolson, PI, tol 1e-3 | 6.15% failed | 6.15% | 6.15% | 6.15% |
| Ring index 2, backwards_euler, fixed 1e-7 s | 100% failed (Newton) | 100% | 100% | 100% |
| Ring index 2, radau_iia_5, fixed 1e-7 s | 100% failed (Newton) | 100% | 100% | 100% |

The NAND gate reaches the same simplified system under every variant, so its kernels and results are unchanged. The ring modulator in index-2 form fails every float32 trajectory on current `main` (radau_iia_5 and rodas3p adaptive at 1e-4 and 1e-6 as well), so the rollback cannot change its solve outcome in the benchmark; `test_ring_modulator_index2_backwards_euler` (float64, 2 us) passes on real GPU under the specified rewrite.

## Sequence

1. Items 5, 6, 7 (ports and specifications; no algorithm change).
2. Items 3 and 4.
3. Item 2, then item 1.
4. Item 8.
5. Run `tests/odesystems/symbolic/structural/`, then the full simulator and real-GPU suites.

## Documentation and notices

- Module docstrings name only the MIT sources and the Pantelides / Mattsson-Söderlind / Carpanzano papers; `tearing.py`'s docstring drops `carpanzano_tearing.jl`, and `structural/AGENTS.md` drops ModelingToolkitTearing.
- `THIRD_PARTY_LICENSES` gains: StateSelection.jl, MIT, Copyright (c) JuliaHub, Inc. and other contributors; ModelingToolkit.jl, MIT, Copyright (c) 2018-2026 Yingbo Ma, Christopher Rackauckas, Julia Computing, and contributors; Modia.jl, MIT, Copyright (c) 2017-2018 ModiaSim developers (Modia tearing in `tearing.py`).

## Done criteria

- Every function in `structural/` traces to an MIT source above, a cited paper, cubie's own history, or a specification in this document.
- Structural, simulator and real-GPU suites pass.
- Notices above are in place.
