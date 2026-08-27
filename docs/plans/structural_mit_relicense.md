# Structural package MIT relicense plan

Goal: every file under `src/cubie/odesystems/symbolic/structural/` derives only from MIT-licensed sources or from cubie's own design, so cubie ships under MIT.

## Provenance facts

- `structural/` (PR #605, merged 2026-07-14) was ported from `JuliaComputing/StateSelection.jl` at 0add0a8 (2026-07-13) and its subpackage `lib/ModelingToolkitTearing`. That repository is AGPL-3.0 since e33acb22 (2025-12-01).
- MIT baselines: StateSelection.jl at 74df007e (2025-12-01, Copyright JuliaHub, Inc.); ModelingToolkit.jl at c4177c335 (2025-12-02, last commit before it depended on StateSelection.jl; the MTK repository is MIT at every date); BipartiteGraphs.jl (MIT).
- Every construct listed under "Rewrite" below was first committed upstream between 2026-03-02 and 2026-07-07 by JuliaHub staff and exists in no MIT tree.

## Rules for the rewrite

- Work only from the MIT baselines above, the Carpanzano (2000) paper, and cubie's own specifications in this document. Do not open the AGPL tree.
- Each rewritten function is specified here by behaviour; implement from the specification.
- Keep the public entry point `structural_simplify(StructuralState) -> SimplifiedSystem` and the `SimplifiedSystem` contract unchanged.
- Determinism (results independent of declaration and equation order) is preserved by cubie's own ordering scheme (item 6), not by upstream tie-break code.

## Work items

### 1. Carpanzano tearing (`tearing.py:511-807`, `dummy_derivatives.py:343-400`)

Delete `find_single_solvable_eq`, `carpanzano_tear_scc`, `CarpanzanoTearing`. Reimplement from Carpanzano, E. (2000), "Order Reduction of General Nonlinear DAE Systems by Automatic Tearing", Math. Comput. Model. Dyn. Syst. 6(2):145-168, algorithm A1 with the §4.3 tear-variable heuristics.

Specification:
- Input: one SCC (variables, equations), the incidence graph, the solvable-edge subgraph, the matching (all SCC variables unassigned on entry).
- Loop while unmatched solvable variables remain: match any equation incident on exactly one unmatched variable via a solvable edge, derivative variables first; otherwise choose a tear variable by the paper's heuristics (minimum-incidence equations, then a variable present but not solvable there; else maximum incidence, fewest solvable equations), ties broken by item 6 ranks; remove it from the candidate set.
- Output: matching with solved variables assigned and tear variables unassigned.
- Default tearing after dummy-derivative selection stays Carpanzano.

### 2. Exact integer-linear SCC matching (`tearing.py:409-508`, `singularity_removal.py:183-235`, `system_structure.py:563-700`)

Delete `exact_scc_matching`, `RestrictedBareissContext`, `_eq_derivative_mm`. Reimplement:
- Square SCC (n >= 2) whose equations are all rows of the integer-linear matrix `mm`, each row's columns equal to the equation's incidence: run fraction-free Bareiss (existing MIT `clil.bareiss`) on those rows with pivot columns restricted to the SCC's variables, derivative variables preferred; no unrestricted tier.
- Full rank: replace each equation's row in `mm`, the incidence graph and the solvable graph by its reduced row; match each equation to its pivot; record `(equation, columns, coefficients)` rewrites for `reassemble` to apply to the symbolic equations.
- Rank deficient: warn and fall back to item 1.
- `StructuralState.eq_derivative` keeps `mm` in sync when Pantelides differentiates an integer-linear equation: the differentiated row has the same coefficients on each variable's derivative column; a variable without a derivative column drops the row from `mm`.

### 3. Bareiss pivot policy (`singularity_removal.py:40-125, 130-180, 279-420, 442-460`)

Delete `_MMSortKey`, `sort_mm_rows`, `_uf_find`, `_uf_union`, `PivotInfo`, the `var_priorities`/`valid_pivot_mask` logic in `find_first_linear_variable` and `BareissContext`, and the connected-component partition in `aag_bareiss`. Reimplement:
- Pivot search per elimination step: among rows not yet used, candidate columns filtered by the current tier mask and by "column not already pivoted"; choose the row with the fewest nonzeros, then the column with the lowest item-6 rank.
- Rows are processed in the order (nonzero count, item-6 ranks of columns, coefficients, equation index) before elimination.
- Rows sharing no column are eliminated as separate groups, groups ordered by their smallest row index.
- Return values: `(rank1, rank2, rank3, pivots)` as today; `structural_singularity_removal` keeps its `return_pivots` form.

### 4. Integer-matrix rebasing (`singularity_removal.py:502-587`, `alias_elimination.py:480-527`, `simplify.py:320-323`)

Delete `get_new_mm` and `_add_row_coeffs`. Reimplement one routine: given `mm`, an old-to-new equation map, an old-to-new variable map, and `aliases` (removed variable -> surviving variable, or -> integer linear combination of surviving variables), produce the rebased `mm`: rows of removed equations drop; each removed variable in a kept row is substituted by its alias; a kept row containing a removed variable with no alias drops; columns are merged and sorted, zero coefficients removed. The `trivial_tearing` alias update builds `aliases` from each torn equation's own row (coefficient of the torn variable is -1), substituting earlier aliases.

### 5. Index compaction and interface split (`system_structure.py:238-335, 849-912`)

Delete `get_old_to_new_idxs`, `default_rm_eqs_vars`, `rm_eqs_vars`, `possibly_explicit_equations`, `trivial_tearing_postprocess`. Reimplement: removing sorted equation and variable index lists from a `SystemStructure` renumbers `graph`, `solvable_graph`, `var_to_diff`, `eq_to_diff`, `fullvars`, `state_priorities`, `canonical_ranks` and the equation list, and returns the two old-to-new maps (`-1` for removed). `trivial_tearing` finds its candidates (equations of the form `var = expr` with `var` an algebraic unknown that appears in no other untorn equation) directly.

### 6. Deterministic ordering (`system_structure.py:337-360, 427-475, 504-512, 913-1059`; consumers `dummy_derivatives.py:158`, `alias_elimination.py:152`, `tearing.py:565`, `reassemble.py:342-343, 970-974`)

Delete `_canonical_sort_key`, `_build_canonical_ranks`, `_num_float`, `_ieee_pow`, `_expression_sort_key`, `__expression_sort_key`, `_equation_sort_key`. Reimplement:
- Variable rank: position in the list of `fullvars` sorted by (base symbol name, derivative order); a derivative variable created later takes its base's rank plus its order.
- Equation order at `StructuralState` construction: sort by the tuple of sorted incident variable ranks, then by the tuple of integer coefficients on those variables (0 for non-linear incidence), then by original index.
- All tie-breaks (alias target choice, dummy-derivative column order, tear-variable choice) use the variable rank.

### 7. Full-matching consistency and Modia priority order (`tearing.py:78-104, 234-235, 265-307, 319-322, 362, 377-406`)

Delete `update_full_var_eq_matching` and rewrite: after tearing an SCC, copy the torn assignments into `full_var_eq_matching`; match each still-unassigned SCC variable to an unmatched SCC equation incident on it, else to the first remaining equation. Rewrite `TearingResult` and `ModiaTearing` in place. Modia keeps trying candidate variables lowest state priority first.

### 8. Reassembly (`reassemble.py:296-325, 735-896`; `simplify.py` options)

- Dummy-derivative singleton SCC insertion: the singleton SCC solving `D(x) = x_t` is placed before the earlier of (its original position, the SCC containing `D(D(x))`); at equal positions, longer derivative chains first. Rewrite from this rule.
- Remove `inline_linear_sccs` and `analytical_linear_scc_limit` from `structural_simplify`, `default_reassemble` and `generate_system_equations`; delete `_get_linear_scc_linsol`. Nothing in `src/` enables the option.

## Unchanged (MIT ancestry confirmed)

`bipartite.py`, `digraph.py`, `diffgraph.py`, `clil.py`, `pantelides.py`, `consistency.py`, `symbolics.py`, `errors.py`; `alias_elimination.py` except item 4; `dummy_derivatives.py` except items 1 and 6; `reassemble.py` except item 8; `simplify.py` pipeline; Modia core of `tearing.py`; `aag_bareiss`, `do_bareiss`, `structural_singularity_removal`, `force_var_to_zero`, `find_linear_variables`, `IgnoreUnderconstrainedVariable`; incidence, solvability, derivative-graph and `_build_state_priorities` code in `system_structure.py`.

## Sequence

1. Items 5, 6, 7 (mechanical; no algorithm change).
2. Items 3 and 4.
3. Item 2, then item 1.
4. Item 8.
5. Run `tests/odesystems/symbolic/structural/`, `tests/odesystems/symbolic/test_dae_parser.py`, the DAE solve tests, then the full simulator and real-GPU suites.

## Documentation and notices

- Module docstrings name only the MIT sources: StateSelection.jl (MIT era), ModelingToolkit.jl, BipartiteGraphs.jl, the Pantelides / Mattsson-Söderlind / Carpanzano papers.
- `THIRD_PARTY_LICENSES` gains: StateSelection.jl, MIT, Copyright (c) JuliaHub, Inc. and other contributors; ModelingToolkit.jl, MIT, Copyright (c) 2018-2026 Yingbo Ma, Christopher Rackauckas, Julia Computing, and contributors.
- `structural/AGENTS.md` provenance line lists the same sources.

## Done criteria

- No function in `structural/` corresponds to code first published after 2025-12-01 in `JuliaComputing/StateSelection.jl`.
- Structural, DAE, simulator and real-GPU suites pass.
- Notices above are in place.
