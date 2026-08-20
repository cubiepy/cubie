# Documentation review artifacts

## Review state

The review covers every tracked repository entry. Existing documentation,
source, tests, and configuration remain unchanged. The files in this
directory contain inventories and proposed text for later approval.

The documentation corpus contains 149 reStructuredText pages. Every page was
read and checked against current source, tests, exports, configuration, and
runtime-resolved autodoc targets where dependencies allowed import.

The repository manifest contains 585 tracked entries. The assigned
documentation, package, test, instruction, and mirror lanes cover 524 entries.
The repository-support lane is the exact 61-entry complement. One NPZ file is
binary and is inventoried as data. Generated data, schemas, and fixtures were
traversed and checked structurally.

## Artifact map

| Review lane | Inventory | Proposed rewrites | Coverage receipt |
|---|---|---|---|
| User-facing pages and support prose | [Inventory](user_facing_inventory.md) | [Proposals](user_facing_rewrite_proposals.md) | 42 files and 6,973 lines |
| Developer, batch, and integrator API pages | [Inventory](batch_integrator_inventory.md) | [Proposals](batch_integrator_rewrite_proposals.md) | 69 pages, 1,917 RST lines, and 3,396 rendered docstring lines |
| Memory, ODE-system, output, and GUI pages | [Inventory](systems_output_inventory.md) | [Proposals](systems_output_rewrite_proposals.md) | 51 pages, 695 RST lines, 39,147 source lines, 21,456 test lines, and 1,068 lines in ten nested instruction files |
| Root and batch source and tests | [Inventory](root_batch_source_inventory.md) | [Proposals](root_batch_supplemental_proposals.md) | 70 files and 45,405 lines |
| Integrator source and tests | [Inventory](integrator_source_inventory.md) | [Proposals](integrator_supplemental_proposals.md) | 115 entries and 34,556 newly read text lines |
| Repository support files | [Inventory](repository_support_inventory.md) | [Proposals](repository_support_supplemental_proposals.md) | 61 files and 65,778 lines |
| Combined result | [Completeness and freshness ledger](completeness_freshness_inventory.md) | Use the six proposal files above | 585 tracked entries and 149 documentation pages |

Lane line totals are not additive. Some source and instruction files were read
by more than one lane as evidence. The manifest reconciliation uses unique
tracked paths.

## Proposal use

The proposal files contain exact replacements with source evidence. They do
not authorize changes to the reviewed files. Apply factual corrections before
language-only rewrites so later wording is based on current behavior.

`CHANGELOG.md` was read and inventoried. Historical punctuation has no rewrite
proposal because project instructions prohibit editing that file.
