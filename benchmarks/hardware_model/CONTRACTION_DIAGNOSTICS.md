# Contraction-dependent full/rolled differences in three exact pairs

The complete full/local and stage-count-one/local outputs become
bitwise identical for RK23, Kvaerno3/LU and Radau3/BiCGSTAB on the frozen
FP32 Lorenz grid when contraction is disabled in both source generation
and final linking. These are independently verified functional
interventions, not default-compiler validation or global accuracy claims.

| Exact pair | Source contract=False, final FMA=True: failing elements / trajectories | Maximum absolute difference | Final FMA=False |
|---|---:|---:|---|
| RK23 | 122822 / 67806 | 4.57763671875e-5 | Bitwise equal; zero failures |
| Kvaerno3/LU | 236059 / 120994 | 8.678436279296875e-5 | Bitwise equal; zero failures |
| Radau3/BiCGSTAB | 357583 / 169486 | 1.6498565673828125e-4 | Bitwise equal; zero failures |

Every comparison retains the original atol=rtol=1e-6 gate. Each direct
true image first reproduces its own source-contract-disabled reference
twice. Each false image then runs twice with finite FP32 state, raw
SUCCESS zero and exact repeatability. The implicit references were
captured independently because the source flag can alter their outputs.
Their original fast-math-bank comparisons remain separate: Kvaerno3/LU
had235728 failing elements; Radau3/BiCGSTAB had357588. Neither original
failure is replaced by the intervention result.

The final links use one exact retained original LTOIR per source, with
the two final FMA settings. True images equal their captured source
cubins. Native FFMA counts are full/rolled44/20 for RK23,492/197 for
Kvaerno3/LU and6793/6787 for Radau3/BiCGSTAB; every false image has zero
FFMA. The earlier final-only flag experiment on original RK23 IR left
the cubins byte-identical, while the source flag alone retained its
numerical discrepancy. The two contraction controls are therefore
distinct in this installed compilation pipeline.

All direct launches use2048 blocks of(1,128,1),4 bytes dynamic shared
memory and262144 trajectories, exceeding two full occupancy waves for
each actual image. Native resources are validation observations only.
RK23 and Kvaerno3 register counts stay unchanged across their final-link
pair. Radau3 full changes127→128 registers and rolled142→147; residency
remains4 and3 blocks per SM respectively. Do not transfer an
unchanged-register statement to that pair. No deliberate timing bank was
collected. Original Solver verbose logs contain incidental timing values,
which are retained and excluded from prediction or performance analysis.

The narrow conclusion is that the observed discrepancy in these three
exact source/policy pairs depends on contraction-enabled code generation.
This intervention does not isolate a particular FFMA site or prove that
all native scheduling/forms besides FFMA are unchanged. It does not show
that unrolling is semantically incorrect, establish other workloads'
numerical behavior, approve a production flag change, fit any model
constant or admit the old numerically failing timing bank. Default-flag
heuristic recommendations still need their own unchanged numerical gate
and independent oracle evidence where required.

## Authoritative provenance

`R` denotes
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`; `V`
denotes its `verification` directory. External manifests remain
authoritative even when source files are mirrored into the research PR.

- RK23 source/reference experiment:
  `R/numerical_contraction_native_e2/receipt.json`.
- RK23 original-IR capture and links:
  `V/numerical_contract_ir_author_e2/receipt.json` and
  `V/numerical_contract_link_e2/receipt.json`.
- RK23 direct execution and independent result approval:
  `R/numerical_direct_native_e1/receipt.json` and
  `V/numerical_direct_native_independent_e1/receipt.json`:
  8 arrays,688 driver calls,632 parameter records.
- Implicit own references and independent approval:
  `R/numerical_implicit_capture_native_e1/receipt.json` and
  `V/numerical_implicit_capture_native_independent_e1/receipt.json`.
- Implicit singleton links:
  `R/numerical_implicit_link_e1/receipt.json`.
- Implicit direct execution and independent result approval:
  `R/numerical_implicit_direct_native_e1/receipt.json` and
  `V/numerical_implicit_direct_native_independent_e1/receipt.json`:
  16 arrays,1376 driver calls,1264 parameter records.
- The exact sources to mirror without editing are recorded in
  `V/contraction_diagnostic_source_manifest_e1.json`. Their captured
  paths, dependent wrapper and comparator, original plans and receipts
  must remain available; a renamed source copy is provenance and is not
  a replacement freeze or an independently rebound run request.
