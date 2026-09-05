# RK23 contraction diagnostic

For the frozen Lorenz RK23 full/local and stage-count-one/local pair,
removing native fused multiply-add contraction makes the complete FP32
outputs bitwise identical. The original comparison failure remains
preserved; this result applies to a separately compiled arithmetic
intervention, not a repair of the original validation bank.

| Source contract | Final link FMA | Full/rolled native FFMA sites | Failing elements | Failing trajectories | Maximum absolute difference |
|---|---|---:|---:|---:|---:|
| False | True | 44 / 20 | 122822 | 67806 | 4.57763671875e-5 |
| False | False | 0 / 0 | 0 | 0 | 0 |

The second row is bitwise equality, stronger than merely passing the
unchanged `atol=rtol=1e-6` comparison. All eight complete state/status
snapshots are finite FP32 with raw SUCCESS status zero. Each image's two
executions repeat exactly. The true baselines reproduce the previously
retained own outputs byte for byte before the false interventions run.

The batch has 262144 runs with 2048 blocks of (1,128,1) and 4 bytes dynamic
shared memory. The direct driver reports 12 resident blocks per SM for
both full images and 10 for both rolled images on 56 SMs, satisfying the
two-wave requirement for every image. Native resources are unchanged
within each true/false source pair: full 40 registers and 0 local bytes;
rolled 44 registers and 40 local bytes. The rolled frame is addressable
local storage; this is not evidence of an allocator-spill intervention.
These are validation observations, never predictor inputs or fitted
coefficients. No timing was measured.

## What this establishes

The observed cross-policy numerical discrepancy depends on contraction-
enabled code generation for this exact RK23 workload. Changing the final
link option on the same source-contract-disabled IR eliminates native
FFMA in both images and eliminates their output discrepancy. Other link
options, source inputs, constants, algorithm settings, memory layout,
duration, geometry and comparison tolerances remain bound to their
preserved records. This is a controlled compiler/arithmetic intervention.

It does not identify a particular FFMA site as the sole source of the
difference. The compiler may also change scheduling and other native
forms when contraction is disabled. It does not prove global numerical
accuracy, show that unrolling is semantically wrong, or establish the
cause of any other workload's failed comparison. It does not recommend
changing production flags or reinterpret the original timing bank as
passing its numerical gate.

The public source flag alone was insufficient in the preceding retained
experiment: source `contract=False` still produced native FFMA and the
same RK23 pair discrepancy. The preceding final-only link experiment on
original source IR produced byte-identical cubins for final FMA true and
false. The successful intervention combines source contraction removal
with final-link contraction removal. These earlier failed interventions
are preserved and explain why the two controls are separate.

## Provenance and audit

- Native execution: `R/numerical_direct_native_e1/receipt.json` and its
  eight NPZ files. Runner SHA 31a0fb330b855d959dfc4eddc68020d3cdfd3a222e2b4880e2a7586bd28b53ad.
- Frozen source and IR: `V/numerical_contract_ir_author_e2/receipt.json`.
  Full captured source cubin equals its retained predecessor; rolled
  equivalence is limited to the independently reviewed internal constant
  symbol renumbering contract. Code and data are unchanged.
- Four offline images: `V/numerical_contract_link_e2/receipt.json`.
  Each true image exactly equals its captured cubin. Each false image
  uses exactly the same saved IR as its true counterpart.
- Independent CPU admission: `V/numerical_direct_independent_e1/receipt.json`.
- Independent native-result approval:
  `V/numerical_direct_native_independent_e1/receipt.json`. Its separate
  saved-data pass verifies all eight arrays, 688 calls and 632 parameter
  checks, preserving the true baseline failure and false-image equality.
- This author's saved-result audit: sibling `audit.py` and `receipt.json`.
  It re-hashes inputs, re-disassembles all four exact images, recomputes
  whole-array checks and comparisons, and checks all 688 recorded Driver
  API calls, including 632 parameter offset/size responses across eight
  launches. Every parameter byte is reconstructed from recorded actual
  source-array and scalar values. Function/module ownership and unload
  order are checked for each lifetime.
- Synchronization and memory-manager lifecycle are bound to the exact
  independently reviewed source. Those non-recorded calls are not
  presented as a separately traced runtime API stream.

Here `R` is
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`, and
`V` is its `verification` subdirectory. The final independent native-output
approval is the separate verifier receipt above; this author audit does
not grant it.

## Applying the diagnostic to other failed comparisons

The method can be applied without fitting constants or changing the
model. Each proposed family/action needs its own preserved constructor,
source settings, exact original grid and constants, duration, reference
outputs and existing numerical tolerances. Capture that exact source
with the public contraction flag disabled, retain its original IR, and
link the same IR under the two final FMA settings. Verify the actual
native forms rather than assuming a flag guarantees zero FFMA.

Before an intervention executes, the direct true image must reproduce
that source's retained own output and pass the actual parameter-count,
offset, size, scalar and array ownership gates. The general batch kernel
currently supplies 15 source arguments, but another workload's signature
or launch geometry must be checked rather than inferred from RK23. A
different ABI or explicit fused intrinsic requires a separately reviewed
form; the existing RK23-only runner intentionally refuses unsupported
signatures and cases.

The per-workload conclusion is conditional: whether contraction removal
removes, reduces, preserves or changes the existing discrepancy. Every
failure remains retained. Equality for RK23 cannot be assigned to the
other failed actions without those measurements. None of this needs
new latency multipliers, timing regression, numerical tolerances or
hardware-model parameters. This is diagnostic coverage, not a change to
the candidate ranking or default-selection model.
