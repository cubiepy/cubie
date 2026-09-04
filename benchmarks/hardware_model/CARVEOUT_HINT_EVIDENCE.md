# The preference-only run did not establish a physical carveout contrast

The accepted ordinary run and both saved profiles pass the retained
source, binary, work, state and cleanup audit. The 64-percent preference
was read back correctly, but its separately profiled launch used the same
8,192-byte shared-memory configuration as the 8-percent control. These
launches therefore do not provide the intended 8 KiB versus 64 KiB
physical contrast. No causal slowdown or cache-capacity conclusion is
assigned to this experiment.

## Raw evidence and independent checks

All paths below are under
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.

- Ordinary: `stage_base_carveout_ordinary_e1/result.json` and
  `samples.jsonl`, six full warm NPZ snapshots and retained native files.
- Profiles: `profile_a_carveout_control8_e1` and
  `profile_b_carveout_shared64_e1`, including each `profile.ncu-rep`,
  `metrics.csv`, command/source snapshot and `benchmark/result.json`.
- Independent receipt and executable CPU audit:
  `verification/carveout_hint_independent_20260905/receipt.json` and
  `audit.py`. Both reports were independently re-imported with `ncu
  --import --page raw --csv`; each import exited zero and reproduced the
  original units and single kernel row exactly. The fresh re-exports are
  retained beside the receipt. No compilation or GPU launch was used by
  the audit.

The ordinary protocol has two blocks, six samples per slot per block,
and 36 accepted measurement rows. All are at least 20 ms; the smallest
is 21.541599 ms. Raw membership, mirrored order, every preference
transition, same CUfunction handle, native specialization count, native
binary and pinned resource/geometry receipts pass revalidation. All six
warm snapshots exactly match the original accepted baseline's FP32 state
array of shape `(2, 32, 262144)` and its 262,144 zero status codes. Both
single-solve profile snapshots match those same arrays exactly.

The common cubin SHA256 is
`5271c0bee09b7a4db4871b78e7b2c45e9af3964169275447813d8ead648aceee`.
Compiled resources are 255 registers/thread and a 688-byte local frame.
The actual pinned block has 64 threads, four resident blocks/SM and 256
resident threads/SM, with 18.285714 full occupancy waves. Dynamic shared
reservation remains four bytes/block. Profiles independently report
64-thread blocks, 4,096 grid blocks, 255 used registers/thread and a
four-block register occupancy limit.

| Profile arm | Requested shared bytes | Preference getter | `launch__shared_mem_config_size` | Converted bytes |
|---|---:|---:|---|---:|
| `control8` | 8,192 | 8 percent | `8.192000 Kbyte` | 8,192 |
| `shared64` | 65,536 | 64 percent | `8.192000 Kbyte` | 8,192 |

The exported `Kbyte` unit is decimal, so the conversion multiplies by
1,000. Getter readback is a requested preference, not an observation of
physical configuration. Both original preferences were restored to -1;
every Solver closed, cache-root overrides were restored and cleanup
errors are empty. Profile and report-import exit codes are zero.

Actual configuration is established only for the two separately
profiled launches. The ordinary run correctly retains
`ordinary_actual_carveout_observed=false`. Its successful numerical and
timing protocol does not turn the hint into a forced hardware setting.
Profile event durations remain excluded from the ordinary timing bank.
Local L1 load/store sector totals and lookup misses are retained in the
receipt, but they are not interpreted as responses to a changed cache
capacity because the required physical treatment did not occur.

The [CUDA Programming Guide's L1/shared-memory balance section](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#configuring-l1-shared-memory-balance)
permits the driver to select a shared-memory capacity different from the
function preference. This documents why a getter alone cannot establish
the treatment; it does not identify why this driver chose 8 KiB here.
