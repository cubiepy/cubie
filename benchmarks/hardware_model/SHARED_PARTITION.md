# Driver-selected shared partition

Production never sets a carveout preference, so the driver's default
choice fixes the L1/shared split at every launch. `partition_check.py`
records that choice for ordinary `Solver.solve` launches and
`partition_rule_probe.py` maps it for a trivial kernel over block size
and dynamic shared bytes; both read `sharedMemoryExecuted` from CUPTI
kernel activity (collector in `cupti_sources/cupti_carveout_author_e1`).

Raw records: `cubie-notes/hardware_unroll_placement/partition_check_e1`
(63 production launches, 7 configs x 3 placements x 3 block sizes) and
`partition_rule_e1` (156 launches, 6 block sizes x 13 shared sizes).

## Observations

- Requesting a preference through the public compile option changes the
  compatibility handle's attribute and occupancy query but the launched
  kernel records no request and an unchanged partition
  (`partition_rule_e1_carveout8`, `partition_rule_e1_carveout100`).
  Occupancy queries on that handle therefore diverge from the launch
  once a preference is set; with no preference they agree.
- The partition is not fixed at 100 KiB. All-local kernels land on 8,
  16 or 32 KiB depending on resident blocks; every shared placement at a
  production block size landed on 100 KiB except the single-buffer
  lorenz96_20/kvaerno3_bicgstab rows (64 KiB at 64 and 128 threads,
  32 KiB at 256 threads).
- With one resident block the driver picks the smallest supported size
  holding dynamic bytes plus the 1 KiB block reservation. With more
  resident blocks it picks one or two supported sizes above that fit,
  so the resident-block need is a lower bound, not the rule.

| block | resident blocks | dynamic bytes | executed partition |
|---:|---:|---:|---:|
| 1024 | 1 | 4 to 6144 | 8 KiB |
| 1024 | 1 | 8192 to 12288 | 16 KiB |
| 1024 | 1 | 16384 to 24576 | 32 KiB |
| 512 | 3 | 4 | 8 KiB |
| 512 | 3 | 512 to 1024 | 16 KiB |
| 512 | 3 | 2048 to 4096 | 32 KiB |
| 512 | 3 | 6144 to 16384 | 64 KiB |
| 256 | 6 | 4 | 16 KiB |
| 256 | 6 | 512 to 1024 | 32 KiB |
| 256 | 6 | 2048 to 8192 | 64 KiB |
| 128 | 12 | 4 to 512 | 32 KiB |
| 128 | 12 | 1024 to 4096 | 64 KiB |
| 64 | 24 | 4 | 32 KiB |
| 64 | 24 | 512 to 1024 | 64 KiB |
| 64 | 24 | 2048 and above | 100 KiB |

Unified L1 plus shared is 128 KiB on this SM89 device, so the L1 data
capacity available to a launch is 128 KiB minus the executed partition.

## Model use

The placement term reads the partition from these tables by resident
blocks and per-block shared bytes; where a pair is not tabulated the
next larger tabulated need is used. Making the partition a model choice
requires the backend to set the preference on the launched function.
