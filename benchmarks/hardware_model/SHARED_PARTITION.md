# Driver-selected shared partition

`partition_check.py` records the CUPTI `sharedMemoryExecuted` of ordinary
`Solver.solve` launches; `partition_rule_probe.py` records it for a
trivial kernel over block size and dynamic shared bytes. Both use the
collector in `cupti_sources/cupti_carveout_author_e1`. Raw records:
`cubie-notes/hardware_unroll_placement/partition_check_e1` (63 launches)
and `partition_rule_e1` (156 launches).

Production sets no carveout preference. The `shared_memory_carveout`
compile option sets the compatibility handle only: the launched kernel
records no request and an unchanged partition, and occupancy queries on
that handle then differ from the launch (`partition_rule_e1_carveout8`,
`partition_rule_e1_carveout100`).

The driver picks the smallest supported size holding one block's
dynamic bytes plus the 1 KiB reservation, and one or two sizes above
the resident-block need when more blocks are resident.

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

Production rows: all-local kernels land on 8, 16 or 32 KiB; shared
placements at production block sizes land on 100 KiB except
lorenz96_20/kvaerno3_bicgstab single-buffer rows (64 KiB at 64 and 128
threads, 32 KiB at 256 threads). L1 data capacity is 128 KiB minus the
executed partition. `placement_score.py` reads the partition by resident
blocks and per-block shared bytes from these tables.

## Preference on the launched function

With fork PR 18 (`carveout-launched-function`) the `shared_memory_carveout`
option reaches the launched CUfunction. `partition_rule_probe.py --carveout`
records: `cubie-notes/hardware_unroll_placement/partition_preference_e1`.

| preference | executed partition |
|---|---|
| none | driver lookup above |
| 0 | smallest supported size holding one block |
| 50 | 64 KiB from 3072 dynamic bytes, 100 KiB below |
| 100 | 100 KiB |
