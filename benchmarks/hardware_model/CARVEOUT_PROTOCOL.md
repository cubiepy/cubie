# Same-cubin carveout preference control

The accepted stage-base placement comparison changed several things at
once. Matched profiles reported shared configuration 8192 to 65536 bytes,
unchanged 255 registers/thread and 256 resident threads/SM, and increased
L2 read traffic despite somewhat fewer local-load sectors. Its binaries
also differed. A buffer-placement timing difference therefore cannot be
assigned exclusively to explicit shared loads or to the shared/L1 split.

`carveout_probe.py` reconstructs only the completed cohort's local
baseline. It preserves the original generated-source snapshot, source,
compiler, input, unroll, duration, cubin/PTX/SASS and pinned geometry
gates. It uses one Solver and one existing native specialization and
loaded CUfunction. The treatment changes only that function's preferred
shared-memory carveout. No source, driver policy, clocks, launch dynamic
shared reservation or compiled resource setting is modified.

The instrument requests 8 KiB and 64 KiB explicitly. Their integer
percentages are `ceil(100*target_bytes/queried_max_shared_bytes_per_SM)`.
For the queried Ada maximum of 102400 bytes these are 8 and 64 percent.
The 8 KiB arm targets the observed original default configuration; it
does not use the unspecified `-1` preference, which allows driver choices
to retain a prior configuration. Both explicit requests remain hints.

## API and source evidence

- [Ada Tuning Guide, Unified Shared Memory/L1/Texture Cache](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#unified-shared-memory-l1-texture-cache)
  lists supported shared capacities 0, 8, 16, 32, 64 and 100 KiB, unified
  capacity 128 KiB and a 1 KiB per-block driver reservation. This
  experiment queries the maximum instead of substituting a constant.
- [CUDA Programming Guide, Configuring L1/Shared Memory Balance](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#configuring-l1-shared-memory-balance)
  defines the percentage of maximum supported shared capacity and
  upward rounding to supported capacities. It explicitly permits the
  driver to select a different capacity.
- [CUDA Driver API types](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)
  defines `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`, numeric
  value 9, and the default value -1. The instrument uses the installed
  CUDA binding enum, not a model coefficient.
- Installed `numba_cuda_mlir/compiler.py:78-99` implements
  `CUFunc.set_shared_memory_carveout()` through `cuFuncSetAttribute`.
  The matching `cuFuncGetAttribute` binding convention is used at
  `compiler.py:167-170` for native resource queries.
- Installed `numba_cuda_mlir/descriptor.py:2211-2218` applies a configured
  dispatcher option to the loaded function. The instrument rejects that
  option when present so the dispatcher cannot overwrite the treatment.

Before and after every treatment setting, the instrument checks that the
actual loaded handle, native specialization count, cubin bytes, entry,
register/local-frame counts and driver-queried pinned occupancy remain
unchanged. Setter arguments, previous preference and getter readback are
retained. Geometry must satisfy two full occupancy waves. The original
function preference is restored on exit and the owned Solver is closed;
cleanup errors make the attempt fail.

## Ordinary and profile protocols

Ordinary execution has two paired blocks, six samples per slot per block.
Slots are `control8/shared64/control8_repeat`, reversed on alternate
samples. These are replicated control slots using the same loaded
function, not independent baseline compilers. Each block first saves
full warm state and status arrays for all three slots and requires exact
agreement with the original accepted baseline warm arrays. It then
settles under the recorded harness duration and collects the mirrored
samples. Every solve retains timing/status/finiteness and treatment
receipts; every measurement must be at least 20 ms. The original solve
duration is fixed. A short sample fails the attempt instead of changing
duration and silently losing the original state comparison.

Profile mode requires a completed ordinary result and revalidates raw
row membership, all six warm NPZs, 36 measurement rows, ordinary source
and native artifact hashes. It recompiles the same baseline and checks
both original and ordinary native identities. It then applies exactly
one selected preference and performs one state-only solve, retaining
exact original warm-state/status agreement. Profile times cannot enter
the ordinary sample set.

Example commands, run with frozen source and harness on `PYTHONPATH`:

```powershell
python benchmarks/hardware_model/carveout_probe.py --cohort-dir C:\local_working_projects\cubie-notes\hardware_unroll_placement\stage_base_placement_e1 --out C:\local_working_projects\cubie-notes\hardware_unroll_placement\stage_base_carveout_ordinary_e1 --execute
python benchmarks/hardware_model/carveout_probe.py --cohort-dir C:\local_working_projects\cubie-notes\hardware_unroll_placement\stage_base_placement_e1 --ordinary-dir C:\local_working_projects\cubie-notes\hardware_unroll_placement\stage_base_carveout_ordinary_e1 --profile-arm control8 --out PROFILE_OUTPUT --execute
```

Use the reviewed elevated session for each profile command, with research
script SHA and `runtime_tree=epoch_ff3a567f`; repeat for `shared64` into a
different fresh output. No tool in this module starts Nsight Compute.

Require successful profile/import exits, one intended solve kernel,
matching cubin/input/source/state/geometry and the actual
`launch__shared_mem_config_size` value. The control must report 8192 bytes
and treatment 65536 bytes. Nsight exported units are significant: the
existing baseline exports `8.192000 Kbyte`, meaning 8192 bytes, not
8.192 KiB. Record the unit and its conversion explicitly. Useful
contrasts alongside LaunchStats/Occupancy are:

- `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`
- `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum`
- `lts__t_sectors_op_read.sum`, `lts__t_sectors_op_write.sum`
- executed instruction counts, active/eligible warp metrics and source
  local-memory sites using the same metric definitions as the original
  matched placement profiles.

Neither attribute getter nor occupancy API reports the actual carveout.
Ordinary results deliberately retain
`ordinary_actual_carveout_observed=false`; even successful separate
profiles establish the configuration only for those profiled launches.
If a profile does not honor its requested configuration, the physical
contrast remains unresolved. Do not relabel the hint as a forced cache
capacity or use any failed/short/profile timing as an ordinary sample.
No fitted penalty or new pre-compile default is supplied by this probe.
