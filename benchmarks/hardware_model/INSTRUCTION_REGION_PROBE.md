# Complete-region instruction warm reference

`instruction_region_probe.py` provides a separate probe epoch. Existing
`instruction_fill_probe.py` images, source snapshots and measurements are
unchanged. The motivation is the independently observed 256 KiB,
224-byte-gap warm anomaly: the old priming path visited the victim but
skipped checksum and ending-clock instructions. `stall_no_inst` samples
also appeared at the checksum's next-sector instruction. This is a
specific incomplete warm reference to isolate, not an assigned latency.

Each trial executes the aggressor once, then copies the volatile runtime
warm flag into a phase register. Both modes enter the same beginning
clock, branch across the selected nonexecuted padding, execute the same
eight victim FFMAs and checksum/guard, and read the same ending clock.
Only **after** the ending clock does the phase controller act:

1. Cold mode retains that interval and advances to the next trial.
2. Warm mode clears the phase, discards the first complete interval and
   branches to the same beginning clock. It retains the second interval.

There is no warm-phase branch between the two source clock reads. The
padding branch and endpoint guard are common to both modes. Native
preflight must verify this ordering survives optimization, rather than
inferring it from PTX. It must also verify the two modes share all PCs
from the beginning clock through the ending clock.

The warm-up traverses both clock instructions, checksum and guard as
well as the victim. It does not claim to warm all levels of instruction
cache or to stop replacement/prefetch activity during the repeat. All
measured differences remain complete-region composites until their
native and hardware-counter evidence supports a narrower assignment.

## Geometry and numerical contract

The public arguments retain selectable 32/256 KiB aggressor payloads
and entry padding counts `0, 8, 16, 32, 64, 128, 256, 512, 1024`. The
default is 32 KiB, 1,024 padding FFMAs and one active warp. Actual native
gaps and accessed instruction footprints require disassembly. The
guarded padding is semantically unexecuted for warm flags zero/one;
it may still be fetched by hardware.

A 1,024-thread CTA retains all 32 allocated warps through a final block
barrier. The launch uses two full occupancy waves. Each active warp
retains exactly `N` uint64 intervals in either mode. Each clock PC must
execute `N * (1 + warm)` times per active warp; each aggressor FFMA
executes `N`, and each victim FFMA executes `N * (1 + warm)`.

Eight independent FP32 recurrence chains start at `seed` through
`seed + 7`, using volatile global-load-produced multiplier and increment
registers equal to one. The exact final integer checksum is:

`8 * seed + 28 + N * (aggressor_ffmas + 8 * (1 + warm))`.

The host enforces a checksum below `2**24`. Discarding the first warm
interval does not discard its arithmetic: endpoints include both full
victim passes. Inactive endpoint/timestamp values remain checked. Kernel
event times and all retained per-trial timestamps are saved; the ordinary
summary compares cold/warm medians for each matched `N`/`2N` pair.

## Actions

Source emission uses no assembler or GPU:

```powershell
python -m benchmarks.hardware_model.instruction_region_probe emit `
  --body-kib 256 --entry-padding-ffmas 0 --out C:/research/region_256_gap0
```

The root GPU owner runs `prepare` with explicit `--ptxas` and
`--nvdisasm` paths to produce source, cubin and SASS. Offline assembly is
also permitted to an explicitly authorized CPU investigator; `prepare`
does not launch a kernel. Use `--optimization-level 0` for the separately
qualified layout-preserving experimental form. Level 3 remains available
and is the explicit default. Preparation binds the level, exact assembler
command and command-file hash, plus observed clock/victim PCs and native
control instructions. These observations require an independent native
path review before launch. After that review, the root can run:

```powershell
python -m benchmarks.hardware_model.instruction_region_probe run `
  --out C:/research/region_256_gap0 `
  --review C:/research/region_256_gap0/native_review.json `
  --populations 1 --iterations 64 --pairs 3 --label ordinary_e1
```

The existing hash-checking instruction profile wrapper can load the
prepared `kind: victim` image and retain its arrays without changing the
endpoint formula. A new native gate and counter audit must use the
**doubled warm clock execution count** and complete-region PC contract;
the old gate's pre-checksum phase-control expectation does not apply.
Application replay, warmup launches and cache/clock-control settings must
be recorded as in the matched gap captures. Profile timestamp values
remain functional evidence, not uninstrumented timing measurements.

No fill latency or prefetch depth is assigned by this generator.

## Observed assembler-layout limitation

The first 18 level-3 images move guarded padding after the ending clock;
all requested gaps collapse to 32 native bytes. They are retained as
failed requested-layout evidence under `instruction_region_e1`. Adding
`--dont-merge-basicblocks` at level 3 does not prevent that relocation.
The documented option prevents basic-block merging, not arbitrary block
layout. Levels 1 and 2 also collapse the tested gaps. See NVIDIA's
[PTX compiler options](https://docs.nvidia.com/cuda/archive/13.1.1/ptx-compiler-api/index.html).

The offline level-0 variants retain different native gaps without spills.
Tested requested padding counts 0, 8 and 1,024 produce actual gaps 320,
576 and 16,832 bytes, with 34 registers. Additional clock-result and
checksum MOVs distinguish this native form from level 3. These numbers
are observations of specific images, not a promise of stable layout or
an affine rule to apply to unreviewed images. In particular, the native
gap does not always increase by exactly sixteen bytes per padding FFMA.
This new complete-region form cannot alone reproduce or causally isolate
the old 224-byte-gap anomaly. Each assembled image must retain its actual
PC map and compiler-level qualification.
