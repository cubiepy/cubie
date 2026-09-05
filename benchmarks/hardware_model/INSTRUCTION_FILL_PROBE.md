# Instruction delivery probes

`instruction_fill_probe.py` emits explicit PTX, prepares a native image
without launching it, and measures independently reviewed images. Its
output is physical microbenchmark evidence. It does not fit solver
timings or assign an instruction-cache penalty automatically.

Every launch uses 1024 threads per CTA and twice the device SM count
in CTAs. An initial and final CTA barrier retain all 32 allocated
warps while a runtime argument selects 1, 4, 8, 16, or 32 working
warps. The driver occupancy query must report one resident CTA per
SM and zero local allocation. Working warps and allocated warps are
separate recorded quantities. This experiment changes active demand;
it does not reproduce the earlier eight-resident-warp geometry.

Eight distinct FP32 chains use runtime multiplier and increment
registers, both with value one. Their initial values are seed through
seed plus seven. Volatile global loads obtain the coefficients,
iteration count, bound, warm mode and recurrent output pointers before
the measured region. The pack contains seven uint64 words; scalar
controls use the low uint32 word. Native inspection must verify these
remain register operands, with no recurrent parameter/constant reads.
A checksum
and an endpoint-dependent branch precede the ending clock. Every
active thread must return the exact integer checksum; inactive
threads preserve their initialized sentinel. All arithmetic remains
inside the consecutive-integer FP32 range. This is an observable
integer-valued FP32 recurrence, not a solver workload.

## Stream service

The `stream` image repeats a straight FFMA payload between two clock
reads. Requested body KiB count only the intended 16-byte FFMA
instructions. Native review records the actual hot PC span and
control instructions. The ordinary run alternates N and 2N, retaining
all raw arrays, one warmup pair, and each measured pair. Its paired
clock difference divided by the added FFMA count includes recurrent
loop control. It is a stream service interval, not a dependent cache
fill latency.

A small first bank is 32, 192, and 256 KiB payloads with all five
working populations. Add 512 KiB if the two larger streams have not
reached a stable miss/service regime. Run 64 and 128 iterations.
Profile selected control and saturated cases at both iteration
counts. Conserve native dynamic instructions, instruction GCC
requests/hits/misses, and GCC-origin L2 requests and sectors. Native
register-only recurrence removes a recurring constant-operand source
from that comparison; setup and output remain separate traffic.

## Same-PC victim contrast

Each `victim` trial traverses an aggressor outside the timed bracket.
The cold mode then times a victim path. The warm mode first traverses
the same victim PCs untimed and returns to the same clock controller
before timing them. A runtime phase variable controls that return.
The victim has 1, 2, 4, or 8 eight-FFMA targets with conditional
branches over observable padding between targets. A separate default
16 KiB padding region separates the clock controller and first target
in the requested layout. Padding branches are unvisited for admitted
inputs. Native layout, target alignment, and shared target PCs must
be checked after ptxas; the PTX ordering is not a native layout claim.

`--entry-padding-ffmas` admits 0, 8, 16, 32, 64, 128, 256, 512,
or 1024 FFMAs, retaining 1024 as the default. With a fixed 256 KiB
aggressor and one eight-FFMA victim, this finite gap sweep probes how
far sequential fetch can make a nearby target ready before its timed
branch arrives. The recorded requested padding bytes exclude control
instructions; use actual SASS PC gaps when interpreting the result.
Cold/warm contrast versus this gap can constrain effective hardware
lookahead without selecting a prefetch depth from solver timings.
The source-PC scheduler's next-PC demand mode remains a named policy
hypothesis; it does not claim the hardware lacks sequential prefetch.

Each trial writes its timestamp after its ending clock. Both modes
execute the same timed clock/checksum/control sequence. Warm mode
adds one untimed victim traversal, which the exact endpoint includes.
Start with one victim target, 256 KiB aggressor, and one working warp.
Use the 32 KiB aggressor as a cheap non-eviction control. Increase
targets or aggressor only to resolve the measured attribution.

Cold minus warm clocks are initially an unassigned composite. Body
size alone does not establish eviction. GCC instruction counters
must establish the relevant miss contrast; L2 request hit/miss and
sector counters must establish the backing state. Aggregate misses
cannot automatically be assigned to the timed victim: aggressor and
controller also fetch instructions. Same-PC native mapping, control
variants, and where available source/PC sampling must support that
attribution. A repeatable delta with a conserved victim miss count
can qualify a critical-path delivery interval for this motif. It
does not isolate an intrinsic hardware latency by subtracting clock
or branch constants.

## Preparation and review

`emit` writes PTX and source identity using CPU only. `prepare` also
runs ptxas and nvdisasm, saves their full output, and hashes the
executables and artifacts. Each output directory must be new.

```powershell
python -m benchmarks.hardware_model.instruction_fill_probe prepare `
  --out C:/local_working_projects/cubie-notes/hardware_unroll_placement/instruction_stream32_e1 `
  --kind stream --body-kib 32 --ptxas <ptxas.exe> `
  --nvdisasm <nvdisasm.exe>
```

Independent native review must bind PTX, cubin, SASS, and generator
hashes; identify all hot instructions and timed PCs; verify eight
distinct register-only chains; verify the ending-clock dependency;
verify CFG, padding and barriers; and record actual hot spans and
target gaps. A review receipt has `status: "PASS"`, `ptx_sha256`,
`cubin_sha256`, `sass_sha256`, and an `actual_hot_body` record. That
record should contain named PC intervals, static instruction counts,
and dynamic count formulas distinguishing hot payload, recurrence
control, timed administration, untimed aggressor, and harness.

```powershell
python -m benchmarks.hardware_model.instruction_fill_probe run `
  --out <prepared-directory> --review <native-review.json> `
  --iterations 64 --pairs 3 --populations 1,4,8,16,32
```

For one Nsight launch use `--profile-once --populations 1 --factor 1`
and a new `--label`; use `--warm 0` or `--warm 1` for victim modes.
The benchmark keeps profiling output distinct from ordinary timing.
`launch.json` records the actual device, geometry, kernel resources,
review identity, and declared hot scope. Every endpoint/timestamp
array is saved and hashed. No outlier is discarded.

## Physical interpretation

The first useful result is a measured target-device delivery curve
versus active demand and working instruction footprint. A conserved
miss and sector plateau can identify a request initiation service;
the victim contrast can constrain readiness. These are distinct
quantities. Instruction delivery overlaps independent warp execution
and data/arithmetic work. A model must gate fetch readiness and
reserve a shared initiation resource rather than add total fetch
time to execution time. Domain, line granule, and prefetch behavior
remain explicitly qualified until this bank resolves them. No
application-fitted prefetch depth belongs in this experiment.
