# Arithmetic service evidence

## Qualified FFMA instrument

The qualified SM89 FFMA binary has a 257-instruction dependent body and
five retained loop-administration instructions. Its hot native loop has 262
instructions, while the complete kernel native inventory has 336 instruction
addresses. It uses 20 registers per thread and has no local or static memory.
The launch reserves 8,192 bytes of shared memory so exactly one 1,024-thread
block resides on each of 56 SMs. A 112-block launch therefore executes two
complete occupancy waves. The same binary selects either one complete active
warp or all 32 warps; every allocated warp remains at the final CTA barrier.

The cubin SHA256 is
`7ddf4759a965764ebfef032ca06e31a24ed94d42897889c4426a4d619df6ee7c`.
The independently reviewed native certificate is
`verification/arithmetic_ffma257_independent_20260905/receipt.json`, SHA256
`bd562f7a7b0eea53e9171bb296d39c7b7b6b1dac841ace653ecc6bc11ae014e4`.
It binds the scalar ABI, exact body, result dependency, runtime count,
timestamp guards, convergence control, output stores, and final barrier. The
one- and 32-warp paths use the same native artifact.

Each active lane follows the exact dyadic recurrence `x = -x + 1`. Odd N and
even 2N have different expected endpoints. The repaired ordinary bank contains
48 measurement launches at N=65,539, arranged as 12 paired N/2N observations
for each population. Across the four calibration and 48 measurement launches,
all 3,075,072 retained active FP32 outputs match the independent bit-exact
oracle. The 48 measurements account for 2,838,528 of those endpoints.

## Ordinary observations

For one active warp, every paired median N-to-2N increment is exactly
`1044/257 = 4.062256809338521` cycles per target operation. Paired minima range
from 4.054474648801204 to 4.067930266132566 cycles per operation, with median
`1044/257`. The interval contains the retained loop control and guard, so this
is a lower-envelope dependent-chain observation for the admitted kernel, not
an intrinsic SM89 FFMA latency.

For all 32 active warps, paired median increments range from
8.025848630360763 to 8.26016130948377 cycles per lane operation, with median
8.025976053228295. Counted target work divided by the sum of each CTA's occupied
SM-clock envelope ranges from 3.7488452478674437 to 3.866492802465206 warp
FFMA instructions per SM cycle, with median 3.8664509676250995. This is achieved
service for this exact two-wave workload. It is neither a device wall-clock
rate nor an architectural initiation interval.

The author raw audit is
`verification/arithmetic_ffma257_ordinary_audit_e2_20260905/receipt.json`.
The independent audit is
`verification/arithmetic_ffma257_ordinary_independent_e2_v2_20260905/receipt.json`,
SHA256
`760ea7778d3ce72c26cf3cb380104e54bca3f3cb1a847f596860cd8330cb44ba`.
Both retain the two unusual minimum tails rather than replacing them with the
otherwise repeated `1044/257` value.

## Counter qualification

Four matched Nsight Compute reports cover one warp and 32 warps at N and 2N.
All reports reimport successfully and contain one kernel row plus all 336
source-counter addresses. Hardware and software warp-instruction totals agree
exactly:

| Active warps | Count | Target FFMA warp instructions | All software and hardware warp instructions |
|---:|---:|---:|---:|
| 1 | N | 1,886,474,576 | 1,923,245,296 |
| 1 | 2N | 3,772,949,152 | 3,846,421,712 |
| 32 | N | 60,367,186,432 | 61,541,849,600 |
| 32 | 2N | 120,734,372,864 | 123,083,494,912 |

The predicated-on thread count is exactly 32 times each target warp count. All
addresses outside the hot loop have zero N-to-2N delta. The terminal
call-shaped exit at PC `0x1200` is inside the audited hot-loop range and also
has zero predicated N-to-2N delta. The FMA-pipe counter exceeds the exact target
count by 10,864 warp instructions in each one-warp report and 14,336 in each
32-warp report. Those fixed residuals remain recorded without attribution or
subtraction.

The independent saved-report review is
`verification/arithmetic_ffma_profile_independent_20260905/receipt.json`,
SHA256
`596a0750dd1022aeaf3b093935e2f9deb4bfeceb80ba1da986d61bfce29eccc0`.
It reimports the four reports, rechecks raw arrays and PC rows, and confirms
the actual 8,192-byte shared reservation, one resident block per SM, and two
waves. Profile event times are excluded from the ordinary distributions.

## Model use

The observations qualify an FFMA work counter and two population-dependent
service distributions. They do not isolate a native latency or physical
pipeline topology. `ARITHMETIC_PROXY_CATALOG.json` therefore keeps the
four-cycle Turing dependent latency as a named transfer assumption and obtains
the CC8.9 aggregate FP32 capacity from NVIDIA's published 128 results per SM
clock. One full-warp FFMA produces 32 results under that convention, yielding
the explicit aggregate capacity reservation of 0.25 SM cycle per warp
instruction. No solver time, native register label, or fitted correction enters
that catalog.

The hardware capacity source is NVIDIA's
[CUDA Best Practices Guide, Table 5](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions).
The latency proxy remains labeled to Jia et al.'s
[Turing microbenchmark Table 4.1](https://arxiv.org/pdf/1903.07486#page=40).
The one-warp ordinary observation is retained as workload evidence and does not
silently replace either sourced field.
