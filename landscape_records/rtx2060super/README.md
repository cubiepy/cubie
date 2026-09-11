# cubie PR 927 auto_performance landscape, RTX 2060 SUPER

- Device: `device NVIDIA GeForce RTX 2060 SUPER cc (7, 5) SMs 34 L2 4 MiB shared/SM 64 KiB opt-in 64 KiB icache 64 KiB` (icache measured)
- Driver 595.84, CUDA 13.3, backend mlir (cubie-numba-cuda-mlir 0.5.1.1), Python 3.12.3, cubie 0.12.0
- Commit fab9a4a5094549aa94ea4e0e0ef9e1f93576f19b, branch chore/policy-landscape-turing-validation
- SM clock locked at 1470 MHz; memory clock lock unsupported on this GPU, read 6801 MHz throughout
- No other compute process on the GPU during the runs

## Commands (worktree root, in order, 2026-09-11 09:10 to 12:30 NZST)

    PY=.venv/bin/python
    OUT=~/landscape_2060
    COMMON="--blocksizes 32,64,128,256 --workers 4"
    $PY benchmarks/verify_performance_policy.py --out $OUT/verify_full.jsonl --log $OUT/verify_full.log --systems chain20,chain64,chain32_c8,lorenz96_10,lorenz $COMMON
    $PY benchmarks/verify_performance_policy.py --out $OUT/verify_full.jsonl --log $OUT/verify_full.log --systems lorenz96_40 --algos kvaerno3,kvaerno5 $COMMON
    $PY benchmarks/verify_performance_policy.py --out $OUT/verify_full.jsonl --log $OUT/verify_full.log --systems lorenz96_40 --algos radau_iia_3,radau_iia_5,kvaerno3_bicgstab,radau_iia_5_bicgstab,bogacki-shampine-32,vern7,tsit5,rosenbrock23 --n-runs 32768 --duration-scale 0.5 $COMMON
    $PY benchmarks/verify_performance_policy.py --out $OUT/verify_fabbri.jsonl --log $OUT/verify_fabbri.log --systems fabbri --algos radau_iia_5 --n-runs 32768 $COMMON
    $PY benchmarks/verify_performance_policy.py --out $OUT/verify_full.jsonl --score > $OUT/verify_full.score.txt
    $PY benchmarks/verify_performance_policy.py --out $OUT/verify_fabbri.jsonl --score > $OUT/verify_fabbri.score.txt

Every invocation exited 0 first time (2 h 11 min, 40 min, 23 min, 5 min); nothing rerun.

## Results

- verify_full.jsonl: 60 rows; verify_fabbri.jsonl: 1 row. Every arm has sass_bytes; no arm has error set.

## Anomalies

- verify_fabbri.log line 4: a compile worker reported `compile FAILED fabbri/radau_iia_5 plain: SyntaxError` on a half-written generated module; the main process rebuilt the arm from cache and the row records no error.
- Fabbri solves report 6 to 17 of 32768 runs failed (STEP_TOO_SMALL / NEWTON_DIVERGENCE); see the score output's failed-run section.
