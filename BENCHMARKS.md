# TACHYON-π benchmarks

Machine: AMD Ryzen 9 9950X (16C/32T, AVX-512), 128 GB DDR5, Debian 13, GCC 14.2.0,
GMP 6.3 (`mpz` = FFT/Toom-Cook multiplication), `-O3 -march=native`, leaf=8, guard=30.
Timings: `--benchmark` medians of K trials, in-process, **excluding** disk writes
(`--out` costs extra) and process init; digits = significant decimal digits of π.
Raw records: `benchmarks/results.jsonl` (append-only; one JSON per run).

## Baseline (stage 2): Chudnovsky BS + GMP, **single thread**, 2026-11-03

| digits | total | split | sqrt | final div | convert | peak RSS | digits/s |
|---|---|---|---|---|---|---|---|
| 10k | 0.00034 s | 0.00017 | 0.00004 | 0.00007 | 0.00005 | 4 MB | 2.9e7 |
| 100k | 0.0103 s | 0.0056 | 0.0011 | 0.0020 | 0.0015 | 6 MB | 9.7e6 |
| 1M | 0.207 s | 0.128 | 0.019 | 0.024 | 0.036 | 23 MB | 4.8e6 |
| 10M | 3.88 s | 2.59 | 0.278 | 0.379 | 0.631 | 211 MB | 2.6e6 |
| 100M | 89.0 s¹ | 57.7 | 3.80 | 4.74 | 9.36 | 2321 MB | 1.12e6 |

¹ Pre BUG-1-fix figures: `set_u128` limb-order bug inflated P(0,N) for D>=10M, corrupting
100M digits AND wasting word-work on oversized operands. Post-fix 100M baseline (2026-11-05):
**62.76 s** total (split 43.8, sqrt 3.70, final 4.97, convert 10.26), RSS 2076 MB, --verify PASS.
Earlier GPU/hybrid rows (E4) were measured with the same oversized-P penalty; relative
speedups remain meaningful but absolute times will improve on re-measurement.

- Trials: K=5 (≤10M), K=3 (100M). Variation < 2% at every size.
- Asymptotic slope: 10M→100M is 22.9x for 10x digits (≈ n·log n, GMP Schönhage–Strassen regime).
- Phase shares at 100M: split 64.8%, convert 10.5%, final-div 5.3%, sqrt 4.3% (remainder =
  init/output/timing overhead).

## Reference bars on this machine

| program | digits | time | threads | notes |
|---|---|---|---|---|
| y-cruncher v0.8.7.9547 (fresh, this host, 2026-11-04) | 1M | 0.011 s | 32 | custom compute + verify, no disk write |
| y-cruncher v0.8.7.9547 (fresh, this host, 2026-11-04) | 10M | 0.115 s | 32 | custom compute + verify, no disk write |
| y-cruncher v0.8.7.9547 (fresh, this host, 2026-11-04) | 100M | 1.814 s | 32 | bench mode, incl. verify, no disk write |
| y-cruncher v0.8.7.9547 (fresh, this host, 2026-11-04) | 1B | 27.97 s | 32 | bench mode, incl. verify, no disk write |
| tachyon-pi (this work) | 1M | 0.095 s | 32 | E7c: pmulp + parallel split/convert; no verify, no write |
| tachyon-pi (this work) | 10M | 1.117 s | 32 | E7c (median of 3, cooldown; phases 0.34/0.27/0.28/0.23) |
| tachyon-pi (this work) | 100M | 62.8 s | 1 | exact integer arithmetic, no approximations (post BUG-1 fix) |
| tachyon-pi (this work) | 100M | 25.4 s | 32 | E5a split + E6 convert (post BUG-2); superseded by E7c row below |
| tachyon-pi (this work) | 100M | **20.48 s** | 32 | E7c (median of 3, cooldown, var <0.5 %; split 9.60, sqrt 3.66, final 3.81, convert 3.32; peak RSS 5.00 GB); --verify PASS 20.6–21.6 s |
| tachyon-pi (this work) | 1B | 317.9 s | 32 | E7c first 1B run, --verify **PASS**; split 222.1, sqrt 48.0, final 47.8; peak RSS 29.2 GB; wall 9:51 incl. startup |

Gap to y-cruncher @100M: **11.3x** at 32 threads (was 14.0x post-E6; 80x at 1 thread
pre-fix; 17.0x pre-E6/BUG-2).
Gap @10M 9.7x (was 13.8x), @1M 8.6x (was 9.7x), @1B 11.4x (first 1B run; y-cruncher's
27.97 s incl. verify vs our 317.9 s incl. verify). Note the y-cruncher figures *include* their verification pass
while tachyon totals exclude verification, so the true compute gap is slightly larger.
Closing plan: E8 parallel AVX-512 FFT multiplication (split phase is now 47 % of total
and mul-bound with single-threaded GMP FFT), concurrent CPU+GPU execution (E5b),
cost-balanced split (E5c), transform-reuse DAG (H3).

### y-cruncher re-benchmark conditions (fresh bars)

Binary: official Linux static build v0.8.7 Build 9547, auto-tuned profile
`24-ZN5 ~ Komari (20241107)` (correct Zen 5 path). Host: same machine that runs tachyon
(AMD Ryzen 9 9950X, 32 threads, 132 GiB RAM, est. 5.46 GHz). Commands:
`./y-cruncher skip-warnings bench {100m|1b} -od:0` and
`./y-cruncher skip-warnings custom pi -dec:N -verify:1 -od:0` (bench mode rejects < 20M).
`-od:0` suppresses the output file, so disk writes are excluded on both sides.
Digit convention: both compute floor(10^k · pi) style digit counts; "Last Decimal Digits"
and "Spot Check: Good through N" confirmed in every run.
Start-to-end wall overhead was ≤ 0.08 s (100M), ≤ 1 s (1B); we compare compute times.
The older bars (100M 1.107 s, 1B 14.670 s) came from an earlier session under a different
digit-count definition and almost certainly excluded verification; the fresh bars supersede them.

## GPU backend (stage 3, E4): CUDA three-prime exact NTT, 2026-11-04

RTX 5070 Ti (sm_120, GDDR7 ~896 GB/s). Local LLM occupies most VRAM (~1.4 GB free);
backend self-probes free memory and falls back to CPU gracefully (workspace 2^24 ≈ 0.55 GiB).

Multiplication medians of 5, `gpu-bench` (e2e = full `tachyon_cuda_mul` incl. transfers):

| operand bits | GMP ms | GPU e2e ms | speedup |
|---|---|---|---|
| 2^19 | 0.95 | 0.35 | 2.7x |
| 2^21 | 3.77 | 0.73 | 5.2x |
| 2^23 | 19.8 | 2.11 | 9.4x |
| 2^25 | 100.8 | 10.5 | 9.6x |
| 2^26 | 211.7 | 23.4 | 9.1x |
| 2^27 | 445.3 | 90.6 | 4.9x (Karatsuba) |
| 2^28 | 928.2 | 309.3 | 3.0x (Karatsuba) |

Crossover ≈ 2^19 operand bits; GPU pipeline ≈ 95% of e2e (transfer overhead minor).
Raw throughput ~1.4e10 NTT butterflies/s, memory-bound at N ≥ 2^23.

π end-to-end (medians of 3, single CPU thread + GPU):

| digits | cpu (s) | gpu (s) | speedup | hybrid (s) |
|---|---|---|---|---|
| 2M | 0.497 | 0.364 | 1.37x | — |
| 10M | 3.810 | 2.418 | 1.58x | 2.443 |
| 30M | 16.29 | 9.13 | 1.78x | 9.45 |

Correctness: gpu & hybrid outputs byte-identical to `machin-check` (1M) and pass BBP
`--verify` (10M). Hybrid v1 (threshold dispatch) ≡ gpu today; its value arrives with
CPU/GPU concurrency (E5). Still 3.6x behind y-cruncher at 100M (multi-core + pipelining gap).

## E5a thread scaling (stage 4, 2026-11-05, post BUG-1 fix)

`--threads T` task-parallel split (cpu backend only); totals exclude disk write.

| digits | threads | split s | total s | total speedup vs T1 |
|---|---|---|---|---|
| 10M | 1  | 2.515 | 3.781 | 1.0x |
| 10M | 4  | 1.046 | 2.302 | 1.6x |
| 10M | 16 | 0.725 | 1.982 | 1.9x |
| 10M | 32 | 0.714 | 1.968 | 1.9x |
| 100M | 1  | 43.8  | 62.76 | 1.0x |
| 100M | 16 | 12.2  | 31.01 | 2.0x |
| 100M | 32 | 12.1  | 30.87 | 2.0x |

- Split speedup saturates ~3.6x (16T≈32T; SMT-neutral). Bottleneck is the serial spine
  of the binary tree (Amdahl), not memory bandwidth. Cost-balanced splitting (E5c) and
  CPU/GPU concurrency (E5b) are the paths past it.
- Correctness at every row: outputs byte-identical across thread counts; 10M T16 and
  100M T16/T32 --verify PASS.

## E6 + BUG-2: parallel conversion in production path (2026-11-04)

`mpz_get_str_parallel` (E6) at 32T: 100M convert 10.61->3.51 s (3.0x; serial pow-chain
bounds parallelism ~3-4x), 10M 0.631->0.234, 1M 0.036->0.019; bit-exact vs GMP (convert-test).
BUG-2: `compute_pi` unconditionally ran a throwaway *serial* mpz_get_str for timing
bookkeeping that production paths then duplicated; removed behind `PiOptions::time_convert`
(--benchmark only). 100M T32 total 30.9->25.4 s; `--out` wall 26.1 s *including* disk write.
Remaining serial tail: sqrt 3.89 + final-div 5.18 (GMP, single-thread) = 36% of total;
split spine 12.83 (3.6x scaling ceiling, E5c); convert serial remainder ~1.3 s.
