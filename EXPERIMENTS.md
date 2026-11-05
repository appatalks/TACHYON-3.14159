# TACHON-π experiment ledger

Format: hypothesis → implementation → expected → measured → decision.
All measurements on: AMD Ryzen 9 9950X (16C/32T, AVX-512), Debian 13, GCC 14.2,
GMP 6.3, `-O3 -march=native`, single thread unless stated. Medians of repeated
trials after warm-up. Nothing here is extrapolated or quoted from other machines.

## E0 — Baseline bring-up correctness (2026-11-03)

- Hypothesis: Chudnovsky binary splitting with exact integers produces floor(π·10^D)
  within guard tolerance; independent checks can certify digits without trusting the engine.
- Implementation: src/pi_chudnovsky.cpp (GMP backend), verify.cpp (BBP hex-digit spot
  checks with per-digit rigorous error bounds), tests/machin_check.cpp (Machin's formula,
  pure integer recurrence, fully independent summation).
- Correctness: ALL TESTS PASSED (`tests/run_tests.sh`):
  - 101 digits + embedded 100-digit reference: ok
  - 10k / 100k digits + BBP spot checks at positions {13, 1000, 50000, D/3}: PASS
  - 1M digits: byte-exact match vs independent Machin checker (1,000,002 bytes)
  - leaf invariance: 5000 digits, leaf=1 vs leaf=64: identical
- Five baseline math bugs fixed en route (see RESEARCH.md §7 for the corrections log).
- Decision: RETAIN baseline; proceed to performance work.

## E1 — Leaf size sweep (binary-splitting granularity)

- Hypothesis: leaf size (number of terms folded with small-int arithmetic per leaf)
  changes total runtime measurably.
- Implementation: `--leaf L`, 10M digits, 3 trials each, wall time.

| leaf | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| wall (s) | 3.96 | 3.95 | 3.99 | 3.89 | 3.95 | 3.97 | 3.92 | 3.97 |

- Measured: flat within ±2% (noise). Tree granularity is irrelevant while GMP mul
  dominates; small-int folding saves only per-leaf overhead, which is <1% of runtime.
- Decision: REJECT as optimization lever; keep leaf=8 default. Revisit for GPU dispatch
  batching (larger leaves = fewer, bigger GPU multiplications) in the hybrid engine.

## E2 — Phase profile at scale (from --benchmark phase timers)

- Measured (single thread, medians of 5):

| digits | total | split | sqrt | final div | convert | peak RSS |
|---|---|---|---|---|---|---|
| 10k | 0.0003s | 0.0002 | ~0 | 0.0001 | 0.0001 | 4 MB |
| 100k | 0.0103s | 0.0056 | 0.0011 | 0.0020 | 0.0015 | 6 MB |
| 1M | 0.207s | 0.128 | 0.019 | 0.024 | 0.036 | 23 MB |
| 10M | 3.876s | 2.589 | 0.278 | 0.379 | 0.631 | 211 MB |

- Analysis: binary splitting ≈ 67% at 10M and rising (asymptotically dominant);
  decimal conversion ≈ 16% (GMP default divide-and-conquer; y-cruncher's basecase
  split-radix is faster — a later lever); sqrt+final ≈ 17%.
- Decision: the multiplication backend is the right target (H1: GPU exact NTT).
  Conversion cost noted for a dedicated experiment later (E6+).

## E3 — 100M-digit baseline

- Measured (3 trials, median): 89.0 s total = split 57.7 + convert 9.36 + final 4.74 +
  sqrt 3.80; peak RSS 2.32 GB. 10M→100M scaling 22.9x for 10x digits (≈ n·log n).
- Analysis: split-phase share rises with size (67%→65%… dominant as expected); confirms
  H1 (multiplication backend) as the primary lever. y-cruncher does 100M in 1.107 s on
  this host with 32 threads; per-thread gap ≈ 2.5x before parallelism — reachable only
  with a fast custom NTT + parallel pipeline + GPU offload.
- Decision: RETAIN as reference baseline for all stage-3+ comparisons.

## E4 — CUDA three-prime NTT vs GMP crossover (2026-11-04)

- Hypothesis: exact GPU NTT beats GMP above a moderate operand size on RTX 5070 Ti.
- Implementation (`src/cuda/cuda_backend.cu`, 824 LoC): primes 167772161/469762049/754974721,
  28-bit limbs, exact CRT (bound 2^80 < 2^85.6); per prime repack → bitrev → 24 DIT stages →
  fused pointwise(+1/N) → inverse → Garner `crt_accum` (atomicAdd) → chunked parallel carry →
  pack64 → D2H direct into `mpz_limbs_write`. CPU fallback N ≤ 2^15; Karatsuba above N = 2^24.
  VRAM budget `min(free + dev_used − 300MB, 10GB)`; workspace(2^24) ≈ 0.55 GiB.
- Correctness: own harness `tests/gpu-mul-test` (25 trials incl. signs/zero, sizes to 2^27 bits
  through Karatsuba) PASS; `--backend gpu --verify` (BBP) 10M PASS; byte-identical to
  `machin-check` at 1M for gpu AND hybrid; full suite + ctest green.
- Multiplication medians of 5 (bench/gpu_bench, e2e = whole tachyon_cuda_mul):

  | bits | GMP ms | GPU e2e ms | ratio |
  |---|---|---|---|
  | 2^18 | 0.36 | 0.36 (CPU path) | 1.0 |
  | 2^19 | 0.95 | 0.35 | 2.7 |
  | 2^20 | 1.58 | 0.44 | 3.6 |
  | 2^21 | 3.77 | 0.73 | 5.2 |
  | 2^22 | 8.13 | 1.18 | 6.9 |
  | 2^23 | 19.8 | 2.11 | 9.4 |
  | 2^24 | 40.5 | 4.43 | 9.2 |
  | 2^25 | 100.8 | 10.5 | 9.6 |
  | 2^26 | 211.7 | 23.4 | 9.1 |
  | 2^27 | 445.3 | 90.6 | 4.9 (Karatsuba) |
  | 2^28 | 928.2 | 309.3 | 3.0 (Karatsuba) |

- Analysis: GPU pipeline is ≈95% of e2e time → host transfer/API overhead is NOT the limit;
  raw NTT throughput is (~1.4e10 butterflies/s, memory-bound at N ≥ 2^23; Karatsuba 3x-work
  penalty visible ≥ 2^27). Crossover ≈ 2^19 operand bits. 1M-digit π still fully CPU (top node
  ≈ 1.2M bits but mid-tree nodes dominate count).
- π end-to-end (medians of 3): 2M 0.497→0.364 (1.37x); 10M 3.809→2.418 (1.58x);
  30M 16.29→9.13 (1.78x). Gain grows with size as large nodes dominate.
- Hybrid v1 (`hybrid-e1`: GMP below 2^19 bits, GPU above; tests/gpu-mul invariance + Machin
  byte-match): 10M 2.44s, 30M 9.45s — equals GPU (internal N≤2^15 cutoff already implements
  the same crossover). Distinct value requires CPU/GPU CONCURRENCY (E5).
- Cost model upgraded to v2 with E4-calibrated constants on both sides (GMP word-step fit +
  GPU butterfly throughput, same kMuls factor each; dashboard mapping is now data-driven:
  10M hybrid plan estimates 3.13 s CPU-only vs 3.8 s measured, 2.1x planned speedup).
- Decision: RETAIN GPU backend. Level-1 milestone (measurable improvement over CPU-only
  baseline) DEMONSTRATED. Next: E5 concurrent CPU+GPU execution.

## E5a — Task-parallel CPU binary splitting (2026-11-05)

- Hypothesis: a fork-join task scheduler over the *same* exact BS tree (bit-identical
  results) gives near-linear split speedup on 32 threads, since subtrees are independent.
- Implementation (`bs_parallel` in src/pi_chudnovsky.cpp): task queue + condvar, worker
  pool (caller included); leaves >= span_min sized for ~8x thread count; each leaf task
  owns its own Ctx scratch (zero sharing); combine continuations fire on a shared atomic
  counter when both children finish. Gated to `backend == cpu` (CUDA backend uses one
  global workspace; E5b needs CUDA streams).
- Correctness: 1M digits, threads 1..32, all six outputs byte-identical (single md5);
  10M T16 --verify PASS; 100M T1 output byte-identical to T16/T32 outputs.
- Scaling @ 10M digits (post set_u128 fix, single trial each):

  | threads | split s | total s | bs speedup |
  |---|---|---|---|
  | 1  | 2.515 | 3.781 | 1.0x |
  | 2  | 1.478 | 2.740 | 1.7x |
  | 4  | 1.046 | 2.302 | 2.4x |
  | 8  | 0.808 | 2.066 | 3.1x |
  | 16 | 0.725 | 1.982 | 3.5x |
  | 32 | 0.714 | 1.968 | 3.5x |

  Split saturates ~3.5x because the tree's *top* levels are serial (only 1-2-4... ready
  tasks early); Amdahl on the critical path dominates, not memory bandwidth. Total speedup
  1.9x (convert+sqrt+final ~1.26 s serial). 100M scaling: see BENCHMARKS.
- Decision: RETAIN. Deeper win requires *cost-balanced* splitting (unequal spans so the
  serial spine shrinks) — logged as E5c candidate.

## BUG-1 post-mortem — silent P corruption for D >= ~10M (found 2026-11-05, FIXED)

- Symptom: 100M result 92.95M digits with wrong leading digits; deterministic across
  runs and thread counts. Length correct up to D=48M, wrong above ~49M. 20M/40M --verify
  showed exactly 1 BBP-position failure (leading digits still correct).
- Diagnosis chain: per-node digit-count instrumentation (TACHYON_DEBUG_SIZES) vs exact
  lgamma forecasts localized corruption to *every leaf containing k >= 635131*, the exact
  k where p_k = (6k-5)(2k-1)(6k-1) crosses 2^64. Q was always exact.
- Root cause: `set_u128` used `mpz_import(..., order = -1, ...)` with limbs {hi, lo}.
  mpz_import's `order` reverses *iteration order*, not limb significance: with order=-1
  the word array is read little-endian, so v >= 2^64 became lo*2^64 + hi. The swapped
  digit-count forecast matches measurement digit-for-digit (e.g. leaf [705148,705154):
  got 229, swap-predicted 229, correct 116).
- Blast radius: P(0,N) inflated by 10^(14.18 * (N - 635130)) digits; T = Tl*Qr + Tr*Pl
  only visibly corrupted once the P excess exceeded the half-tree term count (~49M
  digits); every mul touching P was also doing ~14-39% extra word work (100M split
  57.7 s -> 43.8 s after fix).
- Fix: build the 128-bit value directly (mpz_set_ui / mpz_mul_2exp / mpz_add_ui), no import.
- Regression: ALL PASS — run_tests.sh (incl. 1M Machin cross-check), ctest 3/3,
  --verify at 10M/20M/40M/50M/100M (0 failures), parallel path T16.
- Lesson: mpz_sizeinbase on intermediates is a cheap invariant tripwire; add the
  P/Q/T digit-count assertion permanently to a debug test. (Fixed value now: 100M
  total 62.76 s, split 43.8 s, RSS 2076 MB.)

## E6 + BUG-2: parallel decimal conversion lands; dead 10.6 s serial conversion removed (2026-11-04)

- E6 (`mpz_get_str_parallel`, fork-join D&C over a shared 10^(2^k) pow table, 8192-digit
  leaves via mpz_get_str) was already coded but unmeasured; output bit-exactness confirmed
  by `convert-test` (1M/10M: byte-identical to mpz_get_str, PASS).
- Measured conversion speedup at 32 threads: 1M 0.036->0.019 s (1.9x), 10M 0.631->0.234
  (2.7x), 100M 10.61->3.51 s (3.0x). Below-linear by design: the serial pow-chain
  (sum ~2*M(10^D)) plus top-level split bounds parallelism to ~3-4x at this scale.
- BUG-2 found while measuring: `compute_pi` always ran a *throwaway* serial
  `mpz_get_str` purely to fill `tm.convert`, and production paths (main, dashboard jobs)
  then converted *again* in parallel. At 100M that is 10.6 s of discarded work per run
  (~1/3 of total wall). Fix: `PiOptions::time_convert` (set only under --benchmark), and
  the timed throwaway now uses the parallel converter so phase timings reflect reality.
- Post-fix 100M T32: bench total **25.41 s** (split 12.83, sqrt 3.89, final 5.18, convert
  3.51), normal `--out` run 26.1 s wall *including* 100 MB disk write; --verify PASS.
  10M T32 1.588 s, 1M T32 0.106 s. Decision: KEEP both changes.

## Planned next

- E5b: CUDA-stream concurrency — dispatch GPU muls on non-blocking streams while CPU
  workers chew independent subtrees (the real hybrid value; today hybrid ≡ gpu serial).
- E5c: cost-balanced splitting — choose split points so the serial spine holds a smaller
  share of total work (unequal spans sized by estimated mul cost), attacking the ~3.6x
  split-scaling ceiling found in E5a.
- E6: DONE (above). Further convert win possible: cache the 10^(2^k) pow table across
  runs in server mode (~1 s at 100M) and push fork depth past log2(threads).
- E7: transform-reuse DAG (H3): memoized forward transforms across the 3 products per combine.
- Re-measure GPU/hybrid end-to-end rows post BUG-1-fix (E4 rows carry the oversized-P penalty).
