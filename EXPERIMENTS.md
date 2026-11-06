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

## E7 / E7b / E7c: parallel Karatsuba mul lands; parallel reciprocal tail REJECTED (2026-11-05)

- E7 hypothesis: a fork-join Karatsuba multiplication (pmulp, 3-limb base, threshold
  2048 limbs, global thread pool) inside eng.mul would accelerate every large product
  (split tree, final num-mul, sqrt operand) with one mechanism.
- E7 first integration REGRESSED 100M --verify to **63.25 s**. Two root causes found:
  (1) the newton-reciprocal tail helpers (rec_par/divrec/sqrtrec) never scale-doubled:
      the per-level shift was `m+GUARD` instead of `GUARD`, so x stayed at scale n+32
      while D·x was taken at scale n+m -> the ladder converged but ~3x slower than
      needed, and the final exact fix-up (V = 2^2n - D·R, c = fdiv_q(V,D)) silently
      absorbed any residual error (correct-but-slow masked the bug for weeks);
  (2) after fixing the ladder, a post-loop boost `mpz_mul_2exp(x,x,m-GUARD)`
      double-counted the scale and overshot x by ~2^(m-32), making the exact
      fix-up divide a ~2^n-sized V by D (full extra division) instead of ~n bits.
- Probe facts at nb=2^28 (session probe_tailperf / probe_div, probe_div PASS vs
  current source): mpz_tdiv_q 2.154 s vs full reciprocal divrec 3.90 s (par-mul);
  mpz_sqrt 2.254 s vs sqrtrec 4.24 s (hybrid par L² + GMP e-div) / 7.62 s (full).
  GMP's internal divide-and-conquer division does ~1/3 the multiplication work of
  a reciprocal ladder plus exact fix-up at these shapes; the production final
  division (num ~1.5x bits of den, quotient ~nb/2 bits) took the FULL ladder path.
- E7c dispatch decision (measured): production tail = GMP (mpz_sqrt, mpz_pow_ui,
  plain mpz_tdiv_q; divrec reciprocal path kept only when quotient <= nb/8, which
  production never hits). `tail-test` harness (differential vs GMP, all sizes/
  threads) PASSES at 32 threads and stays in-tree as regression cover for the
  still-used helpers. KEEP parallel mul (pmulp) in split tree, final num-mul, and
  conversion; REJECT reciprocal-based tail for production.
- Note mpz_pow_ui is also faster than the parallel pow_par here (+1.2 s penalty in
  sqrt phase at 100M when it was enabled): exponent 128 is tiny, so pow_par's
  fork/join overhead exceeds its parallel win.
- Post-E7c clean benchmarks (cooldown between runs, T32, median of 3, no verify,
  no disk write; variance < 0.5 %): 1M **0.095 s** (was 0.106), 10M **1.117 s**
  (was 1.588; -30 %), 100M **20.48 s** (was 25.41; -19 %: split 9.60, sqrt 3.66,
  final 3.81, convert 3.32; peak RSS 5.00 GB). --verify PASS at 10M (0.91 s
  phases) and 100M (20.6-21.6 s incl. contention runs). Measurement hygiene:
  back-to-back 100M runs and concurrent compiles inflate totals by 2-4 s (thermal
  + scheduler contention); the 24.3 s median seen mid-session was contention,
  20.5 s is reproducible after cooldown.
- Gap to y-cruncher @100M narrows 14.0x -> **11.3x** (20.48 vs 1.814); @10M
  13.8x -> 9.7x; @1M 9.7x -> 8.6x.

## CPU exact-NTT prototype: exactness PASS, scalar performance REJECTED (2026-11-05)

Added `src/cpu_ntt.cpp` as an opt-in three-prime exact NTT backend using the
same 28-bit representation and CRT proof as the CUDA path. `cpu-ntt-test`
passes signed products through the maximum supported transform tier
(N=2^24). It is deliberately gated behind `TACHYON_CPU_NTT=1` and is not on
the production path.

The first scalar OpenMP implementation is substantially slower than the
existing fork-join Karatsuba multiplier: at 2^26 operand bits, NTT was 1.95 s
vs pmulp 0.063 s; at 2^27 bits, 2.73 s vs 0.164 s. The 2^28-bit NTT probe
falls back because the required transform exceeds N=2^24. Root cause is the
per-butterfly scalar modular multiply/reduction and stage barriers, not the
exact CRT reconstruction. A useful CPU record attempt still requires
Montgomery/Barrett arithmetic plus AVX-512 vectorized butterflies (issue #1).

## E6 follow-up: cached decimal power table (2026-11-05)

`mpz_get_str_parallel` now retains the largest `10^(2^k)` table in-process,
so repeated server/benchmark conversions do not rebuild the serial squaring
chain. Conversion at 100M T32 measured **3.05 s** (previous quiet rows were
about 3.3 s); the total was 20.81 s in that loaded run, so this is a useful
tail reduction but not a new end-to-end record. The cache is immutable while
workers read it and replaces the previous per-call table lifetime.

## Four-prime GPU tier: verified 500M result (2026-11-05)

The fourth-prime exact NTT tier builds and passes direct multiplication tests
through N=2^25, including skewed operands. End-to-end GPU verification passes
at 200M, 300M, 400M, and **500M** digits. The verified 500M run takes
**186.95 s** excluding decimal conversion; the benchmark run including
conversion takes 251.21 s.

The prior failed 500M verification was from an earlier build/state. In the
current build, CPU/GPU split-tree fingerprints match exactly and an exhaustive
GMP oracle over every GPU-sized product (`N>=2^25`) finds no mismatch. The
four-prime tier therefore closes the old 500M GPU cliff: 186.95 s verified is
2.67x faster than the 498.7 s CPU/T1 baseline and 10.3x faster than the old
1917.2 s GPU fallback.

The same backend with `--threads 32` and hybrid dispatch improves the verified
500M result to **162.50 s** (split 110.41, sqrt 22.56, final 29.53), BBP PASS;
including decimal conversion the benchmark is **184.83 s**. A low-risk LIFO
queue experiment regressed hybrid 100M from 18.22 s to 21.87 s, so FIFO remains
the production scheduler. E5c still needs a cost-aware priority scheduler,
not a simple queue-order swap.

### E5c task-granularity sweep

Added the opt-in `TACHYON_TASKS_PER_THREAD` scheduler knob. At 100M hybrid/T32,
the measured medians were: factor 4 = 18.50 s, factor 8 = 18.35 s, factor
16 = **17.77 s**, factor 32 = 19.32 s. The default remains factor 8 because
factor 16 was not yet revalidated at 500M before resources were released; the
result is retained as the next-session tuning lead, not a new record.

## Planned next

- E5b: MEASURED after fixing a scheduler gate — `backend=hybrid` was accidentally
  excluded from `bs_parallel`, making the original hybrid matrix serial. Corrected
  T32 medians: hybrid 1.215/4.228/17.512 s vs CPU 1.127/4.331/20.878 s at
  10M/30M/100M; hybrid wins at 30M and 100M, but 10M remains launch-bound.
- E5c: cost-balanced splitting — choose split points so the serial spine holds a smaller
  share of total work (unequal spans sized by estimated mul cost), attacking the ~3.6x
  split-scaling ceiling found in E5a.
- E6: DONE (above). Further convert win possible: cache the 10^(2^k) pow table across
  runs in server mode (~1 s at 100M) and push fork depth past log2(threads).
- E7: DONE (above). Transform-reuse DAG (H3): memoized forward transforms across the
  3 products per combine still unexplored.
- E8 (record path): own parallel AVX-512 FFT/NTT multiplication on CPU. GMP's
  Schönhage-Strassen is single-threaded at these sizes (split phase 9.6 s of 20.5 s
  total is dominated by ~332M-bit muls); a 32-thread FFT mul is the only realistic
  route toward y-cruncher's 1.814 s @100M.
- 1B-digit feasibility: first run in progress (see BENCHMARKS.md).
- Re-measure GPU/hybrid end-to-end rows post BUG-1-fix (E4 rows carry the oversized-P penalty).
  DONE 2026-11-05: T32 E5b matrix in `BENCHMARKS.md` and
  `benchmarks/e5b_gpu_matrix_20261105.jsonl`; GPU wins at 100M, corrected hybrid
  wins at 30M and 100M. T1 200M/500M rows remain the large-size crossover evidence:
  parity at 200M and the four-prime tier now passes 500M; further GPU work can
  target larger transforms and reducing the remaining sqrt/final/convert phases.
