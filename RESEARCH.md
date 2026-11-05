# TACHYON-π — Research Notes & Strategy

Hardware: AMD Ryzen 9 9950X (16C/32T, AVX-512 F/BW/DQ/VL/IFMA/VBMI/BF16), NVIDIA RTX 5070 Ti
(16GB GDDR7, sm_120, ~700GB/s bandwidth, ~1.4GB VRAM free while the local LLM is resident),
128GB DDR5, Debian, GCC 14.2, CMake 3.31, CUDA 13.1, GMP 6.3, FFTW3.

## 1. SOTA targets (measured elsewhere, same-CPU where possible)

| System | 100M digits | 1B digits | source |
|---|---|---|---|
| y-cruncher v0.8.5, Ryzen 9 9950X (same CPU, 48GB RAM) | **1.107s** | **14.670s** | numberworld.org/y-cruncher/benchmarks/charts/100m.html, 1b.html |
| y-cruncher best (2× Xeon 8488C) | 0.531s | 5.149s | same charts |
| gpupi, RTX 5090 GPU (compute-only) | 801ms @250M | **3509.7ms** | github.com/fanhqme2/gpupi (self-reported, unverified) |

→ Level-2 target on THIS machine: beat y-cruncher's 1.107s @100M and 14.67s @1B.
→ RTX 5070 Ti has ~44% of 5090's bandwidth: gpupi-equivalent alone likely ~7–8s @1B ⇒
hybrid CPU+GPU overlap is where the edge must come from.

## 2. Algorithm research matrix

| Method | Complexity | Parallelism | GPU fit | Verdict |
|---|---|---|---|---|
| Chudnovsky + binary splitting | O(M(n) log² n) | tree-parallel, big mults at top | excellent (few huge mults) | **Primary engine** |
| Gauss–Legendre / Brent–Salamin | O(M(n) log n) but ~log n full-precision iterations, each needs full-precision mul+sqrt | mostly serial chain | poor (serial deps) | Reject as primary |
| Borwein quartic | same class as AGM | serial | poor | Reject |
| Ramanujan–Sato | same as Chudnovsky, 14.18 vs ~14.18 digits/term | same | same | No advantage (Chudnovsky is optimal-class of these) |
| BBP/Bellard digit extraction | O(M(d) log n) per block of digits | trivially parallel across blocks | good but re-computes per block | Verification-only role |
| CPU FFT/NTT mult | M(n)=O(n log n) | SIMD + threads | — | Required backend |
| Exact single-prime NTT (96-bit, 2^96−2^32+1, à la gpupi) | O(n log n), exact | massive thread parallelism | **best fit for consumer GPU** (1 transform buffer per product, no CRT ×3 of 31-bit primes) | Core GPU backend |
| Float FFT w/ error bound (Borwein) | O(n log n) | good | cuFFT availability | Risky bounds at 4e9-bit sizes; exact NTT preferred |
| Multi-prime CRT NTT (y-cruncher style, 3–4×31-bit) | same asymptotics | good | 3–4× transform traffic vs single 96-bit prime | Inferior for VRAM-bandwidth-bound GPU |
| Newton reciprocal/div/sqrt | O(M(n)) | low | usable | Needed for final div/sqrt; keep on CPU (GMP is excellent) |
| RNS/Rader | — | — | — | Subsumed by NTT choices |

Convergence note (rule: faster convergence ≠ cheaper): Chudnovsky gives 14.18 digits/term;
AGM doubles bits per iteration requiring ~2.5 full-precision squarings per iteration ⇒
total work ~O(M(n) log n) with a large constant and a serial chain. Binary splitting's
tree exposes top-heavy parallelism that maps exactly onto 32 CPU threads + 1 GPU. Settled: Chudnovsky.

## 3. Key prior-art lessons (from gpupi / y-cruncher internals)

1. **Single large modulus beats CRT-of-small-primes on bandwidth-limited GPUs**: one
   96-bit-prime transform pipeline moves ~1/3 the data of three 31-bit-prime pipelines.
   gpupi uses p = 2^96 − 2^32 + 1 with PTX carry-chain reduction.
2. **Transform sharing**: in binary splitting the same subtree product is squared/multiplied
   repeatedly; caching forward transforms (and fused pointwise inverse) removes ~1/3 of transform work.
3. **Common-factor cancellation** in blocks (gpupi: 136-term blocks; Chudnovsky term factors
   share heavily) shrinks all products ⇒ directly reduces every downstream multiply.
4. y-cruncher uses size-tiered mult: basecase → Karatsuba → float-FFT → NTT/VST, AVX-512 tuned,
   and divide-and-conquer radix conversion. Its 9950X numbers are the bar.
5. Decimal conversion & final sqrt/div are NOT negligible at 100M–1B digits; measure phases separately.

## 4. Three highest-value hypotheses

**H1 (core engineering, high confidence):** *Heterogeneous cost-balanced binary splitting.*
A work-stealing scheduler evaluates the splitting tree with cost-weighted dispatch: large
mults (≥ threshold T, tuned by autotune) go to the GPU exact-NTT pipeline, mid-size to
AVX-512-friendly GMP on CPU cores, and CPU+GPU run **concurrently** on independent subtrees
(pipelined via CUDA streams + pinned memory). Novelty vs gpupi (GPU-serial pipeline) and
y-cruncher (CPU-parallel or GPU-serial): the *joint overlap schedule* + measured dispatch
threshold. Falsifier: if PCIe 5.0 transfer + launch overhead > GPU speedup for all sizes,
GPU only helps at ≥1B digits; measure the crossover.

**H2 (arithmetic, medium-high confidence):** *Single-prime 96-bit exact NTT tuned for sm_120.*
p = 2^96−2^32+1 (or 2^96−512·2^24+1 style), radix-2/4 decimation-in-frequency with shared
forward transforms, fused pointwise-multiply+inverse, in-register carry-chain modular ops.
Target ≥3× GMP's large-mul throughput per watt-hour and ≥2× on wall time for ≥2^26-limb
products. Falsifier: measured mul-throughput curve vs GMP.

**H3 (unconventional, medium confidence — the ambitious bet):** *Algebraic contraction of the
splitting tree: "folded leaves + fused 2×2 matrix products with transform reuse".*
Binary splitting for Chudnovsky evaluates products of 2×2 matrices M_k=[[a_k,0],[b_k,t-acc]]…
observation: the matrices are lower-triangular ⇒ each combine costs 3 big mults (a·a', t=a·t'+b·t'·†)
not 4 (already known). The novel part: (i) *fold* s consecutive leaves symbolically into one
2×2 matrix using only small-int arithmetic before entering the big-int domain (exact common-factor
cancellation inside the fold ⇒ strictly smaller big operands — quantify bit-reduction: expected
~O(s) bits saved per fold vs naive, measurable via term-growth analysis), (ii) *schedule transform
reuse across the three products of every combine* — the same operand is forward-transformed once
and consumed by several pointwise products + inverse choices within one kernel graph, a
"transform memoization DAG" (this is the graph-based answer to the Central Innovation Challenge:
the DAG's value = Σ saved transforms, provable ≥ 1/3 of transform work when reuse factor ≥2),
(iii) precision-aware truncation of the t-accumulator branches (t needs only D+guard bits;
b-branches are exact — truncation is provably safe since t's error only affects the final
quotient beyond guard digits; derive guard bound). Falsifier: if fold+cancellation yields <5%
bit reduction at 100M digits, or transform memoization gains <10% at realistic cache sizes, reject.

## 5. Verification plan

- ≤1M digits: diff against canonical pi-million.txt (MIT project, md5 d1a2d700b5a8440597e1c6568bfb6419 — recompute locally) and Python mpmath.
- Every run: BBP (Bailey–Borwein–Plouffe) hex-digit extraction at 3 positions (independent formula & summation order, long-double, ~13-hex-digit guard analysis) + Bellard variant.
- ≤1e6: independent Machin-style π/4 arctan series with integer arithmetic (fully independent of Chudnovsky).
- Large: gcd/bit-invariants, plus cross-backend agreement (CPU vs GPU must produce identical digits).
- Guard digits: fixed +20 decimal guard through series; error bound derived in ARCHITECTURE.md.

## 6. Decision log (running)

- 2026-11-03: Chudnovsky binary splitting chosen as sole primary; AGM/Borwein rejected (serial chain; §2).
- 2026-11-03: Exact single-prime NTT chosen over float-FFT (no error-bound proof needed) and over 31-bit CRT (VRAM bandwidth). GPU model resident: must cap VRAM usage (--gpu-mem) and provide auto CPU fallback.
- 2026-11-03: Baseline = GMP single-thread first, then hetero scheduler. Benchmark phases: split / sqrt / div / convert separately.

## 7. Correctness corrections log (baseline bring-up, 2026-11-03)

Five math/API bugs found and fixed in the first working Chudnovsky engine (all verified
against exact Python/Fraction references and the canonical first 100 digits):

1. Leaf fold order: must be `T_k = T_{k-1}*q_k + (-1)^k (A+Bk) P_k` with P folded *first*
   (term uses cumulative P_k, not per-step p_k).
2. Term numerator is `p(k) = (6k-5)(2k-1)(6k-1)` (not (6k-3)); ~3x bit growth per term otherwise.
3. Combine `T = T_l*Q_r + T_r*P_l` (was using R.P where R.Q belongs).
4. Closing constant: `pi = 426880*sqrt(10005)*Q/T`. The constant 1000522583297352258426438131
   belongs to the BBP formula, not Chudnovsky's closing. Proof: 640320^3 = 144*426880^2*10005.
5. `mpz_import` with order=-1 wants most-significant limb FIRST (set_u128 limb order).

Release-only heap crash (segfault / "double free or corruption (out)" under -O2/-O3, clean
under ASan - classic uninitialized-heap-pointer signature, layout-dependent): root cause was a
missing `mpz_init(rem)` in bbp_check_hex — an uninitialized stack mpz_t whose garbage `_mp_d`
happened to be a freeable-looking pointer under ASan's allocator pattern. Backtraced via
MALLOC_CHECK_=3 + execinfo in a 20-line repro (no gdb available). Fix: mpz_init(rem).

Also fixed: machin_check.cpp series started at u_0 = 10^P instead of floor(10^P/m)
(missing 1/m factor); produced ~1.179 instead of pi.

BBP design note: kNC capped at 11 verified hex digits — the rigorous error bound grows as
16^i * 16(K+2) * 2^-63, which exceeds ~1e-3 past i~11 in long double, making deeper digits
routinely inconclusive (return -1, never a false pass).

## 6a. Decision log (continued)

- 2026-11-03: Baseline PASSED full suite: 101/10k/100k BBP checks, 1M-digit byte-exact
  cross-check vs independent Machin integer-arithmetic checker, leaf-size invariance.
- 2026-11-03: E1 leaf sweep at 10M digits: 3.87-3.99s for leaf in {1..256}, flat within noise.
  Binary-splitting tree shape is NOT a lever when GMP mul dominates; keep leaf=8 default.
- 2026-11-03: Baseline perf (single-thread, medians of 5): 1M digits 0.207s; 10M digits 3.88s
  (bs 67%, convert 16%, final 10%, sqrt 7%). y-cruncher bar (same machine, multithreaded):
  ~1.1s @100M. Single-thread hybrid's realistic target: match y-cruncher at <=10M, stay
  within 2-3x at 100M; the NTT backend is the main lever.
