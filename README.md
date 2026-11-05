# TACHYON-π

A CPU+GPU arbitrary-precision π computation engine, built as an open experimental
platform for testing whether heterogeneous, hardware-aware computation graphs can
beat fixed-strategy big-integer pipelines. Target hardware: AMD Ryzen 9 9950X
(16C/32T, AVX-512) + NVIDIA RTX 5070 Ti (16 GB) + 128 GB DDR5.

Current best: **20.5 s @100M digits** and **318 s @1B digits** (BBP-verified),
on a 16-core/32-thread CPU; a CUDA exact-NTT mul backend wins single-threaded
by up to 1.8x below 30M digits. See `RESEARCH.md` (hypotheses + decision log),
`EXPERIMENTS.md` (ledger), `BENCHMARKS.md` (numbers), `ARCHITECTURE.md`
(design + error model).

## Build

```sh
sudo apt install libgmp-dev cmake build-essential   # CUDA optional
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Use

```sh
./build/tachyon-pi --digits 1000000 --verify          # compute + BBP spot checks
./build/tachyon-pi --digits 10000000 --benchmark --trials 5 --json out.jsonl
./build/tachyon-pi --digits 100000000 --threads 32 --out pi-100M.txt
./build/tachyon-pi --digits 30000000 --backend gpu    # big muls on the CUDA NTT
./build/tachyon-pi --digits 10000 --leaf 32 --guard 40 --quiet
./build/tachyon-pi --serve 8080                       # web dashboard (see below)
```

Options: `--digits N` (significant digits, N≥2) · `--backend cpu|gpu|hybrid` ·
`--verify` · `--benchmark --trials K` · `--json FILE` · `--out FILE` ·
`--threads T` (**default 1** — pass `--threads 32` for the multicore numbers) ·
`--leaf L` · `--guard G` · `--gpu-mem MB` · `--quiet` · `--serve [PORT]`
(default 8080).

## Web dashboard

`./build/tachyon-pi --serve 8080` starts a zero-dependency HTTP + SSE server
(listens on `127.0.0.1` only) that serves an embedded single-page dashboard at
`http://127.0.0.1:8080/`. Tabs: **Run** (submit live jobs, watch per-phase
progress over SSE), **Mapping** (cost-model device-placement DAG per digit
target), **Digits** (per-job head/tail, digit-frequency histogram, pattern
search), **Benchmarks** (log-log speed chart vs the y-cruncher reference).

JSON/SSE API: `GET /health` · `GET /api/jobs` · `GET /api/job?id&what=meta|digits|stats` ·
`GET /api/stream?id=N` (SSE) · `POST /api/run {"digits","backend","verify",...}` ·
`GET /api/mapping?digits&backend` · `GET /api/benchmarks` · `GET /api/history`.
UI-submitted runs are appended to `benchmarks/server_runs.jsonl`. The Mapping
cost model is `cost-model-v2`, calibrated against the E4 crossover measurements
(GMP per-mul fit within 5%, GPU butterfly throughput fit at the 2^24-product row;
model total within 1.2x of the measured 10M-digit split phase).

Output format: `3.` followed by (N−1) decimals + newline; `--out` files are
byte-comparable across runs and against `machin-check`.

## Architecture (as built)

π = 426880·√K·Q / T via Chudnovsky binary splitting (K = 1000522583297352258426438131),
in four phases:

| phase | implementation |
|---|---|
| split tree | CPU task pool (E5): 32-bit-limb products folded in u64/u128 leaves, big products via the engine's mul — home-grown 32-lane AVX-512 parallel Karatsuba (`pmulp`, E7) at ≤2^28-bit operands, `mpz_mul` (GMP, FFT) above |
| sqrt(K)·Q scale | `mpz_sqrt` (GMP asm; E7c proved parallel reciprocal-sqrt loses ~1.9x at production sizes) |
| final num-mul + division | `pmulp` for the num-mul; `mpz_tdiv_q` (GMP divide-and-conquer) for the division (E7c) |
| decimal conversion | 3-level divide-and-conquer, parallel (E6, 3.0x) |

`--backend gpu|hybrid` swaps the *big multiplications* onto a CUDA exact-NTT
engine: three-prime NTT (167772161 / 469762049 / 754974721) over 28-bit limbs
with Garner mixed-radix CRT — coefficient bound 2^80 < M ≈ 2^85.6, so
reconstruction is exact and bit-identical to `mpz_mul` (no floating point
anywhere). GPU-side tricks: fused pointwise+1/N kernel, one half-size twiddle
table per prime read backwards for the inverse (E4), chunk-parallel carry
resolution (the naive serial carry was 94.5 % of GPU time at N = 2^24),
`__brev` instead of an LUT. E8 (in progress): per-prime 3-stream overlap,
pinned host staging, device-resident cached twiddle tables (~25 ms saved per
N = 2^26 mul), and a 250 ms-cached memory-budget gate so the driver isn't
polled on every mul. N > 2^24 recurses through Karatsuba into GPU-sized
products; anything that doesn't fit the live VRAM budget (or N ≤ 2^15) falls
back to CPU `mpz_mul`, so the backend self-adapts to a GPU shared with other
processes.

`bench/gpu_bench.cpp` measures the raw mul backend; `bench/mul_bench.cpp` the
CPU side. `--gpu-mem MB` caps the device budget.

## Correctness

Three independent layers, all passing (`bash tests/run_tests.sh`):
1. BBP hex-digit spot checks with per-digit rigorous error bounds (any size, never
   false-passes; inconclusive is reported as such).
2. Byte-exact diff against `machin-check` — Machin's formula evaluated with pure
   integer recurrences (fully independent summation) — at 1M digits.
3. Embedded reference for the first 100 digits; leaf-size invariance checks.
Exactness model and guard-digit derivation: `ARCHITECTURE.md`.

## Records (same machine, medians of 3 after cooldown, all BBP-verified)

Best per size (CPU, `--threads 32`, no verify/disk-write):

| digits | TACHYON best | y-cruncher bar (same CPU) | gap |
|---|---|---|---|
| 1M | 0.095 s | 0.011 s | 8.6x |
| 10M | 1.117 s | 0.115 s | 9.7x |
| 100M | **20.48 s** (split 9.60, sqrt 3.66, final 3.81, convert 3.32) | 1.814 s | 11.3x |
| 1B | **317.9 s** (verify PASS, peak RSS 29.2 GB) | 14.67 s | 21.7x |

(y-cruncher also lists a 1.107 s @100M mode; we cite its conservative
normal-mode figures.) Single-thread CPU reference: 1M 0.24 s, 10M 3.81 s,
100M 63.8 s. With `--backend gpu` (single thread, E4-era build): 10M 2.42 s
and 30M 9.13 s — **1.6–1.8x faster than the same-thread CPU path**. The GPU
advantage shrinks with size: parity at 200M, then 3.8x behind at 500M, because
the CUDA NTT caps at N=2^24 limbs and larger products fall back to device
Karatsuba (see `BENCHMARKS.md` crossover table). Combining CUDA muls with the
32-thread split tree (E5b/E8) is in progress and needs a GPU window free of
other workloads.

Measurement hygiene (learned the hard way, see `EXPERIMENTS.md`): concurrent
compiles, back-to-back runs, or an actively generating local LLM inflate T32
totals by 2–4 s / ~25 % (e.g. 100M measured 26.3 s with llama-server active
vs 20.5 s quiet); always cool down between runs and say which config produced
a number. Honest summary: no published world record has been *broken* — we
report measured deltas (100M went 62.8 s single-thread → 30.9 s at E5a/32T →
25.4 s at E6 → 20.5 s at E7c),
not record claims.
