# TACHYON-π

A CPU+GPU arbitrary-precision π computation engine, built as an open experimental
platform for testing whether heterogeneous, hardware-aware computation graphs can
beat fixed-strategy big-integer pipelines. Target hardware: AMD Ryzen 9 9950X
(16C/32T, AVX-512) + NVIDIA RTX 5070 Ti (16 GB) + 128 GB DDR5.

Status: **stage 2 complete** — verified single-thread CPU Chudnovsky baseline;
CUDA exact-NTT backend and hybrid scheduler in development. See `RESEARCH.md`
(hypotheses + decision log), `EXPERIMENTS.md` (ledger), `BENCHMARKS.md` (numbers),
`ARCHITECTURE.md` (design + error model).

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
./build/tachyon-pi --digits 100000000 --out pi-100M.txt
./build/tachyon-pi --digits 10000 --leaf 32 --guard 40 --quiet
./build/tachyon-pi --serve 8080                       # web dashboard (see below)
```

Options: `--digits N` (significant digits, N≥2) · `--backend cpu|gpu|hybrid` ·
`--verify` · `--benchmark --trials K` · `--json FILE` · `--out FILE` ·
`--threads T` · `--leaf L` · `--guard G` · `--gpu-mem MB` · `--quiet` ·
`--serve [PORT]` (default 8080).

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

## Correctness

Three independent layers, all passing (`bash tests/run_tests.sh`):
1. BBP hex-digit spot checks with per-digit rigorous error bounds (any size, never
   false-passes; inconclusive is reported as such).
2. Byte-exact diff against `machin-check` — Machin's formula evaluated with pure
   integer recurrences (fully independent summation) — at 1M digits.
3. Embedded reference for the first 100 digits; leaf-size invariance checks.
Exactness model and guard-digit derivation: `ARCHITECTURE.md`.

## Headline baseline (single thread, medians of 5, 2026-11-03)

| digits | total | digits/s |
|---|---|---|
| 1M | 0.207 s | 4.8M |
| 10M | 3.88 s | 2.6M |
| 100M | 89.0 s | 1.1M |

y-cruncher reference bar on the same machine: ~1.1 s @100M (multithreaded).
