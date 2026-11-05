# TACHYON-π architecture

## Module map

```
src/
  main.cpp            CLI: options, timing, phase report, verify orchestration, JSON records
  pi_chudnovsky.*     π algorithm: term_count, binary-splitting tree (BSNode), closing
  engine.*            MulEngine abstraction: cpu-gmp today; CUDA NTT backend lands stage 3
  verify.*            BBP hex-digit spot checks with rigorous per-digit error bounds
  tests/
    machin_check.cpp  independent π (Machin formula, integer recurrences) for byte-exact diffs
tests/run_tests.sh    correctness suite (smoke, BBP, 1M Machin diff, leaf invariance)
benchmarks/           results.jsonl (one JSON record per benchmark run)
RESEARCH.md           algorithm research matrix, hypotheses H1-H3, decision + corrections log
EXPERIMENTS.md        hypothesis → implementation → measured → decision ledger
BENCHMARKS.md         end-to-end numbers vs y-cruncher bar on this machine
```

## Data flow (CPU baseline)

1. **Term count:** N = (D + guard)/log10(10^? per Chudnovsky term ≈ 14.18142587) + 4.
2. **Binary splitting** over k ∈ [0, N): recursively split range; at leaves (≤ `leaf` terms)
   fold with machine-word arithmetic:
   - `p(k) = (6k−5)(2k−1)(6k−1)` (< 2^85), `q(k) = k³·10939058860032000`
   - leaf: `P*=p; T = T·q + (−1)^k (A+Bk)·P` (P folded first!), `Q*=q`
   - combine: `P = P_l·P_r`, `Q = Q_l·Q_r`, `T = T_l·Q_r + T_r·P_l`  (3 big mults)
3. **Closing:** `π = 426880·√10005·Q/T` — sqrt via Newton (GMP mpz_sqrtrem), exact integer
   division at scale 10^(D+guard+2); `out = floor(426880·Q·S / (T·10^(d2−D)))`.
   Identity: 640320³ = 144·426880²·10005 ⇒ the famous 1000522583297352258426438131 is BBP's,
   not Chudnovsky's closing constant.
4. **Output:** `out = floor(π·10^D)` (D+1 significant digits); `to_decimal` formats "3."+…
5. **Verification (optional, `--verify`):** BBP hex spot checks (below) + embedded 100-digit
   reference; `tests/machin_check` for byte-exact cross-diffs at ≤1M digits.

## Guard-digit error model

BS is exact integer arithmetic ⇒ the only inexactness is the closing sqrt/division, computed
at working scale 10^(D+guard) with guard=30 decimal digits by default, and the final
truncation `floor(π·10^D)`. The integer result satisfies 0 ≤ π·10^D − out < 10^-(guard−3):
√10005 is computed to 10^(D+guard+2) scale (sqrtres error < 1 ulp), the multiplication/division
rounding adds < 4 ulp at that scale, so total error < 10^-27 · 10^D ⇒ prints exact floor
unless π·10^D lies within 10^(D−27) of an integer (probability ~10^-27; detectable by
increasing `--guard`).

## BBP spot check (verify.cpp) — what is proven vs tested

- Target side exact: R = (floor(π·10^D)·16^p) mod 10^D, goal = floor(R·16^kNC/10^D) gives
  the kNC hex digits of π at position p+1 exactly (guard: (p+kNC+2)·log10(16) ≤ D).
- Series side: Plouffe identity {16^p·π} = Σ mod 1 evaluated in long double with exact
  integer residues 16^(p−k) mod (8k+j) for k ≤ p (u64 modpow); tail bounded by 16^-(kNC+4).
- Each digit accepted only if distance to digit boundary exceeds the rigorous bound
  bnd_i = 16^i·16(K+2)·2^-63 + 10^(log10 16·(p+i)−D) + 16^(i−kNC−4); otherwise
  "inconclusive" (never a false pass). kNC=11: beyond that the bound exceeds ~10^-3.
- Status: this *tests* the decimal result against an independent formula at chosen offsets
  (positions {13, 1000, 50000, D/3+7}); a full-hexid digit-stream check is O(D) more work.

## Backend abstraction (stage 3 seam)

`MulEngine` (engine.hpp): `mul(mpz_t r, const mpz_t a, const mpz_t b)`, `name()`.
The BS tree calls only P/Q/T products through the engine ⇒ CUDA NTT swaps in without
touching the algorithm. Dispatch (stage 4) will be operand-size- and live-VRAM-aware;
`--gpu-mem` caps device memory (the resident LLM leaves ~1.4 GB of the 16 GB).

## Web frontend (http_server + jobs + embedded dashboard)

`--serve [PORT]` runs a hand-rolled POSIX HTTP/1.1 server (`http_server.cpp`,
thread-per-connection capped at 16, `127.0.0.1` only) that serves the embedded
single-page dashboard (`web_dashboard.hpp`, one raw-string `DASHBOARD_HTML`) plus a
JSON/SSE API. A `JobManager` (`jobs.cpp`) owns one worker thread that serializes
compute jobs and keeps mutable `JobSnapshot`s (phase + fraction + timing + head/tail
+ kept decimal up to 10,000,010 chars); the engine drives progress through the
`PiOptions::progress` callback. The dashboard polls `/api/jobs` for the job list and
opens `EventSource /api/stream?id=N` for live per-phase progress (SSE `data:` frames
every 200 ms until done/failed). The Mapping tab renders the E4-calibrated cost-model-v2 placement
DAG from `/api/mapping`; the Digits tab reads `/api/job?what=digits|stats`; the
Benchmarks tab plots `/api/benchmarks` (CLI) + `/api/history` (UI runs) against the
y-cruncher reference. The dashboard is dependency-free vanilla JS/SVG; the inline JS
is `node --check` clean and every `$()` id resolves in the HTML.

## Threading model

Baseline is single-threaded (deterministic, no sync noise). Planned: CPU-side parallel
subtrees (work-stealing over ready left/right children) + GPU stream overlap; scheduler
decides per-node by estimated cost vs measured throughput.

## Build

CMake ≥ 3.22, C++20, GMP required (`libgmp-dev`), CUDA optional (arch 120 auto for
sm_120/Blackwell). `cmake -B build && cmake --build build -j`. Tests: `bash tests/run_tests.sh`.
