// TACHYON-pi core: pi via Chudnovsky binary splitting (exact integer arithmetic).
//
// Formula (Chudnovsky brothers):  1/pi = 12 * sum (-1)^k (6k)!(A+Bk) / ((3k)!(k!)^3 C^{3k})
// with A=13591409, B=545140134, C=640320. Term ratios give the splitting factors
//   p(k) = -(6k-5)(2k-1)(6k-1),  q(k) = k^3 * C^3/24  (C^3/24 = 10939058860032000).
// Closing identity:  pi = 426880 * sqrt(10005) * Q(0,N) / T(0,N).
// Exact since C^{3/2}/12 = 426880^2 * 10005 with C = 640320 (verified:
// 640320^3 = 144 * 426880^2 * 10005).
//
// All arithmetic below is exact integers. The only inexact steps are the final
// integer sqrt (floor) and the two truncating divisions; the total error is
// < 10^-(guard-3) by the budget in ARCHITECTURE.md, so with guard=30 the first
// `digits` decimals are exact unless pi sits within 10^-29 of a decimal
// boundary at that scale (excluded empirically by BBP checks + reference diffs).
#include "pi_chudnovsky.hpp"
#include "convert.hpp"
#include "engine.hpp"
#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using ull = unsigned long long;
using u64 = unsigned long long;
using u128 = unsigned __int128;
using clk = std::chrono::steady_clock;

namespace {

constexpr ull A_CONST = 13591409ULL;
constexpr ull B_CONST = 545140134ULL;
constexpr ull C3_OVER_24 = 10939058860032000ULL;  // 640320^3/24, exact in u64

inline void set_u128(mpz_t z, u128 v) {
  // NB: mpz_import's `order` only reverses WORD ITERATION ORDER, not limb
  // significance; with order=-1 the array is still interpreted little-endian
  // (op[0] = least significant). Passing {hi,lo} silently swapped the halves
  // for every p_k >= 2^64 (k >= 635131, i.e. D >= ~10M) -- the 100M-digit
  // corruption bug. Build the value directly instead: no import at all.
  if (v >> 64) {
    mpz_set_ui(z, (u64)(v >> 64));
    mpz_mul_2exp(z, z, 64);
    mpz_add_ui(z, z, (u64)v);
  } else {
    mpz_set_ui(z, (u64)v);
  }
}

struct Node { mpz_t P, Q, T; };

// Scratch is indexed by recursion depth: zero mpz allocations inside the
// splitting tree itself (only the 64-deep pool is ever allocated).
struct Scratch {
  Node child[2];
  mpz_t t1, t2;
};

struct Ctx {
  std::vector<Scratch> sc;
  unsigned leaf = 8;
  MulEngine* eng = nullptr;
  // progress reporting (optional)
  const std::function<void(const char*, double)>* prog = nullptr;
  ull leaf_total = 1, leaf_done = 0, report_stride = ~0ull;
};

#ifdef TACHYON_DEBUG_SIZES
// Exact-enough (long double) digit forecasts for products over k in [a,b).
long double dbg_log10_P(ull a, ull b) {  // prod (6k-5)(2k-1)(6k-1)
  long double s = 0;
  for (ull k = a; k < b; ++k)
    s += log10l((long double)(6 * k - 5)) + log10l((long double)(2 * k - 1)) +
         log10l((long double)(6 * k - 1));
  return s;
}
long double dbg_log10_Q(ull a, ull b) {  // prod k^3 * C3_OVER_24
  long double s = 0;
  for (ull k = a; k < b; ++k)
    s += 3 * log10l((long double)k) + log10l((long double)C3_OVER_24);
  return s;
}
#endif

// Leaf: sequential fold of terms [a,b), b-a <= ctx.leaf.
void leaf(Node& out, mpz_t qtmp, mpz_t ttmp, ull a, ull b) {
  mpz_set_ui(out.P, 1);
  mpz_set_ui(out.Q, 1);
  mpz_set_ui(out.T, 0);
  for (ull k = a; k < b; ++k) {
    if (k == 0) {
      mpz_set_ui(ttmp, A_CONST);
      mpz_add(out.T, out.T, ttmp);
      continue;
    }
    u128 p128 = (u128)(6 * k - 5) * (2 * k - 1) * (6 * k - 1);  // <2^85 for k<2^40
    u64 lin = A_CONST + B_CONST * k;                            // <2^63 for k<2^33
    // P_k = P_{k-1} * p_k ;  T_k = T_{k-1} * q_k + (-1)^k * (A+B*k) * P_k
    mpz_set_ui(qtmp, k);                                        // q = k^3*C3_OVER_24
    mpz_mul_ui(qtmp, qtmp, k);
    mpz_mul_ui(qtmp, qtmp, k);
    mpz_mul_ui(qtmp, qtmp, C3_OVER_24);
    set_u128(ttmp, p128);
    mpz_mul(out.P, out.P, ttmp);                                // P *= p first
    mpz_mul_ui(ttmp, out.P, lin);                               // t = lin*P_k
    mpz_mul(out.T, out.T, qtmp);                                // T *= q
    if (k & 1) mpz_sub(out.T, out.T, ttmp);
    else mpz_add(out.T, out.T, ttmp);
    mpz_mul(out.Q, out.Q, qtmp);                                // Q *= q
  }
#ifdef TACHYON_DEBUG_SIZES
  {  // flag leaves whose P deviates from theory by > 6 digits
    long double t = dbg_log10_P(a, b);
    long double got = (long double)mpz_sizeinbase(out.P, 10);
    if (fabsl(got - t) > 6.0L)
      std::fprintf(stderr, "[leaf] [%llu,%llu) P got=%.0Lf want=%.0Lf d=%+.0Lf"
                   " | Q d=%+.1Lf\n", a, b, got, t, got - t,
                   (long double)mpz_sizeinbase(out.Q, 10) - dbg_log10_Q(a, b));
  }
#endif
}

void combine(Node& out, const Node& L, const Node& R, mpz_t t1, mpz_t t2,
             MulEngine& eng) {
  // T = Tl*Qr + Tr*Pl   (P = Pl*Pr, Q = Ql*Qr)
  eng.mul(out.P, L.P, R.P);
  eng.mul(t1, L.T, R.Q);
  eng.mul(t2, R.T, L.P);
  mpz_add(out.T, t1, t2);
  eng.mul(out.Q, L.Q, R.Q);
}

inline void ctx_init(Ctx& c, int pools) {
  c.sc.resize(pools);
  for (auto& s : c.sc) {
    for (auto& n : s.child) { mpz_init(n.P); mpz_init(n.Q); mpz_init(n.T); }
    mpz_init(s.t1); mpz_init(s.t2);
  }
}
inline void ctx_free(Ctx& c) {
  for (auto& s : c.sc) {
    for (auto& n : s.child) { mpz_clear(n.P); mpz_clear(n.Q); mpz_clear(n.T); }
    mpz_clear(s.t1); mpz_clear(s.t2);
  }
  c.sc.clear();
}

void bs(Node& out, Ctx& ctx, int depth, ull a, ull b) {
  if (b - a <= ctx.leaf) {
    leaf(out, ctx.sc[depth].t1, ctx.sc[depth].t2, a, b);
    if (ctx.prog && ++ctx.leaf_done % ctx.report_stride == 0)
      (*ctx.prog)("split", (double)ctx.leaf_done / (double)ctx.leaf_total);
    return;
  }
  ull m = (a + b) >> 1;
  Node& L = ctx.sc[depth].child[0];
  Node& R = ctx.sc[depth].child[1];
  bs(L, ctx, depth + 1, a, m);
  bs(R, ctx, depth + 1, m, b);
#ifdef TACHYON_DEBUG_SIZES
  if (depth <= 0)
    std::fprintf(stderr, "[n] d%d [%llu,%llu) L:Q=%zu T=%zu P=%zu | R:Q=%zu T=%zu P=%zu\n",
                 depth, a, b, mpz_sizeinbase(L.Q, 10), mpz_sizeinbase(L.T, 10),
                 mpz_sizeinbase(L.P, 10), mpz_sizeinbase(R.Q, 10),
                 mpz_sizeinbase(R.T, 10), mpz_sizeinbase(R.P, 10));
#endif
  combine(out, L, R, ctx.sc[depth].t1, ctx.sc[depth].t2, *ctx.eng);
}

// ---------------------------------------------------------------------------
// E5a: task-parallel binary splitting (threads > 1).
//
// Design: classic fork-join over the same exact tree, so results are
// bit-identical to the sequential path regardless of schedule. A task that
// covers more than `span_min` terms splits into two child tasks plus a
// combine continuation that fires when both children finish (atomic counter).
// Workers share one LIFO-ish queue; `span_min` is chosen so the number of
// leaf tasks is ~8x the worker count (load balance) while peak extra memory
// stays at one full P/Q/T working set (~3x output size), same order as the
// sequential path. Requires a thread-safe engine (GMP ops on distinct
// operands are); gated to the cpu backend until the GPU path adds streams.
struct ParCtx {
  MulEngine* eng = nullptr;
  ull span_min = 8, leaf = 8, N = 0, leaf_total = 1;
  std::atomic<ull> leaf_done{0};
  const std::function<void(const char*, double)>* prog = nullptr;
  std::mutex pmx;
  std::deque<std::function<void()>> q;
  std::mutex qmx;
  std::condition_variable cv;
  long outstanding = 0;  // under qmx: queued + running tasks
  bool stop = false;

  void push(std::function<void()> f) {
    { std::lock_guard lk(qmx); ++outstanding; q.push_back(std::move(f)); }
    cv.notify_one();
  }
  bool pop(std::function<void()>& f) {
    std::unique_lock lk(qmx);
    cv.wait(lk, [&] { return stop || !q.empty(); });
    if (q.empty()) return false;
    f = std::move(q.front());
    q.pop_front();
    return true;
  }
  void task_done() {
    std::lock_guard lk(qmx);
    if (--outstanding == 0) { stop = true; cv.notify_all(); }
  }
};

void run_sub(ParCtx& pc, Node* out, ull a, ull b, const std::function<void()>& done);

void spawn_sub(ParCtx& pc, Node* out, ull a, ull b, std::function<void()> done) {
  pc.push([&pc, out, a, b, done = std::move(done)]() mutable {
    run_sub(pc, out, a, b, done);
  });
}

void run_sub(ParCtx& pc, Node* out, ull a, ull b, const std::function<void()>& done) {
  if (b - a <= pc.span_min) {
    Ctx c;
    c.leaf = pc.leaf;
    c.eng = pc.eng;
    ctx_init(c, 32);  // recursion depth within one task is log2(span_min/leaf)
    bs(*out, c, 0, a, b);
    ctx_free(c);
    ull d = (b - a + pc.leaf - 1) / pc.leaf;
    ull done_n = pc.leaf_done.fetch_add(d) + d;
    if (pc.prog) {
      std::lock_guard lk(pc.pmx);
      (*pc.prog)("split", (double)done_n / (double)pc.leaf_total);
    }
    done();
    return;
  }
  ull m = (a + b) >> 1;
  Node* L = new Node;
  Node* R = new Node;
  mpz_init(L->P); mpz_init(L->Q); mpz_init(L->T);
  mpz_init(R->P); mpz_init(R->Q); mpz_init(R->T);
  auto pend = std::make_shared<std::atomic<int>>(2);
  auto comb = std::make_shared<std::function<void()>>([&pc, out, L, R, done]() {
    Ctx c;
    c.eng = pc.eng;
    ctx_init(c, 1);
    combine(*out, *L, *R, c.sc[0].t1, c.sc[0].t2, *pc.eng);
    ctx_free(c);
    mpz_clear(L->P); mpz_clear(L->Q); mpz_clear(L->T); delete L;
    mpz_clear(R->P); mpz_clear(R->Q); mpz_clear(R->T); delete R;
    done();
  });
  auto kid_done = [pend, comb](ParCtx& p) {
    if (pend->fetch_sub(1) == 1) p.push([comb] { (*comb)(); });
  };
  spawn_sub(pc, L, a, m, [kid_done, &pc] { kid_done(pc); });
  spawn_sub(pc, R, m, b, [kid_done, &pc] { kid_done(pc); });
}

// top = BS(0,N) computed by t workers (caller's thread is one of them).
// Bit-identical to bs(); only the schedule differs.
void bs_parallel(Node& top, MulEngine& eng, ull N, ull leaf, unsigned t,
                 const std::function<void(const char*, double)>* prog) {
  ParCtx pc;
  pc.eng = &eng;
  pc.leaf = leaf;
  pc.N = N;
  pc.leaf_total = (N + leaf - 1) / leaf;
  pc.prog = prog;
  // span_min: enough leaf-chunks per task to make ~8t tasks (balance) but
  // never so large that fewer than 2 tasks exist.
  ull cpt = std::max<ull>(1, (pc.leaf_total + 8ull * t - 1) / (8ull * t));
  ull span_min = leaf;
  while (span_min < cpt * leaf && span_min < N) span_min <<= 1;
  pc.span_min = span_min;

  std::mutex rm;
  std::condition_variable rcv;
  bool root_done = false;
  spawn_sub(pc, &top, 0, N, [&] {
    std::lock_guard lk(rm);
    root_done = true;
    rcv.notify_all();
  });
  auto worker = [&pc] {
    std::function<void()> f;
    while (pc.pop(f)) {
      f();
      pc.task_done();
    }
  };
  std::vector<std::thread> th;
  for (unsigned i = 1; i < t; ++i) th.emplace_back(worker);
  worker();
  for (auto& x : th) x.join();
  std::unique_lock lk(rm);
  rcv.wait(lk, [&] { return root_done; });
}

inline ull term_count(ull digits, ull guard) {
  // Chudnovsky: ~14.1814259 decimal digits per term (see ARCHITECTURE.md).
  return (ull)((long double)(digits + guard) / 14.18142587l) + 4;
}

void gmp_free_str(char* s) {
  void (*freefunc)(void*, size_t);
  mp_get_memory_functions(nullptr, nullptr, &freefunc);
  freefunc(s, strlen(s) + 1);
}

// ---- E7c/E7d: parallel tail operations (division, sqrt, pow) --------------
// All use eng.mul (parallel Karatsuba over GMP when the engine is
// cpu-gmp-par); bit-identical results by construction, so no fallback needed
// beyond the size gates inside each routine.

// r = base^exp via binary exponentiation with engine multiplications.
// mpz_pow_ui's small-exponent chain (10^k, k ~ 2*1e8 bits worth) is dominated
// by ~28 huge squarings; routing them through eng.mul parallelises them.
void pow_par(MulEngine& eng, mpz_t r, const mpz_t base, ull exp) {
  if (exp == 0) { mpz_set_ui(r, 1); return; }
  mpz_t acc, sq;
  mpz_init_set_ui(acc, 1);
  mpz_init_set(sq, base);
  int hi = 64 - __builtin_clzll(exp);
  for (int i = 0; i < hi; ++i) {
    if ((exp >> i) & 1) eng.mul(acc, acc, sq);
    if (i + 1 < hi) eng.mul(sq, sq, sq);
  }
  mpz_set(r, acc);
  mpz_clear(acc);
  mpz_clear(sq);
}

// R = floor(2^(2n)/D) computed exactly, n = bitlen(D), D > 0.
// Seed: dh = D >> (n-32) (top 32 bits, in [2^31, 2^32)), x0 = floor((2^64-1)/dh).
//   |x0 - floor(2^(n+32)/D)| <= 5 (dh >= 2^31), so a short linear loop turns
//   the bootstrap into the exact seed X0 = floor(2^(n+32)/D) (c tracks D*x).
// Newton doubling: with deficit e >= 0 and scale s = n+m, x = 2^(n+m)/D - e:
//   w  = 2^(n+m+1) - D*x = 2^(n+m) + D*e
//   x' = floor(x*w / 2^n) = floor(2^(n+2m)/D - D*e^2/2^n)
// so the precision doubles (scale n+m -> n+2m) and e' = D*e^2/2^n + theta,
// theta in [0,1).  Since D/2^n >= 1/2 the map has no attracting fixed point:
// once D*e^2 > 2^n the ULP error grows doubly-exponentially (e.g. the last
// steps at n ~ 2^22 can be off by thousands of ULP).  That is harmless for
// correctness -- x stays a LOWER bound at every step -- but rules out
// per-unit fix-up loops, which is what hung the tail benchmark before.
// Exact bulk fix-up instead, via the floor identity
//   floor(V/D) = R + floor((V - D*R)/D)     valid for ANY integer R,
// applied once to the signed residual V = 2^(2n) - D*R (small division).
void rec_par(MulEngine& eng, mpz_t R, const mpz_t D) {
  const mp_size_t n = (mp_size_t)mpz_sizeinbase(D, 2);
  mpz_t x, u, w, V, c;
  mpz_inits(x, u, w, V, c, nullptr);
  mpz_tdiv_q_2exp(w, D, (unsigned long)(n - 32));         // top 32 bits >= 2^31
  mpz_ui_pow_ui(x, 2, 64); mpz_sub_ui(x, x, 1);           // 2^64 - 1
  mpz_tdiv_q_ui(x, x, mpz_get_ui(w));                     // bootstrap seed
  mpz_set_ui(V, 1);
  mpz_mul_2exp(V, V, (unsigned long)(n + 32));            // V = 2^(n+32)
  mpz_mul(c, D, x);                                       // n x 32: linear cost
  while (mpz_cmp(c, V) > 0) { mpz_sub_ui(x, x, 1); mpz_sub(c, c, D); }
  while (true) {
    mpz_add(u, c, D);
    if (mpz_cmp(u, V) > 0) break;
    mpz_set(c, u); mpz_add_ui(x, x, 1);
  }                                       // x = floor(2^(n+32)/D), exact
  // Truncated Newton doubling (E7b). Invariant: x <= 2^(n+m)/D with deficit
  // < 2^(m+2). Each level keeps only the TOP m+G bits of D and truncates both
  // products, so every multiplication is BALANCED (m+G) x m: GMP's unbalanced
  // full-width n x m path was ~190x worse per bit^2 and dominated the tail
  // (measured 14.8s vs 2.15s for GMP's own division at 2^28 bits). The
  // truncation error is < 2^(m+1) ULP at scale n+m, which the exact final
  // fix-up below absorbs in one shot, so R stays EXACT.
  mp_size_t m = 32;
  const mp_size_t GUARD = 64;                                 // truncation guard bits
  for (; m < n - GUARD; m *= 2) {
    mpz_tdiv_q_2exp(u, D, (unsigned long)(n - m - GUARD));    // D top (m+G) bits
    eng.mul(w, u, x);                                     // (m+G) x m  balanced
    mpz_fdiv_q_2exp(w, w, (unsigned long)m);              // top m+G+1 bits
    if (mpz_sizeinbase(w, 2) > (unsigned long)(m + GUARD)) mpz_sub_ui(w, w, 1);
    mpz_set_ui(u, 1);
    mpz_mul_2exp(u, u, (unsigned long)(m + GUARD + 1));
    mpz_sub(u, u, w);                                     // W = 2^(m+G+1) - u > 0
    eng.mul(w, x, u);                                     // m x (m+G+1) balanced
    mpz_fdiv_q_2exp(x, w, (unsigned long)GUARD);          // scale n+m -> n+2m
  }
  // Each level doubles the scale (x -> x*2^m/2^GUARD), so on exit x already
  // lives at scale n+m, m = 2^k >= n-GUARD, deficit e with |e| bits ~ 2^k/16.
  // No boost needed; the exit shift below lands R at scale 2n.
  // No final full-width level: exit at the first m >= n - GUARD. R must land
  // at scale 2n, so shift x by |n-m| in the appropriate direction; either way
  // the deficit of R is < 2^(n+2) + 1 ULP, i.e. the fix-up quotient c below is
  // at most ~5, so c*D stays linear even though m may be < n.
  // On exit x <= floor(2^(n+m)/D) with deficit < 2^(m+2), m = 2^k >= n-GUARD.
  if (m >= n) mpz_fdiv_q_2exp(R, x, (unsigned long)(m - n));
  else        mpz_mul_2exp(R, x, (unsigned long)(n - m)); // ~floor(2^(2n)/D)
  mpz_set_ui(u, 1);
  mpz_mul_2exp(u, u, (unsigned long)(2 * n));             // 2^(2n)
  eng.mul(w, D, R);                                       // parallel n x n
  mpz_sub(u, u, w);                                       // signed residual
  mpz_fdiv_q(c, u, D);                                    // small quotient
  mpz_add(R, R, c);                                       // exact in one shot
  mpz_clears(x, u, w, V, c, nullptr);
}

// q = floor(A/D), r = A - q*D, EXACT, via the exact reciprocal above (all big
// products route through eng.mul, i.e. the parallel Karatsuba on cpu-gmp-par).
// Precondition A >= 0, D > 0.  Falls back to mpz_tdiv_qr when D is small or
// bits(A) >= 2*bits(D) (the deficit bound below needs bits(A) < 2*bits(D)).
void divrec(mpz_t q, mpz_t r, const mpz_t A, const mpz_t D, MulEngine& eng,
            int depth) {
  (void)depth;
  const mp_size_t nb = (mp_size_t)mpz_sizeinbase(D, 2);
  // E7c (measured): the reciprocal ladder + fix-up does ~3x the mul-work of
  // GMP's recursive division, so it only pays off when the quotient is small
  // (bits(A) - nb <= nb/8).  For large quotients GMP's DC division wins even
  // with a single-threaded GMP core (probe @2^28 bits: mpz_tdiv_q 2.17s vs
  // divrec-par 3.90s).  Fallback: D small, or bits(A) >= 2nb (deficit bound),
  // or quotient > nb/8 bits.
  if (nb < (1 << 21) || mpz_sizeinbase(A, 2) >= 2 * nb ||
      8 * (mpz_sizeinbase(A, 2) - nb) > nb) {
    mpz_tdiv_qr(q, r, A, D);
    return;
  }
  mpz_t R, t, f;
  mpz_inits(R, t, f, nullptr);
  rec_par(eng, R, D);                                   // R = floor(2^(2nb)/D)
  eng.mul(t, A, R);
  mpz_fdiv_q_2exp(q, t, (unsigned long)(2 * nb));       // q <= Q*, deficit
  eng.mul(t, q, D);                                     //   <= A*epsR/2^(2nb)
  mpz_sub(r, A, t);                                     // signed residual
  mpz_fdiv_q(f, r, D);                                  // bulk fix-up: one shot
  if (mpz_sgn(f)) {
    mpz_add(q, q, f);
    mpz_mul(t, f, D);                                   // f is tiny: linear
    mpz_sub(r, r, t);
  }
  if (std::getenv("TAILDBG")) {                         // oracle: compare vs exact
    mpz_t qa, ra;
    mpz_inits(qa, ra, nullptr);
    mpz_tdiv_qr(qa, ra, A, D);
    if (mpz_cmp(q, qa) || mpz_cmp(r, ra)) {
      mpz_sub(qa, q, qa);                               // q - Q*
      std::fprintf(stderr, "[divrec] MISMATCH nb=%lld bitsA=%lld qerr_bits=%lld sign=%d\n",
                   (long long)nb, (long long)mpz_sizeinbase(A, 2),
                   (long long)mpz_sizeinbase(qa, 2), mpz_sgn(qa));
    } else {
      std::fprintf(stderr, "[divrec] ok nb=%lld bitsA=%lld\n",
                   (long long)nb, (long long)mpz_sizeinbase(A, 2));
    }
    mpz_clears(qa, ra, nullptr);
  }
  mpz_clears(R, t, f, nullptr);
}

// s = isqrt(N), exact, split-and-correct (N >= 1; s may alias N).
//   j  = bl/4;  H = N >> 2j;  s0 = isqrt(H)  (recursive)   [2j = 2*(bl/4), NOT bl/2]
//   L  = s0 << j   (L^2 = s0^2 * 2^(2j) <= H * 2^(2j) <= N, so L <= s*)
//   e  = floor((N - L^2) / (2L + 1))          [the one expensive division]
//   s  = L + e, then exact fix-up.
// Write s* = isqrt(N), e* = s* - L >= 0, r = N - s*^2 in [0, 2s*]. Then
//   N - L^2 = e*(2L+1) + (e*^2 + r),
// so e = e* + floor((e*^2 + r)/(2L+1)) >= e*: s never underestimates.
// Bounds: H >= 2^(bl/2 - 1) gives s0 >= 2^(bl/4)/sqrt(2) - 1, and
//   e* < (sqrt(H+1) - sqrt(H) + 1) * 2^(bl/4) + 1 < 1.6 * 2^(bl/4),
// hence e*^2 + r < (2.6 + 2) * 2^(bl/2) while 2L+1 > 1.4 * 2^(bl/2):
// the excess e - e* is < 4, i.e. s in [s*, s*+3].
// Fix-up is exact and cheap: with num = e*(2L+1) + rem from the division,
//   N - s^2 = rem + e - e^2   (e is only ~bl/4 bits: tiny squaring),
// then decrement s while N - s^2 < 0 using  N-(s-1)^2 = (N-s^2) + 2s - 1.
// At most 3 iterations; no full-size squaring is ever needed.
void sqrtrec(mpz_t s, const mpz_t N, MulEngine& eng, int depth) {
  mp_size_t bl = mpz_sizeinbase(N, 2);
  if (bl < (1 << 23) || depth > 6) {
    mpz_sqrt(s, N);
    return;
  }
  mpz_t H, s0, L, num, den, e, rem, t;
  mpz_init(H); mpz_init(s0); mpz_init(L); mpz_init(num);
  mpz_init(den); mpz_init(e); mpz_init(rem); mpz_init(t);
  // Split by 2*(bl/4), NOT bl/2: when bl/2 is odd (bl = 4k+2/4k+3), shifting
  // by bl/2 breaks the identity L^2 = s0^2 * 2^(2*(bl/4)) and L lands a factor
  // sqrt(2) below s*, exploding the fix-up count (the E7 tail-test failure).
  mpz_tdiv_q_2exp(H, N, (unsigned long)((bl / 4) * 2));
  sqrtrec(s0, H, eng, depth + 1);
  mpz_mul_2exp(L, s0, (unsigned long)(bl / 4));      // L = s0 << (bl/4)
  eng.mul(num, L, L);                                // L^2 (parallel mul)
  mpz_sub(num, N, num);                              // N - L^2 >= 0
  mpz_mul_2exp(den, L, 1);
  mpz_add_ui(den, den, 1);                           // 2L + 1
  divrec(e, rem, num, den, eng, 0);                  // parallel division
  mpz_add(s, L, e);                                  // s >= isqrt(N)
  mpz_mul(t, e, e);                                  // t = rem + e - e^2
  mpz_sub(t, rem, t);
  mpz_add(t, t, e);
  long dn = 0, up = 0;
  while (mpz_sgn(t) < 0) {                           // s too big (<= 3 times)
    if (++dn > 64) {                                 // proof violated: report, don't hang
      std::fprintf(stderr, "[sqrtrec] DOWN runaway bl=%lld ebits=%lld remsz=%lld\n",
                   (long long)bl, (long long)mpz_sizeinbase(e, 2),
                   (long long)mpz_sizeinbase(rem, 2));
      break;
    }
    mpz_mul_2exp(rem, s, 1);
    mpz_sub_ui(rem, rem, 1);                         // 2s - 1
    mpz_add(t, t, rem);
    mpz_sub_ui(s, s, 1);
  }
  mpz_mul_2exp(rem, s, 1);
  mpz_add_ui(rem, rem, 1);                           // 2s + 1
  while (mpz_cmp(t, rem) >= 0) {                     // belt-and-braces upward
    if (++up > 64) {
      std::fprintf(stderr, "[sqrtrec] UP runaway bl=%lld ebits=%lld remsz=%lld\n",
                   (long long)bl, (long long)mpz_sizeinbase(e, 2),
                   (long long)mpz_sizeinbase(rem, 2));
      break;
    }
    mpz_sub(t, t, rem);
    mpz_add_ui(s, s, 1);
    mpz_add_ui(rem, rem, 2);
  }
  mpz_clear(H); mpz_clear(s0); mpz_clear(L); mpz_clear(num);
  mpz_clear(den); mpz_clear(e); mpz_clear(rem); mpz_clear(t);
}

// Bit-exactness selftest for the E7c/E7d tail primitives: divrec vs
// mpz_tdiv_qr, sqrtrec vs mpz_sqrt, pow_par vs mpz_pow_ui, on both the serial
// and parallel engines, across gate boundaries and structured edge cases.
int tail_selftest_impl(unsigned threads) {
  GmpEngine ser;
  ParallelGmpEngine par(threads > 1 ? threads : 1);
  MulEngine* engs[2] = {&ser, &par};
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 0x5EED1234ull);
  int fails = 0;
  mpz_t A, D, X, q1, r1, q2, r2, N, s1, s2, u;
  mpz_inits(A, D, X, q1, r1, q2, r2, N, s1, s2, u, nullptr);

  auto el = [](clk::time_point a) {
    return std::chrono::duration<double>(clk::now() - a).count(); };
  for (int ei = 0; ei < 2; ++ei) {
    MulEngine* eng = engs[ei];
    const char* en = ei ? "par" : "ser";
    // --- exact division ----------------------------------------------------
    for (mp_size_t db : {mp_size_t(1 << 20), mp_size_t(1 << 22), mp_size_t(1 << 24)}) {
      for (int it = 0; it < 4; ++it) {
        auto tc = clk::now();
        mp_size_t na = db + (mp_size_t)gmp_urandomm_ui(rs, (unsigned long)(db / 2));
        mpz_urandomb(A, rs, na);
        mpz_setbit(A, na - 1);
        mpz_urandomb(D, rs, db);
        mpz_setbit(D, db - 1);
        mpz_tdiv_qr(q1, r1, A, D);
        std::fprintf(stderr, "[tail] %s divrec db=%d it=%d start\n", en, (int)db, it);
        divrec(q2, r2, A, D, *eng, 0);
        std::fprintf(stderr, "[tail] %s divrec db=%d it=%d done %.2fs\n", en, (int)db, it, el(tc));
        if (mpz_cmp(q1, q2) || mpz_cmp(r1, r2)) {
          std::fprintf(stderr, "[tail] divrec mismatch D=%lldbits A=%lldbits\n",
                       (long long)db, (long long)na);
          ++fails;
        }
      }
      // exact multiple (R1 = 0 chains), remainder max, and A < D edges.
      mpz_urandomb(D, rs, db);
      mpz_setbit(D, db - 1);
      mpz_urandomb(X, rs, (unsigned long)(db + db / 3));  // pushes na ~ 2*nd
      mpz_setbit(X, db + db / 3 - 1);
      mpz_mul(A, D, X);
      mpz_tdiv_qr(q1, r1, A, D);
      std::fprintf(stderr, "[tail] %s divrec multiple db=%d start\n", en, (int)db);
      auto tc = clk::now();
      divrec(q2, r2, A, D, *eng, 0);
      std::fprintf(stderr, "[tail] %s divrec multiple db=%d done %.2fs\n", en, (int)db, el(tc));
      if (mpz_cmp(q1, q2) || mpz_cmp(r1, r2) || mpz_sgn(r2) != 0) {
        std::fprintf(stderr, "[tail] divrec multiple mismatch D=%lldbits\n",
                     (long long)db);
        ++fails;
      }
      mpz_sub_ui(X, D, 1);      // X free here: scratch for D-1
      mpz_add(A, A, X);         // remainder at max = D-1
      mpz_tdiv_qr(q1, r1, A, D);
      std::fprintf(stderr, "[tail] %s divrec max-rem db=%d start\n", en, (int)db);
      auto tc2 = clk::now();
      divrec(q2, r2, A, D, *eng, 0);
      std::fprintf(stderr, "[tail] %s divrec max-rem db=%d done %.2fs\n", en, (int)db, el(tc2));
      if (mpz_cmp(q1, q2) || mpz_cmp(r1, r2)) {
        std::fprintf(stderr, "[tail] divrec max-rem mismatch D=%lldbits\n",
                     (long long)db);
        ++fails;
      }
      mpz_urandomb(A, rs, db / 2);
      mpz_tdiv_qr(q1, r1, A, D);
      std::fprintf(stderr, "[tail] %s divrec small-A db=%d start\n", en, (int)db);
      auto tc3 = clk::now();
      divrec(q2, r2, A, D, *eng, 0);
      std::fprintf(stderr, "[tail] %s divrec small-A db=%d done %.2fs\n", en, (int)db, el(tc3));
      if (mpz_cmp(q1, q2) || mpz_cmp(r1, r2)) {
        std::fprintf(stderr, "[tail] divrec small-A mismatch\n");
        ++fails;
      }
    }
    // --- exact integer sqrt --------------------------------------------------
    for (mp_size_t nb : {mp_size_t(1 << 22), mp_size_t(1 << 24), mp_size_t(1 << 25)}) {
      for (int it = 0; it < 4; ++it) {
        auto tc = clk::now();
        mpz_urandomb(N, rs, nb);
        mpz_setbit(N, nb - 1);
        mpz_sqrt(s1, N);
        std::fprintf(stderr, "[tail] %s sqrtrec nb=%d it=%d start\n", en, (int)nb, it);
        sqrtrec(s2, N, *eng, 0);
        std::fprintf(stderr, "[tail] %s sqrtrec nb=%d it=%d done %.2fs\n", en, (int)nb, it, el(tc));
        if (mpz_cmp(s1, s2)) {
          std::fprintf(stderr, "[tail] sqrtrec mismatch N=%lldbits\n",
                       (long long)nb);
          ++fails;
        }
      }
      // perfect square, r at max (u^2 + 2u), r = 1: exercise both fix-up loops.
      mpz_urandomb(u, rs, (unsigned long)(nb / 2));
      mpz_setbit(u, nb / 2 - 1);
      mpz_mul(N, u, u);
      mpz_sqrt(s1, N);
      std::fprintf(stderr, "[tail] %s sqrtrec perfect nb=%d start\n", en, (int)nb);
      auto ts = clk::now();
      sqrtrec(s2, N, *eng, 0);
      std::fprintf(stderr, "[tail] %s sqrtrec perfect nb=%d done %.2fs\n", en, (int)nb, el(ts));
      if (mpz_cmp(s1, s2) || mpz_cmp(s2, u)) {
        std::fprintf(stderr, "[tail] sqrtrec perfect-square mismatch\n");
        char fn[64];
        std::snprintf(fn, sizeof fn, "/tmp/failN%s.bin", en);
        if (FILE *f = std::fopen(fn, "ab")) {
          mpz_out_raw(f, N);
          mpz_out_raw(f, s2);  // what the engine produced
          std::fclose(f);
        }
        ++fails;
      }
      mpz_addmul_ui(N, u, 2);  // N = u^2 + 2u, r = 2u = max
      mpz_sqrt(s1, N);
      std::fprintf(stderr, "[tail] %s sqrtrec max-r nb=%d start\n", en, (int)nb);
      auto ts2 = clk::now();
      sqrtrec(s2, N, *eng, 0);
      std::fprintf(stderr, "[tail] %s sqrtrec max-r nb=%d done %.2fs\n", en, (int)nb, el(ts2));
      if (mpz_cmp(s1, s2)) {
        std::fprintf(stderr, "[tail] sqrtrec max-r mismatch\n");
        char fn2[64];
        std::snprintf(fn2, sizeof fn2, "/tmp/failN%s.bin", en);
        if (FILE *f = std::fopen(fn2, "ab")) {
          mpz_out_raw(f, N);
          mpz_out_raw(f, s2);
          std::fclose(f);
        }
        ++fails;
      }
    }
    // --- power ---------------------------------------------------------------
    mpz_set_ui(u, 10);
    for (unsigned long long e : {1ull, 2ull, 3ull, 1023ull, 1ull << 27}) {
      std::fprintf(stderr, "[tail] %s pow e=%llu start\n", en, e);
      auto tp = clk::now();
      mpz_pow_ui(q1, u, e);
      pow_par(*eng, q2, u, e);
      std::fprintf(stderr, "[tail] %s pow e=%llu done %.2fs\n", en, e, el(tp));
      if (mpz_cmp(q1, q2)) {
        std::fprintf(stderr, "[tail] pow_par mismatch e=%llu\n", e);
        ++fails;
      }
    }
  }
  mpz_clears(A, D, X, q1, r1, q2, r2, N, s1, s2, u, nullptr);
  gmp_randclear(rs);
  return fails;
}

}  // namespace

int tachyon_tail_selftest(unsigned threads) { return tail_selftest_impl(threads); }

void compute_pi(const PiOptions& opt, mpz_t out, PiTimings& tm) {
  MulEngine& eng = get_engine(opt.backend, opt.threads);
  const ull D = opt.digits;
  const ull g = opt.guard;
  const ull d2 = D + g + 2;  // working scale for sqrt/division
  const ull N = term_count(D, g + 10);

  auto t0 = clk::now();

  auto rep = [&](const char* phase, double frac) {
    if (opt.progress) opt.progress(phase, frac);
  };
  Ctx ctx;
  ctx.leaf = opt.leaf;
  ctx.eng = &eng;
  ctx.prog = opt.progress ? &opt.progress : nullptr;
  ctx.leaf_total = (N + opt.leaf - 1) / opt.leaf;
  ctx.report_stride = (ctx.leaf_total + 999) / 1000;  // ~1000 split updates max
  ctx.sc.resize(64);
  for (auto& s : ctx.sc) {
    for (auto& n : s.child) { mpz_init(n.P); mpz_init(n.Q); mpz_init(n.T); }
    mpz_init(s.t1); mpz_init(s.t2);
  }
  Node top;
  mpz_init(top.P); mpz_init(top.Q); mpz_init(top.T);

  unsigned par_threads = 1;
  if (opt.threads > 1) {
    unsigned hw = std::thread::hardware_concurrency();
    par_threads = hw ? std::min<unsigned>(static_cast<unsigned>(opt.threads), hw)
                     : static_cast<unsigned>(opt.threads);
  }

  rep("split", 0.0);
  if (par_threads > 1 && (opt.backend == "cpu" || opt.backend == "gpu")) {
    // E5a: task-parallel split (GMP ops on distinct operands are thread-safe).
    // E8: the gpu backend joins the parallel schedule; large products queue on
    // the device mutex inside tachyon_cuda_mul while workers keep chewing
    // reductions and small products on their own cores.
    bs_parallel(top, eng, N, opt.leaf, par_threads, opt.progress ? &opt.progress : nullptr);
  } else {
    bs(top, ctx, 0, 0, N);
  }
  rep("split", 1.0);
#ifdef TACHYON_DEBUG_SIZES
  std::fprintf(stderr, "[dbg] N=%llu Qdigits=%zu Tdigits=%zu Pdigits=%zu\n", N,
               mpz_sizeinbase(top.Q, 10), mpz_sizeinbase(top.T, 10),
               mpz_sizeinbase(top.P, 10));
#endif

  auto t1 = clk::now();

  rep("sqrt", 0.0);
  // S = floor(sqrt(K * 10^(2*d2)))  =>  |S - sqrt(K)*10^d2| < 1
  // E7: with the cpu-gmp-par engine, pow/sqrt/mul route through parallel
  // primitives (bit-identical by construction; see pow_par/sqrtrec/divrec).
  const bool tail_par = std::strcmp(eng.name(), "cpu-gmp-par") == 0;
  mpz_t S, num, den, ten;
  mpz_init(S); mpz_init(num); mpz_init(den); mpz_init_set_ui(ten, 10);
  {
    mpz_t K;
    mpz_init_set_ui(K, 10005);
    // E7c (measured): mpz_pow_ui beats pow_par for 10^(2*d2) (sqrt phase
    // 3.7s -> 4.9s); GMP's small-base power is already near-optimal here.
    mpz_pow_ui(S, ten, 2 * d2);
    mpz_mul(S, K, S);
    // E7c (measured): reciprocal-based sqrtrec/divrec lose to GMP's DC
    // division at production sizes (probe @2^28 bits: mpz_sqrt 2.25s vs
    // sqrtrec 7.62s; mpz_tdiv_q 2.17s vs divrec 3.90s) because the
    // reciprocal ladder + exact fix-up does ~3x the mul-work of recursive
    // division.  sqrtrec/divrec stay available and test-covered (tail-test,
    // probes) but production sqrt/div run on GMP; the parallel-mul win is
    // banked in the split tree and decimal conversion instead.
    mpz_sqrt(S, S);
    mpz_clear(K);
  }

  auto t2 = clk::now();

  rep("final", 0.0);
  // out = floor( 426880*Q*S / (T * 10^(d2-D)) )  => floor(pi*10^D)
  mpz_mul_ui(num, top.Q, 426880);
  if (tail_par) eng.mul(num, num, S);
  else mpz_mul(num, num, S);
  mpz_pow_ui(den, ten, d2 - D);
  mpz_mul(den, den, top.T);
  mpz_tdiv_q(out, num, den);  // E7c: GMP quotient-only div beats divrec
                              // (2.15s vs 2.61s @probe scale; remainder is
                              // discarded anyway)

  auto t3 = clk::now();

  rep("convert", 0.0);
  if (opt.time_convert) {  // timed throwaway conversion (benchmark only; production
                           // paths convert via to_decimal -> mpz_get_str_parallel)
    unsigned tc = opt.threads > 1 ? (unsigned)opt.threads : 1;
    char* str = mpz_get_str_parallel(out, tc);
    std::free(str);
  }

  auto t4 = clk::now();

  tm.bs = std::chrono::duration<double>(t1 - t0).count();
  tm.sqrt = std::chrono::duration<double>(t2 - t1).count();
  tm.final = std::chrono::duration<double>(t3 - t2).count();
  tm.convert = std::chrono::duration<double>(t4 - t3).count();
  tm.total = tm.bs + tm.sqrt + tm.final + tm.convert;
  rep("done", 1.0);

  mpz_clear(S); mpz_clear(num); mpz_clear(den); mpz_clear(ten);
  mpz_clear(top.P); mpz_clear(top.Q); mpz_clear(top.T);
  for (auto& s : ctx.sc) {
    for (auto& n : s.child) { mpz_clear(n.P); mpz_clear(n.Q); mpz_clear(n.T); }
    mpz_clear(s.t1); mpz_clear(s.t2);
  }
}

std::string to_decimal(const mpz_t scaled, unsigned long long digits, unsigned threads) {
  char* str = mpz_get_str_parallel(scaled, threads);
  std::string all(str);
  std::free(str);
  // all = floor(pi*10^D) has D+1 digits: leading '3', then D decimals (truncated).
  std::string s;
  s.reserve(digits + 2);
  s += all[0];
  s += '.';
  unsigned long long take = digits - 1;
  if (take + 1 > all.size()) take = all.size() - 1;
  s.append(all, 1, take);
  return s;
}
