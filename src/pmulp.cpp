// E7: mpz_mul_parallel -- Karatsuba divide & conquer fork-joined over GMP.
//
// Correctness: every path computes the exact integer product; the recursion
// is the algebraic identity (a1 B + a0)(b1 B + b0) = z2 B^2 + z1 B + z0 with
// z1 = (a0+a1)(b0+b1) - z0 - z2. No truncation anywhere, so the result is
// bit-identical to mpz_mul regardless of schedule or thread count.
//
// Thread-safety: GMP mpz ops on distinct operands are thread-safe; each
// recursive frame allocates its own temporaries. The caller's `c` is only
// written at the very end (via a private temp), so aliasing c==a or c==b is
// safe.
#include "pmulp.hpp"
#include "cpu_ntt.hpp"
#include <atomic>
#include <mutex>
#include <thread>
#include <cstdlib>

namespace {

typedef unsigned long ul_;
typedef unsigned long long ull_;

// Operand bits below which recursion is not worth its ~1.4x work overhead
// (GMP is already Karatsuba there, so parallel D&C only adds overhead).
// Calibrated by mul-bench (E7): FFT-region operands >= ~2^21 bits gain.
ull_ kParMinBits = 1u << 21;

// Recursion is bounded by operand size; worker threads are a global resource
// handed out by token so a deep tree on 32MB operands does not stall at
// budget exhaustion (old /3 budget model stopped at depth ~4 = 81 leaves and
// left 0.5 s of merge work on the critical path).
// Depth sweep (E7, 32T): depth 2 dominates 3..8 at every size >= 2^22 bits
// (2.1-3.8x); deeper trees inflate work too much over GMP's own FFT/Toom.
std::atomic<int> g_tokens{0};
int kMaxDepth = 2;

inline mp_size_t min_limbs(mpz_srcptr a, mpz_srcptr b) {
  mp_size_t na = mpz_sgn(a) ? (mp_size_t)mpz_size(a) : 1;
  mp_size_t nb = mpz_sgn(b) ? (mp_size_t)mpz_size(b) : 1;
  return na < nb ? na : nb;
}

int g_seed_threads = 1;

// Claim one worker token (bounded total threads across the whole forest).
inline bool claim_token() {
  int t = g_tokens.load(std::memory_order_relaxed);
  while (t > 0 && !g_tokens.compare_exchange_weak(t, t - 1, std::memory_order_relaxed)) {}
  return t > 0;
}

// Magnitude-only parallel multiply: c = |a|*|b| (a, b >= 0 assumed).
void mul_mag(mpz_t c, mpz_srcptr a, mpz_srcptr b, int depth) {
  mp_size_t na = (mp_size_t)mpz_size(a), nb = (mp_size_t)mpz_size(b);
  mp_size_t lo = na < nb ? na : nb;
  ull_ bits = (ull_)lo * GMP_LIMB_BITS;
  if (bits < kParMinBits || depth > kMaxDepth) { mpz_mul(c, a, b); return; }

  // Ensure na >= nb by swapping roles.
  if (na < nb) { mpz_srcptr t = a; a = b; b = t; na = nb; }
  const mp_size_t k = (na + 1) / 2;          // split limb count
  const mp_size_t shift = k * GMP_LIMB_BITS;

  mpz_t a0, a1, b0, b1, z0, z1, z2, u, v;
  mpz_init(a0); mpz_init(a1); mpz_init(b0); mpz_init(b1);
  mpz_init(z0); mpz_init(z1); mpz_init(z2); mpz_init(u); mpz_init(v);
  mpz_fdiv_r_2exp(a0, a, shift);             // a0 = a mod 2^shift
  mpz_fdiv_q_2exp(a1, a, shift);             // a1 = a >> shift
  if (nb <= k) { mpz_set(b0, b); mpz_set_ui(b1, 0); }
  else { mpz_fdiv_r_2exp(b0, b, shift); mpz_fdiv_q_2exp(b1, b, shift); }

  // Spawn the two cheaper branches if workers remain; compute
  // z1 = (a0+a1)(b0+b1) - z0 - z2 on this thread either way.
  bool spawn0 = claim_token();
  std::thread t0, t2;
  if (spawn0) t0 = std::thread([&] { mul_mag(z0, a0, b0, depth + 1); g_tokens.fetch_add(1, std::memory_order_relaxed); });
  else mul_mag(z0, a0, b0, depth + 1);
  const bool has_z2 = mpz_sgn(a1) && mpz_sgn(b1);
  bool spawn2 = has_z2 && claim_token();
  if (spawn2) t2 = std::thread([&] { mul_mag(z2, a1, b1, depth + 1); g_tokens.fetch_add(1, std::memory_order_relaxed); });
  else if (has_z2) mul_mag(z2, a1, b1, depth + 1);
  // else z2 stays 0.

  mpz_add(u, a0, a1);
  if (mpz_sgn(b1)) mpz_add(v, b0, b1); else mpz_set(v, b0);
  mul_mag(z1, u, v, depth + 1);              // runs on this thread
  if (t2.joinable()) t2.join();
  if (t0.joinable()) t0.join();
  mpz_sub(z1, z1, z0);
  mpz_sub(z1, z1, z2);

  // c = ((z2 << shift) + z1) << shift + z0
  mpz_mul_2exp(z2, z2, shift);
  mpz_add(z2, z2, z1);
  mpz_mul_2exp(z2, z2, shift);
  mpz_add(z2, z2, z0);
  mpz_set(c, z2);

  mpz_clear(a0); mpz_clear(a1); mpz_clear(b0); mpz_clear(b1);
  mpz_clear(z0); mpz_clear(z1); mpz_clear(z2); mpz_clear(u); mpz_clear(v);
}

}  // namespace

void mpz_mul_parallel(mpz_t c, const mpz_t a, const mpz_t b, unsigned threads) {
  if (threads < 3 || (ull_)min_limbs(a, b) * GMP_LIMB_BITS < kParMinBits) {
    mpz_mul(c, a, b);
    return;
  }
  const ull_ min_bits = (ull_)min_limbs(a, b) * GMP_LIMB_BITS;
  // The scalar CPU NTT is an opt-in research backend. Its exactness is
  // validated, but its current modular butterflies are slower than pmulp;
  // only an explicitly requested run may select it.
  if (std::getenv("TACHYON_CPU_NTT") && min_bits >= (1ull << 27) &&
      mpz_mul_cpu_ntt(c, a, b, threads)) {
    return;
  }
  pmul_seed(threads);
  int sg = mpz_sgn(a) * mpz_sgn(b);
  if (sg == 0) { mpz_set_ui(c, 0); return; }
  mpz_t ta, tb;
  mpz_init(ta); mpz_init(tb);
  if (mpz_sgn(a) < 0) mpz_neg(ta, a); else mpz_set(ta, a);
  if (mpz_sgn(b) < 0) mpz_neg(tb, b); else mpz_set(tb, b);
  mul_mag(c, ta, tb, 1);
  if (sg < 0) mpz_neg(c, c);
  mpz_clear(ta); mpz_clear(tb);
}

// ---- exported shared-token API (used by parallel division / sqrt) ----

void pmul_seed(unsigned threads) {
  // Pool size is set once per process (engines use a fixed thread count);
  // tokens are shared across all concurrent parallel ops so the total spawn
  // count stays within the configured cap.
  static std::once_flag once;
  std::call_once(once, [&] {
    if (threads < 3) threads = 3;
    g_tokens.store((int)threads - 1);
    g_seed_threads = (int)threads;
    if (const char* e = getenv("TACHYON_PMUL_DEPTH")) kMaxDepth = atoi(e);
    if (const char* e = getenv("TACHYON_PMUL_MINBITS")) kParMinBits = atoll(e);
  });
}

bool pmul_claim() { return claim_token(); }
void pmul_release() { g_tokens.fetch_add(1, std::memory_order_relaxed); }
bool pmul_pool_active() { return g_seed_threads >= 3; }
