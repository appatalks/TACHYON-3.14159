#include "verify.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

// BBP hex-digit spot check.
//
// Plouffe identity:  pi = sum_k 16^-k * (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)).
// Therefore {16^p * pi} (p = pos-1, fractional part) has as its hex digits exactly
// pi's hex digits at 1-based positions pos, pos+1, ...
//
// The sum is evaluated mod 1 in long double (x87, 64-bit mantissa, eps ~ 1.1e-19):
//  * for k <= p the term 16^(p-k)/(8k+j) is replaced by the EXACT residue
//    16^(p-k) mod (8k+j) divided by (8k+j)  (u64 modpow; d < 2^63 in practice);
//  * for k > p the terms are < 1/16 and are summed directly (exact binary scales).
// Only K = p + NC + 4 terms are needed; the tail is < 16^-(NC+4).
//
// Target side is exact: with pi_D = pi_scaled = floor(pi*10^D),
//   R = (pi_D * 16^p) mod 10^D,  goal = floor(R * 16^NC / 10^D)
// gives the NC hex digits of {16^p * pi_D/10^D}; the gap to true pi contributes
// 16^p * 10^-D, which the guard condition makes negligible.
//
// Each digit is accepted only if the running value is farther from the next
// digit boundary than a rigorous error bound
//   bnd_i = 16^i * 16*(K+2)*2^-63 + 10^(log10(16)*(p+i) - D) + 16^(i-NC-4);
// otherwise the check is inconclusive.

namespace {

constexpr int kNC = 11;  // hex digits checked; beyond ~12 the long-double
                         // error bound bnd_i = 16^i*16(K+2)*2^-63 makes the
                         // check routinely inconclusive.
constexpr long double kLog10_16 = 1.2041199826712886L;
constexpr long double kEps63 = 1.0842021724855044e-19L;  // 2^-63
const char kHex[] = "0123456789abcdef";

uint64_t modpow64(uint64_t a, uint64_t e, uint64_t m) {
  uint64_t r = 1 % m;
  a %= m;
  while (e) {
    if (e & 1) r = (uint64_t)((unsigned __int128)r * a % m);
    a = (uint64_t)((unsigned __int128)a * a % m);
    e >>= 1;
  }
  return r;
}

void gmp_free(char* s) {
  void (*freefunc)(void*, size_t);
  mp_get_memory_functions(nullptr, nullptr, &freefunc);
  freefunc(s, std::strlen(s) + 1);
}

}  // namespace

// returns 1 = verified, 0 = MISMATCH, -1 = inconclusive
int bbp_check_hex(const mpz_t pi_scaled, unsigned long long D,
                  unsigned long long pos) {
  if (pos < 1) return -1;
  const unsigned long long p = pos - 1;
  // pi-truncation must stay far below 16^-NC after scaling
  if ((long double)(p + kNC + 2) * kLog10_16 > (long double)D) return -1;

  const unsigned long long K = p + kNC + 4;

  // ---- BBP fractional sum, kept in [0,1) ----
  long double s = 0.0L;
  for (unsigned long long k = 0; k <= K; ++k) {
    if (k <= p) {
      const uint64_t e = p - k;
      const uint64_t d1 = 8 * k + 1, d4 = 8 * k + 4, d5 = 8 * k + 5, d6 = 8 * k + 6;
      s += 4.0L * (long double)modpow64(16, e, d1) / (long double)d1;
      s -= 2.0L * (long double)modpow64(16, e, d4) / (long double)d4;
      s -= 1.0L * (long double)modpow64(16, e, d5) / (long double)d5;
      s -= 1.0L * (long double)modpow64(16, e, d6) / (long double)d6;
    } else {
      long double w = 1.0L;  // 16^(p-k), an exact binary fraction
      for (unsigned long long i = 0; i < k - p; ++i) w *= 0.0625L;
      s += w * (4.0L / (long double)(8 * k + 1) - 2.0L / (long double)(8 * k + 4) -
                1.0L / (long double)(8 * k + 5) - 1.0L / (long double)(8 * k + 6));
    }
    s -= floorl(s);
  }

  // ---- exact target digits from the decimal result ----
  mpz_t den, num, rem, goal;
  mpz_init(rem);
  mpz_init_set_ui(den, 10);
  mpz_pow_ui(den, den, (unsigned long)D);
  mpz_init(num); mpz_setbit(num, (mp_bitcnt_t)(4 * p));  // 16^p
  mpz_mul(num, num, pi_scaled);
  mpz_tdiv_r(rem, num, den);
  mpz_init(goal); mpz_setbit(goal, (mp_bitcnt_t)(4 * kNC));
  mpz_mul(goal, goal, rem);
  mpz_fdiv_q(goal, goal, den);
  char* gs = mpz_get_str(nullptr, 16, goal);
  std::string want(gs);
  gmp_free(gs);
  if (want.size() < (size_t)kNC) want = std::string((size_t)kNC - want.size(), '0') + want;
  want = want.substr(want.size() - kNC, kNC);
  mpz_clears(den, num, rem, goal, (mpz_ptr)nullptr);

  // ---- digit-by-digit comparison with error margins ----
  const long double err0 = 16.0L * (long double)(K + 2) * kEps63;
  long double x = s;
  long double scale = 1.0L;
  for (int i = 0; i < kNC; ++i) {
    x *= 16.0L;
    scale *= 16.0L;
    const long double trunc = powl(10.0L, kLog10_16 * (long double)(p + i) - (long double)D);
    const long double bnd = scale * err0 + trunc + powl(16.0L, (long double)(i - kNC - 4));
    const long double di = floorl(x);
    const long double frac = x - di;
    if (frac < bnd || frac > 1.0L - bnd) return -1;  // ambiguous at digit i
    const int got = (int)di;
    if (kHex[got] != want[i]) return 0;              // mismatch
    x = frac;
  }
  return 1;
}

int bbp_verify(const mpz_t pi_scaled, unsigned long long D, int verbose) {
  static const unsigned long long fixed_pos[] = {1, 4, 13, 100, 1000, 100000, 1000000};
  int fails = 0, ran = 0, skip = 0;
  auto try_pos = [&](unsigned long long pos) {
    if (pos < 1 || (long double)(pos + kNC + 2) * kLog10_16 > (long double)D) {
      ++skip;
      return;
    }
    ++ran;
    const int r = bbp_check_hex(pi_scaled, D, pos);
    if (verbose)
      std::printf("  bbp hex @%llu: %s\n", pos,
                  r == 1 ? "ok" : r == 0 ? "FAIL" : "inconclusive");
    if (r == 0) ++fails;
  };
  for (unsigned long long q : fixed_pos) try_pos(q);
  try_pos(D / 3 + 1);
  try_pos(D * 7 / 10 + 1);
  if (verbose && skip) std::printf("  (%d positions skipped: precision too small)\n", skip);
  if (fails) return fails;
  return ran ? 0 : -1;
}
