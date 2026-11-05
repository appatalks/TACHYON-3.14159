// E7 correctness: mpz_mul_parallel must be bit-identical to mpz_mul.
#include "../src/pmulp.hpp"
#include <cstdio>
#include <cstdlib>
#include <gmp.h>
#include <random>
#include <vector>

int main() {
  std::mt19937_64 rng(0xE7);
  std::vector<unsigned> bits = {1, 7, 63, 64, 65, 1000, 70000, 200000,
                                (1u << 21) - 1, 1u << 21, (1u << 21) + 1,
                                1u << 22, 1u << 23, 1u << 24, 3u * (1u << 24)};
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 0xE7);
  int fails = 0;
  mpz_t a, b, r, p;
  mpz_init(a); mpz_init(b); mpz_init(r); mpz_init(p);
  for (unsigned bits_a : bits) {
    for (int skew = 0; skew < 3; ++skew) {
      unsigned bits_b = skew == 0 ? bits_a : (skew == 1 ? bits_a / 2 + 1 : bits_a * 2 + 1);
      mpz_urandomb(a, rs, bits_a);
      mpz_urandomb(b, rs, bits_b);
      mpz_mul_ui(b, b, 1234567);  // make unlikely to be trivially small
      for (int sgn = 0; sgn < 4 && !fails; ++sgn) {
        mpz_set(a, a);
        mpz_t aa, bb;
        mpz_init_set(aa, a); mpz_init_set(bb, b);
        if (sgn & 1) mpz_neg(aa, aa);
        if (sgn & 2) mpz_neg(bb, bb);
        mpz_mul(r, aa, bb);
        mpz_mul_parallel(p, aa, bb, 32);
        if (mpz_cmp(r, p) != 0) {
          std::printf("FAIL bits=%u skew=%d sgn=%d\n", bits_a, skew, sgn);
          ++fails;
        }
        // aliasing: c == a
        mpz_set(aa, aa);
        mpz_mul_parallel(aa, aa, bb, 8);
        if (mpz_cmp(r, aa) != 0) {
          std::printf("FAIL alias bits=%u skew=%d sgn=%d\n", bits_a, skew, sgn);
          ++fails;
        }
        mpz_clear(aa); mpz_clear(bb);
      }
    }
  }
  // zeros and ones
  mpz_set_ui(a, 0); mpz_urandomb(b, rs, 100000);
  mpz_mul(r, a, b); mpz_mul_parallel(p, a, b, 32);
  if (mpz_cmp(r, p)) { std::printf("FAIL zero\n"); ++fails; }
  mpz_set_ui(a, 1);
  mpz_mul(r, a, b); mpz_mul_parallel(p, a, b, 32);
  if (mpz_cmp(r, p)) { std::printf("FAIL one\n"); ++fails; }
  (void)fails;
  gmp_randclear(rs);
  mpz_clear(a); mpz_clear(b); mpz_clear(r); mpz_clear(p);
  if (fails) { std::printf("mul-test: %d FAILURES\n", fails); return 1; }
  std::printf("mul-test: OK (bit-identical at all sizes)\n");
  return 0;
}
