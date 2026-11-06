#include "../src/cpu_ntt.hpp"

#include <cstdio>
#include <gmp.h>
#include <random>
#include <vector>

int main() {
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 0xC0FFEE);
  mpz_t a, b, ref, got;
  mpz_inits(a, b, ref, got, nullptr);
  int failures = 0;
  for (unsigned bits : {1u << 16, 1u << 17, 1u << 18, 1u << 20,
                        1u << 27}) {
    const int trials = bits >= (1u << 27) ? 1 : 3;
    for (int it = 0; it < trials; ++it) {
      mpz_urandomb(a, rs, bits);
      mpz_urandomb(b, rs, bits - 3);
      if (it & 1) mpz_neg(a, a);
      if (it == 2) mpz_neg(b, b);
      mpz_mul(ref, a, b);
      if (!mpz_mul_cpu_ntt(got, a, b, 8) || mpz_cmp(ref, got) != 0) {
        std::printf("cpu-ntt-test: mismatch at %u bits, iteration %d\n", bits,
                    it);
        ++failures;
      }
      if (it & 1) mpz_neg(a, a);
      if (it == 2) mpz_neg(b, b);
    }
  }
  mpz_clears(a, b, ref, got, nullptr);
  gmp_randclear(rs);
  if (failures) return 1;
  std::puts("cpu-ntt-test: PASS");
  return 0;
}
