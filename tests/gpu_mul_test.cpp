// Stage 3 correctness: GPU exact-NTT multiplication vs GMP across sizes,
// signs, zero, and the Karatsuba fallback (operands beyond one transform).
// Exits 0 on pass, 77 to signal "no CUDA device; skipped".
#include <gmp.h>
#include <cstdio>
#include <cstdlib>

extern "C" int tachyon_cuda_probe();
extern "C" void tachyon_cuda_mul(mpz_ptr c, mpz_srcptr a, mpz_srcptr b);

static gmp_randstate_t rs;

static void check_bits(unsigned bits_a, unsigned bits_b, int trials) {
  mpz_t a, b, r1, r2;
  mpz_init(a); mpz_init(b); mpz_init(r1); mpz_init(r2);
  for (int t = 0; t < trials; ++t) {
    mpz_urandomb(a, rs, bits_a);
    mpz_urandomb(b, rs, bits_b);
    if (t & 1) mpz_neg(a, a);
    if (t & 2) mpz_neg(b, b);
    mpz_mul(r1, a, b);                       // GMP reference
    tachyon_cuda_mul(r2, a, b);              // GPU path (dispatch internal)
    if (mpz_cmp(r1, r2) != 0) {
      std::printf("FAIL bits=%u x %u trial=%d\n", bits_a, bits_b, t);
      mpz_out_str(stdout, 16, a); std::printf(" * ");
      mpz_out_str(stdout, 16, b); std::printf("\n  got ");
      mpz_out_str(stdout, 16, r2); std::printf("\n  want ");
      mpz_out_str(stdout, 16, r1); std::printf("\n");
      return exit(1);
    }
  }
  mpz_clear(a); mpz_clear(b); mpz_clear(r1); mpz_clear(r2);
  std::printf("  ok  %u-bit x %u-bit (%d trials, mixed signs)\n", bits_a,
              bits_b, trials);
}

int main() {
  if (tachyon_cuda_probe() != 1) {
    std::printf("gpu-mul-test: no CUDA device; skipping\n");
    return 77;
  }
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 20261103);

  // exact small cases
  mpz_t a, b, r1, r2;
  mpz_init_set_si(a, -123456789); mpz_init_set_si(b, 987654321);
  mpz_init(r1); mpz_init(r2);
  tachyon_cuda_mul(r2, a, b); mpz_mul(r1, a, b);
  if (mpz_cmp(r1, r2)) { std::printf("FAIL small signed\n"); return 1; }
  mpz_set_ui(a, 0); tachyon_cuda_mul(r2, a, b);
  if (mpz_sgn(r2) != 0) { std::printf("FAIL zero\n"); return 1; }
  std::printf("  ok  small signed / zero edge cases\n");

  check_bits(100, 100, 20);
  check_bits(4096, 4096, 10);
  check_bits(65536, 65536, 5);
  check_bits(262144, 262144, 3);
  check_bits(1u << 18, 1u << 18, 2);
  check_bits(1u << 20, 1u << 20, 2);
  check_bits(1u << 22, 1u << 22, 2);
  check_bits(1u << 24, 1u << 24, 2);   // near single-transform limit
  check_bits(1u << 26, 1u << 26, 2);   // Karatsuba territory
  check_bits(1u << 27, 1u << 27, 2);   // two Karatsuba levels possible
  check_bits(1u << 28, 1u << 28, 1);   // four-prime N=2^25 tier
  check_bits(1u << 29, 1u << 28, 1);   // skewed N=2^25 tier
  check_bits(1u << 31, 1u << 31, 1);   // outer N=2^28 Karatsuba boundary
  check_bits(1u << 29, 1u << 27, 1);   // skewed N=2^25 tier
  std::printf("gpu-mul-test: PASS\n");
  return 0;
}
