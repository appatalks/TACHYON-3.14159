// E7: serial mpz_mul vs parallel mpz_mul_parallel across operand sizes.
#include "../src/pmulp.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <gmp.h>
#include <vector>
#include <thread>

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  unsigned T = argc > 1 ? (unsigned)std::atoi(argv[1]) : 32;
  int reps = argc > 2 ? std::atoi(argv[2]) : 3;
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 42);
  mpz_t a, b, s, p;
  mpz_init(a); mpz_init(b); mpz_init(s); mpz_init(p);
  std::printf("threads=%u reps=%d  (square operands)\n", T, reps);
  std::printf("%10s  %9s  %9s  %6s\n", "operand bits", "serial s", "par s", "speedup");
  unsigned bits0 = argc > 3 ? (unsigned)std::atoi(argv[3]) : (1u << 20);
  unsigned bits1 = argc > 4 ? (unsigned)std::atoi(argv[4]) : (1u << 28);
  for (unsigned bits = bits0; bits <= bits1; bits <<= 1) {
    mpz_urandomb(a, rs, bits);
    mpz_urandomb(b, rs, bits);
    double best_s = 1e9, best_p = 1e9;
    for (int i = 0; i < reps; ++i) {
      auto t0 = clk::now();
      mpz_mul(s, a, b);
      auto t1 = clk::now();
      best_s = std::min(best_s, std::chrono::duration<double>(t1 - t0).count());
    }
    for (int i = 0; i < reps; ++i) {
      auto t0 = clk::now();
      mpz_mul_parallel(p, a, b, T);
      auto t1 = clk::now();
      best_p = std::min(best_p, std::chrono::duration<double>(t1 - t0).count());
    }
    if (mpz_cmp(s, p) != 0) std::printf("MISMATCH at %u bits!\n", bits);
    std::printf("%10u  %9.4f  %9.4f  %6.2fx\n", bits, best_s, best_p, best_s / best_p);
  }
  gmp_randclear(rs);
  mpz_clear(a); mpz_clear(b); mpz_clear(s); mpz_clear(p);
  return 0;
}
