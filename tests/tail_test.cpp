// E7 tail-primitive correctness: divrec/sqrtrec/pow_par vs GMP, bit-exact.
#include "../src/pi_chudnovsky.hpp"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  unsigned t = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 32;
  int fails = tachyon_tail_selftest(t);
  if (fails)
    std::printf("TAIL-TEST FAIL (%d mismatches)\n", fails);
  else
    std::printf("TAIL-TEST OK (threads=%u)\n", t);
  return fails != 0;
}
