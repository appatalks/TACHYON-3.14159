// E6 correctness: mpz_get_str_parallel must be digit-identical to mpz_get_str.
#include "../src/convert.hpp"

#include <gmp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void gmp_free_str(char* s) {
  void (*freefunc)(void*, size_t);
  mp_get_memory_functions(nullptr, nullptr, &freefunc);
  freefunc(s, std::strlen(s) + 1);
}

static const unsigned kAll5[] = {1u, 2u, 5u, 16u, 32u, 0u};
static const unsigned kFast3[] = {1u, 8u, 32u, 0u};

static int check(mpz_srcptr x, const char* label, const unsigned* thread_list = nullptr) {
  const unsigned* tl = thread_list ? thread_list : kAll5;
  char* ref = mpz_get_str(nullptr, 10, x);
  for (const unsigned* tp = tl; *tp; ++tp) {
    const unsigned t = *tp;
    char* got = mpz_get_str_parallel(x, t);
    if (std::strcmp(ref, got) != 0) {
      std::printf("FAIL [%s threads=%u]\n  ref head=%.40s...\n  got head=%.40s...\n", label, t,
                  ref, got);
      std::free(got);
      gmp_free_str(ref);
      return 1;
    }
    std::free(got);
  }
  gmp_free_str(ref);
  return 0;
}

int main() {
  int fail = 0;
  mpz_t x;
  mpz_init(x);

  // Edge cases.
  mpz_set_ui(x, 0);
  fail |= check(x, "zero");
  mpz_set_ui(x, 1);
  fail |= check(x, "one");
  mpz_set_si(x, -7);
  fail |= check(x, "negative");
  mpz_set_si(x, -1);
  fail |= check(x, "neg one");

  // Powers of ten and neighbors (padding / sizeinbase-overestimate edges).
  for (unsigned e = 0; e <= 20; ++e) {
    mpz_ui_pow_ui(x, 10, e);
    fail |= check(x, "10^e");
    mpz_add_ui(x, x, 1);
    fail |= check(x, "10^e+1");
    mpz_ui_pow_ui(x, 10, e);
    mpz_sub_ui(x, x, 1);
    fail |= check(x, "10^e-1");
  }

  // Random values across many magnitudes.
  gmp_randstate_t st;
  gmp_randinit_default(st);
  gmp_randseed_ui(st, 20260725);
  for (int bits = 1; bits <= 40000; bits = bits < 100 ? bits + 1 : bits * 3) {
    mpz_urandomb(x, st, bits);
    fail |= check(x, "random");
    mpz_neg(x, x);
    fail |= check(x, "random neg");
  }

  // Large values: 1M and 10M digits, compared digit-for-digit.
  for (unsigned long d : {1000000UL, 10000000UL}) {
    mpz_ui_pow_ui(x, 10, d);
    mpz_sub_ui(x, x, 1);  // d digits of 9s... actually d+1->d digits
    mpz_ui_pow_ui(x, 10, d - 1);
    mpz_add_ui(x, x, 1);  // 1 followed by d-2 zeros then 1: d digits
    mpz_t p;
    mpz_init(p);
    mpz_ui_pow_ui(p, 10, d);
    mpz_sub_ui(x, p, 1);  // exactly d digits: 999...9
    mpz_clear(p);
    fail |= check(x, "10^d-1", kFast3);
    mpz_ui_pow_ui(x, 10, d);  // d+1 digits
    fail |= check(x, "10^d", kFast3);
    mpz_addmul_ui(x, x, 1234567);  // irregular shape without prime search
    mpz_add_ui(x, x, 7654321);
    fail |= check(x, "scrambled", kFast3);
    std::printf("  %lu digits: ok\n", d);
  }

  gmp_randclear(st);
  mpz_clear(x);
  std::printf("%s\n", fail ? "CONVERT-TEST FAILED" : "CONVERT-TEST PASSED");
  return fail;
}
