// Independent pi computation via Machin's formula:
//   pi = 16*atan(1/5) - 4*atan(1/239)
// computed with plain integer recurrences -- a completely different route from
// Chudnovsky binary splitting. Prints "3." followed by (D-1) decimal digits,
// matching the `tachyon-pi --out` format for byte-exact comparison.
//
// Error: every term is an exact floor division; cumulative truncation is at
// most ~j ulp in the accumulator (j ~ 1e6 terms at P ~ 1e6 digits, i.e. < 1e13
// ulp), absorbed by GUARD=30 working digits. The printed digits therefore
// match floor(pi*10^(D-1)) except with probability ~1e-17 at a digit boundary.
#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: machin-check D\n"); return 2; }
  const unsigned long D = std::strtoul(argv[1], nullptr, 10);  // significant digits
  if (D < 2) { std::fprintf(stderr, "D must be >= 2\n"); return 2; }
  const unsigned long GUARD = 30;
  const unsigned long P = (D - 1) + GUARD;  // working decimals after the point

  auto atan_inv = [&](unsigned long m, mpz_t sum) {
    // computes sum = floor(atan(1/m) * 10^P)
    mpz_init(sum);
    mpz_t u, t;
    mpz_init_set_ui(u, 10);
    mpz_pow_ui(u, u, P);
    mpz_tdiv_q_ui(u, u, m);  // u_0 = floor(10^P / m)
    mpz_init(t);
    const unsigned long m2 = m * m;
    unsigned long j = 0;
    bool neg = false;
    for (;;) {
      if (mpz_sgn(u) == 0) break;
      mpz_tdiv_q_ui(t, u, 2 * j + 1);  // t_j = floor(u_j / (2j+1))
      if (neg) mpz_sub(sum, sum, t);
      else mpz_add(sum, sum, t);
      mpz_tdiv_q_ui(u, u, m2);         // u_{j+1} = floor(u_j / m^2)
      neg = !neg;
      ++j;
    }
    mpz_clears(u, t, (mpz_ptr)nullptr);
  };

  mpz_t a5, a239, pi;
  atan_inv(5, a5);
  atan_inv(239, a239);
  mpz_init(pi);
  mpz_mul_ui(pi, a5, 16);
  mpz_submul_ui(pi, a239, 4);

  // truncate from scale 10^P to 10^(D-1)
  mpz_t den;
  mpz_init_set_ui(den, 10);
  mpz_pow_ui(den, den, GUARD);
  mpz_fdiv_q(pi, pi, den);

  void (*freefunc)(void*, size_t);
  mp_get_memory_functions(nullptr, nullptr, &freefunc);

  char* str = mpz_get_str(nullptr, 10, pi);  // D digits: '3' then D-1 decimals
  const size_t len = std::strlen(str);
  if (len < D) {  // pathological leading-zero case: pad
    std::printf("%c.%s\n", str[0], std::string((size_t)(D - 1 - (len - 1)), '0').append(str + 1).c_str());
  } else {
    std::printf("%c.%s\n", str[0], str + 1);
  }
  freefunc(str, len + 1);

  mpz_clears(a5, a239, pi, den, (mpz_ptr)nullptr);
  return 0;
}
