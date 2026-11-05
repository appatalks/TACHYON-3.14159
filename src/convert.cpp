// E6: parallel divide-and-conquer decimal conversion.
//
// Cost model: mpz_get_str is already a D&C conversion but single-threaded.
// The speedup of parallelising is bounded by the serial power-of-10 squaring
// chain (Sum M(10^2^k) ~ 2*M(D_top)) plus the serial top-level split; deeper
// levels are fully independent subtrees.  We share one read-only pow table
// P[k] = 10^(2^k) so every node at the same depth divides by the same
// divisor, fork-join the big (right, power-of-2 sized) subtree, and convert
// <= LEAF-digit leaves with plain mpz_get_str into a padded output buffer.
//
// Thread-safety: GMP mpz ops on distinct operands are thread-safe; the pow
// table is read-only after construction; each worker owns its mpz temporaries.
#include "convert.hpp"

#include <gmp.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr mp_size_t LEAF = 8192;  // digits per sequential leaf

inline unsigned log2_floor(mp_size_t v) {
  unsigned k = 0;
  while ((mp_size_t{1} << (k + 1)) <= v) ++k;
  return k;
}

struct Job {
  mpz_ptr n;  // value < 10^len (leading zeros possible); owned by caller frame
  mp_size_t len;
  char* out;  // exactly `len` chars written, zero-padded
  unsigned depth;
};

void convert_rec(const Job& j, const std::vector<mpz_t>& pow10, unsigned max_depth) {
  if (j.len <= LEAF) {
    char tmp[LEAF + 4];
    mpz_get_str(tmp, 10, j.n);  // n >= 0, no sign, <= LEAF digits
    const std::size_t got = std::strlen(tmp);
    const std::size_t pad = static_cast<std::size_t>(j.len) - got;
    std::memset(j.out, '0', pad);
    std::memcpy(j.out + pad, tmp, got);
    return;
  }
  // Split point 2^floor(log2(len-1)) so the left half is never empty
  // (len a power of two would give lq = 0 and infinite recursion).
  const unsigned k = log2_floor(j.len - 1);
  const mp_size_t s = mp_size_t{1} << k;  // right-half width (power of 2)
  const mp_size_t lq = j.len - s;         // left-half width (>= 1)

  mpz_t q, r;
  mpz_init2(q, mpz_sizeinbase(j.n, 2));  // upper bound, no realloc later
  mpz_init2(r, mpz_sizeinbase(j.n, 2));
  mpz_tdiv_qr(q, r, j.n, pow10[k]);      // q = n / 10^s, r = n mod 10^s

  if (j.depth < max_depth) {
    // Fork the full power-of-2 right subtree; recurse the smaller left inline.
    Job rj{r, s, j.out + lq, j.depth + 1};
    std::thread th([&] { convert_rec(rj, pow10, max_depth); });
    Job qj{q, lq, j.out, j.depth + 1};
    convert_rec(qj, pow10, max_depth);
    th.join();
  } else {
    Job rj{r, s, j.out + lq, j.depth + 1};
    convert_rec(rj, pow10, max_depth);
    Job qj{q, lq, j.out, j.depth + 1};
    convert_rec(qj, pow10, max_depth);
  }
  mpz_clear(q);
  mpz_clear(r);
}

}  // namespace

char* mpz_get_str_parallel(mpz_srcptr x, unsigned threads) {
  if (threads < 2) threads = 1;
  const int neg = mpz_sgn(x) < 0;
  if (mpz_sgn(x) == 0) {
    char* s = static_cast<char*>(std::malloc(2));
    s[0] = '0';
    s[1] = '\0';
    return s;
  }

  // sizeinbase can overestimate decimal digits by 1; extra leading zeros are
  // stripped at the end, so it is safe.
  mp_size_t nd = mpz_sizeinbase(x, 10);
  const unsigned kmax = log2_floor(nd);  // need P[kmax] = 10^(2^kmax)

  // Serial squaring chain: P[k] = P[k-1]^2.
  std::vector<mpz_t> pow10(kmax + 1);
  for (unsigned i = 0; i <= kmax; ++i) mpz_init(pow10[i]);
  mpz_set_ui(pow10[0], 10);
  for (unsigned i = 1; i <= kmax; ++i) mpz_mul(pow10[i], pow10[i - 1], pow10[i - 1]);

  std::size_t const total = static_cast<std::size_t>(nd) + 2;  // sign + NUL
  char* str = static_cast<char*>(std::malloc(total));
  std::size_t const off = neg ? 1 : 0;
  if (neg) str[0] = '-';

  mpz_t copy;
  mpz_init_set(copy, x);  // GMP >= 6.3: mpz_init_copy is mpz_init_set
  if (neg) mpz_neg(copy, copy);

  const unsigned max_depth = log2_floor(threads ? threads : 1);
  convert_rec(Job{copy, nd, str + off, 0}, pow10, max_depth);
  mpz_clear(copy);

  str[off + nd] = '\0';
  // Strip spurious leading zeros (sizeinbase overestimate); keep >= 1 digit.
  char* p = str + off;
  std::size_t digits = static_cast<std::size_t>(nd);
  while (digits > 1 && p[0] == '0') {
    ++p;
    --digits;
  }
  if (p != str + off) {
    std::memmove(str + off, p, digits);
    str[off + digits] = '\0';
  }

  for (unsigned i = 0; i <= kmax; ++i) mpz_clear(pow10[i]);
  return str;
}
