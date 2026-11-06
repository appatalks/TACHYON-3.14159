#include "cpu_ntt.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr uint32_t P1 = 167772161u;
constexpr uint32_t P2 = 469762049u;
constexpr uint32_t P3 = 754974721u;
constexpr int kLimbBits = 28;
constexpr uint32_t kLimbMask = (1u << kLimbBits) - 1u;
constexpr uint32_t kMaxN = 1u << 24;
constexpr uint32_t kMinN = 1u << 12;

std::mutex g_ntt_mutex;

uint32_t mod_pow(uint32_t base, uint64_t exp, uint32_t p) {
  uint64_t r = 1;
  uint64_t x = base % p;
  while (exp) {
    if (exp & 1u) r = (r * x) % p;
    x = (x * x) % p;
    exp >>= 1;
  }
  return static_cast<uint32_t>(r);
}

uint32_t mod_inv(uint32_t x, uint32_t p) { return mod_pow(x, p - 2u, p); }

uint32_t next_pow2(uint32_t x) {
  uint32_t n = 1;
  while (n < x) n <<= 1;
  return n;
}

template <uint32_t P>
inline uint32_t mul_mod(uint32_t a, uint32_t b) {
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % P);
}

template <uint32_t P>
void ntt(std::vector<uint32_t>& a, bool inverse, uint32_t primitive_root,
         unsigned threads) {
  const uint32_t n = static_cast<uint32_t>(a.size());
  for (uint32_t i = 1, j = 0; i < n; ++i) {
    uint32_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }

  const uint32_t root = mod_pow(primitive_root, (P - 1u) / n, P);
  const uint32_t wroot = inverse ? mod_inv(root, P) : root;
  for (uint32_t len = 2; len <= n; len <<= 1) {
    const uint32_t half = len >> 1;
    const uint32_t wlen = mod_pow(wroot, n / len, P);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (long long block = 0; block < static_cast<long long>(n);
         block += len) {
      uint64_t w = 1;
      const uint32_t start = static_cast<uint32_t>(block);
      for (uint32_t j = 0; j < half; ++j) {
        const uint32_t u = a[start + j];
        const uint32_t v = mul_mod<P>(
            a[start + j + half], static_cast<uint32_t>(w));
        uint32_t lo = u + v;
        if (lo >= P) lo -= P;
        const uint32_t hi = u >= v ? u - v : u + P - v;
        a[start + j] = lo;
        a[start + j + half] = hi;
        w = (w * wlen) % P;
      }
    }
  }

  if (inverse) {
    const uint32_t inv_n = mod_inv(n, P);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (long long i = 0; i < static_cast<long long>(n); ++i)
      a[static_cast<size_t>(i)] =
          mul_mod<P>(a[static_cast<size_t>(i)], inv_n);
  }
}

std::vector<uint32_t> limbs28(mpz_srcptr x) {
  const size_t bits = mpz_sizeinbase(x, 2);
  const size_t n = (bits + kLimbBits - 1) / kLimbBits;
  std::vector<uint32_t> out(n, 0);
  const mp_limb_t* src = mpz_limbs_read(x);
  const size_t words = static_cast<size_t>(mpz_size(x));
  for (size_t i = 0; i < n; ++i) {
    const size_t bit = i * kLimbBits;
    const size_t wi = bit / GMP_LIMB_BITS;
    const unsigned sh = static_cast<unsigned>(bit % GMP_LIMB_BITS);
    uint64_t v = wi < words ? static_cast<uint64_t>(src[wi]) >> sh : 0;
    if (sh != 0 && wi + 1 < words)
      v |= static_cast<uint64_t>(src[wi + 1]) << (GMP_LIMB_BITS - sh);
    out[i] = static_cast<uint32_t>(v & kLimbMask);
  }
  return out;
}

template <uint32_t P, uint32_t G>
void convolve_prime(const std::vector<uint32_t>& x,
                    const std::vector<uint32_t>& y, uint32_t n,
                    unsigned threads, std::vector<uint32_t>& out) {
  std::vector<uint32_t> a(n, 0), b(n, 0);
  for (size_t i = 0; i < x.size(); ++i) a[i] = x[i] % P;
  for (size_t i = 0; i < y.size(); ++i) b[i] = y[i] % P;
  ntt<P>(a, false, G, threads);
  ntt<P>(b, false, G, threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (long long i = 0; i < static_cast<long long>(n); ++i)
    a[static_cast<size_t>(i)] =
        mul_mod<P>(a[static_cast<size_t>(i)], b[static_cast<size_t>(i)]);
  ntt<P>(a, true, G, threads);
  out.swap(a);
}

void pack_result(mpz_ptr out, const std::vector<uint32_t>& limbs,
                 bool negative) {
  const size_t nbits = limbs.size() * kLimbBits;
  const size_t nwords = (nbits + GMP_LIMB_BITS - 1) / GMP_LIMB_BITS;
  mp_limb_t* dst = mpz_limbs_write(out, nwords);
  std::memset(dst, 0, nwords * sizeof(mp_limb_t));
  for (size_t i = 0; i < limbs.size(); ++i) {
    const size_t bit = i * kLimbBits;
    const size_t wi = bit / GMP_LIMB_BITS;
    const unsigned sh = static_cast<unsigned>(bit % GMP_LIMB_BITS);
    const uint64_t v = limbs[i];
    dst[wi] |= static_cast<mp_limb_t>(v << sh);
    if (sh > GMP_LIMB_BITS - kLimbBits && wi + 1 < nwords)
      dst[wi + 1] |= static_cast<mp_limb_t>(v >> (GMP_LIMB_BITS - sh));
  }
  mpz_limbs_finish(out, nwords);
  if (negative) mpz_neg(out, out);
}

}  // namespace

bool mpz_mul_cpu_ntt(mpz_t c, const mpz_t a, const mpz_t b,
                     unsigned threads) {
#ifndef _OPENMP
  (void)c;
  (void)a;
  (void)b;
  (void)threads;
  return false;
#else
  if (mpz_sgn(a) == 0 || mpz_sgn(b) == 0) {
    mpz_set_ui(c, 0);
    return true;
  }
  if (threads < 2) return false;

  const size_t la = (mpz_sizeinbase(a, 2) + kLimbBits - 1) / kLimbBits;
  const size_t lb = (mpz_sizeinbase(b, 2) + kLimbBits - 1) / kLimbBits;
  const uint64_t need = static_cast<uint64_t>(la) + lb - 1;
  const uint64_t n64 = next_pow2(static_cast<uint32_t>(need));
  if (n64 < kMinN || n64 > kMaxN) return false;

  const unsigned t = std::max(1u, threads);
  std::lock_guard<std::mutex> lock(g_ntt_mutex);
  const std::vector<uint32_t> x = limbs28(a);
  const std::vector<uint32_t> y = limbs28(b);
  std::vector<uint32_t> r1, r2, r3;
  const uint32_t n = static_cast<uint32_t>(n64);
  convolve_prime<P1, 3>(x, y, n, t, r1);
  convolve_prime<P2, 3>(x, y, n, t, r2);
  convolve_prime<P3, 11>(x, y, n, t, r3);

  const uint32_t inv12 = mod_inv(P1 % P2, P2);
  const uint32_t p12_mod_p3 =
      static_cast<uint32_t>((static_cast<uint64_t>(P1) * P2) % P3);
  const uint32_t inv123 = mod_inv(p12_mod_p3, P3);
  std::vector<uint32_t> result(need + 4, 0);
  __uint128_t carry = 0;
  for (size_t i = 0; i < need; ++i) {
    const uint32_t x1 = r1[i];
    const uint32_t d2 =
        static_cast<uint32_t>((r2[i] + P2 - (x1 % P2)) % P2);
    const uint32_t x2 = mul_mod<P2>(d2, inv12);
    const uint32_t mixed = static_cast<uint32_t>(
        (static_cast<uint64_t>(x1) +
         static_cast<uint64_t>(P1) * x2) %
        P3);
    const uint32_t d3 =
        static_cast<uint32_t>((r3[i] + P3 - mixed) % P3);
    const uint32_t x3 = mul_mod<P3>(d3, inv123);
    const __uint128_t coeff =
        static_cast<__uint128_t>(x1) +
        static_cast<__uint128_t>(P1) *
            (static_cast<uint64_t>(x2) +
             static_cast<uint64_t>(P2) * x3);
    const __uint128_t v = coeff + carry;
    result[i] = static_cast<uint32_t>(v & kLimbMask);
    carry = v >> kLimbBits;
  }
  size_t used = need;
  while (carry != 0) {
    result[used++] = static_cast<uint32_t>(carry & kLimbMask);
    carry >>= kLimbBits;
  }
  result.resize(used);
  while (result.size() > 1 && result.back() == 0) result.pop_back();
  pack_result(c, result, mpz_sgn(a) != mpz_sgn(b));
  return true;
#endif
}
