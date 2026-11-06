#pragma once

#include <gmp.h>

// Exact three-prime CPU NTT multiplication. Returns true only when the
// requested operand sizes fit the implemented transform tier and the product
// was written to c; callers may fall back to another exact multiplier.
bool mpz_mul_cpu_ntt(mpz_t c, const mpz_t a, const mpz_t b, unsigned threads);
