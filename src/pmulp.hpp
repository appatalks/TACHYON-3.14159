// E7: multi-threaded exact integer multiplication.
//
// Divide-and-conquer Karatsuba (2-way split, 3 products) over single-threaded
// GMP leaves, fork-joined across up to `threads` workers. In the regime where
// GMP itself uses FFT/Toom (M(2n) ~ 2.2-2.8 M(n)), one Karatsuba level buys
// ~1.5-1.9x wall-time at ~1.1-1.4x work; ~log2(threads) levels saturate the
// cores. Results are bit-identical to mpz_mul at all sizes (exact integers).
#pragma once
#include <gmp.h>

void mpz_mul_parallel(mpz_t c, const mpz_t a, const mpz_t b, unsigned threads);

// Shared worker-token pool (E7c/E7d): parallel division and sqrt fork-join
// through the same global cap as parallel multiplication. pmul_seed must be
// called once with the intended thread count before any parallel op.
void pmul_seed(unsigned threads);
bool pmul_claim();     // take a worker token; false if the pool is empty
void pmul_release();   // refund one
bool pmul_pool_active();  // true once seeded with >= 3 threads
