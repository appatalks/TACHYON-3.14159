// Independent verification utilities.
//  * BBP hex-digit spot checks: Bailey-Borwein-Plouffe formula computes the
//    hex digits of pi at arbitrary positions using a completely different
//    series and summation order than Chudnovsky binary splitting.
//    Error analysis: long double (64-bit mantissa, eps~1.1e-19); modular-exponent
//    residues are exact integers < 2^63; per-addition rounding ~eps, over d
//    additions the absolute error is <~ d*1e-19 + tail 16^-21. Therefore 13 hex
//    digits are reliable for d < 1e5, 11 digits for d < 1e7 (enforced by
//    ncmp below).
//  * Reference digit comparison (first 100 digits embedded; optional file).
#pragma once
#include <gmp.h>

// Returns 1 if the BBP check at hex position `pos` (1-based, after the hex
// point) agrees with the decimal-scaled integer `pi_scaled` = floor(pi*10^D).
int bbp_check_hex(const mpz_t pi_scaled, unsigned long long D, unsigned long long pos);

// Runs BBP checks at a set of positions; returns number failed.
int bbp_verify(const mpz_t pi_scaled, unsigned long long D, int verbose);
