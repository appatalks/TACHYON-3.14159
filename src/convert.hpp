// E6: parallel divide-and-conquer decimal conversion.
#pragma once
#include <gmp.h>

// Convert |x| to a NUL-terminated decimal string using `threads` worker
// threads (fork-join at the top log2(threads) levels of the D&C split tree).
// Result is allocated with malloc(); caller must free().
// Bit-exact identical output to mpz_get_str(10).
char* mpz_get_str_parallel(mpz_srcptr x, unsigned threads);
