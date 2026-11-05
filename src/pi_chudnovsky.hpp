// TACHYON-pi core: pi to N digits via Chudnovsky binary splitting, exact ints.
#pragma once
#include <gmp.h>
#include <functional>
#include <string>

struct PiTimings {
  double bs = 0, sqrt = 0, final = 0, convert = 0, write = 0, total = 0;
};

struct PiOptions {
  unsigned long long digits = 1000;  // total significant decimal digits of pi
  unsigned leaf = 8;                 // binary-splitting leaf size (terms)
  unsigned guard = 30;               // extra decimal digits of working precision
  std::string backend = "cpu";       // cpu | gpu | hybrid
  int threads = 0;                   // 0 = default (baseline: 1)
  unsigned long long gpu_mem_mb = 0; // 0 = auto-probe (stage 3+)
  bool time_convert = false;         // benchmark: run+time a throwaway decimal conversion
  // Optional progress hook, invoked from the compute thread at phase
  // boundaries and periodically during the split phase (fraction 0..1).
  std::function<void(const char* phase, double frac)> progress;
};

// Computes out = floor(pi * 10^D) where D = options.digits.
// The exact value satisfies 0 <= pi*10^D - out < 10^-(guard-3).
void compute_pi(const PiOptions& opt, mpz_t out, PiTimings& tm);

// Returns decimal expansion "3.14159..." with exactly `digits` significant digits.
// Uses the E6 parallel D&C converter when threads > 1.
std::string to_decimal(const mpz_t scaled, unsigned long long digits, unsigned threads = 1);

// E7: bit-exactness selftest of the parallel tail primitives (divrec,
// sqrtrec, pow_par) against GMP. Returns the number of failures (0 = pass).
int tachyon_tail_selftest(unsigned threads);
