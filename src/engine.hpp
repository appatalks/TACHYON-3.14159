// Multiplication engine abstraction. Stage 2: GMP passthrough. Stage 3/4 will
// provide exact-NTT CUDA and heterogeneous cost-model dispatch behind this same
// interface so the binary-splitting core never changes.
#pragma once
#include <gmp.h>
#include <atomic>
#include <cstddef>
#include <string>
#include "pmulp.hpp"

struct MulEngine {
  virtual ~MulEngine() = default;
  // c = a * b (c may alias a or b).
  virtual void mul(mpz_t c, const mpz_t a, const mpz_t b) = 0;
  virtual const char* name() const = 0;
};

struct GmpEngine : MulEngine {
  void mul(mpz_t c, const mpz_t a, const mpz_t b) override { mpz_mul(c, a, b); }
  const char* name() const override { return "cpu-gmp"; }
};

// E7: GMP with multi-threaded FFT-region multiplication. Bit-identical to
// GmpEngine; spawn threads are drawn from a global token pool so several
// concurrent muls (e.g. inside the parallel split) stay within the thread cap.
struct ParallelGmpEngine : MulEngine {
  std::atomic<unsigned> threads{1};
  explicit ParallelGmpEngine(unsigned t = 1) { threads.store(t); }
  void mul(mpz_t c, const mpz_t a, const mpz_t b) override {
    mpz_mul_parallel(c, a, b, threads.load(std::memory_order_relaxed));
  }
  const char* name() const override { return "cpu-gmp-par"; }
};

// Returns the engine for a backend name. Unknown/gpu/hybrid fall back to CPU
// with a one-line notice until the CUDA/NTT stages land. threads > 1 selects
// the parallel-multiplication CPU engine (E7).
MulEngine& get_engine(const std::string& backend, int threads = 0);
