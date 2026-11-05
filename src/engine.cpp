#include "engine.hpp"
#include <cstdio>

#ifdef TACHYON_HAVE_CUDA
extern "C" int tachyon_cuda_probe();      // from cuda/cuda_backend.cu
extern "C" void tachyon_cuda_mul(mpz_ptr c, mpz_srcptr a, mpz_srcptr b);

namespace {
// Exact three-prime NTT multiplication on the GPU. Small-operand dispatch
// (below the launch-overhead crossover) lives inside tachyon_cuda_mul itself,
// so this engine is a clean passthrough.
struct CudaEngine : MulEngine {
  void mul(mpz_t c, const mpz_t a, const mpz_t b) override { tachyon_cuda_mul(c, a, b); }
  const char* name() const override { return "cuda-ntt3"; }
};

// Hybrid v1 (stage 4): per-product device choice from the E4-measured
// crossover. End-to-end GPU win measured at >= 2^19 operand bits (2.74x);
// below that GMP wins (launch + transfer floor). The GPU path itself falls
// back to mpz_mul below N=2^16, so this threshold is the only extra gate.
struct HybridEngine : MulEngine {
  void mul(mpz_t c, const mpz_t a, const mpz_t b) override {
    size_t bits = mpz_sizeinbase(a, 2), bb = mpz_sizeinbase(b, 2);
    if (bb < bits) bits = bb;  // dispatch on the SMALLER operand (both must be big)
    if (bits >= kGpuMinBits) tachyon_cuda_mul(c, a, b);
    else mpz_mul(c, a, b);
  }
  const char* name() const override { return "hybrid-e1"; }
  static const size_t kGpuMinBits = 1u << 19;  // E4 crossover (smaller operand)
};
}  // namespace
#endif

MulEngine& get_engine(const std::string& backend, int threads) {
  static GmpEngine gmp;
  static ParallelGmpEngine par;
  if (backend == "cpu") {
    if (threads > 1) {
      par.threads.store((unsigned)threads);
      return par;
    }
    return gmp;
  }
#ifdef TACHYON_HAVE_CUDA
  if (backend == "gpu" && tachyon_cuda_probe() == 1) {
    static CudaEngine cuda;
    return cuda;
  }
  if (backend == "hybrid" && tachyon_cuda_probe() == 1) {
    static HybridEngine hyb;
    return hyb;
  }
#endif
  std::fprintf(stderr, "[tachyon] notice: backend '%s' unavailable; using cpu-gmp\n",
               backend.c_str());
  return gmp;
}
