// E4: multiplication crossover benchmark — GMP vs GPU exact-NTT path.
// Same operands, median of K timed runs each, sizes sweeping 2^14..2^28 bits.
// Prints a table and (with --json FILE) appends JSONL rows for EXPERIMENTS.md.
#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

extern "C" int tachyon_cuda_probe();
extern "C" void tachyon_cuda_mul(mpz_ptr c, mpz_srcptr a, mpz_srcptr b);
extern "C" double tachyon_cuda_last_gpu_ms();  // microseconds of GPU pipeline in last mul

static double time_mul(void (*fn)(mpz_ptr, mpz_srcptr, mpz_srcptr), mpz_ptr c,
                       mpz_srcptr a, mpz_srcptr b, int trials) {
  std::vector<double> v;
  for (int i = 0; i < trials; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn(c, a, b);
    auto t1 = std::chrono::high_resolution_clock::now();
    v.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  const char* json = nullptr;
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
  if (tachyon_cuda_probe() != 1) { std::printf("no CUDA device\n"); return 2; }

  gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs, 42);
  mpz_t a, b, c; mpz_init(a); mpz_init(b); mpz_init(c);
  int trials = 5;

  std::printf("%10s  %10s  %10s  %10s  %8s\n",
              "bits", "gmp_ms", "e2e_ms", "gpu_ms", "ratio");
  std::FILE* jf = json ? std::fopen(json, "a") : nullptr;
  for (unsigned bits = 14; bits <= 28; ++bits) {
    mpz_urandomb(a, rs, 1u << bits);
    mpz_urandomb(b, rs, 1u << bits);
    double gmp_ms = time_mul(mpz_mul, c, a, b, trials) * 1e3;
    double e2e_ms = time_mul(tachyon_cuda_mul, c, a, b, trials) * 1e3;
    double gpu_ms = tachyon_cuda_last_gpu_ms();  // backend returns microseconds
    std::printf("%10u  %10.3f  %10.3f  %10.3f  %8.2f\n", 1u << bits,
                gmp_ms, e2e_ms, gpu_ms, gmp_ms / e2e_ms);
    if (jf) std::fprintf(jf, "{\"bits\":%u,\"gmp_ms\":%.6f,\"e2e_ms\":%.6f,"
                             "\"gpu_ms\":%.6f}\n", 1u << bits, gmp_ms, e2e_ms, gpu_ms);
  }
  if (jf) std::fclose(jf);
  return 0;
}
