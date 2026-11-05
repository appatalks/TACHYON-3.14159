/*
 * TACHYON-pi  |  Stage 3: GPU exact-integer multiplication backend.
 * ===========================================================================
 * extern "C" entry points (resolved from libtachyon_core):
 *   void   tachyon_cuda_mul(mpz_ptr c, mpz_srcptr a, mpz_srcptr b)
 *   int    tachyon_cuda_probe(void)
 *   double tachyon_cuda_last_gpu_ms(void)
 *
 * ALGORITHM: three-prime Number-Theoretic Transform (NTT) convolution of
 * base-2^28 limb vectors + Garner's mixed-radix CRT reconstruction.
 *
 *   p1 = 167772161 = 5  * 2^25 + 1,  primitive root g1 = 3
 *   p2 = 469762049 = 7  * 2^26 + 1,  primitive root g2 = 3
 *   p3 = 754974721 = 45 * 2^24 + 1,  primitive root g3 = 11
 *
 * Max transform length N = 2^24 (p3 has 2-adicity exactly 24; p1/p2 admit
 * longer transforms, so all three share a principal 2^24-th root of unity).
 *
 * EXACTNESS ARGUMENT.  With limb width B = 2^28 and transform length N,
 * every convolution coefficient obeys
 *      0 <= c_k <= (#pairs feeding k) * (B-1)^2 < N * B^2 = 2^24 * 2^56 = 2^80.
 * The modulus product M = p1*p2*p3 ~= 2^85.6 > 2^80, so the residue triple
 * (c_k mod p1, c_k mod p2, c_k mod p3) determines c_k uniquely over the
 * integers: the CRT reconstruction is EXACT.  No floating point is used
 * anywhere in the numeric path, so the result matches mpz_mul bit-for-bit.
 *
 * PIPELINE (device work on the default stream, timed end-to-end):
 *   repack the base-2^64 GMP digit image -> 28-bit limbs reduced mod p_i
 *       (one kernel per operand per prime)
 *   bit-reversal permutation (__brev on the fly, no uploaded LUT)
 *   iterative Cooley-Tukey DIT NTT, log2(N) launches, one stage each
 *   fused pointwise: one kernel forms A[i]*B[i] and folds the 1/N factor into
 *       it, so no separate elementwise/scaling passes are needed; the inverse
 *       then reuses the very same bitrev + ascending-DIT schedule, driven by
 *       the same twiddle table read backwards (inverse DFT = forward DFT with
 *       w^-1, and w^-k = -w^(N/2-k) because w^(N/2) == -1); each prime thus
 *       needs a single N/2+1 entry table, i.e. 1/4 of the build time, memory
 *       and PCIe traffic that separate forward/inverse tables would cost
 *   Garner CRT kernel: per index, mixed-radix digits (x1,x2,x3) give
 *       t = x1 + p1*(x2 + p2*x3) < 2^80 held in __uint128_t; its three
 *       base-2^28 limbs are accumulated WITHOUT carry into uint32 acc[] via
 *       atomicAdd (each acc slot takes <= 3 contributions of < 2^28, i.e.
 *       < 2^30, far below 2^32, so no intermediate overflow is possible;
 *       three different threads legitimately share one slot address)
 *   carry resolution: chunk-local normalisation (carry-in 0) + an O(#chunks)
 *       scan of the chunk carry-outs + a short per-chunk fixup with the true
 *       carry-in + a fully parallel 28->64 bit repacking.  A single-threaded
 *       serial carry would cost 0.81 s at N = 2^24 (95 % of the pipeline);
 *       this costs about 1 ms and is exact (see the comment at carry_local)
 *   product digits copied straight into GMP limb storage, sign applied.
 *   (mpz_export/mpz_import are deliberately avoided: with size == 8 this
 *   GMP 6.3.0 build faults on export and mis-imports; mpz_limbs_read /
 *   mpz_limbs_write is exact, faster, and needs no byte swapping.)
 *
 * mulmod: (uint64)a*b % P with P a compile-time constant template argument
 * (P < 2^31 so the 64-bit product is exact); nvcc strength-reduces the
 * constant remainder into reciprocal multiplication.  The plain `% P` form
 * is kept for auditability instead of hand-written Barrett.
 *
 * SMALL N: N <= 2^15 -> plain mpz_mul on the CPU (launch overhead dominates;
 * the spec's 2^11 sits far below the measured crossover, see kNMinGpu);
 * last_gpu_ms reports 0.0.  LARGE N: N > 2^24 -> standard
 * three-recursion Karatsuba at the mpz level, split at a 28-bit limb
 * boundary so every recursive product is either GPU-feasible (N <= 2^24)
 * or strictly smaller, guaranteeing termination.
 *
 * DEVIATIONS FROM SPEC (deliberate; each is measured or memory-forced):
 *   - Bit-reversal computed with __brev instead of an uploaded LUT: identical
 *     permutation, 64 MB less device memory, trivial cost.
 *   - One twiddle table per prime holding w^k for k <= N/2, re-uploaded per
 *     prime into a single device buffer instead of six resident per-prime
 *     tables: the inverse reads the same table backwards with a sign flip, so
 *     this costs 1/4 of the build time, memory and PCIe traffic.  Host mirrors
 *     are cached (2-entry LRU keyed by N) and built with a compile-time
 *     modulus, because a Chudnovsky run changes sizes - and hence tables - on
 *     nearly every iteration.
 *   - Pointwise multiply fused into the first inverse stage with the 1/N
 *     factor folded in, instead of a separate element-wise kernel.
 *   - CRT output accumulated into the base-2^28 limb image with 32-bit atomic
 *     adds (three values < 2^28 never overflow a uint32) instead of a 128-bit
 *     scratch array.
 *   - The spec's serial single-thread carry pass is replaced by a chunked
 *     parallel resolution (normalise each chunk with carry-in 0, chain the
 *     carry-outs across chunks, re-walk each chunk with its true carry-in).
 *     nsys showed the serial version costing 94.5% of GPU time at N = 2^24
 *     (0.81 s of 0.86 s); the parallel one costs about 1 ms.
 *   - mpz_limbs_read / mpz_limbs_write instead of mpz_export / mpz_import:
 *     this GMP (6.3.0) raises SIGFPE in mpz_export and silently mis-reads in
 *     mpz_import when size == 8, exactly the 64-bit limb case used here.
 *   - CPU cutoff raised from the spec's N <= 2^11 to N <= 2^15 (measured).
 *   - If the workspace does not fit the driver's free-memory budget, the call
 *     falls back to an exact mpz_mul on the CPU instead of aborting, and a
 *     Karatsuba recursion whose leaves all fell back reports last_gpu_ms = 0.
 *
 * Thread-safe: the device section of gpu_mul_abs serialises on an internal
 * mutex (E8), so tachyon_cuda_mul may be called from every split worker;
 * budget gating, size math and the small-N / Karatsuba-fallback paths run
 * lock-free on the calling thread.
 */

#include <cuda_runtime.h>
#include <gmp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>

#define CUDA_CHK(expr)                                                       \
  do {                                                                       \
    const cudaError_t _e_ = (expr);                                          \
    if (_e_ != cudaSuccess) {                                                \
      std::fprintf(stderr,                                                   \
                   "TACHYON-pi CUDA failure: %s (%s) at %s:%d\n",            \
                   cudaGetErrorString(_e_), cudaGetErrorName(_e_), __FILE__, \
                   __LINE__);                                                \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

/* Forward declaration so the Karatsuba path below can recurse into it. */
extern "C" void tachyon_cuda_mul(mpz_ptr c, mpz_srcptr a, mpz_srcptr b);

namespace {

/* ------------------------------------------------------------------ *
 * Constants                                                          *
 * ------------------------------------------------------------------ */
constexpr uint32_t kP[3] = {167772161u, 469762049u, 754974721u};
constexpr uint32_t kG[3] = {3u, 3u, 11u};

constexpr int kLimbBits = 28;
constexpr uint32_t kLimbMask = (1u << kLimbBits) - 1u;

/* Spec asks for a 2^11 cutoff because launch overhead dominates tiny sizes -
 * it does, far more than assumed: ~160 kernel launches per call put the real
 * crossover at N = 2^15/2^16 (warm, tables cached, measured on this GPU):
 *   operand bits    N     mpz_mul / our e2e
 *      131072       2^14   0.49x  (GPU 2x slower)
 *      262144       2^15   1.06x  (break-even)
 *      524288       2^16   2.56x
 *     2097152       2^18   5.21x
 * The spec's 2^11 would route a large share of a real Chudnovsky run through a
 * path 5-12x slower than mpz_mul, so the cutoff here is 2^15. */
constexpr uint32_t kNMinGpu = 1u << 15; /* below this: CPU mpz_mul  */
constexpr uint32_t kNMaxGpu = 1u << 24; /* above this: Karatsuba    */
constexpr uint32_t kMaxHalfLimbs = kNMaxGpu / 2; /* 2^23 limbs per factor */
constexpr int kThreads = 256;
constexpr size_t kHeadroomBytes = 300ull << 20; /* free-memory slack */
constexpr size_t kWorkspaceCeil = 10ull << 30;  /* hard 10 GB ceiling */

/* ------------------------------------------------------------------ *
 * Host-side modular helpers (setup only; never on the hot path)      *
 * ------------------------------------------------------------------ */
uint32_t mod_pow(uint32_t base, uint64_t exp, uint32_t p) {
  uint64_t r = 1, x = base % p;
  while (exp) {
    if (exp & 1) r = (r * x) % p;
    x = (x * x) % p;
    exp >>= 1;
  }
  return (uint32_t)r;
}

uint32_t mod_inv(uint32_t v, uint32_t p) { return mod_pow(v, p - 2, p); }

int ceil_log2(uint32_t n) {
  int lg = 0;
  uint32_t v = 1;
  while (v < n) { v <<= 1; ++lg; }
  return lg;
}

uint64_t next_pow2_u64(uint64_t v) {
  uint64_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

/* ------------------------------------------------------------------ *
 * Device code                                                        *
 * ------------------------------------------------------------------ */

__device__ __forceinline__ uint32_t bit_reverse(uint32_t i, int logn) {
  return __brev(i) >> (32 - logn);
}

__global__ void bitrev_perm_kernel(uint32_t* __restrict__ x, int logn) {
  const uint32_t n = 1u << logn;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint32_t r = bit_reverse(i, logn);
    if (r > i) {
      const uint32_t t = x[i];
      x[i] = x[r];
      x[r] = t;
    }
  }
}

/* Modular multiply.  P is a template constant (all three primes are < 2^31),
 * so the 64x32 product and the constant `% P` compile to a reciprocal
 * multiply - no division instruction and no Barrett round-off to audit. */
template <uint32_t P>
__device__ __forceinline__ uint32_t mul_mod(uint32_t a, uint32_t b) {
  return (uint32_t)(((uint64_t)a * b) % P);
}

/* One Cooley-Tukey DIT butterfly stage (s = 0 .. logn - 1) over a buffer that
 * is already in bit-reversed order.  Shared by both directions: the inverse
 * DFT is the forward DFT run with the inverse root, then scaled by 1/N. */
/* INV = false multiplies by w^ti, INV = true by w^-ti.  A stage only ever
 * needs ti <= N/2, and w^-(ti) = -(w^(N/2 - ti)) because w^(N/2) == -1 (w has
 * exact order N), so ONE half-size table drives both directions - nothing else
 * has to be built, stored or uploaded for the inverse transform. */
template <uint32_t P, bool INV>
__global__ void ntt_stage_kernel(uint32_t* __restrict__ x,
                                 const uint32_t* __restrict__ tw, int logn,
                                 int s) {
  const uint32_t n = 1u << logn;
  const uint32_t half = 1u << s;
  const uint32_t wstep = n >> (s + 1);
  const uint32_t nf = n >> 1; /* butterflies per stage */
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t f = blockIdx.x * blockDim.x + threadIdx.x; f < nf; f += stride) {
    const uint32_t j = f & (half - 1u);
    const uint32_t base = ((f >> s) << (s + 1));
    const uint32_t u = x[base + j];
    const uint32_t ti = j * wstep; /* < n/2, so the table covers it */
    const uint32_t w = INV ? (P - tw[(n >> 1) - ti]) : tw[ti];
    const uint32_t t = mul_mod<P>(x[base + j + half], w);
    uint32_t lo = u + t;
    if (lo >= P) lo -= P; /* u + t < 2P < 2^32 */
    const uint32_t hi = (u >= t) ? (u - t) : (u + P - t);
    x[base + j] = lo;
    x[base + j + half] = hi;
  }
}

/* Fused pointwise spectral product and 1/N scale:  dst[i] = a[i]*b[i]*scale.
 * The 1/N factor is folded in here (it commutes with the linear inverse
 * transform), so no separate scaling pass is needed. */
template <uint32_t P>
__global__ void ntt_pointwise_kernel(uint32_t* __restrict__ dst,
                                     const uint32_t* __restrict__ a,
                                     const uint32_t* __restrict__ b,
                                     int logn, uint32_t scale) {
  const uint32_t n = 1u << logn;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    dst[i] = mul_mod<P>(mul_mod<P>(a[i], b[i]), scale);
}

/* Repack 28-bit limbs out of a raw little-endian 64-bit word image.
 * Limb j straddles words:  limb = (w[j*28/64] >> (j*28%64))
 *                                 | (w[j*28/64 + 1] << (64 - j*28%64)).
 * Words past the end read as zero (zero extension); shift-by-64 is avoided.
 * All three primes exceed 2^27, so one conditional subtract reduces a
 * 28-bit limb modulo P. */
__global__ void repack_kernel(const uint64_t* __restrict__ wsrc,
                              uint32_t nwords, uint32_t* __restrict__ dst,
                              uint32_t n, uint32_t p) {
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint64_t bit = (uint64_t)i * kLimbBits;
    const uint64_t wi = bit >> 6;
    const uint32_t bo = (uint32_t)(bit & 63ull);
    const uint64_t lo = (wi < nwords) ? wsrc[wi] : 0ull;
    uint64_t limb;
    if (bo == 0u) {
      limb = lo;
    } else {
      const uint64_t hi = (wi + 1ull < nwords) ? wsrc[wi + 1ull] : 0ull;
      limb = (lo >> bo) | (hi << (64u - bo));
    }
    uint32_t v = (uint32_t)(limb & kLimbMask);
    if (v >= p) v -= p;
    dst[i] = v;
  }
}

/* Garner / mixed-radix CRT + carry-free accumulation into base-2^28 slots.
 * x1 < p1 < p2 and x2 < p2 < p3, so no divisions are needed at all. */
__device__ __host__ constexpr uint32_t c_mod_pow(uint32_t b, uint64_t e,
                                                 uint32_t p) {
  uint64_t r = 1, x = b % p;
  while (e) {
    if (e & 1) r = (r * x) % p;
    x = (x * x) % p;
    e >>= 1;
  }
  return (uint32_t)r;
}

template <uint32_t QA, uint32_t QB, uint32_t QC>
__global__ void crt_accum_kernel(const uint32_t* __restrict__ c1,
                                 const uint32_t* __restrict__ c2,
                                 const uint32_t* __restrict__ c3,
                                 uint32_t* __restrict__ acc, uint32_t n) {
  constexpr uint32_t kInvAmodB = c_mod_pow(QA, QB - 2, QB);
  constexpr uint32_t kInvABmodC =
      c_mod_pow((uint32_t)(((uint64_t)QA * QB) % QC), QC - 2, QC);
  constexpr uint32_t kAmodC = QA % QC;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint32_t x1 = c1[i];
    uint32_t d2 = c2[i] + QB - x1; /* x1 < QB */
    if (d2 >= QB) d2 -= QB;
    const uint32_t x2 = mul_mod<QB>(d2, kInvAmodB);
    const uint32_t mixed = (uint32_t)(((uint64_t)x1 + (uint64_t)kAmodC * x2) % QC);
    uint32_t d3 = c3[i] + QC - mixed;
    if (d3 >= QC) d3 -= QC;
    const uint32_t x3 = mul_mod<QC>(d3, kInvABmodC);

    const __uint128_t t = ( (__uint128_t)(uint64_t)x1)
                        + ((__uint128_t)(uint64_t)QA) * ((uint64_t)x2 + (uint64_t)QB * x3);
    /* Each slot acc[k] receives the matching limb of t_k, t_{k-1} and
     * t_{k-2}, i.e. three different threads touch the same address, so the
     * accumulation must be atomic.  Values stay < 3 * 2^28 < 2^30, far under
     * 2^32, so no overflow is possible and plain 32-bit atomicAdd suffices
     * (this also halves the accumulator's device footprint). */
    atomicAdd(acc + i, (uint32_t)(t & kLimbMask));
    atomicAdd(acc + i + 1, (uint32_t)((t >> kLimbBits) & kLimbMask));
    atomicAdd(acc + i + 2, (uint32_t)(t >> (2 * kLimbBits)));
  }
}

/* ------------------------------------------------------------------ *
 * Carry resolution: carry-free 28-bit slots -> canonical limbs       *
 * ------------------------------------------------------------------ */
/* acc[] holds the carry-free base-2^28 limb sums (every slot < 2^30).  Making
 * them canonical is one serial carry chain, which a single thread cannot do
 * quickly at these sizes: measured on this GPU it cost 0.81 s at N = 2^24,
 * i.e. ~95 % of the whole pipeline.  The chain is therefore resolved in four
 * steps, all parallel except an O(#chunks) scan:
 *
 *   carry_local - each chunk (kCarryChunk limbs, one walking thread) is
 *       normalised with carry-in 0, in place.  Per chunk we keep its carry-out
 *       `local` and the smallest carry-in that would push one MORE carry past
 *       the chunk end.  Normalising first makes that threshold trivial: with
 *       canonical limbs, a carry-in d < 2^28 escapes chunk b iff every limb
 *       after the first is 2^28 - 1 and d >= 2^28 - limb[0].
 *   carry_scan  - one thread chains the chunk carry-outs,
 *       C_{b+1} = local[b] + (C_b escapes ? 1 : 0), and publishes C_b per
 *       chunk; #chunks <= 2^14, so this is microseconds.  Whatever carry is
 *       left after the last chunk becomes one extra output limb.
 *   carry_fix   - each chunk re-walks with carry-in C_b.  That extra carry is
 *       < 2^28, so it either dies inside the chunk (every higher limb is then
 *       already canonical) or escapes exactly where carry_scan said it would;
 *       the walk stops as soon as the carry dies, so in practice this is O(1)
 *       per chunk.
 *   pack64      - canonical 28-bit limbs -> base-2^64 words, fully parallel.
 *
 * Correctness of the split rests on C_b < 2^28 at every boundary: slots are
 * < 2^30, so a chunk carry-out is <= 3 and an escaping carry adds 1, giving
 * C_b <= 4 throughout. */
constexpr uint32_t kLimbBase = 1u << kLimbBits;
constexpr int kCarryChunk = 1024;  /* limbs per carry-resolution chunk */
constexpr int kCarryThreads = 32;  /* thread 0 of each block does the walk */

__global__ void carry_local_kernel(uint32_t* __restrict__ acc,
                                   uint32_t* __restrict__ meta,
                                   uint32_t nlimbs) {
  if (threadIdx.x != 0) return;
  const uint32_t b = blockIdx.x;
  const uint32_t lo = b * (uint32_t)kCarryChunk;
  const uint32_t hi = min(lo + (uint32_t)kCarryChunk, nlimbs);
  uint32_t c = 0, w0 = 0;
  bool prop = true; /* every limb after the first is still all-ones */
  for (uint32_t i = lo; i < hi; ++i) {
    const uint64_t v = (uint64_t)acc[i] + c;
    const uint32_t w = (uint32_t)(v & kLimbMask);
    c = (uint32_t)(v >> kLimbBits);
    acc[i] = w;
    if (i == lo) w0 = w;
    else if (w != kLimbMask) prop = false;
  }
  /* meta[2b] = carry-out with carry-in 0; meta[2b+1] = escape threshold. */
  meta[2 * b] = c;
  meta[2 * b + 1] = prop ? (kLimbBase - w0) : kLimbBase;
}

__global__ void carry_scan_kernel(uint32_t* __restrict__ acc,
                                  uint32_t* __restrict__ meta, uint32_t nlimbs,
                                  uint32_t nchunks) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint32_t c = 0;
  for (uint32_t b = 0; b < nchunks; ++b) {
    const uint32_t escape = (c >= meta[2 * b + 1]) ? 1u : 0u;
    meta[2 * b + 1] = c; /* reuse the slot as this chunk's carry-in */
    c = meta[2 * b] + escape;
  }
  acc[nlimbs] = c; /* the carry that leaves the last chunk: one extra limb */
}

__global__ void carry_fix_kernel(uint32_t* __restrict__ acc,
                                 const uint32_t* __restrict__ meta,
                                 uint32_t nlimbs) {
  if (threadIdx.x != 0) return;
  const uint32_t b = blockIdx.x;
  uint32_t d = meta[2 * b + 1];
  if (d == 0) return;
  const uint32_t lo = b * (uint32_t)kCarryChunk;
  const uint32_t hi = min(lo + (uint32_t)kCarryChunk, nlimbs);
  for (uint32_t i = lo; i < hi && d != 0; ++i) {
    const uint32_t v = acc[i] + d; /* acc[i] < 2^28, d <= 4: no wrap */
    if (v >= kLimbBase) {
      acc[i] = v - kLimbBase;
      d = 1;
    } else {
      acc[i] = v;
      d = 0;
    }
  }
  /* d != 0 here means the carry escaped, exactly as carry_scan predicted. */
}

/* Word i covers bits [64i, 64i+64) of the limb string, i.e. limbs
 * k = 64i/28 .. k+3 (four, because a word can straddle that many). */
__global__ void pack64_kernel(const uint32_t* __restrict__ acc,
                              uint64_t* __restrict__ out, uint32_t nlimbs,
                              uint32_t nwords) {
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nwords;
       i += stride) {
    const uint64_t bit = (uint64_t)i * 64u;
    const uint32_t k = (uint32_t)(bit / (uint64_t)kLimbBits);
    const uint32_t sh0 = (uint32_t)(bit % (uint64_t)kLimbBits);
    uint64_t word = 0;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t idx = k + j;
      const uint64_t limb = (idx < nlimbs) ? (uint64_t)acc[idx] : 0ull;
      const uint32_t off = (uint32_t)kLimbBits * j;
      if (off >= sh0) {
        const uint32_t sh = off - sh0;
        if (sh < 64u) word |= limb << sh;
      } else {
        word |= limb >> (sh0 - off); /* only j == 0, shift < 28 */
      }
    }
    out[i] = word;
  }
}

/* ------------------------------------------------------------------ *
 * Host-side cache + orchestration                                    *
 * ------------------------------------------------------------------ */

struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
};

/* A/B are per-prime so the three prime pipelines can run concurrently on
 * separate streams (the operands are re-repacked per prime anyway). */
DevBuf g_A[3], g_B[3], g_C[3], g_W, g_acc, g_out, g_cmeta;
size_t g_dev_used = 0; /* bytes currently held by this cache */

/* Pinned host staging: pageable cudaMemcpy silently de-syncs the pipeline,
 * and the twiddle tables plus operand images are large enough to matter. */
void* g_pin_in = nullptr;
size_t g_pin_in_bytes = 0;
void* g_pin_out = nullptr;
size_t g_pin_out_bytes = 0;

cudaStream_t g_stream[3] = {nullptr, nullptr, nullptr};
cudaEvent_t g_ev_in = nullptr;

void* pin_ensure(void** pp, size_t& cap, size_t bytes) {
  if (cap >= bytes) return *pp;
  if (*pp) {
    cudaFreeHost(*pp);
    *pp = nullptr;
    cap = 0;
  }
  CUDA_CHK(cudaHostAlloc(pp, bytes, cudaHostAllocDefault));
  cap = bytes;
  return *pp;
}

void streams_ensure() {
  static const bool ready = []() {
    for (int i = 0; i < 3; ++i)
      CUDA_CHK(cudaStreamCreateWithFlags(&g_stream[i], cudaStreamNonBlocking));
    CUDA_CHK(cudaEventCreateWithFlags(&g_ev_in, cudaEventDisableTiming));
    return true;
  }();
  (void)ready;
}

double g_last_gpu_ms = 0.0;
uint64_t g_gpu_calls = 0; /* number of times gpu_mul_abs actually ran */

void* buf_ensure(DevBuf& b, size_t bytes) {
  if (b.bytes >= bytes) return b.p;
  if (b.p) {
    CUDA_CHK(cudaFree(b.p));
    g_dev_used -= b.bytes;
    b.p = nullptr;
    b.bytes = 0;
  }
  CUDA_CHK(cudaMalloc(&b.p, bytes));
  b.bytes = bytes;
  g_dev_used += bytes;
  return b.p;
}


/* w^k for k = 0 .. N/2 (the only exponents a stage can ask for). */
template <uint32_t P>
void build_twiddle(uint32_t* tab, uint32_t N, uint32_t g) {
  const uint32_t step = mod_pow(g, ((uint64_t)P - 1) / N, P);
  uint64_t a = 1;
  for (uint32_t k = 0; k <= (N >> 1); ++k) {
    tab[k] = (uint32_t)a;
    a = (a * step) % P;
  }
}

/* Cached host twiddle mirrors, 2-entry LRU keyed by N.  Tables for a given
 * N are built once with a multiply recurrence and reused for every later
 * call of the same size (the Chudnovsky loop hammers one size). */
struct TwEntry {
  uint32_t n = 0;
  uint32_t* fwd[3] = {nullptr, nullptr, nullptr};
  /* Device mirrors, uploaded once per table and reused across the many
   * Chudnovsky iterations that share one transform size.  At N = 2^26 a
   * fresh pageable upload of all three tables costs ~25 ms per multiply;
   * cached, the amortized cost is near zero. */
  uint32_t* dev[3] = {nullptr, nullptr, nullptr};
  size_t dev_bytes = 0;
};
TwEntry g_tw_cache[2];
int g_tw_victim = 0;

TwEntry* twiddle_host(uint32_t N) {
  for (TwEntry& e : g_tw_cache)
    if (e.n == N) return &e;
  TwEntry& e = g_tw_cache[g_tw_victim];
  g_tw_victim ^= 1;
  if (e.n != 0) {
    for (int i = 0; i < 3; ++i) {
      std::free(e.fwd[i]);
      e.fwd[i] = nullptr;
      if (e.dev[i]) {
        CUDA_CHK(cudaFree(e.dev[i]));
        e.dev[i] = nullptr;
      }
    }
    g_dev_used -= e.dev_bytes;
    e.dev_bytes = 0;
    e.n = 0;
  }
  for (int i = 0; i < 3; ++i) {
    e.fwd[i] = (uint32_t*)std::malloc(sizeof(uint32_t) * ((size_t)N / 2 + 1));
    if (!e.fwd[i]) {
      std::fprintf(stderr, "TACHYON-pi: host twiddle allocation failed\n");
      std::exit(1);
    }
  }
  /* P is a template argument so the recurrence's `% P` compiles to a
   * reciprocal multiply instead of a 64-bit division.  That is worth ~5x here,
   * and it matters: the Chudnovsky loop changes operand sizes (hence N, hence
   * the tables) on nearly every iteration. */
  build_twiddle<kP[0]>(e.fwd[0], N, kG[0]);
  build_twiddle<kP[1]>(e.fwd[1], N, kG[1]);
  build_twiddle<kP[2]>(e.fwd[2], N, kG[2]);
  e.n = N;
  return &e;
}

/* Device budget: min(free + our own cached footprint - 300 MB, 10 GB).
 * The GPU is shared, so the free figure is refreshed periodically -- but NOT
 * on every mul: with a foreign process (e.g. an LLM) holding the device,
 * cudaMemGetInfo on every dispatched mul burned ~15% of the whole 200M-digit
 * run in driver round-trips while every gate failed anyway.  A 250 ms TTL
 * keeps adaptation fast enough and the attempt cost near zero.  The gate is
 * evaluated by mul worker threads without the device mutex, so the cached
 * reading is atomic (benign races by design). */
size_t workspace_cap() {
  static std::atomic<long long> s_cap_ns{0};  // steady-clock time of reading
  static std::atomic<size_t> s_cap_val{0};    // cached cap
  const long long now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  if (now_ns - s_cap_ns.load(std::memory_order_relaxed) <
      250ll * 1000ll * 1000ll)
    return s_cap_val.load(std::memory_order_relaxed);
  s_cap_ns.store(now_ns, std::memory_order_relaxed);
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return s_cap_val = 0;
  const size_t avail = free_b + g_dev_used;
  if (avail <= kHeadroomBytes) return s_cap_val = 0;
  const size_t cap = avail - kHeadroomBytes;
  return s_cap_val = cap < kWorkspaceCeil ? cap : kWorkspaceCeil;
}

size_t words_for_limbs(uint32_t limbs) {
  return ((size_t)limbs * kLimbBits + 63) / 64;
}

/* Total device footprint needed for a transform of length N. */
size_t workspace_need(uint32_t N) {
  const size_t n = N;
  const size_t wout = words_for_limbs(N) + 4;
  /* Both 64-bit operand images together: la + lb - 1 <= N, so between them
   * they occupy at most words_for_limbs(N) + a few words. */
  const size_t wsrc = words_for_limbs(N) + 8;
  return 4 * n * 9 /* A x3, B x3, C x3 (per-prime for stream overlap) */ +
         6 * n /* device twiddle tables, 3 x (N/2 + 1) */ +
         4 * (n + 3) /* acc, 32-bit */ +
         8 * wout + 8 * wsrc + (1u << 20) /* slack: cmeta + alignment */;
}

/* Same pipeline with ONE shared A/B pair on a single stream (E4 layout):
 * 46 % less device footprint, used automatically when the three-stream
 * overlap cannot be afforded.  The device twiddle cache and pinned staging
 * still apply, so this tier still beats the old E4 numbers. */
size_t workspace_need_lean(uint32_t N) {
  const size_t n = N;
  const size_t wout = words_for_limbs(N) + 4;
  const size_t wsrc = words_for_limbs(N) + 8;
  return 4 * n * 5 /* A,B,C1..3 */ +
         6 * n /* device twiddle tables */ +
         4 * (n + 3) /* acc, 32-bit */ +
         8 * wout + 8 * wsrc + (1u << 20);
}

dim3 grid_for(uint32_t n) {
  uint32_t blocks = (n + (uint32_t)kThreads - 1u) / (uint32_t)kThreads;
  if (blocks > 65535u) blocks = 65535u;
  return dim3(blocks, 1, 1);
}

size_t limbs_of(mpz_srcptr x) {
  return (mpz_sizeinbase(x, 2) + (size_t)kLimbBits - 1) / (size_t)kLimbBits;
}

/* Forward NTT (in place): bit-reversal permutation, then DIT stages. */
template <uint32_t P>
void ntt_forward(uint32_t* x, const uint32_t* tw, int logn, cudaStream_t st) {
  bitrev_perm_kernel<<<grid_for(1u << logn), kThreads, 0, st>>>(x, logn);
  for (int s = 0; s < logn; ++s)
    ntt_stage_kernel<P, false><<<grid_for(1u << (logn - 1)), kThreads, 0, st>>>(
        x, tw, logn, s);
}

/* Inverse NTT: fused pointwise product + 1/N scale, then the same bit-reversal
 * + ascending DIT structure as the forward transform, driven by the inverse
 * twiddle table (the inverse DFT is the forward DFT with the inverse root). */
template <uint32_t P>
void ntt_inverse_fused(uint32_t* dst, const uint32_t* a, const uint32_t* b,
                       const uint32_t* tw, int logn, uint32_t inv_n,
                       cudaStream_t st) {
  ntt_pointwise_kernel<P><<<grid_for(1u << logn), kThreads, 0, st>>>(
      dst, a, b, logn, inv_n);
  bitrev_perm_kernel<<<grid_for(1u << logn), kThreads, 0, st>>>(dst, logn);
  for (int s = 0; s < logn; ++s)
    ntt_stage_kernel<P, true><<<grid_for(1u << (logn - 1)), kThreads, 0, st>>>(
        dst, tw, logn, s);
}

/* GPU pipeline for |a| * |b| with a pre-computed power-of-two N.  Writes the
 * unsigned product into `res`.  Returns false when the device workspace
 * cannot be afforded, in which case the caller falls back to the CPU.
 *
 * Host <-> device transfer uses GMP's limb API (mpz_limbs_read /
 * mpz_limbs_write) rather than mpz_export/mpz_import: on this GMP 6.3.0
 * build mpz_export with size == 8 raises SIGFPE and mpz_import(size == 8)
 * returns a wrong value, while the limb API is both correct and a literal
 * copy of the base-2^64 digit array we want (zero conversion work). */
bool gpu_mul_abs(mpz_ptr res, mpz_srcptr xa, mpz_srcptr xb, uint32_t N) {
  static_assert(sizeof(mp_limb_t) == 8, "backend assumes 64-bit GMP limbs");

  const size_t la = limbs_of(xa), lb = limbs_of(xb);
  const mp_size_t na = mpz_size(xa); /* base-2^64 words of |a| */
  const mp_size_t nb = mpz_size(xb);
  const int logn = ceil_log2(N);

  const size_t cap_now = workspace_cap();
  const bool overlap = workspace_need(N) <= cap_now;
  if (!overlap && workspace_need_lean(N) > cap_now) {
    if (std::getenv("TACHYON_GPU_DEBUG"))
      std::fprintf(stderr, "[gpu-gate] N=%u need=%zu/%zu cap=%zu used=%zu\n",
                   N, workspace_need(N), workspace_need_lean(N), cap_now,
                   g_dev_used);
    return false;
  }

  /* The CRT kernel spreads coefficient k over acc[k..k+2], and the top
   * coefficient (index la + lb - 2) therefore needs limbs up to index
   * la + lb, i.e. la + lb + 1 limbs in total - one MORE than the la + lb - 1
   * "obvious" limb count, because a product of two la/limb numbers can reach
   * 2 * (la - 1) limbs plus a carry.  Words are sized from that count. */
  const size_t carry_limbs = la + lb + 1;
  /* One further limb holds the carry that leaves the final chunk. */
  const mp_size_t out_limbs = (mp_size_t)(carry_limbs + 1);
  const mp_size_t nwords_out =
      (mp_size_t)(((uint64_t)out_limbs * kLimbBits + 63) / 64);
  const uint32_t nchunks =
      (uint32_t)((carry_limbs + kCarryChunk - 1) / kCarryChunk);

  /* E8: the device cache and streams are process-wide singletons; with the
   * task-parallel split several workers can hold GPU-sized products at once,
   * so the whole device section serialises on this mutex.  Everything above
   * (size math + budget gate on the cached cap) is read-only and lock-free,
   * which is what keeps the foreign-occupant fallback path cheap. */
  static std::mutex g_dev_mtx;
  std::lock_guard<std::mutex> dev_lk(g_dev_mtx);

  streams_ensure();

  /* Grow-on-demand device cache.  The lean tier shares one A/B pair across
   * the three primes (everything then runs in order on stream 0). */
  for (int i = 0; i < 3; ++i) {
    if (overlap || i == 0) {
      buf_ensure(g_A[i], 4ull * N);
      buf_ensure(g_B[i], 4ull * N);
    }
    buf_ensure(g_C[i], 4ull * N);
  }
  /* +3 slots: the CRT kernel writes acc[i+1] / acc[i+2] for i up to N - 1. */
  buf_ensure(g_acc, 4ull * ((size_t)N + 3));
  buf_ensure(g_out, 8ull * (size_t)nwords_out + 8);
  buf_ensure(g_cmeta, 8ull * ((size_t)nchunks + 2));
  buf_ensure(g_W, 8ull * ((size_t)na + (size_t)nb + 4));

  uint32_t* A[3] = {(uint32_t*)g_A[0].p, (uint32_t*)g_A[1].p, (uint32_t*)g_A[2].p};
  uint32_t* B[3] = {(uint32_t*)g_B[0].p, (uint32_t*)g_B[1].p, (uint32_t*)g_B[2].p};
  if (!overlap) {
    A[1] = A[2] = A[0];
    B[1] = B[2] = B[0];
  }
  uint32_t* C[3] = {(uint32_t*)g_C[0].p, (uint32_t*)g_C[1].p, (uint32_t*)g_C[2].p};
  uint32_t* acc = (uint32_t*)g_acc.p;
  uint64_t* outw = (uint64_t*)g_out.p;
  uint32_t* cmeta = (uint32_t*)g_cmeta.p;
  uint64_t* W = (uint64_t*)g_W.p;
  const uint64_t woff_b = (uint64_t)na; /* second operand image offset (words) */

  const mp_limb_t* da = mpz_limbs_read(xa);
  const mp_limb_t* db = mpz_limbs_read(xb);
  TwEntry* tw = twiddle_host(N);

  const auto t0 = std::chrono::steady_clock::now();

  /* Operand images ride a pinned staging buffer: one host memcpy, then a
   * genuinely asynchronous PCIe burst on stream 0; the other two streams
   * wait on an event for the same image. */
  const size_t insz = 8ull * ((size_t)na + (size_t)nb + 4);
  uint64_t* pinIn = (uint64_t*)pin_ensure(&g_pin_in, g_pin_in_bytes, insz);
  std::memcpy(pinIn, da, 8ull * (size_t)na);
  std::memcpy(pinIn + na, db, 8ull * (size_t)nb);
  CUDA_CHK(cudaMemcpyAsync(W, pinIn, 8ull * ((size_t)na + (size_t)nb),
                           cudaMemcpyHostToDevice, g_stream[0]));
  if (overlap) {
    CUDA_CHK(cudaEventRecord(g_ev_in, g_stream[0]));
    CUDA_CHK(cudaStreamWaitEvent(g_stream[1], g_ev_in, 0));
    CUDA_CHK(cudaStreamWaitEvent(g_stream[2], g_ev_in, 0));
  }

  for (int prime = 0; prime < 3; ++prime) {
    cudaStream_t st = overlap ? g_stream[prime] : g_stream[0];
    /* Twiddles live on the device for the whole lifetime of the cache entry;
     * the first use of a size uploads, the rest of the bucket rides free. */
    if (!tw->dev[prime]) {
      const size_t tb = 4ull * ((size_t)N / 2 + 1);
      CUDA_CHK(cudaMalloc((void**)&tw->dev[prime], tb));
      tw->dev_bytes += tb;
      g_dev_used += tb;
      CUDA_CHK(cudaMemcpyAsync(tw->dev[prime], tw->fwd[prime], tb,
                               cudaMemcpyHostToDevice, st));
    }
    const uint32_t* twF = tw->dev[prime];
    switch (prime) {
      case 0: {
        constexpr uint32_t P = kP[0];
        repack_kernel<<<grid_for(N), kThreads, 0, st>>>(W, (uint32_t)na, A[0], N, P);
        repack_kernel<<<grid_for(N), kThreads, 0, st>>>(W + woff_b, (uint32_t)nb, B[0], N, P);
        ntt_forward<P>(A[0], twF, logn, st);
        ntt_forward<P>(B[0], twF, logn, st);
        ntt_inverse_fused<P>(C[0], A[0], B[0], twF, logn, mod_inv(N, P), st);
        break;
      }
      case 1: {
        constexpr uint32_t P = kP[1];
        repack_kernel<<<grid_for(N), kThreads, 0, st>>>(W, (uint32_t)na, A[1], N, P);
        repack_kernel<<<grid_for(N), kThreads, 0, st>>>(W + woff_b, (uint32_t)nb, B[1], N, P);
        ntt_forward<P>(A[1], twF, logn, st);
        ntt_forward<P>(B[1], twF, logn, st);
        ntt_inverse_fused<P>(C[1], A[1], B[1], twF, logn, mod_inv(N, P), st);
        break;
      }
      default: {
        constexpr uint32_t P = kP[2];
        repack_kernel<<<grid_for(N), kThreads, 0, st>>>(W, (uint32_t)na, A[2], N, P);
        repack_kernel<<<grid_for(N), kThreads, 0, st>>>(W + woff_b, (uint32_t)nb, B[2], N, P);
        ntt_forward<P>(A[2], twF, logn, st);
        ntt_forward<P>(B[2], twF, logn, st);
        ntt_inverse_fused<P>(C[2], A[2], B[2], twF, logn, mod_inv(N, P), st);
        break;
      }
    }
  }

  /* The CRT stage consumes all three residue planes; join the side streams
   * and run the epilogue on stream 0. */
  CUDA_CHK(cudaStreamSynchronize(g_stream[1]));
  CUDA_CHK(cudaStreamSynchronize(g_stream[2]));
  CUDA_CHK(cudaMemsetAsync(acc, 0, 4ull * ((size_t)N + 3), g_stream[0]));
  crt_accum_kernel<kP[0], kP[1], kP[2]>
      <<<grid_for(N), kThreads, 0, g_stream[0]>>>(C[0], C[1], C[2], acc, N);
  carry_local_kernel<<<nchunks, kCarryThreads, 0, g_stream[0]>>>(
      acc, cmeta, (uint32_t)carry_limbs);
  carry_scan_kernel<<<1, 1, 0, g_stream[0]>>>(acc, cmeta,
                                              (uint32_t)carry_limbs, nchunks);
  carry_fix_kernel<<<nchunks, kCarryThreads, 0, g_stream[0]>>>(
      acc, cmeta, (uint32_t)carry_limbs);
  pack64_kernel<<<grid_for((uint32_t)nwords_out), kThreads, 0, g_stream[0]>>>(
      acc, outw, (uint32_t)out_limbs, (uint32_t)nwords_out);

  /* The product lands straight in `res` digit storage, via pinned staging. */
  mp_limb_t* dst = mpz_limbs_write(res, nwords_out);
  const size_t outsz = 8ull * (size_t)nwords_out;
  uint64_t* pinOut = (uint64_t*)pin_ensure(&g_pin_out, g_pin_out_bytes, outsz);
  CUDA_CHK(cudaMemcpyAsync(pinOut, outw, outsz, cudaMemcpyDeviceToHost,
                           g_stream[0]));
  CUDA_CHK(cudaStreamSynchronize(g_stream[0]));
  std::memcpy(dst, pinOut, outsz);
  g_last_gpu_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
  mpz_limbs_finish(res, nwords_out);
  ++g_gpu_calls;
  return true;
}

/* Three-recursion Karatsuba at the mpz level, splitting at a whole 28-bit
 * limb boundary.  The split point is capped at 2^23 limbs so that the low
 * product is GPU-feasible outright; the high product and the middle term
 * recurse with strictly smaller operands, so the recursion terminates. */
void karatsuba_abs(mpz_ptr res, mpz_srcptr xa, mpz_srcptr xb) {
  const size_t la = limbs_of(xa), lb = limbs_of(xb);
  size_t L = la > lb ? la : lb;
  size_t m = L / 2;
  if (m < 1) m = 1;
  if (m > (size_t)kMaxHalfLimbs) m = (size_t)kMaxHalfLimbs;
  if (m >= L) m = L - 1; /* keep both halves non-empty: sizes shrink */
  const unsigned long shift = (unsigned long)m * kLimbBits;

  mpz_t x0, x1, y0, y1, z0, z1, z2, s1, s2;
  mpz_init(x0); mpz_init(x1); mpz_init(y0); mpz_init(y1);
  mpz_init(z0); mpz_init(z1); mpz_init(z2); mpz_init(s1); mpz_init(s2);

  mpz_tdiv_r_2exp(x0, xa, shift);
  mpz_tdiv_q_2exp(x1, xa, shift);
  mpz_tdiv_r_2exp(y0, xb, shift);
  mpz_tdiv_q_2exp(y1, xb, shift);

  tachyon_cuda_mul(z0, x0, y0);
  tachyon_cuda_mul(z2, x1, y1);
  mpz_add(s1, x0, x1);
  mpz_add(s2, y0, y1);
  tachyon_cuda_mul(z1, s1, s2);
  mpz_sub(z1, z1, z0);
  mpz_sub(z1, z1, z2);

  mpz_mul_2exp(res, z2, shift);
  mpz_add(res, res, z1);
  mpz_mul_2exp(res, res, shift);
  mpz_add(res, res, z0);

  mpz_clear(x0); mpz_clear(x1); mpz_clear(y0); mpz_clear(y1);
  mpz_clear(z0); mpz_clear(z1); mpz_clear(z2); mpz_clear(s1); mpz_clear(s2);
}

}  /* namespace */

/* ------------------------------------------------------------------ *
 * Public entry points                                                *
 * ------------------------------------------------------------------ */

/* 1 iff a usable (sm_70+) CUDA device is present.  Probed once. */
extern "C" int tachyon_cuda_probe(void) {
  static const int usable = []() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0;
    if (prop.major < 7) return 0;
    return 1;
  }();
  return usable;
}

/* Milliseconds of the device section of the most recent tachyon_cuda_mul
 * (transfers + kernels + final sync, host wall clock).  0.0 when that call
 * took the small-N CPU path. */
extern "C" double tachyon_cuda_last_gpu_ms(void) { return g_last_gpu_ms; }

/* Exact c = a * b for arbitrary-signed operands. */
extern "C" void tachyon_cuda_mul(mpz_ptr c, mpz_srcptr a, mpz_srcptr b) {
  if (mpz_sgn(a) == 0 || mpz_sgn(b) == 0) {
    mpz_set_ui(c, 0);
    g_last_gpu_ms = 0.0;
    return;
  }

  const size_t la = limbs_of(a), lb = limbs_of(b);
  const uint64_t need = (uint64_t)la + (uint64_t)lb - 1;
  const uint64_t n64 = next_pow2_u64(need);

  /* Small products: GMP is simply faster than a kernel launch storm. */
  if (n64 <= (uint64_t)kNMinGpu) {
    mpz_mul(c, a, b);
    g_last_gpu_ms = 0.0;
    return;
  }

  const int sgn = mpz_sgn(a) * mpz_sgn(b);
  mpz_t xa, xb, prod;
  mpz_init(xa);
  mpz_init(xb);
  mpz_init(prod);
  mpz_abs(xa, a);
  mpz_abs(xb, b);

  if (n64 <= (uint64_t)kNMaxGpu) {
    if (!gpu_mul_abs(prod, xa, xb, (uint32_t)n64)) {
      /* Device budget exhausted: exact CPU fallback. */
      mpz_mul(c, a, b);
      g_last_gpu_ms = 0.0;
      mpz_clear(xa);
      mpz_clear(xb);
      mpz_clear(prod);
      return;
    }
  } else {
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t calls0 = g_gpu_calls;
    karatsuba_abs(prod, xa, xb);
    /* Wall time of the whole recursive device workload.  If every leaf fell
     * back to CPU the recursion touched no device at all, so report 0.0. */
    g_last_gpu_ms = (g_gpu_calls == calls0)
                        ? 0.0
                        : std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
  }

  if (sgn < 0) mpz_neg(prod, prod);
  mpz_set(c, prod);
  mpz_clear(xa);
  mpz_clear(xb);
  mpz_clear(prod);
}
