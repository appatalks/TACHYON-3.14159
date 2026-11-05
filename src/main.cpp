// tachyon-pi: heterogeneous pi engine CLI.
#include "pi_chudnovsky.hpp"
#include "verify.hpp"
#include "engine.hpp"
#include "http_server.hpp"
#include "web_dashboard.hpp"
#include <sys/resource.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <algorithm>

static const char* REF100 =
    "3.14159265358979323846264338327950288419716939937510"
    "58209749445923078164062862089986280348253421170679";  // 3. + 99 decimals

static long peak_rss_kb() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss;  // KB on Linux
}

static void usage() {
  std::printf(
    "tachyon-pi --digits N [--backend cpu|gpu|hybrid] [--verify] [--benchmark]\n"
    "           [--trials K] [--out FILE] [--json FILE] [--threads T] [--leaf L]\n"
    "           [--guard G] [--gpu-mem MB] [--quiet]\n"
    "  --digits N     total significant decimal digits of pi (N>=2)\n"
    "  --verify       BBP hex spot checks + reference-digits check\n"
    "  --benchmark    run warmup + trials, report medians and phase breakdown\n"
    "  --out FILE     write full decimal expansion (timing reported separately)\n"
    "  --json FILE    append machine-readable benchmark record\n"
    "  --serve [PORT] start the dashboard web server (default 8080) instead of computing\n");
}

int main(int argc, char** argv) {
  PiOptions opt;
  bool verify = false, bench = false, quiet = false, serve_mode = false;
  int trials = 3, serve_port = 8080;
  std::string outfile, jsonfile;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "error: %s needs a value\n", what); std::exit(2); }
      return argv[++i];
    };
    if (a == "--digits") opt.digits = std::strtoull(next("--digits"), nullptr, 10);
    else if (a == "--backend") opt.backend = next("--backend");
    else if (a == "--threads") opt.threads = std::atoi(next("--threads"));
    else if (a == "--leaf") opt.leaf = std::atoi(next("--leaf"));
    else if (a == "--guard") opt.guard = std::atoi(next("--guard"));
    else if (a == "--gpu-mem") opt.gpu_mem_mb = std::strtoull(next("--gpu-mem"), nullptr, 10);
    else if (a == "--trials") trials = std::atoi(next("--trials"));
    else if (a == "--out") outfile = next("--out");
    else if (a == "--json") jsonfile = next("--json");
    else if (a == "--serve") {
      serve_mode = true;
      if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
        serve_port = std::atoi(argv[++i]);
    }
    else if (a == "--verify") verify = true;
    else if (a == "--benchmark") bench = true;
    else if (a == "--quiet") quiet = true;
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else { std::fprintf(stderr, "error: unknown arg '%s'\n", a.c_str()); usage(); return 2; }
  }
  if (serve_mode) return tachyon::serve(serve_port, DASHBOARD_HTML);
  if (opt.digits < 2) { std::fprintf(stderr, "error: --digits must be >= 2\n"); return 2; }
  if (opt.guard < 10) opt.guard = 30;
  if (trials < 1) trials = 1;

  mpz_t res;
  mpz_init(res);
  PiTimings tm, tm_last;
  std::vector<double> totals;
  int rc = 0;

  opt.time_convert = bench;
  const int runs = bench ? trials + 1 : 1;  // first run is warmup for --benchmark
  for (int r = 0; r < runs; ++r) {
    tm = PiTimings{};
    compute_pi(opt, res, tm);
    if (!outfile.empty()) {
      auto w0 = std::chrono::steady_clock::now();
      std::string s = to_decimal(res, opt.digits, opt.threads > 1 ? (unsigned)opt.threads : 1);
      std::ofstream f(outfile);
      f << s << "\n";
      f.close();
      auto w1 = std::chrono::steady_clock::now();
      tm.write = std::chrono::duration<double>(w1 - w0).count();
    }
    if (bench && r > 0) totals.push_back(tm.total);
    tm_last = tm;
    if (bench && !quiet)
      std::printf("trial %d/%d total %.4fs%s\n", r, runs - 1, tm.total,
                  r == 0 ? " (warmup)" : "");
  }

  double median = tm_last.total;
  if (!totals.empty()) {
    std::sort(totals.begin(), totals.end());
    median = totals[totals.size() / 2];
  }

  if (!quiet) {
    std::string s = to_decimal(res, opt.digits, opt.threads > 1 ? (unsigned)opt.threads : 1);
    if (opt.digits <= 70) std::printf("%s\n", s.c_str());
    else std::printf("%s...%s\n", s.substr(0, 62).c_str(), s.substr(s.size() - 12).c_str());
    std::printf("[tachyon] backend=%s engine=%s digits=%llu leaf=%u guard=%u\n",
                opt.backend.c_str(), get_engine(opt.backend, opt.threads).name(), opt.digits,
                opt.leaf, opt.guard);
    std::printf("[timing]  split %.4fs | sqrt %.4fs | final %.4fs | convert %.4fs",
                tm_last.bs, tm_last.sqrt, tm_last.final, tm_last.convert);
    if (tm_last.write > 0) std::printf(" | write %.4fs", tm_last.write);
    std::printf("\n[timing]  total %.4fs", bench ? median : tm_last.total);
    if (bench && totals.size() > 1) {
      std::printf(" (median of %zu)", totals.size());
    }
    std::printf("\n[resource] peak RSS %.1f MB\n", peak_rss_kb() / 1024.0);
    std::printf("[rate]    %.3g digits/s (total)\n", opt.digits / (bench ? median : tm_last.total));
  }

  if (verify) {
    std::printf("[verify]  BBP hex-digit spot checks (independent formula):\n");
    int fails = 0;
    unsigned long long pos[4] = {13, 1000, 50000, opt.digits / 3 + 7};
    for (auto p : pos) {
      if ((long double)p + 14 < (long double)opt.digits * 0.83L)
        fails += !bbp_check_hex(res, opt.digits, p);
    }
    // embedded reference: first 101 significant digits
    std::string s = to_decimal(res, opt.digits, opt.threads > 1 ? (unsigned)opt.threads : 1);
    size_t n = std::strlen(REF100);
    if (s.size() + 1 >= n && opt.digits >= 100) {
      if (s.compare(0, n - 2, REF100, n - 2) != 0) { fails++; std::printf("  reference 101 digits: FAIL\n"); }
      else std::printf("  reference first %zu digits: ok\n", n - 2);
    }
    std::printf("[verify]  %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    rc = fails ? 1 : 0;
  }

  if (!jsonfile.empty()) {
    std::ofstream f(jsonfile, std::ios::app);
    f << "{\"digits\":" << opt.digits << ",\"backend\":\"" << opt.backend
      << "\",\"engine\":\"" << get_engine(opt.backend, opt.threads).name()
      << "\",\"leaf\":" << opt.leaf << ",\"threads\":" << opt.threads
      << ",\"trials\":" << (bench ? trials : 1)
      << ",\"median_total\":" << median << ",\"bs\":" << tm_last.bs
      << ",\"sqrt\":" << tm_last.sqrt << ",\"final\":" << tm_last.final
      << ",\"convert\":" << tm_last.convert << ",\"write\":" << tm_last.write
      << ",\"peak_rss_mb\":" << peak_rss_kb() / 1024.0 << "}\n";
  }

  mpz_clear(res);
  return rc;
}
