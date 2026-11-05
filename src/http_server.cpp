// Minimal embedded HTTP/1.1 + SSE server. POSIX sockets, thread-per-connection
// (capped), Connection: close. Serves the dashboard and the JSON/SSE API.
#include "http_server.hpp"
#include "jobs.hpp"
#include "engine.hpp"
#include "pi_chudnovsky.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace tachyon {
namespace {

std::atomic<int> g_active{0};

struct Req {
  std::string method, path, query, body;
  std::string qparam(const std::string& key) const {
    size_t i = query.find(key + "=");
    if (i == std::string::npos || (i > 0 && query[i - 1] != '?')) {
      i = query.find("&" + key + "=");
      if (i == std::string::npos) return "";
      i++;
    }
    size_t s = i + key.size() + 1, e = query.find('&', s);
    return query.substr(s, e == std::string::npos ? std::string::npos : e - s);
  }
};

bool read_head(int fd, Req& r) {
  std::string buf;
  char tmp[4096];
  size_t hdr_end = std::string::npos;
  while (hdr_end == std::string::npos && buf.size() < 65536) {
    ssize_t n = recv(fd, tmp, sizeof tmp, 0);
    if (n <= 0) return false;
    buf.append(tmp, n);
    hdr_end = buf.find("\r\n\r\n");
  }
  if (hdr_end == std::string::npos) return false;
  std::string head = buf.substr(0, hdr_end);
  r.body = buf.substr(hdr_end + 4);
  size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
  r.method = head.substr(0, sp1);
  std::string uri = head.substr(sp1 + 1, sp2 - sp1 - 1);
  size_t q = uri.find('?');
  r.path = q == std::string::npos ? uri : uri.substr(0, q);
  r.query = q == std::string::npos ? "" : uri.substr(q + 1);
  size_t cl = head.find("\r\nContent-Length: ");
  if (cl != std::string::npos) {
    size_t want = std::strtoul(head.c_str() + cl + 18, nullptr, 10);
    if (want > 65536) want = 65536;
    while (r.body.size() < want) {
      ssize_t n = recv(fd, tmp, sizeof tmp, 0);
      if (n <= 0) break;
      r.body.append(tmp, n);
    }
    r.body.resize(want);
  }
  return true;
}

void send_all(int fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    ssize_t n = send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
    if (n <= 0) return;
    off += n;
  }
}

void respond(int fd, int code, const char* ctype, const std::string& body,
             const char* extra = "") {
  const char* status = code == 200 ? "OK" : code == 400 ? "Bad Request"
                     : code == 404 ? "Not Found" : code == 409 ? "Conflict"
                     : code == 503 ? "Service Unavailable" : "Internal Server Error";
  std::ostringstream h;
  h << "HTTP/1.1 " << code << ' ' << status << "\r\n"
    << "Content-Type: " << ctype << "\r\n"
    << "Content-Length: " << body.size() << "\r\n"
    << "Connection: close\r\n"
    << "Cache-Control: no-store\r\n"
    << extra << "\r\n";
  std::string hs = h.str();
  send_all(fd, hs);
  send_all(fd, body);
}

std::string u64s(unsigned long long v) { return std::to_string(v); }

// ---- tiny JSON field extraction for POST /api/run ----
unsigned long long jget_u(const std::string& b, const std::string& key,
                          unsigned long long def) {
  size_t i = b.find("\"" + key + "\"");
  if (i == std::string::npos) return def;
  i = b.find(':', i);
  if (i == std::string::npos) return def;
  return std::strtoull(b.c_str() + i + 1, nullptr, 10);
}
bool jget_b(const std::string& b, const std::string& key, bool def) {
  size_t i = b.find("\"" + key + "\"");
  if (i == std::string::npos) return def;
  i = b.find(':', i);
  if (i == std::string::npos) return def;
  size_t j = b.find_first_not_of(" \t", i + 1);
  if (j == std::string::npos) return def;
  return b.compare(j, 4, "true") == 0;
}
std::string jget_s(const std::string& b, const std::string& key, const std::string& def) {
  size_t i = b.find("\"" + key + "\"");
  if (i == std::string::npos) return def;
  i = b.find(':', i);
  if (i == std::string::npos) return def;
  i = b.find('"', i);
  if (i == std::string::npos) return def;
  size_t e = b.find('"', i + 1);
  return e == std::string::npos ? def : b.substr(i + 1, e - i - 1);
}

std::string snap_json(const JobSnapshot& s) {
  char buf[1200];
  std::snprintf(buf, sizeof buf,
    "{\"id\":%llu,\"state\":\"%s\",\"phase\":\"%s\",\"frac\":%.4f,"
    "\"digits\":%llu,\"backend\":\"%s\",\"leaf\":%u,\"guard\":%u,\"verify\":%s,"
    "\"verify_fails\":%d,\"bs\":%.5f,\"sqrt\":%.5f,\"final\":%.5f,\"convert\":%.5f,"
    "\"verify_sec\":%.4f,\"total_sec\":%.4f,\"peak_rss_mb\":%.1f,"
    "\"head\":\"%s\",\"tail\":\"%s\",\"digits_kept\":%s,\"error\":\"%s\"}",
    s.id, s.state.c_str(), s.phase.c_str(), s.frac, s.opt.digits, s.opt.backend.c_str(),
    s.opt.leaf, s.opt.guard, s.verify ? "true" : "false", s.verify_fails,
    s.tm.bs, s.tm.sqrt, s.tm.final, s.tm.convert, s.verify_sec, s.total_sec,
    s.peak_rss_mb, s.head.c_str(), s.tail.c_str(), s.digits_kept ? "true" : "false",
    s.error.c_str());
  return buf;
}

std::string read_file_jsonl_array(const char* path) {
  std::string out = "[";
  std::ifstream f(path);
  std::string line;
  bool first = true;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (!first) out += ',';
    out += line;
    first = false;
  }
  out += ']';
  return out;
}

// ---- scheduler cost model v0 (device mapping) ----
// Bits per Chudnovsky term: 14.1814 decimal digits = 47.09 bits.
// CPU mul time model: calibrated to the measured single-thread baseline at 1M/10M/100M
// (see BENCHMARKS.md); GPU + transfer models are placeholders until E4 calibration.
std::string mapping_json(unsigned long long D, const std::string& backend) {
  const unsigned leaf = 8;
  const double bits_per_term = 47.09;
  const double D_bits = (double)D * 3.321928;
  unsigned long long N = (unsigned long long)((double)(D + 40) / 14.18142587) + 4;
  int levels = 1;
  while ((unsigned long long)leaf << levels < N) ++levels;
  bool gpu_ok = (backend == "gpu" || backend == "hybrid");

  // model constants (cost-model-v2: both sides fit to E4 measurements, 2026-11-04)
  // Convention: `bits` below is the node's PRODUCT size (combine result). Both sides are
  // per-combine estimates and carry the same kMuls factor, so the dispatch ratio is honest.
  // GMP per-mul: kGmpMsPerWordStep * (bits/64) * log2(bits/64) ms fits E4 rows within ~20%
  // (2^24-product row 33.6 vs measured 40.5 ms). Each BS combine does ~3 full-size products
  // (P*Q, Q*Q, T1*Q2, T2*P1 -> ~3.1 equivalent); kMuls also reproduces the measured
  // 10M-digit split phase: model 3.3 s vs measured 3.8 s.
  const double kGmpMsPerWordStep = 1.0e-5;
  const double kMuls = 3.1;
  // E4-calibrated (2026-11-04): RTX 5070 Ti, three-prime 28-bit-limb NTT, LLM resident.
  // One GPU mul = 3 transforms of nb points (nb = next_pow2(product_bits/28)); per-mul rate
  // fits 3*nb*log2(nb)/gpu_ms = 2.5e7 butterflies/ms at the 2^24-product row (4.43 ms).
  const double gpu_ops_per_ms = 2.5e7;        // butterflies/ms (E4 fit; was 6e9 pre-E4 guess)
  const double pcie_GBs = 26.0;               // effective pageable+copy pipeline
  const double launch_ms = 0.24;              // ~240 kernel launches, E4-measured floor

  std::ostringstream o;
  o << "{\"digits\":" << D << ",\"terms\":" << N << ",\"levels\":" << levels
    << ",\"model\":\"cost-model-v2 (calibrated by E4, 2026-11-04)\""
    << ",\"gpu_available\":" << (gpu_ok ? "true" : "false") << ",\"layers\":[";
  double cpu_total = 0, gpu_best_total = 0;
  long long gpu_nodes = 0, cpu_nodes = 0;
  for (int l = 0; l < levels; ++l) {
    unsigned long long span = (unsigned long long)leaf << l;
    unsigned long long count = (N + span - 1) / span;
    if (count < 1) count = 1;
    double bits = (double)span * bits_per_term;
    if (bits > D_bits) bits = D_bits;
    double n = bits / 64.0;  // 64-bit words
    double cpu_ms = count * kMuls * kGmpMsPerWordStep * n * std::log2(n + 2.0);
    // E4-calibrated GPU model: transform length nb = next_pow2(product_bits/28) (operands are
    // half-product size, limbs are 28-bit); 3 transforms (two forward + one inverse) of
    // nb*log2(nb) butterflies per mul; above nb=2^24 the backend recurses into 3-way
    // Karatsuba (measured ~3x work per extra level).
    double nb = std::pow(2.0, std::ceil(std::log2(bits / 28.0 + 1.0)));
    double kar = 1.0;
    while (nb > 16777216.0) { nb /= 2.0; kar *= 3.0; }
    double gpu_ms = count * kMuls * kar * (launch_ms + 3.0 * nb * std::log2(nb + 2.0) / gpu_ops_per_ms);
    double xfer_ms = count * (2.0 * bits / 8.0) / (pcie_GBs * 1e6);  // MB/ms scale
    bool use_gpu = gpu_ok && gpu_ms + xfer_ms < 0.8 * cpu_ms && bits >= 524288.0 &&
                   (backend == "gpu" || l >= levels / 2);  // hybrid: upper tree half
    if (use_gpu) {
      ++gpu_nodes;
      gpu_best_total += gpu_ms + xfer_ms;
      cpu_total += cpu_ms;
    } else {
      ++cpu_nodes;
      gpu_best_total += cpu_ms;
      cpu_total += cpu_ms;
    }
    char buf[256];
    std::snprintf(buf, sizeof buf,
        "{\"level\":%d,\"span_terms\":%llu,\"count\":%llu,\"node_bits\":%.0f,"
        "\"device\":\"%s\",\"est_cpu_ms\":%.3f,\"est_gpu_ms\":%.3f,\"est_xfer_ms\":%.3f}",
        l, span, count, bits,
        use_gpu ? "gpu" : "cpu",
        cpu_ms, gpu_ms, xfer_ms);
    if (l) o << ',';
    o << buf;
  }
  o << "],\"summary\":{\"cpu_nodes\":" << cpu_nodes << ",\"gpu_nodes\":" << gpu_nodes
    << ",\"cpu_only_est_ms\":" << (long long)cpu_total
    << ",\"planned_est_ms\":" << (long long)gpu_best_total
    << ",\"est_speedup\":" << (gpu_best_total > 0 ? cpu_total / gpu_best_total : 1.0) << "}"
    << ",\"phases\":[{\"name\":\"split\",\"share\":0.65},{\"name\":\"convert\",\"share\":0.11},"
    << "{\"name\":\"final\",\"share\":0.05},{\"name\":\"sqrt\",\"share\":0.04},"
    << "{\"name\":\"overhead\",\"share\":0.15}]";
  // measured shares for reference when history exists
  o << "}";
  return o.str();
}

void handle(int fd, const Req& r, const char* html, JobManager& jm) {
  if (r.path == "/" || r.path == "/index.html") {
    respond(fd, 200, "text/html; charset=utf-8", html);
    return;
  }
  if (r.path == "/health") {
    respond(fd, 200, "application/json",
            "{\"ok\":true,\"cuda\":"
#ifdef TACHYON_HAVE_CUDA
            "true"
#else
            "false"
#endif
            ",\"active\":" + u64s(g_active.load()) + "}");
    return;
  }
  if (r.path == "/api/jobs") {
    std::string out = "[";
    auto v = jm.list();
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) out += ',';
      out += snap_json(v[i]);
    }
    out += ']';
    respond(fd, 200, "application/json", out);
    return;
  }
  if (r.path == "/api/job") {
    unsigned long long id = std::strtoull(r.qparam("id").c_str(), nullptr, 10);
    std::string what = r.qparam("what");
    if (what == "digits") {
      std::string s;
      if (!jm.digits(id, s)) { respond(fd, 404, "application/json", "{\"error\":\"no digits kept\"}"); return; }
      size_t off = std::strtoul(r.qparam("offset").c_str(), nullptr, 10);
      size_t maxn = std::strtoul(r.qparam("max").c_str(), nullptr, 10);
      if (maxn == 0 || maxn > 10000000) maxn = 10000000;
      respond(fd, 200, "text/plain; charset=utf-8", s.substr(std::min(off, s.size()), maxn));
      return;
    }
    if (what == "stats") {
      long long freq[10];
      std::string meta;
      if (!jm.stats(id, freq, meta)) { respond(fd, 404, "application/json", "{\"error\":\"no digits kept\"}"); return; }
      std::string out = "{\"freq\":[";
      for (int i = 0; i < 10; ++i) { if (i) out += ','; out += std::to_string(freq[i]); }
      out += "]";
      if (!meta.empty() && meta.front() == '{') meta.erase(0, 1);  // splice in meta object
      out += "," + meta;
      respond(fd, 200, "application/json", out);
      return;
    }
    JobSnapshot s;
    if (!jm.snapshot(id, s)) { respond(fd, 404, "application/json", "{\"error\":\"no such job\"}"); return; }
    respond(fd, 200, "application/json", snap_json(s));
    return;
  }
  if (r.path == "/api/stream") {
    unsigned long long id = std::strtoull(r.qparam("id").c_str(), nullptr, 10);
    const char* hdr =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
    send_all(fd, hdr);
    for (int tick = 0; tick < 36000; ++tick) {  // 30 min cap
      JobSnapshot s;
      if (!jm.snapshot(id, s)) { send_all(fd, "event: err\ndata: {\"error\":\"gone\"}\n\n"); break; }
      send_all(fd, "data: " + snap_json(s) + "\n\n");
      if (s.state == "done" || s.state == "failed") break;
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return;
  }
  if (r.path == "/api/run" && r.method == "POST") {
    PiOptions opt;
    opt.digits = jget_u(r.body, "digits", 0);
    opt.backend = jget_s(r.body, "backend", "cpu");
    opt.leaf = (unsigned)jget_u(r.body, "leaf", 8);
    opt.guard = (unsigned)jget_u(r.body, "guard", 30);
    bool verify = jget_b(r.body, "verify", false);
    if (opt.digits < 2 || opt.digits > 1000000000ull) {
      respond(fd, 400, "application/json", "{\"error\":\"digits must be in [2, 1e9]\"}");
      return;
    }
    if (opt.backend != "cpu" && opt.backend != "gpu" && opt.backend != "hybrid") {
      respond(fd, 400, "application/json", "{\"error\":\"bad backend\"}");
      return;
    }
    if (opt.guard < 10) opt.guard = 30;
    if (opt.leaf < 1 || opt.leaf > 1024) opt.leaf = 8;
    unsigned long long id = jm.submit(opt, verify);
    respond(fd, 200, "application/json", "{\"id\":" + u64s(id) + "}");
    return;
  }
  if (r.path == "/api/mapping") {
    unsigned long long D = std::strtoull(r.qparam("digits").c_str(), nullptr, 10);
    if (D < 100) D = 1000000;
    respond(fd, 200, "application/json", mapping_json(D, r.qparam("backend")));
    return;
  }
  if (r.path == "/api/benchmarks") {
    respond(fd, 200, "application/json", read_file_jsonl_array("benchmarks/results.jsonl"));
    return;
  }
  if (r.path == "/api/history") {
    respond(fd, 200, "application/json", jm.history_json());
    return;
  }
  respond(fd, 404, "application/json", "{\"error\":\"not found\"}");
}

}  // namespace

int serve(int port, const char* html) {
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) return 1;
  int one = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons((unsigned short)port);
  if (bind(srv, (sockaddr*)&addr, sizeof addr) < 0) {
    std::fprintf(stderr, "error: cannot bind 127.0.0.1:%d (%s)\n", port, std::strerror(errno));
    return 1;
  }
  if (listen(srv, 32) < 0) return 1;
  std::printf("[tachyon] dashboard: http://127.0.0.1:%d\n", port);
  std::fflush(stdout);

  JobManager jm;
  for (;;) {
    int fd = accept(srv, nullptr, nullptr);
    if (fd < 0) continue;
    if (g_active.load() > 16) {
      respond(fd, 503, "application/json", "{\"error\":\"too many connections\"}");
      close(fd);
      continue;
    }
    ++g_active;
    std::thread([fd, html, &jm] {
      Req r;
      if (read_head(fd, r)) {
        try {
          handle(fd, r, html, jm);
        } catch (...) {
          respond(fd, 500, "application/json", "{\"error\":\"internal\"}");
        }
      }
      close(fd);
      --g_active;
    }).detach();
  }
}

}  // namespace tachyon
