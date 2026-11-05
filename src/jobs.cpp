// Job manager implementation: serialized worker, snapshots, JSONL history.
#include "jobs.hpp"
#include "verify.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <sys/resource.h>

namespace tachyon {
namespace {

std::string jesc(const std::string& s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      default: if ((unsigned char)c >= 32) o += c;
    }
  }
  return o;
}

std::string job_json(const JobManager* /*m*/, const std::string& id, const std::string& state,
                     const std::string& phase, double frac, const PiOptions& opt, bool verify,
                     int vf, const PiTimings& tm, double vsec, double tsec, double rss,
                     const std::string& head, const std::string& tail, bool kept,
                     const std::string& error) {
  char buf[1024];
  std::snprintf(buf, sizeof buf,
      "{\"id\":%s,\"state\":\"%s\",\"phase\":\"%s\",\"frac\":%.4f,"
      "\"digits\":%llu,\"backend\":\"%s\",\"leaf\":%u,\"guard\":%u,\"verify\":%s,"
      "\"verify_fails\":%d,\"bs\":%.5f,\"sqrt\":%.5f,\"final\":%.5f,\"convert\":%.5f,"
      "\"verify_sec\":%.4f,\"total_sec\":%.4f,\"peak_rss_mb\":%.1f,"
      "\"head\":\"%s\",\"tail\":\"%s\",\"digits_kept\":%s,\"error\":\"%s\"}",
      id.c_str(), state.c_str(), phase.c_str(), frac, opt.digits,
      jesc(opt.backend).c_str(), opt.leaf, opt.guard, verify ? "true" : "false", vf,
      tm.bs, tm.sqrt, tm.final, tm.convert, vsec, tsec, rss, jesc(head).c_str(),
      jesc(tail).c_str(), kept ? "true" : "false", jesc(error).c_str());
  return buf;
}

}  // namespace

JobManager::JobManager() {
  worker_ = std::thread([this] {
    for (;;) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock lk(mu_);
        queue_.erase(std::remove_if(queue_.begin(), queue_.end(),
                        [](auto& j) { return j->state == "cancelled"; }),
                     queue_.end());
        if (queue_.empty()) {
          if (stop_) return;
          lk.unlock();
          std::this_thread::sleep_for(std::chrono::milliseconds(120));
          continue;
        }
        job = queue_.front();
        queue_.pop_front();
        job->state = "running";
      }
      run(job.get());
      persist(job.get());
      std::lock_guard lk(mu_);
      finished_.push_back(job);
      if (finished_.size() > 200) finished_.pop_front();
    }
  });
}

JobManager::~JobManager() {
  {
    std::lock_guard lk(mu_);
    stop_ = true;
  }
  if (worker_.joinable()) worker_.join();
}

unsigned long long JobManager::submit(const PiOptions& opt, bool verify) {
  std::lock_guard lk(mu_);
  auto j = std::make_shared<Job>();
  j->id = next_id_++;
  j->opt = opt;
  j->opt.progress = nullptr;  // rebound in run()
  j->verify = verify;
  j->t_start = std::chrono::steady_clock::now();
  jobs_[j->id] = j;
  queue_.push_back(j);
  return j->id;
}

void JobManager::run(Job* job) {
  job->opt.progress = [this, job](const char* phase, double frac) {
    std::lock_guard lk(mu_);
    job->phase = phase;
    job->frac = frac;
  };
  try {
    mpz_t res;
    mpz_init(res);
    compute_pi(job->opt, res, job->tm);
    std::string s = to_decimal(res, job->opt.digits,
                               job->opt.threads > 1 ? (unsigned)job->opt.threads : 1);
    {
      std::lock_guard lk(mu_);
      job->head = s.size() > 74 ? s.substr(0, 62) + "\xe2\x80\xa6" : s;
      job->tail = s.size() > 74 ? s.substr(s.size() - 12) : "";
      if (s.size() <= 10000010) {  // keep up to ~10M digits in RAM
        job->decimal = std::move(s);
        job->digits_kept = true;
      }
    }
    if (job->verify) {
      {
        std::lock_guard lk(mu_);
        job->state = "verifying";
        job->phase = "bbp";
      }
      auto v0 = std::chrono::steady_clock::now();
      int f = bbp_verify(res, job->opt.digits, 0);
      if (f < 0) {  // precision too small for BBP: reference fallback
        f = (job->decimal.compare(0, 53,
              "3.14159265358979323846264338327950288419716939937510") != 0) ? 1 : 0;
      }
      auto v1 = std::chrono::steady_clock::now();
      std::lock_guard lk(mu_);
      job->verify_fails = f;
      job->verify_sec = std::chrono::duration<double>(v1 - v0).count();
    }
    mpz_clear(res);
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    std::lock_guard lk(mu_);
    job->peak_rss_mb = ru.ru_maxrss / 1024.0;
    job->total_sec = job->tm.total + job->verify_sec;
    job->state = "done";
    job->phase = "done";
    job->frac = 1.0;
  } catch (const std::exception& e) {
    std::lock_guard lk(mu_);
    job->state = "failed";
    job->error = e.what();
  } catch (...) {
    std::lock_guard lk(mu_);
    job->state = "failed";
    job->error = "unknown exception";
  }
}

void JobManager::persist(Job* job) {
  std::lock_guard lk(mu_);
  std::string line = job_json(this, std::to_string(job->id), job->state, job->phase, job->frac,
                              job->opt, job->verify, job->verify_fails, job->tm, job->verify_sec,
                              job->total_sec, job->peak_rss_mb, job->head, job->tail,
                              job->digits_kept, job->error);
  std::FILE* f = std::fopen("benchmarks/server_runs.jsonl", "a");
  if (f) {
    std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
  }
}

bool JobManager::snapshot(unsigned long long id, JobSnapshot& out) {
  std::lock_guard lk(mu_);
  auto it = jobs_.find(id);
  if (it == jobs_.end()) return false;
  Job& j = *it->second;
  out = JobSnapshot{};
  out.id = j.id; out.state = j.state; out.phase = j.phase; out.frac = j.frac;
  out.opt = j.opt; out.opt.progress = nullptr;
  out.verify = j.verify; out.verify_fails = j.verify_fails; out.tm = j.tm;
  out.verify_sec = j.verify_sec; out.total_sec = j.total_sec; out.peak_rss_mb = j.peak_rss_mb;
  out.head = j.head; out.tail = j.tail; out.digits_kept = j.digits_kept; out.error = j.error;
  return true;
}

std::vector<JobSnapshot> JobManager::list() {
  std::vector<JobSnapshot> v;
  std::lock_guard lk(mu_);
  for (auto& [id, j] : jobs_) {
    JobSnapshot s;
    s.id = j->id; s.state = j->state; s.phase = j->phase; s.frac = j->frac;
    s.opt = j->opt; s.opt.progress = nullptr;
    s.verify = j->verify; s.verify_fails = j->verify_fails; s.tm = j->tm;
    s.verify_sec = j->verify_sec; s.total_sec = j->total_sec; s.peak_rss_mb = j->peak_rss_mb;
    s.head = j->head; s.tail = j->tail; s.digits_kept = j->digits_kept; s.error = j->error;
    v.push_back(std::move(s));
  }
  return v;
}

bool JobManager::digits(unsigned long long id, std::string& out) {
  std::lock_guard lk(mu_);
  auto it = jobs_.find(id);
  if (it == jobs_.end() || !it->second->digits_kept) return false;
  out = it->second->decimal;
  return true;
}

bool JobManager::stats(unsigned long long id, long long freq[10], std::string& meta) {
  std::lock_guard lk(mu_);
  auto it = jobs_.find(id);
  if (it == jobs_.end() || !it->second->digits_kept) return false;
  const std::string& s = it->second->decimal;
  for (int i = 0; i < 10; ++i) freq[i] = 0;
  for (char c : s)
    if (c >= '0' && c <= '9') ++freq[c - '0'];
  char buf[128];
  std::snprintf(buf, sizeof buf, "{\"scanned\":%zu,\"kept\":true}", s.size());
  meta = buf;
  return true;
}

std::string JobManager::history_json() {
  std::string out = "[";
  bool first = true;
  std::FILE* f = std::fopen("benchmarks/server_runs.jsonl", "r");
  if (f) {
    char line[4096];
    while (std::fgets(line, sizeof line, f)) {
      size_t n = std::strlen(line);
      while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
      if (!n) continue;
      if (!first) out += ',';
      out += line;
      first = false;
    }
    std::fclose(f);
  }
  out += ']';
  return out;
}

}  // namespace tachyon
