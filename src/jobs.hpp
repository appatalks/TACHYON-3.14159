// Background compute jobs for the embedded dashboard server.
#pragma once
#include "pi_chudnovsky.hpp"
#include <gmp.h>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tachyon {

struct JobSnapshot {
  unsigned long long id = 0;
  std::string state;  // queued|running|verifying|done|failed
  std::string phase;  // split|sqrt|final|convert|bbp|reference|done
  double frac = 0;    // progress of current phase (split only sub-divides)
  PiOptions opt;
  bool verify = false;
  int verify_fails = -1;  // -1 = not verified yet
  PiTimings tm;
  double verify_sec = 0, total_sec = 0, peak_rss_mb = 0;
  std::string head, tail;  // decimal display fragments
  bool digits_kept = false;
  std::string error;
};

class JobManager {
 public:
  JobManager();
  ~JobManager();

  unsigned long long submit(const PiOptions& opt, bool verify);
  bool snapshot(unsigned long long id, JobSnapshot& out);
  std::vector<JobSnapshot> list();
  // digits: full decimal "3.14159..." (kept up to 10M digits); empty if not kept.
  bool digits(unsigned long long id, std::string& out);
  // frequency stats of the decimal digits (0-9); counts digits_kept content.
  bool stats(unsigned long long id, long long freq[10], std::string& meta);
  // append-only JSONL history of finished jobs (server side, separate from CLI).
  std::string history_json();

 private:
  struct Job {
    unsigned long long id;
    PiOptions opt;
    bool verify;
    std::string state = "queued", phase = "queued", error;
    double frac = 0;
    int verify_fails = -1;
    PiTimings tm;
    double verify_sec = 0, total_sec = 0, peak_rss_mb = 0;
    std::string head, tail, decimal;  // decimal empty when not kept
    bool digits_kept = false;
    std::chrono::steady_clock::time_point t_start;
  };

  void run(Job* job);
  void persist(Job* job);

  std::mutex mu_;
  std::map<unsigned long long, std::shared_ptr<Job>> jobs_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::deque<std::shared_ptr<Job>> finished_;  // JSONL history mirror
  unsigned long long next_id_ = 1;
  bool stop_ = false;
  std::thread worker_;
};

}  // namespace tachyon
