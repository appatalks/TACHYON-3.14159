// Minimal embedded HTTP/1.1 + SSE server for the dashboard (no dependencies).
#pragma once

namespace tachyon {

// Blocks until interrupted. `html` is served at "/".
// Returns non-zero on bind/listen failure.
int serve(int port, const char* html);

}  // namespace tachyon
