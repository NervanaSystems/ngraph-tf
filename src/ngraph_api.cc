#include "ngraph/runtime/backend.hpp"

#include "ngraph_api.h"

namespace tensorflow {
namespace ngraph_bridge {
namespace config {

static bool _is_enabled = true;
static bool _is_logging_placement = false;

extern "C" {
void ngraph_enable() { enable(); }
void ngraph_disable() { disable(); }
bool ngraph_is_enabled() { return is_enabled(); }

size_t ngraph_backends_len() { return backends_len(); }
bool ngraph_list_backends(char** backends, int backends_len) {
  const auto ngraph_backends = list_backends();
  if (backends_len != ngraph_backends.size()) {
    return false;
  }

  for (size_t idx = 0; idx < backends_len; idx++) {
    backends[idx] = strdup(ngraph_backends[idx].c_str());
  }
  return true;
}
bool ngraph_set_backend(const char* backend) {
  try {
    set_backend(string(backend));
  } catch (const runtime_error& e) {
    return false;
  }
  return true;
}

void ngraph_start_logging_placement() { start_logging_placement(); }
void ngraph_stop_logging_placement() { stop_logging_placement(); }
bool ngraph_is_logging_placement() { return is_logging_placement(); }
}

void enable() { _is_enabled = true; }
void disable() { _is_enabled = false; }
bool is_enabled() { return _is_enabled; }

size_t backends_len() { return list_backends().size(); }
vector<string> list_backends() {
  return ngraph::runtime::Backend::get_registered_devices();
}
void set_backend(const string& type) { ngraph::runtime::Backend::create(type); }

void start_logging_placement() { _is_logging_placement = true; }
void stop_logging_placement() { _is_logging_placement = false; }
bool is_logging_placement() { return _is_logging_placement; }
}
}
}
