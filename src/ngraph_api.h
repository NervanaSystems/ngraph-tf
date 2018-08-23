#pragma once

#include <string.h>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace config {
extern "C" {
extern void ngraph_enable();
extern void ngraph_disable();
extern bool ngraph_is_enabled();

extern size_t ngraph_backends_len();
extern bool ngraph_list_backends(char** backends, int backends_len);
extern bool ngraph_set_backend(const char* backend);
}

extern void enable();
extern void disable();
extern bool is_enabled();

extern size_t backends_len();
// TODO: why is this not const?
extern vector<string> list_backends();
extern void set_backend(const string& type);
}
}
}
