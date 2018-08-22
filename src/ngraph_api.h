#pragma once

namespace tensorflow {
namespace ngraph_bridge {
namespace config {
extern "C" {
extern void ngraph_enable();
extern void ngraph_disable();
extern bool ngraph_is_enabled();
}

extern void enable();
extern void disable();
extern bool is_enabled();
}
}
}
