#pragma once

extern "C" {
namespace tensorflow {
namespace ngraph_bridge {
namespace config {

extern void enable();
extern void disable();
extern bool is_enabled();
}
}
}
}
