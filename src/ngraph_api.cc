extern "C" {
namespace tensorflow {
namespace ngraph_bridge {
namespace config {

static bool ngraph_is_enabled = true;

void enable() { ngraph_is_enabled = true; }

void disable() { ngraph_is_enabled = false; }

bool is_enabled() { return ngraph_is_enabled; }
}
}
}
}
