namespace tensorflow {
namespace ngraph_bridge {
namespace config {
extern "C" {

static bool _is_enabled = true;

void ngraph_enable() { _is_enabled = true; }
void ngraph_disable() { _is_enabled = false; }
bool ngraph_is_enabled() { return _is_enabled; }
}

void enable() { ngraph_enable(); }
void disable() { ngraph_disable(); }
bool is_enabled() { return ngraph_is_enabled(); }
}
}
}
