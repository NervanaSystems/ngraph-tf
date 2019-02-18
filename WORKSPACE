workspace(name = "ngraph_bridge")
load("//tf_configure:tf_configure.bzl", "tf_configure")

tf_configure(
    name = "local_config_tf",
)

new_http_archive(
    name = "ngraph",
    build_file = "ngraph.BUILD",
    sha256 = "1efd0cae2bc8febe40863727fcadf7eecbf7724073c5ddd2c95c6db00dd70985",
    strip_prefix = "ngraph-0.14.0-rc.1",
    urls = [
        "https://mirror.bazel.build/github.com/NervanaSystems/ngraph/archive/v0.14.0-rc.1.tar.gz",
        "https://github.com/NervanaSystems/ngraph/archive/v0.14.0-rc.1.tar.gz",
    ],
)

new_http_archive(
    name = "nlohmann_json_lib",
    build_file = "nlohmann_json.BUILD",
    sha256 = "e0b1fc6cc6ca05706cce99118a87aca5248bd9db3113e703023d23f044995c1d",
    strip_prefix = "json-3.5.0",
    urls = [
        "https://mirror.bazel.build/github.com/nlohmann/json/archive/v3.5.0.tar.gz",
        "https://github.com/nlohmann/json/archive/v3.5.0.tar.gz",
    ],
)
