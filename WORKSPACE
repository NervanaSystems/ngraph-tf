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
    sha256 = "9f3549824af3ca7e9707a2503959886362801fb4926b869789d6929098a79e47",
    strip_prefix = "json-3.1.1",
    urls = [
        "https://mirror.bazel.build/github.com/nlohmann/json/archive/v3.1.1.tar.gz",
        "https://github.com/nlohmann/json/archive/v3.1.1.tar.gz",
    ],
)
