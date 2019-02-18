licenses(["notice"])  # 3-Clause BSD

exports_files(["LICENSE.MIT"])
load("@ngraph_bridge//:cxx_abi_option.bzl", "CXX_ABI")

cc_library(
    name = "nlohmann_json_lib",
    hdrs = glob([
        "include/nlohmann/**/*.hpp",
    ]),
    copts = [
        "-I external/nlohmann_json_lib",
    ]+ CXX_ABI,
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
