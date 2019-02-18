licenses(["notice"])  # 3-Clause BSD
exports_files(["LICENSE"])

load("@ngraph_bridge//:cxx_abi_option.bzl", "CXX_ABI")

cc_library(
    name = "ngraph_headers",
    hdrs = glob(["src/ngraph/**/*.hpp"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ngraph_core",
    srcs = glob([
        "src/ngraph/*.cpp",
        "src/ngraph/autodiff/*.cpp",
        "src/ngraph/builder/*.cpp",
        "src/ngraph/descriptor/*.cpp",
        "src/ngraph/descriptor/layout/*.cpp",
        "src/ngraph/op/*.cpp",
        "src/ngraph/op/experimental/generate_mask.cpp",
        "src/ngraph/op/experimental/quantized_avg_pool.cpp",
        "src/ngraph/op/experimental/quantized_conv.cpp",
        "src/ngraph/op/experimental/quantized_conv_bias.cpp",
        "src/ngraph/op/experimental/quantized_conv_relu.cpp",
        "src/ngraph/op/experimental/quantized_max_pool.cpp",
        "src/ngraph/op/experimental/shape_of.cpp",
        "src/ngraph/op/util/*.cpp",
        "src/ngraph/pattern/*.cpp",
        "src/ngraph/pattern/*.hpp",
        "src/ngraph/pass/*.cpp",
        "src/ngraph/pass/*.hpp",
        "src/ngraph/runtime/*.cpp",
        "src/ngraph/type/*.cpp",
        ],
        exclude = [
        "src/ngraph/ngraph.cpp",
    ]),
    deps = [
        ":ngraph_headers",
        "@nlohmann_json_lib",
    ],
    copts = [
        "-I external/ngraph/src",
        "-I external/nlohmann_json_lib/include/",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        '-D NGRAPH_VERSION=\\"0.14.0-rc.1\\"',
        "-D NGRAPH_DEX_ONLY",
        '-D PROJECT_ROOT_DIR=\\"\\"',
    ] + CXX_ABI,
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_binary(
    name = 'libinterpreter_backend.so',
    srcs = glob([
        "src/ngraph/except.hpp",
        "src/ngraph/runtime/interpreter/*.cpp",
        "src/ngraph/state/rng_state.cpp",
    ]),
    deps = [
        ":ngraph_headers",
        ":ngraph_core",
    ],
    copts = [
        "-I external/ngraph/src",
        "-I external/ngraph/src/ngraph",
        "-I external/nlohmann_json_lib/include/",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        '-D NGRAPH_VERSION=\\"0.14.0-rc.1\\"',
        "-D NGRAPH_DEX_ONLY",
        '-D PROJECT_ROOT_DIR=\\"\\"',
    ] + CXX_ABI,
    linkshared = 1,
    visibility = ["//visibility:public"],
)
 
