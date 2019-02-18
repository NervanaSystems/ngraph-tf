load("//:cxx_abi_option.bzl", "CXX_ABI")

cc_binary(
    name = 'libngraph_bridge.so',
    srcs = [
        "src/ngraph_api.cc",
        "src/ngraph_api.h",
        "src/ngraph_assign_clusters.cc",
        "src/ngraph_assign_clusters.h",
        "src/ngraph_builder.cc",
        "src/ngraph_builder.h",
        "src/ngraph_backend_manager.h",
        "src/ngraph_backend_manager.cc",
        "src/ngraph_capture_variables.cc",
        "src/ngraph_capture_variables.h",
        "src/ngraph_cluster_manager.cc",
        "src/ngraph_cluster_manager.h",
        "src/ngraph_conversions.h",
        "src/ngraph_deassign_clusters.cc",
        "src/ngraph_deassign_clusters.h",
        "src/ngraph_encapsulate_clusters.cc",
        "src/ngraph_encapsulate_clusters.h",
        "src/ngraph_encapsulate_op.cc",
        "src/ngraph_freshness_tracker.cc",
        "src/ngraph_freshness_tracker.h",
        "src/ngraph_mark_for_clustering.cc",
        "src/ngraph_mark_for_clustering.h",
        "src/ngraph_rewrite_for_tracking.cc",
        "src/ngraph_rewrite_for_tracking.h",
        "src/ngraph_rewrite_pass.cc",
        "src/ngraph_tracked_variable.cc",
        "src/ngraph_utils.cc",
        "src/ngraph_utils.h",
        "src/ngraph_version_utils.h",
        "src/tf_deadness_analysis.cc",
        "src/tf_deadness_analysis.h",
        "src/tf_graphcycles.cc",
        "src/tf_graphcycles.h",
        "src/version.h",
        "src/version.cc",
        "logging/ngraph_log.h",
        "logging/ngraph_log.cc",
        "logging/tf_graph_writer.h",
        "logging/tf_graph_writer.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@ngraph//:ngraph_headers",
        "@ngraph//:ngraph_core",
    ],
    copts = [
        "-pthread", 
        "-std=c++11", 
        "-DNDEBUG",
        "-D SHARED_LIB_PREFIX=\\"lib\\"",
        "-D SHARED_LIB_SUFFIX=\\".so\\"",
        "-I logging",
        "-I external/ngraph/src",
    ] + CXX_ABI,
    visibility = ["//visibility:public"],
)
