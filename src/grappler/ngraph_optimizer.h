/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#pragma once

#ifndef NGRAPH_TF_NGRAPHOPTIMIZER_H_
#define NGRAPH_TF_NGRAPHOPTIMIZER_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

#include "ngraph_api.h"
#include "ngraph_assign_clusters.h"
#include "ngraph_capture_variables.h"
#include "ngraph_deassign_clusters.h"
#include "ngraph_encapsulate_clusters.h"
#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_rewrite_for_tracking.h"
#include "tf_graph_writer.h"

#include <iomanip>

namespace tensorflow {

namespace ngraph_bridge {

// Custom Grappler Optimizer for NGraph-TF
class NgraphOptimizer : public tensorflow::grappler::CustomGraphOptimizer {
 public:
  NgraphOptimizer() = default;
  ~NgraphOptimizer() override = default;

  string name() const override { return "NgraphOptimizer"; };

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return Status::OK();
  }

  Status Optimize(tensorflow::grappler::Cluster* cluster,
                  const tensorflow::grappler::GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(tensorflow::grappler::Cluster* cluster,
                const tensorflow::grappler::GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

 protected:
  void DumpGraphs(Graph& graph, int idx, std::string filename_prefix,
                  std::string title);

 private:
  static std::string DotFilename(std::string kind, int idx) {
    return GraphFilenamePrefix(kind, idx) + ".dot";
  }

  static std::string PbtxtFilename(std::string kind, int idx) {
    return GraphFilenamePrefix(kind, idx) + ".pbtxt";
  }

  static std::string GraphFilenamePrefix(std::string kind, int idx) {
    std::stringstream ss;
    ss << kind << "_" << std::setfill('0') << std::setw(4) << idx;
#if defined NGRAPH_DISTRIBUTED
    ngraph::Distributed dist;
    int Rank_ID = dist.get_rank();
    ss << "_" << std::setfill('0') << std::setw(4) << Rank_ID;
#endif
    return ss.str();
  }

  static std::string GraphFilenamePrefix(std::string kind, int idx,
                                         int sub_idx) {
    std::stringstream ss;
    ss << GraphFilenamePrefix(kind, idx) << "_" << std::setfill('0')
       << std::setw(4) << sub_idx;
#if defined NGRAPH_DISTRIBUTED
    ngraph::Distributed dist;
    int Rank_ID = dist.get_rank();
    ss << "_" << std::setfill('0') << std::setw(4) << Rank_ID;
#endif
    return ss.str();
  }

  static int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static bool DumpAllGraphs() {
    return std::getenv("NGRAPH_TF_DUMP_GRAPHS") != nullptr;
  }

  static bool DumpPrecaptureGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_PRE_CAPTURED_GRAPHS") != nullptr;
  }
  static bool DumpCapturedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_CAPTURED_GRAPHS") != nullptr;
  }

  static bool DumpUnmarkedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_UNMARKED_GRAPHS") != nullptr;
  }

  static bool DumpMarkedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_MARKED_GRAPHS") != nullptr;
  }

  static bool DumpClusteredGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_CLUSTERED_GRAPHS") != nullptr;
  }

  static bool DumpDeclusteredGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS") != nullptr;
  }

  static bool DumpEncapsulatedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_ENCAPSULATED_GRAPHS") != nullptr;
  }

  static bool DumpTrackedGraphs() {
    return DumpAllGraphs() ||
           std::getenv("NGRAPH_TF_DUMP_TRACKED_GRAPHS") != nullptr;
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

int NgraphOptimizer::s_serial_counter = 0;
mutex NgraphOptimizer::s_serial_counter_mutex;

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_TF_NGRAPHOPTIMIZER_H_
