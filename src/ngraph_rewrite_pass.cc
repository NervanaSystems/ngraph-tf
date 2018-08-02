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
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph_assign_clusters.h"
#include "ngraph_deassign_clusters.h"
#include "ngraph_encapsulate_clusters.h"
#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"

#include "tf_graph_writer.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

class NGraphRewritePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) {
    // If we don't get a main graph, log that fact and bail.
    if (options.graph == nullptr) {
      NGRAPH_VLOG(0) << "NGraphRewritePass: options.graph == nullptr";
      return Status::OK();
    }

    // For filename generation purposes, grab a fresh index.
    int idx = FreshIndex();
    if(DumpOriginalGraphs()) {
      DumpGraphs(options, idx, "original", "Original Graph");
    }

    // mark for clustering
    TF_RETURN_IF_ERROR(MarkForClustering(options.graph->get()));
    if(DumpMarkedGraphs()) {
      DumpGraphs(options, idx, "marked", "Graph Marked for Clustering");
    }

    // assign clusters
    TF_RETURN_IF_ERROR(AssignClusters(options.graph->get()));
    if(DumpClusteredGraphs()) {
      DumpGraphs(options, idx, "clustered", "Graph with Clusters Assigned");
    }

    // de-assign clusters
    TF_RETURN_IF_ERROR(DeassignClusters(options.graph->get()));
    if(DumpDeclusteredGraphs()) {
      DumpGraphs(options, idx, "declustered", "Graph with Trivial Clusters De-Assigned");
    }

    // encapsulate
    TF_RETURN_IF_ERROR(EncapsulateClusters(options.graph->get()));
    if(DumpEncapsulatedGraphs()) {
      DumpGraphs(options, idx, "encapsulated", "Graph with Clusters Encapsulated");
    }

    return Status::OK();
  }
 private:
  void DumpGraphs(const GraphOptimizationPassOptions& options, int idx, std::string filename_prefix, std::string title) {
    // If we have a "main" graph, dump that.
    if(options.graph != nullptr) {
      auto dot_filename = DotFilename(filename_prefix,idx);
      auto pbtxt_filename = PbtxtFilename(filename_prefix,idx);
      NGRAPH_VLOG(0) << "Dumping main graph to " << dot_filename;
      NGRAPH_VLOG(0) << "Dumping main graph to " << pbtxt_filename;

      GraphToDotFile(options.graph->get(), dot_filename, title);
      GraphToPbTextFile(options.graph->get(), pbtxt_filename);
    }

    // If we have partition graphs (we shouldn't), dump those.
    if (options.partition_graphs != nullptr) {
      int sub_idx = 0;

      for (auto& kv : *options.partition_graphs) {
        auto dot_filename = DotFilename(filename_prefix,idx,sub_idx);
        auto pbtxt_filename = PbtxtFilename(filename_prefix,idx,sub_idx);
        NGRAPH_VLOG(0) << "Dumping subgraph " << sub_idx << " to " << dot_filename;
        NGRAPH_VLOG(0) << "Dumping subgraph " << sub_idx << " to " << pbtxt_filename;

        Graph* pg = kv.second.get();

        GraphToDotFile(pg, dot_filename, title);
        GraphToPbTextFile(pg, pbtxt_filename);

        sub_idx++;
      }
    }
  }

  int FreshIndex() {
    mutex_lock l(s_serial_counter_mutex);
    return s_serial_counter++;
  }

  static bool DumpAllGraphs() { return std::getenv("NGRAPH_TF_DUMP_GRAPHS") != nullptr; }
  static bool DumpOriginalGraphs() { return DumpAllGraphs() || std::getenv("NGRAPH_TF_DUMP_ORIGINAL_GRAPHS") != nullptr; }
  static bool DumpMarkedGraphs() { return DumpAllGraphs() || std::getenv("NGRAPH_TF_DUMP_MARKED_GRAPHS") != nullptr; }
  static bool DumpClusteredGraphs() { return DumpAllGraphs() || std::getenv("NGRAPH_TF_DUMP_CLUSTERED_GRAPHS") != nullptr; }
  static bool DumpDeclusteredGraphs() { return DumpAllGraphs() || std::getenv("NGRAPH_TF_DUMP_DECLSUTERED_GRAPHS") != nullptr; }
  static bool DumpEncapsulatedGraphs() { return DumpAllGraphs() || std::getenv("NGRAPH_TF_DUMP_ENCAPSULATED_GRAPHS") != nullptr; }

  static std::string DotFilename(std::string kind, int idx) {
    return GraphFilenamePrefix(kind,idx) + ".dot";
  }
  static std::string PbtxtFilename(std::string kind, int idx) {
    return GraphFilenamePrefix(kind,idx) + ".pbtxt";
  }
  static std::string DotFilename(std::string kind, int idx, int sub_idx) {
    return GraphFilenamePrefix(kind,idx,sub_idx) + ".dot";
  }
  static std::string PbtxtFilename(std::string kind, int idx, int sub_idx) {
    return GraphFilenamePrefix(kind,idx,sub_idx) + ".pbtxt";
  }
  static std::string GraphFilenamePrefix(std::string kind, int idx) {
    std::stringstream ss;
    ss << kind << "_" << idx;
    return ss.str();
  }
  static std::string GraphFilenamePrefix(std::string kind, int idx, int sub_idx) {
    std::stringstream ss;
    ss << GraphFilenamePrefix(kind,idx) << "_" << sub_idx;
    return ss.str();
  }

  static int s_serial_counter GUARDED_BY(s_serial_counter_mutex);
  static mutex s_serial_counter_mutex;
};

int NGraphRewritePass::s_serial_counter = 0;
mutex NGraphRewritePass::s_serial_counter_mutex;

}  // namespace ngraph_bridge

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      ngraph_bridge::NGraphRewritePass);
}  // namespace tensorflow
