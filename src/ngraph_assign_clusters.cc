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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/device_name_utils.h"

#include "ngraph_assign_clusters.h"
#include "ngraph_cluster_manager.h"
#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_utils.h"
#include "tf_graphcycles.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

namespace {
  struct Cluster {
    int index;
    std::set<tensorflow::Node*> nodes;
  };
}

Status AssignClusters(Graph* graph) {
  std::map<Node*, std::shared_ptr<Cluster>> cluster_map;

  GraphCycles gc;

  for (auto node : graph->op_nodes()) {
    int new_index = gc.NewNode();
    NGRAPH_VLOG(5) << "Creating cycle graph node: " << new_index << " for "
                   << node->name() << "[" << node->type_string() << "]";
    cluster_map[node] = std::make_shared<Cluster>();
    cluster_map[node]->index = new_index;
    cluster_map[node]->nodes.insert(node);
  }

  for (auto edge : graph->edges()) {
    Node* src = edge->src();
    Node* dst = edge->dst();

    // Skip source/sink
    if (!src->IsOp() || !dst->IsOp()) {
      continue;
    }

    // Skip NextIteration
    if (src->IsNextIteration() || dst->IsNextIteration()) {
      continue;
    }

    if (!gc.InsertEdge(cluster_map[src]->index, cluster_map[dst]->index)) {
      NGRAPH_VLOG(5) << "Failing due to cycle";
      return errors::Unimplemented(
          "Input graph has a cycle (inserting an edge from ",
          src->DebugString(), " to ", dst->DebugString(),
          " would create a cycle)");
    }
  }

  NGRAPH_VLOG(2) << "Starting contraction";
  bool changed;

  do {
    changed = false;

    for (auto edge : graph->edges()) {
      Node* src = edge->src();
      Node* dst = edge->dst();

      if (!src->IsOp() || !dst->IsOp()) {
        continue;
      }

      if (!NodeIsMarkedForClustering(src) || !NodeIsMarkedForClustering(dst)) {
        NGRAPH_VLOG(5) << "Skipping: " << src->name() << " -> " << dst->name();
        continue;
      }

      int src_index = cluster_map[src]->index;
      int dst_index = cluster_map[dst]->index;

      if (gc.HasEdge(src_index, dst_index) &&
          gc.ContractEdge(src_index, dst_index)) {
        // using cluster_map[dst]->nodes in the loop directly appears to
        // invalidate the iterator when `node` == `dst`
        // this happens with clang but not gcc
        auto cluster_dst = cluster_map[dst];
        for (auto node : cluster_dst->nodes) {
          cluster_map[src]->nodes.insert(node);
          cluster_map[node] = cluster_map[src];
        }
        changed = true;
      }
    }
  } while (changed);
  NGRAPH_VLOG(2) << "Contraction done";

  NGRAPH_VLOG(2) << "Starting tagging";
  std::set<Cluster*> seen;

  for (auto kv : cluster_map) {
    auto cluster = kv.second.get();
    bool has_ngraph_ops = false;
    bool has_non_ngraph_ops = false;

    for (auto node : cluster->nodes) {
      if (NodeIsMarkedForClustering(node)) {
        has_ngraph_ops = true;
      }
      else {
        has_non_ngraph_ops = true;
      }
    }

    if (has_ngraph_ops && has_non_ngraph_ops) {
      NGRAPH_VLOG(2) << "Cluster " << cluster->index << " has both nGraph and non-nGraph nodes";
      for (auto node : cluster->nodes) {
        NGRAPH_VLOG(2) << (NodeIsMarkedForClustering(node) ? "nGraph node: " : "non-nGraph node: ") << node->name() << " [" << node->type_string() << "]";
      }
      return errors::Internal("Cluster ", cluster->index, " has both nGraph and non-nGraph nodes");
    }

    if (!has_ngraph_ops) {
      continue;
    }

    if (seen.count(cluster) == 0) {
      int cluster_idx = NGraphClusterManager::NewCluster();

      for (auto node : cluster->nodes) {
        NGRAPH_VLOG(2) << ">> cluster " << cluster_idx << ": " << node
                       << " :: " << node->name() << " [" << node->type_string()
                       << "]";

        if (!NodeIsMarkedForClustering(node)) {
          return errors::Internal(
              "Node ", node->DebugString(),
              " was not marked for clustering but was placed in an nGraph cluster.");
        }

        node->AddAttr("_ngraph_cluster", cluster_idx);
      }
    }
  }
  NGRAPH_VLOG(2) << "Tagging done";

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
