/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph_catalog.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// Used by only NGraphVar or NGraphAssign
Status GetSharedName(Node* node, string* shared_name) {
  if (node->type_string() == "NGraphVariable") {
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shared_name", shared_name));
    if (shared_name->empty()) {
      (*shared_name) = node->name();
    }
    return Status::OK();
  }

  auto temp = node;
  while (temp->type_string() != "NGraphVariable") {
    Node* input_0;
    TF_RETURN_IF_ERROR(temp->input_node(0, &input_0));
    temp = input_0;
  }
  GetSharedName(temp, shared_name);

  return Status::OK();
}

// 1. Populate the input_variable_map
// 2. Attach Graph Ids to the node

Status EnterInCatalog(Graph* graph, int graph_id) {
  // Topological Sort
  vector<Node*> ordered;
  GetReversePostOrder(*graph, &ordered);

  for (auto node : ordered) {
    // Update the input variable map
    if (IsNGVariableType(node->type_string())) {
      string node_key = NGraphCatalog::CreateNodeKey(graph_id, node->name(), 0);
      string shared_name;
      TF_RETURN_IF_ERROR(GetSharedName(node, &shared_name));
      NGraphCatalog::AddCatalog(node_key, shared_name);
      // add_graph_id.push_back(node);
      NGRAPH_VLOG(1) << "Adding in Catalog ";
      NGRAPH_VLOG(1) << "Key: " << node_key;
      NGRAPH_VLOG(1) << "Value: " << shared_name;

    } else if (node->type_string() == "NGraphEncapsulate") {
      // input catalog
      for (auto edge : node->in_edges()) {
        if (edge->src()->IsOp() && !edge->IsControlEdge() &&
            IsNGVariableType(edge->src()->type_string())) {
          auto src = edge->src();
          string node_key = NGraphCatalog::CreateNodeKey(graph_id, node->name(),
                                                         edge->dst_input());
          string shared_name;
          TF_RETURN_IF_ERROR(GetSharedName(src, &shared_name));
          NGraphCatalog::AddCatalog(node_key, shared_name);
          NGRAPH_VLOG(1) << "Adding in Catalog ";
          NGRAPH_VLOG(1) << "Key: " << node_key;
          NGRAPH_VLOG(1) << "Value: " << shared_name;
        }
      }

      // output ng-copy map catalog
      unordered_set<int> op_index_to_copy;
      NGRAPH_VLOG(1) << "Finding Output Copy required for " << node->name();
      for (auto edge : node->out_edges()) {
        if (edge->dst()->IsOp() && !edge->IsControlEdge() &&
            !IsNGVariableType(edge->dst()->type_string())) {
          NGRAPH_VLOG(1) << "Output Copy required for " << node->name()
                         << " ,index: " << edge->src_output() << " dstOpType "
                         << edge->dst()->type_string();
          op_index_to_copy.insert(edge->src_output());
        }
      }
      NGraphCatalog::AddEncapCopyOutputCatalog(node->name(), op_index_to_copy);

      // add_graph_id.push_back(node);
    }  // end of node is type NGraphEncapsulate

    // Update the output tensor map

    if (IsNGVariableType(node->type_string())) {
      for (auto edge : node->in_edges()) {
        if (!edge->src()->IsOp() || edge->IsControlEdge() ||
            IsRefType(edge->dst()->input_type(edge->dst_input())) ||
            edge->src()->type_string() != "NGraphEncapsulate") {
          continue;
        }

        NGRAPH_VLOG(1) << "Get " << node->type_string()
                       << "and input is from NGraphEncapsulate";

        auto src = edge->src();
        int src_output = edge->src_output();
        string node_key;
        if (src_output == 0) {
          node_key = to_string(graph_id) + "_" + src->name();
        } else {
          node_key =
              NGraphCatalog::CreateNodeKey(graph_id, src->name(), src_output);
        }
        // Will be updated with real tensors in Encapsulate
        NGraphCatalog::AddOutputCatalog(node_key, nullptr);
        NGRAPH_VLOG(1) << "Adding in output Catalog ";
        NGRAPH_VLOG(1) << "Key: " << node_key;
      }
    }  // end of if node of type NGraphAssign
  }    // enter in catalog

  // for (auto node : add_graph_id) {
  //   node->AddAttr("ngraph_graph_id", graph_id);
  // }

  NGRAPH_VLOG(1) << "Entered in Catalog";
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
