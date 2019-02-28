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
    TF_RETURN_IF_ERROR(node->input_node(0, &input_0));
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

  vector<Node*> add_graph_id;

  for (auto node : ordered) {
    if (IsNGVariableType(node->type_string())) {
      string node_key = NGraphCatalog::CreateNodeKey(graph_id, node->name(), 0);
      string shared_name;
      TF_RETURN_IF_ERROR(GetSharedName(node, &shared_name));
      NGraphCatalog::AddCatalog(node_key, shared_name);
      add_graph_id.push_back(node);
      NGRAPH_VLOG(1) << "Adding in Catalog ";
      NGRAPH_VLOG(1) << "Key: " << node_key;
      NGRAPH_VLOG(1) << "Value: " << shared_name;
    } else if (node->type_string() == "NGraphEncapsulate") {
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
      add_graph_id.push_back(node);
    }
  }

  for (auto node : add_graph_id) {
    node->AddAttr("ngraph_graph_id", graph_id);
  }

  NGRAPH_VLOG(1) << "Entered in Catalog";
}

}  // namespace ngraph_bridge

}  // namespace tensorflow