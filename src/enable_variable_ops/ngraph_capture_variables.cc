/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

#include "ngraph_api.h"
#include "ngraph_capture_variables.h"
#include "ngraph_replace_op_utilities.h"
#include "ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// Utility function to check if placement on the NGRAPH device has been
// requested.
//
// FIXME(amprocte): stubbed out for now because NGRAPH device is gone.
//
static bool NGraphPlacementRequested(const Node* node) { return true; }

//
// Main entry point for the variable-capture.
//
Status CaptureVariables(Graph* graph, std::vector<string> skip_these_nodes) {
  const static std::map<
      const string,
      const pair<string,
                 function<Status(
                     Graph * graph, Node * node, Node * *replacement,
                     const string replacement_node_name,
                     const string replacement_op_type, const bool just_looking,
                     const bool outputs_ng_supported, const int graph_id,
                     const bool is_backend_set)>>>
      CAPTURE_REPLACE_OP_MAP{
          {"ApplyGradientDescent", std::make_pair("NGraphApplyGradientDescent",
                                                  ReplaceApplyGradientDescent)},
          {"Assign", std::make_pair("NGraphAssign", ReplaceAssign)},
          {"AssignAdd", std::make_pair("NGraphAssignAdd", ReplaceAssign)},
          {"AssignSub", std::make_pair("NGraphAssignSub", ReplaceAssign)},
          {"VariableV2", std::make_pair("NGraphVariable", ReplaceVariable)}};

  std::vector<Node*> replaced_nodes;
  for (auto node : graph->op_nodes()) {
    if (NGraphPlacementRequested(node)) {
      auto itr = CAPTURE_REPLACE_OP_MAP.find(node->type_string());
      if (itr != CAPTURE_REPLACE_OP_MAP.end()) {
        NGRAPH_VLOG(1) << "Capturing: " << node->name();
        Node* replacement;

        // Create the replacement node
        TF_RETURN_IF_ERROR((itr->second.second)(graph, node, &replacement,
                                                node->name(), itr->second.first,
                                                false, false, 0, false));

        std::vector<const Edge*> edges;

        NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                       << replacement->DebugString();

        // Though edges will be removed when we remove the node
        // we specifically remove the edges to be sure
        // We only add incoming control edges. Input data edges are
        // added via node def
        for (auto edge : node->in_edges()) {
          NGRAPH_VLOG(4) << "Replacing: " << edge->DebugString();
          if (!edge->IsControlEdge()) continue;
          graph->AddEdge(edge->src(), edge->src_output(), replacement,
                         edge->dst_input());
          graph->RemoveEdge(edge);
        }

        for (auto edge : node->out_edges()) {
          edges.push_back(edge);
        }

        for (auto edge : edges) {
          NGRAPH_VLOG(4) << "Replacing: " << edge->DebugString();
          graph->AddEdge(replacement, edge->src_output(), edge->dst(),
                         edge->dst_input());
          graph->RemoveEdge(edge);
        }

        replaced_nodes.push_back(node);
      }

    }  // end of checking NGraphPlacementRequested
  }    // end of looping through nodes in the graph

  for (auto node : replaced_nodes) {
    NGRAPH_VLOG(4) << "Removing: " << node->name();
    graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
