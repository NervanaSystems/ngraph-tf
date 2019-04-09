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
#include "tensorflow/core/graph/types.h"

#include "ngraph_rewrite_for_tracking.h"
#include "ngraph_utils.h"
#include "ngraph_replace_op_utilities.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// Main entry point for rewrite-for-tracking.
//
Status RewriteForTracking(Graph* graph, int graph_id) {
  std::vector<Node*> replaced_nodes;

  for (auto node : graph->op_nodes()) {
    if (IsNGVariableType(node->type_string())) {
      NGRAPH_VLOG(1) << "Checking: " << DebugNode(node) << " " << node->name();

      bool just_looking = true;
      bool outputs_ng_supported = true;

      // Check if all the outputs of this node are supported by nGraph
      for (auto edge : node->out_edges()) {
        auto dst = edge->dst();
        NGRAPH_VLOG(1) << "dst node " << DebugNode(dst);
        if (dst->IsOp() && !edge->IsControlEdge() &&
            !IsNGSupportedType(dst->type_string())) {
          NGRAPH_VLOG(1) << "Dst node ngraph doesn't support ";
          outputs_ng_supported = false;
          break;
        }
      }

      // If any of the nodes reading from this Variable node read the data as
      // reference then we dont track it, else we do
      for (auto edge : node->out_edges()) {
        if (edge->dst()->IsOp() && !edge->IsControlEdge() &&
            IsRefType(edge->dst()->input_type(edge->dst_input()))) {
          // if the output reference is read by NGraph supported ops, do not
          // turn off just_looking
          if (!IsNGVariableType(edge->dst()->type_string())) {
            NGRAPH_VLOG(1) << DebugNode(edge->dst())
                           << "needs reference, setting just_looking to false";
            just_looking = false;
            break;
          }
        }
      }

      NGRAPH_VLOG(1) << "Just Looking: " << PrintBool(just_looking);
      NGRAPH_VLOG(1) << "Outputs supported by nGraph: "
                     << PrintBool(outputs_ng_supported);
      NGRAPH_VLOG(1) << "Requires Replacement "
                     << PrintBool(just_looking || !outputs_ng_supported);

      std::string node_new_name = node->name();

      if (just_looking) {
        node_new_name += "/peek";
      }

      if (!outputs_ng_supported) {
        node_new_name += "/non_ng_outputs";
      }

      node_new_name += "/gid_" + to_string(graph_id);
      NGRAPH_VLOG(1) << "Replacing " << node->name() << " New Node name "
                     << node_new_name;

      Node* replacement;
      // TODO(amprocte): Do we need to copy "_" attributes?
      if (node->type_string() == "NGraphVariable") {
        TF_RETURN_IF_ERROR(ReplaceNGraphVariable(
            graph, node, &replacement, node_new_name, just_looking,
            outputs_ng_supported, graph_id));
      } else if (IsNGAssignType(node->type_string())) {
        TF_RETURN_IF_ERROR(ReplaceNGraphAssign(graph, node, &replacement,
                                               node_new_name, just_looking,
                                               outputs_ng_supported, graph_id));
      }

      // Only add incoming control edges. Incoming data edges
      // are already added when building node def
      NGRAPH_VLOG(4) << "Replacing in-edges that are control edges ";
      for (auto edge : node->in_edges()) {
        if (edge->IsControlEdge()) {
          graph->AddEdge(edge->src(), -1, replacement, -1);
          graph->RemoveEdge(edge);
        }
      }

      NGRAPH_VLOG(4) << "Replacing out-edges";
      std::vector<const Edge*> edges;
      for (auto edge : node->out_edges()) {
        edges.push_back(edge);
      }

      for (auto edge : edges) {
        graph->AddEdge(replacement, edge->src_output(), edge->dst(),
                       edge->dst_input());
        graph->RemoveEdge(edge);
      }
      NGRAPH_VLOG(1) << "Replaced " << edges.size() << " of output edges ";

      replaced_nodes.push_back(node);

    }  // end of checking if it is NGVariableType
  }    // end of looping through the nodes in the graph
  for (auto node : replaced_nodes) {
    graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
