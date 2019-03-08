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
#include "ngraph_rewrite_for_tracking.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/types.h"

#include "ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status ReplaceNGraphVariable(Graph* graph, Node* node, Node** replacement,
                             std::string node_new_name, bool just_looking,
                             bool outputs_ng_supported) {
  NGRAPH_VLOG(1) << "Replacing NGraphVariable " << node->name();

  TensorShape shape;
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shape", &shape));
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "dtype", &dtype));

  std::string container;
  std::string shared_name;
  int graph_id;
  if (GetNodeAttr(node->attrs(), "container", &container) != Status::OK()) {
    container = "";
  }
  if (GetNodeAttr(node->attrs(), "shared_name", &shared_name) != Status::OK()) {
    shared_name = "";
  }
  if (GetNodeAttr(node->attrs(), "ngraph_graph_id", &graph_id) !=
      Status::OK()) {
    graph_id = 0;
  }

  // NGRAPHVARIABLE
  TF_RETURN_IF_ERROR(
      NodeBuilder(node_new_name, node->type_string())
          .Attr("shape", shape)
          .Attr("dtype", dtype)
          .Attr("container", container)
          .Attr("shared_name",
                (shared_name.empty() ? node->name() : shared_name))
          .Attr("just_looking", just_looking)
          .Attr("copy_to_tf", !outputs_ng_supported)
          .Attr("ngraph_graph_id", graph_id)
          .Device(node->assigned_device_name())
          .Finalize(graph, &(*replacement)));

  (*replacement)->set_assigned_device_name(node->assigned_device_name());

  // Add edge from the input nodes (to the variable node (NGraphVariable))
  // to the new replacement node (also of type NGraphVariable)
  NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                 << (*replacement)->DebugString();

  // Though edges will be removed when we remove the node
  // we specifically remove the edges to be sure
  for (auto edge : node->in_edges()) {
    NGRAPH_VLOG(4) << "Replacing: " << edge->DebugString();
    graph->AddEdge(edge->src(), edge->src_output(), (*replacement),
                   edge->dst_input());
    graph->RemoveEdge(edge);
  }

  return Status::OK();
}

Status ReplaceNGraphAssign(Graph* graph, Node* node, Node** replacement,
                           std::string node_new_name, bool just_looking,
                           bool outputs_ng_supported) {
  NGRAPH_VLOG(1) << "Replacing NGraphAssign " << node->name();

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

  int graph_id;
  if (GetNodeAttr(node->attrs(), "ngraph_graph_id", &graph_id) !=
      Status::OK()) {
    graph_id = 0;
  }

  NodeBuilder::NodeOut input_ref;
  NodeBuilder::NodeOut input_val;

  for (auto edge : node->in_edges()) {
    if (edge == NULL) {
      NGRAPH_VLOG(1) << "Replacing NGraphAssign, found null edge: ";
      continue;
    }

    // Check REF TYPE RATHER THAN NAME
    if (edge->dst()->IsOp() && !edge->IsControlEdge() &&
        IsRefType(edge->dst()->input_type(edge->dst_input()))) {
      input_ref = NodeBuilder::NodeOut(edge->src(), edge->src_output());
    } else {
      input_val = NodeBuilder::NodeOut(edge->src(), edge->src_output());
    }
  }
  // if NGraphAssign
  TF_RETURN_IF_ERROR(NodeBuilder(node_new_name, "NGraphAssign")
                         .Attr("validate_shape", true)
                         .Attr("use_locking", true)
                         .Attr("T", dtype)
                         .Attr("just_looking", just_looking)
                         .Attr("copy_to_tf", !outputs_ng_supported)
                         .Attr("ngraph_graph_id", graph_id)
                         .Input(input_ref)
                         .Input(input_val)
                         .Device(node->assigned_device_name())
                         .Finalize(graph, &(*replacement)));

  (*replacement)->set_assigned_device_name(node->assigned_device_name());
  return Status::OK();
}

//
// Main entry point for rewrite-for-tracking.
//
Status RewriteForTracking(Graph* graph) {
  std::vector<Node*> replaced_nodes;
  std::set<string> ng_supported_ops = {"NGraphVariable", "NGraphAssign",
                                       "NGraphEncapsulate"};

  for (auto node : graph->op_nodes()) {
    if (node->type_string() == "NGraphVariable" ||
        node->type_string() == "NGraphAssign") {
      NGRAPH_VLOG(1) << "Checking: " << DebugNode(node) << " " << node->name();

      bool just_looking = true;
      bool outputs_ng_supported = true;

      for (auto edge : node->out_edges()) {
        auto dst = edge->dst();
        NGRAPH_VLOG(1) << "dst node " << DebugNode(dst);
        if (dst->IsOp() && !edge->IsControlEdge() &&
            (ng_supported_ops.find(dst->type_string()) ==
             ng_supported_ops.end())) {
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
          if (!IsNGVariableType(edge->dst()->type_string())) {
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

      if (just_looking || !outputs_ng_supported) {
        Node* replacement;

        std::string node_new_name = node->name();
        if (just_looking) {
          node_new_name += "/peek";
        }

        if (!outputs_ng_supported) {
          node_new_name += "/non_ng_outputs";
        }

        NGRAPH_VLOG(1) << "Replacing " << node->name() << " New Node name "
                       << node_new_name;

        // TODO(amprocte): Do we need to copy "_" attributes?
        if (node->type_string() == "NGraphVariable") {
          ReplaceNGraphVariable(graph, node, &replacement, node_new_name,
                                just_looking, outputs_ng_supported);
        } else if (node->type_string() == "NGraphAssign") {
          ReplaceNGraphAssign(graph, node, &replacement, node_new_name,
                              just_looking, outputs_ng_supported);
        }

        std::vector<const Edge*> edges;
        for (auto edge : node->out_edges()) {
          edges.push_back(edge);
        }

        for (auto edge : edges) {
          graph->AddEdge(replacement, edge->src_output(), edge->dst(),
                         edge->dst_input());
          graph->RemoveEdge(edge);
        }

        replaced_nodes.push_back(node);
      } else {
        NGRAPH_VLOG(1)
            << "No replacement (not just looking and all outputs ng support): "
            << node->name();
      }
    }
  }
  for (auto node : replaced_nodes) {
    graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
