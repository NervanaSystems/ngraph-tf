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
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/types.h"

#include "ngraph_rewrite_for_tracking.h"
#include "ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

// TODO(Mingshsan): Rewrite the Replace* function to a helper function
Status ReplaceNGraphVariable(Graph* graph, Node* node, Node** replacement,
                             std::string node_new_name, bool just_looking,
                             bool outputs_ng_supported, int graph_id) {
  NGRAPH_VLOG(1) << "Replacing NGraphVariable " << node->name();

  TensorShape shape;
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shape", &shape));
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "dtype", &dtype));

  std::string container;
  std::string shared_name;
  // int graph_id;
  std::string backend_name;

  if (GetNodeAttr(node->attrs(), "container", &container) != Status::OK()) {
    container = "";
  }
  if (GetNodeAttr(node->attrs(), "shared_name", &shared_name) != Status::OK()) {
    shared_name = "";
  }

  // TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "ngraph_graph_id",
  // &graph_id));

  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), "_ngraph_backend", &backend_name));

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
          .Attr("_ngraph_backend", backend_name)
          .Device(node->assigned_device_name())
          .Finalize(graph, &(*replacement)));

  (*replacement)->set_assigned_device_name(node->assigned_device_name());

  NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                 << (*replacement)->DebugString();

  return Status::OK();
}

Status ReplaceNGraphAssign(Graph* graph, Node* node, Node** replacement,
                           std::string node_new_name, bool just_looking,
                           bool outputs_ng_supported, int graph_id) {
  NGRAPH_VLOG(1) << "Replacing  " << node->name();
  auto node_type = node->type_string();

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

  std::string backend_name;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), "_ngraph_backend", &backend_name));

  NodeBuilder::NodeOut input_ref;
  NodeBuilder::NodeOut input_val;

  for (auto edge : node->in_edges()) {
    if (edge == NULL) {
      NGRAPH_VLOG(1) << "Replacing " << node_type << ", found null edge: ";
      continue;
    }
    if (edge->dst()->IsOp() && !edge->IsControlEdge() &&
        IsRefType(edge->dst()->input_type(edge->dst_input()))) {
      input_ref = NodeBuilder::NodeOut(edge->src(), edge->src_output());
    } else {
      input_val = NodeBuilder::NodeOut(edge->src(), edge->src_output());
    }
  }
  // if NGraphAssign
  TF_RETURN_IF_ERROR(NodeBuilder(node_new_name, node_type)
                         .Attr("validate_shape", true)
                         .Attr("use_locking", true)
                         .Attr("T", dtype)
                         .Attr("just_looking", just_looking)
                         .Attr("copy_to_tf", !outputs_ng_supported)
                         .Attr("ngraph_graph_id", graph_id)
                         .Attr("_ngraph_backend", backend_name)
                         .Input(input_ref)
                         .Input(input_val)
                         .Device(node->assigned_device_name())
                         .Finalize(graph, &(*replacement)));

  (*replacement)->set_assigned_device_name(node->assigned_device_name());
  NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                 << (*replacement)->DebugString();
  return Status::OK();
}

// ReplaceNGraphApplyGradientDescent
Status ReplaceNGraphApplyGradientDescent(
    Graph* graph, Node* node, Node** replacement, std::string node_new_name,
    bool just_looking, bool outputs_ng_supported, int graph_id) {
  NGRAPH_VLOG(1) << "Start replacing NGraphApplyGradientDescent "
                 << node->name();

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));
  bool use_locking;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "use_locking", &use_locking));
  // int graph_id;
  // TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "ngraph_graph_id",
  // &graph_id));
  std::string backend_name;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), "_ngraph_backend", &backend_name));

  NodeBuilder::NodeOut input_var;
  NodeBuilder::NodeOut input_alpha;
  NodeBuilder::NodeOut input_delta;

  // TODO(Mingshan): we may removing the control_edges to the
  // ApplyGradientDescent node
  std::vector<const Edge*> input_edges;
  TF_RETURN_IF_ERROR(node->input_edges(&input_edges));

  NGRAPH_VLOG(1) << "No of input edges to ApplyGradientDescent "
                 << input_edges.size();

  input_var =
      NodeBuilder::NodeOut(input_edges[0]->src(), input_edges[0]->src_output());
  input_alpha =
      NodeBuilder::NodeOut(input_edges[1]->src(), input_edges[1]->src_output());
  input_delta =
      NodeBuilder::NodeOut(input_edges[2]->src(), input_edges[2]->src_output());

  TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "NGraphApplyGradientDescent")
                         .Attr("T", dtype)
                         .Attr("use_locking", use_locking)
                         .Attr("just_looking", just_looking)
                         .Attr("copy_to_tf", !outputs_ng_supported)
                         .Attr("ngraph_graph_id", graph_id)
                         .Attr("_ngraph_backend", backend_name)
                         .Input(input_var)
                         .Input(input_alpha)
                         .Input(input_delta)
                         .Device(node->assigned_device_name())
                         .Finalize(graph, &(*replacement)));

  (*replacement)->set_assigned_device_name(node->assigned_device_name());
  return Status::OK();
}  // end of ReplaceNGraphApplyGradientDescent

//
// Main entry point for rewrite-for-tracking.
//
Status RewriteForTracking(Graph* graph, int graph_id) {
  std::vector<Node*> replaced_nodes;
  std::set<string> ng_supported_ops = {
      "NGraphVariable",    "NGraphAssign",
      "NGraphEncapsulate", "NGraphApplyGradientDescent",
      "NGraphAssignSub",   "NGraphAssignAdd"};

  for (auto node : graph->op_nodes()) {
    if (IsNGVariableType(node->type_string())) {
      NGRAPH_VLOG(1) << "Checking: " << DebugNode(node) << " " << node->name();

      bool just_looking = true;
      bool outputs_ng_supported = true;

      // Check if all the outputs of this node ngraph supports
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
          // NGVariableType = NGVariable || NGraphAssign ||
          // NGraphApplyGradientDescent
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
      // TODO(mingshan): Combine this three to one helper function
      if (node->type_string() == "NGraphVariable") {
        ReplaceNGraphVariable(graph, node, &replacement, node_new_name,
                              just_looking, outputs_ng_supported, graph_id);
      } else if (IsNGAssignType(node->type_string())) {
        ReplaceNGraphAssign(graph, node, &replacement, node_new_name,
                            just_looking, outputs_ng_supported, graph_id);
      }

      // Only add incoming control edges. Incoming data edges
      // are already added when building node def
      NGRAPH_VLOG(4) << "Replacing in-edges that are control edges ";
      for (auto edge : node->in_edges()) {
        if (edge == NULL) continue;
        if (edge->IsControlEdge()) {
          NGRAPH_VLOG(4) << "Added edge: " << edge->DebugString();
          NGRAPH_VLOG(4) << "SRC " << edge->src() << " DST " << replacement;
          graph->AddEdge(edge->src(), -1, replacement, -1);
          NGRAPH_VLOG(4) << "Removing edge: " << edge->DebugString();
          graph->RemoveEdge(edge);
          // NGRAPH_VLOG(4) << "Removed " << edge->DebugString();
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
