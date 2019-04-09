/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include "ngraph_utils.h"
using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status ReplaceApplyGradientDescent(Graph* graph, Node* node, Node** replacement,
                                   const string node_new_name,
                                   const bool just_looking,
                                   const bool outputs_ng_supported,
                                   const int graph_id) {
  NGRAPH_VLOG(1) << "Start replacing NGraphApplyGradientDescent "
                 << node->name();

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));
  bool use_locking;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "use_locking", &use_locking));

  std::string backend_name;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), "_ngraph_backend", &backend_name));

  NodeBuilder::NodeOut input_var;
  NodeBuilder::NodeOut input_alpha;
  NodeBuilder::NodeOut input_delta;

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
}  // end of ReplaceApplyGradientDescent

Status ReplaceAssign(Graph* graph, Node* node, Node** replacement,
                     const string replacement_node_name,
                     const string replacement_node_type,
                     const bool just_looking, const bool outputs_ng_supported,
                     const int graph_id) {
  NGRAPH_VLOG(1) << "Replacing  " << node->name();

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

  std::string backend_name;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), "_ngraph_backend", &backend_name));

  NodeBuilder::NodeOut input_ref;
  NodeBuilder::NodeOut input_val;

  for (auto edge : node->in_edges()) {
    if (edge == NULL) {
      NGRAPH_VLOG(1) << "Replacing " << replacement_node_type
                     << ", found null edge: ";
      continue;
    }
    if (edge->dst()->IsOp() && !edge->IsControlEdge() &&
        IsRefType(edge->dst()->input_type(edge->dst_input()))) {
      input_ref = NodeBuilder::NodeOut(edge->src(), edge->src_output());
    } else {
      input_val = NodeBuilder::NodeOut(edge->src(), edge->src_output());
    }
  }

  TF_RETURN_IF_ERROR(NodeBuilder(replacement_node_name, replacement_node_type)
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

Status ReplaceVariable(Graph* graph, Node* node, Node** replacement,
                       const string replacement_node_name,
                       const string replacement_node_type,
                       const bool just_looking, const bool outputs_ng_supported,
                       const int graph_id) {
  NGRAPH_VLOG(1) << "Replacing NGraphVariable " << node->name();

  TensorShape shape;
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shape", &shape));
  TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "dtype", &dtype));

  std::string container;
  std::string shared_name;
  std::string backend_name;

  if (GetNodeAttr(node->attrs(), "container", &container) != Status::OK()) {
    container = "";
  }
  if (GetNodeAttr(node->attrs(), "shared_name", &shared_name) != Status::OK()) {
    shared_name = "";
  }

  TF_RETURN_IF_ERROR(
      GetNodeAttr(node->attrs(), "_ngraph_backend", &backend_name));

  TF_RETURN_IF_ERROR(
      NodeBuilder(replacement_node_name, replacement_node_type)
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

}  // namespace ngraph_bridge

}  // namespace tensorflow
