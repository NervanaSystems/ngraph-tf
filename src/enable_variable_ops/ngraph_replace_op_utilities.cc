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
  std::string backend_name;

  if (GetNodeAttr(node->attrs(), "container", &container) != Status::OK()) {
    container = "";
  }
  if (GetNodeAttr(node->attrs(), "shared_name", &shared_name) != Status::OK()) {
    shared_name = "";
  }

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





}  // namespace ngraph_bridge

}  // namespace tensorflow
