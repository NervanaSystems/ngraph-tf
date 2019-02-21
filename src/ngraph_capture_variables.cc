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

#include "ngraph_api.h"
#include "ngraph_capture_variables.h"
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
Status CaptureVariables(Graph* graph) {
  if (config::IsEnabled() == false) {
    return Status::OK();
  }

  std::vector<Node*> replaced_nodes;

  for (auto node : graph->op_nodes()) {
    if (NGraphPlacementRequested(node)) {
      if (node->type_string() == "VariableV2") {
        NGRAPH_VLOG(1) << "Capturing: " << node->name();

        TensorShape shape;
        DataType dtype;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "shape", &shape));
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "dtype", &dtype));

        std::string container;
        std::string shared_name;
        if (GetNodeAttr(node->attrs(), "container", &container) !=
            Status::OK()) {
          container = "";
        }
        if (GetNodeAttr(node->attrs(), "shared_name", &shared_name) !=
            Status::OK()) {
          shared_name = "";
        }

        Node* replacement;

        // TODO(amprocte): Do we need to copy "_" attributes?
        TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "NGraphVariable")
                               .Attr("shape", shape)
                               .Attr("dtype", dtype)
                               .Attr("container", container)
                               .Attr("shared_name", shared_name)
                               .Device(node->assigned_device_name())
                               .Finalize(graph, &replacement));

        replacement->set_assigned_device_name(node->assigned_device_name());

        std::vector<const Edge*> edges;

        // Add edge from the input nodes (to the variable node (VariableV2))
        // to the replacement node (NGraphVariable)
        NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                       << replacement->DebugString();

        // Though edges will be removed when we remove the node
        // we specifically remove the edges to be sure
        for (auto edge : node->in_edges()) {
          NGRAPH_VLOG(4) << "Replacing: " << edge->DebugString();
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

      else if (node->type_string() == "Assign") {
        NGRAPH_VLOG(1) << "Capturing: " << node->name();

        DataType dtype;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));
        Node* replacement;
        NGRAPH_VLOG(1) << "Got dtype" << DataTypeString(dtype);

        NodeBuilder::NodeOut input_ref;
        NodeBuilder::NodeOut input_val;

        for (auto edge : node->in_edges()) {
          if (edge == NULL) {
            NGRAPH_VLOG(1) << "Found null edge: ";
            continue;
          }
          NGRAPH_VLOG(1) << "Found in edge: " << edge->src()->name();
          NGRAPH_VLOG(1) << "Found in edge: " << edge->src()->type_string();
          NGRAPH_VLOG(1) << "Found in edge: " << edge->DebugString();
          // Check REF TYPE RATHER THAN NAME
          if (edge->src()->type_string() == "NGraphVariable" ||
              edge->src()->type_string() == "VariableV2") {
            input_ref = NodeBuilder::NodeOut(edge->src(), edge->src_output());
          } else {
            input_val = NodeBuilder::NodeOut(edge->src(), edge->src_output());
          }
        }
        NGRAPH_VLOG(1) << "Found inputs";
        // TODO(amprocte): Do we need to copy "_" attributes?
        TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "NGraphAssign")
                               .Attr("validate_shape", true)
                               .Attr("use_locking", true)
                               .Attr("T", dtype)
                               .Input(input_ref)
                               .Input(input_val)
                               .Device(node->assigned_device_name())
                               .Finalize(graph, &replacement));
        NGRAPH_VLOG(1) << "Constructed Node Def";

        replacement->set_assigned_device_name(node->assigned_device_name());

        std::vector<const Edge*> edges;

        // Add edge from the input nodes (to the variable node (VariableV2))
        // to the replacement node (NGraphVariable)
        NGRAPH_VLOG(4) << "Replacing Node " << node->DebugString() << " with "
                       << replacement->DebugString();

        // Though edges will be removed when we remove the node
        // we specifically remove the edges to be sure
        for (auto edge : node->in_edges()) {
          NGRAPH_VLOG(4) << "Replacing: " << edge->DebugString();
          graph->AddEdge(edge->src(), edge->src_output(), replacement,
                         edge->dst_input());

          NGRAPH_VLOG(4) << "Added Edge: ";
          graph->RemoveEdge(edge);
          NGRAPH_VLOG(4) << "Removed Edge: ";
        }

        NGRAPH_VLOG(4) << "Getting out edges: ";
        for (auto edge : node->out_edges()) {
          edges.push_back(edge);
        }
        NGRAPH_VLOG(4) << "Got out edges: ";

        for (auto edge : edges) {
          NGRAPH_VLOG(4) << "Replacing: " << edge->DebugString();
          graph->AddEdge(replacement, edge->src_output(), edge->dst(),
                         edge->dst_input());
          graph->RemoveEdge(edge);
        }

        replaced_nodes.push_back(node);
        NGRAPH_VLOG(1) << "Replaced";
      }
    }
  }

  for (auto node : replaced_nodes) {
    NGRAPH_VLOG(4) << "Removing: " << node->name();
    graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
