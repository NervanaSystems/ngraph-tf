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
#include "ngraph_replace_optimizers.h"
#include "ngraph_api.h"
#include "ngraph_capture_variables.h"
#include "ngraph_utils.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

using namespace std;
namespace ng = ngraph;
namespace tensorflow {

namespace ngraph_bridge {

Status ReplaceOptimizers(Graph* graph, int graph_id) {
  // Go over the nodes and replace variable modifiers with the computation graph
  // Add Assign Op in their place

  for (auto node : graph->op_nodes()) {
    if (node->type_string() == "NGraphAssignSub") {
      NodeBuilder::NodeOut input_ref;
      NodeBuilder::NodeOut input_val;

      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

      for (auto edge : node->in_edges()) {
        if (edge == NULL) {
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
      NGRAPH_VLOG(1) << "Has assigned device " << node->assigned_device_name();

      Node* sub_op;
      string new_name_sub = node->name() + "_Sub";
      TF_RETURN_IF_ERROR(NodeBuilder(new_name_sub, "Sub")
                             .Input(input_ref)
                             .Input(input_val)
                             .Attr("T", dtype)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &(sub_op)));
      sub_op->set_assigned_device_name(node->assigned_device_name());
      NodeBuilder::NodeOut ndef_sub_op = NodeBuilder::NodeOut(sub_op, 0);

      NGRAPH_VLOG(1) << "Sub op name: " << sub_op->name();
      NGRAPH_VLOG(1) << "Sub op assigned device: "
                     << sub_op->assigned_device_name();

      Node* ngraphassign_op;
      string new_name_ngassign = node->name() + "_NGraphAssign";

      TF_RETURN_IF_ERROR(NodeBuilder(new_name_ngassign, "NGraphAssign")
                             .Attr("validate_shape", true)
                             .Attr("use_locking", true)
                             .Attr("T", dtype)
                             .Attr("ngraph_graph_id", 0)
                             .Input(input_ref)
                             .Input(ndef_sub_op)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &ngraphassign_op));
      ngraphassign_op->set_assigned_device_name(node->assigned_device_name());
      NGRAPH_VLOG(1) << "Assign op name: " << ngraphassign_op->name();
      NGRAPH_VLOG(1) << "Assign op assigned device: "
                     << ngraphassign_op->assigned_device_name();

      for (auto edge : node->in_edges()) {
        if (edge->IsControlEdge()) {
          graph->AddEdge(edge->src(), edge->src_output(), ngraphassign_op,
                         edge->dst_input());
          graph->RemoveEdge(edge);
        }
      }
      NGRAPH_VLOG(1) << "Added input control edges";
      std::vector<const Edge*> edges;
      for (auto edge : node->out_edges()) {
        edges.push_back(edge);
      }

      NGRAPH_VLOG(1) << "Got output edges";

      for (auto edge : edges) {
        graph->AddEdge(ngraphassign_op, edge->src_output(), edge->dst(),
                       edge->dst_input());
        graph->RemoveEdge(edge);
      }

      NGRAPH_VLOG(1) << "Replaced output edges";

      graph->RemoveNode(node);
      NGRAPH_VLOG(1) << "Removing node";

    }  // AssignSub
    else if (node->type_string() == "NGraphAssignAdd") {
      NGRAPH_VLOG(1) << " Replace Optimizers ";
      NodeBuilder::NodeOut input_ref;
      NodeBuilder::NodeOut input_val;

      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

      for (auto edge : node->in_edges()) {
        if (edge == NULL) {
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
      NGRAPH_VLOG(1) << "Has assigned device " << node->assigned_device_name();

      Node* add_op;
      string new_name_add = node->name() + "_Add";
      TF_RETURN_IF_ERROR(NodeBuilder(new_name_add, "Add")
                             .Input(input_ref)
                             .Input(input_val)
                             .Attr("T", dtype)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &(add_op)));
      add_op->set_assigned_device_name(node->assigned_device_name());
      NodeBuilder::NodeOut ndef_add_op = NodeBuilder::NodeOut(add_op, 0);

      NGRAPH_VLOG(1) << "Sub op name: " << add_op->name();
      NGRAPH_VLOG(1) << "Sub op assigned device: "
                     << add_op->assigned_device_name();

      Node* ngraphassign_op;
      string new_name_ngassign = node->name() + "_NGraphAssign";

      TF_RETURN_IF_ERROR(NodeBuilder(new_name_ngassign, "NGraphAssign")
                             .Attr("validate_shape", true)
                             .Attr("use_locking", true)
                             .Attr("T", dtype)
                             .Attr("ngraph_graph_id", 0)
                             .Input(input_ref)
                             .Input(ndef_add_op)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &ngraphassign_op));
      ngraphassign_op->set_assigned_device_name(node->assigned_device_name());
      NGRAPH_VLOG(1) << "Assign op name: " << ngraphassign_op->name();
      NGRAPH_VLOG(1) << "Assign op assigned device: "
                     << ngraphassign_op->assigned_device_name();

      for (auto edge : node->in_edges()) {
        if (edge->IsControlEdge()) {
          graph->AddEdge(edge->src(), edge->src_output(), ngraphassign_op,
                         edge->dst_input());
          graph->RemoveEdge(edge);
        }
      }

      std::vector<const Edge*> edges;
      for (auto edge : node->out_edges()) {
        edges.push_back(edge);
      }

      for (auto edge : edges) {
        graph->AddEdge(ngraphassign_op, edge->src_output(), edge->dst(),
                       edge->dst_input());
        graph->RemoveEdge(edge);
      }

      graph->RemoveNode(node);
      NGRAPH_VLOG(1) << "Replaced AssignAdd";

    }  // AssignAdd

    else if (node->type_string() == "NGraphApplyGradientDescent") {
      NodeBuilder::NodeOut input_var;
      NodeBuilder::NodeOut input_alpha;
      NodeBuilder::NodeOut input_delta;

      // TODO(Mingshan): we may removing the control_edges to the
      // ApplyGradientDescent node
      std::vector<const Edge*> input_edges;
      TF_RETURN_IF_ERROR(node->input_edges(&input_edges));

      NGRAPH_VLOG(1) << "No of input edges to ApplyGradientDescent "
                     << input_edges.size();

      input_var = NodeBuilder::NodeOut(input_edges[0]->src(),
                                       input_edges[0]->src_output());
      input_alpha = NodeBuilder::NodeOut(input_edges[1]->src(),
                                         input_edges[1]->src_output());
      input_delta = NodeBuilder::NodeOut(input_edges[2]->src(),
                                         input_edges[2]->src_output());

      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "T", &dtype));

      Node* mul_op;
      string new_name_mul = node->name() + "_Mul";
      TF_RETURN_IF_ERROR(NodeBuilder(new_name_mul, "Mul")
                             .Input(input_alpha)
                             .Input(input_delta)
                             .Attr("T", dtype)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &(mul_op)));
      mul_op->set_assigned_device_name(node->assigned_device_name());
      NodeBuilder::NodeOut ndef_mul_op = NodeBuilder::NodeOut(mul_op, 0);

      Node* sub_op;
      string new_name_sub = node->name() + "_Sub";
      TF_RETURN_IF_ERROR(NodeBuilder(new_name_sub, "Sub")
                             .Input(input_var)
                             .Input(ndef_mul_op)
                             .Attr("T", dtype)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &(sub_op)));
      sub_op->set_assigned_device_name(node->assigned_device_name());
      NodeBuilder::NodeOut ndef_sub_op = NodeBuilder::NodeOut(sub_op, 0);

      Node* ngraphassign_op;
      string new_name_ngassign = node->name() + "_NGraphAssign";

      TF_RETURN_IF_ERROR(NodeBuilder(new_name_ngassign, "NGraphAssign")
                             .Attr("validate_shape", true)
                             .Attr("use_locking", true)
                             .Attr("T", dtype)
                             .Attr("ngraph_graph_id", 0)
                             .Input(input_var)
                             .Input(ndef_sub_op)
                             .Device(node->assigned_device_name())
                             .Finalize(graph, &ngraphassign_op));
      ngraphassign_op->set_assigned_device_name(node->assigned_device_name());
      NGRAPH_VLOG(1) << "Assign op name: " << ngraphassign_op->name();
      NGRAPH_VLOG(1) << "Assign op assigned device: "
                     << ngraphassign_op->assigned_device_name();

      for (auto edge : node->in_edges()) {
        if (edge->IsControlEdge()) {
          graph->AddEdge(edge->src(), edge->src_output(), ngraphassign_op,
                         edge->dst_input());
          graph->RemoveEdge(edge);
        }
      }

      std::vector<const Edge*> edges;
      for (auto edge : node->out_edges()) {
        edges.push_back(edge);
      }

      for (auto edge : edges) {
        graph->AddEdge(ngraphassign_op, edge->src_output(), edge->dst(),
                       edge->dst_input());
        graph->RemoveEdge(edge);
      }

      graph->RemoveNode(node);
      NGRAPH_VLOG(1) << "Replaced ApplyGradientDescent";
    }
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
