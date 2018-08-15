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

#include "ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// The "marking" pass checks every node with requested placement on nGraph,
// and either rejects the placement request, or tags it with suitable metadata.
//
// For now we assume that every node has nGraph placement requested, unless the
// environment variable NGRAPH_TF_DISABLE is set. (TODO(amprocte): implement
// something better.)
//
// Each TensorFlow op supported by nGraph has a "confirmation function"
// associated with it. When the confirmation pass encounters a node of op "Op",
// the confirmation function for "Op" first checks if this particular instance
// of the op can be placed on nGraph, possibly attaching extra metadata to the
// node for later use, and returns "true" if placement is allowed. Every
// confirmed op has the attribute "_ngraph_marked_for_clustering" set to
// "true".
//
// See the body of "MarkForClustering" for more details on what a "confirmation
// function" does.
//

using ConfirmationFunction = std::function<Status(Node*, bool*)>;

//
// Utility function to check if placement on the NGRAPH device has been
// requested.
//
// FIXME(amprocte): stubbed out for now because NGRAPH device is gone.
//
static bool NGraphPlacementRequested(const Node* node) { return true; }

//
// Marks the input indices in "inputs" as static, i.e., inputs that must be
// driven either by an _Arg or by a Const in the encapsulated graph.
//
static inline void SetStaticInputs(Node* n, std::vector<int32> inputs) {
  n->AddAttr("_ngraph_static_inputs", inputs);
}

//
// Main entry point for the marking pass.
//
Status MarkForClustering(Graph* graph) {
  //
  // If NGRAPH_TF_DISABLE is set we will not mark anything; all subsequent
  // passes become a no-op.
  //
  if (std::getenv("NGRAPH_TF_DISABLE") != nullptr) {
    return Status::OK();
  }

  //
  // A map of op types (e.g. "Add") to type constraint maps. For (fake)
  // example:
  //
  //  type_constraint_map["Cast"]["SrcT"] = {DT_FLOAT, DT_BOOL};
  //  type_constraint_map["Cast"]["DstT"] = {DT_DOUBLE, DT_INT16};
  //
  // ...would mean that for the "Cast" op, the "SrcT" type variable can be
  // DT_FLOAT or DT_BOOL, and the "DstT" type variable can be DT_DOUBLE or
  // DT_INT16.
  //
  static std::map<std::string, std::map<std::string, gtl::ArraySlice<DataType>>>
      type_constraint_map;

  //
  // A map of op types (e.g. "Add") to confirmation functions. These can be
  // used to check arbitrary constraints, and attach information to the node
  // in the process. For example:
  // TODO(amprocte): following example is stale.
  //
  //    confirmation_functions["MyOp"] = [](Node* n, bool* result) {
  //      Node* tf_arg_node;
  //      TF_RETURN_IF_ERROR(n->input_node(0, &tf_arg_node));
  //
  //      std::vector<int64> tf_const_data;
  //      if (ExtractConstantData(tf_arg_node, &tf_const_data) !=
  //              Status::OK() ||
  //          tf_const_data.size() != 1) {
  //        *result = false;
  //        return Status::OK();
  //      }
  //
  //      n->AddAttr("_ngraph_myop_constant_input", tf_const_data[0]);
  //      *result = true;
  //      return Status::OK();
  //    };
  //
  // The foregoing function checks every "MyOp" node to make sure that its
  // zeroth input node is a constant scalar, and if it is, extracts the value
  // of that scalar, and attaches it to the node as the
  // "_ngraph_myop_constant_input" attribute. Placement fails if the input is
  // not a constant scalar (since "false" is written to *result).
  //
  static std::map<std::string, ConfirmationFunction> confirmation_functions;

  mutex init_mu;
  static bool initialized = false;

  // If the type constraint and confirmation function maps have not been
  // initialized, initialize them.
  //
  // IF YOU ARE ADDING A NEW OP IMPLEMENTATION, ADD TYPE CONSTRAINTS AND A
  // CONFIRMATION FUNCTION FOR THE OP HERE. The constraint function should
  // refuse placement if the node is not supported in the builder, and tag
  // the node with any data that will be needed in case the graph is broken
  // up in a later rewrite pass (for example, constant data).
  {
    mutex_lock l(init_mu);

    if (!initialized) {
      //
      // Initialize type constraint map.
      //
      type_constraint_map["Abs"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Add"]["T"] = NGraphNumericDTypes();
      type_constraint_map["AddN"]["T"] = NGraphNumericDTypes();
      type_constraint_map["AvgPool"]["T"] = NGraphNumericDTypes();
      type_constraint_map["AvgPoolGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["BatchMatMul"]["T"] = NGraphNumericDTypes();
      type_constraint_map["BiasAdd"]["T"] = NGraphNumericDTypes();
      type_constraint_map["BiasAddGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Cast"]["SrcT"] = NGraphDTypes();
      type_constraint_map["Cast"]["DstT"] = NGraphDTypes();
      type_constraint_map["ConcatV2"]["T"] = NGraphDTypes();
      type_constraint_map["ConcatV2"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["Const"]["dtype"] = NGraphDTypes();
      type_constraint_map["Conv2D"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Conv2DBackpropInput"]["T"] = NGraphNumericDTypes();
      type_constraint_map["DepthwiseConv2dNative"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Equal"]["T"] = NGraphDTypes();
      type_constraint_map["Exp"]["T"] = NGraphNumericDTypes();
      type_constraint_map["ExpandDims"]["T"] = NGraphDTypes();
      type_constraint_map["Floor"]["T"] = NGraphNumericDTypes();
      type_constraint_map["FloorDiv"]["T"] = NGraphNumericDTypes();
      type_constraint_map["FloorMod"]["T"] = NGraphNumericDTypes();
      type_constraint_map["FusedBatchNorm"]["T"] = NGraphNumericDTypes();
      type_constraint_map["FusedBatchNormGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Greater"]["T"] = NGraphDTypes();
      type_constraint_map["GreaterEqual"]["T"] = NGraphDTypes();
      type_constraint_map["Identity"]["T"] = NGraphDTypes();
      type_constraint_map["L2Loss"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Less"]["T"] = NGraphDTypes();
      type_constraint_map["LessEqual"]["T"] = NGraphDTypes();
      type_constraint_map["Log"]["T"] = NGraphNumericDTypes();
      // LogicalAnd and LogicalNot have no type attributes ("T", if it existed,
      // would always be bool).
      type_constraint_map["MatMul"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Maximum"]["T"] = NGraphNumericDTypes();
      type_constraint_map["MaxPool"]["T"] = NGraphNumericDTypes();
      type_constraint_map["MaxPoolGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Mean"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Mean"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["Minimum"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Mul"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Neg"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Pack"]["T"] = NGraphDTypes();
      type_constraint_map["Pad"]["T"] = NGraphDTypes();
      type_constraint_map["Pad"]["Tpaddings"] = NGraphIndexDTypes();
      type_constraint_map["Pow"]["T"] = NGraphNumericDTypes();
      type_constraint_map["PreventGradient"]["T"] = NGraphDTypes();
      type_constraint_map["Prod"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Prod"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["RealDiv"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Reciprocal"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Relu"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Relu6"]["T"] = NGraphNumericDTypes();
      type_constraint_map["ReluGrad"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Reshape"]["T"] = NGraphDTypes();
      type_constraint_map["Reshape"]["Tshape"] = NGraphIndexDTypes();
      type_constraint_map["Rsqrt"]["T"] = NGraphDTypes();
      type_constraint_map["Slice"]["T"] = NGraphDTypes();
      type_constraint_map["Slice"]["Index"] = NGraphIndexDTypes();
      type_constraint_map["Sign"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Sigmoid"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Snapshot"]["T"] = NGraphDTypes();
      type_constraint_map["Softmax"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Split"]["T"] = NGraphDTypes();
      type_constraint_map["SplitV"]["T"] = NGraphDTypes();
      type_constraint_map["SplitV"]["Tlen"] = NGraphIndexDTypes();
      type_constraint_map["Square"]["T"] = NGraphDTypes();
      type_constraint_map["SquaredDifference"]["T"] = NGraphDTypes();
      type_constraint_map["Squeeze"]["T"] = NGraphDTypes();
      type_constraint_map["StridedSlice"]["T"] = NGraphDTypes();
      type_constraint_map["StridedSlice"]["Index"] = NGraphIndexDTypes();
      type_constraint_map["Sub"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Sum"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Sum"]["Tidx"] = NGraphIndexDTypes();
      type_constraint_map["Tanh"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Tile"]["T"] = NGraphNumericDTypes();
      type_constraint_map["Tile"]["Tmultiples"] = NGraphIndexDTypes();
      type_constraint_map["Transpose"]["T"] = NGraphDTypes();
      type_constraint_map["Transpose"]["Tperm"] = NGraphIndexDTypes();
      type_constraint_map["Unpack"]["T"] = NGraphDTypes();

      //
      // Initialize confirmation function map.
      //

      // Trivial confirmation function which always accepts placement.
      ConfirmationFunction always = [](Node* n, bool* result) {
        *result = true;
        return Status::OK();
      };

      //
      // Please keep these in alphabetical order by op name.
      //
      confirmation_functions["Abs"] = always;
      confirmation_functions["Add"] = always;
      confirmation_functions["AddN"] = always;
      confirmation_functions["AvgPool"] = always;
      confirmation_functions["AvgPoolGrad"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {0});
        *result = true;
        return Status::OK();
      };
      confirmation_functions["BatchMatMul"] = always;
      confirmation_functions["BiasAdd"] = always;
      confirmation_functions["BiasAddGrad"] = always;
      confirmation_functions["Cast"] = always;

      // Constraint: axis selection input must be static.
      confirmation_functions["ConcatV2"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {n->num_inputs() - 1});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Const"] = always;

      confirmation_functions["Conv2D"] = always;
      confirmation_functions["Conv2DBackpropFilter"] = [](Node* n,
                                                          bool* result) {
        SetStaticInputs(n, {1});
        *result = true;
        return Status::OK();
      };
      confirmation_functions["Conv2DBackpropInput"] = [](Node* n,
                                                         bool* result) {
        SetStaticInputs(n, {0});
        *result = true;
        return Status::OK();
      };
      confirmation_functions["DepthwiseConv2dNative"] = always;
      confirmation_functions["Equal"] = always;
      confirmation_functions["Exp"] = always;
      confirmation_functions["ExpandDims"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Fill"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {0});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Floor"] = always;
      confirmation_functions["FloorDiv"] = always;
      confirmation_functions["FloorMod"] = always;
      confirmation_functions["FusedBatchNorm"] = always;
      confirmation_functions["FusedBatchNormGrad"] = always;
      confirmation_functions["Greater"] = always;
      confirmation_functions["GreaterEqual"] = always;
      confirmation_functions["Identity"] = always;
      confirmation_functions["L2Loss"] = always;
      confirmation_functions["Less"] = always;
      confirmation_functions["LessEqual"] = always;
      confirmation_functions["Log"] = always;
      confirmation_functions["LogicalAnd"] = always;
      confirmation_functions["LogicalNot"] = always;
      confirmation_functions["MatMul"] = always;
      confirmation_functions["Maximum"] = always;
      confirmation_functions["MaxPool"] = always;
      confirmation_functions["MaxPoolGrad"] = always;

      // Constraint: reduction-axes input must be static.
      confirmation_functions["Mean"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Minimum"] = always;
      confirmation_functions["Mul"] = always;
      confirmation_functions["Neg"] = always;

      // Constraint: padding-widths input must be static.
      confirmation_functions["Pad"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Pow"] = always;
      confirmation_functions["PreventGradient"] = always;

      // Constraint: reduction-axes input must be static.
      confirmation_functions["Prod"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["RealDiv"] = always;
      confirmation_functions["Reciprocal"] = always;
      confirmation_functions["Relu"] = always;
      confirmation_functions["Relu6"] = always;
      confirmation_functions["ReluGrad"] = always;

      // Constraint: shape input must be static.
      confirmation_functions["Reshape"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Rsqrt"] = always;
      confirmation_functions["Sigmoid"] = always;
      confirmation_functions["Sign"] = always;

      // Constraint: begin and size input must be static.
      confirmation_functions["Slice"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1,2});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Snapshot"] = always;
      confirmation_functions["Softmax"] = always;

      // Constraint: num splits input must be static.
      confirmation_functions["Split"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {0});
        *result = true;
        return Status::OK();
      };

      // Constraint: size splits, num splits inputs must be static.
      confirmation_functions["SplitV"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1,2});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Square"] = always;
      confirmation_functions["SquaredDifference"] = always;
      confirmation_functions["Squeeze"] = always;

      // Constraint: begin, end, and stride inputs must be static.
      confirmation_functions["StridedSlice"] = [](Node* n, bool* result) {
        // reject if tf.newaxis in strided slice
        // TODO support tf.newaxis
        int tf_new_axis_mask;
        TF_RETURN_IF_ERROR(
            GetNodeAttr(n->attrs(), "new_axis_mask", &tf_new_axis_mask));
        if (tf_new_axis_mask != 0) {
          *result = false;
          return Status::OK();
        }

        SetStaticInputs(n, {1,2,3});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Pack"] = always;
      confirmation_functions["Sub"] = always;

      // Constraints: reduction-axes input must be static.
      confirmation_functions["Sum"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Tanh"] = always;
      confirmation_functions["Tile"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1});
        *result = true;
        return Status::OK();
      };

      // Constraint: permutation input must be static.
      confirmation_functions["Transpose"] = [](Node* n, bool* result) {
        SetStaticInputs(n, {1});
        *result = true;
        return Status::OK();
      };

      confirmation_functions["Unpack"] = always;

      initialized = true;
    }
  }

  for (auto node : graph->op_nodes()) {
    if (NGraphPlacementRequested(node)) {
      bool type_constraints_ok = true;

      // First check type constraints.
      for (auto& name_and_set : type_constraint_map[node->type_string()]) {
        auto& type_attr_name = name_and_set.first;
        auto& allowed_types = name_and_set.second;

        DataType dt;

        if (GetNodeAttr(node->attrs(), type_attr_name, &dt) != Status::OK() ||
            std::find(allowed_types.begin(), allowed_types.end(), dt) ==
                allowed_types.end()) {
          type_constraints_ok = false;
          break;
        }
      }

      // If type constraints are satisfied, check for a confirmation
      // function.
      bool confirmed = false;
      if (type_constraints_ok) {
        auto it = confirmation_functions.find(node->type_string());

        if (it != confirmation_functions.end()) {
          TF_RETURN_IF_ERROR(it->second(node, &confirmed));
        }
      }

      // Set the _ngraph_marked_for_clustering attribute if type constraints
      // are satisfied and the confirmation function (if any) has returned
      // true.
      if (confirmed) {
        NGRAPH_VLOG(4) << "Accepting: " << node->name() << "["
                       << node->type_string() << "]";
        // TODO(amprocte): move attr name to a constant
        node->AddAttr("_ngraph_marked_for_clustering", true);
      } else {
        NGRAPH_VLOG(4) << "Rejecting: " << node->name() << "["
                       << node->type_string() << "]";
      }
    }
  }

  return Status::OK();
}

bool NodeIsMarkedForClustering(const Node* node) {
  bool is_marked;
  // TODO(amprocte): move attr name to a constant
  return (GetNodeAttr(node->attrs(), "_ngraph_marked_for_clustering",
                      &is_marked) == Status::OK() &&
          is_marked);
}

void GetStaticInputs(const Node* node, std::vector<int32>* inputs) {
  if (GetNodeAttr(node->attrs(), "_ngraph_static_inputs", inputs) != Status::OK()) {
    *inputs = std::vector<int32>{};
  }
}

bool InputIsStatic(const Node* node, int index) {
  std::vector<int32> inputs;
  GetStaticInputs(node,&inputs);

  for (auto i : inputs) {
    if (i == index) {
      return true;
    }
  }

  return false;
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
