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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph_utils.h"

using namespace std;
namespace ngraph_bridge {

// TODO(amprocte): this decl should probably be in a header.
extern const char* const DEVICE_NGRAPH;

class NGraphConfirmPass : public tensorflow::GraphOptimizationPass {
 public:
  tf::Status Run(const tf::GraphOptimizationPassOptions& options) {
    return ConfirmPlacement(options.graph->get());
  }

 private:
  static bool NGraphPlacementRequested(const tf::Node* node) {
    tf::DeviceNameUtils::ParsedName parsed;

    if (!tf::DeviceNameUtils::ParseFullName(node->requested_device(),
                                            &parsed)) {
      return false;
    }

    return (parsed.has_type && parsed.type == DEVICE_NGRAPH);
  }

  static tf::Status ExtractConstantData(tf::Node* node, std::vector<tf::int64>* values) {
    if (node->type_string() != "Const") {
      return tf::errors::InvalidArgument("Tried to extract constant data from a non-Const node");
    }

    tf::DataType dtype;
    TF_RETURN_IF_ERROR(tf::GetNodeAttr(node->attrs(), "dtype", &dtype));

    tf::TensorShapeProto shape_proto;

    switch (dtype) {
      case tf::DataType::DT_INT32:
        {
          std::vector<tf::int32> values_int32;
          TF_RETURN_IF_ERROR(ValuesFromConstNode<tf::int32>(node->def(), &shape_proto, &values_int32));
          values->resize(values_int32.size());
          for (size_t i = 0; i < values_int32.size(); i++) {
            (*values)[i] = (tf::int64)values_int32[i];
          }
        }
        break;
      case tf::DataType::DT_INT64:
        TF_RETURN_IF_ERROR(ValuesFromConstNode<tf::int32>(node->def(), &shape_proto, values));
        break;
      default:
        return tf::errors::InvalidArgument("Tried to extract constant data from a Const node that is neither DT_INT32 nor DT_INT64");
    }

    return tf::Status::OK();
  }

  tf::Status ConfirmPlacement(tf::Graph* graph) {
    std::map<std::string, std::function<tf::Status(tf::Node*, bool*)>>
        confirmation_functions;

    auto unconditional = [](tf::Node* n, bool* result) { *result = true; return tf::Status::OK(); };

    confirmation_functions["Abs"] = unconditional;
    confirmation_functions["Add"] = unconditional;
    confirmation_functions["AvgPool"] = unconditional;
    confirmation_functions["BiasAdd"] = unconditional;
    confirmation_functions["ConcatV2"] = [](tf::Node* n, bool *result) {
      tf::Node* tf_axis_node;
      TF_RETURN_IF_ERROR(n->input_node(n->num_inputs() - 1, &tf_axis_node));

      std::vector<tf::int64> tf_static_axis;
      if (ExtractConstantData(tf_axis_node, &tf_static_axis) != tf::Status::OK() || tf_static_axis.size() != 1) {
        *result = false;
        return tf::Status::OK();
      }

      n->AddAttr("_ngraph_concat_static_axis", tf_static_axis[0]);
      *result = true;
      return tf::Status::OK();
    };
    confirmation_functions["Conv2D"] = unconditional;
    //confirmation_functions["DepthwiseConv2dNative"] = unconditional;
    confirmation_functions["Equal"] = unconditional;
    confirmation_functions["Floor"] = unconditional;
    confirmation_functions["FusedBatchNorm"] = unconditional;
    confirmation_functions["MatMul"] = unconditional;
    confirmation_functions["MaxPool"] = unconditional;
    confirmation_functions["Mean"] = [](tf::Node* n, bool *result) {
      bool tf_keep_dims;

      if (tf::GetNodeAttr(n->attrs(), "keep_dims", &tf_keep_dims) != tf::Status::OK()) {
        tf_keep_dims = false;
      }

      tf::Node* tf_axes_node;
      TF_RETURN_IF_ERROR(n->input_node(1, &tf_axes_node));

      std::vector<tf::int64> tf_static_axes;
      if (ExtractConstantData(tf_axes_node, &tf_static_axes) != tf::Status::OK()) {
        *result = false;
        return tf::Status::OK();
      }

      n->AddAttr("_ngraph_mean_static_axes", tf_static_axes);
      *result = true;
      return tf::Status::OK();
    };
    confirmation_functions["Mul"] = unconditional;
    confirmation_functions["Pad"] = [](tf::Node* n, bool *result) {
      tf::Node* tf_paddings_node;
      TF_RETURN_IF_ERROR(n->input_node(1, &tf_paddings_node));

      std::vector<tf::int64> tf_static_paddings;
      if (ExtractConstantData(tf_paddings_node, &tf_static_paddings) != tf::Status::OK()) {
        *result = false;
        return tf::Status::OK();
      }

      n->AddAttr("_ngraph_pad_static_paddings", tf_static_paddings);
      *result = true;
      return tf::Status::OK();
    };
    confirmation_functions["Relu"] = unconditional;
    confirmation_functions["Relu6"] = unconditional;
    confirmation_functions["Reshape"] = [](tf::Node* n, bool *result) {
      tf::Node* tf_shape_node;
      TF_RETURN_IF_ERROR(n->input_node(1, &tf_shape_node));

      std::vector<tf::int64> tf_static_shape;
      if (ExtractConstantData(tf_shape_node, &tf_static_shape) != tf::Status::OK()) {
        *result = false;
        return tf::Status::OK();
      }

      n->AddAttr("_ngraph_reshape_static_shape", tf_static_shape);
      *result = true;
      return tf::Status::OK();
    };
    confirmation_functions["Sign"] = unconditional;
    confirmation_functions["Snapshot"] = unconditional;
    confirmation_functions["Squeeze"] = unconditional;
    confirmation_functions["Sum"] = [](tf::Node* n, bool *result) {
      bool tf_keep_dims;

      if (tf::GetNodeAttr(n->attrs(), "keep_dims", &tf_keep_dims) != tf::Status::OK()) {
        tf_keep_dims = false;
      }

      tf::Node* tf_axes_node;
      TF_RETURN_IF_ERROR(n->input_node(1, &tf_axes_node));

      std::vector<tf::int64> tf_static_axes;
      if (ExtractConstantData(tf_axes_node, &tf_static_axes) != tf::Status::OK()) {
        *result = false;
        return tf::Status::OK();
      }

      n->AddAttr("_ngraph_sum_static_axes", tf_static_axes);
      *result = true;
      return tf::Status::OK();
    };
    confirmation_functions["Transpose"] = [](tf::Node* n, bool *result) {
      tf::Node* tf_permutation_node;
      TF_RETURN_IF_ERROR(n->input_node(1, &tf_permutation_node));

      std::vector<tf::int64> tf_static_permutation;
      if (ExtractConstantData(tf_permutation_node, &tf_static_permutation) != tf::Status::OK()) {
        *result = false;
        return tf::Status::OK();
      }

      n->AddAttr("_ngraph_transpose_static_permutation", tf_static_permutation);
      *result = true;
      return tf::Status::OK();
    };

    for (auto node : graph->op_nodes()) {
      if (NGraphPlacementRequested(node)) {
        bool confirmed = false;

        auto it = confirmation_functions.find(node->type_string());

        if (it != confirmation_functions.end()) {
          TF_RETURN_IF_ERROR(it->second(node,&confirmed));
        }

        if (confirmed) {
          node->AddAttr("_kernel", "ngraph");
        }
        else {
          NGRAPH_VLOG(4) << "Rejecting: " << node->name() << "[" << node->type_string() << "]";
        }
      }
    }

    return tf::Status::OK();
  }
};
}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 90,
                      ngraph_bridge::NGraphConfirmPass);
}  // namespace tensorflow
