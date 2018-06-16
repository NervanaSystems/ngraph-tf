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

  tf::Status ConfirmPlacement(tf::Graph* graph) {
    std::map<std::string, std::function<bool(const tf::Node*)>>
        confirmation_functions;

    auto unconditional = [](const tf::Node* n) { return true; };

    confirmation_functions["Abs"] = unconditional;
    confirmation_functions["Add"] = unconditional;
    confirmation_functions["AvgPool"] = unconditional;
    confirmation_functions["BiasAdd"] = unconditional;
    // TODO: ConcatV2 only works when given constant concatenation axis
    confirmation_functions["ConcatV2"] = unconditional;
    confirmation_functions["Const"] = unconditional;
    confirmation_functions["Conv2D"] = unconditional;
    confirmation_functions["DepthwiseConv2dNative"] = unconditional;
    confirmation_functions["Equal"] = unconditional;
    confirmation_functions["Floor"] = unconditional;
    confirmation_functions["FusedBatchNorm"] = unconditional;
    confirmation_functions["Identity"] = unconditional;
    confirmation_functions["MatMul"] = unconditional;
    confirmation_functions["MaxPool"] = unconditional;
    // TODO: Mean only works when keep_dims = false
    // TODO: Mean only works when given constant reduction axes
    confirmation_functions["Mean"] = unconditional;
    confirmation_functions["Mul"] = unconditional;
    confirmation_functions["NoOp"] = unconditional;
    // TODO: Pad only works when given constant padding widths
    confirmation_functions["Pad"] = unconditional;
    confirmation_functions["Relu"] = unconditional;
    confirmation_functions["Relu6"] = unconditional;
    // TODO: Reshape only works when given constant result shape
    confirmation_functions["Reshape"] = unconditional;
    confirmation_functions["Sign"] = unconditional;
    confirmation_functions["Snapshot"] = unconditional;
    confirmation_functions["Squeeze"] = unconditional;
    // TODO: Sum only works when keep_dims = false
    // TODO: Sum only works when given constant reduction axes
    confirmation_functions["Sum"] = unconditional;
    // TODO: Transpose only works when given constant axis permutation
    confirmation_functions["Transpose"] = unconditional;

    for (auto node : graph->op_nodes()) {
      if (NGraphPlacementRequested(node)) {
        auto it = confirmation_functions.find(node->type_string());

        if (it != confirmation_functions.end() && it->second(node)) {
          node->AddAttr("_kernel", "ngraph");
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
