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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/default/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "ngraph/runtime/backend.hpp"
#include "ngraph_backend_manager.h"
#include "ngraph_freshness_tracker.h"
#include "ngraph_utils.h"
#include "ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

/* -------------------------------------------------
//
// NGraphAssignOp
//
---------------------------------------------------*/

// Computes *input[0] = input[1]
class NGraphAssignOp : public OpKernel {
 public:
  explicit NGraphAssignOp(OpKernelConstruction* context) : OpKernel(context) {
    // OP_REQUIRES_OK(context,
    //                context->GetAttr("use_locking", &use_exclusive_lock_));
    // OP_REQUIRES_OK(context,
    //                context->GetAttr("validate_shape", &validate_shape_));
    // OP_REQUIRES(context, IsRefType(context->input_type(0)),
    //             errors::InvalidArgument("lhs input needs to be a ref type"));
    // if (!context
    //          ->GetAttr("_grappler_relax_allocator_constraints",
    //                    &relax_constraints_)
    //          .ok()) {
    //   relax_constraints_ = false;
    // }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& rhs = context->input(1);

    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    // get the nGraphTensor
    //string ng_variable_name = context->
    auto itr = BackendManager::ng_variable_map_.find("Var1");
    NGRAPH_VLOG(1)<<"In map ? "<<(itr==BackendManager::ng_variable_map_.end() ? "No " : "Yes");

    NGRAPH_VLOG(1)<<" Map size "<<BackendManager::ng_variable_map_.size();
    for(auto it: BackendManager::ng_variable_map_){
      NGRAPH_VLOG(1)<<"Key "<<it.first <<" Val "<<it.second;
    }
    shared_ptr<ngraph::runtime::Tensor> ng_tensor_to_assign = BackendManager::ng_variable_map_["Var1"];
    
    NGRAPH_VLOG(1)<<"In Assign Kernel : is Var ng-Tenssor null "<<(ng_tensor_to_assign==NULL? "Yes": "No");
    void* tf_src_ptr = (void*)DMAHelper::base(&rhs);
    ng_tensor_to_assign->write(tf_src_ptr, 0, ng_tensor_to_assign->get_element_count() * ng_tensor_to_assign->get_element_type().size());

    NGRAPH_VLOG(1)<<"In Assign Kernel : Print NG Tensor ";
    //PrintNGTensor(ng_tensor_to_assign);
  }
};

REGISTER_OP("NGraphAssign")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true");

REGISTER_KERNEL_BUILDER(Name("NGraphAssign").Device(DEVICE_CPU),
                        NGraphAssignOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
